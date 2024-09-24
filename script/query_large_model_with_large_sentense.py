import time
import numpy as np
import h5py
import pandas as pd
from tqdm import tqdm
import json
import os
from fastchat.model.model_adapter import (
        load_model,
        get_conversation_template,
    )

from fastchat.serve.inference import *
@torch.inference_mode()
def generate_with_start_kvcache(
    model,
    tokenizer,
    params: Dict,
    device: str,
    context_len: int,
    stream_interval: int = 2,
    judge_sent_end: bool = False,
    start_kvcache= None,
    return_kvcache:bool=False
):
    if hasattr(model, "device"):device = model.device

    # Read parameters
    prompt = params["prompt"]
    len_prompt = len(prompt)
    temperature = float(params.get("temperature", 1.0))
    repetition_penalty = float(params.get("repetition_penalty", 1.0))
    top_p = float(params.get("top_p", 1.0))
    top_k = int(params.get("top_k", -1))  # -1 means disable
    max_new_tokens = int(params.get("max_new_tokens", 256))
    echo = bool(params.get("echo", False))
    stop_str = params.get("stop", None)
    stop_token_ids = params.get("stop_token_ids", None) or []
    stop_token_ids.append(tokenizer.eos_token_id)

    logits_processor = prepare_logits_processor(
        temperature, repetition_penalty, top_p, top_k
    )
    input_ids = tokenizer(prompt).input_ids

    if model.config.is_encoder_decoder:
        max_src_len = context_len
    else:  # truncate
        max_src_len = context_len - max_new_tokens - 1

    input_ids = input_ids[-max_src_len:]

    output_ids = list(input_ids)
    input_echo_len = len(input_ids)

    if model.config.is_encoder_decoder:
        encoder_output = model.encoder(input_ids=torch.as_tensor([input_ids], device=device))[0]
        start_ids = torch.as_tensor(
            [[model.generation_config.decoder_start_token_id]],
            dtype=torch.int64,
            device=device,
        )
    out = None
    
    past_key_values = start_kvcache
    sent_interrupt = False
    for i in range(max_new_tokens):
        if i == 0:  # prefill
            if model.config.is_encoder_decoder:
                out = model.decoder(input_ids=start_ids,encoder_hidden_states=encoder_output,use_cache=True)
                logits = model.lm_head(out[0])
            else:
                
                cover_token = past_key_values[-1][0].shape[-2] if past_key_values is not None else 0
                out = model(torch.as_tensor([input_ids[cover_token:]], device=device), use_cache=True,past_key_values=past_key_values)
                logits = out.logits
            past_key_values = out.past_key_values
        else:  # decoding
            if model.config.is_encoder_decoder:
                out = model.decoder(
                    input_ids=torch.as_tensor([[token] if not sent_interrupt else output_ids],device=device,),
                    encoder_hidden_states=encoder_output,
                    use_cache=True,
                    past_key_values=past_key_values if not sent_interrupt else None,
                )
                sent_interrupt = False
                logits = model.lm_head(out[0])
            else:
                out = model(
                    input_ids=torch.as_tensor([[token] if not sent_interrupt else output_ids],device=device,),
                    use_cache=True,
                    past_key_values=past_key_values if not sent_interrupt else None,
                )
                sent_interrupt = False
                logits = out.logits
            past_key_values = out.past_key_values

        if logits_processor:
            if repetition_penalty > 1.0:
                tmp_output_ids = torch.as_tensor([output_ids], device=logits.device)
            else:
                tmp_output_ids = None
            last_token_logits = logits_processor(tmp_output_ids, logits[:, -1, :])[0]
        else:
            last_token_logits = logits[0, -1, :]

        if device == "mps":
            # Switch to CPU by avoiding some bugs in mps backend.
            last_token_logits = last_token_logits.float().to("cpu")

        if temperature < 1e-5 or top_p < 1e-8:  # greedy
            _, indices = torch.topk(last_token_logits, 2)
            tokens = [int(index) for index in indices.tolist()]
        else:
            probs = torch.softmax(last_token_logits, dim=-1)
            indices = torch.multinomial(probs, num_samples=2)
            tokens = [int(token) for token in indices.tolist()]
        token = tokens[0]
        output_ids.append(token)

        if token in stop_token_ids:
            stopped = True
        else:
            stopped = False
        
        # Yield the output tokens
        if stopped:
            break
    #print(len(output_ids))
    tmp_output_ids = output_ids[input_echo_len:]
    output = tokenizer.decode(
                tmp_output_ids,
                skip_special_tokens=True,
                spaces_between_special_tokens=False,
                clean_up_tokenization_spaces=True,
            )
    # Finish stream event, which contains finish reason
    if i == max_new_tokens - 1:
        finish_reason = "length"
    elif stopped:
        finish_reason = "stop"
    else:
        finish_reason = None
    
    output = {
        "text": output,
        "usage": {
            "prompt_tokens": input_echo_len,
            "completion_tokens": i,
            "total_tokens": input_echo_len + i,
        },
        "finish_reason": finish_reason,
    }

    if return_kvcache:
        output['last_kvcache'] = past_key_values
    return output
    


ROOTDIR = 'data/unarXive_quantum_physics/'

print("loading csv files...........")
sentense_ids = pd.read_csv(os.path.join(ROOTDIR,"unarXive_quantum_physics.clear.sections.id.csv"))
sentense_ids = list(sentense_ids.groupby('paper_id'))
print("done~!")
need_question_ids = list(range(len(sentense_ids)))

print("loading finished ids.........")
with open(os.path.join(ROOTDIR, "query.question.results.good_questions.ids.json"), 'r') as f:
    good_question_ids = json.load(f)
good_question_ids= [int(a) for a,b in good_question_ids]
print(len(good_question_ids))
print(good_question_ids[:20])
print("done~!")
good_question_ids=set(good_question_ids)
need_question_ids=set(list(range(len(sentense_ids)))) - good_question_ids
need_question_ids = list(need_question_ids)
print(f"remain {len(need_question_ids)}/{len(sentense_ids)} items")


SAVEPATH = os.path.join(ROOTDIR, "full_paper_question_results")
sectionsf = h5py.File('data/unarXive_quantum_physics/unarXive_quantum_physics.clear.sections.h5', 'r')
abstractf = h5py.File('data/unarXive_quantum_physics/unarXive_quantum_physics.clear.abstract.h5', 'r')
titlef = h5py.File('data/unarXive_quantum_physics/unarXive_quantum_physics.clear.title.h5', 'r')

print("loading model...........")
model_path = 'pretrain_weights/vicuna/vicuna-7b-v1.5-16k'
# llm       = LLM(model=model_path)


model, tokenizer = load_model(
    model_path,
    "cuda",
    1,
    load_8bit=False
)


def deal_with_id(__id):
    
    
    _id  = need_question_ids[__id]
    paper_id,group = sentense_ids[_id]
    section_ids = np.sort(group['section_num'].values)
    abstract= abstractf.get(f'abstract/{paper_id}')[()].decode('utf-8').replace('\n', " ").replace('  '," ")
    title   = titlef.get(f'abstract/{paper_id}')[()].decode('utf-8').replace('\n'," ").replace('  '," ")
    content = [sectionsf.get(f'{paper_id}/{sentence_id}')[()].decode('utf-8').replace('\n', " ").replace('  '," ") for sentence_id in section_ids]
    content = "\n".join(content)
    
    qs  = f"""Here's a research paper titled as "{title}". Find its abstract below: "{abstract}".Based on the abstract, provide a brief summary.  Now, please proceed to review the main content of the paper: \n \"\"\"\n{content}"\n\"\"\"\n Create a comprehensive outline for this paper, detailing its logical structure. Take a step-by-step approach to this task."""
    conv = get_conversation_template(model_path)
    conv.append_message(conv.roles[0], qs)
    conv.append_message(conv.roles[1], None)
    history = generate_with_start_kvcache(model,tokenizer,{'prompt':conv.get_prompt(),'max_new_tokens':5000,'temperature':0.7},'cuda',context_len=16000,return_kvcache=True)
    
    output_dict = {'paper_id': paper_id,'outlines':history["text"]}
    for cluster in ['Result','Abstract','Introduction','Methodology','Literature Review','Discussion']:
        conv = get_conversation_template(model_path)
        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], history["text"])
        conv.append_message(conv.roles[0], f"""I need you formulate five insightful question about the "{cluster}" part of this paper based on the information given above. Make the question as short as possible. The question should not be general like 'What is the main purpose'. I need at least one 'Why' question, one 'What' question and one 'How' question. """)
        conv.append_message(conv.roles[1], None)
        downstringquestion = generate_with_start_kvcache(model,tokenizer,{'prompt':conv.get_prompt(),'max_new_tokens':2000,'temperature':0.7},'cuda',context_len=16000,start_kvcache=history['last_kvcache'])
        output_dict[f'question_for_{cluster}'] = downstringquestion['text']
    
    return output_dict


total_chunk = 1000
index_range = np.linspace(0, len(need_question_ids),
                          total_chunk+1).astype('int')
cost_list = []
if __name__ == '__main__':
    # for key,val in deal_with_id(0).items():
    #     print(f"================= {key} ===============")
    #     print(val)
    import torch
    for i in tqdm(range(total_chunk)):
        lock_file = f'lock/lock.{i:05d}_{total_chunk:05d}'
        if os.path.exists(lock_file):
            print(f"{lock_file} exist, continue....")
            continue
        print(f'create lock file at {lock_file}')
        os.system(f'touch {lock_file}')
        start = index_range[i]
        end = index_range[i+1]
        print(f'deal with sentense from {start} - {end}')
        now = time.time()
        result = {}
        for _id in tqdm(range(start, end)):
            try:
                result[_id] = deal_with_id(_id)
                #print(f"{_id}=>{sentense_ids.iloc[_id]['paper_id']}|{sentense_ids.iloc[_id]['section_num']}==> {result[_id]}")
            except:
                print(f"{_id}=>{sentense_ids[_id][0]}==> fail!!! ")
                torch.cuda.empty_cache()
        print(f"cost {time.time() - now}")
        with open(f"{SAVEPATH}/type_{start:08d}_{end:08d}.json", 'w') as ff:
            json.dump(result, ff)
