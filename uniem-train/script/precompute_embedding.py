from train_qlora import *
replace_llama_attn_with_flash_attn()
model_name_or_path = "/mnt/petrelfs/zhangtianning.di/.cache/huggingface/hub/models--PY007--TinyLlama-1.1B-intermediate-step-240k-503b/snapshots/a016b960b941eb6eb7884e363b935370bafa5932/"
model_max_length   = 32000

tokenizer = transformers.AutoTokenizer.from_pretrained(model_name_or_path,
        model_max_length=model_max_length,
        padding_side="right",
        use_fast=False,
    )
if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.unk_token

model = transformers.AutoModelForCausalLM.from_pretrained(model_name_or_path,
        device_map='auto',
        quantization_config=None,
    )

model = LlamaLastWeightedEmbedder(model)
data_module = make_supervised_data_module(tokenizer=tokenizer, 
                                          data_args=None, 
                                          model_max_length = model_max_length, 
                                          return_mapping=True,
                                          dispatch_batches=True)
model = model.cuda()

question_dataset = data_module['train_dataset'].question_memory
print(f"totally question:{len(question_dataset)}")

need_question_ids = np.array_split(np.arange(len(question_dataset)),len(question_dataset)//8)

indexes = np.array(question_dataset.index)
tensors = question_dataset.tensor
@torch.no_grad()
def deal_with_question_id(__id):
    _id = need_question_ids[__id]
    question_index = indexes[_id]
    question_ids   = torch.from_numpy(tensors[_id].astype('int')).cuda()
    with torch.cuda.amp.autocast():
        question_embedding = model(question_ids)    
    return {'index': question_index, 'embedding': question_embedding.detach().cpu().numpy().astype(np.float32)}


# answer_dataset = data_module['train_dataset'].answer_memory
# need_question_ids = list(range(len(answer_dataset)))
# @torch.no_grad()
# def deal_with_answer_id(__id):
#     _id = need_question_ids[__id]
#     answer_index = answer_dataset.index[_id]
#     answer_ids   = answer_dataset.tensor[_id]
#     with torch.cuda.amp.autocast():
#         answer_embedding = model(answer_ids)    
#     return {'answer_index': _id, 'answer_embedding': answer_embedding}


import time
total_chunk = 100
index_range = np.linspace(0, len(need_question_ids),total_chunk+1).astype('int')
ROOTPATH = '/mnt/petrelfs/zhangtianning.di/projects/llm/uniem-train/data/unarXive_quantum_physics/llama_question_embedding/alpha'
LOCKPATH = os.path.join(ROOTPATH,'lock')
SPLTPATH = os.path.join(ROOTPATH,'split')
os.makedirs(LOCKPATH, exist_ok=True)
os.makedirs(SPLTPATH, exist_ok=True)
cost_list = []
if __name__ == '__main__':

    for i in tqdm(range(total_chunk)):
        lock_file = os.path.join(LOCKPATH,f'lock.{i:05d}_{total_chunk:05d}')
        if os.path.exists(lock_file):
            print(f"{lock_file} exist, continue....")
            continue
        print(f'create lock file at {lock_file}')
        os.system(f'touch {lock_file}')
        start = index_range[i]
        end = index_range[i+1]
        print(f'deal with sentense from {start} - {end}')
        now = time.time()
        results = {}
        for _id in tqdm(range(start, end)):
            result = deal_with_question_id(_id)
            for key, val in result.items(): 
                if key not in results:
                    results[key] = []
                results[key].append(val)
        results['index'] = np.concatenate(results['index'])
        results['embedding'] = np.concatenate(results['embedding'])
        print(f"cost {time.time() - now}")
        np.save(os.path.join(SPLTPATH,f'question_embedding.{i:05d}_{total_chunk:05d}.npy'), results['embedding'])
        np.save(os.path.join(SPLTPATH,f'question_index.{i:05d}_{total_chunk:05d}.npy'), results['index'])