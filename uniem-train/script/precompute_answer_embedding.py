import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__),
                                             os.path.pardir)))
from train_qlora import *

#model_name_or_path = "/mnt/petrelfs/zhangtianning.di/.cache/huggingface/hub/models--PY007--TinyLlama-1.1B-intermediate-step-240k-503b/snapshots/a016b960b941eb6eb7884e363b935370bafa5932/"
model_name_or_path = "pretrain_weights/models--jinaai--jina-embeddings-v2-base-en/snapshots/7302ac470bed880590f9344bfeee32ff8722d0e5"
model_max_length   = 8000
tokenizer = transformers.AutoTokenizer.from_pretrained(model_name_or_path,
        model_max_length=model_max_length,
        padding_side="right",
        use_fast=False,
    )
if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.unk_token
# replace_llama_attn_with_flash_attn()
# model = transformers.AutoModelForCausalLM.from_pretrained(model_name_or_path,
#         device_map='auto',
#         quantization_config=None,
#     )
# model = LlamaLastWeightedEmbedder(model)

from models.jinabert.modeling_bert import JinaBertModel
from models.jinabert.configuration_bert import JinaBertConfig
config                 = JinaBertConfig.from_pretrained(os.path.join(model_name_or_path, 'config.json'))
config.alibi_scaling   = 1
config.embedding_model = True
config.attn_implementation = 'flashV2'
model = JinaBertModel.from_pretrained(model_name_or_path,config=config,device_map=None)
model = BertEmbedder(model)
model = model.cuda()

# ====================================================================
# data_module = make_supervised_data_module(tokenizer=tokenizer, data_args=None, model_max_length = model_max_length, return_mapping=True,dispatch_batches=True)
# answer_dataset = data_module['train_dataset'].answer_memory
# indexes = np.array(answer_dataset.index)
# @torch.no_grad()
# def deal_with_answer_id(__id):
#     _id = need_answer_ids[__id]
#     answer_index = indexes[_id]
#     answer_ids   = torch.from_numpy(tensors[_id].astype('int')).cuda()
#     with torch.cuda.amp.autocast(enabled=True, dtype=torch.float16):
#         answer_embedding = model(answer_ids)    
#     return {'index': answer_index, 'embedding': answer_embedding.detach().cpu().numpy()}

datapair_path = 'data/unarXive_quantum_physics/pair.answer_version_b.question_version_a.json'
data_module = make_supervised_data_module2(tokenizer=tokenizer, dummy_data='only_answer',model_max_length = model_max_length, datapair_path=datapair_path, 
                                           use_reference=False, dispatch_batches=True)
answer_dataset = data_module['train_dataset']

answer_tokens_unique_id = answer_dataset.answer_tokens_unique_id
answer_tokens = answer_dataset.answer_tokens
answer_tokens_unique_id_keys = list(answer_tokens_unique_id.keys())

# ==========================================================
batch_size = 3
need_answer_ids = np.array_split(np.arange(len(answer_tokens_unique_id_keys)),len(answer_tokens_unique_id_keys)//batch_size)


exit()
@torch.no_grad()
def deal_with_answer_id(__id):
    _ids = need_answer_ids[__id]
    answer_keys  = [answer_tokens_unique_id_keys[i] for i in _ids]
    answer_index = [answer_tokens_unique_id[k] for k in answer_keys]
    answer_token = torch.from_numpy(answer_tokens[answer_index].astype('int')).cuda()
    #.cuda()
    with torch.cuda.amp.autocast(enabled=True, dtype=torch.float16):
        answer_embedding = model(answer_token)
        assert not answer_embedding.isinf().any()
        assert not answer_embedding.isnan().any()
    return {'index': answer_keys, 'embedding': answer_embedding.detach().cpu().numpy()}


def split_run_embedding(total_chunk):
    import time
    index_range = np.linspace(0, len(need_answer_ids),total_chunk+1).astype('int')
    SAVEPATH = 'data/unarXive_quantum_physics/answer_version_b/jina_embedding/alpha'
    LOCKPATH = os.path.join(SAVEPATH,'lock')
    SPLTPATH = os.path.join(SAVEPATH,'split')
    os.makedirs(LOCKPATH, exist_ok=True)
    os.makedirs(SPLTPATH, exist_ok=True)
    cost_list = []
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
            result = deal_with_answer_id(_id)
            for key, val in result.items(): 
                if key not in results:
                    results[key] = []
                results[key].append(val)
        results['index'] = np.concatenate(results['index'])
        results['embedding'] = np.concatenate(results['embedding'])
        print(f"cost {time.time() - now}")
        np.save(os.path.join(SPLTPATH,f'answer_embedding.{i:05d}_{total_chunk:05d}.npy'), results['embedding'])
        np.save(os.path.join(SPLTPATH,f'answer_embedding.{i:05d}_{total_chunk:05d}.idx.npy'), results['index'])

if __name__ == '__main__':
    split_run_embedding(100)