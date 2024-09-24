import time
import numpy as np
import h5py
import pandas as pd
from tqdm import tqdm
import json
import os
from transformers import AutoTokenizer
ROOTDIR = 'data/unarXive_quantum_physics/answer_version_b/'
SAVEPATH = os.path.join(ROOTDIR, 'jina_answer_token','split')
os.makedirs(SAVEPATH, exist_ok=True)
print("loading csv files...........")

sentense_ids = pd.read_csv(os.path.join(ROOTDIR,"paper_index_to_content.csv"))
print("done~!")
need_question_ids = list(range(len(sentense_ids)))

tokenizer = AutoTokenizer.from_pretrained("pretrain_weights/models--jinaai--jina-embeddings-v2-base-en/snapshots/7302ac470bed880590f9344bfeee32ff8722d0e5", model_max_length=8192, padding_side="right", use_fast=False)
if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.unk_token

def preprocess(string):
    input_ids = tokenizer(
        string,
        return_tensors="pt",
        padding="max_length",
        max_length=tokenizer.model_max_length,
        truncation=True,
    ).input_ids.numpy().astype('uint16') # <--- no need to use int64 as the vocab size is 30528
    return input_ids

def deal_with_id(__id):

    _id          = need_question_ids[__id]
    paper_index  = sentense_ids.iloc[_id].paper_index
    paper_content=sentense_ids.iloc[_id].text
    content = paper_content.replace('  '," ") 
    answer_token = preprocess(content)
    return answer_token




import time
total_chunk = 1000
index_range = np.linspace(0, len(need_question_ids),total_chunk+1).astype('int')
cost_list = []

if __name__ == '__main__':

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
        index_store  = []
        tensor_store = []
        for _id in tqdm(range(start, end)):
            
            index_store.append(_id)
            tensor_store.append(deal_with_id(_id))
            
        print(f"cost {time.time() - now}")
        file_name = f"{SAVEPATH}/token_{start:08d}_{end:08d}"
        index_store = np.array(index_store,dtype=np.int64)
        tensor_store= np.concatenate(tensor_store)
        np.save(file_name+'.idx', index_store)
        np.save(file_name, tensor_store)