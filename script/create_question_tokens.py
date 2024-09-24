import pandas as pd
import os
from transformers import AutoTokenizer
import concurrent.futures
from tqdm.auto import tqdm
import numpy as np

ROOTDIR = 'data/unarXive_quantum_physics/question_version_a/'
SAVEPATH = os.path.join(ROOTDIR, 'jina_question_token/split')
os.makedirs(SAVEPATH, exist_ok=True)
questions_df= pd.read_csv(os.path.join(ROOTDIR,"query_full_paper.question.good_questions.csv"))
need_question_ids = list(range(len(questions_df)))

# print("loading finished ids.........")
# with open(os.path.join(ROOTDIR, "query.question.results.good_questions.ids.json"), 'r') as f:
#     good_question_ids = json.load(f)
# good_question_ids= [int(a) for a,b in good_question_ids]
good_question_ids=[]
good_question_ids=set(good_question_ids)
need_question_ids=set(list(range(len(questions_df)))) - good_question_ids
need_question_ids = list(need_question_ids)

tokenizer = AutoTokenizer.from_pretrained(
        "pretrain_weights/models--jinaai--jina-embeddings-v2-base-en/snapshots/7302ac470bed880590f9344bfeee32ff8722d0e5",
        model_max_length=64,
        padding_side="right",
        use_fast=True,
    )
if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.unk_token

def preprocess(string):
    input_ids = tokenizer(
        string,
        return_tensors="pt",
        padding="max_length",
        max_length=tokenizer.model_max_length,
        truncation=True,
    ).input_ids.numpy().asdtype('uint16') # <--- no need to use int64 as the vocab size is 30528
    return input_ids
def deal_with_id(__id):

    _id  = need_question_ids[__id]
    question=questions_df.iloc[_id]['question']
    question_token = preprocess(question)
    return question_token

import time
total_chunk = 100
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
            try:
                index_store.append(_id)
                tensor_store.append(deal_with_id(_id))
            except:
                print(f"{_id}==> fail!!! ")
        print(f"cost {time.time() - now}")
        file_name = f"{SAVEPATH}/tensor_{start:08d}_{end:08d}"
        index_store = np.array(index_store,dtype=np.int64)
        tensor_store= np.concatenate(tensor_store)
        np.save(file_name+'.idx', index_store)
        np.save(file_name, tensor_store)