import time
import numpy as np
import h5py
import pandas as pd
from tqdm import tqdm
import json
import os
from transformers import AutoTokenizer
ROOTPATH = 'data/unarXive_quantum_physics/question_version_a/cache/query_full_paper.question.good_questions.openai.embedding.jsonl'
SAVEPATH = 'data/unarXive_quantum_physics/question_version_a/openai.ada2_question_embedding/'
ValidCSV = 'data/unarXive_quantum_physics/question_version_a/query_full_paper.question.good_questions.csv'

valid_metadata = pd.read_csv(ValidCSV)

import time
total_chunk = 100
index_range = np.linspace(0, len(valid_metadata),total_chunk+1).astype('int')
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
        end   = index_range[i+1]
        print(f'deal with sentense from {start} - {end}')
        now = time.time()
        validationQs = []
        embeddings   = []
        question_indexs = []
        with open(ROOTPATH,'r') as f:
            for line_id, line in tqdm(enumerate(f), total=len(valid_metadata)):
                if line_id < start: continue
                if line_id >=  end: break
                data = json.loads(line)
                true_input = data[0]['input']
                embedding  = np.array(data[1]['data'][0]['embedding']).astype(np.float32)
                question_index = data[2]['question_row']
                validationQ= valid_metadata.iloc[question_index]['question'].strip() == data[0]['input'].strip()
                question_indexs.append(question_index)
                validationQs.append(validationQ)
                embeddings.append(embedding)
        embeddings  = np.stack(embeddings).astype(np.float32)
        validationQs= np.array(validationQs)
        question_indexs = np.array(question_indexs)
        file_name = os.path.join(SAVEPATH, f"embedding_{start:08d}_{end:08d}")
        np.save(file_name+'.idx', question_indexs)
        np.save(file_name+'.verify', validationQs)
        np.save(file_name, embeddings)