import pandas as pd
import h5py
import numpy as np
import os
os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')

from torch.utils.data import Dataset
from uniem.data_structures import RecordType, PairRecord, TripletRecord, ScoredPairRecord

from uniem.finetuner import FineTuner


# rawdata = pd.read_csv("/mnt/lustre/weigengchen/from_s3/unarXive.clear/query.question.results.good_questions.csv")
rawdata = pd.read_csv("/mnt/petrelfs/pangxinle/workspace/from_s3/query.question.results.good_questions.csv")
sectionsf = h5py.File('/mnt/petrelfs/pangxinle/workspace/from_s3/unarXive_quantum_physics.clear.sections.h5', 'r')


class UnarXive_Question_Sentense_Dataset(Dataset):
    def __init__(self, rawdata):
        self.rawdata      = rawdata
    def __len__(self):
        return int(len(self.rawdata) * 0.85)

    def get_data(self,idx):
        paper_id, sentense_id , question = rawdata.iloc[idx][['paper_id','sentense_id','question']]
        evidence = sectionsf.get(f'{paper_id}/{sentense_id}')[()].decode('utf-8')
        return question,evidence
    def __getitem__(self, idx):
        question,evidence = self.get_data(idx)
        fail_count = 0
        while len(evidence.split())>1024:
            fail_count+=1
            idx = np.random.randint(len(self))
            question,evidence = self.get_data(idx)
            if fail_count>10:
                raise NotImplementedError("too many fails~")
        return dict(text=question, text_pos=evidence)

dataset = UnarXive_Question_Sentense_Dataset(rawdata)

'''
# 指定训练的模型
finetuner = FineTuner.from_pretrained('sentence-transformers/all-MiniLM-L12-v2', dataset=dataset)

#output_dir = '/mnt/lustre/weigengchen/checkpoints/sbert/unarXive_quantum_ph/all-MiniLM-L12-v2/finetune-8-21'

fintuned_model = finetuner.run(epochs=10, batch_size=2048,
                               log_with="wandb",num_max_checkpoints=50,
                               save_on_epoch_end=True, shuffle=True)
'''