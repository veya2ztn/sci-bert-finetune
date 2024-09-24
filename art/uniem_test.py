import pandas as pd
import h5py
import numpy as np
import os 

os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')
from torch.utils.data import Dataset
from uniem.data_structures import RecordType, PairRecord, TripletRecord, ScoredPairRecord
from tqdm import tqdm

class UnarXive_KeyWord_Abstract_Dataset(Dataset):
    def __init__(self, rawdata=None):
        self.rawdata      = pd.read_csv("data/unarXive/unArXiv.key_word_from_abstract.csv",low_memory=False)
    def __len__(self):
        return len(self.rawdata)
    
    def get_data(self,idx):
        paper_id, abstract = self.rawdata.iloc[idx][['paper_id','abstract']]
        abstract = str(abstract).replace("\n"," ")
        key_words = []
        for key in self.rawdata.keys():
            if 'key_word' in key:
                word = str(self.rawdata.iloc[idx][key])
                if word is None:continue
                if word == 'nan':continue
                key_words.append(word)
        index = np.random.randint(0,len(key_words))
        key_word = key_words[index]
    
        return key_word,abstract
    def __getitem__(self, idx):
        question,evidence = self.get_data(idx)
        fail_count = 0
        while len(evidence.split())>1024 or len(evidence.split())<10:
            fail_count+=1
            idx = np.random.randint(len(self.rawdata))
            question,evidence = self.get_data(idx)
            if fail_count>10:
                raise NotImplementedError("too many fails~")
        return dict(text=question, text_pos=evidence)
dataset = UnarXive_KeyWord_Abstract_Dataset()

# rawdata = pd.read_csv("data/unarXive.clear/query.question.results.good_questions.csv")
# sectionsf = h5py.File('data/unarXive.clear/unarXive.clear.sections.h5', 'r')
# class UnarXive_Question_Sentense_Dataset(Dataset):
#     def __init__(self, rawdata):
#         self.rawdata      = rawdata
#     def __len__(self):
#         return len(self.rawdata)
    
#     def get_data(self,idx):
#         paper_id, sentense_id , question = self.rawdata.iloc[idx][['paper_id','sentense_id','question']]
#         evidence = sectionsf.get(f'{paper_id}/{sentense_id}')[()].decode('utf-8')
#         return question,evidence
#     def __getitem__(self, idx):
#         question,evidence = self.get_data(idx)
#         fail_count = 0
#         while len(evidence.split())>1024:
#             fail_count+=1
#             idx = np.random.randint(len(self.rawdata))
#             question,evidence = self.get_data(idx)
#             if fail_count>10:
#                 raise NotImplementedError("too many fails~")
#         return dict(text=question, text_pos=evidence)

# dataset = UnarXive_Question_Sentense_Dataset(rawdata)



from uniem.finetuner import FineTuner
finetuner = FineTuner.from_pretrained('pretrain_weights/moka-ai_m3e/moka-ai_m3e-base', dataset=dataset)
# for i in tqdm(range(len(dataset))):
#     data = dataset[i]
#     text = finetuner.tokenizer(data['text'])
#     evdi = finetuner.tokenizer(data['text_pos'])

import time
fintuned_model = finetuner.run(epochs=40, output_dir=f'checkpoints/uniem/unarXive.keyword.abstract/finetuned-m3e-base-{time.strftime("%m_%d_%H_%M")}',
                               batch_size=64,log_with="tensorboard",num_max_checkpoints=10,
                               save_on_epoch_end=True)