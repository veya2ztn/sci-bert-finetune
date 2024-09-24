import pandas as pd
import numpy as np 
from torch.utils.data import Dataset, DataLoader
from typing import List, Dict, Any, Union, Optional
import torch
import os
import json
from tqdm.auto import tqdm

class Dummy_Question_Sentense_Dataset(Dataset):
    def __init__(self):
        pass
    def __len__(self):
        return 9600
    
    def __getitem__(self, idx):
        return dict(text=fake.pystr(min_chars=10,max_chars=30), 
                    text_pos=fake.pystr(min_chars=100,max_chars=300))


def preprocess(string, tokenizer):
        input_ids = tokenizer(
            string,
            return_tensors="pt",
            padding="max_length",
            max_length=tokenizer.model_max_length,
            truncation=True,
        ).input_ids
        return input_ids

# class LazyPairDatasetWithIndex(Dataset):
#     """Dataset for supervised fine-tuning."""

#     def __init__(self, raw_data, tokenizer: transformers.PreTrainedTokenizer):
#         super().__init__()
#         self.tokenizer = tokenizer
#         self.raw_data = raw_data
#         self.cached_data_dict = {}

#     def __len__(self):
#         return len(self.raw_data)

    
#     def __getitem__(self, i) -> Dict[str, torch.Tensor]:
#         if i in self.cached_data_dict:
#             return self.cached_data_dict[i]

#         question_ids = preprocess(self.raw_data.iloc[i]["question"], self.tokenizer)
#         answer_ids   = preprocess(self.raw_data.iloc[i]["answer"],   self.tokenizer)
#         question_index=self.raw_data.iloc[i]["question_index"]
#         answer_index=self.raw_data.iloc[i]["answer_index"]

#         ret = {
#             'question_index': question_index,
#             'answer_index': answer_index,
#             'question_ids': question_ids,
#             'answer_ids': answer_ids,
#         }
#         self.cached_data_dict[i] = ret
        
#         return ret

def create_entire_question_answer_mapping_from_pair(question_answer_pair):
    question_to_answer  = defaultdict(list)
    answer_to_question  = defaultdict(list)
    question_to_pos     = {}
    answer_to_pos       = {}
    pos_to_question     = []
    pos_to_answer       = []
    for i,(k,v) in enumerate(question_answer_pair):
        question_to_answer[k].append(v)
        answer_to_question[v].append(k)
        if k not in question_to_pos:
            question_to_pos[k] = len(question_to_pos) # map question1 into 1
            pos_to_question.append(k) # map 1 into question1
        if v not in answer_to_pos:  
            answer_to_pos[v] = len(answer_to_pos)
            pos_to_answer.append(v)
    return question_to_answer, answer_to_question, question_to_pos, answer_to_pos, pos_to_question, pos_to_answer

from collections import defaultdict
from memory import KeyTensorMemory
class FullPaperDatasetWithIndex(Dataset):
    """
    Dataset maintain its own table regist of the unique id of question and answer Pair
    For example, 
    [[question1, answer1],
     [question2, answer2],
     ......
     [questionN, answerN]]
    The unique id should be any hashable object, such as int, str, tuple, etc. 
    The return value is 
    {
        'question_index': question_token_index_in_this_dataset,
        'answer_index'  : answer_token_index_in_this_dataset,
        'question_ids'  : question_token,
        'answer_ids'    : answer_token,
    }
    Then the dataset should also provide the mapping that convert the index into its real unique_id
    """
    def __init__(self, question_answer_pair, question_token_path, answer_token_path,
                 dummy=False,
                 mapping_system = None):
        super().__init__()
        self.question_answer_pair= question_answer_pair
        if mapping_system is None:
            mapping_system = create_entire_question_answer_mapping_from_pair(question_answer_pair)
        (self.question_to_answer, self.answer_to_question, 
             question_to_pos, answer_to_pos, 
             pos_to_question, pos_to_answer) = mapping_system
        
        self.question_to_pos = question_to_pos
        self.answer_to_pos   = answer_to_pos
        self.pos_to_question = pos_to_question
        self.pos_to_answer   = pos_to_answer
        ### thus the dataset only allocate the resource from the question_to_pos.keys() and answer_to_pos.keys()
        
        if not dummy:
            used_question_keys   = list(self.question_to_answer.keys())
            #print(f"used_question_keys ==> {len(used_question_keys)} ==> {np.max(used_question_keys)}")
            self.question_memory = self.allocate_resource(question_token_path, used_question_keys)
            used_answer_keys     = list(self.answer_to_question.keys())
            #print(f"used_answer_keys ==> {len(used_answer_keys)} ==> {np.max(used_answer_keys)}")
            self.answer_memory   = self.allocate_resource(answer_token_path, used_answer_keys)
            assert question_to_pos == self.question_memory.index_to_pos, 'check the question_to_pos'
            assert answer_to_pos   == self.answer_memory.index_to_pos,  'check the answer_to_pos'


    @staticmethod
    def allocate_resource(global_dataset:Union[List[str],str], ordered_keys):

        
        if isinstance(global_dataset, str):global_dataset = [global_dataset]

        if global_dataset[0][-4:] == '.npy': ### then it is a idx, array dataset
            global_memory = KeyTensorMemory.load_from_path(global_dataset)
            return global_memory.clip_via_keys(ordered_keys)
        else:
            raise NotImplementedError

    @staticmethod
    def get_filted_answer_path(max_token_set, ROOTDIR):
        tokenlimit = [0, 4000, 8000, 12000, 16000, 24000, 28000, 32000]
        paths  = []
        for i in range(len(tokenlimit)-1):
            start = tokenlimit[i]
            end   = tokenlimit[i+1]
            if end > max_token_set:break
            paths.append(os.path.join(ROOTDIR, 'llama_answer_token', f'llama_answer_token_{start:05d}_{end:05d}.npy'))
        return paths
    
    @staticmethod
    def get_QA_index_mapping(question_to_answer, filted_answer_path_list):
        answer_indexs = []
        assert len(filted_answer_path_list)>0, "no data!!!"
        #print(filted_answer_path_list)
        for _path in filted_answer_path_list:
            the_index  = np.load(_path.replace('.npy','.key.npy'))
            answer_indexs.append(the_index)
        answer_indexs_set= set(np.concatenate(answer_indexs))
        filter_question_to_answer = {}
        filter_answer_to_question = {}
        for question_id, answer_id in tqdm(question_to_answer.items(), desc='filtering QA index mapping'):
            if answer_id not in answer_indexs_set:continue
            filter_question_to_answer[question_id] = answer_id
            if answer_id not in filter_answer_to_question:filter_answer_to_question[answer_id]=[]
            filter_answer_to_question[answer_id].append(question_id)
        return filter_question_to_answer,filter_answer_to_question
       

    def __len__(self):
        return len(self.question_answer_pair)
    
    def checkdata(self):
        ROOTDIR='data/unarXive_quantum_physics/'
        questions_df = pd.read_csv(os.path.join(ROOTDIR,"query_full_paper.question.good_questions.csv"))
        print("loading csv files...........")
        sentense_ids = pd.read_csv(os.path.join(ROOTDIR,"unarXive_quantum_physics.clear.sections.id.csv"))
        sentense_ids = list(sentense_ids.groupby('paper_id'))
        print("done~!")
        for i in tqdm(range(len(self.question_to_answer))):
            data = self.__getitem__(i)
            question_index  = self.question_memory.get_key_of_index(data['question_index'])
            answer_index    = self.answer_memory.get_key_of_index(data['answer_index'])
            raw             = questions_df.iloc[question_index]
            paper_unique_id = raw['paper_id']
            paper_real_id,_ = sentense_ids[answer_index]
            assert paper_unique_id == paper_real_id, f"paper_unique_id {paper_unique_id} != paper_real_id {paper_real_id}"


    def get_keys_of_question(self, index):
        return self.question_memory.get_key_of_index(index)
    

    def get_keys_of_answer(self, index):
        return self.answer_memory.get_key_of_index(index)


    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        question_key , answer_key = self.question_answer_pair[i]
        question_token = self.question_memory.get_tensor_of_key(question_key)
        answer_token   = self.answer_memory.get_tensor_of_key(answer_key)
        
        question_token_index = self.question_memory.get_index_of_key(question_key)
        answer_token_index   = self.answer_memory.get_index_of_key(answer_key)

        ret = {
            'question_index': question_token_index,
            'answer_index':   answer_token_index,
            'question_ids':   torch.LongTensor(question_token.astype('int')).unsqueeze(0),
            'answer_ids':     torch.LongTensor(answer_token.astype('int')).unsqueeze(0),
        }
        
        
        return ret

import time
def load_wave_data(wavedata_path_list):
    if not isinstance(wavedata_path_list, list):
        return None,h5py.File(wavedata_path_list, 'r')
    data_list = []
    index_list = []
    for data_path in tqdm(wavedata_path_list):
        #tqdm.write(data_path)
        data = np.load(data_path)
        data_list.append(data)
        index_path = data_path.replace(".verify.npy", ".npy").replace(".npy", ".idx.npy")
        
        index_list.append(np.load(index_path, allow_pickle=True))
        #tqdm.write(f"done!", end='\n')
    print("concat~!")
    now = time.time()
    data_list = np.concatenate(data_list) if len(data_list) > 1 else data_list[0]
    index_list = np.concatenate(index_list) if len(index_list) > 1 else index_list[0]
    index_map = dict([[name, i] for i, name in enumerate(index_list)])
    print(f"done, cost {time.time() - now}")
    return index_map, data_list

class PaperDatasetWithReference(Dataset):
    """
    Dataset maintain its own table regist of the unique id of question and answer Pair
    For example, 
    [[question1, answer1],
     [question2, answer2],
     ......
     [questionN, answerN]]
    The unique id should be any hashable object, such as int, str, tuple, etc. 
    The return value is 
    {
        'question_index': question_token_index_in_this_dataset,
        'answer_index'  : answer_token_index_in_this_dataset,
        'question_ids'  : question_token,
        'answer_ids'    : answer_token,
    }
    Then the dataset should also provide the mapping that convert the index into its real unique_id
    """
    def __init__(self, question_answer_pair, 
                       question_token_path, answer_token_path, 
                       reference_question_embedding_path,
                       reference_answer_embedding_path,
                       tokenizer,
                       dummy=False):
        super().__init__()
        self.question_answer_pair= question_answer_pair
        ### thus the dataset only allocate the resource from the question_to_pos.keys() and answer_to_pos.keys()
        self.dummy = dummy
        self.tokenizer = tokenizer
        self.question_tokens_unique_id =  self.question_tokens = self.answer_tokens_unique_id = self.answer_tokens = None
        if (not dummy or dummy =="only_question"):
            self.question_tokens_unique_id, self.question_tokens     =  self.load_resource(question_token_path, flag='question')  
        if (not dummy or dummy =="only_answer"):
            self.answer_tokens_unique_id  , self.answer_tokens       =  self.load_resource(answer_token_path, flag='paper')  

        if self.question_tokens_unique_id is not None:
            self.question_index2key = {i:k for k,i in self.question_tokens_unique_id.items()}
        if self.answer_tokens_unique_id is not None:
            self.answer_index2key = {i:k for k,i in self.answer_tokens_unique_id.items()}

        if reference_question_embedding_path is not None and not dummy:
            self.ref_question_unique_id   , self.ref_question_embedding =  self.load_resource(reference_question_embedding_path, flag='numpy')
            self.ref_answer_unique_id     , self.ref_answer_embedding   =  self.load_resource(reference_answer_embedding_path, flag='numpy')
        else:
            self.ref_question_unique_id   , self.ref_question_embedding = None, None
            self.ref_answer_unique_id     , self.ref_answer_embedding   = None, None

    @staticmethod
    def load_resource(path, flag='question'):
        if isinstance(path, str):
            if path.endswith('.npy'):
                index = np.load(path.replace('.npy','.idx.npy'))
                index_map = {name:i for i, name in enumerate(index)}
                return index_map, np.load(path) ## the index and the array
            elif path.endswith('.csv'):
                df = pd.read_csv(path)
                index = df[f'{flag}_index'].values 
                index_map = {name:i for i, name in enumerate(index)}
                output = [index_map]
                if flag in df:
                    output.append(df[flag].values)
                elif 'text' in df:
                    output.append(df['text'].values)
                elif 'content' in df:
                    output.append(df['content'].values)
                else:
                    raise NotImplementedError
                return output
            else:
                raise NotImplementedError
        elif isinstance(path, list):
            
            if path[0].endswith('.npy'):
                return load_wave_data(path)
             
            else:
                raise NotImplementedError

    def __len__(self):
        return len(self.question_answer_pair)
    
    @property
    def extra_answer_needed_keys(self):
        return []
    
    @property
    def whole_answer_index(self):
        return np.arange(len(self.answer_tokens_unique_id))
    
    @property
    def whole_answer_token(self):
        return self.answer_tokens


    def cache_whole_the_tokens(self):
        """
        This is used for cache the whole string tokens.
        However, it is better use a  Dataloader to load once the data which will auto multiprocess.
        """
        pass
    
    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        if self.dummy:
            return {
                'question_index': torch.randint(0, 768, (1,)),
                'answer_index':   torch.randint(0, 768, (1,)),
                'reference_question_embedding': torch.randn(1536),
                'reference_answer_embedding'  : torch.randn(1536),
                'question_ids':   torch.randint(0, 768, (8000,)),
                'answer_ids':     torch.randint(0, 768, (8000,)),
                }

        question_key , answer_key = self.question_answer_pair[i]
        
        question_index = self.question_tokens_unique_id[question_key]
        question_token = self.question_tokens[question_index]
        question_index = [question_index]
        if isinstance(answer_key,list):
            answer_index = [self.answer_tokens_unique_id[k] for k in answer_key]
            answer_token = [self.answer_tokens[i] for i in answer_index]
        else:
            answer_index = self.answer_tokens_unique_id[answer_key]
            answer_token = self.answer_tokens[answer_index]
            answer_index = [answer_index]

        if isinstance(question_token, str):
            question_token = preprocess(question_token, self.tokenizer)
        else:
            question_token = torch.LongTensor(question_token.astype('int'))
        
        if isinstance(answer_token, str):
            answer_token = preprocess(answer_token, self.tokenizer)
        elif isinstance(answer_token, list) and isinstance(answer_token[0], str):
            answer_token = [preprocess(t, self.tokenizer) for t in answer_token]
        elif isinstance(answer_token, list) and isinstance(answer_token[0], np.ndarray):
            answer_token = torch.stack([torch.LongTensor(t.astype('int')) for t in answer_token])
        else:
            answer_token = torch.LongTensor(answer_token.astype('int'))
        
        ret = {
            'question_ids':   question_token,
            'answer_ids':     answer_token,
            'question_index':   torch.LongTensor(question_index), ## question_key may not be int, lets record the index
            'answer_index':   torch.LongTensor(answer_index),
        }
        if self.ref_question_embedding is not None:
            ref_question_embedding = self.ref_question_embedding[self.ref_question_unique_id[question_key]]
            ref_answer_embedding   = self.ref_answer_embedding[self.ref_answer_unique_id[answer_key]]
            ret['reference_question_embedding'] = torch.from_numpy(ref_question_embedding)
            ret['reference_answer_embedding']   = torch.from_numpy(ref_answer_embedding)
        else:
            ## return the pair order 
            ret['reference_question_embedding'] = torch.LongTensor([i])
            ret['reference_answer_embedding']   = torch.LongTensor([i])
        return ret


class QADataCollatorWithNegative:
    def __init__(self, buffer, negative_sampler_num):
        self.buffer = buffer
        self.negative_sampler_num =negative_sampler_num

    def __call__(self, records: List[Dict[str, Any]]) -> Dict[str, Any]:
        question_index  = [record['question_index'] for record in records]
        answer_index    = [record['answer_index']   for record in records]
        question_ids    = [record['question_ids']   for record in records]
        answer_ids      = [record['answer_ids']     for record in records]
        
        # extra_question_index     = torch.randint(0, self.buffer.shape['question'][0], (self.negative_sampler_num,))
        # extra_answer_index       = torch.randint(0, self.buffer.shape['answer'][0], (self.negative_sampler_num,))
        # extra_question_embedding = torch.randn((self.negative_sampler_num, self.buffer.shape['question'][1]))
        # extra_answer_embedding   = torch.randn((self.negative_sampler_num, self.buffer.shape['answer'][1])) 
        (extra_question_index, extra_question_embedding,
         extra_answer_index  ,   extra_answer_embedding)= self.buffer.get_cached_question_and_answer(
                                        self.negative_sampler_num, 
                                        exclude_question_indexes=question_index,
                                        exclude_answer_indexes  =answer_index)

        question_index  = torch.LongTensor(question_index)
        answer_index    = torch.LongTensor(answer_index)
        question_ids    = torch.cat(question_ids)
        answer_ids      = torch.cat(answer_ids)
        sample =  {
            'question_index':question_index,
            'answer_index':answer_index,
            'question_ids': question_ids,
            'answer_ids': answer_ids,
            'extra_answer_index':extra_answer_index,
            'extra_answer_embedding':extra_answer_embedding
        }
        if extra_question_index is not None:
            sample['extra_question_index']=extra_question_index
            sample['extra_question_embedding']=extra_question_embedding
        return sample

from transformers.trainer import DataCollatorWithPadding
class QADataCollatorWithPadding(DataCollatorWithPadding):

    def __call__(self, records: List[Dict[str, Any]]) -> Dict[str, Any]:
        question_index  = [record['question_index'] for record in records]
        answer_index    = [record['answer_index']   for record in records]
        question_ids    = [record['question_ids']   for record in records]
        answer_ids      = [record['answer_ids']     for record in records]
    

        question_index  = torch.LongTensor(question_index)
        answer_index    = torch.LongTensor(answer_index)
        question_ids    = torch.cat(question_ids)
        answer_ids      = torch.cat(answer_ids)
        question_token_size = question_ids.shape[1]
        model_max_length = self.tokenizer.model_max_length
        if question_token_size < model_max_length:
            question_ids = torch.nn.functional.pad(question_ids,(0,model_max_length-question_token_size,0,0))
        
        return {
            'question_index':question_index,
            'answer_index':answer_index,
            'question_ids': question_ids,
            'answer_ids': answer_ids,
        }


class QADataCollatorWithPaddingReference(DataCollatorWithPadding):

    def __call__(self, records: List[Dict[str, Any]]) -> Dict[str, Any]:
        # keys = records[0].keys()
        # ret = {k:torch.stack([record[k] for record in records]) for k in keys}

        reference_question_embedding= torch.stack([record['reference_question_embedding'] for record in records]) if 'reference_question_embedding' in records[0] else None 
        reference_answer_embedding  = torch.stack([record['reference_answer_embedding']   for record in records]) if 'reference_answer_embedding' in records[0] else None 

        question_ids    = torch.stack([record['question_ids']   for record in records])
        question_index  = torch.stack([record['question_index']   for record in records])

        

        if len(records[0]['answer_ids'].shape) == 2: # (S,L_max)
            max_length   = max([len(record['answer_ids']) for record in records])
            answer_index   = torch.stack([torch.nn.functional.pad(record['answer_index'],(0, max_length-len(record['answer_index'])), value=-1) for record in records])
            answer_ids = torch.stack([torch.nn.functional.pad(record['answer_ids'],  (0,0, 0, max_length-len(record['answer_ids'])), value=-1) for record in records])
        else:
            answer_ids      = torch.stack([record['answer_ids']     for record in records])
            answer_index      = torch.stack([record['answer_index']     for record in records])
        ret = {
            'question_ids':question_ids,
            'answer_ids':answer_ids,
            'question_index':question_index,
            'answer_index':answer_index
        }

        if reference_question_embedding is not None:
            ret['reference_question_embedding']=reference_question_embedding
            ret['reference_answer_embedding']  =reference_answer_embedding
            
        return ret
    
if __name__ == "__main__":
    
    # TPATH="data/unarXive_quantum_physics/query_full_paper.question_answer_map.json"
    # with open(TPATH,'r') as f:data = json.load(f)
    # question_to_answer={}
    # for i,(key, val) in enumerate(data.items()):
    #     question_to_answer[int(key)] = int(val)
        
    # tokenizer = transformers.AutoTokenizer.from_pretrained(
    #     "../pretrain_weights/vicuna/vicuna-7b-v1.5-16k/",
    #     model_max_length=128,
    #     padding_side="right",
    #     use_fast=False,
    # )
    # if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.unk_token


    max_token_set = 16000
    
    ROOTDIR = 'data/unarXive_quantum_physics/'
    with open(os.path.join(ROOTDIR,'query_full_paper.question_answer_map.json'),'r') as f:
        question_to_answer = json.load(f)
    question_to_answer = dict([(int(k),int(v)) for k,v in question_to_answer.items()])
    answer_token_path = FullPaperDatasetWithIndex.get_filted_answer_path(max_token_set, ROOTDIR = ROOTDIR)
    question_token_path = os.path.join(ROOTDIR, 'llama_question_token/llama_question_token.npy')
    
    filter_question_to_answer,filter_answer_to_question = FullPaperDatasetWithIndex.get_QA_index_mapping(
        question_to_answer, answer_token_path
    )
    question_answer_pair = list(filter_question_to_answer.items())
    train_dataset = FullPaperDatasetWithIndex(question_answer_pair, question_token_path, answer_token_path,
                                              dummy=False)
    print(f"dataset created!!")
    train_dataset.checkdata()
    # #dataloader = DataLoader(dataset=datasets,batch_size=32,num_workers=16)
    # for _ in tqdm(dataloader):
    #     pass