
from transformers import AutoTokenizer
from uniem.finetuner import FineTuner
from uniem.training_strategy import BitFitTrainging
from uniem.model import PoolingStrategy, create_uniem_embedder
from transformers import AutoTokenizer, AutoModelForCausalLM,LlamaForCausalLM
import torch
import pandas as pd
from torch.utils.data import Dataset
import h5py
import numpy as np
import os,time
from datasets import load_dataset

from fastchat.train.llama_flash_attn_monkey_patch import (
    replace_llama_attn_with_flash_attn,
)
replace_llama_attn_with_flash_attn()

os.environ['TOKENIZERS_PARALLELISM']='false'
# rawdata = pd.read_csv("data/unarXive.clear/query.question.results/query.question.results.good_questions1.csv")
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

# class Dummy_Question_Sentense_Dataset(Dataset):
#     def __init__(self, dataset=None):
#         self.dataset = dataset
#     def __len__(self):
#         return len(self.dataset['train']) if self.dataset is None else 9600
    
#     def __getitem__(self, idx):
#         return dict(text=self.dataset['train'][idx]['summary_text'][:100], 
#                     text_pos=self.dataset['train'][idx]['chapter'])
#dataset = Dummy_Question_Sentense_Dataset(load_dataset('/home/zhangtianning/.cache/huggingface/datasets/kmfoda___booksum/'))
    
# df = pd.read_json('data/uniem_example/riddle.jsonl', lines=True)
# df = df.rename(columns={'instruction': 'text', 'output': 'text_pos'})
# dataset=df.to_dict('records')

from faker import Faker
fake = Faker()

class Dummy_Question_Sentense_Dataset(Dataset):
    def __init__(self):
        pass
    def __len__(self):
        return 9600
    
    def __getitem__(self, idx):
        return dict(text=fake.pystr(min_chars=10,max_chars=30), 
                    text_pos=fake.pystr(min_chars=100,max_chars=300))
dataset = Dummy_Question_Sentense_Dataset()   


def modify_llama_model(embedder):
    # Modify the "o_proj" layer in each LlamaDecoderLayer
    for block in embedder:
        for layer in embedder.encoder.model.layers:
            layer.self_attn.o_proj.bias = torch.nn.Parameter(torch.zeros_like(layer.self_attn.o_proj.weight[:,0]))
    # Modify the "gate_proj" layer in each LlamaDecoderLayer
    for layer in embedder.encoder.model.layers:
        layer.mlp.gate_proj.bias = torch.nn.Parameter(torch.zeros_like(layer.mlp.gate_proj.weight[:,0]))
    # Modify the "lm_head" layer
    embedder.encoder.lm_head.bias = torch.nn.Parameter(torch.zeros_like(embedder.encoder.lm_head.weight[:,0]))
    return embedder

#model_path = "pretrain_weights/llama2/llama2-7b-hf/"
model_path = "pretrain_weights/vicuna/vicuna-7b-v1.5-16k"

embedder = create_uniem_embedder(model_path, 
                                 model_class=AutoModelForCausalLM, 
                                 pooling_strategy=PoolingStrategy.llama_last_weighted,
                                 output_hidden_states=True,#torch_dtype=torch.float16
                                 temperature = 1,
                                 top_p = 1
                                 )
tokenizer = AutoTokenizer.from_pretrained(model_path)
embedder  = AutoModelForCausalLM.from_pretrained(model_path)
# pooling_strategy=PoolingStrategy.llama_last_weighted
# from uniem.utils import *
# from uniem.model import *
# #model = model_class.from_pretrained(model_name_or_path,**kargs)  # type: ignore
from transformers import LlamaConfig, LlamaForCausalLM,LlamaTokenizer
# config = LlamaConfig(hidden_size=128,
#         intermediate_size=1024,
#         num_hidden_layers=8)
# model = LlamaForCausalLM(config)
# model = cast(PreTrainedModel, model)
# embedder_cls = StrategyEmbedderClsMap[PoolingStrategy(pooling_strategy)]
# embedder = embedder_cls(model)
# #tokenizer = LlamaTokenizer()
# tokenizer = AutoTokenizer.from_pretrained(model_path)

print("=============== loaded model ====================")

#embedder = modify_llama_model(embedder)
finetuner = FineTuner(embedder, tokenizer, dataset=dataset)
finetuner.tokenizer.pad_token = finetuner.tokenizer.eos_token
finetuner.run(epochs=3, lr=1e-4, training_strategy=BitFitTrainging(),
            output_dir=f'checkpoints/sgpt_llama/unarXive.keyword.abstract/finetuned-m3e-base-{time.strftime("%m_%d_%H_%M")}',
            batch_size=1,log_with="tensorboard",num_max_checkpoints=1000,
            save_on_epoch_end=True)