from abc import ABC
import csv
import numpy as np
from torch.utils.data import Dataset
from megatron import print_rank_0
from megatron.global_vars import get_args
from megatron.data.mask_creation_utils import make_attention_mask
import pandas as pd 

def build_tokens_types_paddings_from_text(src_text, bert_tokenizer, max_seq_length):
    """Build token types and paddings, trim if needed, and pad if needed."""

    src_text_ids = bert_tokenizer.tokenize(src_text)
    return build_tokens_types_paddings_from_ids(src_text_ids,
                                                max_seq_length,
                                                bert_tokenizer.cls,
                                                bert_tokenizer.sep,
                                                bert_tokenizer.pad)


def build_tokens_types_paddings_from_ids(src_ids, max_seq_length, cls_id, sep_id, pad_id):
    enc_ids = []
    tokentypes_enc = []

    # [CLS].
    enc_ids.append(cls_id)
    tokentypes_enc.append(0)

    # A.
    len_src = len(src_ids)
    enc_ids.extend(src_ids)
    tokentypes_enc.extend([0] * len_src)

    # Cap the size.
    if len(enc_ids) > max_seq_length - 1:
        enc_ids = enc_ids[0: max_seq_length - 1]
        tokentypes_enc = tokentypes_enc[0: max_seq_length - 1]

    # [SEP].
    enc_ids.append(sep_id)
    tokentypes_enc.append(0)

    num_tokens_enc = len(enc_ids)
    # Padding.
    padding_length = max_seq_length - len(enc_ids)
    if padding_length > 0:
        enc_ids.extend([pad_id] * padding_length)
        tokentypes_enc.extend([pad_id] * padding_length)

    return enc_ids, tokentypes_enc, num_tokens_enc


def build_sample(query_uid, token_ids, token_types, num_tokens, prefixed_query_text, reference):
    token_ids = np.array(token_ids, dtype=np.int64)
    token_types = np.array(token_types, dtype=np.int64)
    token_mask = make_attention_mask(token_ids, token_ids)

    sample = ({
        'query_uid': query_uid,
        'query_ids_bert': token_ids,
        'query_types': token_types,
        'query_mask_bert': token_mask,
        'prefixed_query_text': prefixed_query_text,
        'reference': reference
    })
    return sample


class OpenQADataset(ABC, Dataset):

    def __init__(self, task_name, dataset_name, datapaths,
                 bert_tokenizer,
                 max_seq_length):
        self.args = args = get_args()
        self.np_rng = np.random.RandomState(seed=args.seed)
        self.task_name = task_name
        self.dataset_name = dataset_name
        self.bert_tokenizer = bert_tokenizer
        self.max_seq_length = max_seq_length
        print_rank_0(' > building {} dataset for {}:'.format(self.task_name,
                                                             self.dataset_name))
        string = '  > paths:'
        for path in datapaths:
            string += ' ' + path
        print_rank_0(string)
        self.samples = []
        for datapath in datapaths:
            self.samples.extend(self.process_samples_from_single_path(datapath))

    def __len__(self):
        return len(self.samples)

    def get_item(self,idx):
        raw_sample = self.samples[idx]
        prefixed_query_text = "{} {}".format("Question:", raw_sample['question'])
        if 'vicuna' in self.args.hf_model_name:
            prefixed_query_text= prefixed_query_text.replace('Question: what', 'Question: What'
                            ).replace('Question: who','Question: Who'
                            ).replace('Question: how','Question: How'
                            ).replace('Question: why','Question: Why'
                            ).replace('Question: where','Question: Where'
                            ).replace('Question: when','Question: When'
                            )
        ques_tokens, tokentypes_enc, num_tokens_ques = build_tokens_types_paddings_from_text(raw_sample['question'],self.bert_tokenizer,self.max_seq_length)
        return raw_sample, ques_tokens, tokentypes_enc, num_tokens_ques, prefixed_query_text
    
    def __getitem__(self, idx):
        raw_sample, ques_tokens, tokentypes_enc, num_tokens_ques, prefixed_query_text = self.get_item(idx)
        
        sample = build_sample(raw_sample['uid'],
                              ques_tokens,
                              tokentypes_enc,
                              num_tokens_ques,
                              prefixed_query_text,
                              raw_sample['answers'])
        return sample

    @staticmethod
    def process_samples_from_single_path(filename):
        print_rank_0(' > Processing {} ...'.format(filename))
        samples = []
        total = 0

        with open(filename, 'r') as ifile:
            reader = csv.reader(ifile, delimiter='\t')
            for row in reader:
                question = row[0]
                answers = eval(row[1])

                total += 1
                # We are keeping the uid as negative to avoid the conflict with evidence ID
                sample = {'uid': -1 * total, 'question': question, 'answers': answers}
                samples.append(sample)

                # if total % 1000 == 0:
                #     print_rank_0('  > processed {} so far ...'.format(total))

        print_rank_0(' >> processed {} samples.'.format(len(samples)))
        return samples


class WhatIsQuestionDataset(ABC, Dataset):

    def __init__(self, task_name, dataset_name, datapaths,bert_tokenizer,
                 max_seq_length):
        self.args = args  = get_args()
        self.np_rng       = np.random.RandomState(seed=args.seed)
        self.task_name    = f"[What is?]"
        self.dataset_name = f"[Judger~~~!]"
        self.bert_tokenizer = bert_tokenizer
        self.max_seq_length = max_seq_length
        print_rank_0(' > building {} dataset for {}:'.format(self.task_name,self.dataset_name))
        string = '  > paths:'
        for path in datapaths:string += ' ' + path
        print_rank_0(string)
        self.samples = []
        for datapath in datapaths:
            self.samples.extend(self.process_samples_from_single_path(datapath))

    def __len__(self):
        return len(self.samples)

    def get_item(self,idx):
        paper_id, key_word = self.samples[idx]
        prefixed_query_text = f"What is {key_word}"
        ques_tokens, tokentypes_enc, num_tokens_ques = build_tokens_types_paddings_from_text(
            key_word,self.bert_tokenizer,self.max_seq_length)
        return paper_id, ques_tokens, tokentypes_enc, num_tokens_ques, prefixed_query_text
    
    def __getitem__(self, idx):
        paper_id, ques_tokens, tokentypes_enc, num_tokens_ques, prefixed_query_text = self.get_item(idx)
        
        sample = build_sample(idx,
                              ques_tokens,
                              tokentypes_enc,
                              num_tokens_ques,
                              prefixed_query_text,
                              paper_id)
        return sample

    @staticmethod
    def process_samples_from_single_path(filename):
        
        print_rank_0(' > Processing {} ...'.format(filename))
        #data = pd.read_csv('data/unArXiv.key_word_from_abstract.csv')
        data = pd.read_csv(filename)
        samples = []
        keys =[ k for k in data.keys() if 'key_word' in k ]
        for key in keys:
            samples+=data[['paper_id',key]].values.tolist()
        samples = [[a,b] for a,b in samples if isinstance(b,str)]
        print_rank_0(' >> processed {} samples.'.format(len(samples)))
        return samples