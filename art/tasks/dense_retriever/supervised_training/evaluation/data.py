from abc import ABC
import csv
from collections import OrderedDict
import torch
from torch.utils.data import DataLoader
import numpy as np
from torch.utils.data import Dataset
from megatron.data.samplers import DistributedBatchSampler
from megatron import print_rank_0, mpu
from megatron.global_vars import get_args, get_tokenizer
from megatron.data.mask_creation_utils import make_attention_mask


def get_qa_dataset(qa_file, split):
    args = get_args()
    tokenizer = get_tokenizer()

    dataset = QADataset("{} Split".format(split),
                        "Question-Answer Pairs",
                        qa_file,
                        tokenizer,
                        args.seq_length_retriever)
    return dataset


def process_qa_batch(batch):
    query_tokens = batch['token_ids'].long().cuda()
    query_mask = (batch['token_mask'] < 0.5).cuda()
    query_types = batch['token_types'].long().cuda()
    query_len = batch['seq_len'].long().cuda()
    reference = batch['reference']
    return query_tokens, query_mask, query_types, query_len, reference


class CustomDataLoader(DataLoader):
    def __init__(self, dataset, eval=False, **kwargs):
        if kwargs.get('collate_fn', None) is None:
            kwargs['collate_fn'] = self._collate_fn
        self.eval = eval
        super().__init__(dataset, **kwargs)

    def _collate_fn(self, batch_data):
        # generate batch
        batch_size = len(batch_data)
        if batch_size == 0:
            raise StopIteration

        tensorized = OrderedDict()
        for d in batch_data:
            for k, v in d.items():
                tensorized.setdefault(k, []).append(v)
        assert len(tensorized) == 5

        tensorized['token_ids'] = torch.LongTensor(np.array(tensorized['token_ids']))
        tensorized['token_mask'] = torch.LongTensor(np.array(tensorized['token_mask']))
        tensorized['token_types'] = torch.LongTensor(np.array(tensorized['token_types']))
        tensorized['seq_len'] = torch.LongTensor(tensorized['seq_len'])
        return tensorized


def get_one_epoch_qa_dataloader(dataset, batch_size=None):
    """Data loader. Note that batch-size is the local (per GPU) batch-size.
       NOTE: This dataloader is not distributed !!!
    """

    args = get_args()

    world_size = mpu.get_data_parallel_world_size()
    rank = mpu.get_data_parallel_rank()

    if batch_size is None:
        batch_size = args.batch_size
    num_workers = args.num_workers
    global_batch_size = batch_size * world_size

    sampler = torch.utils.data.SequentialSampler(dataset)

    # importantly, drop_last must be False to get all the data.
    batch_sampler = DistributedBatchSampler(sampler,
                                            batch_size=global_batch_size,
                                            drop_last=False,
                                            rank=rank,
                                            world_size=world_size)

    # Data loader. Note that batch size is the per GPU batch size.
    data_loader = CustomDataLoader(dataset,
                                   batch_sampler=batch_sampler,
                                   num_workers=num_workers,
                                   pin_memory=True)
    return data_loader


def build_tokens_types_paddings_from_text(src_text, tokenizer, max_seq_length):
    """Build token types and paddings, trim if needed, and pad if needed."""

    src_text_ids = tokenizer.tokenize(src_text)

    return build_tokens_types_paddings_from_ids(src_text_ids,
                                                max_seq_length,
                                                tokenizer.cls,
                                                tokenizer.sep,
                                                tokenizer.pad)


def build_tokens_types_paddings_from_ids(src_ids, max_seq_length, cls_id, sep_id, pad_id):
    # TODO: Design modular interface to reuse this function. This is getting repeated multiple times in different tasks
    """Build token types and paddings, trim if needed, and pad if needed."""

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


def build_sample(token_ids, token_types, num_tokens, reference):
    """Convert to numpy and return a sample consumed by the batch producer."""

    token_ids = np.array(token_ids, dtype=np.int64)
    token_types = np.array(token_types, dtype=np.int64)
    token_mask = make_attention_mask(token_ids, token_ids)

    sample = ({
        'token_ids': token_ids,
        'token_mask': token_mask,
        'token_types': token_types,
        'seq_len': num_tokens,
        'reference': reference
    })
    return sample


class QADataset(ABC, Dataset):
    """Open-Retrieval Question Answer pairs dataset."""

    def __init__(self, task_name, dataset_name, datapath,
                 tokenizer, max_seq_length):
        # Store inputs.
        self.task_name = task_name
        self.dataset_name = dataset_name
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
        print_rank_0(' > building {} dataset for {}:'.format(self.task_name,
                                                             self.dataset_name))
        print_rank_0(datapath)
        self.samples = self.process_samples_from_single_path(datapath)
        print_rank_0('  >> total number of samples: {}'.format(len(self.samples)))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        raw_sample = self.samples[idx]

        ques_tokens, tokentypes_enc, num_tokens_ques = build_tokens_types_paddings_from_text(raw_sample['question'],
                                                                                        self.tokenizer,
                                                                                        self.max_seq_length)
        sample = build_sample(ques_tokens,
                              tokentypes_enc,
                              num_tokens_ques,
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

                sample = {'question': question, 'answers': answers}
                total += 1
                samples.append(sample)

                # if total % 1000 == 0:
                #     print_rank_0('  > processed {} so far ...'.format(total))

        print_rank_0(' >> processed {} samples.'.format(len(samples)))
        return samples
