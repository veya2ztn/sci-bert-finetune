"""Index evidence dataset for use in QA tasks.
   This file is similar to the tools/preprocess_data.py script in the backbones
"""

from megatron.data.pretokenized_evidence import Tokenized_Knowledge_Builder
import argparse
import csv
import multiprocessing
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__),
                                             os.path.pardir)))
import time
import torch
from megatron.tokenizer import build_tokenizer
from megatron.data import indexed_dataset
from tqdm import tqdm

# https://stackoverflow.com/questions/15063936/csv-error-field-larger-than-field-limit-131072
maxInt = sys.maxsize
while True:
    # decrease the maxInt value by factor 10
    # as long as the OverflowError occurs.
    try:
        csv.field_size_limit(maxInt)
        break
    except OverflowError:
        maxInt = int(maxInt/10)


class IdentitySplitter(object):
    def tokenize(self, *text):
        return text

class Encoder(object):
    def __init__(self, args):
        self.args = args

    def initializer(self):
        # Use Encoder class as a container for global data
        Encoder.tokenizer = build_tokenizer(self.args)
        Encoder.splitter  = IdentitySplitter()

    def encode(self, csv_line):
        # line format: doc_id, doc_text, title

        csv_text = csv_line[1]
        csv_title = csv_line[2]
        #print(f"\n csv text is ==>  {csv_line[1]} <==")
        #print(f"\n csv title is ==>  {csv_line[2]} <==")
        ids = {}
        for key in self.args.tsv_keys:
            if key == "text":
                text = csv_text
            elif key == "title":
                text = csv_title
            doc_ids = []
            for sentence in Encoder.splitter.tokenize(text):
                if self.args.tokenizer_type in ['BertWordPieceLowerCase','BertWordPieceCase']:
                    sentence_ids = Encoder.tokenizer.tokenize(sentence)

                    if len(sentence_ids) > 0:
                        doc_ids.append(sentence_ids)
                    else:
                        doc_ids.append([Encoder.tokenizer.cls, Encoder.tokenizer.sep])
                else:
                    sentence_ids = Encoder.tokenizer.encode(sentence)
                    if len(sentence_ids) > 0:
                        doc_ids.append(sentence_ids)
            ids[key] = doc_ids
        #print(f"\n ids ==>  {ids} <==")
        return ids, len(csv_line)

def get_args():
    parser = argparse.ArgumentParser()
    group = parser.add_argument_group(title='input data')
    group.add_argument('--input', type=str, required=True,
                       help='Path to input JSON')
    group.add_argument('--tsv-keys', nargs='+', default=['text'],
                       help='space separate listed of keys to extract from json')

    group = parser.add_argument_group(title='tokenizer')
    group.add_argument('--tokenizer-type', type=str, required=True,
                       #choices=['BertWordPieceLowerCase','BertWordPieceCase','GPT2BPETokenizer','pretrain_weights/vicuna-7b-v1.1'],
                       help='What type of tokenizer to use.')
    group.add_argument('--vocab-file', type=str, default=None,
                       help='Path to the vocab file')

    group = parser.add_argument_group(title='output data')
    group.add_argument('--output-prefix', type=str, required=True,
                       help='Path to binary output file without suffix')
    group.add_argument('--dataset-impl', type=str, default='mmap',
                       choices=['lazy', 'cached', 'mmap'])

    group = parser.add_argument_group(title='runtime')
    group.add_argument('--workers', type=int, default=1,
                       help='Number of worker processes to launch')
    group.add_argument('--log-interval', type=int, default=10000,
                       help='Interval between progress updates')
    args = parser.parse_args()
    args.keep_empty = False

    # some default/dummy values for the tokenizer
    args.rank = 0
    args.vocab_extra_ids = 0
    args.make_vocab_size_divisible_by = 128
    args.model_parallel_size = 1

    return args

def main():
    args = get_args()
    startup_start = time.time()

    print("Opening", args.input)
    tsvfile = open(args.input, 'r', encoding='utf-8')
    reader = csv.reader(tsvfile, delimiter='\t')
    next(reader, None)  # skip the headers

    encoder      = Encoder(args)
    tokenizer    = build_tokenizer(args)
    pool         = multiprocessing.Pool(args.workers, initializer=encoder.initializer)
    encoded_docs = pool.imap(encoder.encode, reader, 25)
    #encoded_docs = map(encoder.encode, fin)
    print(f"Vocab size: {tokenizer.vocab_size}")
    print(f"Output prefix: {args.output_prefix}")
    startup_end = time.time()
    print("Time to startup:", startup_end - startup_start)
    builder = Tokenized_Knowledge_Builder(args, tokenizer)
    builder.build(encoded_docs,args.tsv_keys,args.output_prefix, args.log_interval)
    # Close the .tsv file
    tsvfile.close()

if __name__ == '__main__':
    main()
