"""Index evidence dataset for use in QA tasks.
   This file is similar to the tools/preprocess_data.py script in the backbones
"""

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

    def get_doc_ids(self,text):
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
        return doc_ids
    def encode(self, csv_line):
        # line format: doc_id, doc_text, title
        #print(f"\n csv line is ==>  {csv_line} <==")
        #raise      
        (_id,paper_id,abstract,key_word_1,key_word_2,key_word_3,key_word_4,
        key_word_5,key_word_6,key_word_7,key_word_8,key_word_9,key_word_10) = csv_line
        csv_text  = abstract
        csv_title = csv_line[2]
        #print(f"\n csv text is ==>  {csv_line[1]} <==")
        #print(f"\n csv title is ==>  {csv_line[2]} <==")
        ids = {}
        ids["text"] = self.get_doc_ids(abstract)
        ids["title"]=[]
        for key in csv_line[3:]:
            question = f"what is {key}"
            ids["title"].extend(self.get_doc_ids(question))
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
    reader  = csv.reader(tsvfile, delimiter=',')
    next(reader, None)  # skip the headers

    encoder = Encoder(args)
    tokenizer = build_tokenizer(args)
    pool = multiprocessing.Pool(args.workers, initializer=encoder.initializer)
    encoded_docs = pool.imap(encoder.encode, reader, 25)
    #encoded_docs = map(encoder.encode, fin)
    level = "document"
    print(f"Vocab size: {tokenizer.vocab_size}")
    print(f"Output prefix: {args.output_prefix}")
    output_bin_files = {}
    output_idx_files = {}
    builders = {}
    for key in ['title','text']:
        output_bin_files[key] = "{}_{}_{}.bin".format(args.output_prefix,key, level)
        output_idx_files[key] = "{}_{}_{}.idx".format(args.output_prefix,key, level)
        builders[key] = indexed_dataset.make_builder(output_bin_files[key],
                                                     impl=args.dataset_impl,
                                                     vocab_size=tokenizer.vocab_size)

    startup_end = time.time()
    proc_start = time.time()
    total_bytes_processed = 0
    print("Time to startup:", startup_end - startup_start)

    for i, (doc, bytes_processed) in enumerate(encoded_docs, start=1):
        total_bytes_processed += bytes_processed
        #print(f"doc===> {doc} <===")
        for key, sentences in doc.items():
            #print(f"{key} ===> {sentences} <===")
            for sentence in sentences:
                builders[key].add_item(torch.IntTensor(sentence))
            builders[key].end_document()
  
        if i % args.log_interval == 0:
            current = time.time()
            elapsed = current - proc_start
            mbs = total_bytes_processed/elapsed/1024/1024
            print(f"Processed {i}/{21015325} documents",
                  f"({i/elapsed} docs/s, {mbs} MB/s).",
                  file=sys.stderr)
        #raise
    for key in ['title','text']:#args.tsv_keys:
        builders[key].finalize(output_idx_files[key])

    # Close the .tsv file
    tsvfile.close()

if __name__ == '__main__':
    main()
