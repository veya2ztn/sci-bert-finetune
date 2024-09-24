# coding=utf-8
# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Megatron global variables."""

from fastchat.model.compression import load_compress_model
from transformers import AutoTokenizer, AutoModelForCausalLM
import os
import sys
import time
import csv

import torch
from megatron.data.indexed_dataset import make_dataset as make_indexed_dataset
from megatron.tokenizer import build_tokenizer
from .arguments import parse_args
from transformers import T5Tokenizer, T5ForConditionalGeneration
from megatron.data.pretokenized_evidence import Tokenized_Knowledge_Dataset

_GLOBAL_ARGS = None
_GLOBAL_TOKENIZER = None
_GLOBAL_T0_TOKENIZER = None
_GLOBAL_T0_MODEL = None
_GLOBAL_EVIDENCE_IN_STRING = None
_GLOBAL_TENSORBOARD_WRITER = None
_GLOBAL_TIMERS = None
_GLOBAL_KNOWLEDGE = {}
_GLOBAL_T0_EMBEDDING = None

def isnotebook():
    try:
        from google import colab
        return True
    except: pass
    try:
        shell = get_ipython().__class__.__name__
        if shell == 'ZMQInteractiveShell':
            return True   # Jupyter notebook, Spyder or qtconsole
        elif shell == 'TerminalInteractiveShell':
            return False  # Terminal running IPython
        else:
            return False  # Other type (?)
    except NameError:
        return False      # Probably standard Python interpreter


    
class Config:pass

default_supervised_config = {"num_layers": 24,
                             "num_unique_layers": None,
                             "param_sharing_style": "grouped",
                             "hidden_size": 1024,
                             "ffn_hidden_size": 4096,
                             "num_attention_heads": 16,
                             "kv_channels": 64,
                             "max_position_embeddings": 512,
                             "make_vocab_size_divisible_by": 128,
                             "layernorm_epsilon": 1e-05,
                             "apply_residual_connection_post_layernorm": False,
                             "openai_gelu": False,
                             "onnx_safe": None,
                             "attention_dropout": 0.1,
                             "hidden_dropout": 0.1,
                             "weight_decay": 0.1,
                             "clip_grad": 1.0,
                             "find_unused_parameters": False,
                             "batch_size": 2,
                             "checkpoint_activations": False,
                             "distribute_checkpointed_activations": False,
                             "checkpoint_num_layers": 1,
                             "train_iters": None,
                             "log_interval": 20,
                             "exit_interval": None,
                             "tensorboard_dir": None,
                             "scaled_upper_triang_masked_softmax_fusion": False,
                             "scaled_masked_softmax_fusion": False,
                             "bias_gelu_fusion": False,
                             "bias_dropout_fusion": False,
                             "seed": 1234,
                             "init_method_std": 0.02,
                             "lr": 2e-05,
                             "lr_decay_style": "linear",
                             "lr_decay_iters": None,
                             "min_lr": 0.0,
                             "warmup": 0.01,
                             "override_lr_scheduler": False,
                             "use_checkpoint_lr_scheduler": False,
                             "save": "./checkpoints/dualencoder-mss-dpr-large-epochs20-webq",
                             "save_interval": 5000,
                             "no_save_optim": False,
                             "no_save_rng": False,
                             "load": "./checkpoints/dualencoder-mss-dpr-large-epochs20-webq",
                             "no_load_optim": False,
                             "no_load_rng": False,
                             "finetune": False,
                             "fp16": True,
                             "apply_query_key_layer_scaling": False,
                             "attention_softmax_in_fp32": False,
                             "fp32_allreduce": False,
                             "hysteresis": 2,
                             "loss_scale": None,
                             "loss_scale_window": 1000,
                             "min_scale": 1,
                             "fp16_lm_cross_entropy": False,
                             "model_parallel_size": 1,
                             "distributed_backend": "nccl",
                             "DDP_impl": "torch",
                             "local_rank": None,
                             "lazy_mpu_init": None,
                             "use_cpu_initialization": False,
                             "eval_iters": 100,
                             "eval_interval": 500,
                             "data_path": None,
                             "glob": False,
                             "qa_file_dev": "./data/qas/webq-dev.csv",
                             "qa_file_test": "./data/qas/webq-test.csv",
                             "qa_file_train": None,
                             "split": "969,30,1",
                             "vocab_file": "./bert-vocab/bert-large-uncased-vocab.txt", "merge_file": None, "vocab_extra_ids": 0, "seq_length": 512, "encoder_seq_length": 512, "decoder_seq_length": None, "seq_length_retriever": 256, "sample_rate": 1.0, "mask_prob": 0.15, "short_seq_prob": 0.1, "mmap_warmup": False, "num_workers": 2, "tokenizer_type": "BertWordPieceLowerCase",
                             "data_impl": "infer",
                             "reset_position_ids": False,
                             "lr_decay_iters": 1,
                             "train_iters": 1,
                             "reset_attention_mask": False,
                             "eod_mask_loss": False,
                             "bert_load": "./checkpoints/megatron_bert_345m/release/mp_rank_00/model_optim_rng.pt",
                             "pretrained_dualencoder_load": None,
                             "evidence_data_path": "./data/wikipedia-split/psgs_w100.tsv",
                             "indexed_evidence_bert_tokenized_data_path": "data/evidence-wikipedia-indexed-mmap/wikipedia-evidence_text_document",
                             "indexed_title_bert_tokenized_data_path": "data/evidence-wikipedia-indexed-mmap/wikipedia-evidence_title_document",
                             "indexed_evidence_t0_tokenized_data_path": "data/evidence-wikipedia-indexed-mmap/t0/wikipedia-evidence-t0_text_document",
                             "indexed_title_t0_tokenized_data_path": "data/evidence-wikipedia-indexed-mmap/t0/wikipedia-evidence-t0_title_document",
                             "log_interval_input_data": 100000,
                             "report_topk_accuracies": [
                                 1,
                                 5,
                                 10,
                                 20,
                                 50,
                                 100
                             ],
                             "retriever_score_scaling": True,
                             "inverse_temperature_multiplier": 1.0,
                             "art_training": False,
                             "no_context_embedder_training": False,
                             "no_query_embedder_training": False,
                             "disable_retriever_dropout": False,
                             "index_reload_interval": 500,
                             "max_training_rank": 1,
                             "update_retriever": False,
                             "hf_model_name": "bigscience/T0_3B",
                             "verbalizer": " . Please write a question based on this passage.",
                             "verbalizer_head": "Passage: ",
                             "shard_size": 16,
                             "initialize_t0_model_tokenizer_evidence": False,
                             "t0_model_in_bf16": False,
                             "compute_fresh_evidence_embeddings": False,
                             "faiss_use_gpu": True,
                             "embedding_path": "./embedding-path/psgs_w100-dualencoder-mss-dpr-large-epochs20-webq.pkl",
                             "match": "string",
                             "topk_retrievals": 100,
                             "save_topk_outputs_path": None,
                             "indexer_batch_size": 128,
                             "indexer_log_interval": 1000,
                             "allow_trivial_doc": False,
                             "run_indexer": False,
                             "trec_eval": False,
                             "task": "RETRIEVER",
                             "epochs": 20,
                             "pretrained_checkpoint": None,
                             "keep_last": False,
                             "train_with_neg": True,
                             "train_data": [
                                 "./data/webq/biencoder-webquestions-train.json"
                             ],
                             "valid_data": [
                                 "./data/webq/biencoder-webquestions-dev.json"
                             ],
                             "eval_batch_size": 16,
                             "seq_length_ret": 256,
                             "train_hard_neg": 7,
                             "val_av_rank_hard_neg": 5,
                             "val_av_rank_other_neg": 5,
                             "rank": 0,
                             "world_size": 1,
                             "dynamic_loss_scale": True,
                             "params_dtype": torch.float16,
                             "padded_vocab_size": 30592
                             }

default_unsupervised_config = {
    "num_layers": 12,
    "num_unique_layers": None,
    "param_sharing_style": "grouped",
    "hidden_size": 768,
    "ffn_hidden_size": 3072,
    "num_attention_heads": 12,
    "kv_channels": 64,
    "max_position_embeddings": 512,
    "make_vocab_size_divisible_by": 128,
    "layernorm_epsilon": 1e-05,
    "apply_residual_connection_post_layernorm": False,
    "openai_gelu": False,
    "onnx_safe": None,
    "attention_dropout": 0.1,
    "hidden_dropout": 0.1,
    "weight_decay": 0.1,
    "clip_grad": 1.0,
    "find_unused_parameters": False,
    "batch_size": 4,
    "checkpoint_activations": True,
    "distribute_checkpointed_activations": False,
    "checkpoint_num_layers": 1,
    "train_iters": 1,
    "log_interval": 20,
    "exit_interval": None,
    "tensorboard_dir": None,
    "scaled_upper_triang_masked_softmax_fusion": False,
    "scaled_masked_softmax_fusion": False,
    "bias_gelu_fusion": False,
    "bias_dropout_fusion": False,
    "seed": 1234,
    "init_method_std": 0.02,
    "lr": 2e-05,
    "lr_decay_style": "linear",
    "lr_decay_iters": 1,
    "min_lr": 0.0,
    "warmup": 0.01,
    "override_lr_scheduler": False,
    "use_checkpoint_lr_scheduler": False,
    "save": "./checkpoints/nq-mss-base-init",
    "save_interval": 500,
    "no_save_optim": False,
    "no_save_rng": False,
    "load": "./checkpoints/nq-mss-base-init",
    "no_load_optim": False,
    "no_load_rng": False,
    "finetune": False,
    "fp16": True,
    "apply_query_key_layer_scaling": False,
    "attention_softmax_in_fp32": False,
    "fp32_allreduce": False,
    "hysteresis": 2,
    "loss_scale": None,
    "loss_scale_window": 1000,
    "min_scale": 1,
    "fp16_lm_cross_entropy": False,
    "model_parallel_size": 1,
    "distributed_backend": "nccl",
    "DDP_impl": "local",
    "local_rank": None,
    "lazy_mpu_init": None,
    "use_cpu_initialization": False,
    "eval_iters": 100,
    "eval_interval": 500,
    "data_path": None,
    "glob": False,
    "qa_file_dev": "./data/qas/nq-dev.csv",
    "qa_file_test": "./data/qas/nq-test.csv",
    "qa_file_train": None,
    "split": "969, 30,1", 
    "vocab_file": "./bert-vocab/bert-large-uncased-vocab.txt", 
    "merge_file": None, "vocab_extra_ids": 0, "seq_length": 512, "encoder_seq_length": 512, "decoder_seq_length": None, 
    "seq_length_retriever": 256, "sample_rate": 1.0, "mask_prob": 0.15, "short_seq_prob": 0.1, "mmap_warmup": False, 
    "num_workers": 2, "tokenizer_type": "BertWordPieceLowerCase", "data_impl": "infer", "reset_position_ids": False, 
    "reset_attention_mask": False, "eod_mask_loss": False, "bert_load": None, 
    "pretrained_dualencoder_load": None,#"./checkpoints/mss-retriever-base", 
    "evidence_data_path": "./data/wikipedia-split/psgs_w100.tsv", 
    "indexed_evidence_bert_tokenized_data_path": "./data/evidence-wikipedia-indexed-mmap/bert/wikipedia-evidence-bert_text_document", 
    "indexed_title_bert_tokenized_data_path": "./data/evidence-wikipedia-indexed-mmap/bert/wikipedia-evidence-bert_title_document", 
    "indexed_evidence_t0_tokenized_data_path": "./data/evidence-wikipedia-indexed-mmap/t0/wikipedia-evidence-t0_text_document", 
    "indexed_title_t0_tokenized_data_path": "./data/evidence-wikipedia-indexed-mmap/t0/wikipedia-evidence-t0_title_document", 
    "log_interval_input_data": 100000, "report_topk_accuracies": [1,5,20,50,100],
    "retriever_score_scaling": True,
    "inverse_temperature_multiplier": 1.0,
    "art_training": True,
    "no_context_embedder_training": False,
    "no_query_embedder_training": False,
    "disable_retriever_dropout": False,
    "index_reload_interval": 500,
    "max_training_rank": 1,
    "update_retriever": True,
    "hf_model_name": "bigscience/T0_3B",
    "verbalizer": " . Please write a question based on this passage.",
    "verbalizer_head": "Passage: ",
    "shard_size": 16,
    "initialize_t0_model_tokenizer_evidence": True,
    "t0_model_in_bf16": False,
    "compute_fresh_evidence_embeddings": True,
    "faiss_use_gpu": False,
    "embedding_path": None,#"./embedding-path/art-finetuning-embedding/psgs_w100-retriever-nq-base-topk32-epochs10-bsize64-indexer.pkl",
    "match": "string",
    "topk_retrievals": 32,
    "save_topk_outputs_path": None,
    "indexer_batch_size": 128,
    "indexer_log_interval": 1000,
    "allow_trivial_doc": True,
    "run_indexer": False,
    "trec_eval": False,
    "task": "ZERO-SHOT-RETRIEVER",
    "epochs": 10,
    "pretrained_checkpoint": None,
    "keep_last": False,
    "train_with_neg": False,
    "train_data": ["./data/qas/nq-train.csv"],
    "valid_data": None,
    "eval_batch_size": 1,
    "seq_length_ret": None,
    "train_hard_neg": None,
    "val_av_rank_hard_neg": None,
    "val_av_rank_other_neg": None,
    "rank": 0,
    "world_size": 1,
    "dynamic_loss_scale": True,
    "params_dtype": torch.float16,
    "padded_vocab_size": 30592
}

default_unsupervised_config_vicuna = {
    "num_layers": 12,
    "num_unique_layers": None,
    "param_sharing_style": "grouped",
    "hidden_size": 768,
    "ffn_hidden_size": 3072,
    "num_attention_heads": 12,
    "kv_channels": 64,
    "max_position_embeddings": 512,
    "make_vocab_size_divisible_by": 128,
    "layernorm_epsilon": 1e-05,
    "apply_residual_connection_post_layernorm": False,
    "openai_gelu": False,
    "onnx_safe": None,
    "attention_dropout": 0.1,
    "hidden_dropout": 0.1,
    "weight_decay": 0.1,
    "clip_grad": 1.0,
    "find_unused_parameters": False,
    "batch_size": 4,
    "checkpoint_activations": True,
    "distribute_checkpointed_activations": False,
    "checkpoint_num_layers": 1,
    "train_iters": 1,
    "log_interval": 20,
    "exit_interval": None,
    "tensorboard_dir": None,
    "scaled_upper_triang_masked_softmax_fusion": False,
    "scaled_masked_softmax_fusion": False,
    "bias_gelu_fusion": False,
    "bias_dropout_fusion": False,
    "seed": 1234,
    "init_method_std": 0.02,
    "lr": 2e-05,
    "lr_decay_style": "linear",
    "lr_decay_iters": 1,
    "min_lr": 0.0,
    "warmup": 0.01,
    "override_lr_scheduler": False,
    "use_checkpoint_lr_scheduler": False,
    "save": "./checkpoints/nq-mss-base-init",
    "save_interval": 500,
    "no_save_optim": False,
    "no_save_rng": False,
    "load": "./checkpoints/nq-mss-base-init",
    "no_load_optim": False,
    "no_load_rng": False,
    "finetune": False,
    "fp16": True,
    "apply_query_key_layer_scaling": False,
    "attention_softmax_in_fp32": False,
    "fp32_allreduce": False,
    "hysteresis": 2,
    "loss_scale": None,
    "loss_scale_window": 1000,
    "min_scale": 1,
    "fp16_lm_cross_entropy": False,
    "model_parallel_size": 1,
    "distributed_backend": "nccl",
    "DDP_impl": "local",
    "local_rank": None,
    "lazy_mpu_init": None,
    "use_cpu_initialization": False,
    "eval_iters": 100,
    "eval_interval": 500,
    "data_path": None,
    "glob": False,
    "qa_file_dev": "./data/qas/nq-dev.csv",
    "qa_file_test": "./data/qas/nq-test.csv",
    "qa_file_train": None,
    "split": "969, 30,1", 
    "vocab_file": "./bert-vocab/bert-large-uncased-vocab.txt", 
    "merge_file": None, "vocab_extra_ids": 0, "seq_length": 512, "encoder_seq_length": 512, "decoder_seq_length": None, 
    "seq_length_retriever": 256, "sample_rate": 1.0, "mask_prob": 0.15, "short_seq_prob": 0.1, "mmap_warmup": False, 
    "num_workers": 2, "tokenizer_type": "BertWordPieceLowerCase", "data_impl": "infer", "reset_position_ids": False, 
    "reset_attention_mask": False, "eod_mask_loss": False, "bert_load": None, 
    "pretrained_dualencoder_load": None,#"./checkpoints/mss-retriever-base", 
    "evidence_data_path": "./data/wikipedia-split/psgs_w100.tsv", 
    "indexed_evidence_bert_tokenized_data_path": "./data/evidence-wikipedia-indexed-mmap/bert/wikipedia-evidence-bert_text_document", 
    "indexed_title_bert_tokenized_data_path": "./data/evidence-wikipedia-indexed-mmap/bert/wikipedia-evidence-bert_title_document", 
    "indexed_evidence_t0_tokenized_data_path": "./data/evidence-wikipedia-indexed-mmap/vicuna/wikipedia-evidence-vicuna_text_document", 
    "indexed_title_t0_tokenized_data_path": "./data/evidence-wikipedia-indexed-mmap/vicuna/wikipedia-evidence-vicuna_title_document",
    "log_interval_input_data": 100000, "report_topk_accuracies": [1,5,20,50,100],
    "retriever_score_scaling": True,
    "inverse_temperature_multiplier": 1.0,
    "art_training": True,
    "no_context_embedder_training": False,
    "no_query_embedder_training": False,
    "disable_retriever_dropout": False,
    "index_reload_interval": 500,
    "max_training_rank": 1,
    "update_retriever": True,
    "hf_model_name": "pretrain_weights/vicuna-7b-v1.1",
    "verbalizer": "Based on the given passage, formulate a question that aligns most accurately with the primary fact disclosed within the text. ASSISTANT:",
    "verbalizer_head": "A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user\'s questions. USER: Here is a passage about ",
    "shard_size": 16,
    "initialize_t0_model_tokenizer_evidence": True,
    "t0_model_in_bf16": False,
    "compute_fresh_evidence_embeddings": True,
    "faiss_use_gpu": False,
    "embedding_path": None,
    "match": "string",
    "topk_retrievals": 32,
    "save_topk_outputs_path": None,
    "indexer_batch_size": 128,
    "indexer_log_interval": 1000,
    "allow_trivial_doc": True,
    "run_indexer": False,
    "trec_eval": False,
    "task": "ZERO-SHOT-RETRIEVER",
    "epochs": 10,
    "pretrained_checkpoint": None,
    "keep_last": False,
    "train_with_neg": False,
    "train_data": ["./data/qas/nq-train.csv"],
    "valid_data": None,
    "eval_batch_size": 1,
    "seq_length_ret": None,
    "train_hard_neg": None,
    "val_av_rank_hard_neg": None,
    "val_av_rank_other_neg": None,
    "rank": 0,
    "world_size": 1,
    "dynamic_loss_scale": True,
    "params_dtype": torch.float16,
    "padded_vocab_size": 30592,
    "hf_model_type":"compress"
}

# def get_args():
#     """Return arguments."""
#     _ensure_var_is_initialized(_GLOBAL_ARGS, 'args')
#     return _GLOBAL_ARGS

def get_args():
    if isnotebook():
        config = Config()
        #cfg = default_supervised_config
        cfg = default_unsupervised_config_vicuna
        for key, val in cfg.items():
            setattr(config, key, val)
        print(f'---------> use fixed config <----------')
        config.art_mode = 'question_relative_check'
        return config
    else:
        _ensure_var_is_initialized(_GLOBAL_ARGS, 'args')
        return _GLOBAL_ARGS

#################################################
############ initialize bert token ##############
#################################################
def get_tokenizer():
    """Return tokenizer."""
    _ensure_var_is_initialized(_GLOBAL_TOKENIZER, 'tokenizer')
    return _GLOBAL_TOKENIZER
def rebuild_tokenizer(args):
    global _GLOBAL_TOKENIZER
    _GLOBAL_TOKENIZER = None
    return _build_tokenizer(args)
def _build_tokenizer(args):
    """Initialize tokenizer."""
    global _GLOBAL_TOKENIZER
    _ensure_var_is_not_initialized(_GLOBAL_TOKENIZER, 'tokenizer')
    _GLOBAL_TOKENIZER = build_tokenizer(args)
    return _GLOBAL_TOKENIZER

#################################################
############ initialize t0 system  ##############
#################################################
def get_t0_tokenizer():
    """Return T0 tokenizer."""
    _ensure_var_is_initialized(_GLOBAL_T0_TOKENIZER, 'T0-tokenizer')
    return _GLOBAL_T0_TOKENIZER
def get_t0_model():
    """Return T0 model."""
    _ensure_var_is_initialized(_GLOBAL_T0_MODEL, 'T0-model')
    return _GLOBAL_T0_MODEL

def _build_t0_tokenizer(args):
    """Initialize T0 tokenizer."""
    global _GLOBAL_T0_TOKENIZER
    _ensure_var_is_not_initialized(_GLOBAL_T0_TOKENIZER, 'T0-tokenizer')
    if 'bigscience' in args.hf_model_name:
        _GLOBAL_T0_TOKENIZER = T5Tokenizer.from_pretrained(args.hf_model_name)
    else:
        _GLOBAL_T0_TOKENIZER = AutoTokenizer.from_pretrained(
            args.hf_model_name, use_fast=False)
    return _GLOBAL_T0_TOKENIZER


def get_knowledge_pool():
    """return the knowledge_pool"""
    _ensure_var_is_initialized(_GLOBAL_KNOWLEDGE, 'knowledge')
    return _GLOBAL_KNOWLEDGE

def _build_knowledge_pool_in_bert(args):
    """Initialize T0 tokenizer."""
    global _GLOBAL_KNOWLEDGE
    assert 'bert' not in _GLOBAL_KNOWLEDGE
    _GLOBAL_KNOWLEDGE['bert'] = Tokenized_Knowledge_Dataset(
        (args.indexed_evidence_bert_tokenized_data_path, args.indexed_title_bert_tokenized_data_path),
         args.data_impl,
        get_tokenizer(),args.seq_length_retriever
    )
    return _GLOBAL_KNOWLEDGE


def _build_knowledge_pool_in_t0(args):
    """Initialize T0 tokenizer."""
    global _GLOBAL_KNOWLEDGE
    assert 't0' not in _GLOBAL_KNOWLEDGE
    
    _GLOBAL_KNOWLEDGE['t0']  = Tokenized_Knowledge_Dataset(
        (args.indexed_evidence_t0_tokenized_data_path, args.indexed_title_t0_tokenized_data_path),
         args.data_impl,
        get_t0_tokenizer(),args.seq_length_retriever
    )

    return _GLOBAL_KNOWLEDGE
        


def get_knowledge_embedding_handle():
    """Return T0 model."""
    _ensure_var_is_initialized(_GLOBAL_T0_EMBEDDING, 'embedding')
    return _GLOBAL_T0_EMBEDDING



from fastchat.model.compression import *
#from fastchat.model.compression import load_compress_model
#from optimum.bettertransformer import BetterTransformer
def load_compress_model_(model_path, device, torch_dtype):
    # partially load model
    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
    base_pattern = os.path.join(model_path, "pytorch_model-*.bin")
    files = glob.glob(base_pattern)

    with init_empty_weights():
        config = AutoConfig.from_pretrained(
            model_path, low_cpu_mem_usage=True, torch_dtype=torch_dtype, load_in_8bit=True, device_map="auto"
        )
        model = AutoModelForCausalLM.from_config(config)
        linear_weights = get_compressed_list(model)
        
    compressed_state_dict = {}

    for filename in tqdm(files):
        tmp_state_dict = torch.load(filename)
        for name in tmp_state_dict:
            if name in linear_weights:
                tensor = tmp_state_dict[name].to(device).data.to(torch_dtype)
                compressed_state_dict[name] = compress(
                    tensor, default_compression_config
                )
            else:
                compressed_state_dict[name] = tmp_state_dict[name].to(device)
            tmp_state_dict[name] = None
            tensor = None
            gc.collect()
            torch.cuda.empty_cache()

    for name in model.state_dict():
        if name not in linear_weights:
            set_module_tensor_to_device(
                model, name, device, value=compressed_state_dict[name]
            )
    apply_compressed_weight(model, compressed_state_dict, device)

    model.to(device)
    #model = BetterTransformer.transform(model)
    #model = torch.compile(model)
    return model, tokenizer
def get_device(args):

    device_count = torch.cuda.device_count()
    if torch.distributed.is_initialized():

        if args.rank == 0:
            print('torch distributed is already initialized, '
                  'skipping initialization ...', flush=True)
        args.rank = torch.distributed.get_rank()
        args.world_size = torch.distributed.get_world_size()
        device = torch.device(f'cuda:{args.rank}')
    else:

        if args.rank == 0:
            print('> initializing torch distributed ...', flush=True)
        # Manually set the device ids.
        if device_count > 0:
            device = args.rank % device_count
            if args.local_rank is not None:
                assert args.local_rank == device, \
                    'expected local-rank to be the same as rank % device-count.'
            else:
                args.local_rank = device
            device = torch.device(f'cuda:{args.local_rank}')
            
    return device

from bits4llama.llama import load_quant
import time
def _build_t0_model(args):
    """Initialize T0 model."""
    global _GLOBAL_T0_MODEL
    _ensure_var_is_not_initialized(_GLOBAL_T0_MODEL, 'T0-model')
    if 'bigscience' in args.hf_model_name:
        _GLOBAL_T0_MODEL = T5ForConditionalGeneration.from_pretrained(args.hf_model_name, torch_dtype=torch.bfloat16 if args.t0_model_in_bf16 else torch.float32)
    else:
        if args.hf_model_type == 'compress':
            _GLOBAL_T0_MODEL, _ = load_compress_model_(
                model_path=args.hf_model_name, device=f'cuda:{args.rank}', torch_dtype=torch.float16
            )
        elif args.hf_model_type == 'fast':
            _GLOBAL_T0_MODEL = AutoModelForCausalLM.from_pretrained(
            args.hf_model_name, low_cpu_mem_usage=True, torch_dtype=torch.float16#, load_in_8bit=True, device_map="auto"#"balanced_low_0"
        )
        else:
            device = get_device(args)#torch.device(f'cuda:{args.rank}')
            torch.cuda.set_device(device) # important for 4bits model load
            _GLOBAL_T0_MODEL = load_quant(
                args.hf_model_name, args.hf_model_type, 4, 128,device=device).to(device)
            
    # if args.t0_model_in_bf16:
    #     _GLOBAL_T0_MODEL = _GLOBAL_T0_MODEL.bfloat16()

    for param in _GLOBAL_T0_MODEL.parameters():
        param.requires_grad = False
    return _GLOBAL_T0_MODEL


#################################################
########## initialize large evidence  ############
#################################################
def get_evidence_in_string():
    """Return T0 model."""
    #return None
    _ensure_var_is_initialized(_GLOBAL_EVIDENCE_IN_STRING, 'wikipedia-evidence-from-DPR-paper')
    return _GLOBAL_EVIDENCE_IN_STRING
def _load_wikipedia_evidence(args):
    """Load the DPR wikipedia evidence file"""
    global _GLOBAL_EVIDENCE_IN_STRING
    _ensure_var_is_not_initialized(_GLOBAL_EVIDENCE_IN_STRING, 'wikipedia-evidence-from-DPR-paper')
    _GLOBAL_EVIDENCE_IN_STRING = process_samples_from_single_path(args)
    return _GLOBAL_EVIDENCE_IN_STRING

import h5py
def process_samples_from_single_path(args):
    if not args.evidence_data_path: return None
    if '.hdf5' in args.evidence_data_path:
        return h5py.File(args.evidence_data_path, 'r')['data']
    if args.local_rank == 0:
        print(' > Processing {} ...'.format(args.evidence_data_path))
    total = 0
    id2text = []

    with open(args.evidence_data_path) as tsvfile:
        reader = csv.reader(tsvfile, delimiter='\t')
        next(reader, None)  # skip the headers

        # Fill some random text tuple in the first index
        id2text.append(("text", "title"))

        for row in reader:
            # file format: doc_id, doc_text, title
            doc_id = int(row[0])
            text = row[1]
            title = row[2]

            # doc_id is specified by the index of the list
            id2text.append((text, title))

            total += 1
            if total % 1000000 == 0:
                if args.local_rank == 0:
                    print('  > processed {} rows so far ...'.format(total))

    if args.local_rank == 0:
        print(' >> processed {} samples.'.format(total))

    return id2text


#################################################
############## initialize utils #################
#################################################
def get_tensorboard_writer():
    """Return tensorboard writer. It can be None so no need
    to check if it is initialized."""
    return _GLOBAL_TENSORBOARD_WRITER
def get_timers():
    """Return timers."""
    _ensure_var_is_initialized(_GLOBAL_TIMERS, 'timers')
    return _GLOBAL_TIMERS
def set_global_variables(extra_args_provider=None, args_defaults={},
                         ignore_unknown_args=False):
    """Set args, tokenizer, tensorboard-writer, and timers."""
    args = _parse_args(extra_args_provider=extra_args_provider,
                       defaults=args_defaults,
                       ignore_unknown_args=ignore_unknown_args)
    _ = _build_tokenizer(args)
    _ = _build_knowledge_pool_in_bert(args)
    if args.initialize_t0_model_tokenizer_evidence:
        _ = _build_t0_tokenizer(args)
        _ = _build_knowledge_pool_in_t0(args)
        _ = _build_t0_model(args)
        _ = _load_wikipedia_evidence(args)

    _set_tensorboard_writer(args)
    _set_timers()
def _parse_args(extra_args_provider=None, defaults={},
                ignore_unknown_args=False):
    """Parse entire arguments."""
    global _GLOBAL_ARGS
    _ensure_var_is_not_initialized(_GLOBAL_ARGS, 'args')
    _GLOBAL_ARGS = parse_args(extra_args_provider=extra_args_provider,
                              defaults=defaults,
                              ignore_unknown_args=ignore_unknown_args)
    return _GLOBAL_ARGS


def _set_tensorboard_writer(args):
    """Set tensorboard writer."""
    global _GLOBAL_TENSORBOARD_WRITER
    _ensure_var_is_not_initialized(_GLOBAL_TENSORBOARD_WRITER,
                                   'tensorboard writer')

    if hasattr(args, 'tensorboard_dir') and \
       args.tensorboard_dir and args.rank == 0:
        try:
            from torch.utils.tensorboard import SummaryWriter
            print('> setting tensorboard ...')
            _GLOBAL_TENSORBOARD_WRITER = SummaryWriter(
                log_dir=args.tensorboard_dir)
        except ModuleNotFoundError:
            print('WARNING: TensorBoard writing requested but is not '
                  'available (are you using PyTorch 1.1.0 or later?), '
                  'no TensorBoard logs will be written.', flush=True)
def _set_timers():
    """Initialize timers."""
    global _GLOBAL_TIMERS
    _ensure_var_is_not_initialized(_GLOBAL_TIMERS, 'timers')
    _GLOBAL_TIMERS = Timers()


def _ensure_var_is_initialized(var, name):
    """Make sure the input variable is not None."""
    assert var is not None, '{} is not initialized.'.format(name)


def _ensure_var_is_not_initialized(var, name):
    """Make sure the input variable is not None."""
    assert var is None, '{} is already initialized.'.format(name)


class _Timer:
    """Timer."""

    def __init__(self, name):
        self.name_ = name
        self.elapsed_ = 0.0
        self.started_ = False
        self.start_time = time.time()

    def start(self):
        """Start the timer."""
        assert not self.started_, 'timer has already been started'
        torch.cuda.synchronize()
        self.start_time = time.time()
        self.started_ = True

    def stop(self):
        """Stop the timer."""
        assert self.started_, 'timer is not started'
        torch.cuda.synchronize()
        self.elapsed_ += (time.time() - self.start_time)
        self.started_ = False

    def reset(self):
        """Reset timer."""
        self.elapsed_ = 0.0
        self.started_ = False

    def elapsed(self, reset=True):
        """Calculate the elapsed time."""
        started_ = self.started_
        # If the timing in progress, end it first.
        if self.started_:
            self.stop()
        # Get the elapsed time.
        elapsed_ = self.elapsed_
        # Reset the elapsed time
        if reset:
            self.reset()
        # If timing was in progress, set it back.
        if started_:
            self.start()
        return elapsed_


class Timers:
    """Group of timers."""

    def __init__(self):
        self.timers = {}

    def __call__(self, name):
        if name not in self.timers:
            self.timers[name] = _Timer(name)
        return self.timers[name]

    def write(self, names, writer, iteration, normalizer=1.0, reset=False):
        """Write timers to a tensorboard writer"""
        # currently when using add_scalars,
        # torch.utils.add_scalars makes each timer its own run, which
        # polutes the runs list, so we just add each as a scalar
        assert normalizer > 0.0
        for name in names:
            value = self.timers[name].elapsed(reset=reset) / normalizer
            writer.add_scalar(name + '_time', value, iteration)

    def log(self, names, normalizer=1.0, reset=True):
        """Log a group of timers."""
        assert normalizer > 0.0
        string = 'time (ms)'
        for name in names:
            elapsed_time = self.timers[name].elapsed(
                reset=reset) * 1000.0 / normalizer
            string += ' | {}: {:.2f}'.format(name, elapsed_time)
        if torch.distributed.is_initialized():
            if torch.distributed.get_rank() == 0:
                print(string, flush=True)
        else:
            print(string, flush=True)


