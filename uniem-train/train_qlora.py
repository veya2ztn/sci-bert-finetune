# Usage: deepspeed train_lora.py --deepspeed <$PATH_TO_DEEPSPEED_CONFIG>

# Adopted from tatsu-lab@stanford_alpaca. Below is the original copyright:
#    Copyright 2023 Rohan Taori, Ishaan Gulrajani, Tianyi Zhang, Yann Dubois, Xuechen Li
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.

import os
import logging
import pathlib
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
import transformers
from transformers import Trainer, BitsAndBytesConfig
from transformers.integrations import deepspeed
import torch

os.environ['WANDB_CONSOLE']='off'
# os.environ["WANDB_MODE"] = "offline"
from fastchat.train.llama2_flash_attn_monkey_patch import (
    replace_llama_attn_with_flash_attn,
)

try:
    from llama_attn_replace import replace_llama_attn
except:
    pass

from utils import get_peft_state_maybe_zero_3,upgrade_transformer_progressbar
upgrade_transformer_progressbar()

from arguements import print_namespace_tree, ModelArguments, DataArguments,TrainingArguments, SelfDistributedArguments, LoraArguments
from uniem.SharedMemoryBuffer import AlongAnswerNumpyBufferForIndex,OnlyAnswerNumpyBufferForIndex
from uniem.model import LlamaLastWeightedEmbedder, BertEmbedder
from typing import List, Dict, Any, Union, Optional
from extra_structure import *
from trainer_gradient_cache import RealtimeEmbeddingTrainer
def get_local_rank():
    rank = int(os.environ.get("RANK",0))
    local_rank = int(os.environ.get("LOCAL_RANK",0))
    
    return rank + local_rank

def smart_tokenizer_and_embedding_resize(
    special_tokens_dict: Dict,
    tokenizer: transformers.PreTrainedTokenizer,
    model: transformers.PreTrainedModel,
):
    """Resize tokenizer and embedding.

    Note: This is the unoptimized version that may make your embedding size not be divisible by 64.
    """
    num_new_tokens = tokenizer.add_special_tokens(special_tokens_dict)
    model.resize_token_embeddings(len(tokenizer))

    if num_new_tokens > 0:
        input_embeddings = model.get_input_embeddings().weight.data
        output_embeddings = model.get_output_embeddings().weight.data

        input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)
        output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)

        input_embeddings[-num_new_tokens:] = input_embeddings_avg
        output_embeddings[-num_new_tokens:] = output_embeddings_avg

def print_trainable_parameters(args, model):
    """
    Prints the number of trainable parameters in the model.
    """
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    #if args.bits == 4: trainable_params /= 2
    print(
        f"trainable params: {trainable_params} || "
        f"all params: {all_param} || "
        f"trainable: {100 * trainable_params / all_param}"
    )
from transformers.integrations import WandbCallback

def get_model_and_tokenizer(training_args:TrainingArguments, model_args:ModelArguments, compute_dtype, device_map=None, q_lora=False):
    #print(f"loading model from {model_args.model_name_or_path}")
    if 'jina' in model_args.model_name_or_path.lower():
        assert training_args.full_finetune, "your are using a embedder model directly, diable lora"
        assert not q_lora, "your are using a embedder model directly, disable qlora"
        from models.jinabert.modeling_bert import JinaBertModel
        from models.jinabert.configuration_bert import JinaBertConfig
        config                 = JinaBertConfig.from_pretrained(os.path.join(model_args.model_name_or_path, 'config.json'))
        config.alibi_scaling   = model_args.extrapolation_scaling
        config.embedding_model = True
        if training_args.flash_attn:
            config.attn_implementation = 'flashV2'
        model                  = JinaBertModel.from_pretrained(model_args.model_name_or_path,config=config,device_map=device_map)
        
    else:
        model = transformers.AutoModelForCausalLM.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=training_args.cache_dir,
            device_map=device_map,
            quantization_config=BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=compute_dtype,
            )
            if q_lora else None,
        )
    #print(f"loading tokenizer from {model_args.model_name_or_path}")
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        model_max_length=training_args.model_max_length * model_args.extrapolation_scaling,
        padding_side="right",
        use_fast=False,
    )
    if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.unk_token
    return model, tokenizer

from transformers.trainer_utils import EvalPrediction
def format_topk_rank(results:EvalPrediction)->Dict:
    rank_order = results.label_ids
    assert rank_order.shape[1] == 4
    rank_max, rank_meddian, rank_mean, rank_min = rank_order.T
    
    output = {}
    for k in [100,50,10,1]:
        output[f'top{k}/mean']   = (rank_mean <= k).mean()
        output[f'top{k}/median'] = (rank_meddian <= k).mean()
        output[f'top{k}/min']    = (rank_min <= k).mean()
        output[f'top{k}/max']    = (rank_max <= k).mean()
    
    order_distance = results.predictions  
    rank_max, rank_meddian, rank_mean, rank_min = order_distance.T
    output[f'distance/max']    = rank_max.mean()
    output[f'distance/mean']   = rank_mean.mean()
    output[f'distance/min']    = rank_min.mean()
    output[f'distance/median'] = rank_meddian.mean()
    return output


def train():
    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments, LoraArguments, SelfDistributedArguments)
    )
    (
        model_args,
        data_args,
        training_args,
        lora_args,
        distributed_arg,
    ) = parser.parse_args_into_dataclasses()
    local_rank = get_local_rank()
    model_args.extrapolation_scaling = training_args.extrapolation_scaling
    data_args.dummy_data = training_args.dummy_data

    if local_rank == 0:
        print_namespace_tree(model_args)
        print_namespace_tree(data_args)
        print_namespace_tree(training_args)
        print_namespace_tree(lora_args)
        print_namespace_tree(distributed_arg)
    if training_args.flash_attn:
        if training_args.use_long_lora:
            replace_llama_attn(True,False)
        else:
            replace_llama_attn_with_flash_attn()
        

    
    device_map = None

    if distributed_arg.self_distributed_init:
        distributed_initial(distributed_arg)
        slurm_distributed_initial(distributed_arg)
    else:
        world_size = int(os.environ.get("WORLD_SIZE", 1))
        ddp = world_size != 1
        if lora_args.q_lora:
            device_map = {"": int(os.environ.get("LOCAL_RANK",0))} if ddp else 'auto'
            if len(training_args.fsdp) > 0 or deepspeed.is_deepspeed_zero3_enabled():
                logging.warning(
                    "FSDP and ZeRO3 are both currently incompatible with QLoRA."
                )

    compute_dtype = (
        torch.float16
        if training_args.fp16
        else (torch.bfloat16 if training_args.bf16 else torch.float32)
    )
    model, tokenizer = get_model_and_tokenizer(training_args, model_args, compute_dtype, device_map=device_map, q_lora=lora_args.q_lora)

    if not training_args.full_finetune:
        
        lora_config = LoraConfig(
            r=lora_args.lora_r,
            lora_alpha=lora_args.lora_alpha,
            target_modules=lora_args.lora_target_modules,
            lora_dropout=lora_args.lora_dropout,
            bias=lora_args.lora_bias,
            task_type="CAUSAL_LM",
        )

        if lora_args.q_lora:
            model = prepare_model_for_kbit_training( model, use_gradient_checkpointing=training_args.gradient_checkpointing)
            if not ddp and torch.cuda.device_count() > 1:
                # keeps Trainer from trying its own DataParallelism when more than 1 gpu is available
                model.is_parallelizable = True
                model.model_parallel = True

        #print(model.config)
        model = get_peft_model(model, lora_config)
        if training_args.start_lora_path:
            model.load_adapter(training_args.start_lora_path, 'default', is_trainable=True)
            
        if training_args.flash_attn:
            for name, module in model.named_modules():
                if "norm" in name:
                    module = module.to(compute_dtype)
                if "lm_head" in name or "embed_tokens" in name:
                    if hasattr(module, "weight"):
                        module = module.to(compute_dtype)
    #model.print_trainable_parameters()
    print_trainable_parameters(model_args, model)
    if training_args.gradient_checkpointing:
        model.enable_input_require_grads()

    
    negative_sampler_num = -1# training_args.negative_sampler_num
    realtime_embedding_mode = True
    del training_args.negative_sampler_num
    if negative_sampler_num > 0 :
        model = PairQAmodel(LlamaLastWeightedEmbedder(model))
        data_module = make_supervised_data_module(tokenizer=tokenizer, data_args=data_args,
                        model_max_length = training_args.model_max_length, return_mapping=True,dispatch_batches=training_args.dispatch_batches)
        
        if local_rank == 0:
            print(f'loading preload embedding, which will cost 10 - 20s')
            preloaded_root = 'data/unarXive_quantum_physics' 
            preloaded = {'answer':[np.load(os.path.join(preloaded_root,'llama_answer_embedding/alpha','index.npy')),
                                   np.load(os.path.join(preloaded_root,'llama_answer_embedding/alpha','embedding.npy'))],
                        'question':[np.load(os.path.join(preloaded_root,'llama_question_embedding/alpha','index.npy')),
                                    np.load(os.path.join(preloaded_root,'llama_question_embedding/alpha','embedding.npy'))]}
        else:
            preloaded = False
        if torch.cuda.device_count() > 1 and torch.distributed.is_available():torch.distributed.barrier()
        
        buffer        = AlongAnswerNumpyBufferForIndex(model.config.hidden_size, data_module.pop('mapping_system'),preloaded=preloaded)
        data_collator = QADataCollatorWithNegative(buffer=buffer,negative_sampler_num= negative_sampler_num)
        trainer       = BufferWithTrainer(buffer=buffer, negative_sampler_num= negative_sampler_num, 
                                    model=model, tokenizer=tokenizer, data_collator=data_collator,
                                    args=training_args, **data_module) # modify the make_supervised_data_module
    elif realtime_embedding_mode:
        if 'jina' in model_args.model_name_or_path.lower():
            model = BertEmbedder(model)
        else:
            model = LlamaLastWeightedEmbedder(model)
        model = PairQAGradientCacheModel(model,return_loss=True,label_temperature=training_args.label_temperature)
        data_module = make_supervised_data_module2(tokenizer=tokenizer, dummy_data=data_args.dummy_data,
                        model_max_length = training_args.model_max_length, 
                        datapair_path=training_args.datapair_path, 
                        evalpair_tuple=(training_args.eval_datapair_path,
                                        training_args.eval_datapair_question_token_path,
                                        training_args.eval_datapair_answer_token_path
                                        ),
                        use_reference=training_args.use_reference_label,
                        dispatch_batches=training_args.dispatch_batches, add_eval_dataset=training_args.add_eval_dataset)
        data_collator= QADataCollatorWithPaddingReference(tokenizer)
        training_args.logging_steps = 1
        knowledge_buffer_tuple=None
        compute_metrics = None 
        if training_args.add_eval_dataset: 
            assert training_args.knowledge_buffer, "you need to provide a knowledge buffer for evaluation"
            assert training_args.evaluation_strategy != 'no', "you need to provide a evaluation strategy"
            assert training_args.evaluation_strategy == 'epoch' or training_args.eval_steps >= 100, "so far, better avoid freuqent evaluation (commend this line if you real want to)"
            compute_metrics                 = format_topk_rank
        if training_args.knowledge_buffer:
            knowledge_buffer_embedding_list = [training_args.knowledge_buffer]
            knowledge_buffer_tuple          = load_wave_data(knowledge_buffer_embedding_list)
        trainer = RealtimeEmbeddingTrainer(knowledge_buffer=knowledge_buffer_tuple, model=model, 
                                           tokenizer=tokenizer, args=training_args, data_collator=data_collator, 
                                           compute_metrics= compute_metrics,
                                           **data_module)
    else:
        model = PairQAmodel(LlamaLastWeightedEmbedder(model),return_loss=True)
        data_module = make_supervised_data_module(tokenizer=tokenizer, data_args=data_args,
                    model_max_length = training_args.model_max_length, return_mapping=False,dispatch_batches=training_args.dispatch_batches)
        data_collator= QADataCollatorWithPadding(tokenizer)
        trainer = Trainer(model=model, tokenizer=tokenizer, args=training_args, data_collator=data_collator, **data_module)
    model.config.use_cache = False

    # if trainer.args.create_embedding: ### the transformer trainer has quite wired multi inference parallelism, so the correct way is put model here!!
    #     assert trainer.args.embedding_offline_path
    #     trainer.create_offline_embedding()
    #     trainer.accelerator.print(f'save embedding to {trainer.args.embedding_offline_path}')
    #     trainer.accelerator.wait_for_everyone()
    #     exit()

    if list(pathlib.Path(training_args.output_dir).glob("checkpoint-*")):
        trainer.train(resume_from_checkpoint=True)
    else:
        trainer.train()
    trainer.save_state()

    unwrapped_model = model
    while hasattr(unwrapped_model, 'module'):unwrapped_model = unwrapped_model.module
    model = unwrapped_model
    # check if zero3 mode enabled
    if deepspeed.is_deepspeed_zero3_enabled():
        # use deepspeed engine internal function to gather state dict
        # state_dict_zero3 contains whole parameters of base and lora adapters
        # we will not extract lora parameters since peft save_pretrained will do that
        # https://github.com/huggingface/peft/blob/3714aa2fff158fdfa637b2b65952580801d890b2/src/peft/peft_model.py#L125
        # https://github.com/huggingface/peft/blob/3714aa2fff158fdfa637b2b65952580801d890b2/src/peft/utils/save_and_load.py#L19
        state_dict_zero3 = trainer.model_wrapped._zero3_consolidated_16bit_state_dict()
        if training_args.local_rank == 0:
            state_dict = state_dict_zero3
    else:
        # in other mode we use original code from fastchat team, to make sure our change is minimum
        state_dict = get_peft_state_maybe_zero_3(
            model.named_parameters(), lora_args.lora_bias
        )

    if training_args.local_rank == 0:
        if not training_args.full_finetune:
            model.embedder.encoder.save_pretrained(training_args.output_dir, state_dict=state_dict)
        else:
            model.embedder.encoder.save_pretrained(training_args.output_dir)


def find_free_port():
    import socket
    s = socket.socket()
    s.bind(('', 0))            # Bind to a free port provided by the host.
    return s.getsockname()[1]  # Return the port number assigned.


def distributed_initial(args):
    import os
    ngpus = ngpus_per_node = torch.cuda.device_count()
    args.world_size = -1
    args.dist_file  = None
    args.rank       = 0
    args.dist_backend = "nccl"
    args.multiprocessing_distributed = ngpus>1
    args.ngpus_per_node = ngpus_per_node
    if not hasattr(args,'train_set'):args.train_set='large'
    ip = os.environ.get("MASTER_ADDR", "127.0.0.1")
    port = find_free_port()#os.environ.get("MASTER_PORT", f"{find_free_port()}" )
    args.port = port
    hosts = int(os.environ.get("WORLD_SIZE", "1"))  # number of nodes
    rank = int(os.environ.get("RANK", "0"))  # node id
    gpus = torch.cuda.device_count()  # gpus per node
    args.dist_url = f"tcp://{ip}:{port}"
    if args.world_size == -1 and "SLURM_NPROCS" in os.environ:
        args.world_size = int(os.environ["SLURM_NPROCS"])
        args.rank       = int(os.environ["SLURM_PROCID"])
        jobid           = os.environ["SLURM_JOBID"]

        hostfile        = "dist_url." + jobid  + ".txt"
        if args.dist_file is not None:
            args.dist_url = "file://{}.{}".format(os.path.realpath(args.dist_file), jobid)
        elif args.rank == 0:
            import socket
            ip = socket.gethostbyname(socket.gethostname())
            port = find_free_port()
            args.dist_url = "tcp://{}:{}".format(ip, port)
            #with open(hostfile, "w") as f:f.write(args.dist_url)
        else:
            import os
            import time
            while not os.path.exists(hostfile):
                time.sleep(1)
            with open(hostfile, "r") as f:
                args.dist_url = f.read()
        print("dist-url:{} at PROCID {} / {}".format(args.dist_url, args.rank, args.world_size))
    else:
        args.world_size = 1
    args.distributed = args.world_size > 1 or args.multiprocessing_distributed
    return args
import torch.distributed as dist
def slurm_distributed_initial(args):
    local_rank = get_local_rank()
    if args.dist_url == "env://" and args.rank == -1:args.rank = int(os.environ["RANK"])
    if args.multiprocessing_distributed:
        args.rank = args.rank * args.ngpus_per_node + local_rank
    print(f"start init_process_group,backend={args.dist_backend}, init_method={args.dist_url},world_size={args.world_size}, rank={args.rank}")
    dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,world_size=args.world_size, rank=args.rank)

if __name__ == "__main__":
    
    train()
