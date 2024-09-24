import transformers
from dataclasses import dataclass, field
from fastchat.train.train import (
    DataArguments,
    ModelArguments,
    make_supervised_data_module,
)
import typing

@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: typing.Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    model_max_length: int = field(
        default=512,
        metadata={
            "help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."
        },
    )
    flash_attn: bool = True
    use_long_lora: bool = False
    remove_unused_columns: bool = False
    report_to: typing.List[str] = field(default_factory=lambda: [])
    negative_sampler_num: int = 200
    generate_chunk_size:int = 3
    full_finetune: bool = False
    start_lora_path: str = None
    label_temperature: float = 0.01
    extrapolation_scaling:  typing.Optional[float] = field(default=None)
    dummy_data: bool = False
    datapair_path: str = 'data/unarXive_quantum_physics/pair.answer_version_b.question_version_a.json'
    use_reference_label: bool = True
    knowledge_buffer: str = None
    add_eval_dataset: bool=False
    eval_mode: str = 'precompute_embedding'
    embedding_offline_path: str = None
    create_embedding: bool = False
    eval_datapair_path: str = None
    eval_datapair_question_token_path: str = None
    eval_datapair_answer_token_path: str = None
    real_batch_size: int = None
@dataclass
class LoraArguments:
    lora_r: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    lora_target_modules: typing.List[str] = field(
        default_factory=lambda: ["q_proj", "v_proj"]
    )
    lora_weight_path: str = ""
    lora_bias: str = "none"
    q_lora: bool = False

@dataclass
class SelfDistributedArguments:
    self_distributed_init: bool = False


import argparse
def print_namespace_tree(namespace, indent=0):
    namespace = vars(namespace) if not isinstance(namespace, dict) else namespace
    for key, value in namespace.items():
        print(' ' * indent, end='')
        if isinstance(value, (dict, argparse.Namespace)):
            print(key)
            print_namespace_tree(value, indent + 4)
        else:
            print(f"{key:30s} ---> {value}")

def convert_namespace_tree(namespace):
    namespace = vars(namespace)
    if isinstance(namespace,dict):
        return dict([(key, convert_namespace_tree(val)) for key, val in namespace.items()])
    else:
        return namespace
