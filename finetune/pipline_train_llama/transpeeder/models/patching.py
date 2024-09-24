""" https://github.com/lm-sys/FastChat/blob/main/fastchat/train/llama_flash_attn_monkey_patch.py.
"""

from typing import List, Optional, Tuple, Dict

import torch
import transformers
from transformers.models.llama.modeling_llama import apply_rotary_pos_emb

from einops import rearrange
from flash_attn.flash_attn_interface import flash_attn_varlen_qkvpacked_func
from flash_attn.bert_padding import unpad_input, pad_input


def smart_tokenizer_and_embedding_resize(
    special_tokens_dict: Dict,
    tokenizer: transformers.PreTrainedTokenizer,
    model: transformers.PreTrainedModel,
):
    """Resize tokenizer and embedding.

    Note: This is the unoptimized version that may make your embedding size not be divisible by 64.
    """
    # TODO: padding embedding size for being divisible by 64.
    num_new_tokens = tokenizer.add_special_tokens(special_tokens_dict)
    model.resize_token_embeddings(len(tokenizer))

    if num_new_tokens > 0:
        input_embeddings = model.get_input_embeddings().weight.data
        output_embeddings = model.get_output_embeddings().weight.data

        input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)
        output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)

        input_embeddings[-num_new_tokens:] = input_embeddings_avg
        output_embeddings[-num_new_tokens:] = output_embeddings_avg


def llama_flash_attn_forward(
    self,
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.Tensor] = None,
    past_key_value: Optional[Tuple[torch.Tensor]] = None,
    output_attentions: bool = False,
    use_cache: bool = False,
) -> Tuple[torch.Tensor, Optional[torch.Tensor],
            Optional[Tuple[torch.Tensor]]]:
    """Input shape: Batch x Time x Channel

    attention_mask: [bsz, q_len]
    """
    bsz, q_len, _ = hidden_states.size()

    query_states = self.q_proj(hidden_states).view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
    key_states = self.k_proj(hidden_states).view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
    value_states = self.v_proj(hidden_states).view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
    # [bsz, q_len, nh, hd]
    # [bsz, nh, q_len, hd]

    kv_seq_len = key_states.shape[-2]
    assert past_key_value is None, "past_key_value is not supported"

    cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)
    query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)
    # [bsz, nh, t, hd]
    assert not output_attentions, "output_attentions is not supported"
    assert not use_cache, "use_cache is not supported"

    # Flash attention codes from
    # https://github.com/HazyResearch/flash-attention/blob/main/flash_attn/flash_attention.py

    # transform the data into the format required by flash attention
    qkv = torch.stack([query_states, key_states, value_states], dim=2) # [bsz, nh, 3, q_len, hd]
    qkv = qkv.transpose(1, 3) # [bsz, q_len, 3, nh, hd]
    # We have disabled _prepare_decoder_attention_mask in LlamaModel
    # the attention_mask should be the same as the key_padding_mask
    attention_mask = torch.ones((bsz, q_len), device=qkv.device)
    key_padding_mask = attention_mask


    if key_padding_mask is None:
        qkv = rearrange(qkv, 'b s ... -> (b s) ...')
        max_s = q_len
        cu_q_lens = torch.arange(0, (bsz + 1) * q_len, step=q_len, dtype=torch.int32,
                                device=qkv.device)
        output = flash_attn_varlen_qkvpacked_func(
            qkv, cu_q_lens, max_s, 0.0,
            softmax_scale=None, causal=True
        )
        output = rearrange(output, '(b s) ... -> b s ...', b=bsz)
    else:
        nheads = qkv.shape[-2]
        x = rearrange(qkv, 'b s three h d -> b s (three h d)')
        x_unpad, indices, cu_q_lens, max_s = unpad_input(x, key_padding_mask)
        x_unpad = rearrange(x_unpad, 'nnz (three h d) -> nnz three h d', three=3, h=nheads)
        output_unpad = flash_attn_varlen_qkvpacked_func(
            x_unpad, cu_q_lens, max_s, 0.0,
            softmax_scale=None, causal=True
        )
        output = rearrange(pad_input(rearrange(output_unpad, 'nnz h d -> nnz (h d)'),
                                    indices, bsz, q_len),
                        'b s (h d) -> b s h d', h=nheads)
    return self.o_proj(rearrange(output,
                                    'b s h d -> b s (h d)')), None, None


# Disable the transformation of the attention mask in LlamaModel as the flash attention
# requires the attention mask to be the same as the key_padding_mask
def _prepare_decoder_attention_mask(self,
                                    attention_mask,
                                    input_shape,
                                    inputs_embeds,
                                    past_key_values_length):
    # [bsz, seq_len]
    return attention_mask


def replace_llama_attn_with_flash_attn():
    transformers.models.llama.modeling_llama.LlamaModel._prepare_decoder_attention_mask = _prepare_decoder_attention_mask
    transformers.models.llama.modeling_llama.LlamaAttention.forward = llama_flash_attn_forward

def modify_llama_model(embedder):
    # Modify the "o_proj" layer in each LlamaDecoderLayer
    for layer in embedder.encoder.model.layers:
        layer.self_attn.o_proj.bias = torch.nn.Parameter(torch.zeros_like(layer.self_attn.o_proj.weight[:,0]))
    # Modify the "gate_proj" layer in each LlamaDecoderLayer
    for layer in embedder.encoder.model.layers:
        layer.mlp.gate_proj.bias = torch.nn.Parameter(torch.zeros_like(layer.mlp.gate_proj.weight[:,0]))
    # Modify the "lm_head" layer
    embedder.encoder.lm_head.bias = torch.nn.Parameter(torch.zeros_like(embedder.encoder.lm_head.weight[:,0]))
    return embedder

from transformers.models.llama.modeling_llama import LlamaAttention,LlamaMLP
def addbias(module):
    if isinstance(module, LlamaAttention):
        module.o_proj.bias = torch.nn.Parameter(torch.zeros_like(module.o_proj.weight[:,0]))
    elif isinstance(module, LlamaMLP):
        module.gate_proj.bias = torch.nn.Parameter(torch.zeros_like(module.gate_proj.weight[:,0]))
    else:
        for name, child in module.named_children():
            addbias(child)

import deepspeed
from deepspeed.runtime.state_dict_factory import SDLoaderFactory
def load_state_dir_withoutbias(self, load_dir, checkpoint_engine, strict=True):
    for idx, layer in enumerate(self.forward_funcs):
        # Functions, etc. will not have state_dicts
        if not hasattr(layer, 'load_state_dict'):
            continue

        # get all checkpoint files for the layer.
        model_ckpt_list = self.ckpt_layer_path_list(load_dir, idx)
        mp_rank = self._grid.get_slice_parallel_rank()
        mp_world_size = self._grid.get_slice_parallel_world_size()

        sd_loader = SDLoaderFactory.get_sd_loader(model_ckpt_list,
                                                    version=2.0,
                                                    checkpoint_engine=checkpoint_engine)
        load_path, checkpoint, _ = sd_loader.load(mp_world_size, mp_rank, module_key=None, is_pipe_parallel=True)
        layer.load_state_dict(checkpoint,strict=False)

        # if self._grid.data_parallel_id == 0:
        #     logger.info(
        #         f'RANK={self.global_rank} Loaded layer={idx+self._local_start} file={load_path}'
        #     )

    self._synchronize_tied_weights()

def replace_state_load_to_skip_bias():
    deepspeed.runtime.pipe.module.PipelineModule.load_state_dir =  load_state_dir_withoutbias


