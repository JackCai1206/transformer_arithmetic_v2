import torch
from torch import nn
from torch.nn import functional as F
from typing import List, Optional, Tuple, Union
import math

from transformers import LlamaConfig, LlamaForCausalLM
from transformers.models.falcon.modeling_falcon import FalconRotaryEmbedding
from transformers.models.llama.modeling_llama import BaseModelOutputWithPast, LlamaModel, LlamaAttention, apply_rotary_pos_emb, repeat_kv
from transformers.cache_utils import DynamicCache, Cache

from lib.modeling.llama import LlamaModelWithNoPE

class LlamaTempSoftAttnConfig(LlamaConfig):
    use_lpe: bool = False
    temp_beta: float = 3.0 # traininable temperature parameter (inside softmax -> T = beta log n)
    fix_beta: bool = False


class LlamaTempSoftAttn(LlamaAttention):
    def __init__(self, config: LlamaConfig, layer_idx: Optional[int] = None):
        super().__init__(config)
        self.config = config
        self.layer_idx = layer_idx

        # Trainable temperature parameter beta
        self.temp_beta = config.temp_beta
        self.temp_beta = nn.Parameter(torch.tensor(self.temp_beta))
        if config.fix_beta:
            self.temp_beta.requires_grad = False
        
        self.attention_dropout = config.attention_dropout
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.max_position_embeddings = config.max_position_embeddings
        self.rope_theta = config.rope_theta
        self.is_causal = True

        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=config.attention_bias)
        self.k_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=config.attention_bias)
        self.v_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=config.attention_bias)
        self.o_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=config.attention_bias)
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,  # NOTE: we manually pass this
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        bsz, q_len, _ = hidden_states.size() # batch size, query length, hidden size

        if self.config.pretraining_tp > 1:
            key_value_slicing = (self.num_key_value_heads * self.head_dim) // self.config.pretraining_tp
            query_slices = self.q_proj.weight.split(
                (self.num_heads * self.head_dim) // self.config.pretraining_tp, dim=0
            )
            key_slices = self.k_proj.weight.split(key_value_slicing, dim=0)
            value_slices = self.v_proj.weight.split(key_value_slicing, dim=0)

            query_states = [F.linear(hidden_states, query_slices[i]) for i in range(self.config.pretraining_tp)]
            query_states = torch.cat(query_states, dim=-1)

            key_states = [F.linear(hidden_states, key_slices[i]) for i in range(self.config.pretraining_tp)]
            key_states = torch.cat(key_states, dim=-1)

            value_states = [F.linear(hidden_states, value_slices[i]) for i in range(self.config.pretraining_tp)]
            value_states = torch.cat(value_states, dim=-1)

        else:
            query_states = self.q_proj(hidden_states)
            key_states = self.k_proj(hidden_states)
            value_states = self.v_proj(hidden_states)

        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        past_key_value = getattr(self, "past_key_value", past_key_value)
        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        if past_key_value is not None:
            # sin and cos are specific to RoPE models; cache_position needed for the static cache
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)

        if attention_mask is not None:  # no matter the length, we just slice it
            # if cache_position is not None:
            #     causal_mask = attention_mask[:, :, cache_position, : key_states.shape[-2]]
            # print(attention_mask.shape) # torch.Size([1024, 76])
            # print(key_states.shape) # torch.Size([1024, 6, 76, 64])
            causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
            attn_weights = attn_weights + causal_mask


        #############################################################################################
        # #NOTE: for attn_weights, apply softmax with temperature parameter by multiplying each with T = beta log n
        # where n is the length of the input sequence
        #############################################################################################
        # Compute the scaling factor T for each position
        position_indices = torch.arange(1, q_len + 1, device=hidden_states.device).float()  # [1, 2, ..., q_len]
        scaling_factors = self.temp_beta * torch.log(position_indices)  # Compute T = beta * log(i)

        # Reshape scaling_factors to be compatible with attn_weights
        # We want to scale across the last dimension (q_len), so we reshape it accordingly
        scaling_factors = scaling_factors.view(1, 1, 1, q_len)  # Shape: (1, 1, 1, q_len)

        # Scale the attention weights with T
        attn_weights = attn_weights * scaling_factors  # Scale the logits # TODO: check if this should be division or multiplication -> it is multiplication

        # upcast attention to fp32
        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype) 
        attn_weights = nn.functional.dropout(attn_weights, p=self.attention_dropout, training=self.training)
        attn_output = torch.matmul(attn_weights, value_states)

        if attn_output.size() != (bsz, self.num_heads, q_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(bsz, self.num_heads, q_len, self.head_dim)}, but is"
                f" {attn_output.size()}"
            )

        attn_output = attn_output.transpose(1, 2).contiguous()

        attn_output = attn_output.reshape(bsz, q_len, -1)

        if self.config.pretraining_tp > 1:
            attn_output = attn_output.split(self.hidden_size // self.config.pretraining_tp, dim=2)
            o_proj_slices = self.o_proj.weight.split(self.hidden_size // self.config.pretraining_tp, dim=1)
            attn_output = sum([F.linear(attn_output[i], o_proj_slices[i]) for i in range(self.config.pretraining_tp)])
        else:
            attn_output = self.o_proj(attn_output)

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights, past_key_value


class LlamaModelTempSoftAttn(LlamaModelWithNoPE):
    def __init__(self, config: LlamaTempSoftAttnConfig):
        super().__init__(config)
        for i, layer in enumerate(self.layers):
            layer.self_attn = LlamaTempSoftAttn(config, layer_idx=i)


class LlamaForCausalLMTempSoftAttn(LlamaForCausalLM):
    def __init__(self, config: LlamaConfig):
        super().__init__(config)
        self.model = LlamaModelTempSoftAttn(config)
        self.post_init()

        # for name, param in self.model.layers[0].named_parameters():
        #     print(name, param)

        # self.model.layers[0].self_attn.temp_beta


