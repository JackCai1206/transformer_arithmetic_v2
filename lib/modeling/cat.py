from transformers.models.llama.modeling_llama import LlamaForCausalLM, LlamaAttention, LlamaConfig, apply_rotary_pos_emb, repeat_kv
from transformers import Cache
import torch
import torch.nn as nn
from typing import Optional, Tuple
from torch.nn import functional as F
import math

class ConvLlamaAttention(LlamaAttention):
    def __init__(self, config: LlamaConfig, layer_idx: int):
        super().__init__(config, layer_idx)
        # self.query_conv = nn.Conv1d(config.hidden_size, config.hidden_size, kernel_size=5, padding=2)
        # self.key_conv   = nn.Conv1d(config.hidden_size, config.hidden_size, kernel_size=5, padding=2)
        # self.value_conv = nn.Conv1d(config.hidden_size, config.hidden_size, kernel_size=5, padding=2)
        self.query_conv = nn.Identity()
        self.key_conv = nn.Identity()
        self.value_conv = nn.Identity()

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        bsz, q_len, _ = hidden_states.size()

        q_conv = self.query_conv(hidden_states.transpose(1, 2)).transpose(1, 2)
        query_states = self.q_proj(q_conv)

        k_conv = self.key_conv(hidden_states.transpose(1, 2)).transpose(1, 2)
        key_states = self.k_proj(k_conv)
        
        v_conv = self.value_conv(hidden_states.transpose(1, 2)).transpose(1, 2)
        value_states = self.v_proj(v_conv)

        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        cos, sin = self.rotary_emb(value_states, position_ids)
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        if past_key_value is not None:
            # sin and cos are specific to RoPE models; cache_position needed for the static cache
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)

        assert attention_mask is not None
        if attention_mask is not None:  # no matter the length, we just slice it
            causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
            attn_weights = attn_weights + causal_mask

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

        attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)

        attn_output = self.o_proj(attn_output)

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights, past_key_value

class ConvLlamaForCausalLM(LlamaForCausalLM):
    def __init__(self, config):
        super().__init__(config)
        for i, l in enumerate(self.model.layers):
            l.self_attn = ConvLlamaAttention(config, i)
        self.post_init()
