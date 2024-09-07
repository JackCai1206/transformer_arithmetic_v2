from typing import List, Tuple
import torch
from torch import FloatTensor, LongTensor, Tensor
from transformers import LlamaForCausalLM
from transformers.cache_utils import Cache
from transformers.modeling_outputs import CausalLMOutputWithPast

class LlamaRandPosId(LlamaForCausalLM):
    def forward(self, input_ids: LongTensor = None, attention_mask: Tensor | None = None, position_ids: LongTensor | None = None, past_key_values: Cache | List[FloatTensor] | None = None, inputs_embeds: FloatTensor | None = None, labels: LongTensor | None = None, use_cache: bool | None = None, output_attentions: bool | None = None, output_hidden_states: bool | None = None, return_dict: bool | None = None, cache_position: LongTensor | None = None) -> Tuple | CausalLMOutputWithPast:
        if self.training:
            # position_ids = torch.arange(input_ids.size(1), device=input_ids.device) + torch.randint(0, self.config.k, (input_ids.size(0), 1), device=input_ids.device)
            all_ids = torch.arange(0, self.config.max_position_embeddings, device=input_ids.device).repeat(input_ids.size(0), 1)
            position_ids = all_ids.take_along_dim(torch.sort(torch.rand(*all_ids.shape))[1], 1)[:, :input_ids.size(1)].sort(1)[0]
            breakpoint()
        return super().forward(input_ids, attention_mask, position_ids, past_key_values, inputs_embeds, labels, use_cache, output_attentions, output_hidden_states, return_dict, cache_position)
