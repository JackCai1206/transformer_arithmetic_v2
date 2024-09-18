from typing import List, Tuple, Union
import torch
from torch import FloatTensor, LongTensor, Tensor
from transformers import LlamaForCausalLM
from transformers.cache_utils import Cache
from transformers.modeling_outputs import CausalLMOutputWithPast

class LlamaRandPosId(LlamaForCausalLM):
    def forward(
        self,
        input_ids: LongTensor = None,
        attention_mask: Union[Tensor, None] = None,  # Replacing | with Union
        position_ids: Union[LongTensor, None] = None,
        past_key_values: Union[Cache, List[FloatTensor], None] = None,
        inputs_embeds: Union[FloatTensor, None] = None,
        labels: Union[LongTensor, None] = None,
        use_cache: Union[bool, None] = None,
        output_attentions: Union[bool, None] = None,
        output_hidden_states: Union[bool, None] = None,
        return_dict: Union[bool, None] = None,
        cache_position: Union[LongTensor, None] = None
    ) -> Union[Tuple, CausalLMOutputWithPast]:  # Using Union for return type as well
        if self.training:
            # position_ids = torch.arange(input_ids.size(1), device=input_ids.device) + torch.randint(0, self.config.k, (input_ids.size(0), 1), device=input_ids.device)
            all_ids = torch.arange(0, self.config.max_position_embeddings, device=input_ids.device).repeat(input_ids.size(0), 1)
            position_ids = all_ids.take_along_dim(torch.sort(torch.rand(*all_ids.shape))[1], 1)[:, :input_ids.size(1)].sort(1)[0]
            breakpoint()
        return super().forward(input_ids, attention_mask, position_ids, past_key_values, inputs_embeds, labels, use_cache, output_attentions, output_hidden_states, return_dict, cache_position)
