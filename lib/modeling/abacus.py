from typing import List, Optional, Tuple, Union
import torch
import random
from transformers import FalconConfig, FalconForCausalLM, LlamaConfig, LlamaForCausalLM
from transformers.models.falcon.modeling_falcon import FalconRotaryEmbedding
from transformers.models.llama.modeling_llama import BaseModelOutputWithPast, LlamaModel, LlamaRMSNorm
from transformers.cache_utils import DynamicCache, Cache

# """Implementation of abacus embeddings"""
# # Example of how to extract digit tokens to pass into constructor
# # digit_tokens = tokenizer.convert_tokens_to_ids(['0','1','2','3','4','5','6','7','8','9'])

# class Abacus(torch.nn.Module):
#     """
#     Abacus Embeddings, learned emebddings resued for each digit.
#     Integers must be reversed for this to work correctly.
#     Transformers Can Do Arithmetic with the Right Embeddings, McLeish et al. (2024)
#     """
#     def __init__(self, digit_tokens, embedding_dim, max_seq_length=1024, max_k=300):
#         """
#         digit_tokens (list): list of the tokens for each of the 10 digits, `digit_tokens = tokenizer.convert_tokens_to_ids(['0','1','2','3','4','5','6','7','8','9'])`
#         embedding_dim (int): dimension to embed into
#         max_seq_length (int): maximum number of embeddings that can be trained
#         max_k (int): maximum k value which we randomly shift by during training
#         """
#         super().__init__()
#         self.embedding = torch.nn.Embedding(max_seq_length, embedding_dim)
#         self.register_buffer("digits", torch.tensor(digit_tokens), persistent=False)

#         self.max_k = max_k

#     def helper(self, mask, device):
#         """
#         Converts a binary mask of digit locations into spans of consecutive digits
#         """
#         mask_shape = mask.shape
        
#         # Create a shifted version of the mask to detect changes from 0 to 1
#         shifted_mask = torch.cat([torch.zeros((mask_shape[0], 1), device=device, dtype=mask.dtype), mask[:, :-1]], dim=1)
#         starts = (shifted_mask != mask) & mask
        
#         # Generate IDs for each segment of 1s, processing row-wise
#         segment_ids = torch.cumsum(starts, dim=1)

#         # Generate an index array row-wise
#         index = torch.arange(mask.size(1)).repeat(mask.size(0), 1).to(device)
        
#         # Reset index at the start of each segment
#         reset_index = torch.zeros_like(mask).long()
#         second_term = index * starts.long()
#         reset_index = reset_index.scatter_add(1, segment_ids, second_term)
        
#         # Calculate positions in segment
#         positions = index - reset_index.gather(1, segment_ids) + 1
        
#         # Ensure only values within 1-segments are non-zero
#         result = positions * mask

#         return result

#     def forward(self, input_ids):
#         """
#         input_ids (tensor): a batch of inputs, each row is a sample
#         """
        
#         mask = torch.isin(input_ids, self.digits)
#         output = self.helper(mask, input_ids.device)

#         k=0
#         if self.training:
#             k = random.randint(0, self.max_k)
#             output[output>0] += k # as we already have ones in the tensor, the tensor values will be k+1

#         return self.embedding(output)

class AbacusConfig:
    digit_tokens: List[int] = None
    max_k: int = 200
    return_position_ids: bool = False

class AbacusMixin:
    def helper(self, mask, device):
        """
        Converts a binary mask of digit locations into spans of consecutive digits
        """
        mask_shape = mask.shape
        
        # Create a shifted version of the mask to detect changes from 0 to 1
        shifted_mask = torch.cat([torch.zeros((mask_shape[0], 1), device=device, dtype=mask.dtype), mask[:, :-1]], dim=1)
        starts = (shifted_mask != mask) & mask
        
        # Generate IDs for each segment of 1s, processing row-wise
        segment_ids = torch.cumsum(starts, dim=1)

        # Generate an index array row-wise
        index = torch.arange(mask.size(1)).repeat(mask.size(0), 1).to(device)
        
        # Reset index at the start of each segment
        reset_index = torch.zeros_like(mask).long()
        second_term = index * starts.long()
        reset_index = reset_index.scatter_add(1, segment_ids, second_term)
        
        # Calculate positions in segment
        positions = index - reset_index.gather(1, segment_ids) + 1

        # Ensure only values within 1-segments are non-zero
        result = positions * mask

        return result

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs) # this will initialize the model
        assert self.config is not None
        self.register_buffer("digits", torch.tensor(self.config.digit_tokens), persistent=False)
        self.abacus_embedding = torch.nn.Embedding(self.config.max_position_embeddings, self.config.hidden_size)
        self.offset = None

    def get_abacus_position_ids(self, input_ids, use_cache=None):
        mask = torch.isin(input_ids, self.digits)

        if not self.training:
            if use_cache:
                position_ids = (mask).int() + self.offset * (mask).int()
                self.offset = position_ids
            else:
                position_ids = self.helper(mask, input_ids.device)
                self.offset = position_ids[:, -1:]
        else:
            position_ids = self.helper(mask, input_ids.device)

        k=0
        if self.training:
            k = random.randint(0, self.config.max_k)
            position_ids[position_ids>0] += k

        if self.config.return_position_ids:
            return position_ids
        else:
            return self.abacus_embedding(position_ids)
    

# class AbacusFalconConfig(FalconConfig, AbacusConfig):
#     pass

# class AbacusFalconRotaryEmbedding(FalconRotaryEmbedding):
#     def forward(self, x, seq_len=None):
#         return super().forward(x, self.max_position_embeddings)

# class AbacusFalconModel(AbacusMixin, FalconForCausalLM):
#     def __init__(self, config: AbacusFalconConfig):
#         super().__init__(config)
#         for layer in self.transformer.h:
#             layer.self_attention.rotary_emb = AbacusFalconRotaryEmbedding(
#                 layer.self_attention.head_dim,
#                 max_position_embeddings=config.max_position_embeddings,
#                 base=config.rope_theta
#             )
    
#     def forward(self, input_ids, *args, labels=None, attention_mask=None, position_ids=None, **kwargs):
#         position_ids = self.get_abacus_position_ids(input_ids)
#         return super().forward(input_ids, position_ids=position_ids, *args, labels=labels, attention_mask=attention_mask, **kwargs)


class AbacusLlamaConfig(LlamaConfig, AbacusConfig):
    partial_rotary_factor: float = 1
    pass

class AbacusLlamaModel(AbacusMixin, LlamaModel):
    def __init__(self, config: AbacusLlamaConfig):
        super().__init__(config)

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Union[Cache, List[torch.FloatTensor]]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
    ) -> Union[Tuple, BaseModelOutputWithPast]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError(
                "You cannot specify both input_ids and inputs_embeds at the same time, and must specify either one"
            )

        if self.gradient_checkpointing and self.training and use_cache:
            # logger.warning_once(
            #     "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`."
            # )
            use_cache = False

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        return_legacy_cache = False
        if (
            use_cache and not isinstance(past_key_values, Cache) and not self.training
        ):  # kept for BC (non `Cache` `past_key_values` inputs)
            return_legacy_cache = True
            past_key_values = DynamicCache.from_legacy_cache(past_key_values)
            # logger.warning_once(
            #     "We detected that you are passing `past_key_values` as a tuple and this is deprecated and will be removed in v4.43. "
            #     "Please use an appropriate `Cache` class (https://huggingface.co/docs/transformers/v4.41.3/en/internal/generation_utils#transformers.Cache)"
            # )

        if cache_position is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            cache_position = torch.arange(
                past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
            )
        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)

        causal_mask = self._update_causal_mask(
            attention_mask, inputs_embeds, cache_position, past_key_values, output_attentions
        )
        hidden_states = inputs_embeds

        # create position embeddings to be shared across the decoder layers
        B, L = hidden_states.size()[:2]
        D = self.config.hidden_size // self.config.num_attention_heads
        position_embeddings = torch.ones(B, L, D, device=hidden_states.device), torch.zeros(B, L, D, device=hidden_states.device)
        if self.config.rope_theta != torch.inf:
            rotary_dim = int(self.config.partial_rotary_factor * D)
            cos, sin = self.rotary_emb(hidden_states, position_ids)
            position_embeddings[0][:, :, :rotary_dim] = cos
            position_embeddings[1][:, :, :rotary_dim] = sin

        # Add abacus embeddings
        abacus_embeddings = self.get_abacus_position_ids(input_ids, use_cache=past_key_values is not None and past_key_values.get_seq_length() > 0)
        hidden_states += abacus_embeddings
        # hidden_states = self.norm(hidden_states)

        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        next_decoder_cache = None

        for decoder_layer in self.layers:
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            if self.gradient_checkpointing and self.training:
                layer_outputs = self._gradient_checkpointing_func(
                    decoder_layer.__call__,
                    hidden_states,
                    causal_mask,
                    position_ids,
                    past_key_values,
                    output_attentions,
                    use_cache,
                    cache_position,
                    position_embeddings,
                )
            else:
                layer_outputs = decoder_layer(
                    hidden_states,
                    attention_mask=causal_mask,
                    position_ids=position_ids,
                    past_key_value=past_key_values,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                    cache_position=cache_position,
                    position_embeddings=position_embeddings,
                )

            hidden_states = layer_outputs[0]

            if use_cache:
                next_decoder_cache = layer_outputs[2 if output_attentions else 1]

            if output_attentions:
                all_self_attns += (layer_outputs[1],)

        hidden_states = self.norm(hidden_states)

        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        next_cache = next_decoder_cache if use_cache else None
        if return_legacy_cache:
            next_cache = next_cache.to_legacy_cache()

        if not return_dict:
            return tuple(v for v in [hidden_states, next_cache, all_hidden_states, all_self_attns] if v is not None)
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
        )

class AbacusLlamaForCausalLM(LlamaForCausalLM):
    def __init__(self, config: AbacusLlamaConfig):
        super().__init__(config)
        self.model = AbacusLlamaModel(config)
        self.post_init()
