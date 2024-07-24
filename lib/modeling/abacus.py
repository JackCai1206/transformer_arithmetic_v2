from typing import List
import torch
import random
from transformers import GPT2Config, GPT2LMHeadModel, FalconConfig, FalconForCausalLM, LlamaConfig, LlamaForCausalLM
from transformers.models.falcon.modeling_falcon import FalconRotaryEmbedding

"""Implementation of abacus embeddings"""
# Example of how to extract digit tokens to pass into constructor
# digit_tokens = tokenizer.convert_tokens_to_ids(['0','1','2','3','4','5','6','7','8','9'])

class Abacus(torch.nn.Module):
    """
    Abacus Embeddings, learned emebddings resued for each digit.
    Integers must be reversed for this to work correctly.
    Transformers Can Do Arithmetic with the Right Embeddings, McLeish et al. (2024)
    """
    def __init__(self, digit_tokens, embedding_dim, max_seq_length=1024, max_k=300):
        """
        digit_tokens (list): list of the tokens for each of the 10 digits, `digit_tokens = tokenizer.convert_tokens_to_ids(['0','1','2','3','4','5','6','7','8','9'])`
        embedding_dim (int): dimension to embed into
        max_seq_length (int): maximum number of embeddings that can be trained
        max_k (int): maximum k value which we randomly shift by during training
        """
        super().__init__()
        self.embedding = torch.nn.Embedding(max_seq_length, embedding_dim)
        self.register_buffer("digits", torch.tensor(digit_tokens), persistent=False)

        self.max_k = max_k

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

    def forward(self, input_ids):
        """
        input_ids (tensor): a batch of inputs, each row is a sample
        """
        
        mask = torch.isin(input_ids, self.digits)
        output = self.helper(mask, input_ids.device)

        k=0
        if self.training:
            k = random.randint(0, self.max_k)
            output[output>0] += k # as we already have ones in the tensor, the tensor values will be k+1

        return self.embedding(output)

class AbacusConfig:
    digit_tokens: List[int] = None
    max_k: int = 128

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
    
    def get_abacus_position_ids(self, input_ids):
        mask = torch.isin(input_ids, self.digits)
        position_ids = self.helper(mask, input_ids.device)

        k=0
        if self.training:
            k = random.randint(0, self.config.max_k)
            position_ids[position_ids>0] += k

        return position_ids
    

class AbacusFalconConfig(FalconConfig, AbacusConfig):
    pass

class AbacusFalconRotaryEmbedding(FalconRotaryEmbedding):
    def forward(self, x, seq_len=None):
        return super().forward(x, self.max_position_embeddings)

class AbacusFalconModel(AbacusMixin, FalconForCausalLM):
    def __init__(self, config: AbacusFalconConfig):
        super().__init__(config)
        for layer in self.transformer.h:
            layer.self_attention.rotary_emb = AbacusFalconRotaryEmbedding(
                layer.self_attention.head_dim,
                max_position_embeddings=config.max_position_embeddings,
                base=config.rope_theta
            )
    
    def forward(self, input_ids, *args, labels=None, attention_mask=None, position_ids=None, **kwargs):
        position_ids = self.get_abacus_position_ids(input_ids)
        return super().forward(input_ids, position_ids=position_ids, *args, labels=labels, attention_mask=attention_mask, **kwargs)


class AbacusGPT2Config(GPT2Config, AbacusConfig):
    pass

class AbacusGPT2Model(AbacusMixin, GPT2LMHeadModel):
    def __init__(self, config: AbacusGPT2Config):
        super().__init__(config)

    def forward(self, input_ids, *args, labels=None, attention_mask=None, position_ids=None, **kwargs):
        position_ids = self.get_abacus_position_ids(input_ids)
        return super().forward(input_ids, position_ids=position_ids, *args, labels=labels, attention_mask=attention_mask, **kwargs)
