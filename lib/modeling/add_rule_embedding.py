from torch import nn
import torch
from transformers import LlamaForCausalLM, LlamaConfig

class AddRuleEmbedding(nn.Module):
    def __init__(self, embedding: nn.Embedding, add_rules={}):
        super().__init__()
        self.embedding = embedding
        self.add_rules = add_rules

    def forward(self, x):
        emb = self.embedding.forward(x)
        for key, value in self.add_rules.items():
            summands = self.embedding.forward(torch.tensor(key, device=x.device))
            emb[x == value] = summands.sum(0)
        return emb

class LlamaConfigWithAddRules(LlamaConfig):
    add_rules = ''

class LlamaModelWithAddRules(LlamaForCausalLM):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model.embed_tokens = AddRuleEmbedding(self.model.embed_tokens, add_rules=eval(self.config.add_rules))
        self.post_init()
