from transformers.models.gpt_neox import GPTNeoXConfig, GPTNeoXModel
from torch import nn
 
class DeepThinkingModel(GPTNeoXModel):
    def __init__(self, config: GPTNeoXConfig):
        super().__init__(config)
        self.config = config
        self.gate = nn.Linear(config.hidden_size, 1)
 
    def forward(self, input_ids, **kwargs):
        device = input_ids.device if input_ids is not None else inputs_embeds.device
        position_ids = torch.arange(past_length, seq_length + past_length, dtype=torch.long, device=device)
        position_ids = position_ids.unsqueeze(0)
        while :
            outputs = super().forward(input_ids, **kwargs)
            hidden_states = outputs.last_hidden_state
            r = self.gate(hidden_states) # (batch_size, seq_len, 1)
