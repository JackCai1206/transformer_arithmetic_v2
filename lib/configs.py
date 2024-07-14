from dataclasses import dataclass
import json
from typing import Optional, Union

@dataclass
class ScriptArguments:
    foo: str = 'bar'

@dataclass
class ModelArguments:
    model_id: Optional[str] = None
    from_pretrained: bool = False
    architecture: Optional[str] = 'mamba'
    use_lora: bool = False
    hidden_size: Optional[int] = 768
    num_attention_heads: Optional[int] = 12
    state_size: Optional[int] = 16
    num_layers: Optional[int] = 32
    max_position_embeddings: Optional[int] = 1024

@dataclass
class DataArguments:
    num_train: int = 2_000_000
    num_eval: int = 100
    n_digits_train: int = 20
    n_digits_train_min: int = 1
    n_digits_eval_start: int = 15
    n_digits_eval_end: int = 30
    n_digits_eval_step: int = 5
    block_size: Optional[int] = 1024
    op: str = '+'
    nproc: int = 8
    format: Optional[Union[dict, str]] = 'reverse'
    
    def __post_init__(self):
        if self.format.startswith("{"):
            self.format = json.loads(self.format)
