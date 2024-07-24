from dataclasses import dataclass, field
import json
from typing import Optional, Union
from trl import DPOConfig

@dataclass
class ScriptArguments:
    foo: str = 'bar'
    do_dpo: bool = False
    dpo_config: DPOConfig = None

@dataclass
class ModelArguments:
    model_id: Optional[str] = None
    from_pretrained: bool = False
    architecture: Optional[str] = 'mamba'
    use_lora: bool = False
    hidden_size: Optional[int] = 768
    intermediate_size: Optional[int] = 3072
    num_attention_heads: Optional[int] = 12
    state_size: Optional[int] = 16
    num_layers: Optional[int] = 32
    max_position_embeddings: Optional[int] = 1024
    freeze: Optional[list[str]] = field(default_factory=lambda: [])

    def __post_init__(self):
        if self.intermediate_size is None:
            self.intermediate_size = 4 * self.hidden_size

@dataclass
class DataArguments:
    num_train: int = 20_000_000
    num_eval: int = 100
    n_digits_train: int = 20
    n_digits_train_min: int = 1
    n_digits_eval_start: int = 15
    n_digits_eval_end: int = 30
    n_digits_eval_step: int = 5
    block_size: Optional[int] = 1024
    op: str = 'add'
    nproc: int = 8
    format: Optional[Union[dict, str]] = 'reverse'
    
    def __post_init__(self):
        if self.format.startswith("{"):
            self.format = json.loads(self.format)
