from dataclasses import dataclass, field
import json
from typing import Optional, Tuple, Union
from trl import DPOConfig
import numpy as np

@dataclass
class ScriptArguments:
    foo: str = 'bar'
    do_dpo: bool = False

@dataclass
class ModelArguments:
    model_id: Optional[str] = None
    from_pretrained: bool = False
    use_lora: bool = False
    architecture: Optional[str] = 'mamba'
    rope_theta: Optional[float] = np.inf
    hidden_size: Optional[int] = 768
    intermediate_size: Optional[int] = 3072
    num_attention_heads: Optional[int] = 12
    state_size: Optional[int] = 16
    num_layers: Optional[int] = 32
    max_position_embeddings: Optional[int] = 1024
    freeze: Optional[str] = None
    freeze_except: Optional[str] = None

    def __post_init__(self):
        if self.intermediate_size is None:
            self.intermediate_size = 4 * self.hidden_size

@dataclass
class DataArguments:
    num_train: int = 20_000_000
    num_eval: int = 100
    num_dpo_data: int = 10000
    n_digits_train: Optional[Union[Tuple[Tuple[int]], str]] = '1,20'
    n_digits_eval: Optional[Union[Tuple[int], str]] = '15,30,5'
    n_digits_dpo: Optional[Union[Tuple[int], str]] = '1,20'
    block_size: Optional[int] = 1024
    op_train: Optional[Union[Tuple[str], str]] = 'add'
    op_eval: Optional[Union[Tuple[str], str]] = 'add'
    op_dist_train: Optional[Union[Tuple[float], str]] = '1'
    op_dist_eval: Optional[Union[Tuple[float], str]] = '1'
    nproc: int = 16
    format_train: Optional[Union[Tuple[str], str]] = 'reverse'
    format_eval: Optional[Union[Tuple[str], str]] = 'reverse'
    add_special_tokens: bool = True
    show_task_ids: bool = True

    def __post_init__(self):
    #     if self.format.startswith("{"):
    #         self.format = json.loads(self.format)
        self.op_dist_train = tuple(map(float, self.op_dist_train.split(' ')))
        self.op_train = tuple(self.op_train.split(' '))
        self.format_train = tuple(self.format_train.split(' '))
        self.n_digits_train = tuple(map(lambda x: tuple(map(int, x.split(','))), self.n_digits_train.split(' ')))
        self.op_eval = tuple(self.op_eval.split(' '))
        self.op_dist_eval = tuple(map(float, self.op_dist_eval.split(' ')))
        self.format_eval = tuple(self.format_eval.split(' '))
        self.n_digits_eval = tuple(map(int, self.n_digits_eval.split(',')))
        self.n_digits_dpo = tuple(map(int, self.n_digits_dpo.split(',')))
