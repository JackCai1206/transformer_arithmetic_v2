from dataclasses import dataclass, field
import json
from typing import Optional, Tuple, Union
from trl import DPOConfig
import numpy as np
from transformers import Seq2SeqTrainingArguments

@dataclass
class MyTrainingArguments(Seq2SeqTrainingArguments):
    do_backtrack_decoding: bool = False # Automatically adds backtrack tokens during generation if the model generates the wrong token
    do_backtrack_decoding2: bool = False
    do_backtrack_eval: bool = False # erases backtrack tokens during evaluation
    backtrack_decoding_multiplier: Optional[int] = 3 # Multiplier for the number of max_new_tokens

    get_real_label: Optional[bool] = False # if True, then evaluate with the real label instead of the generated label (for self-improve)

    do_dpo: bool = False

    early_stopping: Optional[bool] = False # Stop training when the model reaches a certain metric
    do_beam_search: Optional[bool] = False # Use beam search during generation
    num_beams: Optional[int] = 1 # Number of beams for beam search
    # num_return_sequences: Optional[int] = 5 # Number of sequences to return for each input
    log_beta: Optional[bool] = False # Log beta values to wandb
    track_num_tokens_seen_by_task: bool = False
    early_stop: bool = False

    output_dir: Optional[str] = 'out'
    should_save: Optional[bool] = True # If false, surpress creating new directories for Trainer


@dataclass
class ScriptArguments:
    foo: str = 'bar'
    ref_model: bool = False
    ref_model_path: Optional[str] = None
    dpo_beta: Optional[float] = 0.1
    eval_more: Optional[bool] = False # use compute_metrics_new instead of compute_metrics if True
    wandb_project: Optional[str] = 'backtrack'
    result_name: Optional[str] = 'result'
    
    
@dataclass
class ModelArguments:
    model_id: Optional[str] = None
    from_pretrained: bool = False
    use_lora: bool = False
    architecture: Optional[str] = 'mamba'
    rope_theta: Optional[float] = np.inf
    partial_rotary_factor: Optional[float] = 1.0
    hidden_size: Optional[int] = 768
    intermediate_size: Optional[int] = 3072
    num_attention_heads: Optional[int] = 12
    state_size: Optional[int] = 16
    num_layers: Optional[int] = 32
    max_position_embeddings: Optional[int] = 1024
    freeze: Optional[str] = None
    freeze_except: Optional[str] = None
    fix_beta: Optional[bool] = False
    attention_dropout: Optional[float] = 0.0

    def __post_init__(self):
        if self.intermediate_size is None:
            self.intermediate_size = 4 * self.hidden_size

@dataclass
class DataArguments:
    use_iterable_dataset: bool = True
    num_train: Optional[Union[Tuple[Tuple[int]], str]] = '20_000_000'
    num_eval: int = 100
    eval_samples_file: str = 'data'
    train_file: Optional[str] = 'data/train'
    num_dpo_data: int = 10000
    n_digits_train: Optional[Union[Tuple[Tuple[int]], str]] = '1,20'
    n_digits_eval: Optional[Union[Tuple[int], str]] = '15,30,5'
    n_digits_dpo: Optional[Union[Tuple[int], str]] = '1,20'
    block_size: Optional[int] = 1024
    op_train: Optional[Union[Tuple[str], str]] = 'add'
    op_eval: Optional[Union[Tuple[str], str]] = 'add'
    op_dist_train: Optional[Union[Tuple[float], str]] = '1'
    op_dist_eval: Optional[Union[Tuple[float], str]] = '1'
    nproc: int = 8
    format_train: Optional[Union[Tuple[str], str]] = 'reverse'
    format_eval: Optional[Union[Tuple[str], str]] = 'reverse'
    add_special_tokens: bool = True
    show_task_ids: bool = True
    padding_side: str = 'right'
    use_train_attention_mask: bool = True
    train_pad_to: Optional[int] = None
    eval_pad_to: Optional[int] = None
    backtrack_p: Optional[float] = 0.2
    backtrack_mask: Optional[bool] = False
    mixture_scheduling_kwargs: Optional[Union[dict, str]] = field(default_factory=dict)
    dpo_format: Optional[str] = None # 'repeat_penalty' or 'backtrack_reward'
    load_as_iterable_dataset: bool = True
    no_seed_for_data: bool = False
    data_from_saved_train_base: Optional[str] = None # original data that was used to train initial model, directory after 'data/train'
    data_from_saved_train_new: Optional[Union[Tuple[str], str]] = None # directory after 'data/train_from_model-.../'
    eval_file_from_model: Optional[str] = None # directory after 'data/eval_from_model-.../'
    train_file_from_model: Optional[str] = None # directory after 'data/train_from_model-.../'

    
    def __post_init__(self):
    #     if self.format.startswith("{"):
    #         self.format = json.loads(self.format)
        self.op_dist_train = tuple(map(lambda x: tuple(map(float, x.split(','))), self.op_dist_train.split(' ')))
        self.op_train = tuple(self.op_train.split(' '))
        self.format_train = tuple(self.format_train.split(' '))
        self.n_digits_train = tuple(map(lambda x: tuple(map(int, x.split(','))), self.n_digits_train.split(' ')))
        self.num_train = tuple(map(int, self.num_train.split(' ')))
        if len(self.num_train) == 1:
            self.num_train = self.num_train * len(self.op_train)
        assert len(self.num_train) == len(self.op_train) == len(self.op_dist_train[0]) == len(self.format_train) == len(self.n_digits_train), 'You must provide the same number of values for num_train, op_train, op_dist_train, format_train, and n_digits_train'
        self.op_eval = tuple(self.op_eval.split(' '))
        self.op_dist_eval = tuple(map(float, self.op_dist_eval.split(' ')))
        self.format_eval = tuple(self.format_eval.split(' '))
        self.n_digits_eval = tuple(map(int, self.n_digits_eval.split(',')))
        self.n_digits_dpo = tuple(map(int, self.n_digits_dpo.split(',')))
        # assert len(self.op_eval) == len(self.op_dist_eval) == len(self.format_eval) == len(self.n_digits_eval), 'You must provide the same number of values for op_eval, op_dist_eval, format_eval, and n_digits_eval'
        self.mixture_scheduling_kwargs = json.loads(self.mixture_scheduling_kwargs) if isinstance(self.mixture_scheduling_kwargs, str) else self.mixture_scheduling_kwargs
        self.data_from_saved_train_new = tuple(self.data_from_saved_train_new.split(',')) if self.data_from_saved_train_new is not None else None