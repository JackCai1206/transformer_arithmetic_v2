import os
from preamble import get_args, get_tokenizer, get_all_datasets, get_model, prepare_train_args, get_trainer
import torch


from lib.configs import ModelArguments, DataArguments
from transformers import HfArgumentParser, set_seed, Seq2SeqTrainingArguments
from typing import cast, Optional, Tuple, Union
from dataclasses import dataclass

@dataclass
class ScriptArguments:
    foo: str = 'bar'
    result_name: str = 'result'


@dataclass
class MyTrainingArguments(Seq2SeqTrainingArguments):
    do_backtrack_decoding: bool = False # Automatically adds backtrack tokens during generation if the model generates the wrong token
    do_backtrack_decoding2: bool = False # Automatically adds backtrack tokens during generation if the model generates the wrong token
    do_backtrack_eval: bool = False # erases backtrack tokens during evaluation
    early_stopping: Optional[bool] = False # Stop training when the model reaches a certain metric
    do_beam_search: Optional[bool] = False # Use beam search during generation
    num_beams: Optional[int] = 1 # Number of beams for beam search
    # num_return_sequences: Optional[int] = 5 # Number of sequences to return for each input
    log_beta: Optional[bool] = False # Log beta values to wandb
    track_num_tokens_seen_by_task: bool = False
    early_stop: bool = False


def get_args():
    args, model_args, data_args, train_args = HfArgumentParser((ScriptArguments, ModelArguments, DataArguments, MyTrainingArguments)).parse_args_into_dataclasses()
    args = cast(ScriptArguments, args)
    model_args = cast(ModelArguments, model_args)
    train_args = cast(MyTrainingArguments, train_args)
    data_args = cast(DataArguments, data_args)
    data_args.block_size = model_args.max_position_embeddings
    if model_args.rope_theta == 'Inf':
        model_args.rope_theta = torch.inf

    set_seed(train_args.seed)

    return args, model_args, data_args, train_args

# turn off wandb
os.environ["WANDB_MODE"] = "disabled"

args, model_args, data_args, train_args = get_args()

tokenizer = get_tokenizer(model_args, data_args)

train_dataset, eval_datasets = get_all_datasets(train_args, data_args, tokenizer)

model = get_model(train_args, model_args, tokenizer)

train_args.do_train = True # making it false will automatically create -eval directory which is not desirable
train_args = prepare_train_args(train_args, model_args, data_args, tokenizer)

trainer = get_trainer(args, data_args, model_args, model, tokenizer, train_args, train_dataset, eval_datasets)

trainer._load_from_checkpoint(resume_from_checkpoint=train_args.resume_from_checkpoint)

result = trainer.evaluate()

# save results
import pandas as pd
import numpy as np

# model_name = train_args.resume_from_checkpoint.split('/')[1]
dir_name = '/'.join(train_args.resume_from_checkpoint.split('/')[:-1])

result_dict = {}

for i in range(data_args.n_digits_eval[0], data_args.n_digits_eval[1]):
    result_dict[f'{i}_digit_acc'] = result[f"eval_{i}-add-reverse_accuracy"]
    result_dict[f'{i}_digit_dist'] = result[f"eval_{i}-add-reverse_distance"]

df = pd.DataFrame(result_dict, index=[0])
df.to_csv(f'{dir_name}/{args.result_name}.csv')

print(f"Results saved to {dir_name}/{args.result_name}.csv")
