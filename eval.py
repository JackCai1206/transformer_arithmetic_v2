import os
from preamble import get_args, get_tokenizer, get_all_datasets, get_model, prepare_train_args, get_trainer
import torch


from lib.configs import ModelArguments, DataArguments, MyTrainingArguments, ScriptArguments
from transformers import HfArgumentParser, set_seed, Seq2SeqTrainingArguments
from typing import cast, Optional, Tuple, Union
from dataclasses import dataclass


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

model = get_model(train_args, model_args, tokenizer)

train_args.do_train = True # making it false will automatically create -eval directory which is not desirable
train_args = prepare_train_args(train_args, model_args, data_args, tokenizer)
train_args.do_train = False # reset to original value
train_args.should_save = False

train_dataset, eval_datasets = get_all_datasets(train_args, data_args, tokenizer)

if train_args.do_backtrack_decoding2 and train_args.backtrack_decoding_multiplier >= 10:
    args.eval_more = True

trainer = get_trainer(args, data_args, model_args, model, tokenizer, train_args, train_dataset, eval_datasets)

if train_args.do_dpo:
    train_args.save_safetensors = True

trainer._load_from_checkpoint(resume_from_checkpoint=train_args.resume_from_checkpoint)

dir_name = '/'.join(train_args.resume_from_checkpoint.split('/')[:-1])
# check if the directory exists (assert)
assert os.path.exists(dir_name)

result = trainer.evaluate()

print(result)

# save results
import pandas as pd
import numpy as np


result_dict = {}

for i in range(data_args.n_digits_eval[0], data_args.n_digits_eval[1], data_args.n_digits_eval[2]):
    result_dict[f'{i}_digit_acc'] = result[f"eval_{i}-add-reverse_accuracy"]
    result_dict[f'{i}_digit_dist'] = result[f"eval_{i}-add-reverse_distance"]

    if args.eval_more:
        result_dict[f'{i}_digit_avg_first_wrong_loc'] = result[f"eval_{i}-add-reverse_avg_first_wrong_loc"]
        result_dict[f'{i}_digit_avg_backtrack_count'] = result[f"eval_{i}-add-reverse_avg_backtrack_count"]
        result_dict[f'{i}_digit_avg_first_backtrack_loc'] = result[f"eval_{i}-add-reverse_avg_first_backtrack_loc"]
    
    if train_args.get_real_label:
        result_dict[f'{i}_digit_real_acc'] = result[f"eval_{i}-add-reverse_real_accuracy"]

df = pd.DataFrame(result_dict, index=[0])

if args.eval_more:
    args.result_name += '_new' # to differentiate between old and new eval results

df.to_csv(f'{dir_name}/{args.result_name}.csv')

print(f"Results saved to {dir_name}/{args.result_name}.csv")
