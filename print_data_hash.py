import os
from preamble import get_args, get_tokenizer, get_all_datasets, get_model, prepare_train_args, get_trainer

args, model_args, data_args, train_args = get_args()

tokenizer = get_tokenizer(model_args, data_args)

train_dataset, eval_datasets = get_all_datasets(train_args, data_args, tokenizer)
