# Generate self-improve data from the model (answer_model), where the target (label) is the output of the model

import os
from preamble import get_args, get_tokenizer, get_all_datasets, get_model, prepare_train_args, get_trainer, get_all_datasets_from_model
import torch

args, model_args, data_args, train_args = get_args()

tokenizer = get_tokenizer(model_args, data_args)

train_dataset, eval_datasets = get_all_datasets(train_args, data_args, tokenizer) # load from correct labeled data

'''
train_dataset[0]: {'input_ids': [], 'labels': [], 'attention_mask': []}
eval_datasets['1-add-reverse'][0]: {'eval_input_ids': [], 'eval_labels': [], 'eval_attention_mask': [], 'eval_loss_mask': []}
tokenizer.decode(train_dataset[0]['input_ids']): '[BOS]C3+58=880[EOS]'
tokenizer.decode(train_dataset[0]['labels']): '[MASK][MASK][MASK][MASK][MASK][MASK][MASK]880[EOS]'
train_dataset[0]['attention_mask']: [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1] # length: input_ids = labels
----
tokenizer.decode(eval_datasets['3-add-reverse']['eval_input_ids'][0]): '[BOS]C338+451='
tokenizer.decode(eval_datasets['3-add-reverse']['eval_labels'][0]): '7890[EOS]'
eval_datasets['3-add-reverse']['eval_attention_mask'][0] : [1, 1, 1, 1, 1, 1, 1, 1, 1, 1] # length: eval_input_ids
eval_datasets['3-add-reverse']['eval_loss_mask'][0]: [1, 1, 1, 1, 1] # length: eval_labels
'''


answer_model = get_model(train_args, model_args, tokenizer)
train_args = prepare_train_args(train_args, model_args, data_args, tokenizer)

print('*' * 50 +'\nLoading model from:', train_args.resume_from_checkpoint)
answer_model.load_state_dict(torch.load(os.path.join(train_args.resume_from_checkpoint, 'pytorch_model.bin')))
# answer_model.load_state_dict(train_args.output_dir)
answer_model = answer_model.to(torch.bfloat16).to(train_args.device)
answer_model.eval()


# now we use the train_dataset and make a new dataset with the same prompt but with the labels from the model
model_id = '/'.join(train_args.resume_from_checkpoint.split('/')[-2:]).replace('/', '_')
data_args.eval_file_from_model = f'data/eval_from_model-{model_id}'
data_args.train_file_from_model = f'data/train_from_model-{model_id}'
data_args.nproc = 1

# Get self-improve data (which is automatically saved in data/eval_from_model-, data/train_from_model-)
train_dataset_from_model, eval_datasets_from_model = get_all_datasets_from_model(answer_model, train_dataset, eval_datasets, train_args, data_args, tokenizer)

exit()

# train_file_from_model