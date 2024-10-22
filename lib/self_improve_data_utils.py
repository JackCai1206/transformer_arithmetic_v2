from dataclasses import dataclass
import itertools
import os
import shutil
from typing import List, Tuple, Union, Dict

import torch
from torch.nn.utils.rnn import pad_sequence

from .configs import DataArguments, MyTrainingArguments

import random
import numpy as np
from datasets import IterableDataset, Dataset, interleave_datasets, disable_caching, enable_caching, load_dataset
from transformers import PreTrainedTokenizer, set_seed, Seq2SeqTrainingArguments
from transformers.data.data_collator import _torch_collate_batch
from trl.trainer.utils import DPODataCollatorWithPadding

import torch

def data_generator_from_model(
    answer_model,  # model to generate answers
    dataset,
    tokenizer,  # tokenizer to encode/decode (PreTrainedTokenizer)
    train: bool = True,
    shard: List[range] = None,
    n_digits: int = None,
    seed: int = None,
    device: str = 'cuda',
):
    assert len(shard) == 1, f'Shard should be a list of one range, but got: {shard}'
    
    input_ids, labels = 'input_ids', 'labels'
    
    if not train:
        seed = 1000 + seed
        input_ids = 'eval_input_ids'
        labels = 'eval_labels'

    random.seed(seed + shard[0].start)
    
    dataset_size = len(dataset)  # Get the size of the dataset
    print(f'Dataset size: {dataset_size}')

    for idx in shard[0]:  # Use idx to iterate over the range in the shard
        if idx >= dataset_size:  # Check if the index is out of bounds
            print(f'Warning: Index {idx} is out of bounds for dataset size {dataset_size}. Skipping.')
            continue
        
        data = dataset[idx]  # Access dataset using idx
            
        # Convert input_ids to a tensor
        input_ids_tensor = torch.tensor(data[input_ids], dtype=torch.long).unsqueeze(0).to(device)  # Add batch dimension
        input_length = input_ids_tensor.size(1)
        attention_mask = torch.ones_like(input_ids_tensor)
        prompt = tokenizer.decode(data[input_ids], skip_special_tokens=True)


        if train:
            prompt_question = prompt.split('=')[0].replace('C', '')
            # prompt_question_length = len(prompt_question) + 3 # [BOS], C, =
            prompt_question_length = sum(torch.Tensor(data[labels]) == -100).item()
            a, b = prompt_question.split('+')
            num_a, num_b = len(a), len(b)

            n_digits = max(num_a, num_b)
            input_ids_tensor = input_ids_tensor[:, :prompt_question_length] # cut it to prompt_question only 
            input_length = input_ids_tensor.size(1)

            data[labels] = data[labels][prompt_question_length:] # cut it to the target only
            prompt = 'C' + prompt_question + '='

        # Decode the real target from the dataset
        real_target = tokenizer.decode(data[labels], skip_special_tokens=True)
        
        # Generate target output using the answer model
        # with torch.amp.autocast('cuda'):
        target_ids = answer_model.generate(input_ids_tensor, max_new_tokens=n_digits + 1, eos_token_id=tokenizer.eos_token_id, attention_mask=attention_mask, pad_token_id=tokenizer.pad_token_id)
        target = tokenizer.decode(target_ids[0][input_length:], skip_special_tokens=True)
        
        loss_mask = [1] * len(target)

        yield {
            'prompt': prompt,
            'target': target,       # Generated target from the model
            'real_target': real_target,  # Original target from the dataset
            'loss_mask': loss_mask, 
        }



def get_dataset_display_name(n_digits, op, format):
    return f'{n_digits}-{op}-{format}'



def get_train_dataset_from_model(answer_model, train_dataset, train_args: MyTrainingArguments, args: DataArguments, tokenizer: PreTrainedTokenizer):
    def add_special_tokens(batch, add_eos=True):
        batch['prompt'] = [tokenizer.bos_token + i for i in batch['prompt']]
        if add_eos:
            batch['target'] = [i + tokenizer.eos_token for i in batch['target']]
            batch['real_target'] = [i + tokenizer.eos_token for i in batch['real_target']]
            batch['loss_mask'] = [i + [1] for i in batch['loss_mask']]
        return batch

    def mask_target(target_ids, loss_mask):
        return [t if m == 1 else -100 for t, m in zip(target_ids, loss_mask)]

    def tokenization(batch):
        batch_new = {}
        prompt_ids = tokenizer(batch['prompt'], padding='do_not_pad', add_special_tokens=False)
        target_ids = tokenizer(batch['target'], padding='do_not_pad', add_special_tokens=False)
        batch_new['input_ids'] = [p + t for p, t in zip(prompt_ids['input_ids'], target_ids['input_ids'])]
        batch_new['labels'] = [[-100] * (len(p)) + mask_target(t, m) for p, t, m in zip(prompt_ids['input_ids'], target_ids['input_ids'], batch['loss_mask'])]
        batch_new['attention_mask'] = [p + t for p, t in zip(prompt_ids['attention_mask'], target_ids['attention_mask'])]
        # batch_new['labels'] = [p + t for p, t in zip(prompt_ids, target_ids)]
        # batch_new['example_ids'] = [[i] * len(p + t) for i, (p, t) in enumerate(zip(prompt_ids, target_ids))]
        return batch_new

    ds_list = []
    if len(args.op_dist_train) > 1:
        fracs = [max(x,y) for x,y in zip(*args.op_dist_train)]
    else:
        fracs = args.op_dist_train[0] # fracs = (1.0,)
    for opi, frac in enumerate(fracs):
        ds_class = Dataset
        os.makedirs(args.train_file_from_model, exist_ok=True)
        train_file = os.path.join(args.train_file_from_model, f'{args.op_train[opi]}-{args.format_train[opi]}-{args.n_digits_train[opi][0]}_{args.n_digits_train[opi][1]}-{args.num_train[opi]}')
        
        if not args.no_seed_for_data:
            train_file += f'-{train_args.seed}'
        else:
            train_file += '-43'
        
        kwargs = {'num_proc': args.nproc}
        kwargs2 = {'keep_in_memory': False, 'cache_dir': train_file, 'split': 'train'}

        ds = ds_class.from_generator(
            data_generator_from_model,
            gen_kwargs={
                'answer_model': answer_model,
                'dataset': train_dataset,
                'tokenizer': tokenizer,
                'train': True, 
                'shard': [range(i * round((args.num_eval * frac) // args.nproc), (i + 1) * round((args.num_eval * frac) // args.nproc)) for i in range(args.nproc)],
                'seed': train_args.seed,
                # 'n_digits': n_digits,
                'device': train_args.device
            },
            **(kwargs | kwargs2)
        )
        
        # save as csv (the original version! without special tokens and tokenization)
        ds.to_csv(train_file+'.csv')
        print(f'Saved {train_file}.csv')
        
        ds = ds.map(add_special_tokens, batched=True, batch_size=1000, fn_kwargs={'add_eos': args.add_special_tokens}, **kwargs)
        remove_columns = ['prompt', 'target', 'loss_mask']
        ds = ds.map(tokenization, batched=True, batch_size=1000, remove_columns=remove_columns, **kwargs)
        if not args.use_iterable_dataset and args.load_as_iterable_dataset:
            ds = ds.to_iterable_dataset(num_shards=args.nproc)
        ds_list.append(ds)

    init_probs = [frac / sum(args.op_dist_train[0]) for frac in args.op_dist_train[0]]
    if len(args.op_dist_train) > 1:
        from multiprocessing import Array
        from ctypes import c_double
        init_probs = Array(c_double, init_probs)

    init_probs_uniform = np.full(len(args.op_dist_train[0]), 1/len(args.op_dist_train[0])) # [1.]
    is_close_to_uniform = np.allclose(init_probs, init_probs_uniform) # True
    if is_close_to_uniform:
        init_probs = None

    ds = interleave_datasets(ds_list, probabilities=init_probs, seed=train_args.seed, stopping_strategy='all_exhausted')
    # .map(group_texts, batched=True, batch_size=1000, num_proc=16)
    # print(f'Cleaned up: {ds.cleanup_cache_files()}')

    # This adds the correct attention masks for packed sequences, but its extremely slow
    # ds = ds.with_format('torch').map(add_attn_masks, batched=True, batch_size=1000, num_proc=16)

    # l = []
    print('----------- Examples from train: -------------')
    for example in itertools.islice(ds, 0, 100):
        # print(example['input_ids'])
        print(tokenizer.decode(example['input_ids']))
        # print(example['labels'])
        print(tokenizer.decode(example['labels']))
        # breakpoint()
    #     l.append(len(example['input_ids']))

    # import matplotlib.pyplot as plt
    # plt.hist(l, bins=100)
    # plt.savefig('train_hist.png')
    # breakpoint()
    
    if not args.use_train_attention_mask:
        ds = ds.remove_columns('attention_mask')
    
    return ds


def get_eval_datasets_from_model(answer_model, eval_datasets, train_args: Seq2SeqTrainingArguments, args: DataArguments, tokenizer: PreTrainedTokenizer):
    def add_special_tokens(batch, add_eos=True):
        batch['prompt'] = [tokenizer.bos_token + i for i in batch['prompt']]
        if add_eos:
            batch['target'] = [i + tokenizer.eos_token for i in batch['target']]
            batch['real_target'] = [i + tokenizer.eos_token for i in batch['real_target']]
            batch['loss_mask'] = [i + [1] for i in batch['loss_mask']]
        return batch

    def tokenization(batch):
        batch_new = tokenizer(batch['prompt'], padding='do_not_pad', add_special_tokens=False, return_token_type_ids=False)
        batch_new['labels'] = tokenizer(batch['target'], padding='do_not_pad', add_special_tokens=False)['input_ids']
        for k in batch_new.keys():
            batch_new['eval_' + k] = batch_new.pop(k)
        batch_new['eval_loss_mask'] = batch['loss_mask']
        return batch_new

    ds_list = {}
    unmapped_ds_list = {}

    for key in eval_datasets.keys():
        n_digits = int(key.split('-')[0])
        for opi, frac in enumerate(args.op_dist_eval):
            os.makedirs(args.eval_file_from_model, exist_ok=True)
            eval_file = os.path.join(args.eval_file_from_model, f'{args.op_eval[opi]}-{args.format_eval[opi]}-{n_digits}-{args.num_eval}')
            
            if not args.no_seed_for_data:
                eval_file += f'-{train_args.seed}'
            else:
                eval_file += '-43'
                
            # shards: [range(0, 12), range(12, 24), ...]

            ds0 = Dataset.from_generator(
                data_generator_from_model,
                gen_kwargs={
                    'answer_model': answer_model,
                    'dataset': eval_datasets[key],
                    'tokenizer': tokenizer,
                    'train': False, 
                    'shard': [range(i * round((args.num_eval * frac) // args.nproc), (i + 1) * round((args.num_eval * frac) // args.nproc)) for i in range(args.nproc)],
                    'seed': train_args.seed,
                    'n_digits': n_digits,
                    'device': train_args.device
                },
                num_proc=args.nproc,
                keep_in_memory=True,
                cache_dir=eval_file,
                split='test'
            )
            # for f in ds0.cache_files:
            #     shutil.rmtree(os.path.dirname(f['filename']))
            # ds0.save_to_disk(eval_file) 
            ds = ds0.map(add_special_tokens, batched=True, batch_size=1000, fn_kwargs={'add_eos': args.add_special_tokens})
            ds = ds.map(tokenization, batched=True, batch_size=args.num_eval, remove_columns=['prompt', 'target', 'loss_mask'])

            key = get_dataset_display_name(n_digits, args.op_eval[opi], args.format_eval[opi])
            ds_list[key] = ds
            unmapped_ds_list[key] = ds0
            # print(f'cleaned up {ds_list[n_digits].cleanup_cache_files()}')

            # save as csv (the original version! without special tokens and tokenization)
            ds0.to_csv(eval_file+'.csv')
            print(f'Saved {eval_file}.csv')

    print('----------- Examples from eval: -------------')
    for ds in ds_list.values():
        for example in ds.take(1):
            print(example['eval_input_ids'])
            print(tokenizer.decode(example['eval_input_ids']))
            print(example['eval_labels'])
            print(tokenizer.decode(example['eval_labels']))
    
    return ds_list, unmapped_ds_list
