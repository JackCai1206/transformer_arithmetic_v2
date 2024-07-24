from dataclasses import dataclass
import itertools

import torch
from .configs import DataArguments
from .data_formats import get_copy, get_nar, get_reverse, get_COT, get_interleave_copy

import random
import numpy as np
from datasets import IterableDataset, concatenate_datasets, Dataset
from transformers import PreTrainedTokenizer, set_seed, Seq2SeqTrainingArguments
from transformers.data.data_collator import _torch_collate_batch

def get_line(a, b, op=None, format=None, train=None):
    if op == 'add':
        if format == 'reverse':
            return get_reverse(a, b)
        elif format == 'COT':
            return get_COT(a, b)
    elif op == 'nar':
        return get_nar(a, n=format['n'])
    elif op == 'interleave_copy': 
        return get_interleave_copy(a, b)
    elif op == 'copy':
        return get_copy(a)
    
    raise ValueError(f'Unknown op or format: {op}, {format}')

def data_generator_factory(args: DataArguments, tokenizer: PreTrainedTokenizer, num_examples: int, train: bool = True, n_digits_a: int | None = None, n_digits_b: int | None = None, sample_n_digits: bool = False):
    if n_digits_b is None:
        n_digits_b = n_digits_a
    def generate_data(shard):
        for _ in shard:
            nda = random.randint(args.n_digits_train_min, n_digits_a) if sample_n_digits else n_digits_a
            ndb = random.randint(args.n_digits_train_min, n_digits_b) if sample_n_digits else n_digits_b
            a = str(random.randint(1, 9)) + ''.join([str(random.randint(0, 9)) for _ in range(nda - 1)])
            b = str(random.randint(1, 9)) + ''.join([str(random.randint(0, 9)) for _ in range(ndb - 1)])
            prompt, target = get_line(a, b, op=args.op, format=args.format, train=train)
            prompt = f"{tokenizer.bos_token}{prompt}"
            target = f"{target}{tokenizer.eos_token}"
            yield {
                'prompt': prompt,
                'target': target
            }

    return generate_data

def get_train_dataset(train_args: Seq2SeqTrainingArguments, args: DataArguments, tokenizer: PreTrainedTokenizer):
    def tokenization(batch):
        batch_new = {}
        prompt_ids = tokenizer(batch['prompt'], padding='do_not_pad', add_special_tokens=False)['input_ids']
        target_ids = tokenizer(batch['target'], padding='do_not_pad', add_special_tokens=False)['input_ids']
        batch_new['input_ids'] = [p + t for p, t in zip(prompt_ids, target_ids)]
        batch_new['labels'] = [[-100] * (len(p)) + t for p, t in zip(prompt_ids, target_ids)]
        # batch_new['labels'] = [p + t for p, t in zip(prompt_ids, target_ids)]
        # batch_new['example_ids'] = [[i] * len(p + t) for i, (p, t) in enumerate(zip(prompt_ids, target_ids))]
        return batch_new

    def group_texts(examples):
        # Concatenate all texts.
        concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
        total_length = len(concatenated_examples[list(examples.keys())[0]])
        # We drop the small remainder, we could add padding if the model supported it instead of this drop, you can
        # customize this part to your needs.
        total_length = (total_length // args.block_size) * args.block_size
        # Split by chunks of max_len.
        result = {
            k: [t[i : i + args.block_size] for i in range(0, total_length, args.block_size)]
            for k, t in concatenated_examples.items()
        }
        # result['labels'] = result['input_ids'].copy()

        return result

    def add_attn_masks(examples):
        eids = examples['example_ids']
        attn_mask = (eids[:, :, None] == eids[:, None, :])
        attn_mask = torch.tril(attn_mask, -1)
        examples['attention_mask'] = attn_mask[:, None, ...]

        # Does not support batch processing
        # inputs = examples['input_ids']
        # seq_len = len(inputs)
        # elens = torch.arange(seq_len)[inputs == tokenizer.bos_token_id] + 1
        # attn_mask = torch.zeros(seq_len, seq_len, dtype=bool)
        # for i in range(len(elens)):
        #     if i == 0:
        #         attn_mask[0:elens[i], 0:elens[i]] = 1
        #     else:
        #         attn_mask[elens[i-1]:elens[i], elens[i-1]:elens[i]] = 1
        # if elens[-1] != seq_len:
        #     attn_mask[elens[i-1]:seq_len, elens[i-1]:seq_len] = 1
        # attn_mask = torch.tril(attn_mask, -1)
        # attn_mask = (~attn_mask).float() * torch.finfo(torch.float32).min # not compatible with bf16
        # examples['attention_mask'] = attn_mask[None, ... ]
        return examples

    ds = Dataset.from_generator(
        data_generator_factory(args, tokenizer, args.num_train, train=True, n_digits_a=args.n_digits_train, sample_n_digits=True),
    num_proc=10, gen_kwargs={'shard': list(range(args.num_train))}) \
    .map(tokenization, batched=True, batch_size=1000, num_proc=16, remove_columns=['prompt', 'target']) \
    # .map(group_texts, batched=True, batch_size=1000, num_proc=16)
    # print(f'Cleaned up: {ds.cleanup_cache_files()}')

    # This adds the correct attention masks for packed sequences, but its extremely slow
    # ds = ds.with_format('torch').map(add_attn_masks, batched=True, batch_size=1000, num_proc=16)

    print('----------- Examples from train: -------------')
    for example in itertools.islice(ds, 0, 2):
        print(example['input_ids'])
        print(tokenizer.decode(example['input_ids']))
        print(example['labels'])
        print(tokenizer.decode(example['labels']))
    
    return ds

def get_eval_dataset(train_args: Seq2SeqTrainingArguments, args: DataArguments, tokenizer: PreTrainedTokenizer):
    def tokenization(batch):
        tokenizer.padding_side = 'left'
        batch_new = tokenizer(batch['prompt'], padding='longest', add_special_tokens=False)
        tokenizer.padding_side = 'right'
        batch_new['labels'] = tokenizer(batch['target'], padding='longest', add_special_tokens=False)['input_ids']
        return batch_new

    ds_list = {}
    for n_digits in range(args.n_digits_eval_start, args.n_digits_eval_end + 1, args.n_digits_eval_step):
        ds_list[str(n_digits)] = Dataset.from_generator(
            data_generator_factory(args, tokenizer, args.num_eval, train=False, n_digits_a=n_digits),
        num_proc=10, gen_kwargs={'shard': list(range(args.num_eval))}) \
        .map(tokenization, batched=True, batch_size=args.num_eval, num_proc=16, remove_columns=['prompt', 'target']) \
        .remove_columns(['token_type_ids'])
        # print(f'cleaned up {ds_list[n_digits].cleanup_cache_files()}')


    print('----------- Examples from eval: -------------')
    for ds in ds_list.values():
        for example in itertools.islice(ds, 0, 2):
            print(example['input_ids'])
            print(tokenizer.decode(example['input_ids']))
            print(example['labels'])
            print(tokenizer.decode(example['labels']))

    return ds_list

# @dataclass
# class PromptAnswerDataCollator:
#     tokenizer: PreTrainedTokenizer
    
#     def __call__(self, features):
#         # convert to dict of lists and pop the labels
#         features = {key: [example[key] for example in features] for key in features[0].keys()}
#         labels = features.pop('labels', None)
#         # attention_mask = features.pop('attention_mask', None)

#         # left-pad the inputs
#         prev_padding_side = self.tokenizer.padding_side
#         self.tokenizer.padding_side = 'left'
#         batch = self.tokenizer.pad(features, padding='longest', return_tensors='pt')

#         if train_labels is not None:
#             batch['labels'] = _torch_collate_batch(train_labels, self.tokenizer)

#         if eval_labels is not None:
#             # For evaluation, manually right-pad the labels (everything else is left-padded)
#             self.tokenizer.padding_side = 'right'
#             batch['labels'] = _torch_collate_batch(eval_labels, self.tokenizer)
#         self.tokenizer.padding_side = prev_padding_side

#         # add in 4D attention mask, workaround for https://github.com/huggingface/transformers/issues/32101
#         # if attention_mask is not None:
#         #     if not isinstance(attention_mask[0], torch.Tensor): # list of lists
#         #         batch['attention_mask'] = torch.tensor(attention_mask, dtype=torch.long)
#         #     else:
#         #         batch['attention_mask'] = torch.stack(attention_mask, dim=0)
#         #         # batch['attention_mask'] = (~batch['attention_mask']).float() * torch.finfo(torch.float32).min # not compatible with bf16

#         return batch
