from dataclasses import dataclass
import itertools
from typing import List, Tuple, Union

import torch
from torch.nn.utils.rnn import pad_sequence
from .configs import DataArguments
from .data_formats import get_3parity, get_3sum, get_add1, get_copy, get_cumsum, get_cumsum_gt5, get_forward, get_forward_carry_only, get_forward_no_carry, get_gt5, get_itcopy_rev, get_minimum, get_mult, get_nar, get_parity, get_reverse, get_COT, get_interleave_copy, get_reverse_2op, get_reverse_add, get_reverse_add_automata, get_reverse_add_cont, get_reverse_carry_only, get_reverse_no_carry, get_rot1rev, get_rotate1, get_sd_mult, get_set_diff, get_sort

import random
import numpy as np
from datasets import IterableDataset, Dataset, interleave_datasets
from transformers import PreTrainedTokenizer, set_seed, Seq2SeqTrainingArguments
from transformers.data.data_collator import _torch_collate_batch
from trl.trainer.utils import DPODataCollatorWithPadding

def get_line(a, b, op=None, format=None, train=None):
    if op == 'add':
        if format == 'reverse':
            return get_reverse_add(a, b)
        elif format == 'COT':
            return get_COT(a, b)
        elif format == 'reverse-no-carry':
            return get_reverse_no_carry(a, b)
        elif format == 'reverse-no-carry-random':
            return get_reverse_no_carry(a, b, randomize=True)
        elif format == 'reverse-carry-only':
            return get_reverse_carry_only(a, b)
        elif format == 'reverse-carry-only-random':
            return get_reverse_carry_only(a, b, randomize=True)
        elif format == 'forward':
            return get_forward(a, b)
        elif format == 'forward-no-carry':
            return get_forward_no_carry(a, b)
        elif format == 'forward-carry-only':
            return get_forward_carry_only(a, b)
        elif format == 'cont-tokens':
            return get_reverse_add_cont(a, b)
        elif format == 'automata_A':
            return get_reverse_add_automata(a, b, type='A')
        elif format == 'automata_B':
            return get_reverse_add_automata(a, b, type='B')
        elif format == 'automata_C':
            return get_reverse_add_automata(a, b, type='C')
        elif format == 'add1':
            return get_add1(a)
    elif op == 'sort':
        if format == 'sort':
            return get_sort(a)
        elif format == 'min':
            return get_minimum(a)
        elif format == 'set_diff':
            return get_set_diff(a)
        elif format == 'sort_rev':
            return get_sort(a, reverse=True)
    elif op == 'nar':
        return get_nar(a, n=format['n'])
    elif op == 'copy':
        if format == 'interleave_copy': 
            return get_interleave_copy(a, b)
        elif format == 'reverse_2op':
            return get_reverse_2op(a, b)
        elif format == 'itcopy_rev':
            return get_itcopy_rev(a, b)
        elif format == 'copy':
            return get_copy(a)
    elif op == 'rotate1':
        return get_rotate1(a)
    elif op == 'reverse':
        return get_reverse(a)
    elif op == 'rot1rev':
        return get_rot1rev(a)
    elif op == 'mult':
        if format == 'sd_mult':
            return get_sd_mult(a, b)
        elif format == 'mult':
            return get_mult(a, b)
    elif op == 'cumsum':
        return get_cumsum(a)
    elif op == 'gt5':
        return get_gt5(a)
    elif op == 'cumsum_gt5':
        return get_cumsum_gt5(a)
    elif op == 'boolean':
        if format == '3sum':
            return get_3sum(a)
        elif format == 'parity':
            return get_parity(a)
        elif format == '3parity':
            return get_3parity(a)

    raise ValueError(f'Unknown op or format: {op}, {format}')

def data_generator(
    op: str,
    format: str,
    show_task_ids: bool, 
    n_digits_a_range: Tuple[int] = None,
    n_digits_b_range: Union[Tuple[int], None] = None, # Use Union for Python < 3.10 Tuple[int] | None = None,
    train: bool = True,
    shard: List[int] = None,
):
    if n_digits_b_range is None:
        n_digits_b_range = n_digits_a_range

    assert len(shard) == 1
    for _ in shard[0]:
        if op == 'sort':
            # Generate a random list of digits
            nda = random.sample(range(*n_digits_a_range), 1)[0]
            a = random.sample(range(100), nda)
            prompt, target, loss_mask = get_line(a, None, op=op, format=format, train=train)
        elif op == 'boolean':
            nda = random.sample(range(*n_digits_a_range), 1)[0]
            a = [random.randint(0, 1) for _ in range(nda)]
            prompt, target, loss_mask = get_line(a, None, op=op, format=format, train=train)
        else:
            nda = random.sample(range(*n_digits_a_range), 1)[0]
            if op == 'mult':
                ndb = 2 # always 3 digits for b
            elif op == 'add' and 'automata' in format:
                ndb = nda
            else:
                ndb = random.sample(range(*n_digits_a_range), 1)[0]
            a = str(random.randint(1, 9)) + ''.join([str(random.randint(0, 9)) for _ in range(nda - 1)])
            b = str(random.randint(1, 9)) + ''.join([str(random.randint(0, 9)) for _ in range(ndb - 1)])
            prompt, target, loss_mask = get_line(a, b, op=op, format=format, train=train)
            if not show_task_ids: 
                prompt = prompt[:-2] + prompt[-1]

        if loss_mask is None:
            loss_mask = [1] * len(target)

        yield {
            'prompt': prompt,
            'target': target,
            'loss_mask': loss_mask,
        }



def get_train_dataset(train_args: Seq2SeqTrainingArguments, args: DataArguments, tokenizer: PreTrainedTokenizer):
    def add_special_tokens(batch, add_eos=True):
        batch['prompt'] = [tokenizer.bos_token + i for i in batch['prompt']]
        if add_eos:
            batch['target'] = [i + tokenizer.eos_token for i in batch['target']]
            batch['loss_mask'] = [i + [1] for i in batch['loss_mask']]
        return batch

    def mask_target(target_ids, loss_mask):
        return [t if m == 1 else -100 for t, m in zip(target_ids, loss_mask)]

    def tokenization(batch):
        batch_new = {}
        prompt_ids = tokenizer(batch['prompt'], padding='do_not_pad', add_special_tokens=False)['input_ids']
        target_ids = tokenizer(batch['target'], padding='do_not_pad', add_special_tokens=False)['input_ids']
        batch_new['input_ids'] = [p + t for p, t in zip(prompt_ids, target_ids)]
        batch_new['labels'] = [[-100] * (len(p)) + mask_target(t, m) for p, t, m in zip(prompt_ids, target_ids, batch['loss_mask'])]
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

    ds_list = []
    for opi, frac in enumerate(args.op_dist_train):
        ds = IterableDataset.from_generator(
            data_generator,
            gen_kwargs={
                'train': True,
                'op': args.op_train[opi],
                'format': args.format_train[opi],
                'n_digits_a_range': args.n_digits_train[opi],
                'shard': [range(i * round((args.num_train * frac) // args.nproc), (i + 1) * round((args.num_train * frac) // args.nproc)) for i in range(args.nproc)],
                'show_task_ids': args.show_task_ids
            },
        )
        ds = ds.map(add_special_tokens, batched=True, batch_size=1000, fn_kwargs={'add_eos': args.add_special_tokens})
        ds = ds.map(tokenization, batched=True, batch_size=1000, remove_columns=['prompt', 'target', 'loss_mask'])
        ds_list.append(ds)

    op_dist_train = [frac / sum(args.op_dist_train) for frac in args.op_dist_train]
    ds = interleave_datasets(ds_list, probabilities=op_dist_train).shuffle(train_args.seed)
    # .map(group_texts, batched=True, batch_size=1000, num_proc=16)
    # print(f'Cleaned up: {ds.cleanup_cache_files()}')

    # This adds the correct attention masks for packed sequences, but its extremely slow
    # ds = ds.with_format('torch').map(add_attn_masks, batched=True, batch_size=1000, num_proc=16)

    # l = []
    print('----------- Examples from train: -------------')
    for example in itertools.islice(ds, 0, 1):
        print(example['input_ids'])
        print(tokenizer.decode(example['input_ids']))
        print(example['labels'])
        print(tokenizer.decode(example['labels']))
    #     l.append(len(example['input_ids']))

    # import matplotlib.pyplot as plt
    # plt.hist(l, bins=100)
    # plt.savefig('train_hist.png')
    # breakpoint()
    
    return ds

def get_eval_dataset(train_args: Seq2SeqTrainingArguments, args: DataArguments, tokenizer: PreTrainedTokenizer):
    def add_special_tokens(batch, add_eos=True):
        batch['prompt'] = [tokenizer.bos_token + i for i in batch['prompt']]
        if add_eos:
            batch['target'] = [i + tokenizer.eos_token for i in batch['target']]
            batch['loss_mask'] = [i + [1] for i in batch['loss_mask']]
        return batch

    def tokenization(batch):
        batch_new = tokenizer(batch['prompt'], padding='do_not_pad', add_special_tokens=False, return_token_type_ids=False)
        batch_new['labels'] = tokenizer(batch['target'], padding='do_not_pad', add_special_tokens=False)['input_ids']
        for k in batch_new.keys():
            batch_new['eval_' + k] = batch_new.pop(k)
        return batch_new

    ds_list = {}
    for n_digits in range(*args.n_digits_eval):
        for opi, frac in enumerate(args.op_dist_eval):
            ds = IterableDataset.from_generator(
                data_generator,
                gen_kwargs={
                    'train': False, 
                    'op': args.op_eval[opi],
                    'format': args.format_eval[opi],
                    'n_digits_a_range': (n_digits, n_digits + 1),
                    'shard': [range(round(args.num_eval * frac))],
                    'show_task_ids': args.show_task_ids
                },
            )
            ds = ds.map(add_special_tokens, batched=True, batch_size=1000, fn_kwargs={'add_eos': args.add_special_tokens})
            ds = ds.map(tokenization, batched=True, batch_size=args.num_eval, remove_columns=['prompt', 'target'])

            key = f'{n_digits}-{args.op_eval[opi]}-{args.format_eval[opi]}'
            ds_list[key] = ds            
            # print(f'cleaned up {ds_list[n_digits].cleanup_cache_files()}')

    print('----------- Examples from eval: -------------')
    for ds in ds_list.values():
        for example in itertools.islice(ds, 0, 2):
            print(example['eval_input_ids'])
            print(tokenizer.decode(example['eval_input_ids']))
            print(example['eval_labels'])
            print(tokenizer.decode(example['eval_labels']))

    return ds_list

def get_dpo_dataset(args: DataArguments, tokenizer: PreTrainedTokenizer):
    def rand_trunc(l):
        # return l[:random.randint(0, len(l)-1)]
        return ''.join([str(random.randint(0, 9)) for _ in range(random.randint(0, len(l)-1))])
    
    def get_dpo_format(batch):
        batch['prompt'] = [tokenizer.bos_token + i for i in batch['prompt']]
        batch['chosen'] = [i + tokenizer.eos_token for i in batch['target']]
        batch['rejected'] = [rand_trunc(i) + tokenizer.eos_token for i in batch['target']]
        # batch['rejected'] = [tokenizer.eos_token for i in batch['target']]
        return batch
    
    ds = IterableDataset.from_generator(
        data_generator,
        gen_kwargs={
            'train': True,
            'op': args.op_train[0],
            'format': args.format_train[0],
            'n_digits_a_range': args.n_digits_train[0],
            'shard': [range(i * round((args.num_train * 1) // args.nproc), (i + 1) * round((args.num_train * 1) // args.nproc)) for i in range(args.nproc)],
            'show_task_ids': args.show_task_ids
        },
    ) \
    .map(get_dpo_format, batched=True, batch_size=1000, remove_columns=['target'])

    return ds

class PromptAnswerDataCollator(DPODataCollatorWithPadding):
    # Right pad during training, left pad during eval
    # because during training the position ids starts at the left-most token
    # The other option is to left pad everything and manually cauculate training position ids, but this should be simpler
    left_pad_list: tuple = ('prompt', 'eval_input_ids', 'eval_labels', 'eval_attention_mask')

    def __call__(self, features):
        # convert to dict of lists and pop the labels
        features = {
            key: [example[key] for example in features] for key in features[0].keys()
        }
        padded_batch = {}
        for k, feat in features.items():
            if k in self.left_pad_list:
                to_pad = [torch.LongTensor(ex[::-1]) for ex in feat]
            else:
                to_pad = [torch.LongTensor(ex) for ex in feat]
            
            if k.endswith("input_ids") or k.endswith('eval_labels'):
                if self.pad_token_id is None:
                    raise ValueError(
                        "Padding is enabled, but the tokenizer is not configured with a padding token."
                        " Explicitly set `tokenizer.pad_token` (e.g. `tokenizer.pad_token = tokenizer.eos_token`)"
                        " before calling the trainer."
                    )
                padding_value = self.pad_token_id
            elif k.endswith("labels"):
                padding_value = self.label_pad_token_id
            elif k.endswith("attention_mask"):
                padding_value = 0
            elif k.endswith('loss_mask'):
                padding_value = 0
            else:
                raise ValueError(f"Unexpected key in batch '{k}'")
            
            # remove the eval_ prefix to conform to model input names
            if 'eval_' in k:
                input_k = k.replace('eval_', '')
            else:
                input_k = k

            padded_batch[input_k] = pad_sequence(to_pad, batch_first=True, padding_value=padding_value)

            if k in self.left_pad_list:
                padded_batch[input_k] = padded_batch[input_k].flip(dims=[1])

        # add in 4D attention mask, workaround for https://github.com/huggingface/transformers/issues/32101
        # if attention_mask is not None:
        #     if not isinstance(attention_mask[0], torch.Tensor): # list of lists
        #         batch['attention_mask'] = torch.tensor(attention_mask, dtype=torch.long)
        #     else:
        #         batch['attention_mask'] = torch.stack(attention_mask, dim=0)
        #         # batch['attention_mask'] = (~batch['attention_mask']).float() * torch.finfo(torch.float32).min # not compatible with bf16
        return padded_batch
