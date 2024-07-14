import itertools
from .configs import DataArguments
from .data_formats import get_copy, get_nar, get_reverse, get_COT, get_interleave_copy

import random
import numpy as np
from datasets import IterableDataset, concatenate_datasets, Dataset
from transformers import PreTrainedTokenizer, set_seed, Seq2SeqTrainingArguments

def get_line(a, b, op=None, format=None, train=None):
    if op == '+':
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
            a = ''.join([str(random.randint(0, 9)) for _ in range(nda)])
            b = ''.join([str(random.randint(0, 9)) for _ in range(ndb)])
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
        batch_new['example_ids'] = [[i] * len(p + t) for i, (p, t) in enumerate(zip(prompt_ids, target_ids))]
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
        # attention_masks = []
        # for eids in examples['example_ids']:
        eids = examples['example_ids']
        attn_mask = (eids[:, None] == eids[None, :])[None, ...]
        attn_mask = np.tril(attn_mask, -1)
        attn_mask = (1 - attn_mask) * -10000
        # attention_masks.append(attn_mask)
        # examples['attention_mask'] = attention_masks
        examples['attention_mask'] = attn_mask
        return examples

    ds = Dataset.from_generator(
        data_generator_factory(args, tokenizer, args.num_train, train=True, n_digits_a=args.n_digits_train, sample_n_digits=True),
    num_proc=10, gen_kwargs={'shard': list(range(args.num_train))}) \
    .map(tokenization, batched=True, batch_size=1000, num_proc=16, remove_columns=['prompt', 'target']) \
    .map(group_texts, batched=True, batch_size=1000, num_proc=16).with_format('torch') \
    # .map(add_attn_masks, batched=False, num_proc=4, remove_columns=['example_ids']) # This adds the correct attention masks for packed sequences, but its extremely slow

    print('----------- Examples from train: -------------')
    for example in itertools.islice(ds, 0, 2):
        print(example['input_ids'])
        print(tokenizer.decode(example['input_ids']))
        print(example['labels'])
        print(tokenizer.decode(example['labels']))
    
    return ds

def get_eval_dataset(train_args, args: DataArguments, tokenizer: PreTrainedTokenizer):
    def tokenization(batch):
        batch_new = tokenizer(batch['prompt'], padding='longest', add_special_tokens=False)
        batch_new['labels'] = tokenizer(batch['target'], padding='longest', add_special_tokens=False)['input_ids']
        return batch_new

    ds_list = {
        n_digits: Dataset.from_generator(
            data_generator_factory(args, tokenizer, args.num_eval, train=False, n_digits_a=n_digits),
        num_proc=10, gen_kwargs={'shard': list(range(args.num_eval))}) \
        .map(tokenization, batched=True, batch_size=10000, num_proc=16).with_format('torch')
    for n_digits in range(args.n_digits_eval_start, args.n_digits_eval_end + 1, args.n_digits_eval_step)}


    print('----------- Examples from eval: -------------')
    for ds in ds_list.values():
        for example in itertools.islice(ds, 0, 2):
            print(example['input_ids'])
            print(tokenizer.decode(example['input_ids']))
            print(example['labels'])
            print(tokenizer.decode(example['labels']))

    return ds_list
