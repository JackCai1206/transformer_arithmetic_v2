import itertools
from .configs import DataArguments
from .data_formats import get_nar, get_reverse, get_COT

import random
from datasets import IterableDataset, concatenate_datasets
from transformers import PreTrainedTokenizer

def get_line(a, b, op=None, format=None, train=None):
    if op == '+':
        if format == 'reverse':
            return get_reverse(a, b)
        elif format == 'COT':
            return get_COT(a, b)
    elif op == 'nar':
        return get_nar(a, n=format['n'])
    
    raise ValueError(f'Unknown op or format: {op}, {format}')

def data_generator_factory(num_examples, op, format, tokenizer: PreTrainedTokenizer, train: bool = True, n_digits_a: int = None, n_digits_b: int = None, sample_n_digits: bool = False, n_digits_train_min: int = 1):
    if n_digits_b is None:
        n_digits_b = n_digits_a
    def generate_data():
        for _ in range(num_examples):
            nda = random.randint(n_digits_train_min, n_digits_a) if sample_n_digits else n_digits_a
            ndb = random.randint(n_digits_train_min, n_digits_b) if sample_n_digits else n_digits_b
            a = ''.join([str(random.randint(0, 9)) for _ in range(nda)])
            b = ''.join([str(random.randint(0, 9)) for _ in range(ndb)])
            prompt, target = get_line(a, b, op=op, format=format, train=train)
            prompt = f"{tokenizer.bos_token}{prompt}"
            target = f"{target}{tokenizer.eos_token}"
            yield {
                'prompt': prompt,
                'target': target
            }

    return generate_data

def get_train_dataset(args: DataArguments, tokenizer: PreTrainedTokenizer):
    def tokenization(batch):
        batch_new = {}
        prompt_ids = tokenizer(batch['prompt'], padding='do_not_pad', add_special_tokens=False)['input_ids']
        target_ids = tokenizer(batch['target'], padding='do_not_pad', add_special_tokens=False)['input_ids']
        batch_new['input_ids'] = [p + t for p, t in zip(prompt_ids, target_ids)]
        batch_new['labels'] = [[-100] * (len(p) - 1) + p[-1:] + t for p, t in zip(prompt_ids, target_ids)]
        # for p, l in zip(batch_new['input_ids'], batch_new['labels']):
        #     assert len(p) == len(l), breakpoint()
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

    ds = IterableDataset.from_generator(
        data_generator_factory(args.num_train, args.op, args.format, tokenizer, train=True, n_digits_a=args.n_digits_train, sample_n_digits=True, n_digits_train_min=args.n_digits_train_min),
    ) \
    .map(tokenization, batched=True, batch_size=1000, remove_columns=['prompt', 'target']) \
    .map(group_texts, batched=True, batch_size=1000).with_format('torch')

    print('----------- Examples from train: -------------')
    for example in itertools.islice(ds, 0, 2):
        print(example['input_ids'])
        print(tokenizer.decode(example['input_ids']))
        print(example['labels'])
        print(tokenizer.decode(example['labels']))
    
    return ds

def get_eval_dataset(args: DataArguments, tokenizer: PreTrainedTokenizer):
    def tokenization(batch):
        batch_new = tokenizer(batch['prompt'], padding='longest', add_special_tokens=False)
        batch_new['labels'] = tokenizer(batch['target'], padding='longest', add_special_tokens=False)['input_ids']
        return batch_new

    ds_list = {
        n_digits: IterableDataset.from_generator(
            data_generator_factory(args.num_eval, args.op, args.format, tokenizer, train=False, n_digits_a=n_digits),
        ) \
        .map(tokenization, batched=True, batch_size=1000).with_format('torch')
    for n_digits in range(args.n_digits_eval_start, args.n_digits_eval_end + 1, args.n_digits_eval_step)}


    print('----------- Examples from eval: -------------')
    for ds in ds_list.values():
        for example in itertools.islice(ds, 0, 2):
            print(example['input_ids'])
            print(tokenizer.decode(example['input_ids']))
            print(example['labels'])
            print(tokenizer.decode(example['labels']))

    return ds_list
