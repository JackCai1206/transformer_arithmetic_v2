import os
import re
import string
from functools import partial

import torch
from lib.configs import ScriptArguments, ModelArguments, DataArguments
from lib.data_utils import get_train_dataset, get_eval_dataset, PromptAnswerDataCollator
from lib.eval_utils import compute_metrics
from lib.modeling.add_rule_embedding import LlamaConfigWithAddRules, LlamaModelWithAddRules
from lib.modeling.llama import LlamaForCausalLMWithNoPE, MyLlamaConfig
from lib.modeling.llama_diff_attn import LlamaDiffAttnConfig, LlamaForCausalLMDiffAttn
from lib.modeling.llama_rand_pos_id import LlamaRandPosId
from lib.trainer_utils import AddWandbConfigCallback, EarlyStoppingCallback, Seq2SeqTrainerNoEvalLoss, MyTrainingArguments
from lib.modeling.cat import ConvLlamaForCausalLM
from lib.modeling.abacus import AbacusLlamaForCausalLM, AbacusLlamaModel, AbacusLlamaConfig
from charactertokenizer import CharacterTokenizer

from typing import cast
from transformers import HfArgumentParser, set_seed, GenerationConfig, AutoModelForCausalLM, AutoConfig, PreTrainedModel, LlamaConfig, LlamaForCausalLM, AutoTokenizer, DataCollatorForSeq2Seq, PreTrainedTokenizer
from transformers.trainer_utils import get_last_checkpoint
from datasets import Dataset
from peft import LoraConfig, get_peft_model

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

def get_tokenizer(model_args: ModelArguments, data_args: DataArguments):
    # We don't pad when generating the datasets
    # During eval, the inputs are padded on the left and the labels are padding on the right using custom data collator
    all_chars = string.ascii_letters + string.digits + string.punctuation + ' ' + '\n'
    tokenizer = CharacterTokenizer(all_chars, model_args.max_position_embeddings)
    tokenizer.padding_side == 'left'
    tokenizer.backtrack_token_id = tokenizer.convert_tokens_to_ids(['X'])[0] # use "X" as the backtrack token

    if any([x == 'sort' for x in data_args.op_train]):
        added_tokens = tokenizer.add_tokens([f'[{str(i).zfill(2)}]' for i in range(100)])

    return tokenizer

def get_all_datasets(train_args: MyTrainingArguments, data_args: DataArguments, tokenizer: PreTrainedTokenizer):
    train_dataset, eval_datasets = None, None
    if train_args.do_eval:
        eval_datasets, unmapped_eval_datasets = get_eval_dataset(train_args, data_args, tokenizer)
    if train_args.do_train:
        train_dataset = get_train_dataset(train_args, data_args, tokenizer, no_sample_from=unmapped_eval_datasets)
    tokenizer.padding_side = 'left' # in case it was changed by the data generator
    return train_dataset, eval_datasets

def get_model(train_args: MyTrainingArguments, model_args: ModelArguments, tokenizer: PreTrainedTokenizer):
    if model_args.model_id is not None:
        model: PreTrainedModel
        if model_args.from_pretrained:
            model = AutoModelForCausalLM.from_pretrained(model_args.model_id)
        else:
            model_config = AutoConfig.from_pretrained(model_args.model_id)
            model = AutoModelForCausalLM.from_config(model_config)

        old_tokenizer = AutoTokenizer.from_pretrained(model_args.model_id)
        emb = model.get_input_embeddings()
        assert isinstance(emb, torch.nn.Embedding), "Only nn.Embedding is supported for now"
        all_vocab = [k for k,v in tokenizer.get_vocab().items() if v >= 0] # filter out the -100 token, which will never appear in the input
        old_ids = old_tokenizer.convert_tokens_to_ids(all_vocab)
        new_ids = tokenizer.convert_tokens_to_ids(all_vocab)
        new_emb = torch.nn.Embedding(len(old_ids), emb.embedding_dim)
        new_emb.weight.data[torch.tensor(new_ids)] = emb(torch.tensor(old_ids))
        model.set_input_embeddings(emb)
        model.resize_token_embeddings(len(tokenizer))

        if model_args.freeze_except is not None:
            for p in model.parameters():
                p.requires_grad = False
        for name, module in model.named_modules():
            if model_args.freeze is not None:
                if re.search(model_args.freeze, name) is not None:
                    for p in module.parameters():
                        p.requires_grad = False
                    print(f"Freezing {name}")
            elif model_args.freeze_except is not None:
                if re.search(model_args.freeze_except, name) is not None:
                    for p in module.parameters():
                        p.requires_grad = True
                    print(f"Freezing except {name}")

        model_args.architecture = model.config.architectures[0]
    else:
        if model_args.architecture == "mamba":
            from transformers import MambaConfig, MambaForCausalLM
            model_config = MambaConfig(
                vocab_size=tokenizer.vocab_size,
                hidden_size=model_args.hidden_size,
                state_size=model_args.state_size,
                num_hidden_layers=model_args.num_layers
            )
            model = MambaForCausalLM(model_config)
        elif model_args.architecture == "CAT":
            model_config = LlamaConfig(
                vocab_size=tokenizer.vocab_size,
                hidden_size=model_args.hidden_size,
                intermediate_size=model_args.intermediate_size,
                num_attention_heads=model_args.num_attention_heads,
                num_hidden_layers=model_args.num_layers,
                max_position_embeddings=model_args.max_position_embeddings,
                _attn_implementation='sdpa'
            )
            model = ConvLlamaForCausalLM(model_config)
        elif model_args.architecture == "llama-diff":
            model_config = LlamaDiffAttnConfig(
                vocab_size=tokenizer.vocab_size,
                hidden_size=model_args.hidden_size,
                intermediate_size=model_args.intermediate_size,
                num_attention_heads=model_args.num_attention_heads,
                num_hidden_layers=model_args.num_layers,
                max_position_embeddings=model_args.max_position_embeddings,
                _attn_implementation='flash_attention_2' if train_args.bf16 else 'sdpa',
                # rope_theta=torch.inf
                rope_theta=model_args.rope_theta,
                partial_rotary_factor=model_args.partial_rotary_factor,
                use_rpe=model_args.architecture == 'llama-rpe'
            )
            
            model = LlamaForCausalLMDiffAttn(model_config)
        elif model_args.architecture.startswith("llama"):
            model_config = MyLlamaConfig(
                vocab_size=tokenizer.vocab_size,
                hidden_size=model_args.hidden_size,
                intermediate_size=model_args.intermediate_size,
                num_attention_heads=model_args.num_attention_heads,
                num_hidden_layers=model_args.num_layers,
                max_position_embeddings=model_args.max_position_embeddings,
                _attn_implementation='flash_attention_2' if train_args.bf16 else 'sdpa',
                # rope_theta=torch.inf
                rope_theta=model_args.rope_theta,
                partial_rotary_factor=model_args.partial_rotary_factor,
                use_rpe=model_args.architecture == 'llama-rpe'
            )
            if model_args.architecture == 'llama-random-pos-id':
                model_config.k = 256
                model = LlamaRandPosId(model_config)
            else:
                model = LlamaForCausalLMWithNoPE(model_config)
        elif model_args.architecture == "abacus":
            # model_config = AbacusFalconConfig(
            #     vocab_size=tokenizer.vocab_size,
            #     hidden_size=model_args.hidden_size, # intermediate size is x4 by default
            #     num_attention_heads=model_args.num_attention_heads,
            #     num_hidden_layers=model_args.num_layers,
            #     max_position_embeddings=model_args.max_position_embeddings,
            #     _attn_implementation='eager',
            #     digit_tokens=tokenizer.convert_tokens_to_ids(["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]),
            #     bos_token_id=tokenizer.bos_token_id,
            #     eos_token_id=tokenizer.eos_token_id,
            # )
            model_config = AbacusLlamaConfig(
                vocab_size=tokenizer.vocab_size,
                hidden_size=model_args.hidden_size,
                intermediate_size=model_args.intermediate_size,
                num_attention_heads=model_args.num_attention_heads,
                num_hidden_layers=model_args.num_layers,
                max_position_embeddings=model_args.max_position_embeddings,
                _attn_implementation='flash_attention_2' if train_args.bf16 else 'eager',
                digit_tokens=tokenizer.convert_tokens_to_ids(["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]),
                # activation_function='gelu'
                rope_theta=model_args.rope_theta,
            )
            model = AbacusLlamaForCausalLM(model_config)
        elif model_args.architecture == "add_rule_embedding":
            model_config = LlamaConfigWithAddRules(
                vocab_size=tokenizer.vocab_size,
                hidden_size=model_args.hidden_size,
                intermediate_size=model_args.intermediate_size,
                num_attention_heads=model_args.num_attention_heads,
                num_hidden_layers=model_args.num_layers,
                max_position_embeddings=model_args.max_position_embeddings,
                _attn_implementation='flash_attention_2' if train_args.bf16 else 'eager',
                add_rules=str({tuple(tokenizer.convert_tokens_to_ids(['A', 'B'])): tokenizer.convert_tokens_to_ids('C')}),
                rope_theta=model_args.rope_theta,
                partial_rotary_factor=model_args.partial_rotary_factor
            )
            model = LlamaModelWithAddRules(model_config)
        else:
            raise ValueError(f"Unknown architecture: {model_args.architecture}")
        model_args.model_id = f"{model_args.architecture}-{model_args.hidden_size}-{model_args.num_attention_heads}-{model_args.num_layers}-{model_args.max_position_embeddings}"
    
        if model_args.use_lora:
            lora_config =  LoraConfig(
                r=8,
                task_type="CAUSAL_LM"
            )

            model = get_peft_model(model, lora_config)
            model.print_trainable_parameters()

        print(f"Number of parameters: {model.num_parameters()}")
        print(f"Number of trainable parameters: {model.num_parameters(only_trainable=True)}")

    return model

def prepare_train_args(train_args: MyTrainingArguments, model_args: ModelArguments, data_args: DataArguments, tokenizer: PreTrainedTokenizer):
    train_args.generation_config = GenerationConfig(
        do_sample=False,
        num_beams=1,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
        bos_token_id=tokenizer.bos_token_id,
        use_cache=True,
        max_length=0, # because we set it in the trainer prediction_step
        # forced_eos_token_id=tokenizer.eos_token_id
    )

    train_args.run_name += f"-{model_args.model_id}"
    if model_args.from_pretrained:
        train_args.run_name += "-pretrained"
    if model_args.use_lora:
        train_args.run_name += "-lora"
    if model_args.freeze:
        train_args.run_name += "-frz-" + model_args.freeze
    if model_args.freeze_except:
        train_args.run_name += "-frzex-" + model_args.freeze_except
    if model_args.rope_theta != torch.inf:
        train_args.run_name += f"-rope"
    if len(set(data_args.num_train)) != 1:
        train_args.run_name += f"-train-{data_args.num_train}"
    disp_task = [data_args.format_train[i] if data_args.format_train[i] != 'None' else data_args.op_train[i] for i in range(len(data_args.format_train))]
    train_args.run_name += f'-{disp_task}-digits-{data_args.n_digits_train}'
    translator = str.maketrans('/,', '__', ''.join(set(string.punctuation + string.whitespace) - set('/,_-')))
    train_args.run_name = str.translate(train_args.run_name, translator)

    train_args.output_dir = f"out/{train_args.run_name}"
    train_args.save_safetensors = False # supposed to fix "There were missing keys in the checkpoint model loaded: ['lm_head.weight']."
    train_args.dataloader_num_workers = data_args.nproc
    train_args.dataloader_prefetch_factor = 3
    train_args.remove_unused_columns = False
    
    if train_args.resume_from_checkpoint == 'True':
        # Try finding a checkpoint in the output directory
        try:
            train_args.resume_from_checkpoint = get_last_checkpoint(train_args.output_dir)
        except FileNotFoundError:
            train_args.resume_from_checkpoint = None
    elif train_args.resume_from_checkpoint == 'False':
        train_args.resume_from_checkpoint = None
    else:
        # Try finding a checkpoint in the provided path
        try:
            ckpt_dir = get_last_checkpoint(train_args.resume_from_checkpoint)
            if ckpt_dir is not None:
                train_args.resume_from_checkpoint = ckpt_dir
        except:
            pass
    
    train_args.run_name += f'-seed-{train_args.seed}'
    
    if not train_args.do_train:
        train_args.run_name += '-eval'

    return train_args

def get_trainer(args: ScriptArguments, data_args: DataArguments, model_args: ModelArguments, model: PreTrainedModel, tokenizer: PreTrainedTokenizer, train_args: MyTrainingArguments, train_dataset: Dataset, eval_datasets: Dataset):
    trainer = Seq2SeqTrainerNoEvalLoss(
        model=model,
        tokenizer=tokenizer,
        args=train_args,
        train_dataset=train_dataset if train_args.do_train else None,
        eval_dataset=eval_datasets if train_args.do_eval else None,
        compute_metrics=partial(compute_metrics, tokenizer, args=train_args),
        data_collator=PromptAnswerDataCollator(
            pad_token_id=tokenizer.pad_token_id,
            label_pad_token_id=-100,
            train_pad_side=data_args.padding_side,
            train_pad_to=data_args.train_pad_to if os.environ.get("WORLD_SIZE") is not None else None,
            eval_pad_to=data_args.eval_pad_to if os.environ.get("WORLD_SIZE") is not None else None
        )
    )

    if "LOCAL_RANK" not in os.environ or os.environ["LOCAL_RANK"] == "0":
        AddConfigCB = AddWandbConfigCallback(extra_configs=[args.__dict__, data_args.__dict__, model_args.__dict__])
        trainer.add_callback(AddConfigCB)

    if train_args.metric_for_best_model is not None:
        EarlyStoppingCB = EarlyStoppingCallback(metric_name=train_args.metric_for_best_model, threshold=0.99, patience=1)
        trainer.add_callback(EarlyStoppingCB)

    return trainer
