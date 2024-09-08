from functools import partial
import re
import string

import torch
from lib.configs import ScriptArguments, ModelArguments, DataArguments
from lib.data_utils import get_dpo_dataset, get_train_dataset, get_eval_dataset, PromptAnswerDataCollator
from lib.eval_utils import compute_metrics
from lib.modeling.add_rule_embedding import LlamaConfigWithAddRules, LlamaModelWithAddRules
from lib.modeling.llama import LlamaForCausalLMWithNoPE
from lib.modeling.llama_rand_pos_id import LlamaRandPosId
from lib.trainer_utils import DPOTrainerDefaultEval, Seq2SeqTrainerNoEvalLoss, AddWandbConfigCallback, DPOSeq2SeqConfig
from lib.modeling.cat import ConvLlamaForCausalLM
from lib.modeling.abacus import AbacusLlamaForCausalLM, AbacusLlamaModel, AbacusLlamaConfig
from charactertokenizer import CharacterTokenizer

from typing import cast
from transformers import Seq2SeqTrainingArguments, HfArgumentParser, set_seed, GenerationConfig, AutoModelForCausalLM, AutoConfig, PreTrainedModel, LlamaConfig, LlamaForCausalLM, AutoTokenizer, DataCollatorForSeq2Seq, PreTrainedTokenizer
from datasets import Dataset
from peft import LoraConfig, get_peft_model

args, model_args, data_args, train_args = HfArgumentParser((ScriptArguments, ModelArguments, DataArguments, Seq2SeqTrainingArguments)).parse_args_into_dataclasses()
args = cast(ScriptArguments, args)
model_args = cast(ModelArguments, model_args)
train_args = cast(Seq2SeqTrainingArguments, train_args)
data_args = cast(DataArguments, data_args)
data_args.block_size = model_args.max_position_embeddings

set_seed(train_args.seed)

# We don't pad when generating the datasets
# During eval, the inputs are padded on the left and the labels are padding on the right using custom data collator
all_chars = string.ascii_letters + string.digits + string.punctuation + ' '
tokenizer = CharacterTokenizer(all_chars, model_args.max_position_embeddings)
tokenizer.padding_side == 'left'

if any([x == 'sort' for x in data_args.op_train]):
    added_tokens = tokenizer.add_tokens([f'[{str(i).zfill(2)}]' for i in range(100)])

if train_args.do_train:
    train_dataset = get_train_dataset(train_args, data_args, tokenizer)
if train_args.do_eval or args.do_dpo:
    eval_datasets = get_eval_dataset(train_args, data_args, tokenizer)
if args.do_dpo:
    dpo_dataset = get_dpo_dataset(data_args, tokenizer)
tokenizer.padding_side = 'left' # in case it was changed by the data generator

def get_model(model_args: ModelArguments, tokenizer: PreTrainedTokenizer):
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
        elif model_args.architecture.startswith("llama"):
            model_config = LlamaConfig(
                vocab_size=tokenizer.vocab_size,
                hidden_size=model_args.hidden_size,
                intermediate_size=model_args.intermediate_size,
                num_attention_heads=model_args.num_attention_heads,
                num_hidden_layers=model_args.num_layers,
                max_position_embeddings=model_args.max_position_embeddings,
                _attn_implementation='flash_attention_2' if train_args.bf16 else 'sdpa',
                # rope_theta=torch.inf
                rope_theta=model_args.rope_theta
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
                rope_theta=model_args.rope_theta
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
                add_rules=str({tuple(tokenizer.convert_tokens_to_ids(['A', 'B'])): tokenizer.convert_tokens_to_ids('C')})
            )
            model = LlamaModelWithAddRules(model_config)
        else:
            raise ValueError(f"Unknown architecture: {model_args.architecture}")
        model_args.model_id = f"{model_args.architecture}-{model_args.hidden_size}-{model_args.num_attention_heads}-{model_args.num_layers}-{model_args.max_position_embeddings}"
    
    return model

model = get_model(model_args, tokenizer)
print(model)

if model_args.use_lora:
    lora_config =  LoraConfig(
        r=8,
        task_type="CAUSAL_LM"
    )

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

print(f"Number of parameters: {model.num_parameters()}")
print(f"Number of trainable parameters: {model.num_parameters(only_trainable=True)}")

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

train_args.run_name = f"{model_args.model_id}"
if model_args.from_pretrained:
    train_args.run_name += "-pretrained"
if model_args.use_lora:
    train_args.run_name += "-lora"
if model_args.freeze:
    train_args.run_name += "-frz-" + model_args.freeze
if model_args.freeze_except:
    train_args.run_name += "-frzex-" + model_args.freeze_except
disp_task = [data_args.format_train[i] if data_args.format_train[i] != 'None' else data_args.op_train[i] for i in range(len(data_args.format_train))]
train_args.run_name += f'-{disp_task}-digits-{data_args.n_digits_train}'
translator = str.maketrans('/,', '__', ''.join(set(string.punctuation + string.whitespace) - set('/,_-')))
train_args.run_name = str.translate(train_args.run_name, translator)

train_args.output_dir = f"out/{train_args.run_name}"
train_args.save_safetensors = False # supposed to fix "There were missing keys in the checkpoint model loaded: ['lm_head.weight']."
train_args.dataloader_num_workers = data_args.nproc
train_args.remove_unused_columns = False

trainer = Seq2SeqTrainerNoEvalLoss(
    model=model,
    tokenizer=tokenizer,
    args=train_args,
    train_dataset=train_dataset if train_args.do_train else None,
    eval_dataset=eval_datasets if train_args.do_eval else None,
    compute_metrics=partial(compute_metrics, tokenizer),
    data_collator=PromptAnswerDataCollator(pad_token_id=tokenizer.pad_token_id)
)

AddConfigCB = AddWandbConfigCallback(extra_configs=[args.__dict__, data_args.__dict__, model_args.__dict__])
trainer.add_callback(AddConfigCB)

if train_args.do_train:
    if train_args.resume_from_checkpoint is not None:
        trainer.train(resume_from_checkpoint=train_args.resume_from_checkpoint)
    else:
        # import cProfile
        # with cProfile.Profile() as pr:
        #     trainer.train()
        # pr.dump_stats(f"{train_args.output_dir}/profile")
        trainer.train()
elif train_args.do_eval and not args.do_dpo:
    trainer._load_from_checkpoint(resume_from_checkpoint=train_args.resume_from_checkpoint)
    trainer.evaluate()

if args.do_dpo:
    from trl import DPOTrainer, DPOConfig
    from transformers import EvalPrediction
    from lib.eval_utils import WandbEvalCallback
    from torch.utils.data import DataLoader
    
    train_args.batch_eval_metrics = True
    train_args.report_to = []
    trainer = Seq2SeqTrainerNoEvalLoss(
        model=model,
        tokenizer=tokenizer,
        args=train_args,
        eval_dataset=dpo_dataset,
        compute_metrics=None,
        data_collator=DataCollatorForSeq2Seq(tokenizer, padding='longest')
    )
    if train_args.resume_from_checkpoint is not None:
        trainer._load_from_checkpoint(resume_from_checkpoint=train_args.resume_from_checkpoint)

    # 1. Collect DPO data
    # def get_dpo_data(model: PreTrainedModel, dataloader: DataLoader, tokenizer: PreTrainedTokenizer, generation_config: GenerationConfig):
    #     for batch in dataloader:
    #         outputs = model.generate(**batch, generation_config=generation_config, max_new_tokens=batch['labels'].shape[1])
    #         for i in range(len(batch['input_ids'])):
    #             x = {
    #                 'prompt': '[BOS]' + tokenizer.decode(batch['input_ids'][i], skip_special_tokens=True),
    #                 'chosen': tokenizer.decode(batch['labels'][i], skip_special_tokens=True) + '[EOS]',
    #                 'rejected': tokenizer.decode(outputs[i][len(batch['input_ids'][i]):], skip_special_tokens=True) + '[EOS]'
    #             }
    #             yield x
    
    # dpo_dataset = Dataset.from_generator(get_dpo_data, gen_kwargs={
    #     'model': trainer.model,
    #     'dataloader': trainer.get_eval_dataloader(dpo_dataset),
    #     'tokenizer': tokenizer,
    #     'generation_config': train_args.generation_config
    # })
    dpo_dataset = get_dpo_dataset(data_args, tokenizer)

    # 2. Train DPO model, eval with seq2seq trainer
    # ref_model = get_model(model_args, tokenizer)
    dpo_config = HfArgumentParser((DPOSeq2SeqConfig), allow_abbrev=False).parse_args_into_dataclasses(return_remaining_strings=True)[0]
    dpo_config = cast(DPOSeq2SeqConfig, dpo_config)
    dpo_config.run_name = train_args.run_name + '-dpo'
    dpo_config.output_dir = train_args.output_dir + '-dpo'
    dpo_config.reference_free = True
    dpo_config.beta = 0.5
    dpo_trainer = DPOTrainerDefaultEval(
        model=model,
        ref_model=None,
        train_dataset=dpo_dataset,
        eval_dataset=eval_datasets,
        args=dpo_config,
        tokenizer=tokenizer,
        callbacks=[],
        compute_metrics=partial(compute_metrics, tokenizer)
    )

    dpo_trainer.train(
        ignore_keys_for_eval=['prompt', 'chosen', 'rejected'],
    )
