from functools import partial

import torch
from lib.configs import ScriptArguments, ModelArguments, DataArguments
from lib.data_utils import get_train_dataset, get_eval_dataset
from lib.eval_utils import compute_metrics
from lib.trainer_utils import Seq2SeqTrainerNoEvalLoss, AddWandbConfigCallback
from lib.modeling.cat import ConvLlamaForCausalLM
from lib.modeling.abacus import AbacusFalconConfig, AbacusFalconModel, AbacusGPT2Config, AbacusGPT2Model
from charactertokenizer import CharacterTokenizer

from typing import cast
from transformers import Seq2SeqTrainingArguments, MambaConfig, MambaForCausalLM, HfArgumentParser, set_seed, GenerationConfig, AutoModelForCausalLM, AutoConfig, PreTrainedModel, LlamaConfig, LlamaForCausalLM, AutoTokenizer, DataCollatorForSeq2Seq, PreTrainedTokenizer
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
all_chars = "0123456789+-*/=() \nABCDSP"
tokenizer = CharacterTokenizer(all_chars, model_args.max_position_embeddings)
tokenizer.padding_side == 'left'

if train_args.do_train:
    train_dataset = get_train_dataset(train_args, data_args, tokenizer)
if train_args.do_eval or args.do_dpo:
    eval_datasets = get_eval_dataset(train_args, data_args, tokenizer)
if args.do_dpo:
    dpo_dataset = get_eval_dataset(train_args, data_args, tokenizer)['30']
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

        for name, module in model.named_modules():
            for freeze in model_args.freeze:
                if freeze in name:
                    module.requires_grad = False
                    print(f"Freezing {name}")

        model_args.architecture = model.config.architectures[0]
    else:
        if model_args.architecture == "mamba":
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
        elif model_args.architecture == "llama":
            model_config = LlamaConfig(
                vocab_size=tokenizer.vocab_size,
                hidden_size=model_args.hidden_size,
                intermediate_size=model_args.intermediate_size,
                num_attention_heads=model_args.num_attention_heads,
                num_hidden_layers=model_args.num_layers,
                max_position_embeddings=model_args.max_position_embeddings,
                _attn_implementation='sdpa'
            )
            model = LlamaForCausalLM(model_config)
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
            model_config = AbacusGPT2Config(
                vocab_size=tokenizer.vocab_size,
                n_embd=model_args.hidden_size,
                n_inner=model_args.intermediate_size,
                n_head=model_args.num_attention_heads,
                n_layer=model_args.num_layers,
                max_position_embeddings=model_args.max_position_embeddings,
                _attn_implementation='sdpa',
                digit_tokens=tokenizer.convert_tokens_to_ids(["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]),
                activation_function='gelu'
            )
            model = AbacusGPT2Model(model_config)
        else:
            raise ValueError(f"Unknown architecture: {model_args.architecture}")
        model_args.model_id = f"{model_args.architecture}-{model_args.hidden_size}-{model_args.num_attention_heads}-{model_args.num_layers}-{model_args.max_position_embeddings}"
    
    return model

model = get_model(model_args, tokenizer)
print(model)

if model_args.use_lora:
    lora_config =  LoraConfig(
        r=8,
        target_modules=['c_attn'],
        task_type="CAUSAL_LM"
    )

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

print(f"Number of trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")

train_args.generation_config = GenerationConfig(
    do_sample=False,
    num_beams=1,
    pad_token_id=tokenizer.pad_token_id,
    eos_token_id=tokenizer.eos_token_id,
    use_cache=False,
)

train_args.run_name = f"{model_args.model_id}"
if model_args.from_pretrained:
    train_args.run_name += "-pretrained"
train_args.run_name += f'-{data_args.op}-digits-{data_args.n_digits_train}'
train_args.run_name = train_args.run_name.replace('/', '-')

train_args.output_dir = f"out/{train_args.run_name}"
train_args.save_safetensors = False # supposed to fix "There were missing keys in the checkpoint model loaded: ['lm_head.weight']."

trainer = Seq2SeqTrainerNoEvalLoss(
    model=model,
    tokenizer=tokenizer,
    args=train_args,
    train_dataset=train_dataset if train_args.do_train else None,
    eval_dataset=eval_datasets if train_args.do_eval else None,
    compute_metrics=partial(compute_metrics, tokenizer),
    data_collator=DataCollatorForSeq2Seq(tokenizer, padding='longest')
)

AddConfigCB = AddWandbConfigCallback(extra_configs=[args.__dict__, data_args.__dict__])
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
elif train_args.do_eval:
    trainer._load_from_checkpoint(resume_from_checkpoint=train_args.resume_from_checkpoint)
    trainer.evaluate()
    
if args.do_dpo:
    from trl import DPOTrainer, DPOConfig
    from transformers import EvalPrediction
    from copy import deepcopy
    
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
    trainer._load_from_checkpoint(resume_from_checkpoint=train_args.resume_from_checkpoint)

    # 1. Collect DPO data
    def get_dpo_data(trainer: Seq2SeqTrainerNoEvalLoss):
        inputs = []
        labels = []
        predictions = []
        def yield_dpo_data(pred: EvalPrediction, compute_result=False):
            inputs.extend(trainer.tokenizer.batch_decode(pred.inputs['input_ids'], skip_special_tokens=True))
            labels.extend(trainer.tokenizer.batch_decode(pred.label_ids, skip_special_tokens=True))
            predictions.extend(trainer.tokenizer.batch_decode(pred.predictions[:, pred.inputs['input_ids'].shape[1]:], skip_special_tokens=True))
            return {}

        trainer.compute_metrics = yield_dpo_data
        trainer.evaluate()

        for i in range(len(inputs)):
            yield {
                'prompt': inputs[i],
                'chosen': labels[i],
                'rejected': predictions[i]
            }

    dpo_dataset = Dataset.from_generator(get_dpo_data, gen_kwargs={'trainer': trainer})

    ref_model = get_model(model_args, tokenizer)
    dpo_config = HfArgumentParser((DPOConfig), allow_abbrev=False).parse_args_into_dataclasses(return_remaining_strings=True)[0]
    dpo_config.output_dir += '-dpo'
    dpo_trainer = DPOTrainer(
        model=model,
        ref_model=ref_model,
        train_dataset=dpo_dataset,
        args=dpo_config,
        tokenizer=tokenizer,
        callbacks=[]
    )
    
    dpo_trainer.train()
