from functools import partial

import torch
from lib.configs import ScriptArguments, ModelArguments, DataArguments
from lib.data_utils import get_train_dataset, get_eval_dataset
from lib.eval_utils import compute_metrics
from lib.trainer_utils import Seq2SeqTrainerNoEvalLoss, AddWandbConfigCallback
from lib.modeling.cat import ConvLlamaForCausalLM
from charactertokenizer import CharacterTokenizer

from typing import cast
from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments, MambaConfig, MambaForCausalLM, HfArgumentParser, set_seed, GenerationConfig, AutoModelForCausalLM, AutoConfig, PreTrainedModel, LlamaConfig, LlamaForCausalLM
from peft import LoraConfig, get_peft_model

args, model_args, data_args, train_args = HfArgumentParser((ScriptArguments, ModelArguments, DataArguments, Seq2SeqTrainingArguments)).parse_args_into_dataclasses()
args = cast(ScriptArguments, args)
model_args = cast(ModelArguments, model_args)
train_args = cast(Seq2SeqTrainingArguments, train_args)
data_args = cast(DataArguments, data_args)
data_args.block_size = model_args.max_position_embeddings

set_seed(train_args.seed)

all_chars = "0123456789+-*/=() \nABCDSP"
tokenizer = CharacterTokenizer(all_chars, model_args.max_position_embeddings)

train_dataset = get_train_dataset(train_args, data_args, tokenizer)
eval_datasets = get_eval_dataset(train_args, data_args, tokenizer)

if model_args.model_id is not None:
    model: PreTrainedModel
    if model_args.from_pretrained:
        model = AutoModelForCausalLM.from_pretrained(model_args.model_id)
    else:
        model_config = AutoConfig.from_pretrained(model_args.model_id)
        model = AutoModelForCausalLM.from_config(model_config)

    emb = model.get_input_embeddings()
    assert isinstance(emb, torch.nn.Embedding), "Only nn.Embedding is supported for now"
    tokens = tokenizer.convert_tokens_to_ids(all_chars.split())
    new_emb = torch.nn.Embedding(len(tokens), emb.embedding_dim)
    new_emb.weight.data = emb.weight.data[tokens]
    model.set_input_embeddings(emb)
    
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
            intermediate_size=model_args.hidden_size * 4,
            num_attention_heads=model_args.num_attention_heads,
            num_hidden_layers=model_args.num_layers,
            max_position_embeddings=model_args.max_position_embeddings,
            _attn_implementation='eager'
        )
        model = ConvLlamaForCausalLM(model_config)
    elif model_args.architecture == "llama":
        model_config = LlamaConfig(
            vocab_size=tokenizer.vocab_size,
            hidden_size=model_args.hidden_size,
            intermediate_size=model_args.hidden_size * 4,
            num_attention_heads=model_args.num_attention_heads,
            num_hidden_layers=model_args.num_layers,
            max_position_embeddings=model_args.max_position_embeddings,
            _attn_implementation='eager'
        )
        model = LlamaForCausalLM(model_config)
    else:
        raise ValueError(f"Unknown architecture: {model_args.architecture}")

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
    use_cache=True
)

train_args.run_name = f"{model_args.architecture}-{data_args.op}-digits-{data_args.n_digits_train}"
train_args.output_dir = f"out/{train_args.run_name}"
if train_args.resume_from_checkpoint is not None:
    train_args.resume_from_checkpoint = f"{train_args.output_dir}/checkpoint-{train_args.resume_from_checkpoint}"

trainer = Seq2SeqTrainerNoEvalLoss(
    model=model,
    tokenizer=tokenizer,
    args=train_args,
    train_dataset=train_dataset,
    eval_dataset=eval_datasets,
    compute_metrics=partial(compute_metrics, tokenizer)
)

AddConfigCB = AddWandbConfigCallback(extra_configs=[args.__dict__, data_args.__dict__])
trainer.add_callback(AddConfigCB)

if train_args.do_train:
    if train_args.resume_from_checkpoint is not None:
        trainer._load_from_checkpoint(resume_from_checkpoint=train_args.resume_from_checkpoint)
    else:
        trainer.train()
elif train_args.do_eval:
    trainer._load_from_checkpoint(resume_from_checkpoint=train_args.resume_from_checkpoint)
    trainer.evaluate()
