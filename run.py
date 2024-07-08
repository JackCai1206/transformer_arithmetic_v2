from functools import partial
from lib.configs import ScriptArguments, ModelArguments, DataArguments
from lib.data_utils import get_train_dataset, get_eval_dataset
from lib.eval_utils import compute_metrics
from lib.trainer_utils import Seq2SeqTrainerNoEvalLoss, AddWandbConfigCallback
from charactertokenizer import CharacterTokenizer

from typing import cast
from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments, MambaConfig, MambaForCausalLM, HfArgumentParser, set_seed, GenerationConfig, AutoModelForCausalLM, AutoConfig
from peft import LoraConfig, get_peft_model

args, model_args, data_args, train_args = HfArgumentParser((ScriptArguments, ModelArguments, DataArguments, Seq2SeqTrainingArguments)).parse_args_into_dataclasses()
args = cast(ScriptArguments, args)
model_args = cast(ModelArguments, model_args)
train_args = cast(Seq2SeqTrainingArguments, train_args)
data_args = cast(DataArguments, data_args)


set_seed(train_args.seed)

all_chars = "0123456789+-*/=() \nABCDSP"
tokenizer = CharacterTokenizer(all_chars, model_args.max_position_embeddings)

train_dataset = get_train_dataset(data_args, tokenizer)
eval_datasets = get_eval_dataset(data_args, tokenizer)

if model_args.model_id is not None:
    if model_args.from_pretrained:
        model = AutoModelForCausalLM.from_pretrained(model_args.model_id)
    else:
        model_config = AutoConfig.from_pretrained(model_args.model_id)
        model = AutoModelForCausalLM.from_config(model_config)
else:
    model_config = MambaConfig(
        vocab_size=tokenizer.vocab_size,
        hidden_size=model_args.hidden_size,
        state_size=model_args.state_size,
        num_hidden_layers=model_args.num_layers
    )

    model = MambaForCausalLM(model_config)

if model_args.use_lora:
    lora_config =  LoraConfig(
        r=8,
        target_modules=["x_proj", "embeddings", "in_proj", "out_proj"],
        task_type="CAUSAL_LM",
        bias="none"
    )

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

print(f"Number of trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")

train_args.generation_config = GenerationConfig(
    do_sample=False,
    num_beams=1,
    pad_token_id=tokenizer.pad_token_id,
    eos_token_id=tokenizer.eos_token_id,
    forced_eos_token_id=tokenizer.eos_token_id
)

trainer = Seq2SeqTrainerNoEvalLoss(
    model=model,
    args=train_args,
    train_dataset=train_dataset,
    eval_dataset=eval_datasets,
    compute_metrics=partial(compute_metrics, tokenizer)
)

AddConfigCB = AddWandbConfigCallback(extra_configs=[args.__dict__, data_args.__dict__])
trainer.add_callback(AddConfigCB)

trainer.train()
