import os
from preamble import get_args, get_tokenizer, get_all_datasets_from_saved, get_model, prepare_train_args, get_trainer
import torch

args, model_args, data_args, train_args = get_args()

tokenizer = get_tokenizer(model_args, data_args)

model = get_model(train_args, model_args, tokenizer)

train_args = prepare_train_args(train_args, model_args, data_args, tokenizer)

print('*' * 50 +'\nLoading model from:', train_args.resume_from_checkpoint)
model.load_state_dict(torch.load(os.path.join(train_args.resume_from_checkpoint, 'pytorch_model.bin')))
model = model.to(torch.bfloat16).to(train_args.device)

print('output_dir: ', train_args.output_dir)

# now we use the train_dataset and make a new dataset with the same prompt but with the labels from the model
#--resume_from_checkpoint='out/self_improve/reverse_20000000-llama-384-6-6-1024-reverse-digits-1_10_-seed-44'
model_id = '/'.join(train_args.resume_from_checkpoint.split('/')[-2:]).replace('/', '_') # 
print('model_id:', model_id)
data_args.eval_file_from_model = f'data/eval_from_model-{model_id}'
data_args.train_file_from_model = f'data/train_from_model-{model_id}'

train_dataset, eval_datasets = get_all_datasets_from_saved(train_args, data_args, tokenizer)

trainer = get_trainer(args, data_args, model_args, model, tokenizer, train_args, train_dataset, eval_datasets)

# check local rank
if "LOCAL_RANK" not in os.environ or os.environ["LOCAL_RANK"] == "0":
    import wandb
    wandb.init(project=args.wandb_project, entity="ssdd", name=train_args.run_name)

    # Workaround for incrorrect global metrics
    # define our custom x axis metric
    wandb.define_metric("train/global_step")
    # set all other train/ metrics to use this step
    wandb.define_metric("*", step_metric="train/global_step")

if train_args.do_train:
    if False:
    # if train_args.resume_from_checkpoint is not None:
        trainer.train(resume_from_checkpoint=train_args.resume_from_checkpoint)
    else:
        trainer.train()

elif train_args.do_eval and not train_args.do_dpo:
    trainer._load_from_checkpoint(resume_from_checkpoint=train_args.resume_from_checkpoint)
    trainer.evaluate()

