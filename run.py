import os
from preamble import get_args, get_tokenizer, get_all_datasets, get_model, prepare_train_args, get_trainer

args, model_args, data_args, train_args = get_args()

tokenizer = get_tokenizer(model_args, data_args)

train_dataset, eval_datasets = get_all_datasets(train_args, data_args, tokenizer)

model = get_model(train_args, model_args, tokenizer)

train_args = prepare_train_args(train_args, model_args, data_args, tokenizer)

trainer = get_trainer(args, data_args, model_args, model, tokenizer, train_args, train_dataset, eval_datasets)

# check local rank
# if "LOCAL_RANK" in os.environ and os.environ["LOCAL_RANK"] == "0":
#     import wandb
#     wandb.init(project='LG-inherit', entity="jackcai1206", name=train_args.run_name)

#     # Workaround for incrorrect global metrics
#     # define our custom x axis metric
#     wandb.define_metric("train/global_step")
#     # set all other train/ metrics to use this step
#     wandb.define_metric("*", step_metric="train/global_step")

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
    import torch
    import copy
    
    from typing import cast
    from functools import partial

    from lib.data_utils import get_dpo_dataset
    from lib.eval_utils import compute_metrics
    from lib.trainer_utils import DPOTrainerDefaultEval, DPOSeq2SeqConfig
    
    from transformers import HfArgumentParser
    
    # train_args.batch_eval_metrics = True
    # train_args.report_to = []
    # trainer = Seq2SeqTrainerNoEvalLoss(
    #     model=model,
    #     tokenizer=tokenizer,
    #     args=train_args,
    #     eval_dataset=dpo_dataset,
    #     compute_metrics=None,
    #     data_collator=DataCollatorForSeq2Seq(tokenizer, padding='longest')
    # )
    # if train_args.resume_from_checkpoint is not None:
    #     trainer._load_from_checkpoint(resume_from_checkpoint=train_args.resume_from_checkpoint)

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
    ref_model = None
    if args.ref_model:
        new_model_args = copy.deepcopy(model_args)
        new_model_args.model_id = None
        ref_model = get_model(train_args, new_model_args, tokenizer)
        ref_model_ckpt_path = os.path.join(args.ref_model_path, 'pytorch_model.bin')
        ref_model.load_state_dict(torch.load(ref_model_ckpt_path))        
        ref_model.eval()

        # # check if ref_model is working
        # import ipdb; ipdb.set_trace()
        # trainer2 = get_trainer(args, data_args, model_args, ref_model, tokenizer, train_args, train_dataset, eval_datasets)
        # trainer2.evaluate()

    dpo_config = HfArgumentParser((DPOSeq2SeqConfig), allow_abbrev=False).parse_args_into_dataclasses(return_remaining_strings=True)[0]
    dpo_config = cast(DPOSeq2SeqConfig, dpo_config)
    dpo_config.run_name = train_args.run_name + '-dpo'
    dpo_config.output_dir = train_args.output_dir + '-dpo'
    dpo_config.reference_free = True
    dpo_config.max_length = 1024
    dpo_config.max_prompt_length = 128 # default
    dpo_config.beta = args.dpo_beta

    if train_args.resume_from_checkpoint is not None:
        resume_ckpt_path = os.path.join(train_args.resume_from_checkpoint, 'pytorch_model.bin')
        model.load_state_dict(torch.load(resume_ckpt_path))        

    dpo_trainer = DPOTrainerDefaultEval(
        model=model,
        ref_model=ref_model,
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
