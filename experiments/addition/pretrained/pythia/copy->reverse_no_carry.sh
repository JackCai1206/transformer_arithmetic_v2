set -e

for from_pretrained use_lora max_steps eval_steps in \
    True False 2000 200 \
;do
    CUDA_VISIBLE_DEVICES=1 WANDB_PROJECT=mamba-arithmetic WANDB_MODE=online python run.py \
        --model_id='EleutherAI/pythia-160m' \
        --from_pretrained=$from_pretrained \
        --use_lora=$use_lora \
        \
        \
        --n_digits_train=32 \
        --n_digits_train_min=5 \
        --n_digits_eval_start=16 \
        --n_digits_eval_end=128 \
        --n_digits_eval_step=16 \
        --op='add' \
        --format='reverse-no-carry' \
        \
        \
        --run_name='test' \
        --resume_from_checkpoint='out/EleutherAI-pythia-160m-pretrained-copy--digits-64/checkpoint-500' \
        --output_dir=out \
        --do_train=True \
        --do_eval=True \
        --max_steps=$max_steps \
        --learning_rate=1e-3 \
        --lr_scheduler_type='cosine' \
        --warmup_ratio=0.05 \
        --logging_steps=20 \
        --eval_strategy="steps" \
        --eval_steps=$eval_steps \
        --predict_with_generate \
        --remove_unused_columns=False \
        --eval_on_start=True \
        --per_device_train_batch_size=128 \
        --per_device_eval_batch_size=1024 \
        --gradient_accumulation_steps=16 \
        --include_inputs_for_metrics=True \
        --save_steps=200 \
        --torch_compile=False \
        --bf16=False \
        --tf32=True
done
