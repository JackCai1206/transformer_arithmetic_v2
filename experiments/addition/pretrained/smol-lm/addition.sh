set -e
# Train
CUDA_VISIBLE_DEVICES=0 WANDB_PROJECT=mamba-arithmetic WANDB_MODE=online python run.py \
    --do_dpo=True \
    \
    \
    --model_id='HuggingFaceTB/SmolLM-135M' \
    --from_pretrained=True \
    \
    \
    --n_digits_train=20 \
    --n_digits_train_min=3 \
    --n_digits_eval_start=10 \
    --n_digits_eval_end=128 \
    --n_digits_eval_step=20 \
    --n_digits_dpo=30 \
    --n_digits_dpo_min=25 \
    --num_dpo_data=10000 \
    --op='add' \
    --format='reverse' \
    \
    \
    --resume_from_checkpoint='out/HuggingFaceTB-SmolLM-135M-pretrained-add-digits-20/checkpoint-500' \
    --run_name='test' \
    --output_dir=out \
    --do_train=False \
    --do_eval=False \
    --remove_unused_columns=False \
    --max_steps=1000 \
    --learning_rate=3e-3 \
    --lr_scheduler_type='cosine' \
    --warmup_ratio=0.05 \
    --logging_steps=10 \
    --eval_strategy="steps" \
    --eval_steps=50 \
    --eval_on_start=True \
    --predict_with_generate \
    --per_device_train_batch_size=50 \
    --per_device_eval_batch_size=1024 \
    --gradient_accumulation_steps=16 \
    --include_inputs_for_metrics=True \
    --tf32=True \
    --torch_compile=False
