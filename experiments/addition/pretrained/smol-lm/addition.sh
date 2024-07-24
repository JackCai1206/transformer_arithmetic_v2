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
    --op='add' \
    --format='reverse' \
    \
    \
    --resume_from_checkpoint='out/HuggingFaceTB-SmolLM-135M-pretrained-add-digits-20/checkpoint-500' \
    --run_name='test' \
    --output_dir=out \
    --do_train=False \
    --do_eval=False \
    --max_steps=500 \
    --learning_rate=3e-3 \
    --lr_scheduler_type='cosine' \
    --warmup_steps=200 \
    --logging_steps=20 \
    --eval_strategy="no" \
    --eval_steps=200 \
    --predict_with_generate \
    --per_device_train_batch_size=64 \
    --per_device_eval_batch_size=100 \
    --gradient_accumulation_steps=16 \
    --include_inputs_for_metrics=True \
    --tf32=True \
    --torch_compile=False
