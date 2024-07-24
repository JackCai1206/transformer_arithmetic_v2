set -e

CUDA_VISIBLE_DEVICES=1 WANDB_PROJECT=mamba-arithmetic WANDB_MODE=online python run.py \
    --model_id='Qwen/Qwen2-0.5B' \
    --from_pretrained=True \
    --num_train=2000000 \
    \
    \
    --n_digits_train=128 \
    --n_digits_train_min=32 \
    --n_digits_eval_start=32 \
    --n_digits_eval_end=256 \
    --n_digits_eval_step=32 \
    --op='copy' \
    --format='' \
    \
    \
    --run_name='test' \
    --output_dir=out \
    --do_train=True \
    --do_eval=True \
    --max_steps=5000 \
    --learning_rate=1e-4 \
    --lr_scheduler_type='cosine' \
    --warmup_steps=200 \
    --logging_steps=20 \
    --eval_strategy="steps" \
    --eval_steps=200 \
    --predict_with_generate \
    --per_device_train_batch_size=8 \
    --per_device_eval_batch_size=64 \
    --gradient_accumulation_steps=16 \
    --include_inputs_for_metrics=True \
    --save_steps=200 \
    --torch_compile=True \
    --bf16=False \
    --tf32=True \
