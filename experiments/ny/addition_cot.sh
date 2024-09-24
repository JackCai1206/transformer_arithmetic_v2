set -e

CUDA_VISIBLE_DEVICES=0 WANDB_PROJECT=arithmetic-dpo WANDB_MODE=online python run.py \
    --architecture=llama \
    --hidden_size=384 \
    --num_attention_heads=6 \
    --num_layers=6 \
    --max_position_embeddings=1024 \
    \
    \
    --n_digits_train='1,10' \
    --op_train='add' \
    --format_train='COT' \
    --n_digits_eval='2,17,2' \
    --op_eval='add' \
    --format_eval='COT' \
    \
    \
    --run_name='test' \
    --output_dir=out \
    --do_train=True \
    --do_eval=True \
    --max_steps=5000 \
    --learning_rate=5e-4 \
    --lr_scheduler_type='cosine' \
    --warmup_ratio=0.05 \
    --logging_steps=20 \
    --eval_strategy="steps" \
    --eval_steps=200 \
    --predict_with_generate \
    --per_device_train_batch_size=256 \
    --per_device_eval_batch_size=100 \
    --gradient_accumulation_steps=2 \
    --include_inputs_for_metrics=True \
    --bf16=True \
    --tf32=True \
    --torch_compile=True
