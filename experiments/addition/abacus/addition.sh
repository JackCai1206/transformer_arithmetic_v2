set -e
# Train
CUDA_VISIBLE_DEVICES=1 WANDB_PROJECT=mamba-arithmetic WANDB_MODE=online python run.py \
    --architecture=abacus \
    --hidden_size=1024 \
    --intermediate_size=2048 \
    --num_attention_heads=16 \
    --num_layers=16 \
    --max_position_embeddings=1024 \
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
    --run_name='test' \
    --output_dir=out \
    --do_train=True \
    --do_eval=True \
    --max_steps=10000 \
    --learning_rate=1e-3 \
    --lr_scheduler_type='cosine' \
    --warmup_steps=200 \
    --logging_steps=20 \
    --eval_strategy="steps" \
    --eval_steps=200 \
    --predict_with_generate \
    --per_device_train_batch_size=256 \
    --per_device_eval_batch_size=100 \
    --gradient_accumulation_steps=16 \
    --include_inputs_for_metrics=True \
    --tf32=True \
    --torch_compile=False

# Evaluate
# CUDA_VISIBLE_DEVICES=0 WANDB_PROJECT=mamba-arithmetic WANDB_MODE=online python run.py \
#     --architecture=abacus \
#     --hidden_size=384 \
#     --num_attention_heads=6 \
#     --num_layers=6 \
#     --max_position_embeddings=1024 \
#     \
#     \
#     --n_digits_eval_start=10 \
#     --n_digits_eval_end=128 \
#     --n_digits_eval_step=10 \
#     --op='add' \
#     --format='reverse' \
#     \
#     \
#     --resume_from_checkpoint='out/abacus-384-6-6-1024-add-digits-20/checkpoint-5000' \
#     --run_name='test' \
#     --output_dir=out \
#     --do_train=False \
#     --do_eval=True \
#     --max_steps=1 \
#     --learning_rate=5e-4 \
#     --lr_scheduler_type='cosine' \
#     --warmup_steps=200 \
#     --logging_steps=20 \
#     --eval_strategy="steps" \
#     --eval_steps=200 \
#     --predict_with_generate \
#     --per_device_train_batch_size=32 \
#     --per_device_eval_batch_size=64 \
#     --gradient_accumulation_steps=16 \
#     --include_inputs_for_metrics=True \
#     --tf32=True \
