set -e
# Train
CUDA_VISIBLE_DEVICES=0 WANDB_PROJECT=mamba-arithmetic WANDB_MODE=online python run.py \
    --architecture=CAT \
    --from_pretrained=False \
    --hidden_size=384 \
    --num_attention_heads=6 \
    --num_layers=6 \
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
    --learning_rate=5e-4 \
    --lr_scheduler_type='cosine' \
    --warmup_steps=200 \
    --logging_steps=20 \
    --eval_strategy="steps" \
    --eval_steps=200 \
    --predict_with_generate \
    --per_device_train_batch_size=512 \
    --per_device_eval_batch_size=100 \
    --gradient_accumulation_steps=16 \
    --include_inputs_for_metrics=True \
    --tf32=True \
    --torch_compile=False

# Evaluate
# CUDA_VISIBLE_DEVICES=0 WANDB_PROJECT=mamba-arithmetic WANDB_MODE=online python run.py \
#     --architecture=CAT \
#     --from_pretrained=False \
#     --hidden_size=384 \
#     --num_attention_heads=6 \
#     --num_layers=6 \
#     --max_position_embeddings=1024 \
#     \
#     \
#     --n_digits_train=20 \
#     --n_digits_train_min=3 \
#     --n_digits_eval_start=10 \
#     --n_digits_eval_end=128 \
#     --n_digits_eval_step=20 \
#     --op='add' \
#     --format='reverse' \
#     \
#     \
#     --run_name='test' \
#     --output_dir=out \
#     --do_train=True \
#     --do_eval=True \
#     --max_steps=10000 \
#     --learning_rate=5e-4 \
#     --lr_scheduler_type='cosine' \
#     --warmup_steps=200 \
#     --logging_steps=20 \
#     --eval_strategy="steps" \
#     --eval_steps=200 \
#     --predict_with_generate \
#     --per_device_train_batch_size=1024 \
#     --per_device_eval_batch_size=100 \
#     --gradient_accumulation_steps=8 \
#     --include_inputs_for_metrics=True \
#     --tf32=True \
#     --torch_compile=False
