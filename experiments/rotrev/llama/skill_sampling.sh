set -e

CUDA_VISIBLE_DEVICES=1 WANDB_PROJECT=mamba-arithmetic WANDB_MODE=online python run.py \
    --architecture=llama \
    --from_pretrained=False \
    --hidden_size=768 \
    --intermediate_size=1536 \
    --num_attention_heads=12 \
    --num_layers=12 \
    --max_position_embeddings=1024 \
    \
    \
    --num_train=20000000 \
    --num_eval=1000 \
    --n_digits_train='1,33 1,33 1,17' \
    --op_train='rotate1 reverse rot1rev' \
    --format_train='None None None' \
    --op_dist_train='1 1 1' \
    --n_digits_eval='4,33,4' \
    --op_eval='rotate1 reverse rot1rev' \
    --format_eval='None None None' \
    --op_dist_eval='1 1 1' \
    \
    \
    --run_name='test' \
    --output_dir=out \
    --do_train=True \
    --do_eval=True \
    --max_steps=5000 \
    --learning_rate=1e-3 \
    --lr_scheduler_type='cosine_with_min_lr' \
    --lr_scheduler_kwargs='{"min_lr": 1e-5}' \
    --adam_beta2=0.98 \
    --warmup_ratio=0.05 \
    --logging_steps=20 \
    --eval_strategy="steps" \
    --eval_steps=$eval_steps \
    --predict_with_generate \
    --remove_unused_columns=False \
    --eval_on_start=True \
    --per_device_train_batch_size=480 \
    --per_device_eval_batch_size=1024 \
    --gradient_accumulation_steps=4 \
    --include_inputs_for_metrics=True \
    --save_steps=500 \
    --torch_compile=True \
    --bf16=False \
    --tf32=True
