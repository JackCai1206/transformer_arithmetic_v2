set -e

CUDA_VISIBLE_DEVICES=0 WANDB_PROJECT=mamba-arithmetic WANDB_MODE=online python run.py \
    --architecture=abacus \
    --from_pretrained=False \
    --hidden_size=768 \
    --intermediate_size=1536 \
    --num_attention_heads=12 \
    --num_layers=12 \
    --max_position_embeddings=1024 \
    \
    \
    --num_train=20000000 \
    --num_eval=100 \
    --n_digits_train='1,21' \
    --op_train='add' \
    --format_train='reverse' \
    --op_dist_train='1' \
    --n_digits_eval='8,129,8' \
    --op_eval='add' \
    --format_eval='reverse' \
    --op_dist_eval='1' \
    \
    \
    --run_name='test' \
    --output_dir=out \
    --do_train=True \
    --do_eval=True \
    --max_steps=25000 \
    --learning_rate=5e-4 \
    --lr_scheduler_type='warmup_stable_decay' \
    --lr_scheduler_kwargs='{"num_stable_steps": 20000, "num_decay_steps": 2500}' \
    --adam_beta2=0.99 \
    --adam_epsilon=1e-12 \
    --weight_decay=0.01 \
    --warmup_ratio=0.1 \
    --logging_steps=20 \
    --eval_strategy="steps" \
    --eval_steps=200 \
    --predict_with_generate \
    --remove_unused_columns=False \
    --eval_on_start=True \
    --per_device_train_batch_size=320 \
    --per_device_eval_batch_size=1024 \
    --gradient_accumulation_steps=3 \
    --include_inputs_for_metrics=True \
    --save_steps=500 \
    --torch_compile=True \
    --bf16=False \
    --tf32=True
