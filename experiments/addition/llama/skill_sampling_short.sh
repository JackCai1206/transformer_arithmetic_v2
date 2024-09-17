set -e

for train_high in 256 128 64 32; do
    CUDA_VISIBLE_DEVICES=0 WANDB_PROJECT=mamba-arithmetic WANDB_MODE=online python run.py \
        --architecture=llama \
        --from_pretrained=False \
        --hidden_size=768 \
        --intermediate_size=1536 \
        --num_attention_heads=12 \
        --num_layers=12 \
        --max_position_embeddings=1024 \
        --rope_theta=1e5 \
        \
        \
        --num_train=20000000 \
        --num_eval=128 \
        --n_digits_train='1,'$((train_high+1))' 1,'$((train_high+1))' 1,9' \
        --op_train='add add add' \
        --format_train='reverse-no-carry reverse-carry-only reverse' \
        --op_dist_train='1 1 1' \
        --n_digits_eval=$((train_high/8))','$((train_high+train_high/4+1))','$((train_high/8)) \
        --op_eval='add add add' \
        --format_eval='reverse-no-carry reverse-carry-only reverse' \
        --op_dist_eval='1 1 1' \
        --show_task_ids=True \
        \
        \
        --run_name='test' \
        --output_dir=out \
        --do_train=True \
        --do_eval=True \
        --max_steps=15000 \
        --learning_rate=5e-4 \
        --lr_scheduler_type='warmup_stable_decay' \
        --lr_scheduler_kwargs='{"num_stable_steps": 12000, "num_decay_steps": 1500}' \
        --adam_beta2=0.98 \
        --adam_epsilon=1e-12 \
        --weight_decay=0.01 \
        --warmup_ratio=0.1 \
        --logging_steps=20 \
        --eval_strategy="steps" \
        --eval_steps=1000 \
        --predict_with_generate \
        --remove_unused_columns=False \
        --eval_on_start=False \
        --per_device_train_batch_size=40 \
        --per_device_eval_batch_size=64 \
        --gradient_accumulation_steps=8 \
        --include_inputs_for_metrics=True \
        --save_steps=500 \
        --torch_compile=True \
        --bf16=True \
        --tf32=True
done