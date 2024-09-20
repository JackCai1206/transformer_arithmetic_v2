set -e
    # --learning_rate=5e-4 \
    # --lr_scheduler_type='warmup_stable_decay' \
    # --lr_scheduler_kwargs='{"num_stable_steps": 20000, "num_decay_steps": 2500}' \
    # --adam_beta1=0.9 \
    # --adam_beta2=0.98 \
    # --adam_epsilon=1e-12 \
    # --weight_decay=0.01 \
    # --max_grad_norm=1 \
    # --warmup_ratio=0.1 \

for rope_theta in 1e5; do
    for do_train num_eval in \
        True 1024 \
        False 10000 \
    ; do
    CUDA_VISIBLE_DEVICES=0 WANDB_PROJECT=LG-inherit WANDB_MODE=disabled python run.py \
        --architecture=llama \
        --from_pretrained=False \
        --hidden_size=384 \
        --intermediate_size=1536 \
        --num_attention_heads=6 \
        --num_layers=6 \
        --max_position_embeddings=1024 \
        --rope_theta=$rope_theta \
        \
        \
        --num_train=20000000 \
        --num_eval=$num_eval \
        --n_digits_train='1,33 1,33 1,17' \
        --op_train='add add add' \
        --format_train='reverse-no-carry reverse-carry-only reverse' \
        --op_dist_train='1 1 1' \
        --n_digits_eval='4,49,4' \
        --op_eval='add add add' \
        --format_eval='reverse-no-carry reverse-carry-only reverse' \
        --op_dist_eval='1 1 1' \
        --show_task_ids=True \
        \
        \
        --save_total_limit=1 \
        --run_name='small' \
        --output_dir=out \
        --do_train=$do_train \
        --do_eval=True \
        --max_steps=10000 \
        --learning_rate=5e-4 \
        --lr_scheduler_type='warmup_stable_decay' \
        --lr_scheduler_kwargs='{"num_stable_steps": 8000, "num_decay_steps": 1000}' \
        --adam_beta2=0.98 \
        --adam_epsilon=1e-12 \
        --weight_decay=0.01 \
        --warmup_ratio=0.1 \
        --logging_steps=20 \
        --eval_strategy="steps" \
        --eval_steps=200 \
        --predict_with_generate \
        --remove_unused_columns=False \
        --eval_on_start=False \
        --per_device_train_batch_size=512 \
        --per_device_eval_batch_size=1024 \
        --gradient_accumulation_steps=4 \
        --include_inputs_for_metrics=True \
        --save_steps=500 \
        --torch_compile=True \
        --bf16=False \
        --tf32=True
    done
done
