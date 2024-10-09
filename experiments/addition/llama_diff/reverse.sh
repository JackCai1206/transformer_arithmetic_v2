set -e

for seed in 42; do
    for rope_theta in 1e5 Inf; do
        for resume do_train num_eval in \
            False True 1024 \
            True False 10000 \
        ; do
        CUDA_VISIBLE_DEVICES=1 WANDB_PROJECT=LG-inherit WANDB_MODE=online python run.py \
            --seed=$seed \
            --architecture=llama-diff \
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
            --n_digits_train='1,17' \
            --op_train='add' \
            --format_train='reverse' \
            --op_dist_train='1' \
            --n_digits_eval='4,25,4' \
            --op_eval='add' \
            --format_eval='reverse' \
            --op_dist_eval='1' \
            --show_task_ids=True \
            \
            \
            --save_total_limit=1 \
            --resume_from_checkpoint=$resume \
            --run_name='small' \
            --output_dir=out \
            --do_train=$do_train \
            --do_eval=True \
            --max_steps=15000 \
            --learning_rate=1e-3 \
            --lr_scheduler_type='cosine' \
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
            --per_device_train_batch_size=1024 \
            --per_device_eval_batch_size=1024 \
            --gradient_accumulation_steps=1 \
            --include_inputs_for_metrics=True \
            --save_steps=500 \
            --torch_compile=True \
            --bf16=False \
            --tf32=True
        done
    done
done
