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

for seed in 43; do
    for rope_theta in 1e5 Inf; do
        for resume do_train num_eval in \
            False True 1024 \
            True False 10000 \
        ; do
        CUDA_VISIBLE_DEVICES=0 WANDB_MODE=online python run.py \
            --seed=$seed \
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
            --n_digits_train='1,17' \
            --op_train='add' \
            --format_train='backtrack' \
            --op_dist_train='1' \
            --n_digits_eval='4,33,4' \
            --op_eval='add' \
            --format_eval='reverse' \
            --op_dist_eval='1' \
            --show_task_ids=True \
            --padding_side='right' \
            \
            \
            --do_backtrack_decoding=True \
            --save_total_limit=1 \
            --resume_from_checkpoint=$resume \
            --run_name='backtrack' \
            --output_dir=out \
            --do_train=$do_train \
            --do_eval=True \
            --max_steps=10000 \
            --learning_rate=5e-4 \
            --lr_scheduler_type='warmup_stable_decay' \
            --lr_scheduler_kwargs='{"num_stable_steps": 6000, "num_decay_steps": 2000}' \
            --adam_beta2=0.98 \
            --adam_epsilon=1e-12 \
            --weight_decay=0.01 \
            --warmup_ratio=0.2 \
            --logging_steps=20 \
            --eval_strategy="steps" \
            --eval_steps=500 \
            --predict_with_generate \
            --remove_unused_columns=False \
            --eval_on_start=$resume \
            --per_device_train_batch_size=2048 \
            --per_device_eval_batch_size=1024 \
            --gradient_accumulation_steps=1 \
            --include_inputs_for_metrics=True \
            --save_steps=500 \
            --torch_compile=True \
            --bf16=True \
            --tf32=True 
        done
    done
done
