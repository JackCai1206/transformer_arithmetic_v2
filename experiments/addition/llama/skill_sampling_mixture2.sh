set -e

export NCCL_P2P_DISABLE="1"
export NCCL_IB_DISABLE="1"

for seed in 42 43 44 45 46; do
    for rope_theta in 1e5; do
        for resume do_train num_eval in \
            False True 1024 \
        ; do
        CUDA_VISIBLE_DEVICES=0 WANDB_PROJECT=LG-inherit WANDB_RUN_GROUP=mixture WANDB_MODE=online python run.py \
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
            --n_digits_train='1,65 1,65 1,17' \
            --op_train='add add add' \
            --format_train='reverse-no-carry reverse-carry-only reverse' \
            --op_dist_train='1,1,1' \
            --n_digits_eval='8,73,8' \
            --op_eval='add add add' \
            --format_eval='reverse-no-carry reverse-carry-only reverse' \
            --op_dist_eval='1 1 1' \
            --show_task_ids=True \
            --mixture_scheduling_kwargs='{"schedule": "cosine", "wait_before": 0.0, "wait_after": 0}' \
            \
            \
            --track_num_tokens_seen_by_task=True \
            --early_stop=True \
            --save_total_limit=1 \
            --resume_from_checkpoint=$resume \
            --run_name='mixture-early-stop' \
            --output_dir=out \
            --do_train=$do_train \
            --do_eval=True \
            --max_steps=100000 \
            --learning_rate=7.5e-4 \
            --lr_scheduler_type='warmup_stable_decay' \
            --lr_scheduler_kwargs='{"num_stable_steps": 40000, "num_decay_steps": 50000, "min_lr_ratio": 0.01}' \
            --adam_beta2=0.98 \
            --adam_epsilon=1e-12 \
            --weight_decay=0.01 \
            --warmup_ratio=0.1 \
            --logging_steps=20 \
            --eval_strategy="steps" \
            --eval_steps=500 \
            --predict_with_generate \
            --remove_unused_columns=False \
            --eval_on_start=False \
            --per_device_train_batch_size=1024 \
            --per_device_eval_batch_size=1024 \
            --gradient_accumulation_steps=1 \
            --include_inputs_for_metrics=True \
            --save_steps=1000 \
            --torch_compile=True \
            --bf16=True \
            --tf32=True
        done
    done
done
