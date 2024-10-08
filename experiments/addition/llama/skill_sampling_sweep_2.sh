set -e

    # 15           32          400         3          1024            \
    # 14           32          400         3          1024            \
    # 13           32          400         3          1024            \
    # 12           32          400         3          1024            \
    # 11           32          400         3          1024            \
    # 10           32          400         3          1024            \

    # 16          32          400         3          1024            \

for train_low   train_high  batch_size  grad_acc   eval_batch_size in \
    16           64          300         4          512            \
    16           128         200         6          256            \
; do
    for seed in 42 43 44; do
        for rope_theta in 1e5; do
            for resume do_train num_eval in \
                False True 1024 \
                True False 10000 \
            ; do
                CUDA_VISIBLE_DEVICES=1 WANDB_PROJECT=LG-inherit WANDB_RUN_GROUP=sweep-length WANDB_MODE=online python run.py \
                    --seed=$seed \
                    --architecture=llama \
                    --from_pretrained=False \
                    --hidden_size=768 \
                    --intermediate_size=1536 \
                    --num_attention_heads=12 \
                    --num_layers=12 \
                    --max_position_embeddings=1024 \
                    --rope_theta=$rope_theta \
                    \
                    \
                    --num_train=20000000 \
                    --num_eval=$num_eval \
                    --n_digits_train='1,'$((train_high+1))' 1,'$((train_high+1))' 1,'$((train_low+1)) \
                    --op_train='add add add' \
                    --format_train='reverse-no-carry reverse-carry-only reverse' \
                    --op_dist_train='1,1,1' \
                    --n_digits_eval=$((train_high/8))','$((train_high+train_high/4+1))','$((train_high/8)) \
                    --op_eval='add add add' \
                    --format_eval='reverse-no-carry reverse-carry-only reverse' \
                    --op_dist_eval='1 1 1' \
                    --show_task_ids=True \
                    --padding_side='right' \
                    \
                    \
                    --resume_from_checkpoint=$resume \
                    --save_total_limit=1 \
                    --run_name='sweep' \
                    --output_dir=out \
                    --do_train=$do_train \
                    --do_eval=True \
                    --max_steps=15000 \
                    --learning_rate=2.5e-4 \
                    --lr_scheduler_type='warmup_stable_decay' \
                    --lr_scheduler_kwargs='{"num_stable_steps": 11500, "num_decay_steps": 2000}' \
                    --adam_beta2=0.98 \
                    --adam_epsilon=1e-12 \
                    --weight_decay=0.01 \
                    --warmup_ratio=0.21 \
                    --logging_steps=20 \
                    --eval_strategy="steps" \
                    --eval_steps=250 \
                    --predict_with_generate \
                    --remove_unused_columns=False \
                    --eval_on_start=$resume \
                    --per_device_train_batch_size=$batch_size \
                    --per_device_eval_batch_size=$eval_batch_size \
                    --gradient_accumulation_steps=$grad_acc \
                    --include_inputs_for_metrics=True \
                    --save_steps=500 \
                    --torch_compile=True \
                    --bf16=True \
                    --tf32=True
            done
        done
    done
done