set -e
#     16          32          400         3          1024            \
    # 8           32          400         3          1024            \
    # 4           32          400         3          1024            \
    # 8           16          600         2          1024            \
    # 4           16          600         2          1024            \
    # 4           8           1200        1          1024            \

for train_low   train_high  batch_size  grad_acc   eval_batch_size in \
    10          100         512         2          1024            \
    20          100         512         2          1024            \
    50          100         512         2          1024            \
    25          50          1024        1          1024            \
    10          50          1024        1          1024            \
    5           50          1024        1          1024            \
; do
    for seed in 42 43 44; do
        for rope_theta in 1e5; do
            for resume do_train num_eval in \
                False True 1024 \
                True False 10000 \
            ; do
                CUDA_VISIBLE_DEVICES=1 WANDB_PROJECT=LG-inherit WANDB_RUN_GROUP=sweep-small WANDB_MODE=online python run.py \
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
                    --n_digits_train='1,'$((train_high+1))' 1,'$((train_high+1))' 1,'$((train_low+1)) \
                    --op_train='add add add' \
                    --format_train='reverse-no-carry reverse-carry-only reverse' \
                    --op_dist_train='1,1,1' \
                    --n_digits_eval=$((train_high/8))','$((train_high+train_high/6+1))','$((train_high/8)) \
                    --op_eval='add add add' \
                    --format_eval='reverse-no-carry reverse-carry-only reverse' \
                    --op_dist_eval='1 1 1' \
                    --show_task_ids=True \
                    --padding_side='random' \
                    --use_train_attention_mask=True \
                    \
                    \
                    --resume_from_checkpoint=$resume \
                    --save_total_limit=1 \
                    --run_name='sweep' \
                    --output_dir=out \
                    --do_train=$do_train \
                    --do_eval=True \
                    --max_steps=20000 \
                    --learning_rate=1e-3 \
                    --lr_scheduler_type='warmup_stable_decay' \
                    --lr_scheduler_kwargs='{"num_stable_steps": 10000, "num_decay_steps": 8000}' \
                    --adam_beta2=0.98 \
                    --adam_epsilon=1e-8 \
                    --weight_decay=0.01 \
                    --warmup_ratio=0.1 \
                    --logging_steps=20 \
                    --eval_strategy="steps" \
                    --eval_steps=500 \
                    --predict_with_generate \
                    --remove_unused_columns=False \
                    --eval_on_start=False \
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