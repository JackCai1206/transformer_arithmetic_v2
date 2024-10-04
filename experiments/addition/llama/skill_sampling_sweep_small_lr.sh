set -e
#     16          32          400         3          1024            \
    # 8           32          400         3          1024            \
    # 4           32          400         3          1024            \
    # 8           16          600         2          1024            \
    # 4           16          600         2          1024            \
    # 4           8           1200        1          1024            \

for train_low   train_high  batch_size  grad_acc   eval_batch_size in \
    8           16          512         1          1024            \
; do
    for seed in 42 43 44 45 46; do
        for rope_theta in 1e5; do
            for lr in 5e-3 2.5e-3 1e-3 7.5e-4 5e-4 2.5e-4 1e-4 5e-5 1e-5; do \
                for resume do_train num_eval in \
                    False True 1024 \
                    True False 10000 \
                ; do
                    CUDA_VISIBLE_DEVICES=0 WANDB_PROJECT=mamba-arithmetic WANDB_MODE=online python run.py \
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
                        --op_dist_train='1 1 1' \
                        --n_digits_eval=$((train_high/8))','$((train_high+train_high/4+1))','$((train_high/8)) \
                        --op_eval='add add add' \
                        --format_eval='reverse-no-carry reverse-carry-only reverse' \
                        --op_dist_eval='1 1 1' \
                        --show_task_ids=True \
                        --padding_side='left' \
                        --use_train_attention_mask=False \
                        \
                        \
                        --resume_from_checkpoint=$resume \
                        --save_total_limit=1 \
                        --run_name='sweep' \
                        --output_dir=out \
                        --do_train=$do_train \
                        --do_eval=True \
                        --max_steps=5000 \
                        --learning_rate=$lr \
                        --lr_scheduler_type='cosine' \
                        --adam_beta2=0.98 \
                        --adam_epsilon=1e-8 \
                        --weight_decay=0.01 \
                        --warmup_ratio=0.1 \
                        --logging_steps=20 \
                        --eval_strategy="steps" \
                        --eval_steps=250 \
                        --predict_with_generate \
                        --remove_unused_columns=False \
                        --eval_on_start=False \
                        --per_device_train_batch_size=$batch_size \
                        --per_device_eval_batch_size=$eval_batch_size \
                        --gradient_accumulation_steps=$grad_acc \
                        --include_inputs_for_metrics=True \
                        --save_steps=500 \
                        --torch_compile=True \
                        --bf16=False \
                        --tf32=True
                done
            done
        done
    done
done