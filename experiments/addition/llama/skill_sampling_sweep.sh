set -e
    # 16  800 2  \
    # 32  400 1024 2  \
    # 64  200 512  4  \
    # 128 100 256  8  \
    # 256 50  128  16 \

for train_low in 8 6 4; do
    for train_high batch_size eval_batch_size grad_acc in \
        16  800 1024 2  \
    ; do
        for rope_theta in 1e5 Inf; do
            for do_train num_eval in \
                True 128 \
                False 10000 \
            ; do
                CUDA_VISIBLE_DEVICES=1 WANDB_PROJECT=mamba-arithmetic WANDB_MODE=online python run.py \
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
                    --op_dist_train='1 1 1' \
                    --n_digits_eval=$((train_high/8))','$((train_high+train_high/4+1))','$((train_high/8)) \
                    --op_eval='add add add' \
                    --format_eval='reverse-no-carry reverse-carry-only reverse' \
                    --op_dist_eval='1 1 1' \
                    --show_task_ids=True \
                    \
                    \
                    --resume_from_checkpoint=True \
                    --save_total_limit=1 \
                    --run_name='test' \
                    --output_dir=out \
                    --do_train=$do_train \
                    --do_eval=True \
                    --max_steps=5000 \
                    --learning_rate=5e-4 \
                    --lr_scheduler_type='warmup_stable_decay' \
                    --lr_scheduler_kwargs='{"num_stable_steps": 3000, "num_decay_steps": 2000}' \
                    --adam_beta2=0.98 \
                    --adam_epsilon=1e-12 \
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
                    --bf16=True \
                    --tf32=True
            done
        done
    done
done