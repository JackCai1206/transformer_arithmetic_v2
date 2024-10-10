set -e

for train_low in 16; do
    for train_high batch_size grad_acc in \
        32  2048 1  \
        64  1024 4  \
        128 512  8  \
        256 256  16 \
    ; do
        CUDA_VISIBLE_DEVICES=1 WANDB_PROJECT=mamba-arithmetic WANDB_MODE=online python run.py \
            --architecture=llama \
            --from_pretrained=False \
            --hidden_size=384 \
            --intermediate_size=1536 \
            --num_attention_heads=6 \
            --num_layers=6 \
            --max_position_embeddings=1024 \
            --rope_theta=1e3 \
            \
            \
            --num_train=20000000 \
            --num_eval=512 \
            --n_digits_train='1,'$((train_high+1))' 1,'$((train_high+1))' 1,'$((train_low+1)) \
            --op_train='rotate1 reverse rot1rev' \
            --format_train='None None None' \
            --op_dist_train='1,1,1' \
            --n_digits_eval=$((train_high/8))','$((train_high+train_high/4+1))','$((train_high/8)) \
            --op_eval='rotate1 reverse rot1rev' \
            --format_eval='None None None' \
            --op_dist_eval='1 1 1' \
            --show_task_ids=True \
            \
            \
            --run_name='test' \
            --output_dir=out \
            --do_train=True \
            --do_eval=True \
            --max_steps=5000 \
            --learning_rate=5e-4 \
            --lr_scheduler_type='warmup_stable_decay' \
            --lr_scheduler_kwargs='{"num_stable_steps": 4000, "num_decay_steps": 500}' \
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
            --per_device_eval_batch_size=1024 \
            --gradient_accumulation_steps=$grad_acc \
            --include_inputs_for_metrics=True \
            --save_steps=500 \
            --torch_compile=True \
            --bf16=True \
            --tf32=True
    done
done