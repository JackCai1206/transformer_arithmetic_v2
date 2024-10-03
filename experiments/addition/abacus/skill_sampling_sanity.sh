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

for seed in 42 43 44 45 46; do
    for rope_theta in Inf 1e5; do
        for resume do_train num_eval in \
            True True 512 \
            True False 10000 \
        ; do
        CUDA_VISIBLE_DEVICES=0,1 WANDB_MODE=online torchrun --nproc_per_node=2 run.py \
            --seed=$seed \
            --architecture=abacus \
            --from_pretrained=False \
            --hidden_size=1024 \
            --intermediate_size=2048 \
            --num_attention_heads=16 \
            --num_layers=16 \
            --max_position_embeddings=1024 \
            --rope_theta=$rope_theta \
            \
            \
            --num_train=20_000_000 \
            --num_eval=$num_eval \
            --n_digits_train='1,21' \
            --op_train='add' \
            --format_train='reverse' \
            --op_dist_train='1' \
            --n_digits_eval='10,101,10' \
            --op_eval='add' \
            --format_eval='reverse' \
            --op_dist_eval='1' \
            --show_task_ids=True \
            --padding_side='right' \
            --train_pad_to=75 \
            \
            \
            --save_total_limit=1 \
            --resume_from_checkpoint=$resume \
            --ignore_data_skip=True \
            --run_name='sanity' \
            --output_dir=out \
            --do_train=$do_train \
            --do_eval=True \
            --max_steps=40000 \
            --learning_rate=5e-4 \
            --lr_scheduler_type='warmup_stable_decay' \
            --lr_scheduler_kwargs='{"num_stable_steps": 32000, "num_decay_steps": 4000}' \
            --adam_beta2=0.98 \
            --adam_epsilon=1e-8 \
            --weight_decay=0.01 \
            --warmup_ratio=0.1 \
            --logging_steps=20 \
            --eval_strategy="steps" \
            --eval_steps=500 \
            --predict_with_generate \
            --remove_unused_columns=False \
            --eval_on_start=True \
            --per_device_train_batch_size=300 \
            --per_device_eval_batch_size=256 \
            --gradient_accumulation_steps=4 \
            --include_inputs_for_metrics=True \
            --save_steps=500 \
            --torch_compile=False \
            --bf16=True \
            --tf32=True
        done
    done
done
