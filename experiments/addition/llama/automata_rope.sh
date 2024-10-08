set -e

for seed in 42 43 44 45 46; do
    for rope_theta in 1e5 Inf; do
        for resume do_train num_eval in \
            False True 1024 \
            True False 10000 \
        ; do
        CUDA_VISIBLE_DEVICES=0 WANDB_PROJECT=LG-inherit WANDB_RUN_GROUP=automata WANDB_MODE=online python run.py \
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
            --num_eval=1000 \
            --n_digits_train='1,17 1,33 1,9' \
            --op_train='add add add' \
            --format_train='automata_A automata_B automata_C' \
            --op_dist_train='1 1 1' \
            --n_digits_eval='4,49,4' \
            --op_eval='add add add' \
            --format_eval='automata_A automata_B automata_C' \
            --op_dist_eval='1 1 1' \
            --show_task_ids=True \
            --padding_side='random' \
            \
            \
            --save_total_limit=1 \
            --resume_from_checkpoint=$resume \
            --run_name='base' \
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
            --eval_on_start=$resume \
            --per_device_train_batch_size=1024 \
            --per_device_eval_batch_size=$num_eval \
            --gradient_accumulation_steps=1 \
            --include_inputs_for_metrics=True \
            --save_steps=500 \
            --torch_compile=True \
            --bf16=True \
            --tf32=True
        done
    done
done


# CUDA_VISIBLE_DEVICES=1 WANDB_PROJECT=mamba-arithmetic WANDB_MODE=online python run.py \
#     --architecture=llama \
#     --from_pretrained=False \
#     --hidden_size=768 \
#     --intermediate_size=1536 \
#     --num_attention_heads=12 \
#     --num_layers=12 \
#     --max_position_embeddings=1024 \
#     --rope_theta=1e5 \
#     \
#     \

#     \
#     \
#     --run_name='test' \
#     --output_dir=out \
#     --do_train=True \
#     --do_eval=True \
#     --max_steps=10000 \
#     --lr_scheduler_type='warmup_stable_decay' \
#     --lr_scheduler_kwargs='{"num_stable_steps": 8000, "num_decay_steps": 1000}' \
#     --adam_beta2=0.99 \
#     --weight_decay=0.00 \
#     --max_grad_norm=1 \
#     --warmup_ratio=0.1 \
#     --logging_steps=20 \
#     --eval_strategy="steps" \
#     --eval_steps=100 \
#     --predict_with_generate \
#     --remove_unused_columns=False \
#     --eval_on_start=True \
#     --per_device_train_batch_size=240 \
#     --per_device_eval_batch_size=1024 \
#     --gradient_accumulation_steps=6 \
#     --include_inputs_for_metrics=True \
#     --save_steps=500 \
#     --torch_compile=True \
#     --bf16=True \
#     --tf32=True

