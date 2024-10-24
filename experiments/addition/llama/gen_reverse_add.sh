set -e

for aux_op          aux_format  in \
    pairwise        mod_add:0   \
    state_tracking  carry:10:9  \
; do
    for seed in 42 43 44; do
        for rope_theta in 1e5; do
            for resume do_train num_eval in \
                False True 1024 \
            ; do
            CUDA_VISIBLE_DEVICES=1 WANDB_PROJECT=LG-inherit WANDB_RUN_GROUP=aux-sweep WANDB_MODE=online python run.py \
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
                --n_digits_train='1,33 1,17' \
                --op_train=$aux_op' add' \
                --format_train=$aux_format' reverse' \
                --op_dist_train='1,1' \
                --n_digits_eval='4,37,4' \
                --op_eval=$aux_op' add' \
                --format_eval=$aux_format' reverse' \
                --op_dist_eval='1 1' \
                --show_task_ids=True \
                --disjoint_tokens=False \
                \
                \
                --track_num_tokens_seen_by_task=True \
                --early_stop=True \
                --save_total_limit=1 \
                --resume_from_checkpoint=$resume \
                --run_name='aux-sweep' \
                --output_dir=out \
                --do_train=$do_train \
                --do_eval=True \
                --max_steps=80000 \
                --learning_rate=1e-3 \
                --lr_scheduler_type='warmup_stable_decay' \
                --lr_scheduler_kwargs='{"num_stable_steps": 56000, "num_decay_steps": 16000, "min_lr_ratio": 0.1}' \
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
                --save_steps=500 \
                --torch_compile=True \
                --bf16=True \
                --tf32=True
            done
        done
    done
done