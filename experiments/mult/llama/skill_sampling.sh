set -e

#    --freeze transformer.wpe transformer.h.0 transformer.h.1 transformer.h.2 transformer.h.3 \
# for from_pretrained use_lora max_steps eval_steps in \
#     True False 1000 100 \
#     True True 1000 100 \
#     False False 1000 100 \
# ; do

for seed in 42 43 44; do
    for rope_theta in Inf 1e5; do
        for resume do_train num_eval in \
            False True 1024 \
            True False 10000 \
        ; do
        CUDA_VISIBLE_DEVICES=0 WANDB_MODE=online WANDB_RUN_GROUP=mult python run.py \
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
            --n_digits_train='1,17 1,9 1,17' \
            --op_train='mult mult add' \
            --format_train='sd_mult mult reverse' \
            --op_dist_train='1 1 1' \
            --n_digits_eval='4,17,2' \
            --op_eval='mult mult add' \
            --format_eval='sd_mult mult reverse' \
            --op_dist_eval='1 1 1' \
            --show_task_ids=True \
            --padding_side='right' \
            \
            \
            --save_total_limit=1 \
            --resume_from_checkpoint=$resume \
            --run_name='mult' \
            --output_dir=out \
            --do_train=$do_train \
            --do_eval=True \
            --max_steps=20000 \
            --learning_rate=5e-4 \
            --lr_scheduler_type='warmup_stable_decay' \
            --lr_scheduler_kwargs='{"num_stable_steps": 12000, "num_decay_steps": 4000}' \
            --adam_beta2=0.98 \
            --adam_epsilon=1e-12 \
            --weight_decay=0.01 \
            --warmup_ratio=0.2 \
            --logging_steps=20 \
            --eval_strategy="steps" \
            --eval_steps=200 \
            --predict_with_generate \
            --remove_unused_columns=False \
            --eval_on_start=False \
            --per_device_train_batch_size=400 \
            --per_device_eval_batch_size=1024 \
            --gradient_accumulation_steps=3 \
            --include_inputs_for_metrics=True \
            --save_steps=500 \
            --torch_compile=True \
            --bf16=True \
            --tf32=True
        done
    done
done
