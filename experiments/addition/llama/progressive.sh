set -e

for train_low train_high in \
    1 10 \
;do
    CUDA_VISIBLE_DEVICES=1 WANDB_PROJECT=mamba-arithmetic WANDB_MODE=online python run.py \
        --architecture=llama \
        --from_pretrained=False \
        --hidden_size=384 \
        --intermediate_size=1536 \
        --num_attention_heads=6 \
        --num_layers=6 \
        --max_position_embeddings=1024 \
        \
        \
        --num_train=20000000 \
        --num_eval=1000 \
        --n_digits_train=$train_low','$((train_high+1)) \
        --op_train='add' \
        --format_train='reverse' \
        --op_dist_train='1' \
        --n_digits_eval='1,48,1' \
        --op_eval='add' \
        --format_eval='reverse' \
        --op_dist_eval='1' \
        \
        \
        --save_total_limit=1 \
        --metric_for_best_model='eval_'$train_high'-add-reverse_accuracy' \
        --run_name='test' \
        --output_dir=out \
        --do_train=True \
        --do_eval=True \
        --max_steps=10000 \
        --learning_rate=5e-4 \
        --lr_scheduler_type='warmup_stable_decay' \
        --lr_scheduler_kwargs='{"num_stable_steps": 8000, "num_decay_steps": 1000}' \
        --adam_beta2=0.98 \
        --adam_epsilon=1e-8 \
        --weight_decay=0.01 \
        --warmup_ratio=0.1 \
        --logging_steps=20 \
        --eval_strategy="steps" \
        --eval_steps=1000 \
        --predict_with_generate \
        --remove_unused_columns=False \
        --eval_on_start=True \
        --per_device_train_batch_size=2048 \
        --per_device_eval_batch_size=1024 \
        --gradient_accumulation_steps=1 \
        --include_inputs_for_metrics=True \
        --save_steps=100 \
        --torch_compile=True \
        --bf16=False \
        --tf32=True
done

# for train_low train_high in \
#     2 11 \
#     3 12 \
#     4 13 \
#     5 14 \
#     6 15 \
#     7 16 \
#     8 17 \
#     9 18 \
#     10 19 \
#     11 20 \
#     12 21 \
#     13 22 \
#     14 23 \
#     15 24 \
#     16 25 \
#     17 26 \
#     18 27 \
#     19 28 \
#     20 29 \
#     21 30 \
#     22 31 \
#     23 32 \
#     24 33 \
#     25 34 \
#     26 35 \
#     27 36 \
#     28 37 \
#     29 38 \
#     30 39 \
#     31 40 \
#     32 41 \
#     33 42 \
#     34 43 \
#     35 44 \
#     36 45 \
#     37 46 \
#     38 47 \
#     39 48 \
# ;do
#     CUDA_VISIBLE_DEVICES=1 WANDB_PROJECT=mamba-arithmetic WANDB_MODE=online python run.py \
#         --architecture=llama \
#         --from_pretrained=False \
#         --hidden_size=384 \
#         --intermediate_size=1536 \
#         --num_attention_heads=6 \
#         --num_layers=6 \
#         --max_position_embeddings=1024 \
#         \
#         \
#         --num_train=20000000 \
#         --num_eval=1000 \
#         --n_digits_train=$train_low','$((train_high+1)) \
#         --op_train='add' \
#         --format_train='reverse' \
#         --op_dist_train='1' \
#         --n_digits_eval='1,48,1' \
#         --op_eval='add' \
#         --format_eval='reverse' \
#         --op_dist_eval='1' \
#         \
#         \
#         --ignore_data_skip=True \
#         --resume_from_checkpoint='out/llama-384-6-6-1024-reverse-digits-'$((train_low-1))'_'$train_high'_' \
#         --save_total_limit=1 \
#         --metric_for_best_model='eval_'$train_high'-add-reverse_accuracy' \
#         --run_name='test' \
#         --output_dir=out \
#         --do_train=True \
#         --do_eval=True \
#         --max_steps=10000 \
#         --learning_rate=5e-4 \
#         --lr_scheduler_type='warmup_stable_decay' \
#         --lr_scheduler_kwargs='{"num_stable_steps": 8000, "num_decay_steps": 1000}' \
#         --adam_beta2=0.98 \
#         --adam_epsilon=1e-7 \
#         --weight_decay=0.01 \
#         --warmup_ratio=0.1 \
#         --logging_steps=20 \
#         --eval_strategy="steps" \
#         --eval_steps=100 \
#         --predict_with_generate \
#         --remove_unused_columns=False \
#         --eval_on_start=True \
#         --per_device_train_batch_size=4096 \
#         --per_device_eval_batch_size=1024 \
#         --gradient_accumulation_steps=1 \
#         --include_inputs_for_metrics=True \
#         --save_steps=100 \
#         --torch_compile=True \
#         --bf16=True \
#         --tf32=True
# done
