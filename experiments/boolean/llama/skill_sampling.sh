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

CUDA_VISIBLE_DEVICES=1 WANDB_PROJECT=mamba-arithmetic WANDB_MODE=online python run.py \
    --architecture=llama \
    --from_pretrained=False \
    --hidden_size=768 \
    --intermediate_size=1536 \
    --num_attention_heads=12 \
    --num_layers=12 \
    --max_position_embeddings=1024 \
    \
    \
    --num_train=20000000 \
    --num_eval=1000 \
    --n_digits_train='2,65,2 2,66,2 2,21,2' \
    --op_train='boolean boolean boolean' \
    --format_train='3sum parity 3parity' \
    --op_dist_train='1,1,1' \
    --n_digits_eval='8,97,8' \
    --op_eval='boolean boolean boolean' \
    --format_eval='3sum parity 3parity' \
    --op_dist_eval='1 1 1' \
    --show_task_ids=True \
    \
    \
    --resume_from_checkpoint='out/llama-768-12-12-1024-3sum_parity_3parity-digits-2_65_2_2_66_2_2_21_2/checkpoint-5500' \
    --run_name='test' \
    --output_dir=out \
    --do_train=True \
    --do_eval=True \
    --max_steps=15000 \
    --learning_rate=5e-4 \
    --lr_scheduler_type='warmup_stable_decay' \
    --lr_scheduler_kwargs='{"num_stable_steps": 12000, "num_decay_steps": 1500}' \
    --adam_beta2=0.98 \
    --adam_epsilon=1e-12 \
    --weight_decay=0.01 \
    --warmup_ratio=0.1 \
    --logging_steps=20 \
    --eval_strategy="steps" \
    --eval_steps=200 \
    --predict_with_generate \
    --remove_unused_columns=False \
    --eval_on_start=True \
    --per_device_train_batch_size=320 \
    --per_device_eval_batch_size=1024 \
    --gradient_accumulation_steps=6 \
    --include_inputs_for_metrics=True \
    --save_steps=500 \
    --torch_compile=True \
    --bf16=True \
    --tf32=True
