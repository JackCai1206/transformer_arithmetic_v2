set -e

CUDA_VISIBLE_DEVICES=1 WANDB_PROJECT=mamba-arithmetic WANDB_MODE=online python run.py \
    --architecture=llama \\
    --from_pretrained=False \
    --hidden_size=768 \
    --intermediate_size=3072 \
    --num_attention_heads=12 \
    --num_layers=12 \
    --max_position_embeddings=1024 \
    \
    \
    --num_train=20000000 \
    --num_eval=300 \
    --n_digits_train='1,64 1,64 1,32' \
    --op_train='add add add' \
    --format_train='reverse-no-carry reverse-carry-only reverse' \
    --op_dist_train='1 1 1' \
    --n_digits_eval='8,65,8' \
    --op_eval='add add add' \
    --format_eval='reverse-no-carry reverse-carry-only reverse' \
    --op_dist_eval='1 1 1' \
    \
    \
    --run_name='test' \
    --resume_from_checkpoint='out/llama-768-12-12-1024-reverse-no-carry_reverse-carry-only_reverse-digits-1_64_1_64_1_32/checkpoint-4800' \
    --output_dir=out \
    --do_train=False \
    --do_eval=True \
    --max_steps=5000 \
    --learning_rate=5e-4 \
    --lr_scheduler_type='cosine' \
    --warmup_ratio=0.05 \
    --logging_steps=20 \
    --eval_strategy="steps" \
    --eval_steps=100 \
    --predict_with_generate \
    --remove_unused_columns=False \
    --eval_on_start=True \
    --per_device_train_batch_size=64 \
    --per_device_eval_batch_size=1024 \
    --gradient_accumulation_steps=32 \
    --include_inputs_for_metrics=True \
    --save_steps=200 \
    --torch_compile=True \
    --bf16=False \
    --tf32=True
