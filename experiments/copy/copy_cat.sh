set -e

# --model_id='state-spaces/mamba-130m-hf' \
for architecture in 'CAT'; do
    CUDA_VISIBLE_DEVICES=0 WANDB_PROJECT=mamba-arithmetic WANDB_MODE=online python run.py \
        --architecture=$architecture \
        --hidden_size=384 \
        --num_attention_heads=6 \
        --num_layers=4 \
        --max_position_embeddings=1024 \
        \
        \
        --n_digits_train=128 \
        --n_digits_train_min=32 \
        --n_digits_eval_start=32 \
        --n_digits_eval_end=256 \
        --n_digits_eval_step=32 \
        --op='copy' \
        --format='' \
        \
        \
        --run_name='test' \
        --output_dir=out \
        --do_train=True \
        --do_eval=True \
        --max_steps=5000 \
        --learning_rate=5e-4 \
        --lr_scheduler_type='cosine' \
        --warmup_ratio=0.05 \
        --logging_steps=20 \
        --eval_strategy="steps" \
        --eval_steps=200 \
        --predict_with_generate \
        --per_device_train_batch_size=32 \
        --per_device_eval_batch_size=64 \
        --gradient_accumulation_steps=4 \
        --include_inputs_for_metrics=True \
        --save_steps=200
done
