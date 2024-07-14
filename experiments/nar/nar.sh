set -e

# --model_id='state-spaces/mamba-130m-hf' \
for from_pretrained in True False; do
    CUDA_VISIBLE_DEVICES=1 WANDB_PROJECT=mamba-arithmetic WANDB_MODE=online python run.py \
        --model_id='gpt2' \
        --from_pretrained=$from_pretrained \
        --use_lora=False \
        \
        \
        --n_digits_train=128 \
        --n_digits_train_min=64 \
        --n_digits_eval_start=128 \
        --n_digits_eval_end=256 \
        --n_digits_eval_step=32 \
        --op='nar' \
        --format='{"n":5}' \
        \
        \
        --run_name='test' \
        --output_dir=out \
        --do_train=True \
        --do_eval=True \
        --max_steps=5000 \
        --learning_rate=5e-4 \
        --lr_scheduler_type='cosine' \
        --warmup_steps=200 \
        --logging_steps=20 \
        --eval_strategy="steps" \
        --eval_steps=100 \
        --predict_with_generate \
        --per_device_train_batch_size=4 \
        --per_device_eval_batch_size=64 \
        --gradient_accumulation_steps=32 \
        --include_inputs_for_metrics=True
done
