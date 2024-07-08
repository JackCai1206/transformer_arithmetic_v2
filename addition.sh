# --model_id='state-spaces/mamba-130m-hf' \
CUDA_VISIBLE_DEVICES=1 WANDB_PROJECT=mamba-arithmetic WANDB_MODE=online python run.py \
    --from_pretrained=False \
    --use_lora=False \
    --state_size=100 \
    --num_layers=4 \
\
\
    --n_digits_train=20 \
    --op='+' \
    --format='reverse' \
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
    --eval_steps=200 \
    --predict_with_generate \
    --per_device_train_batch_size=64 \
    --per_device_eval_batch_size=64 \
    --gradient_accumulation_steps=4 \
    --include_inputs_for_metrics=True \
