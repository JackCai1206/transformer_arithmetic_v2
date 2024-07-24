set -e

#    --freeze transformer.wpe transformer.h.0 transformer.h.1 transformer.h.2 transformer.h.3 \
CUDA_VISIBLE_DEVICES=1 WANDB_PROJECT=mamba-arithmetic WANDB_MODE=online python run.py \
    --model_id='EleutherAI/pythia-160m' \
    --from_pretrained=True \
    \
    \
    --num_train=2000000 \
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
    --learning_rate=5e-5 \
    --lr_scheduler_type='cosine' \
    --warmup_steps=200 \
    --logging_steps=20 \
    --eval_strategy="steps" \
    --eval_steps=200 \
    --predict_with_generate \
    --per_device_train_batch_size=10 \
    --per_device_eval_batch_size=64 \
    --gradient_accumulation_steps=16 \
    --include_inputs_for_metrics=True \
    --save_steps=200 \
    --torch_compile=True \
    --bf16=False \
    --tf32=True \
