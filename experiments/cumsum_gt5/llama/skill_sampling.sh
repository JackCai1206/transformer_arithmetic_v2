set -e

#    --freeze transformer.wpe transformer.h.0 transformer.h.1 transformer.h.2 transformer.h.3 \
# for from_pretrained use_lora max_steps eval_steps in \
#     True False 1000 100 \
#     True True 1000 100 \
#     False False 1000 100 \
# ; do
CUDA_VISIBLE_DEVICES=1 WANDB_PROJECT=mamba-arithmetic WANDB_MODE=online python run.py \
    --architecture=llama \
    --from_pretrained=False \
    --hidden_size=768 \
    --intermediate_size=1536 \
    --num_attention_heads=12 \
    --num_layers=12 \
    --max_position_embeddings=1024 \
    --rope_theta=1e3 \
    --partial_rotary_factor=0.125 \
    \
    \
    --num_train=20000000 \
    --num_eval=1000 \
    --n_digits_train='1,25 1,25 1,13' \
    --op_train='cumsum gt5 cumsum_gt5' \
    --format_train='None None None' \
    --op_dist_train='1,1,1' \
    --n_digits_eval='3,31,3' \
    --op_eval='cumsum gt5 cumsum_gt5' \
    --format_eval='None None None' \
    --op_dist_eval='1 1 1' \
    \
    \
    --run_name='test' \
    --output_dir=out \
    --do_train=True \
    --do_eval=True \
    --max_steps=10000 \
    --learning_rate=5e-4 \
    --lr_scheduler_type='warmup_stable_decay' \
    --lr_scheduler_kwargs='{"num_stable_steps": 2000, "num_decay_steps": 7000}' \
    --adam_beta2=0.98 \
    --adam_epsilon=1e-12 \
    --weight_decay=0.01 \
    --warmup_ratio=0.1 \
    --logging_steps=20 \
    --eval_strategy="steps" \
    --eval_steps=200 \
    --predict_with_generate \
    --remove_unused_columns=False \
    --eval_on_start=False \
    --per_device_train_batch_size=480 \
    --per_device_eval_batch_size=1024 \
    --gradient_accumulation_steps=2 \
    --include_inputs_for_metrics=True \
    --save_steps=500 \
    --torch_compile=True \
    --bf16=True \
    --tf32=True
