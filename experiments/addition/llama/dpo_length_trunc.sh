set -e

#    --freeze transformer.wpe transformer.h.0 transformer.h.1 transformer.h.2 transformer.h.3 \
# for from_pretrained use_lora max_steps eval_steps in \
#     True False 1000 100 \
#     True True 1000 100 \
#     False False 1000 100 \
# ; do
for from_pretrained use_lora max_steps eval_steps in \
    False False 10000 100 \
;do
    CUDA_VISIBLE_DEVICES=0 WANDB_PROJECT=mamba-arithmetic WANDB_MODE=online python run.py \
        --architecture=llama \
        --from_pretrained=False \
        --hidden_size=768 \
        --intermediate_size=3072 \
        --num_attention_heads=12 \
        --num_layers=12 \
        --max_position_embeddings=1024 \
        \
        \
        --num_train=20000000 \
        --num_eval=100 \
        --n_digits_train='1,17' \
        --op_train='add' \
        --format_train='reverse' \
        --op_dist_train='1' \
        --n_digits_eval='4,33,4' \
        --op_eval='add' \
        --format_eval='reverse' \
        --op_dist_eval='1' \
        \
        \
        --run_name='test' \
        --output_dir=out \
        --resume_from_checkpoint='out/llama-768-12-12-1024-reverse-digits-1_17_/checkpoint-1000' \
        --do_train=False \
        --do_dpo=True \
        --do_eval=True \
        --max_steps=5000 \
        --learning_rate=1e-4 \
        --lr_scheduler_type='cosine_with_min_lr' \
        --lr_scheduler_kwargs='{"min_lr": 1e-5}' \
        --adam_beta2=0.99 \
        --warmup_ratio=0.05 \
        --logging_steps=20 \
        --eval_strategy="steps" \
        --eval_steps=100 \
        --predict_with_generate \
        --remove_unused_columns=False \
        --eval_on_start=True \
        --per_device_train_batch_size=160 \
        --per_device_eval_batch_size=1024 \
        --gradient_accumulation_steps=12 \
        --include_inputs_for_metrics=True \
        --save_steps=500 \
        --torch_compile=False \
        --bf16=True \
        --tf32=True
done
