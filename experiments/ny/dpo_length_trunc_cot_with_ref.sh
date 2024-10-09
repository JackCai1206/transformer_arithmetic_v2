set -e

#    --freeze transformer.wpe transformer.h.0 transformer.h.1 transformer.h.2 transformer.h.3 \
# for from_pretrained use_lora max_steps eval_steps in \
#     True False 1000 100 \
#     True True 1000 100 \
#     False False 1000 100 \
# ; do

# --resume_from_checkpoint='out/llama-768-12-12-1024-COT-digits-1_17_/checkpoint-1000' \
# --n_digits_eval='2,17,2' \

# CUDA_VISIBLE_DEVICES=1 WANDB_PROJECT=arithmetic-dpo WANDB_MODE=online python run.py \
#     --ref_model=True \
#     --ref_model_path='out/test-llama-384-6-6-1024-COT-digits-1_10_/checkpoint-5000' \
#     \
#     \
#     --architecture=llama \
#     --from_pretrained=False \
#     --hidden_size=384 \
#     --intermediate_size=3072 \
#     --num_attention_heads=6 \
#     --num_layers=6 \
#     --max_position_embeddings=1024 \
#     \
#     \
#     --num_train=20000000 \
#     --num_eval=1000 \
#     --n_digits_train='1,10' \
#     --op_train='add' \
#     --format_train='COT' \
#     --op_dist_train='1' \
#     --n_digits_eval='2,17,2' \
#     --op_eval='add' \
#     --format_eval='COT' \
#     --op_dist_eval='1' \
#     \
#     \
#     --run_name='dpo_ref' \
#     --output_dir=out \
#     --do_train=False \
#     --do_dpo=True \
#     --do_eval=True \
#     --max_steps=5000 \
#     --learning_rate=1e-4 \
#     --lr_scheduler_type='cosine_with_min_lr' \
#     --lr_scheduler_kwargs='{"min_lr": 1e-5}' \
#     --adam_beta2=0.99 \
#     --warmup_ratio=0.05 \
#     --logging_steps=20 \
#     --eval_strategy="steps" \
#     --eval_steps=100 \
#     --predict_with_generate \
#     --remove_unused_columns=False \
#     --eval_on_start=True \
#     --per_device_train_batch_size=128 \
#     --per_device_eval_batch_size=100 \
#     --gradient_accumulation_steps=12 \
#     --include_inputs_for_metrics=True \
#     --save_steps=500 \
#     --torch_compile=True \
#     --bf16=True \
#     --tf32=True


CUDA_VISIBLE_DEVICES=1 WANDB_PROJECT=arithmetic-dpo WANDB_MODE=online python run.py \
    --ref_model=True \
    --ref_model_path='out/test-llama-384-6-6-1024-COT-digits-1_10_/checkpoint-5000' \
    \
    \
    --architecture=llama \
    --from_pretrained=False \
    --hidden_size=384 \
    --intermediate_size=3072 \
    --num_attention_heads=6 \
    --num_layers=6 \
    --max_position_embeddings=1024 \
    \
    \
    --num_train=20000000 \
    --num_eval=1000 \
    --n_digits_train='1,10' \
    --op_train='add' \
    --format_train='COT' \
    --op_dist_train='1' \
    --n_digits_eval='2,17,2' \
    --op_eval='add' \
    --format_eval='COT' \
    --op_dist_eval='1' \
    \
    \
    --run_name='dpo_ref_scratch_lr1e-5' \
    --output_dir=out \
    --do_train=False \
    --do_dpo=True \
    --do_eval=True \
    --max_steps=5000 \
    --learning_rate=1e-5 \
    --lr_scheduler_type='cosine_with_min_lr' \
    --lr_scheduler_kwargs='{"min_lr": 1e-6}' \
    --adam_beta2=0.99 \
    --warmup_ratio=0.05 \
    --logging_steps=20 \
    --eval_strategy="steps" \
    --eval_steps=100 \
    --predict_with_generate \
    --remove_unused_columns=False \
    --eval_on_start=True \
    --per_device_train_batch_size=128 \
    --per_device_eval_batch_size=100 \
    --gradient_accumulation_steps=12 \
    --include_inputs_for_metrics=True \
    --save_steps=500 \
    --torch_compile=True \
    --bf16=True \
    --tf32=True



CUDA_VISIBLE_DEVICES=1 WANDB_PROJECT=arithmetic-dpo WANDB_MODE=online python run.py \
    --ref_model=True \
    --ref_model_path='out/test-llama-384-6-6-1024-COT-digits-1_10_/checkpoint-5000' \
    \
    \
    --architecture=llama \
    --from_pretrained=False \
    --hidden_size=384 \
    --intermediate_size=3072 \
    --num_attention_heads=6 \
    --num_layers=6 \
    --max_position_embeddings=1024 \
    \
    \
    --num_train=20000000 \
    --num_eval=1000 \
    --n_digits_train='1,10' \
    --op_train='add' \
    --format_train='COT' \
    --op_dist_train='1' \
    --n_digits_eval='2,17,2' \
    --op_eval='add' \
    --format_eval='COT' \
    --op_dist_eval='1' \
    \
    \
    --run_name='dpo_ref_scratch_lr1e-6' \
    --output_dir=out \
    --do_train=False \
    --do_dpo=True \
    --do_eval=True \
    --max_steps=5000 \
    --learning_rate=1e-6 \
    --lr_scheduler_type='cosine_with_min_lr' \
    --lr_scheduler_kwargs='{"min_lr": 1e-7}' \
    --adam_beta2=0.99 \
    --warmup_ratio=0.05 \
    --logging_steps=20 \
    --eval_strategy="steps" \
    --eval_steps=100 \
    --predict_with_generate \
    --remove_unused_columns=False \
    --eval_on_start=True \
    --per_device_train_batch_size=128 \
    --per_device_eval_batch_size=100 \
    --gradient_accumulation_steps=12 \
    --include_inputs_for_metrics=True \
    --save_steps=500 \
    --torch_compile=True \
    --bf16=True \
    --tf32=True



CUDA_VISIBLE_DEVICES=1 WANDB_PROJECT=arithmetic-dpo WANDB_MODE=online python run.py \
    --ref_model=True \
    --ref_model_path='out/test-llama-384-6-6-1024-COT-digits-1_10_/checkpoint-5000' \
    \
    \
    --architecture=llama \
    --from_pretrained=False \
    --hidden_size=384 \
    --intermediate_size=3072 \
    --num_attention_heads=6 \
    --num_layers=6 \
    --max_position_embeddings=1024 \
    \
    \
    --num_train=20000000 \
    --num_eval=1000 \
    --n_digits_train='1,10' \
    --op_train='add' \
    --format_train='COT' \
    --op_dist_train='1' \
    --n_digits_eval='2,17,2' \
    --op_eval='add' \
    --format_eval='COT' \
    --op_dist_eval='1' \
    \
    \
    --run_name='dpo_ref_scratch_lr1e-7' \
    --output_dir=out \
    --do_train=False \
    --do_dpo=True \
    --do_eval=True \
    --max_steps=5000 \
    --learning_rate=1e-7 \
    --lr_scheduler_type='cosine_with_min_lr' \
    --lr_scheduler_kwargs='{"min_lr": 1e-8}' \
    --adam_beta2=0.99 \
    --warmup_ratio=0.05 \
    --logging_steps=20 \
    --eval_strategy="steps" \
    --eval_steps=100 \
    --predict_with_generate \
    --remove_unused_columns=False \
    --eval_on_start=True \
    --per_device_train_batch_size=128 \
    --per_device_eval_batch_size=100 \
    --gradient_accumulation_steps=12 \
    --include_inputs_for_metrics=True \
    --save_steps=500 \
    --torch_compile=True \
    --bf16=True \
    --tf32=True



CUDA_VISIBLE_DEVICES=1 WANDB_PROJECT=arithmetic-dpo WANDB_MODE=online python run.py \
    --ref_model=True \
    --ref_model_path='out/test-llama-384-6-6-1024-COT-digits-1_10_/checkpoint-5000' \
    \
    \
    --architecture=llama \
    --from_pretrained=False \
    --hidden_size=384 \
    --intermediate_size=3072 \
    --num_attention_heads=6 \
    --num_layers=6 \
    --max_position_embeddings=1024 \
    \
    \
    --num_train=20000000 \
    --num_eval=1000 \
    --n_digits_train='1,10' \
    --op_train='add' \
    --format_train='COT' \
    --op_dist_train='1' \
    --n_digits_eval='2,17,2' \
    --op_eval='add' \
    --format_eval='COT' \
    --op_dist_eval='1' \
    \
    \
    --run_name='dpo_ref_scratch_lr1e-3' \
    --output_dir=out \
    --do_train=False \
    --do_dpo=True \
    --do_eval=True \
    --max_steps=5000 \
    --learning_rate=1e-3 \
    --lr_scheduler_type='cosine_with_min_lr' \
    --lr_scheduler_kwargs='{"min_lr": 1e-4}' \
    --adam_beta2=0.99 \
    --warmup_ratio=0.05 \
    --logging_steps=20 \
    --eval_strategy="steps" \
    --eval_steps=100 \
    --predict_with_generate \
    --remove_unused_columns=False \
    --eval_on_start=True \
    --per_device_train_batch_size=128 \
    --per_device_eval_batch_size=100 \
    --gradient_accumulation_steps=12 \
    --include_inputs_for_metrics=True \
    --save_steps=500 \
    --torch_compile=True \
    --bf16=True \
    --tf32=True