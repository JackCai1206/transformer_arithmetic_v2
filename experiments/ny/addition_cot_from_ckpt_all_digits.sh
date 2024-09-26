set -e

# start from 1-10 digit CoT trained model and further train on 1-14 digits

CUDA_VISIBLE_DEVICES=0 WANDB_PROJECT=arithmetic-dpo WANDB_MODE=online python run.py \
    --architecture=llama \
    --hidden_size=384 \
    --num_attention_heads=6 \
    --num_layers=6 \
    --max_position_embeddings=1024 \
    \
    \
    --n_digits_train='8,14' \
    --num_train=50000 \
    --op_train='add' \
    --format_train='COT' \
    --n_digits_eval='2,17,2' \
    --op_eval='add' \
    --format_eval='COT' \
    \
    \
    --run_name='cont' \
    --output_dir=out \
    --resume_from_checkpoint='out/test-llama-384-6-6-1024-COT-digits-1_10_/checkpoint-5000' \
    --ignore_data_skip=True \
    --do_train=True \
    --do_eval=True \
    --max_steps=6000 \
    --learning_rate=1e-4 \
    --lr_scheduler_type='cosine' \
    --warmup_ratio=0.05 \
    --logging_steps=20 \
    --eval_strategy="steps" \
    --eval_steps=200 \
    --predict_with_generate \
    --per_device_train_batch_size=256 \
    --per_device_eval_batch_size=100 \
    --gradient_accumulation_steps=2 \
    --include_inputs_for_metrics=True \
    --bf16=True \
    --tf32=True \
    --torch_compile=True



CUDA_VISIBLE_DEVICES=0 WANDB_PROJECT=arithmetic-dpo WANDB_MODE=online python run.py \
    --architecture=llama \
    --hidden_size=384 \
    --num_attention_heads=6 \
    --num_layers=6 \
    --max_position_embeddings=1024 \
    \
    \
    --n_digits_train='6,14' \
    --num_train=50000 \
    --op_train='add' \
    --format_train='COT' \
    --n_digits_eval='2,17,2' \
    --op_eval='add' \
    --format_eval='COT' \
    \
    \
    --run_name='cont' \
    --output_dir=out \
    --resume_from_checkpoint='out/test-llama-384-6-6-1024-COT-digits-1_10_/checkpoint-5000' \
    --ignore_data_skip=True \
    --do_train=True \
    --do_eval=True \
    --max_steps=6000 \
    --learning_rate=1e-4 \
    --lr_scheduler_type='cosine' \
    --warmup_ratio=0.05 \
    --logging_steps=20 \
    --eval_strategy="steps" \
    --eval_steps=200 \
    --predict_with_generate \
    --per_device_train_batch_size=256 \
    --per_device_eval_batch_size=100 \
    --gradient_accumulation_steps=2 \
    --include_inputs_for_metrics=True \
    --bf16=True \
    --tf32=True \
    --torch_compile=True


CUDA_VISIBLE_DEVICES=0 WANDB_PROJECT=arithmetic-dpo WANDB_MODE=online python run.py \
    --architecture=llama \
    --hidden_size=384 \
    --num_attention_heads=6 \
    --num_layers=6 \
    --max_position_embeddings=1024 \
    \
    \
    --n_digits_train='4,14' \
    --num_train=50000 \
    --op_train='add' \
    --format_train='COT' \
    --n_digits_eval='2,17,2' \
    --op_eval='add' \
    --format_eval='COT' \
    \
    \
    --run_name='cont' \
    --output_dir=out \
    --resume_from_checkpoint='out/test-llama-384-6-6-1024-COT-digits-1_10_/checkpoint-5000' \
    --ignore_data_skip=True \
    --do_train=True \
    --do_eval=True \
    --max_steps=6000 \
    --learning_rate=1e-4 \
    --lr_scheduler_type='cosine' \
    --warmup_ratio=0.05 \
    --logging_steps=20 \
    --eval_strategy="steps" \
    --eval_steps=200 \
    --predict_with_generate \
    --per_device_train_batch_size=256 \
    --per_device_eval_batch_size=100 \
    --gradient_accumulation_steps=2 \
    --include_inputs_for_metrics=True \
    --bf16=True \
    --tf32=True \
    --torch_compile=True


    CUDA_VISIBLE_DEVICES=0 WANDB_PROJECT=arithmetic-dpo WANDB_MODE=online python run.py \
    --architecture=llama \
    --hidden_size=384 \
    --num_attention_heads=6 \
    --num_layers=6 \
    --max_position_embeddings=1024 \
    \
    \
    --n_digits_train='2,14' \
    --num_train=50000 \
    --op_train='add' \
    --format_train='COT' \
    --n_digits_eval='2,17,2' \
    --op_eval='add' \
    --format_eval='COT' \
    \
    \
    --run_name='cont' \
    --output_dir=out \
    --resume_from_checkpoint='out/test-llama-384-6-6-1024-COT-digits-1_10_/checkpoint-5000' \
    --ignore_data_skip=True \
    --do_train=True \
    --do_eval=True \
    --max_steps=6000 \
    --learning_rate=1e-4 \
    --lr_scheduler_type='cosine' \
    --warmup_ratio=0.05 \
    --logging_steps=20 \
    --eval_strategy="steps" \
    --eval_steps=200 \
    --predict_with_generate \
    --per_device_train_batch_size=256 \
    --per_device_eval_batch_size=100 \
    --gradient_accumulation_steps=2 \
    --include_inputs_for_metrics=True \
    --bf16=True \
    --tf32=True \
    --torch_compile=True



    CUDA_VISIBLE_DEVICES=0 WANDB_PROJECT=arithmetic-dpo WANDB_MODE=online python run.py \
    --architecture=llama \
    --hidden_size=384 \
    --num_attention_heads=6 \
    --num_layers=6 \
    --max_position_embeddings=1024 \
    \
    \
    --n_digits_train='1,14' \
    --num_train=50000 \
    --op_train='add' \
    --format_train='COT' \
    --n_digits_eval='2,17,2' \
    --op_eval='add' \
    --format_eval='COT' \
    \
    \
    --run_name='cont' \
    --output_dir=out \
    --resume_from_checkpoint='out/test-llama-384-6-6-1024-COT-digits-1_10_/checkpoint-5000' \
    --ignore_data_skip=True \
    --do_train=True \
    --do_eval=True \
    --max_steps=6000 \
    --learning_rate=1e-4 \
    --lr_scheduler_type='cosine' \
    --warmup_ratio=0.05 \
    --logging_steps=20 \
    --eval_strategy="steps" \
    --eval_steps=200 \
    --predict_with_generate \
    --per_device_train_batch_size=256 \
    --per_device_eval_batch_size=100 \
    --gradient_accumulation_steps=2 \
    --include_inputs_for_metrics=True \
    --bf16=True \
    --tf32=True \
    --torch_compile=True

