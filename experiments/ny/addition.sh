set -e

CUDA_VISIBLE_DEVICES=1 WANDB_PROJECT=arithmetic-dpo WANDB_MODE=online python run.py \
    --architecture=llama \
    --hidden_size=384 \
    --num_attention_heads=6 \
    --num_layers=6 \
    --max_position_embeddings=1024 \
    \
    \
    --n_digits_train='1,17' \
    --op_train='add' \
    --format_train='reverse' \
    --n_digits_eval='4,33,4' \
    --op_eval='add' \
    --format_eval='reverse' \
    \
    \
    --run_name='reverse' \
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
    --per_device_train_batch_size=1024 \
    --per_device_eval_batch_size=100 \
    --gradient_accumulation_steps=2 \
    --include_inputs_for_metrics=True \
    --bf16=True \
    --tf32=True \
    --torch_compile=True


CUDA_VISIBLE_DEVICES=1 WANDB_PROJECT=arithmetic-dpo WANDB_MODE=online python run.py \
    --architecture=llama \
    --hidden_size=384 \
    --num_attention_heads=6 \
    --num_layers=6 \
    --max_position_embeddings=1024 \
    --rope_theta=1e5 \
    \
    \
    --n_digits_train='1,17' \
    --op_train='add' \
    --format_train='reverse' \
    --n_digits_eval='4,33,4' \
    --op_eval='add' \
    --format_eval='reverse' \
    \
    \
    --run_name='reverse' \
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
    --per_device_train_batch_size=1024 \
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
    --n_digits_train='1,17' \
    --op_train='add' \
    --format_train='reverse' \
    --n_digits_eval='4,33,4' \
    --op_eval='add' \
    --format_eval='reverse' \
    \
    \
    --run_name='reverse_1e-3' \
    --output_dir=out \
    --do_train=True \
    --do_eval=True \
    --max_steps=5000 \
    --learning_rate=1e-3 \
    --lr_scheduler_type='cosine' \
    --warmup_ratio=0.05 \
    --logging_steps=20 \
    --eval_strategy="steps" \
    --eval_steps=200 \
    --predict_with_generate \
    --per_device_train_batch_size=1024 \
    --per_device_eval_batch_size=100 \
    --gradient_accumulation_steps=2 \
    --include_inputs_for_metrics=True \
    --bf16=True \
    --tf32=True \
    --torch_compile=True


CUDA_VISIBLE_DEVICES=1 WANDB_PROJECT=arithmetic-dpo WANDB_MODE=online python run.py \
    --architecture=llama \
    --hidden_size=384 \
    --num_attention_heads=6 \
    --num_layers=6 \
    --max_position_embeddings=1024 \
    --rope_theta=1e5 \
    \
    \
    --n_digits_train='1,17' \
    --op_train='add' \
    --format_train='reverse' \
    --n_digits_eval='4,33,4' \
    --op_eval='add' \
    --format_eval='reverse' \
    \
    \
    --run_name='reverse_1e-3' \
    --output_dir=out \
    --do_train=True \
    --do_eval=True \
    --max_steps=5000 \
    --learning_rate=1e-3 \
    --lr_scheduler_type='cosine' \
    --warmup_ratio=0.05 \
    --logging_steps=20 \
    --eval_strategy="steps" \
    --eval_steps=200 \
    --predict_with_generate \
    --per_device_train_batch_size=1024 \
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
    --n_digits_train='1,17' \
    --op_train='add' \
    --format_train='reverse' \
    --n_digits_eval='4,33,4' \
    --op_eval='add' \
    --format_eval='reverse' \
    \
    \
    --run_name='reverse' \
    --output_dir=out \
    --do_train=True \
    --do_eval=True \
    --max_steps=10000 \
    --learning_rate=5e-4 \
    --lr_scheduler_type='cosine' \
    --warmup_ratio=0.05 \
    --logging_steps=20 \
    --eval_strategy="steps" \
    --eval_steps=200 \
    --predict_with_generate \
    --per_device_train_batch_size=1024 \
    --per_device_eval_batch_size=100 \
    --gradient_accumulation_steps=2 \
    --include_inputs_for_metrics=True \
    --bf16=True \
    --tf32=True \
    --torch_compile=True


CUDA_VISIBLE_DEVICES=1 WANDB_PROJECT=arithmetic-dpo WANDB_MODE=online python run.py \
    --architecture=llama \
    --hidden_size=384 \
    --num_attention_heads=6 \
    --num_layers=6 \
    --max_position_embeddings=1024 \
    --rope_theta=1e5 \
    \
    \
    --n_digits_train='1,17' \
    --op_train='add' \
    --format_train='reverse' \
    --n_digits_eval='4,33,4' \
    --op_eval='add' \
    --format_eval='reverse' \
    \
    \
    --run_name='reverse' \
    --output_dir=out \
    --do_train=True \
    --do_eval=True \
    --max_steps=10000 \
    --learning_rate=5e-4 \
    --lr_scheduler_type='cosine' \
    --warmup_ratio=0.05 \
    --logging_steps=20 \
    --eval_strategy="steps" \
    --eval_steps=200 \
    --predict_with_generate \
    --per_device_train_batch_size=1024 \
    --per_device_eval_batch_size=100 \
    --gradient_accumulation_steps=2 \
    --include_inputs_for_metrics=True \
    --bf16=True \
    --tf32=True \
    --torch_compile=True




for seed in 43; do
    for rope_theta in Inf; do
        CUDA_VISIBLE_DEVICES=1 WANDB_PROJECT=backtrack WANDB_MODE=online python run.py \
            --seed=$seed \
            --architecture=llama \
            --from_pretrained=False \
            --hidden_size=384 \
            --intermediate_size=1536 \
            --num_attention_heads=6 \
            --num_layers=6 \
            --max_position_embeddings=1024 \
            --rope_theta=$rope_theta \
            \
            \
            --num_train=20000000 \
            --num_eval=1000 \
            --n_digits_train='1,17' \
            --op_train='add' \
            --format_train='reverse' \
            --op_dist_train='1' \
            --n_digits_eval='4,33,4' \
            --op_eval='add' \
            --format_eval='reverse' \
            --op_dist_eval='1' \
            --show_task_ids=True \
            --padding_side='right' \
            \
            \
            --save_steps=1000 \
            --run_name='reverse' \
            --output_dir=out \
            --do_train=True \
            --do_eval=True \
            --max_steps=10000 \
            --learning_rate=5e-4 \
            --lr_scheduler_type='cosine' \
            --warmup_ratio=0.05 \
            --logging_steps=20 \
            --eval_strategy="steps" \
            --eval_steps=200 \
            --predict_with_generate \
            --per_device_train_batch_size=1024 \
            --per_device_eval_batch_size=100\
            --gradient_accumulation_steps=2 \
            --include_inputs_for_metrics=True \
            --torch_compile=True \
            --bf16=True \
            --tf32=True 
    done
done


CUDA_VISIBLE_DEVICES=1 WANDB_PROJECT=backtrack WANDB_MODE=online python run.py \
    --architecture=llama \
    --hidden_size=384 \
    --num_attention_heads=6 \
    --num_layers=6 \
    --max_position_embeddings=1024 \
    --rope_theta=1e5 \
    --attention_dropout=0.1 \
    \
    \
    --n_digits_train='1,17' \
    --op_train='add' \
    --format_train='reverse' \
    --n_digits_eval='4,33,4' \
    --op_eval='add' \
    --format_eval='reverse' \
    \
    \
    --run_name='reverse_dropout' \
    --output_dir=out \
    --do_train=True \
    --do_eval=True \
    --max_steps=10000 \
    --learning_rate=5e-4 \
    --lr_scheduler_type='cosine' \
    --warmup_ratio=0.05 \
    --logging_steps=20 \
    --eval_strategy="steps" \
    --eval_steps=200 \
    --predict_with_generate \
    --per_device_train_batch_size=1024 \
    --per_device_eval_batch_size=100 \
    --gradient_accumulation_steps=2 \
    --include_inputs_for_metrics=True \
    --bf16=True \
    --tf32=True \
    --torch_compile=True



for seed in 43; do
    for rope_theta in Inf; do
        CUDA_VISIBLE_DEVICES=0 WANDB_PROJECT=backtrack WANDB_MODE=online python run.py \
            --seed=$seed \
            --architecture=llama \
            --from_pretrained=False \
            --hidden_size=384 \
            --intermediate_size=1536 \
            --num_attention_heads=6 \
            --num_layers=6 \
            --max_position_embeddings=1024 \
            --rope_theta=$rope_theta \
            --attention_dropout=0.1 \
            \
            \
            --num_train=20000000 \
            --num_eval=1000 \
            --n_digits_train='1,17' \
            --op_train='add' \
            --format_train='reverse' \
            --op_dist_train='1' \
            --n_digits_eval='4,33,4' \
            --op_eval='add' \
            --format_eval='reverse' \
            --op_dist_eval='1' \
            --show_task_ids=True \
            --padding_side='right' \
            \
            \
            --save_steps=1000 \
            --run_name='reverse_dropout' \
            --output_dir=out \
            --do_train=True \
            --do_eval=True \
            --max_steps=10000 \
            --learning_rate=5e-4 \
            --lr_scheduler_type='cosine' \
            --warmup_ratio=0.05 \
            --logging_steps=20 \
            --eval_strategy="steps" \
            --eval_steps=200 \
            --predict_with_generate \
            --per_device_train_batch_size=1024 \
            --per_device_eval_batch_size=100\
            --gradient_accumulation_steps=2 \
            --include_inputs_for_metrics=True \
            --torch_compile=True \
            --bf16=True \
            --tf32=True 
    done
done



for seed in 43; do
    for rope_theta in Inf; do
        for num_train in 5000,10000,50000,100000,500000,1000000,5000000,10000000; do
        CUDA_VISIBLE_DEVICES=0 WANDB_PROJECT=backtrack WANDB_MODE=online python run.py \
            --seed=$seed \
            --architecture=llama \
            --from_pretrained=False \
            --hidden_size=384 \
            --intermediate_size=1536 \
            --num_attention_heads=6 \
            --num_layers=6 \
            --max_position_embeddings=1024 \
            --rope_theta=$rope_theta \
            \
            \
            --num_train=20000000 \
            --num_eval=1000 \
            --n_digits_train='1,17' \
            --op_train='add' \
            --format_train='reverse' \
            --op_dist_train='1' \
            --n_digits_eval='4,33,4' \
            --op_eval='add' \
            --format_eval='reverse' \
            --op_dist_eval='1' \
            --show_task_ids=True \
            --padding_side='right' \
            \
            \
            --save_steps=1000 \
            --run_name="reverse_${num_train}" \
            --output_dir=out \
            --do_train=True \
            --do_eval=True \
            --max_steps=10000 \
            --learning_rate=5e-4 \
            --lr_scheduler_type='cosine' \
            --warmup_ratio=0.05 \
            --logging_steps=20 \
            --eval_strategy="steps" \
            --eval_steps=200 \
            --predict_with_generate \
            --per_device_train_batch_size=1024 \
            --per_device_eval_batch_size=100\
            --gradient_accumulation_steps=2 \
            --include_inputs_for_metrics=True \
            --torch_compile=True \
            --bf16=True \
            --tf32=True 
    done
done

