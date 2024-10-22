set -e

# BEST: SEED: 44
WANDB_PROJECT=self_improve 
num_train=20000000
for seed in 41 42 43 44 45; do
    CUDA_VISIBLE_DEVICES=0 WANDB_MODE=online python run.py \
        --wandb_project=$WANDB_PROJECT \
        --seed=$seed \
        --architecture=llama \
        --hidden_size=384 \
        --intermediate_size=1536 \
        --num_attention_heads=6 \
        --num_layers=6 \
        --max_position_embeddings=1024 \
        \
        \
        --use_iterable_dataset=False \
        --no_seed_for_data=True \
        --num_train=$num_train \
        --num_eval=1000 \
        --n_digits_train='1,10' \
        --op_train='add' \
        --format_train='reverse' \
        --n_digits_eval='1,21,1' \
        --op_eval='add' \
        --format_eval='reverse' \
        --show_task_ids=True \
        --padding_side='right' \
        \
        \
        --save_steps=1000 \
        --run_name="reverse_${num_train}" \
        --output_dir='out/self_improve' \
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
        --per_device_eval_batch_size=1000 \
        --gradient_accumulation_steps=2 \
        --include_inputs_for_metrics=True \
        --bf16=True \
        --tf32=True \
        --torch_compile=True
done


# BEST: SEED: 43, num_train=10000000
WANDB_PROJECT=self_improve 
num_train=10000000
for seed in 41 42 43 44 45; do
    CUDA_VISIBLE_DEVICES=0 WANDB_MODE=online python run.py \
        --wandb_project=$WANDB_PROJECT \
        --seed=$seed \
        --architecture=llama \
        --hidden_size=384 \
        --intermediate_size=1536 \
        --num_attention_heads=6 \
        --num_layers=6 \
        --max_position_embeddings=1024 \
        \
        \
        --use_iterable_dataset=False \
        --no_seed_for_data=True \
        --num_train=$num_train \
        --num_eval=1000 \
        --n_digits_train='1,16' \
        --op_train='add' \
        --format_train='reverse' \
        --n_digits_eval='1,25,1' \
        --op_eval='add' \
        --format_eval='reverse' \
        --show_task_ids=True \
        --padding_side='right' \
        \
        \
        --save_steps=1000 \
        --run_name="reverse_${num_train}" \
        --output_dir='out/self_improve' \
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
        --per_device_eval_batch_size=1000 \
        --gradient_accumulation_steps=2 \
        --include_inputs_for_metrics=True \
        --bf16=True \
        --tf32=True \
        --torch_compile=True
done


##############################
#### SELF-IMPROVE DATASET ####
##############################

# trained on 1-10 digits
# 11 digit
WANDB_PROJECT=self_improve 
seed=44
for num_train in 10000 50000 100000 500000 1000000 5000000; do
    CUDA_VISIBLE_DEVICES=1 WANDB_MODE=online python generate_self_improve_data.py \
        --wandb_project=$WANDB_PROJECT \
        --seed=$seed \
        --architecture=llama \
        --hidden_size=384 \
        --intermediate_size=1536 \
        --num_attention_heads=6 \
        --num_layers=6 \
        --max_position_embeddings=1024 \
        \
        \
        --use_iterable_dataset=False \
        --no_seed_for_data=True \
        --load_as_iterable_dataset=False \
        --num_train=$num_train \
        --num_eval=1000 \
        --n_digits_train='11,12' \
        --op_train='add' \
        --format_train='reverse' \
        --n_digits_eval='1,21,1' \
        --op_eval='add' \
        --format_eval='reverse' \
        --show_task_ids=True \
        --padding_side='right' \
        \
        \
        --resume=True \
        --resume_from_checkpoint='out/self_improve/reverse_20000000-llama-384-6-6-1024-reverse-digits-1_10_-seed-44' \
        --save_steps=1000 \
        --run_name="reverse_${num_train}" \
        --output_dir='out/self_improve' \
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
        --per_device_eval_batch_size=1000 \
        --gradient_accumulation_steps=2 \
        --include_inputs_for_metrics=True \
        --bf16=True \
        --tf32=True \
        --torch_compile=True
done

# 11, 12 digit
WANDB_PROJECT=self_improve 
seed=44
for num_train in 10000 50000 100000 500000 1000000 5000000; do
    CUDA_VISIBLE_DEVICES=1 WANDB_MODE=online python generate_self_improve_data.py \
        --wandb_project=$WANDB_PROJECT \
        --seed=$seed \
        --architecture=llama \
        --hidden_size=384 \
        --intermediate_size=1536 \
        --num_attention_heads=6 \
        --num_layers=6 \
        --max_position_embeddings=1024 \
        \
        \
        --use_iterable_dataset=False \
        --no_seed_for_data=True \
        --load_as_iterable_dataset=False \
        --num_train=$num_train \
        --num_eval=1000 \
        --n_digits_train='11,13' \
        --op_train='add' \
        --format_train='reverse' \
        --n_digits_eval='1,21,1' \
        --op_eval='add' \
        --format_eval='reverse' \
        --show_task_ids=True \
        --padding_side='right' \
        \
        \
        --resume=True \
        --resume_from_checkpoint='out/self_improve/reverse_20000000-llama-384-6-6-1024-reverse-digits-1_10_-seed-44' \
        --save_steps=1000 \
        --run_name="reverse_${num_train}" \
        --output_dir='out/self_improve' \
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
        --per_device_eval_batch_size=1000 \
        --gradient_accumulation_steps=2 \
        --include_inputs_for_metrics=True \
        --bf16=True \
        --tf32=True \
        --torch_compile=True
done



# trained on 1-16 digits
# X digit
WANDB_PROJECT=self_improve 
seed=41
for num_train in 10000 50000 100000 500000 1000000 5000000; do
    CUDA_VISIBLE_DEVICES=1 WANDB_MODE=online python generate_self_improve_data.py \
        --wandb_project=$WANDB_PROJECT \
        --seed=$seed \
        --architecture=llama \
        --hidden_size=384 \
        --intermediate_size=1536 \
        --num_attention_heads=6 \
        --num_layers=6 \
        --max_position_embeddings=1024 \
        \
        \
        --use_iterable_dataset=False \
        --no_seed_for_data=True \
        --load_as_iterable_dataset=False \
        --num_train=$num_train \
        --num_eval=1000 \
        --n_digits_train='' \
        --op_train='add' \
        --format_train='reverse' \
        --n_digits_eval='1,25,1' \
        --op_eval='add' \
        --format_eval='reverse' \
        --show_task_ids=True \
        --padding_side='right' \
        \
        \
        --resume=True \
        --resume_from_checkpoint='out/self_improve/reverse_10000000-llama-384-6-6-1024-reverse-digits-1_16_-seed-41' \
        --save_steps=1000 \
        --run_name="reverse_${num_train}" \
        --output_dir='out/self_improve' \
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
        --per_device_eval_batch_size=1000 \
        --gradient_accumulation_steps=2 \
        --include_inputs_for_metrics=True \
        --bf16=True \
        --tf32=True \
        --torch_compile=True
done


###################
#### DEBUGGING ####
###################

set -e

WANDB_PROJECT=self_improve 
CUDA_VISIBLE_DEVICES=1 WANDB_MODE='disabled' python run.py \
    --wandb_project=$WANDB_PROJECT \
    --seed=43 \
    --architecture=llama \
    --hidden_size=384 \
    --intermediate_size=1536 \
    --num_attention_heads=6 \
    --num_layers=6 \
    --max_position_embeddings=1024 \
    \
    \
    --use_iterable_dataset=False \
    --num_train=1000 \
    --num_eval=100 \
    --n_digits_train='1,3' \
    --op_train='add' \
    --format_train='reverse' \
    --n_digits_eval='1,7,1' \
    --op_eval='add' \
    --format_eval='reverse' \
    --show_task_ids=True \
    --padding_side='right' \
    \
    \
    --save_steps=10 \
    --run_name='reverse' \
    --output_dir='out/self_improve' \
    --do_train=True \
    --do_eval=True \
    --max_steps=100 \
    --learning_rate=5e-4 \
    --lr_scheduler_type='cosine' \
    --warmup_ratio=0.05 \
    --logging_steps=20 \
    --eval_strategy="steps" \
    --eval_steps=200 \
    --predict_with_generate \
    --per_device_train_batch_size=1024 \
    --per_device_eval_batch_size=1000 \
    --gradient_accumulation_steps=2 \
    --include_inputs_for_metrics=True \
    --bf16=True \
    --tf32=True \
    --torch_compile=True


set -e

WANDB_PROJECT=self_improve 
CUDA_VISIBLE_DEVICES=1 WANDB_MODE='disabled' python generate_self_improve_data.py \
    --wandb_project=$WANDB_PROJECT \
    --seed=43 \
    --architecture=llama \
    --hidden_size=384 \
    --intermediate_size=1536 \
    --num_attention_heads=6 \
    --num_layers=6 \
    --max_position_embeddings=1024 \
    \
    \
    --no_seed_for_data=True \
    --use_iterable_dataset=False \
    --load_as_iterable_dataset=False \
    --num_train=1000 \
    --num_eval=100 \
    --n_digits_train='1,3' \
    --op_train='add' \
    --format_train='reverse' \
    --n_digits_eval='1,7,1' \
    --op_eval='add' \
    --format_eval='reverse' \
    --show_task_ids=True \
    --padding_side='right' \
    \
    \
    --resume=True \
    --resume_from_checkpoint=True \
    --save_steps=10 \
    --run_name='reverse' \
    --output_dir='out/self_improve' \
    --do_train=True \
    --do_eval=True \
    --max_steps=100 \
    --learning_rate=5e-4 \
    --lr_scheduler_type='cosine' \
    --warmup_ratio=0.05 \
    --logging_steps=20 \
    --eval_strategy="steps" \
    --eval_steps=200 \
    --predict_with_generate \
    --per_device_train_batch_size=1024 \
    --per_device_eval_batch_size=1000 \
    --gradient_accumulation_steps=2 \
    --include_inputs_for_metrics=True \
    --bf16=True \
    --tf32=True \
    --torch_compile=True
