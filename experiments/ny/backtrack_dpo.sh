set -e

seed=43
rope_theta=Inf
num_train=10000000
num_eval=1000
ref_model_path='backtrack_mask_10000000_p02-llama-384-6-6-1024-backtrack-digits-1_17_-seed-43'

for lr in 1e-4 1e-5 5e-6 2e-6 1e-6 5e-7 2e-7 1e-7; do
    CUDA_VISIBLE_DEVICES=0 WANDB_PROJECT=backtrack-dpo WANDB_MODE=online python run.py \
        --ref_model=True \
        --ref_model_path="out/${ref_model_path}/checkpoint-10000" \
        \
        \
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
        --num_train=$num_train \
        --num_eval=$num_eval \
        --n_digits_train='1,17' \
        --op_train='add' \
        --format_train='backtrack' \
        --op_dist_train='1' \
        --n_digits_eval='4,33,4' \
        --op_eval='add' \
        --format_eval='reverse' \
        --op_dist_eval='1' \
        --show_task_ids=False \
        --padding_side='right' \
        --dpo_format='repeat_penalty' \
        \
        \
        --do_backtrack_decoding=True \
        --save_steps=500 \
        --resume_from_checkpoint="out/${ref_model_path}/checkpoint-10000" \
        --run_name="dpo_repeat_${lr}" \
        --output_dir=out \
        --do_train=False \
        --do_dpo=True \
        --do_eval=True \
        --max_steps=1000 \
        --learning_rate=$lr \
        --lr_scheduler_type='cosine' \
        --warmup_ratio=0.05 \
        --logging_steps=20 \
        --eval_strategy="steps" \
        --eval_steps=100 \
        --predict_with_generate \
        --eval_on_start=True \
        --per_device_train_batch_size=1024 \
        --per_device_eval_batch_size=100 \
        --gradient_accumulation_steps=2 \
        --include_inputs_for_metrics=True \
        --torch_compile=True \
        --bf16=True \
        --tf32=True 
done




############################################
############# DEBUGGING ####################
############################################

lr=1e-4
CUDA_VISIBLE_DEVICES=0 WANDB_PROJECT=backtrack-dpo WANDB_MODE=online python run.py \
    --ref_model=True \
    --ref_model_path="out/${ref_model_path}/checkpoint-10000" \
    \
    \
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
    --num_train=$num_train \
    --num_eval=$num_eval \
    --n_digits_train='1,17' \
    --op_train='add' \
    --format_train='backtrack' \
    --op_dist_train='1' \
    --n_digits_eval='4,33,4' \
    --op_eval='add' \
    --format_eval='reverse' \
    --op_dist_eval='1' \
    --show_task_ids=False \
    --padding_side='right' \
    --dpo_format='backtrack_reward' \
    \
    \
    --do_backtrack_decoding=True \
    --save_steps=500 \
    --resume_from_checkpoint="out/${ref_model_path}/checkpoint-10000" \
    --run_name="dpo_repeat_${lr}" \
    --output_dir=out \
    --do_train=False \
    --do_dpo=True \
    --do_eval=True \
    --max_steps=1000 \
    --learning_rate=$lr \
    --lr_scheduler_type='cosine' \
    --warmup_ratio=0.05 \
    --logging_steps=20 \
    --eval_strategy="steps" \
    --eval_steps=100 \
    --predict_with_generate \
    --eval_on_start=True \
    --per_device_train_batch_size=1024 \
    --per_device_eval_batch_size=100 \
    --gradient_accumulation_steps=2 \
    --include_inputs_for_metrics=True \
    --torch_compile=True \
    --bf16=True \
    --tf32=True 
