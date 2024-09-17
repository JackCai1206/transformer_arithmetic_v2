set -e

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
    --num_eval=1000 \
    --n_digits_train='1,33 1,33 1,5' \
    --op_train='add add add' \
    --format_train='automata_A automata_B automata_C' \
    --op_dist_train='1 1 0' \
    --n_digits_eval='4,49,4' \
    --op_eval='add add add' \
    --format_eval='automata_A automata_B automata_C' \
    --op_dist_eval='1 1 1' \
    --add_special_tokens=False \
    \
    \
    --run_name='test' \
    --output_dir=out \
    --do_train=True \
    --do_eval=True \
    --max_steps=10000 \
    --lr_scheduler_type='warmup_stable_decay' \
    --lr_scheduler_kwargs='{"num_stable_steps": 8000, "num_decay_steps": 1000}' \
    --adam_beta2=0.99 \
    --weight_decay=0.00 \
    --max_grad_norm=1 \
    --warmup_ratio=0.1 \
    --logging_steps=20 \
    --eval_strategy="steps" \
    --eval_steps=100 \
    --predict_with_generate \
    --remove_unused_columns=False \
    --eval_on_start=False \
    --per_device_train_batch_size=160 \
    --per_device_eval_batch_size=1024 \
    --gradient_accumulation_steps=6 \
    --include_inputs_for_metrics=True \
    --save_steps=500 \
    --torch_compile=False \
    --bf16=True \
    --tf32=True

