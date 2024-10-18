# reverse, nope
for seed in 43; do
    for rope_theta in Inf; do
        for num_train in 5000 10000 50000 100000 500000 1000000 5000000 10000000; do
            CUDA_VISIBLE_DEVICES=0 WANDB_PROJECT=backtrack WANDB_MODE=online python eval.py \
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
                --num_eval=10000 \
                --n_digits_train='1,17' \
                --op_train='add' \
                --format_train='reverse' \
                --op_dist_train='1' \
                --n_digits_eval='1,33,1' \
                --op_eval='add' \
                --format_eval='reverse' \
                --op_dist_eval='1' \
                --show_task_ids=True \
                --padding_side='right' \
                \
                \
                --save_steps=1000 \
                --resume_from_checkpoint="out/reverse_${num_train}-llama-384-6-6-1024-reverse-digits-1_17_-seed-43/checkpoint-10000" \
                --run_name="reverse_${num_train}" \
                --output_dir=out \
                --do_train=False \
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
                --per_device_eval_batch_size=256 \
                --gradient_accumulation_steps=2 \
                --include_inputs_for_metrics=True \
                --torch_compile=True \
                --bf16=True \
                --tf32=True 
        done
    done
done


# reverse, rope
for seed in 43; do
    for rope_theta in 1e5; do
        for num_train in 5000 10000 50000 100000 500000 1000000 5000000 10000000; do
            CUDA_VISIBLE_DEVICES=0 WANDB_PROJECT=backtrack WANDB_MODE=online python eval.py \
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
                --num_eval=10000 \
                --n_digits_train='1,17' \
                --op_train='add' \
                --format_train='reverse' \
                --op_dist_train='1' \
                --n_digits_eval='1,33,1' \
                --op_eval='add' \
                --format_eval='reverse' \
                --op_dist_eval='1' \
                --show_task_ids=True \
                --padding_side='right' \
                \
                \
                --save_steps=1000 \
                --resume_from_checkpoint="out/reverse_${num_train}-llama-384-6-6-1024-rope-reverse-digits-1_17_-seed-43/checkpoint-10000" \
                --run_name="reverse_${num_train}" \
                --output_dir=out \
                --do_train=False \
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
                --per_device_eval_batch_size=256 \
                --gradient_accumulation_steps=2 \
                --include_inputs_for_metrics=True \
                --torch_compile=True \
                --bf16=True \
                --tf32=True 
        done
    done
done


# Try decoding2 on Reverse 
for rope_theta in Inf; do
    for p in 0.5; do
        for num_train in 5000 10000 50000 100000 500000 1000000 5000000 10000000; do

            CUDA_VISIBLE_DEVICES=0 WANDB_PROJECT=backtrack WANDB_MODE=online python eval.py \
                --seed=43 \
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
                --num_eval=1000 \
                --n_digits_train='1,17' \
                --op_train='add' \
                --format_train='reverse' \
                --op_dist_train='1' \
                --n_digits_eval='1,33,1' \
                --op_eval='add' \
                --format_eval='reverse' \
                --op_dist_eval='1' \
                --show_task_ids=True \
                --padding_side='right' \
                \
                \
                --result_name='backtrack_decode2' \
                --do_backtrack_eval=True \
                --do_backtrack_decoding2=True \
                --backtrack_decoding_multiplier=10 \
                --eval_more=True \
                --save_steps=1000 \
                --resume_from_checkpoint="out/reverse_${num_train}-llama-384-6-6-1024-reverse-digits-1_17_-seed-43/checkpoint-10000" \
                --run_name="reverse_${num_train}" \
                --output_dir=out \
                --do_train=False \
                --do_eval=True \
                --max_steps=10000 \
                --learning_rate=5e-4 \
                --lr_scheduler_type='cosine' \
                --warmup_ratio=0.05 \
                --logging_steps=20 \
                --eval_strategy="steps" \
                --eval_steps=200 \
                --predict_with_generate \
                --eval_on_start=False \
                --per_device_train_batch_size=1024 \
                --per_device_eval_batch_size=512 \
                --gradient_accumulation_steps=2 \
                --include_inputs_for_metrics=True \
                --torch_compile=True \
                --bf16=True \
                --tf32=True 
        done
    done
done




################################################
################    BACKTRACK    ###############
################################################
# p=0.5, no mask
for rope_theta in Inf; do
    for p in 0.5; do
        for num_train in 5000 10000 50000 100000 500000 1000000 5000000 10000000; do

            CUDA_VISIBLE_DEVICES=0 WANDB_PROJECT=backtrack WANDB_MODE=online python eval.py \
                --seed=43 \
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
                --num_eval=10000 \
                --n_digits_train='1,17' \
                --op_train='add' \
                --format_train='backtrack' \
                --op_dist_train='1' \
                --n_digits_eval='1,33,1' \
                --op_eval='add' \
                --format_eval='reverse' \
                --op_dist_eval='1' \
                --show_task_ids=True \
                --padding_side='right' \
                \
                \
                --result_name='backtrack_decode' \
                --do_backtrack_eval=True \
                --do_backtrack_decoding=True \
                --backtrack_p=$p \
                --backtrack_mask=False \
                --save_steps=1000 \
                --resume_from_checkpoint="out/backtrack_${num_train}_p05-llama-384-6-6-1024-backtrack-digits-1_17_-seed-43/checkpoint-10000" \
                --run_name="backtrack_${num_train}_p${p}" \
                --output_dir=out \
                --do_train=False \
                --do_eval=True \
                --max_steps=10000 \
                --learning_rate=5e-4 \
                --lr_scheduler_type='cosine' \
                --warmup_ratio=0.05 \
                --logging_steps=20 \
                --eval_strategy="steps" \
                --eval_steps=200 \
                --predict_with_generate \
                --eval_on_start=False \
                --per_device_train_batch_size=1024 \
                --per_device_eval_batch_size=512 \
                --gradient_accumulation_steps=2 \
                --include_inputs_for_metrics=True \
                --torch_compile=True \
                --bf16=True \
                --tf32=True 
        done
    done
done


for rope_theta in Inf; do
    for p in 0.5; do
        for num_train in 5000 10000 50000 100000 500000 1000000 5000000 10000000; do

            CUDA_VISIBLE_DEVICES=1 WANDB_PROJECT=backtrack WANDB_MODE=online python eval.py \
                --seed=43 \
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
                --num_eval=10000 \
                --n_digits_train='1,17' \
                --op_train='add' \
                --format_train='backtrack' \
                --op_dist_train='1' \
                --n_digits_eval='1,33,1' \
                --op_eval='add' \
                --format_eval='reverse' \
                --op_dist_eval='1' \
                --show_task_ids=True \
                --padding_side='right' \
                \
                \
                --result_name='backtrack_no_decode' \
                --do_backtrack_eval=True \
                --do_backtrack_decoding=False \
                --backtrack_p=$p \
                --backtrack_mask=False \
                --save_steps=1000 \
                --resume_from_checkpoint="out/backtrack_${num_train}_p05-llama-384-6-6-1024-backtrack-digits-1_17_-seed-43/checkpoint-10000" \
                --run_name="backtrack_${num_train}_p${p}" \
                --output_dir=out \
                --do_train=False \
                --do_eval=True \
                --max_steps=10000 \
                --learning_rate=5e-4 \
                --lr_scheduler_type='cosine' \
                --warmup_ratio=0.05 \
                --logging_steps=20 \
                --eval_strategy="steps" \
                --eval_steps=200 \
                --predict_with_generate \
                --eval_on_start=False \
                --per_device_train_batch_size=1024 \
                --per_device_eval_batch_size=512 \
                --gradient_accumulation_steps=2 \
                --include_inputs_for_metrics=True \
                --torch_compile=True \
                --bf16=True \
                --tf32=True 
        done
    done
done


for rope_theta in Inf; do
    for p in 0.5; do
        for num_train in 5000 10000 50000 100000 500000 1000000 5000000 10000000; do

            CUDA_VISIBLE_DEVICES=0 WANDB_PROJECT=backtrack WANDB_MODE=online python eval.py \
                --seed=43 \
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
                --num_eval=10000 \
                --n_digits_train='1,17' \
                --op_train='add' \
                --format_train='backtrack' \
                --op_dist_train='1' \
                --n_digits_eval='1,33,1' \
                --op_eval='add' \
                --format_eval='reverse' \
                --op_dist_eval='1' \
                --show_task_ids=True \
                --padding_side='right' \
                \
                \
                --result_name='backtrack_decode2' \
                --do_backtrack_eval=True \
                --do_backtrack_decoding2=True \
                --backtrack_p=$p \
                --backtrack_mask=False \
                --save_steps=1000 \
                --resume_from_checkpoint="out/backtrack_${num_train}_p05-llama-384-6-6-1024-backtrack-digits-1_17_-seed-43/checkpoint-10000" \
                --run_name="backtrack_${num_train}_p${p}" \
                --output_dir=out \
                --do_train=False \
                --do_eval=True \
                --max_steps=10000 \
                --learning_rate=5e-4 \
                --lr_scheduler_type='cosine' \
                --warmup_ratio=0.05 \
                --logging_steps=20 \
                --eval_strategy="steps" \
                --eval_steps=200 \
                --predict_with_generate \
                --eval_on_start=False \
                --per_device_train_batch_size=1024 \
                --per_device_eval_batch_size=512 \
                --gradient_accumulation_steps=2 \
                --include_inputs_for_metrics=True \
                --torch_compile=True \
                --bf16=True \
                --tf32=True 
        done
    done
done

# p=0.5, mask
for rope_theta in Inf; do
    for p in 0.5; do
        for num_train in 5000 10000 50000 100000 500000 1000000 5000000 10000000; do

            CUDA_VISIBLE_DEVICES=1 WANDB_PROJECT=backtrack WANDB_MODE=online python eval.py \
                --seed=43 \
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
                --num_eval=10000 \
                --n_digits_train='1,17' \
                --op_train='add' \
                --format_train='backtrack' \
                --op_dist_train='1' \
                --n_digits_eval='1,33,1' \
                --op_eval='add' \
                --format_eval='reverse' \
                --op_dist_eval='1' \
                --show_task_ids=True \
                --padding_side='right' \
                \
                \
                --result_name='backtrack_decode' \
                --do_backtrack_eval=True \
                --do_backtrack_decoding=True \
                --backtrack_p=$p \
                --backtrack_mask=False \
                --save_steps=1000 \
                --resume_from_checkpoint="out/backtrack_mask_${num_train}_p05-llama-384-6-6-1024-backtrack-digits-1_17_-seed-43/checkpoint-10000" \
                --run_name="backtrack_mask_${num_train}_p${p}" \
                --output_dir=out \
                --do_train=False \
                --do_eval=True \
                --max_steps=10000 \
                --learning_rate=5e-4 \
                --lr_scheduler_type='cosine' \
                --warmup_ratio=0.05 \
                --logging_steps=20 \
                --eval_strategy="steps" \
                --eval_steps=200 \
                --predict_with_generate \
                --eval_on_start=False \
                --per_device_train_batch_size=1024 \
                --per_device_eval_batch_size=512 \
                --gradient_accumulation_steps=2 \
                --include_inputs_for_metrics=True \
                --torch_compile=True \
                --bf16=True \
                --tf32=True 
        done
    done
done


for rope_theta in Inf; do
    for p in 0.5; do
        for num_train in 5000 10000 50000 100000 500000 1000000 5000000 10000000; do

            CUDA_VISIBLE_DEVICES=1 WANDB_PROJECT=backtrack WANDB_MODE=online python eval.py \
                --seed=43 \
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
                --num_eval=10000 \
                --n_digits_train='1,17' \
                --op_train='add' \
                --format_train='backtrack' \
                --op_dist_train='1' \
                --n_digits_eval='1,33,1' \
                --op_eval='add' \
                --format_eval='reverse' \
                --op_dist_eval='1' \
                --show_task_ids=True \
                --padding_side='right' \
                \
                \
                --result_name='backtrack_no_decode' \
                --do_backtrack_eval=True \
                --do_backtrack_decoding=False \
                --backtrack_p=$p \
                --backtrack_mask=False \
                --save_steps=1000 \
                --resume_from_checkpoint="out/backtrack_mask_${num_train}_p05-llama-384-6-6-1024-backtrack-digits-1_17_-seed-43/checkpoint-10000" \
                --run_name="backtrack_mask_${num_train}_p${p}" \
                --output_dir=out \
                --do_train=False \
                --do_eval=True \
                --max_steps=10000 \
                --learning_rate=5e-4 \
                --lr_scheduler_type='cosine' \
                --warmup_ratio=0.05 \
                --logging_steps=20 \
                --eval_strategy="steps" \
                --eval_steps=200 \
                --predict_with_generate \
                --eval_on_start=False \
                --per_device_train_batch_size=1024 \
                --per_device_eval_batch_size=512 \
                --gradient_accumulation_steps=2 \
                --include_inputs_for_metrics=True \
                --torch_compile=True \
                --bf16=True \
                --tf32=True 
        done
    done
done


for rope_theta in Inf; do
    for p in 0.5; do
        for num_train in 5000 10000 50000 100000 500000 1000000 5000000 10000000; do

            CUDA_VISIBLE_DEVICES=1 WANDB_PROJECT=backtrack WANDB_MODE=online python eval.py \
                --seed=43 \
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
                --num_eval=10000 \
                --n_digits_train='1,17' \
                --op_train='add' \
                --format_train='backtrack' \
                --op_dist_train='1' \
                --n_digits_eval='1,33,1' \
                --op_eval='add' \
                --format_eval='reverse' \
                --op_dist_eval='1' \
                --show_task_ids=True \
                --padding_side='right' \
                \
                \
                --result_name='backtrack_decode2' \
                --do_backtrack_eval=True \
                --do_backtrack_decoding2=True \
                --backtrack_p=$p \
                --backtrack_mask=False \
                --save_steps=1000 \
                --resume_from_checkpoint="out/backtrack_mask_${num_train}_p05-llama-384-6-6-1024-backtrack-digits-1_17_-seed-43/checkpoint-10000" \
                --run_name="backtrack_mask_${num_train}_p${p}" \
                --output_dir=out \
                --do_train=False \
                --do_eval=True \
                --max_steps=10000 \
                --learning_rate=5e-4 \
                --lr_scheduler_type='cosine' \
                --warmup_ratio=0.05 \
                --logging_steps=20 \
                --eval_strategy="steps" \
                --eval_steps=200 \
                --predict_with_generate \
                --eval_on_start=False \
                --per_device_train_batch_size=1024 \
                --per_device_eval_batch_size=512 \
                --gradient_accumulation_steps=2 \
                --include_inputs_for_metrics=True \
                --torch_compile=True \
                --bf16=True \
                --tf32=True 
        done
    done
done




# p=0.2, no mask
for rope_theta in Inf; do
    for p in 0.2; do
        for num_train in 5000 10000 50000 100000 500000 1000000 5000000 10000000; do

            CUDA_VISIBLE_DEVICES=0 WANDB_PROJECT=backtrack WANDB_MODE=online python eval.py \
                --seed=43 \
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
                --num_eval=10000 \
                --n_digits_train='1,17' \
                --op_train='add' \
                --format_train='backtrack' \
                --op_dist_train='1' \
                --n_digits_eval='1,33,1' \
                --op_eval='add' \
                --format_eval='reverse' \
                --op_dist_eval='1' \
                --show_task_ids=True \
                --padding_side='right' \
                \
                \
                --result_name='backtrack_decode' \
                --do_backtrack_eval=True \
                --do_backtrack_decoding=True \
                --backtrack_p=$p \
                --backtrack_mask=False \
                --save_steps=1000 \
                --resume_from_checkpoint="out/backtrack_${num_train}_p02-llama-384-6-6-1024-backtrack-digits-1_17_-seed-43/checkpoint-10000" \
                --run_name="backtrack_${num_train}_p${p}" \
                --output_dir=out \
                --do_train=False \
                --do_eval=True \
                --max_steps=10000 \
                --learning_rate=5e-4 \
                --lr_scheduler_type='cosine' \
                --warmup_ratio=0.05 \
                --logging_steps=20 \
                --eval_strategy="steps" \
                --eval_steps=200 \
                --predict_with_generate \
                --eval_on_start=False \
                --per_device_train_batch_size=1024 \
                --per_device_eval_batch_size=512 \
                --gradient_accumulation_steps=2 \
                --include_inputs_for_metrics=True \
                --torch_compile=True \
                --bf16=True \
                --tf32=True 
        done
    done
done


for rope_theta in Inf; do
    for p in 0.2; do
        for num_train in 5000 10000 50000 100000 500000 1000000 5000000 10000000; do

            CUDA_VISIBLE_DEVICES=1 WANDB_PROJECT=backtrack WANDB_MODE=online python eval.py \
                --seed=43 \
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
                --num_eval=10000 \
                --n_digits_train='1,17' \
                --op_train='add' \
                --format_train='backtrack' \
                --op_dist_train='1' \
                --n_digits_eval='1,33,1' \
                --op_eval='add' \
                --format_eval='reverse' \
                --op_dist_eval='1' \
                --show_task_ids=True \
                --padding_side='right' \
                \
                \
                --result_name='backtrack_no_decode' \
                --do_backtrack_eval=True \
                --do_backtrack_decoding=False \
                --backtrack_p=$p \
                --backtrack_mask=False \
                --save_steps=1000 \
                --resume_from_checkpoint="out/backtrack_${num_train}_p02-llama-384-6-6-1024-backtrack-digits-1_17_-seed-43/checkpoint-10000" \
                --run_name="backtrack_${num_train}_p${p}" \
                --output_dir=out \
                --do_train=False \
                --do_eval=True \
                --max_steps=10000 \
                --learning_rate=5e-4 \
                --lr_scheduler_type='cosine' \
                --warmup_ratio=0.05 \
                --logging_steps=20 \
                --eval_strategy="steps" \
                --eval_steps=200 \
                --predict_with_generate \
                --eval_on_start=False \
                --per_device_train_batch_size=1024 \
                --per_device_eval_batch_size=512 \
                --gradient_accumulation_steps=2 \
                --include_inputs_for_metrics=True \
                --torch_compile=True \
                --bf16=True \
                --tf32=True 
        done
    done
done


for rope_theta in Inf; do
    for p in 0.2; do
        for num_train in 5000 10000 50000 100000 500000 1000000 5000000 10000000; do

            CUDA_VISIBLE_DEVICES=0 WANDB_PROJECT=backtrack WANDB_MODE=online python eval.py \
                --seed=43 \
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
                --num_eval=10000 \
                --n_digits_train='1,17' \
                --op_train='add' \
                --format_train='backtrack' \
                --op_dist_train='1' \
                --n_digits_eval='1,33,1' \
                --op_eval='add' \
                --format_eval='reverse' \
                --op_dist_eval='1' \
                --show_task_ids=True \
                --padding_side='right' \
                \
                \
                --result_name='backtrack_decode2' \
                --do_backtrack_eval=True \
                --do_backtrack_decoding2=True \
                --backtrack_p=$p \
                --backtrack_mask=False \
                --save_steps=1000 \
                --resume_from_checkpoint="out/backtrack_${num_train}_p02-llama-384-6-6-1024-backtrack-digits-1_17_-seed-43/checkpoint-10000" \
                --run_name="backtrack_${num_train}_p${p}" \
                --output_dir=out \
                --do_train=False \
                --do_eval=True \
                --max_steps=10000 \
                --learning_rate=5e-4 \
                --lr_scheduler_type='cosine' \
                --warmup_ratio=0.05 \
                --logging_steps=20 \
                --eval_strategy="steps" \
                --eval_steps=200 \
                --predict_with_generate \
                --eval_on_start=False \
                --per_device_train_batch_size=1024 \
                --per_device_eval_batch_size=512 \
                --gradient_accumulation_steps=2 \
                --include_inputs_for_metrics=True \
                --torch_compile=True \
                --bf16=True \
                --tf32=True 
        done
    done
done

# p=0.2, mask
for rope_theta in Inf; do
    for p in 0.2; do
        for num_train in 5000 10000 50000 100000 500000 1000000 5000000 10000000; do

            CUDA_VISIBLE_DEVICES=1 WANDB_PROJECT=backtrack WANDB_MODE=online python eval.py \
                --seed=43 \
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
                --num_eval=10000 \
                --n_digits_train='1,17' \
                --op_train='add' \
                --format_train='backtrack' \
                --op_dist_train='1' \
                --n_digits_eval='1,33,1' \
                --op_eval='add' \
                --format_eval='reverse' \
                --op_dist_eval='1' \
                --show_task_ids=True \
                --padding_side='right' \
                \
                \
                --result_name='backtrack_decode' \
                --do_backtrack_eval=True \
                --do_backtrack_decoding=True \
                --backtrack_p=$p \
                --backtrack_mask=False \
                --save_steps=1000 \
                --resume_from_checkpoint="out/backtrack_mask_${num_train}_p02-llama-384-6-6-1024-backtrack-digits-1_17_-seed-43/checkpoint-10000" \
                --run_name="backtrack_mask_${num_train}_p${p}" \
                --output_dir=out \
                --do_train=False \
                --do_eval=True \
                --max_steps=10000 \
                --learning_rate=5e-4 \
                --lr_scheduler_type='cosine' \
                --warmup_ratio=0.05 \
                --logging_steps=20 \
                --eval_strategy="steps" \
                --eval_steps=200 \
                --predict_with_generate \
                --eval_on_start=False \
                --per_device_train_batch_size=1024 \
                --per_device_eval_batch_size=512 \
                --gradient_accumulation_steps=2 \
                --include_inputs_for_metrics=True \
                --torch_compile=True \
                --bf16=True \
                --tf32=True 
        done
    done
done


for rope_theta in Inf; do
    for p in 0.2; do
        for num_train in 5000 10000 50000 100000 500000 1000000 5000000 10000000; do

            CUDA_VISIBLE_DEVICES=0 WANDB_PROJECT=backtrack WANDB_MODE=online python eval.py \
                --seed=43 \
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
                --num_eval=10000 \
                --n_digits_train='1,17' \
                --op_train='add' \
                --format_train='backtrack' \
                --op_dist_train='1' \
                --n_digits_eval='1,33,1' \
                --op_eval='add' \
                --format_eval='reverse' \
                --op_dist_eval='1' \
                --show_task_ids=True \
                --padding_side='right' \
                \
                \
                --result_name='backtrack_no_decode' \
                --do_backtrack_eval=True \
                --do_backtrack_decoding=False \
                --backtrack_p=$p \
                --backtrack_mask=False \
                --save_steps=1000 \
                --resume_from_checkpoint="out/backtrack_mask_${num_train}_p02-llama-384-6-6-1024-backtrack-digits-1_17_-seed-43/checkpoint-10000" \
                --run_name="backtrack_mask_${num_train}_p${p}" \
                --output_dir=out \
                --do_train=False \
                --do_eval=True \
                --max_steps=10000 \
                --learning_rate=5e-4 \
                --lr_scheduler_type='cosine' \
                --warmup_ratio=0.05 \
                --logging_steps=20 \
                --eval_strategy="steps" \
                --eval_steps=200 \
                --predict_with_generate \
                --eval_on_start=False \
                --per_device_train_batch_size=1024 \
                --per_device_eval_batch_size=512 \
                --gradient_accumulation_steps=2 \
                --include_inputs_for_metrics=True \
                --torch_compile=True \
                --bf16=True \
                --tf32=True 
        done
    done
done


for rope_theta in Inf; do
    for p in 0.2; do
        for num_train in 5000 10000 50000 100000 500000 1000000 5000000 10000000; do

            CUDA_VISIBLE_DEVICES=1 WANDB_PROJECT=backtrack WANDB_MODE=online python eval.py \
                --seed=43 \
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
                --num_eval=10000 \
                --n_digits_train='1,17' \
                --op_train='add' \
                --format_train='backtrack' \
                --op_dist_train='1' \
                --n_digits_eval='1,33,1' \
                --op_eval='add' \
                --format_eval='reverse' \
                --op_dist_eval='1' \
                --show_task_ids=True \
                --padding_side='right' \
                \
                \
                --result_name='backtrack_decode2' \
                --do_backtrack_eval=True \
                --do_backtrack_decoding2=True \
                --backtrack_p=$p \
                --backtrack_mask=False \
                --save_steps=1000 \
                --resume_from_checkpoint="out/backtrack_mask_${num_train}_p02-llama-384-6-6-1024-backtrack-digits-1_17_-seed-43/checkpoint-10000" \
                --run_name="backtrack_mask_${num_train}_p${p}" \
                --output_dir=out \
                --do_train=False \
                --do_eval=True \
                --max_steps=10000 \
                --learning_rate=5e-4 \
                --lr_scheduler_type='cosine' \
                --warmup_ratio=0.05 \
                --logging_steps=20 \
                --eval_strategy="steps" \
                --eval_steps=200 \
                --predict_with_generate \
                --eval_on_start=False \
                --per_device_train_batch_size=1024 \
                --per_device_eval_batch_size=512 \
                --gradient_accumulation_steps=2 \
                --include_inputs_for_metrics=True \
                --torch_compile=True \
                --bf16=True \
                --tf32=True 
        done
    done
done


#########################################################
################      multiplier: 10     ################
#########################################################

for rope_theta in Inf; do
    for p in 0.5; do
        for num_train in 5000 10000 50000 100000 500000 1000000 5000000 10000000; do

            CUDA_VISIBLE_DEVICES=0 WANDB_PROJECT=backtrack WANDB_MODE=online python eval.py \
                --seed=43 \
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
                --num_eval=1000 \
                --n_digits_train='1,17' \
                --op_train='add' \
                --format_train='backtrack' \
                --op_dist_train='1' \
                --n_digits_eval='1,33,1' \
                --op_eval='add' \
                --format_eval='reverse' \
                --op_dist_eval='1' \
                --show_task_ids=True \
                --padding_side='right' \
                \
                \
                --result_name='backtrack_decode2' \
                --do_backtrack_eval=True \
                --do_backtrack_decoding2=True \
                --backtrack_decoding_multiplier=10 \
                --eval_more=True \
                --backtrack_p=$p \
                --backtrack_mask=False \
                --save_steps=1000 \
                --resume_from_checkpoint="out/backtrack_${num_train}_p05-llama-384-6-6-1024-backtrack-digits-1_17_-seed-43/checkpoint-10000" \
                --run_name="backtrack_${num_train}_p${p}" \
                --output_dir=out \
                --do_train=False \
                --do_eval=True \
                --max_steps=10000 \
                --learning_rate=5e-4 \
                --lr_scheduler_type='cosine' \
                --warmup_ratio=0.05 \
                --logging_steps=20 \
                --eval_strategy="steps" \
                --eval_steps=200 \
                --predict_with_generate \
                --eval_on_start=False \
                --per_device_train_batch_size=1024 \
                --per_device_eval_batch_size=512 \
                --gradient_accumulation_steps=2 \
                --include_inputs_for_metrics=True \
                --torch_compile=True \
                --bf16=True \
                --tf32=True 
        done
    done
done


for rope_theta in Inf; do
    for p in 0.5; do
        for num_train in 5000 10000 50000 100000 500000 1000000 5000000 10000000; do

            CUDA_VISIBLE_DEVICES=1 WANDB_PROJECT=backtrack WANDB_MODE=online python eval.py \
                --seed=43 \
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
                --num_eval=1000 \
                --n_digits_train='1,17' \
                --op_train='add' \
                --format_train='backtrack' \
                --op_dist_train='1' \
                --n_digits_eval='1,33,1' \
                --op_eval='add' \
                --format_eval='reverse' \
                --op_dist_eval='1' \
                --show_task_ids=True \
                --padding_side='right' \
                \
                \
                --result_name='backtrack_decode2' \
                --do_backtrack_eval=True \
                --do_backtrack_decoding2=True \
                --backtrack_decoding_multiplier=10 \
                --eval_more=True \
                --backtrack_p=$p \
                --backtrack_mask=False \
                --save_steps=1000 \
                --resume_from_checkpoint="out/backtrack_mask_${num_train}_p05-llama-384-6-6-1024-backtrack-digits-1_17_-seed-43/checkpoint-10000" \
                --run_name="backtrack_mask_${num_train}_p${p}" \
                --output_dir=out \
                --do_train=False \
                --do_eval=True \
                --max_steps=10000 \
                --learning_rate=5e-4 \
                --lr_scheduler_type='cosine' \
                --warmup_ratio=0.05 \
                --logging_steps=20 \
                --eval_strategy="steps" \
                --eval_steps=200 \
                --predict_with_generate \
                --eval_on_start=False \
                --per_device_train_batch_size=1024 \
                --per_device_eval_batch_size=512 \
                --gradient_accumulation_steps=2 \
                --include_inputs_for_metrics=True \
                --torch_compile=True \
                --bf16=True \
                --tf32=True 
        done
    done
done

for rope_theta in Inf; do
    for p in 0.2; do
        for num_train in 5000 10000 50000 100000 500000 1000000 5000000 10000000; do

            CUDA_VISIBLE_DEVICES=0 WANDB_PROJECT=backtrack WANDB_MODE=online python eval.py \
                --seed=43 \
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
                --num_eval=1000 \
                --n_digits_train='1,17' \
                --op_train='add' \
                --format_train='backtrack' \
                --op_dist_train='1' \
                --n_digits_eval='1,33,1' \
                --op_eval='add' \
                --format_eval='reverse' \
                --op_dist_eval='1' \
                --show_task_ids=True \
                --padding_side='right' \
                \
                \
                --result_name='backtrack_decode2' \
                --do_backtrack_eval=True \
                --do_backtrack_decoding2=True \
                --backtrack_decoding_multiplier=10 \
                --eval_more=True \
                --backtrack_p=$p \
                --backtrack_mask=False \
                --save_steps=1000 \
                --resume_from_checkpoint="out/backtrack_${num_train}_p02-llama-384-6-6-1024-backtrack-digits-1_17_-seed-43/checkpoint-10000" \
                --run_name="backtrack_${num_train}_p${p}" \
                --output_dir=out \
                --do_train=False \
                --do_eval=True \
                --max_steps=10000 \
                --learning_rate=5e-4 \
                --lr_scheduler_type='cosine' \
                --warmup_ratio=0.05 \
                --logging_steps=20 \
                --eval_strategy="steps" \
                --eval_steps=200 \
                --predict_with_generate \
                --eval_on_start=False \
                --per_device_train_batch_size=1024 \
                --per_device_eval_batch_size=512 \
                --gradient_accumulation_steps=2 \
                --include_inputs_for_metrics=True \
                --torch_compile=True \
                --bf16=True \
                --tf32=True 
        done
    done
done


for rope_theta in Inf; do
    for p in 0.2; do
        for num_train in 5000 10000 50000 100000 500000 1000000 5000000 10000000; do

            CUDA_VISIBLE_DEVICES=1 WANDB_PROJECT=backtrack WANDB_MODE=online python eval.py \
                --seed=43 \
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
                --num_eval=1000 \
                --n_digits_train='1,17' \
                --op_train='add' \
                --format_train='backtrack' \
                --op_dist_train='1' \
                --n_digits_eval='1,33,1' \
                --op_eval='add' \
                --format_eval='reverse' \
                --op_dist_eval='1' \
                --show_task_ids=True \
                --padding_side='right' \
                \
                \
                --result_name='backtrack_decode2' \
                --do_backtrack_eval=True \
                --do_backtrack_decoding2=True \
                --backtrack_decoding_multiplier=10 \
                --eval_more=True \
                --backtrack_p=$p \
                --backtrack_mask=False \
                --save_steps=1000 \
                --resume_from_checkpoint="out/backtrack_mask_${num_train}_p02-llama-384-6-6-1024-backtrack-digits-1_17_-seed-43/checkpoint-10000" \
                --run_name="backtrack_mask_${num_train}_p${p}" \
                --output_dir=out \
                --do_train=False \
                --do_eval=True \
                --max_steps=10000 \
                --learning_rate=5e-4 \
                --lr_scheduler_type='cosine' \
                --warmup_ratio=0.05 \
                --logging_steps=20 \
                --eval_strategy="steps" \
                --eval_steps=200 \
                --predict_with_generate \
                --eval_on_start=False \
                --per_device_train_batch_size=1024 \
                --per_device_eval_batch_size=512 \
                --gradient_accumulation_steps=2 \
                --include_inputs_for_metrics=True \
                --torch_compile=True \
                --bf16=True \
                --tf32=True 
        done
    done
done





################################################
################      DEBUG     ################
################################################
rope_theta=Inf
p=0.5
num_train=5000000
CUDA_VISIBLE_DEVICES=0 WANDB_PROJECT=backtrack WANDB_MODE=online python eval.py \
    --seed=43 \
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
    --num_eval=100 \
    --n_digits_train='1,17' \
    --op_train='add' \
    --format_train='backtrack' \
    --op_dist_train='1' \
    --n_digits_eval='18,19,1' \
    --op_eval='add' \
    --format_eval='reverse' \
    --op_dist_eval='1' \
    --show_task_ids=True \
    --padding_side='right' \
    \
    \
    --result_name='debug_decode' \
    --do_backtrack_eval=True \
    --do_backtrack_decoding=True \
    --backtrack_decoding_multiplier=10 \
    --backtrack_p=$p \
    --backtrack_mask=False \
    --save_steps=1000 \
    --resume_from_checkpoint="out/backtrack_${num_train}_p05-llama-384-6-6-1024-backtrack-digits-1_17_-seed-43/checkpoint-10000" \
    --run_name="backtrack_${num_train}_p${p}" \
    --output_dir=out \
    --do_train=False \
    --do_eval=True \
    --max_steps=10000 \
    --learning_rate=5e-4 \
    --lr_scheduler_type='cosine' \
    --warmup_ratio=0.05 \
    --logging_steps=20 \
    --eval_strategy="steps" \
    --eval_steps=200 \
    --predict_with_generate \
    --eval_on_start=False \
    --per_device_train_batch_size=1024 \
    --per_device_eval_batch_size=512 \
    --gradient_accumulation_steps=2 \
    --include_inputs_for_metrics=True \
    --torch_compile=True \
    --bf16=True \
    --tf32=True 



rope_theta=Inf
p=0.5
num_train=5000000
CUDA_VISIBLE_DEVICES=0 WANDB_PROJECT=backtrack WANDB_MODE=online python eval.py \
    --seed=43 \
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
    --num_eval=100 \
    --n_digits_train='1,17' \
    --op_train='add' \
    --format_train='backtrack' \
    --op_dist_train='1' \
    --n_digits_eval='18,19,1' \
    --op_eval='add' \
    --format_eval='reverse' \
    --op_dist_eval='1' \
    --show_task_ids=True \
    --padding_side='right' \
    \
    \
    --result_name='debug_decode2' \
    --do_backtrack_eval=True \
    --do_backtrack_decoding2=True \
    --backtrack_decoding_multiplier=10 \
    --backtrack_p=$p \
    --backtrack_mask=False \
    --save_steps=1000 \
    --resume_from_checkpoint="out/backtrack_${num_train}_p05-llama-384-6-6-1024-backtrack-digits-1_17_-seed-43/checkpoint-10000" \
    --run_name="backtrack_${num_train}_p${p}" \
    --output_dir=out \
    --do_train=False \
    --do_eval=True \
    --max_steps=10000 \
    --learning_rate=5e-4 \
    --lr_scheduler_type='cosine' \
    --warmup_ratio=0.05 \
    --logging_steps=20 \
    --eval_strategy="steps" \
    --eval_steps=200 \
    --predict_with_generate \
    --eval_on_start=False \
    --per_device_train_batch_size=1024 \
    --per_device_eval_batch_size=512 \
    --gradient_accumulation_steps=2 \
    --include_inputs_for_metrics=True \
    --torch_compile=True \
    --bf16=True \
    --tf32=True 

