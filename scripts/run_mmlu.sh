#! /bin/bash

cd ..

CUDA_VISIBLE_DEVICES=0 python finetune.py \
         --base_model "/path/to/model/Meta-Llama-3.1-8B" \
         --data_path "/path/to/mmlu" \
         --output_dir "/path/to/output_dir" \
         --batch_size 128 \
         --micro_batch_size 8 \
         --num_epochs 1 \
         --learning_rate 3e-4 \
         --cutoff_len 128 \
         --val_set_size 0 \
         --lora_r 32 \
         --lora_alpha 64 \
         --warmup_rates 0.01 \
         --lora_dropout 0.00 \
         --lora_target_modules "q_proj,v_proj,k_proj,o_proj,gate_proj,down_proj,up_proj" \
         --train_on_inputs \
         --group_by_length \
         --add_eos_token \
         --prompt_template_name "alpaca_mmlu" \
         --A_type "sparse" \
         --top_k 8 \
         --bias_u 0.001 \
         --seed 23

CUDA_VISIBLE_DEVICES=0 python evaluate.py \
         --base_model "/path/to/model/Meta-Llama-3.1-8B" \
         --lora_weights "/path/to/output_dir" \
         --test_dataset "/path/to/mmlu" \
         --batch_size 8 \
         --prompt_template "alpaca_mmlu" \
         --max_new_tokens 128 \
         --save_path "/path/to/output_dir/lora_test_mmlu.json" \
         --seed 23