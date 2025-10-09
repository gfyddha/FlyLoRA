#! /bin/bash
# Script: run_merge_lora.sh
# Purpose: Merge multiple LoRA adapters into one averaged adapter

python tools/merge_lora_adapters.py --peft_dirs /path/to/adapter1 /path/to/adapter2 /path/to/adapter3 /path/to/adapter4 --output_dir /path/to/adapter_merged --file_format safetensors