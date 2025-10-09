import os
import glob
import shutil
import torch
from safetensors import safe_open
from collections import defaultdict
from tqdm import tqdm
import argparse


def fast_lora_average(peft_dirs, output_dir, file_format="safetensors"):
    """
    Fast averaging of multiple PEFT adapters' LoRA parameters.
    
    Args:
        peft_dirs: List of directories containing task-specific LoRA adapters.
        output_dir: Directory to save the averaged adapter.
        file_format: Output file format ('safetensors' or 'bin').
    """
    # Collect all parameter file paths
    param_files = []
    config_files = []
    for dir_path in peft_dirs:
        files = glob.glob(os.path.join(dir_path, "adapter_model.safetensors")) \
              + glob.glob(os.path.join(dir_path, "adapter_model.bin"))
        if not files:
            raise ValueError(f"No adapter files found in {dir_path}")
        param_files.append(files[0])  # Use the first adapter file in each directory

        # Locate configuration file
        config_path = os.path.join(dir_path, "adapter_config.json")
        if not os.path.exists(config_path):
            print(f"Warning: adapter_config.json not found in {dir_path}")
        else:
            config_files.append(config_path)

    # Initialize parameter storage
    param_dict = defaultdict(list)
    param_keys = None

    # Load parameters
    for file_path in tqdm(param_files, desc="Loading parameters"):
        # Detect file format automatically
        if file_path.endswith(".safetensors"):
            with safe_open(file_path, framework="pt") as f:
                current_params = {key: f.get_tensor(key) for key in f.keys()}
        else:
            current_params = torch.load(file_path, map_location="cpu")

        # Check key consistency
        if param_keys is None:
            param_keys = set(k for k in current_params if "lora_" in k.lower())
        else:
            current_keys = set(k for k in current_params if "lora_" in k.lower())
            if current_keys != param_keys:
                raise ValueError(f"Inconsistent parameter structures detected in {file_path}")

        # Collect LoRA parameters
        for k in param_keys:
            param_dict[k].append(current_params[k])

    # Compute mean across adapters
    avg_params = {}
    for k in tqdm(param_keys, desc="Averaging parameters"):
        stacked = torch.stack(param_dict[k], dim=0)
        avg_params[k] = torch.mean(stacked, dim=0)

    # Save results
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"adapter_model.{file_format}")

    if file_format == "safetensors":
        from safetensors.torch import save_file
        save_file(avg_params, output_path)
    else:
        torch.save(avg_params, output_path)

    # Copy one config file (if available)
    if config_files:
        try:
            shutil.copy(config_files[0], os.path.join(output_dir, "adapter_config.json"))
            print(f"Configuration file copied to: {os.path.join(output_dir, 'adapter_config.json')}")
        except Exception as e:
            print(f"Failed to copy configuration file: {e}")
    else:
        print("Warning: No valid adapter_config.json files were found.")

    print(f"Averaged adapter saved to: {output_path}")
    return output_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fast LoRA adapter averaging tool")
    parser.add_argument(
        "--peft_dirs",
        type=str,
        nargs="+",
        required=True,
        help="List of directories containing LoRA adapters to be merged"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Output directory for the merged adapter"
    )
    parser.add_argument(
        "--file_format",
        type=str,
        default="safetensors",
        choices=["safetensors", "bin"],
        help="Output file format (default: safetensors)"
    )

    args = parser.parse_args()
    fast_lora_average(args.peft_dirs, args.output_dir, args.file_format)