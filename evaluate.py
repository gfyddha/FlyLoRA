import os
import re
import sys
import json
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import fire
import torch
import random
import transformers
import numpy as np
from tqdm import tqdm
from peft import PeftModel
from datasets import load_dataset
from transformers import BitsAndBytesConfig, AutoModelForCausalLM, AutoTokenizer
from utils.prompter import Prompter

if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

try:
    if torch.backends.mps.is_available():
        device = "mps"
except:  # noqa: E722
    pass

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    transformers.set_seed(seed)
    torch.backends.cudnn.deterministic = True 
    torch.backends.cudnn.benchmark = False

def main(
    load_8bit: bool = False,
    base_model: str = "",
    lora_weights: str = "",
    test_dataset: str = "",
    prompt_template: str = "alpaca",  # The prompt template to use, will default to alpaca.
    save_path: str = "",
    batch_size: int = 8,
    max_new_tokens: int = 128,
    save_every: int = 200,
    seed: int = 23
):
    set_seed(seed)
    base_model = base_model or os.environ.get("BASE_MODEL", "")
    assert (
        base_model
    ), "Please specify a --base_model, e.g. --base_model='huggyllama/llama-7b'"

    prompter = Prompter(template_name=prompt_template)
    tokenizer = AutoTokenizer.from_pretrained(base_model, padding_side='left')

    quantization_config = BitsAndBytesConfig(
        load_in_8bit=False,
        llm_int8_threshold=6.0
    )

    print("start loading model")
    if device == "cuda":
        model = AutoModelForCausalLM.from_pretrained(
            base_model,
            quantization_config=quantization_config,
            torch_dtype=torch.float16,
            device_map="auto",
            use_safetensors=True
        )
        model = PeftModel.from_pretrained(
            model,
            lora_weights,
            torch_dtype=torch.float16,
        )
    elif device == "mps":
        model = AutoModelForCausalLM.from_pretrained(
            base_model,
            device_map={"": device},
            torch_dtype=torch.float16,
        )
        model = PeftModel.from_pretrained(
            model,
            lora_weights,
            device_map={"": device},
            torch_dtype=torch.float16,
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            base_model, 
            device_map={"": device}, 
            low_cpu_mem_usage=True
        )
        model = PeftModel.from_pretrained(
            model,
            lora_weights,
            device_map={"": device},
        )
    print("end loading model")

    # unwind broken decapoda-research config
    model.config.pad_token_id = tokenizer.pad_token_id = 0  # unk
    model.config.bos_token_id = 1
    model.config.eos_token_id = 2

    if not load_8bit:
        model.half()  # seems to fix bugs for some users.

    model.eval()
    if torch.__version__ >= "2" and sys.platform != "win32":
        model = torch.compile(model)

    if "mmlu" in test_dataset:
        data = load_dataset(
            "parquet",
            data_files={
                "test": f"{test_dataset}/all/test-00000-of-00001.parquet"
            }
        )

    correct = 0
    results = []
    outputs = []
    gt = []
    print(tokenizer.eos_token)

    if "mmlu" in test_dataset:
        for start_idx in tqdm(range(0, len(data['test']), batch_size)):
            end_idx = min(start_idx + batch_size, len(data['test']))
            batch = data['test'][start_idx:end_idx]
            answers = [example for example in batch["answer"]]
            questions = [example for example in batch["question"]]
            choices = [example for example in batch["choices"]]

            # generate prompt
            prompts = []
            for i in range(end_idx - start_idx):
                question = questions[i]
                choice = choices[i]
                prompt = prompter.generate_prompt(question, choice)
                prompts.append(prompt)
            inputs = tokenizer(prompts, padding=True, return_tensors="pt")
            input_ids = inputs["input_ids"].to(device)

            with torch.no_grad():
                generation_output = model.generate(
                    input_ids=input_ids,
                    return_dict_in_generate=True,
                    output_scores=True,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    temperature=1.0
                )
            s = generation_output.sequences
            output = tokenizer.batch_decode(s)
            output = [prompter.get_response(otp) for otp in output]

            # extract the answer
            print(output)
            pattern = re.compile(r'([0-3])')
            res = [pattern.findall(otp) for otp in output]
            print(res)
            for r_i in range(len(res)):
                current_res = res[r_i] if res[r_i] != [] else ["fail"]
                current_gt = answers[r_i]
                is_correct = any(str(answer) == str(current_gt) for answer in current_res)
                results.append(1 if is_correct else 0)
                outputs.append(current_res)
                gt.append(current_gt)
                print(f"{current_res} | {current_gt} | {'correct' if is_correct else 'wrong'}")
                if is_correct:
                    correct += 1

            acc = correct / len(results) * 100

            if end_idx % save_every == 0 or end_idx == len(data['test']):
                print(f"{len(results)}/{len(data['test'])}, correct: {correct}, acc: {round(acc, 2)}%, saving to {save_path}")
                write_data = {}
                write_data['acc'] = acc
                write_data['correct'] = correct
                write_data['len'] = len(results)
                write_data['results'] = results
                write_data['outputs'] = outputs
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                with open(save_path, 'w') as f:
                    json.dump(write_data, f, indent=2, separators=(',', ': '))


if __name__ == "__main__":
    fire.Fire(main)