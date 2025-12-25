#!/usr/bin/env python3
"""
Supervised Fine-Tuning on a mixed dataset.

Trains on:
- 250 samples from corruptions dataset (bugfix, without reasoning)
- 250 samples from MG-Verilog dataset (spec-to-rtl)
"""

import torch
import torch.distributed as dist
from torch.nn.utils.rnn import pad_sequence
from dataclasses import dataclass
from typing import Sequence, Dict
import json
import os
import argparse
import math
import re

from transformers import (
    AutoTokenizer,
    PreTrainedTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer
)
from transformers import TrainerCallback
from peft import (
    get_peft_model,
    LoraConfig,
    TaskType
)

from datasets import Dataset
import wandb

# ==== Constants ====
IGNORE_INDEX = -100
DEFAULT_PAD_TOKEN = "[PAD]"
model_name = "Qwen/Qwen2.5-Coder-3B"
corruptions_path = "/afs/cs.stanford.edu/u/ethanboneh/data/rtl_smith_corruptions/train_corrupt_with_spec_headers.jsonl"
mg_verilog_path = "/matx/u/ethanboneh/mg_verilog_dataset.arrow"

# ==== Prompt Templates ====
prompts = {}

prompts['spec-to-rtl'] = {
  'system_msg' : """
You are a Verilog RTL designer that only writes code using correct Verilog syntax.
""",
  'prompt_prefix' : ""
}

prompt_rules_suffix="""
Here are some additional rules and coding conventions.

 - Declare all ports and signals as logic; do not to use wire or reg.

 - For combinational logic with an always block do not explicitly specify
   the sensitivity list; instead use always @(*).

 - All sized numeric constants must have a size greater than zero
   (e.g, 0'b0 is not a valid expression).

 - An always block must read at least one signal otherwise it will never
   be executed; use an assign statement instead of an always block in
   situations where there is no need to read any signals.

 - if the module uses a synchronous reset signal, this means the reset
   signal is sampled with respect to the clock. When implementing a
   synchronous reset signal, do not include posedge reset in the
   sensitivity list of any sequential always block.
"""

prompt_no_explain_suffix="""
Enclose your code with [BEGIN] and [DONE]. Only output the code snippet
and do NOT output anything else.
"""

BUGFIX_INSTRUCTIONS = """
Enclose your code with [BEGIN] and [DONE]. Only output the code snippet
and do NOT output anything else.
"""

# ======================================================================
# Utilities for cleaning + parsing MG-Verilog descriptions
# ======================================================================

def clean_description_text(text):
    text = re.sub(r"<<SYS>>.*?<</SYS>>", "", text, flags=re.DOTALL)
    text = re.sub(r"<s>|</s>", "", text)
    text = re.sub(r"\[INST\]|\[/INST\]", "", text)
    return text.strip()

def extract_prompt_and_module_header(text):
    mh_pos = text.rfind("Module header:")
    if mh_pos == -1:
        raise ValueError("module header not found")
    prompt_part = text[:mh_pos].strip()
    tail = text[mh_pos:].splitlines()
    module_header = None
    for line in tail[1:]:
        l = line.strip()
        if l.startswith("module "):
            module_header = l
            break
    if module_header is None:
        raise ValueError("module header line not found")
    return prompt_part, module_header

def process_prompt(description, selector="detailed_global_summary"):
    raw_question_text = description[selector].strip()
    cleaned = clean_description_text(raw_question_text)
    prompt_part, module_header = extract_prompt_and_module_header(cleaned)
    return prompt_part, module_header

# ======================================================================
# Data Formatting Functions
# ======================================================================

def format_mg_verilog_example(ex):
    """Format MG-Verilog example (spec-to-rtl)."""
    desc = ex["description"]
    code = ex["code"]
    
    # Process and clean the prompt
    prompt_part, module_header = process_prompt(desc)
    
    # Combine module header with code
    if code.strip().startswith("module"):
        full_code = code.strip()
    else:
        if not module_header.endswith(";"):
            full_code = module_header + ";\n" + code.strip()
        else:
            full_code = module_header + "\n" + code.strip()
    
    # Build user prompt (include system message in user content)
    user_content = prompts['spec-to-rtl']['system_msg'].strip() + "\n\n"
    user_content += "Question:\n"
    user_content += prompt_part.strip() + "\n"
    user_content += prompt_rules_suffix.strip() + "\n"
    user_content += prompt_no_explain_suffix.strip() + "\n"
    user_content += "Answer:\n"
    
    return {
        "input": user_content,
        "output": "[BEGIN]\n" + full_code.strip() + "\n[DONE]"
    }

def format_bugfix_example(ex):
    """Format bugfix example (without reasoning trace)."""
    spec = ex.get("spec", "").strip()
    issue_description = ex.get("issue_description", "").strip()
    corrupted_code = ex.get("corrupted_code", "").strip()
    clean_code = ex.get("clean_code", "").strip()
    
    # Build user prompt
    user_content = spec
    
    if issue_description:
        user_content += "\n\nIssue Description:\n"
        user_content += issue_description.strip() + "\n"
    
    user_content += "\nBuggy Code:\n"
    user_content += "```systemverilog\n"
    user_content += corrupted_code.strip() + "\n"
    user_content += "```\n"
    
    user_content += BUGFIX_INSTRUCTIONS.strip() + "\n"
    user_content += "Answer:\n"
    
    output = "[BEGIN]\n" + clean_code.strip() + "\n[DONE]"
    
    return {
        "input": user_content,
        "output": output
    }

# ==== Data Collator ====
@dataclass
class DataCollatorForCausalLM:
    tokenizer: PreTrainedTokenizer
    source_max_len: int
    target_max_len: int
    train_on_source: bool = False
    predict_with_generate: bool = False

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        sources = [f"{self.tokenizer.bos_token}{ex['input']}" for ex in instances]
        targets = [f"{ex['output']}{self.tokenizer.eos_token}" for ex in instances]

        tokenized_sources = self.tokenizer(sources, max_length=self.source_max_len, truncation=True, add_special_tokens=False)
        tokenized_targets = self.tokenizer(targets, max_length=self.target_max_len, truncation=True, add_special_tokens=False)

        input_ids, labels = [], []
        for src, tgt in zip(tokenized_sources["input_ids"], tokenized_targets["input_ids"]):
            input_ids.append(torch.tensor(src + tgt))
            labels.append(torch.tensor([IGNORE_INDEX] * len(src) + tgt))

        input_ids = pad_sequence(input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id)
        labels = pad_sequence(labels, batch_first=True, padding_value=IGNORE_INDEX)
        return {
            "input_ids": input_ids,
            "labels": labels,
            "attention_mask": input_ids.ne(self.tokenizer.pad_token_id)
        }

# ==== Training Function ====
def train(resume_from_checkpoint=False, num_epochs=3, num_corruptions=250, num_mg_verilog=250):
    # Detect number of GPUs early (needed for batch size calculation)
    try:
        use_ddp = dist.is_initialized()
        world_size = dist.get_world_size() if use_ddp else torch.cuda.device_count()
    except (RuntimeError, ValueError):
        use_ddp = False
        world_size = torch.cuda.device_count()
    
    # Calculate batch size configuration to achieve effective batch size of 16
    num_gpus = world_size
    target_effective_batch = 16
    
    if num_gpus == 1:
        per_device_batch = 2
        grad_accum = 8
    elif num_gpus == 2:
        per_device_batch = 1
        grad_accum = 8
    elif num_gpus >= 4:
        per_device_batch = 1
        grad_accum = max(1, target_effective_batch // (per_device_batch * num_gpus))
    else: # Fallback: 3 GPUs
        per_device_batch = 1
        grad_accum = 5
    
    # For model parallelism (device_map="auto"), effective batch is per_device * grad_accum
    trainer_effective_batch = per_device_batch * grad_accum
    
    print(f"Training configuration:")
    print(f"  - GPUs: {num_gpus}")
    print(f"  - Per-device batch size: {per_device_batch}")
    print(f"  - Gradient accumulation steps: {grad_accum}")
    print(f"  - Trainer effective batch size: {trainer_effective_batch}")
    
    # Load corruptions dataset
    print(f"\nLoading corruptions dataset from {corruptions_path}...")
    corruptions_examples = []
    with open(corruptions_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                ex = json.loads(line)
                corruptions_examples.append(ex)
            except json.JSONDecodeError as e:
                print(f"Warning: Skipping invalid JSON line: {e}")
                continue
    
    print(f"Loaded {len(corruptions_examples)} corruptions examples")
    
    # Limit to num_corruptions
    if num_corruptions and num_corruptions < len(corruptions_examples):
        corruptions_examples = corruptions_examples[:num_corruptions]
        print(f"Limited to {len(corruptions_examples)} corruptions examples")
    
    # Load MG-Verilog dataset
    print(f"\nLoading MG-Verilog dataset from {mg_verilog_path}...")
    mg_dataset = Dataset.from_file(mg_verilog_path)
    print(f"Loaded {len(mg_dataset)} MG-Verilog examples")
    
    # Limit to num_mg_verilog
    if num_mg_verilog and num_mg_verilog < len(mg_dataset):
        mg_dataset = mg_dataset.select(range(num_mg_verilog))
        print(f"Limited to {len(mg_dataset)} MG-Verilog examples")
    
    # Format corruptions dataset
    print("\nFormatting corruptions dataset...")
    formatted_corruptions = []
    for ex in corruptions_examples:
        try:
            formatted = format_bugfix_example(ex)
            formatted_corruptions.append(formatted)
        except Exception as e:
            print(f"Warning: Failed to format corruption example: {e}")
            continue
    
    print(f"Formatted {len(formatted_corruptions)} corruptions examples")
    
    # Format MG-Verilog dataset
    print("\nFormatting MG-Verilog dataset...")
    formatted_mg_verilog = []
    for i in range(len(mg_dataset)):
        try:
            ex = mg_dataset[i]
            formatted = format_mg_verilog_example(ex)
            formatted_mg_verilog.append(formatted)
        except Exception as e:
            print(f"Warning: Failed to format MG-Verilog example {i}: {e}")
            continue
    
    print(f"Formatted {len(formatted_mg_verilog)} MG-Verilog examples")
    
    # Combine datasets
    print(f"\nCombining datasets...")
    all_formatted = formatted_corruptions + formatted_mg_verilog
    print(f"Total examples: {len(all_formatted)} ({len(formatted_corruptions)} corruptions + {len(formatted_mg_verilog)} MG-Verilog)")
    
    if len(all_formatted) == 0:
        raise ValueError("No valid examples after formatting!")
    
    # Create HuggingFace dataset
    dataset = Dataset.from_list(all_formatted)
    
    # Split into train/test (10% test)
    dataset = dataset.train_test_split(test_size=0.1, seed=42)
    print(f"Train size: {len(dataset['train'])}, Test size: {len(dataset['test'])}")
    
    # Calculate steps per epoch
    steps_per_epoch = math.ceil(len(dataset['train']) / trainer_effective_batch)
    expected_steps = steps_per_epoch * num_epochs
    print(f"Steps per epoch: {steps_per_epoch}")
    print(f"Expected total steps: {expected_steps}")
    
    # Initialize wandb
    run_name = f"qwen25-coder3b-mixed-{num_corruptions}corrupt-{num_mg_verilog}mgverilog"
    wandb.init(
        project="rtl-smith-sft",
        name=run_name,
        config={
            "model_name": model_name,
            "dataset": "mixed",
            "num_corruptions": num_corruptions,
            "num_mg_verilog": num_mg_verilog,
            "per_device_train_batch_size": per_device_batch,
            "gradient_accumulation_steps": grad_accum,
            "trainer_effective_batch_size": trainer_effective_batch,
            "num_epochs": num_epochs,
            "learning_rate": 1e-6,
        }
    )
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = DEFAULT_PAD_TOKEN
        tokenizer.pad_token_id = tokenizer.convert_tokens_to_ids(DEFAULT_PAD_TOKEN)
    
    # Load model
    print(f"\nLoading model: {model_name}")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
    )
    
    # Setup LoRA
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=256,
        lora_alpha=256,
        lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        bias="none",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    
    # Training arguments
    output_dir = f"/matx/u/ethanboneh/qwen25_coder3b_mixed_{num_corruptions}corrupt_{num_mg_verilog}mgverilog_lora"
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=num_epochs,
        per_device_train_batch_size=per_device_batch,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=grad_accum,
        learning_rate=1e-6,
        bf16=True,
        logging_steps=10,
        save_strategy="steps",
        save_steps=100,
        save_total_limit=3,
        eval_strategy="steps",
        eval_steps=100,
        eval_accumulation_steps=32,
        prediction_loss_only=True,
        report_to="wandb",
        load_best_model_at_end=False,
        ddp_find_unused_parameters=False,
        remove_unused_columns=False,  # Keep input/output columns for data collator
        max_steps=expected_steps,  # Set max_steps to match expected calculation
    )
    
    # Data collator
    data_collator = DataCollatorForCausalLM(
        tokenizer=tokenizer,
        source_max_len=4096,
        target_max_len=2048,
    )
    
    # Create trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["test"],
        data_collator=data_collator,
        tokenizer=tokenizer,
    )
    
    # Train
    if resume_from_checkpoint:
        print(f"\nResuming from checkpoint: {resume_from_checkpoint}")
        trainer.train(resume_from_checkpoint=resume_from_checkpoint)
    else:
        print("\nStarting training from scratch...")
        trainer.train()
    
    # Save final model
    print(f"\nSaving model to {output_dir}")
    trainer.save_model()
    tokenizer.save_pretrained(output_dir)
    
    print(f"\nTraining complete! Model saved to: {output_dir}")
    wandb.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train model on mixed dataset (corruptions + MG-Verilog)")
    parser.add_argument(
        "--resume-from-checkpoint",
        type=str,
        default=None,
        help="Path to checkpoint to resume from (default: None)"
    )
    parser.add_argument(
        "--num-epochs",
        type=int,
        default=3,
        help="Number of training epochs (default: 3)"
    )
    parser.add_argument(
        "--num-corruptions",
        type=int,
        default=250,
        help="Number of corruptions samples to use (default: 250)"
    )
    parser.add_argument(
        "--num-mg-verilog",
        type=int,
        default=250,
        help="Number of MG-Verilog samples to use (default: 250)"
    )
    args = parser.parse_args()
    
    train(
        resume_from_checkpoint=args.resume_from_checkpoint,
        num_epochs=args.num_epochs,
        num_corruptions=args.num_corruptions,
        num_mg_verilog=args.num_mg_verilog
    )

