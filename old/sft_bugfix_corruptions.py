#!/usr/bin/env python3
"""
Supervised Fine-Tuning for Verilog Bug Fixing.

Given issue description, buggy code, and spec, the model learns to output clean code.
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
dataset_path = "/afs/cs.stanford.edu/u/ethanboneh/data/rtl_smith_corruptions/train_corrupt_with_spec.jsonl"

# ==== Prompt Templates ====
BUGFIX_SYSTEM_MSG = """You are a Verilog RTL designer that fixes bugs in SystemVerilog code.
Given a specification, issue description, and buggy code, you output the corrected code.
"""

BUGFIX_INSTRUCTIONS = """
Enclose your code with [BEGIN] and [DONE]. Only output the code snippet
and do NOT output anything else.
"""

# ======================================================================
# Data Formatting
# ======================================================================

def format_example(ex, include_reasoning_trace=False):
    """Format an example for bug-fixing training.
    
    Input: spec, issue_description, corrupted_code (optionally reasoning_trace)
    Output: clean_code (always just code between [BEGIN] and [DONE])
    """
    spec = ex.get("spec", "").strip()
    issue_description = ex.get("issue_description", "").strip()
    corrupted_code = ex.get("corrupted_code", "").strip()
    clean_code = ex.get("clean_code", "").strip()
    reasoning_trace = ex.get("reasoning_trace", "").strip()
    
    # Build user prompt
    # Start with spec (which already includes system message and question)
    user_content = spec
    
    # Add issue description section
    if issue_description:
        user_content += "\n\nIssue Description:\n"
        user_content += issue_description.strip() + "\n"
    
    # Add reasoning trace to input if requested
    if include_reasoning_trace and reasoning_trace:
        user_content += "\n\nReasoning Trace:\n"
        user_content += reasoning_trace.strip() + "\n"
    
    # Add buggy code section
    user_content += "\nBuggy Code:\n"
    user_content += "```systemverilog\n"
    user_content += corrupted_code.strip() + "\n"
    user_content += "```\n"
    
    # Add instructions
    user_content += BUGFIX_INSTRUCTIONS.strip() + "\n"
    user_content += "Answer:\n"
    
    # Output is always just the clean code wrapped in [BEGIN] and [DONE]
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
def train(resume_from_checkpoint=False, num_epochs=3, max_train_samples=500, include_reasoning_trace=False):
    # Detect number of GPUs early (needed for batch size calculation)
    try:
        use_ddp = dist.is_initialized()
        world_size = dist.get_world_size() if use_ddp else torch.cuda.device_count()
    except (RuntimeError, ValueError):
        # Distributed not initialized, use single GPU or device_count
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
    
    effective_batch_size = per_device_batch * grad_accum * num_gpus
    
    print(f"Training configuration:")
    print(f"  - GPUs: {num_gpus}")
    print(f"  - Per-device batch size: {per_device_batch}")
    print(f"  - Gradient accumulation steps: {grad_accum}")
    print(f"  - Effective batch size: {effective_batch_size}")
    
    # Load dataset from JSONL
    print(f"Loading dataset from {dataset_path}...")
    examples = []
    with open(dataset_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                ex = json.loads(line)
                examples.append(ex)
            except json.JSONDecodeError as e:
                print(f"Warning: Skipping invalid JSON line: {e}")
                continue
    
    print(f"Loaded {len(examples)} examples from JSONL")
    
    # Limit to max_train_samples
    if max_train_samples and max_train_samples < len(examples):
        examples = examples[:max_train_samples]
        print(f"Limited to {len(examples)} examples for training")
    
    # Format dataset
    print("Formatting dataset...")
    if include_reasoning_trace:
        print("  - Including reasoning trace in output")
    formatted_examples = []
    for ex in examples:
        try:
            formatted = format_example(ex, include_reasoning_trace=include_reasoning_trace)
            formatted_examples.append(formatted)
        except Exception as e:
            print(f"Warning: Failed to format example: {e}")
            continue
    
    print(f"Formatted {len(formatted_examples)} examples")
    
    if len(formatted_examples) == 0:
        raise ValueError("No valid examples after formatting!")
    
    # Create HuggingFace dataset
    dataset = Dataset.from_list(formatted_examples)
    
    # Split into train/test (10% test)
    dataset = dataset.train_test_split(test_size=0.1, seed=42)
    print(f"Train size: {len(dataset['train'])}, Test size: {len(dataset['test'])}")
    
    # Calculate steps per epoch
    steps_per_epoch = math.ceil(len(dataset['train']) / effective_batch_size)
    print(f"Steps per epoch: {steps_per_epoch}")
    
    # Initialize wandb
    run_name = f"qwen25-coder3b-bugfix-corruptions-{max_train_samples}samples"
    if include_reasoning_trace:
        run_name += "-with-reasoning"
    wandb.init(
        project="rtl-smith-sft",
        name=run_name,
        config={
            "model_name": model_name,
            "dataset": "corruptions",
            "max_train_samples": max_train_samples,
            "include_reasoning_trace": include_reasoning_trace,
            "per_device_train_batch_size": per_device_batch,
            "gradient_accumulation_steps": grad_accum,
            "effective_batch_size": effective_batch_size,
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
    print(f"Loading model: {model_name}")
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
    output_dir_suffix = f"{max_train_samples}samples"
    if include_reasoning_trace:
        output_dir_suffix += "_with_reasoning"
    output_dir = f"/matx/u/ethanboneh/qwen25_coder3b_bugfix_corruptions_{output_dir_suffix}_lora"
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
        print(f"Resuming from checkpoint: {resume_from_checkpoint}")
        trainer.train(resume_from_checkpoint=resume_from_checkpoint)
    else:
        print("Starting training from scratch...")
        trainer.train()
    
    # Save final model
    print(f"Saving model to {output_dir}")
    trainer.save_model()
    tokenizer.save_pretrained(output_dir)
    
    print(f"Training complete! Model saved to: {output_dir}")
    wandb.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train model for Verilog bug fixing")
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
        "--max-train-samples",
        type=int,
        default=500,
        help="Maximum number of training samples (default: 500)"
    )
    parser.add_argument(
        "--include-reasoning-trace",
        action="store_true",
        default=False,
        help="Include reasoning trace in the output along with corrected code (default: False)"
    )
    args = parser.parse_args()
    
    train(
        resume_from_checkpoint=args.resume_from_checkpoint,
        num_epochs=args.num_epochs,
        max_train_samples=args.max_train_samples,
        include_reasoning_trace=args.include_reasoning_trace
    )

