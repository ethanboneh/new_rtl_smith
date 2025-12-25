
import torch
import torch.distributed as dist
from torch.nn.utils.rnn import pad_sequence
from dataclasses import dataclass
from typing import Sequence, Dict
import copy
import os
from os import listdir
import glob
import re
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
dataset_path = "/matx/u/ethanboneh/mg_verilog_dataset.arrow"

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
# Final formatting for SFT
# ======================================================================

def format_example(ex):
    """Format an example for training using conversational prompt-completion format."""
    desc = ex["description"]
    code = ex["code"]

    # Process and clean the prompt
    prompt_part, module_header = process_prompt(desc)
    
    # Combine module header with code
    # The module_header is the module declaration line, code is the rest
    # If code already starts with module, use it as-is, otherwise prepend module_header
    if code.strip().startswith("module"):
        full_code = code.strip()
    else:
        # Ensure module_header ends with a newline or semicolon, then add code
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
    
    # Return prompt and completion as strings for the data collator
    return {
        "input": user_content,
        "output": "[BEGIN]\n" + full_code.strip() + "\n[DONE]"
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
def train(resume_from_checkpoint=False, num_epochs=3, overfit=False, overfit_size=5, max_train_samples=None):
    # Detect number of GPUs early (needed for batch size calculation)
    # Check if distributed is initialized, but handle gracefully if not
    if overfit:
        num_epochs = 30  # Overfit mode: run for 30 epochs
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
    
    # Calculate per_device_batch and grad_accum to achieve target_effective_batch
    # For 7 GPUs: 16/7 ≈ 2.28, so use per_device=2, grad_accum=1 → 2*1*7 = 14 (close)
    # For 1 GPU: 16/1 = 16, so use per_device=2, grad_accum=8 → 2*8*1 = 16
    # For 2 GPUs: 16/2 = 8, so use per_device=2, grad_accum=2 → 2*2*2 = 8 (or per_device=1, grad_accum=8 → 16)
    if num_gpus == 1:
        per_device_batch = 2
        grad_accum = 8
    elif num_gpus == 2:
        per_device_batch = 1
        grad_accum = 8
    elif num_gpus >= 4:
        # For 4+ GPUs, use per_device=1, grad_accum=4 → 1*4*4 = 16 (exact for 4 GPUs)
        # For 7 GPUs: 1*2*7 = 14 (close to 16)
        per_device_batch = 1
        grad_accum = max(1, target_effective_batch // (per_device_batch * num_gpus))
    else:
        # Fallback: 3 GPUs
        per_device_batch = 1
        grad_accum = 5  # 1*5*3 = 15 (close)
    
    effective_batch_size = per_device_batch * grad_accum * num_gpus
    
    # Initialize wandb
    base_output_dir = "/matx/u/ethanboneh"
    wandb_dir = os.path.join(base_output_dir, "wandb_logs")
    os.makedirs(wandb_dir, exist_ok=True)
    
    # Set wandb run name
    run_name = "qwen25-coder3b-mgverilog-lora"
    if overfit:
        run_name = f"{run_name}-overfit-{overfit_size}ex"
    
    wandb.init(
        project="rtl-smith-sft",
        entity="ethanboneh-stanford-university",
        name=run_name,
        dir=wandb_dir,  # Save wandb logs to /matx/u/ethanboneh/wandb_logs
        config={
            "model_name": model_name,
            "dataset_path": dataset_path,
            "source_max_len": 2048,
            "target_max_len": 1024,
            "lora_r": 256,
            "lora_alpha": 512,  # 2 * rank
            "lora_dropout": 0.05,
            "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            "per_device_train_batch_size": per_device_batch,
            "gradient_accumulation_steps": grad_accum,
            "effective_batch_size": effective_batch_size,
            "num_train_epochs": num_epochs,
            "learning_rate": 2e-6,
            "bf16": True,
            "quantization": "none",  # LoRA, not QLoRA
            "overfit_mode": overfit,
            "overfit_size": overfit_size if overfit else None,
        }
    )
    
    # Load dataset
    dataset = Dataset.from_file(dataset_path)
    
    # Format dataset using the same formatting as sft.py
    print("Formatting dataset...")
    print(f"Dataset size before formatting: {len(dataset)}")
    dataset = dataset.map(
        format_example,
        remove_columns=dataset.column_names,
        num_proc=8,
    )
    print(f"Dataset size after formatting: {len(dataset)}")

    print(f"dataset: {dataset.column_names}")
    print(f"example['input']: {dataset[0]['input']}")
    print(f"example['output']: {dataset[0]['output']}")
    
    # Split into train/test
    dataset = dataset.train_test_split(test_size=0.1)
    print(f"Total dataset size: {len(dataset['train']) + len(dataset['test'])}")
    print(f"Train size: {len(dataset['train'])}")
    print(f"Test size: {len(dataset['test'])}")
    
    # Overfit mode: use only a small subset of training data
    if overfit:
        original_train_size = len(dataset['train'])
        # Use only the first N examples for overfitting
        dataset['train'] = dataset['train'].select(range(min(overfit_size, len(dataset['train']))))
        print(f"\n{'='*80}")
        print(f"OVERFIT MODE ENABLED")
        print(f"  - Original train size: {original_train_size}")
        print(f"  - Overfit train size: {len(dataset['train'])}")
        print(f"  - This should allow the model to overfit quickly if setup is correct")
        print(f"{'='*80}\n")
    
    # Limit training dataset size if specified
    if max_train_samples is not None and max_train_samples > 0:
        original_train_size = len(dataset['train'])
        if max_train_samples < original_train_size:
            dataset['train'] = dataset['train'].select(range(max_train_samples))
            print(f"\n{'='*80}")
            print(f"LIMITING TRAINING DATASET SIZE")
            print(f"  - Original train size: {original_train_size}")
            print(f"  - Limited train size: {len(dataset['train'])}")
            print(f"{'='*80}\n")
    
    # Calculate steps per epoch (batch sizes already calculated above)
    # IMPORTANT: For model parallelism (device_map="auto"), the Trainer uses:
    # effective_bsz = per_device_batch * grad_accum (NOT multiplied by num_gpus)
    # because the model is split across GPUs, not the data
    trainer_effective_batch = per_device_batch * grad_accum
    steps_per_epoch = math.ceil(len(dataset['train']) / trainer_effective_batch)
    print(f"Training configuration:")
    print(f"  - GPUs: {num_gpus} (model parallelism with device_map='auto')")
    print(f"  - Per-device batch size: {per_device_batch}")
    print(f"  - Gradient accumulation steps: {grad_accum}")
    print(f"  - Effective batch size (for Trainer): {trainer_effective_batch} ({per_device_batch} × {grad_accum}, NOT multiplied by GPUs for model parallelism)")
    print(f"  - Steps per epoch: {steps_per_epoch}")

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token

    # Load model for LoRA (no quantization)
    # Use "auto" to distribute model across all available GPUs
    # For data parallelism (DDP), set device_map=None and let Trainer handle it
    try:
        if dist.is_initialized():
            # DDP mode: each process loads model on its assigned GPU
            device_map = None  # Let Trainer handle device placement in DDP
            print(f"DDP mode: Using GPU {dist.get_rank()} of {dist.get_world_size()}")
        else:
            # Single process or model parallelism: distribute across all GPUs
            num_gpus = torch.cuda.device_count()
            if num_gpus > 1:
                device_map = "auto"  # Automatically distributes model across available GPUs
                print(f"Model parallelism: Distributing model across {num_gpus} GPUs")
            else:
                device_map = "auto"
                print(f"Single GPU mode: Using GPU 0")
    except (RuntimeError, ValueError):
        # Distributed not initialized, use model parallelism
        num_gpus = torch.cuda.device_count()
        if num_gpus > 1:
            device_map = "auto"
            print(f"Model parallelism: Distributing model across {num_gpus} GPUs")
        else:
            device_map = "auto"
            print(f"Single GPU mode: Using GPU 0")
    
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map=device_map,
        dtype=torch.bfloat16,
        trust_remote_code=True
    )

    # LoRA configuration: r=256, alpha=2*rank=512, all attention and MLP modules
    lora_config = LoraConfig(
        r=256,  # rank (not less than 64)
        lora_alpha=512,  # alpha = 2 * rank
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],  # all attention and MLP modules
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.CAUSAL_LM
        # Note: SVD-based LoRA initialization is not directly supported in PEFT
        # If needed, consider using custom initialization or checking for EVA-based methods
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # Training arguments
    # base_output_dir already defined above for wandb
    # use_ddp and world_size already defined above
    # Only set max_steps when not resuming to override any checkpoint state
    training_args_dict = {
        "output_dir": os.path.join(base_output_dir, "qwen25_coder3b_sft_mgverilog_lora"),
        "per_device_train_batch_size": per_device_batch,  # Per GPU batch size (calculated above)
        "gradient_accumulation_steps": grad_accum,  # Calculated above to achieve effective batch size of 16
        # Effective batch size = per_device_batch_size * gradient_accumulation * num_gpus
        # Target: 16
        "num_train_epochs": num_epochs,  # 3-5 epochs as specified
        "eval_strategy": "no",
        # "eval_steps": 100,
        "save_strategy": "steps",
        "save_steps": 200,
        "save_total_limit": 3,
        "logging_steps": 10,
        "learning_rate": 2e-6,  # Learning rate as specified
        "bf16": True,
        "report_to": "wandb",
        "remove_unused_columns": False,
        "load_best_model_at_end": False,
    }
    
    # If using DDP, add DDP-specific settings
    if use_ddp:
        print(f"Running in DDP mode with {world_size} GPUs")
        training_args_dict["ddp_find_unused_parameters"] = False
        training_args_dict["ddp_backend"] = "nccl"
        # Only rank 0 should save
        try:
            if dist.get_rank() != 0:
                training_args_dict["report_to"] = "none"  # Disable wandb on non-rank-0 processes
        except (RuntimeError, ValueError):
            pass
    else:
        print(f"Running in single-process mode (can use device_map='auto' for model parallelism)")

    expected_steps = steps_per_epoch * num_epochs
    
    # Only set max_steps when not resuming to override any checkpoint state
    if not resume_from_checkpoint:
        training_args_dict["max_steps"] = expected_steps
        print(f"Setting max_steps={expected_steps} to override any checkpoint state")
    if overfit:
        training_args_dict["max_steps"] = 1000
        print(f"Overfit mode: Setting max_steps=1000 to allow model to overfit")
    
    training_args = TrainingArguments(**training_args_dict)

    # Data collator
    collator = DataCollatorForCausalLM(
        tokenizer=tokenizer,
        source_max_len=2048,
        target_max_len=1024
    )

    def compute_num_steps(train_dataset, training_args):
        # batch size AFTER gradient accumulation
        effective_bsz = training_args.per_device_train_batch_size * training_args.gradient_accumulation_steps
        num_update_steps_per_epoch = math.ceil(len(train_dataset) / effective_bsz)
        total_steps = num_update_steps_per_epoch * training_args.num_train_epochs
        return total_steps

    steps = compute_num_steps(dataset["train"], training_args)
    print("Expected total steps:", steps)
    
    class GenerationLoggingCallback(TrainerCallback):
        """Log model generations every N steps to console and wandb."""
        def __init__(self, train_dataset, tokenizer, log_steps=100, num_samples=3):
            self.train_dataset = train_dataset
            self.tokenizer = tokenizer
            self.log_steps = log_steps
            self.num_samples = num_samples
            # Sample indices at initialization (fixed samples to track progress)
            import random
            random.seed(42)  # Fixed seed for reproducibility
            self.sample_indices = random.sample(range(len(train_dataset)), min(num_samples, len(train_dataset)))
        
        def on_step_end(self, args, state, control, model=None, **kwargs):
            """Called at the end of each training step. Log generations periodically."""
            if state.global_step % self.log_steps == 0 and state.global_step > 0:
                self._log_generations(model, state.global_step)
        
        def _log_generations(self, model, step):
            """Generate and log model outputs for sample prompts."""
            if model is None:
                return
            
            # Get the underlying model if it's wrapped (e.g., PEFT)
            if hasattr(model, 'module'):
                gen_model = model.module
            else:
                gen_model = model
            
            was_training = gen_model.training
            gen_model.eval()
            generation_logs = []
            
            print(f"\n{'='*80}")
            print(f"Step {step}: Model Generations")
            print(f"{'='*80}")
            
            with torch.no_grad():
                for idx in self.sample_indices:
                    example = self.train_dataset[idx]
                    prompt = example['input']
                    expected_output = example['output']
                    
                    # Tokenize prompt (add BOS token like in data collator)
                    prompt_text = f"{self.tokenizer.bos_token}{prompt}"
                    prompt_tokens = self.tokenizer(
                        prompt_text,
                        return_tensors="pt",
                        truncation=True,
                        max_length=2048,
                        add_special_tokens=False
                    )
                    
                    # Handle device placement - try to find a device from model
                    # For model parallelism, we'll let the model handle device placement
                    try:
                        # Try to get device from first parameter
                        device = next(gen_model.parameters()).device
                        prompt_tokens = {k: v.to(device) for k, v in prompt_tokens.items()}
                    except (StopIteration, AttributeError):
                        # If model has no parameters or device_map="auto", use CPU and let model handle it
                        device = "cpu"
                        prompt_tokens = {k: v for k, v in prompt_tokens.items()}
                    
                    # Generate
                    try:
                        # Use the model's generate method (works with PEFT models)
                        outputs = gen_model.generate(
                            **prompt_tokens,
                            max_new_tokens=1024,
                            do_sample=True,
                            temperature=0.7,
                            top_p=0.9,
                            pad_token_id=self.tokenizer.pad_token_id,
                            eos_token_id=self.tokenizer.eos_token_id,
                        )
                        
                        # Decode generated text
                        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=False)
                        
                        # Remove the prompt from generated text
                        prompt_decoded = self.tokenizer.decode(prompt_tokens['input_ids'][0], skip_special_tokens=False)
                        if generated_text.startswith(prompt_decoded):
                            generated_output = generated_text[len(prompt_decoded):].strip()
                        else:
                            # Try removing just the prompt part
                            generated_output = generated_text.replace(prompt_decoded, "", 1).strip()
                        
                        # # Extract just the code part if it's wrapped in [BEGIN]...[DONE]
                        # if "[BEGIN]" in generated_output and "[DONE]" in generated_output:
                        #     begin_idx = generated_output.find("[BEGIN]")
                        #     done_idx = generated_output.find("[DONE]")
                        #     if begin_idx != -1 and done_idx != -1:
                        #         generated_output = generated_output[begin_idx+7:done_idx].strip()
                        
                    except Exception as e:
                        import traceback
                        generated_output = f"<Generation error: {str(e)}>\n{traceback.format_exc()}"
                    
                    # Print to console
                    print(f"\n--- Sample {idx} ---")
                    print(f"Prompt:\n{prompt}")
                    print(f"\nExpected Output:\n{expected_output}")
                    print(f"\nGenerated Output:\n{generated_output}")
                    print(f"\n{'─'*80}")
                    
                    # Prepare for wandb logging
                    generation_logs.append({
                        "sample_idx": idx,
                        "prompt": prompt[:800],  # Truncate for wandb
                        "expected_output": expected_output[:1500],
                        "generated_output": generated_output[:1500],
                    })
            
            # Log to wandb
            if wandb.run is not None:
                try:
                    wandb.log({
                        "generations": wandb.Table(
                            columns=["sample_idx", "prompt", "expected_output", "generated_output"],
                            data=[[
                                log["sample_idx"],
                                log["prompt"],
                                log["expected_output"],
                                log["generated_output"]
                            ] for log in generation_logs]
                        )
                    }, step=step)
                except Exception as e:
                    print(f"Warning: Failed to log to wandb: {e}")
            
            # Set back to original training mode
            if was_training:
                gen_model.train()
            print(f"{'='*80}\n")
    
    class StepPrinterCallback(TrainerCallback):
        def on_train_begin(self, args, state, control, **kwargs):
            return

    # Trainer
    trainer = Trainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["test"],
        data_collator=collator,
        callbacks=[
            StepPrinterCallback(),
            GenerationLoggingCallback(
                train_dataset=dataset["train"],
                tokenizer=tokenizer,
                log_steps=100,
                num_samples=3
            )
        ],
    )
    

    # Resume or train
    checkpoint_dir = None
    if resume_from_checkpoint:
        if os.path.isdir(training_args.output_dir):
            checkpoints = glob.glob(os.path.join(training_args.output_dir, "checkpoint-*"))
            if checkpoints:
                # Get the latest checkpoint by sorting by checkpoint number
                checkpoint_dir = max(checkpoints, key=lambda x: int(x.split("-")[-1]))
                print(f"Resuming training from checkpoint: {checkpoint_dir}")
            else:
                print(f"No checkpoints found in {training_args.output_dir}, starting from scratch...")
        else:
            print(f"Output directory {training_args.output_dir} does not exist, starting from scratch...")
    
    if checkpoint_dir:
        try:
            trainer.train(resume_from_checkpoint=checkpoint_dir)
        except (ValueError, RuntimeError) as e:
            print(f"Failed to resume from checkpoint {checkpoint_dir}: {e}")
            print("Starting training from scratch...")
            trainer.train()
    else:
        print("Starting training from scratch...")
        trainer.train()

    # Save LoRA adapter + tokenizer
    final_model_dir = os.path.join(base_output_dir, "qwen25_coder3b_sft_mgverilog_lora")
    model.save_pretrained(final_model_dir)
    tokenizer.save_pretrained(final_model_dir)
    print(f"Model and tokenizer saved to: {final_model_dir}")

# ==== Run ====
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Qwen model with LoRA on MG-Verilog dataset")
    parser.add_argument(
        "--resume-from-checkpoint",
        action="store_true",
        default=False,
        help="Resume training from the latest checkpoint in the output directory (default: False)"
    )
    parser.add_argument(
        "--num-epochs",
        type=int,
        default=3,
        help="Number of training epochs (default: 3, range: 3-5)"
    )
    parser.add_argument(
        "--overfit",
        action="store_true",
        default=False,
        help="Enable overfit mode: train on a very small subset (5 examples) to verify setup is correct (default: False)"
    )
    parser.add_argument(
        "--overfit-size",
        type=int,
        default=5,
        help="Number of examples to use in overfit mode (default: 5)"
    )
    parser.add_argument(
        "--max-train-samples",
        type=int,
        default=None,
        help="Maximum number of training samples to use (default: None, use all). Useful for quick experiments with smaller subsets."
    )
    args = parser.parse_args()
    
    # Validate epochs if specified (skip validation in overfit mode)
    if not args.overfit:
        if args.num_epochs < 3 or args.num_epochs > 5:
            print(f"Warning: num_epochs should be between 3-5, got {args.num_epochs}. Using default: 3")
            args.num_epochs = 3
    
    train(
        resume_from_checkpoint=args.resume_from_checkpoint,
        num_epochs=args.num_epochs,
        overfit=args.overfit,
        overfit_size=args.overfit_size,
        max_train_samples=args.max_train_samples
    )