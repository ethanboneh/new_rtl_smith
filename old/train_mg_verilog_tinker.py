#!/usr/bin/env python3
"""
Train Qwen/Qwen3-4B-Instruct-2507 on MG-Verilog dataset using Tinker.

This script loads the formatted MG-Verilog dataset and trains using Tinker's
supervised learning framework.
"""

import asyncio
import json
import os
import sys
from datetime import datetime

import chz
import datasets
from tinker_cookbook import cli_utils, model_info
from tinker_cookbook.renderers import TrainOnWhat
from tinker_cookbook.supervised import train
from tinker_cookbook.supervised.data import SupervisedDatasetFromHFDataset
from tinker_cookbook.supervised.types import SupervisedDataset, SupervisedDatasetBuilder
from tinker_cookbook.tokenizer_utils import get_tokenizer
import tinker
import torch

# Add tinker-cookbook to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "tinker-cookbook"))


def convert_to_conversation_format(row: dict) -> dict:
    """Convert input/output format to conversation format with messages."""
    # Extract system message from input (first part before "Question:")
    input_text = row["input"]
    output_text = row["output"]
    
    # Split input to extract system message and user question
    if "Question:\n" in input_text:
        parts = input_text.split("Question:\n", 1)
        system_msg = parts[0].strip()
        user_content = "Question:\n" + parts[1].strip() if len(parts) > 1 else ""
    else:
        system_msg = ""
        user_content = input_text.strip()
    
    # Build messages list
    messages = []
    if system_msg:
        messages.append({"role": "system", "content": system_msg})
    messages.append({"role": "user", "content": user_content})
    messages.append({"role": "assistant", "content": output_text})
    
    return {"messages": messages}


@chz.chz
class Config:
    """Configuration for MG-Verilog SFT training with Tinker."""
    
    # Required parameters
    log_path: str = "/matx/u/ethanboneh/tinker-mg-verilog-sft"
    model_name: str = "Qwen/Qwen3-4B-Instruct-2507"
    dataset_path: str = "/matx/u/ethanboneh/mg_verilog_formatted.arrow"
    load_checkpoint_path: str | None = None
    
    # Training parameters
    learning_rate: float = 1e-6
    lr_schedule: str = "linear"
    num_epochs: int = 3
    
    # Model parameters
    lora_rank: int = 128
    
    # Infrastructure parameters
    base_url: str | None = None
    
    # Dataset parameters
    batch_size: int = 16
    max_length: int | None = 4096  # Max sequence length
    train_on_what: TrainOnWhat = TrainOnWhat.ALL_ASSISTANT_MESSAGES
    test_size: int = 1115  # Size of test set (10% of ~11k examples)
    shuffle_seed: int = 42
    
    # Checkpointing and evaluation
    save_every: int = 200
    eval_every: int = 100
    infrequent_eval_every: int = 1000
    
    # Adam optimizer parameters
    adam_beta1: float = 0.9
    adam_beta2: float = 0.95
    adam_eps: float = 1e-8
    
    # Logging parameters
    wandb_project: str | None = "rtl-smith-sft"
    wandb_name: str | None = None
    
    # Behavior
    behavior_if_log_dir_exists: cli_utils.LogdirBehavior = "ask"


def build_dataset_builder(config: Config) -> SupervisedDatasetBuilder:
    """Build a dataset builder from the formatted MG-Verilog dataset."""
    
    def dataset_builder() -> tuple[SupervisedDataset, SupervisedDataset | None]:
        # Load the formatted dataset
        print(f"Loading dataset from {config.dataset_path}...")
        # Try loading as directory first (Arrow format saved with save_to_disk)
        if os.path.isdir(config.dataset_path):
            hf_dataset = datasets.load_from_disk(config.dataset_path)
        else:
            # Fallback to single file (JSONL or Parquet)
            hf_dataset = datasets.Dataset.from_file(config.dataset_path)
        print(f"Loaded {len(hf_dataset)} examples")
        
        # Convert to conversation format
        print("Converting to conversation format...")
        conversations = [convert_to_conversation_format(row) for row in hf_dataset]
        conv_dataset = datasets.Dataset.from_list(conversations)
        
        # Shuffle
        if config.shuffle_seed is not None:
            conv_dataset = conv_dataset.shuffle(seed=config.shuffle_seed)
        
        # Split into train and test
        if config.test_size > 0 and len(conv_dataset) > config.test_size:
            test_ds = conv_dataset.take(config.test_size)
            train_ds = conv_dataset.skip(config.test_size)
            print(f"Train size: {len(train_ds)}, Test size: {len(test_ds)}")
        else:
            train_ds = conv_dataset
            test_ds = None
            print(f"Train size: {len(train_ds)}, Test size: 0")
        
        # Get tokenizer and renderer
        tokenizer = get_tokenizer(config.model_name)
        renderer_name = model_info.get_recommended_renderer_name(config.model_name)
        from tinker_cookbook.renderers import get_renderer
        renderer = get_renderer(renderer_name, tokenizer)
        
        # Map function to convert messages to Datum
        def map_fn(row: dict) -> tinker.Datum:
            from tinker_cookbook.supervised.data import conversation_to_datum
            # Messages are already in the correct format (list of dicts with role/content)
            messages = row["messages"]
            return conversation_to_datum(
                messages,
                renderer,
                max_length=config.max_length,
                train_on_what=config.train_on_what,
            )
        
        # Create supervised datasets
        train_dataset = SupervisedDatasetFromHFDataset(
            train_ds,
            batch_size=config.batch_size,
            map_fn=map_fn,
        )
        
        test_dataset = None
        if test_ds is not None:
            test_dataset = SupervisedDatasetFromHFDataset(
                test_ds,
                batch_size=len(test_ds),  # Use full test set as one batch for eval
                map_fn=map_fn,
            )
        
        return train_dataset, test_dataset
    
    return dataset_builder


def main(config: Config):
    """Main training function."""
    # Check log directory
    cli_utils.check_log_dir(config.log_path, behavior_if_exists=config.behavior_if_log_dir_exists)
    
    # Set wandb name if not provided
    if config.wandb_name is None:
        date_time = datetime.now().strftime("%Y-%m-%d-%H-%M")
        config.wandb_name = f"qwen3-4b-instruct-2507-mgverilog-tinker-{date_time}"
    
    # Build dataset builder
    dataset_builder = build_dataset_builder(config)
    
    # Create training config
    train_config = train.Config(
        log_path=config.log_path,
        model_name=config.model_name,
        load_checkpoint_path=config.load_checkpoint_path,
        dataset_builder=dataset_builder,
        learning_rate=config.learning_rate,
        lr_schedule=config.lr_schedule,
        num_epochs=config.num_epochs,
        lora_rank=config.lora_rank,
        base_url=config.base_url,
        evaluator_builders=[],
        infrequent_evaluator_builders=[],
        save_every=config.save_every,
        eval_every=config.eval_every,
        infrequent_eval_every=config.infrequent_eval_every,
        adam_beta1=config.adam_beta1,
        adam_beta2=config.adam_beta2,
        adam_eps=config.adam_eps,
        wandb_project=config.wandb_project,
        wandb_name=config.wandb_name,
    )
    
    # Run training
    print(f"Starting training with config:")
    print(f"  Model: {config.model_name}")
    print(f"  LoRA rank: {config.lora_rank}")
    print(f"  Learning rate: {config.learning_rate}")
    print(f"  Batch size: {config.batch_size}")
    print(f"  Epochs: {config.num_epochs}")
    print(f"  Max length: {config.max_length}")
    print(f"  Log path: {config.log_path}")
    
    asyncio.run(train.main(train_config))


if __name__ == "__main__":
    chz.nested_entrypoint(main, allow_hyphens=True)

