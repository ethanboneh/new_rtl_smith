#!/usr/bin/env python3
"""
Train Qwen3-8B on VeriThoughts datasets using Tinker.

This script implements SFT (Supervised Fine-Tuning) using Tinker's
distributed training infrastructure.

Usage:
    # Train on reasoning dataset
    python train_tinker.py --dataset reasoning
    
    # Train on instruction dataset  
    python train_tinker.py --dataset instruction
    
    # Train with subsampling
    python train_tinker.py --dataset reasoning --subsample 0.25
    
    # Custom hyperparameters
    python train_tinker.py --dataset instruction --learning-rate 1e-4 --lora-rank 128
"""

import asyncio
import argparse
import json
import os
import sys
from datetime import datetime
from typing import Optional

import chz
import datasets
import tinker
from tqdm import tqdm

# Add tinker-cookbook to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "tinker-cookbook"))

from tinker_cookbook import cli_utils, model_info
from tinker_cookbook.renderers import TrainOnWhat, get_renderer
from tinker_cookbook.supervised import train
from tinker_cookbook.supervised.data import (
    SupervisedDatasetFromHFDataset,
    conversation_to_datum,
)
from tinker_cookbook.supervised.types import SupervisedDataset
from tinker_cookbook.tokenizer_utils import get_tokenizer

from config import (
    MODEL_NAME,
    RENDERER_NAME,
    DATASETS_DIR,
    CHECKPOINTS_DIR,
    WANDB_PROJECT,
    DEFAULT_TRAINING_CONFIG,
    create_directories,
)


@chz.chz
class TrainingConfig:
    """Configuration for VeriThoughts SFT training."""
    
    # Dataset parameters
    dataset_type: str = "reasoning"  # "reasoning" or "instruction"
    subsample_fraction: float = 1.0
    
    # Model parameters
    model_name: str = MODEL_NAME
    lora_rank: int = DEFAULT_TRAINING_CONFIG["lora_rank"]
    
    # Training parameters
    learning_rate: float = DEFAULT_TRAINING_CONFIG["learning_rate"]
    lr_schedule: str = DEFAULT_TRAINING_CONFIG["lr_schedule"]
    num_epochs: int = DEFAULT_TRAINING_CONFIG["num_epochs"]
    batch_size: int = DEFAULT_TRAINING_CONFIG["batch_size"]
    max_length: int = DEFAULT_TRAINING_CONFIG["max_length"]
    
    # Checkpointing
    save_every: int = DEFAULT_TRAINING_CONFIG["save_every"]
    eval_every: int = DEFAULT_TRAINING_CONFIG["eval_every"]
    infrequent_eval_every: int = DEFAULT_TRAINING_CONFIG["infrequent_eval_every"]
    
    # Paths
    log_path: Optional[str] = None
    load_checkpoint_path: Optional[str] = None
    dataset_path: Optional[str] = None  # Override to use custom dataset path
    
    # Logging
    wandb_project: Optional[str] = WANDB_PROJECT
    wandb_name: Optional[str] = None
    
    # Behavior
    behavior_if_log_dir_exists: cli_utils.LogdirBehavior = "ask"
    test_size: int = 500  # Number of examples for test set
    shuffle_seed: int = 42


def build_dataset_path(config: TrainingConfig) -> str:
    """Build the path to the formatted dataset."""
    if config.dataset_path:
        return config.dataset_path
    
    fraction_str = f"_{int(config.subsample_fraction*100)}pct" if config.subsample_fraction < 1.0 else ""
    filename = f"verithoughts_{config.dataset_type}{fraction_str}_formatted.jsonl"
    return os.path.join(DATASETS_DIR, filename)


def build_log_path(config: TrainingConfig) -> str:
    """Build the log path for training."""
    if config.log_path:
        return config.log_path
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_short = config.model_name.split("/")[-1].replace("-", "_")
    fraction_str = f"_{int(config.subsample_fraction*100)}pct" if config.subsample_fraction < 1.0 else ""
    
    run_name = f"{config.dataset_type}{fraction_str}_{model_short}_lr{config.learning_rate}_rank{config.lora_rank}_{timestamp}"
    return os.path.join(CHECKPOINTS_DIR, run_name)


def build_wandb_name(config: TrainingConfig) -> str:
    """Build the wandb run name."""
    if config.wandb_name:
        return config.wandb_name
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    fraction_str = f"_{int(config.subsample_fraction*100)}pct" if config.subsample_fraction < 1.0 else ""
    
    return f"verithoughts_{config.dataset_type}{fraction_str}_{timestamp}"


def load_dataset_from_jsonl(file_path: str, test_size: int, shuffle_seed: int):
    """
    Load dataset from JSONL file and split into train/test.
    
    Args:
        file_path: Path to JSONL file
        test_size: Number of examples for test set
        shuffle_seed: Random seed for shuffling
        
    Returns:
        Tuple of (train_dataset, test_dataset)
    """
    # Load data
    print(f"Loading dataset from {file_path}...")
    data = []
    with open(file_path, 'r') as f:
        for line in f:
            data.append(json.loads(line.strip()))
    
    print(f"Loaded {len(data)} examples")
    
    # Create HuggingFace dataset
    dataset = datasets.Dataset.from_list(data)
    
    # Shuffle
    dataset = dataset.shuffle(seed=shuffle_seed)
    
    # Split
    if test_size > 0 and len(dataset) > test_size:
        test_ds = dataset.select(range(test_size))
        train_ds = dataset.select(range(test_size, len(dataset)))
        print(f"Train: {len(train_ds)}, Test: {len(test_ds)}")
    else:
        train_ds = dataset
        test_ds = None
        print(f"Train: {len(train_ds)}, Test: 0")
    
    return train_ds, test_ds


def build_dataset_builder(config: TrainingConfig):
    """Build the dataset builder for training."""
    
    def dataset_builder() -> tuple[SupervisedDataset, SupervisedDataset | None]:
        # Get dataset path
        dataset_path = build_dataset_path(config)
        
        if not os.path.exists(dataset_path):
            raise FileNotFoundError(
                f"Dataset not found: {dataset_path}\n"
                f"Run 'python prepare_datasets.py --dataset {config.dataset_type} "
                f"--subsample {config.subsample_fraction}' first."
            )
        
        # Load dataset
        train_ds, test_ds = load_dataset_from_jsonl(
            dataset_path,
            test_size=config.test_size,
            shuffle_seed=config.shuffle_seed,
        )
        
        # Get tokenizer and renderer
        tokenizer = get_tokenizer(config.model_name)
        renderer_name = model_info.get_recommended_renderer_name(config.model_name)
        renderer = get_renderer(renderer_name, tokenizer)
        
        # Define map function
        def map_fn(row: dict) -> tinker.Datum:
            return conversation_to_datum(
                row["messages"],
                renderer,
                max_length=config.max_length,
                train_on_what=TrainOnWhat.ALL_ASSISTANT_MESSAGES,
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
                batch_size=min(len(test_ds), config.batch_size),
                map_fn=map_fn,
            )
        
        return train_dataset, test_dataset
    
    return dataset_builder


async def run_training(config: TrainingConfig):
    """Run the training loop."""
    # Set up paths
    log_path = build_log_path(config)
    wandb_name = build_wandb_name(config)
    
    # Check log directory
    cli_utils.check_log_dir(log_path, behavior_if_exists=config.behavior_if_log_dir_exists)
    
    # Build dataset builder
    dataset_builder = build_dataset_builder(config)
    
    # Create training config
    train_config = train.Config(
        log_path=log_path,
        model_name=config.model_name,
        load_checkpoint_path=config.load_checkpoint_path,
        dataset_builder=dataset_builder,
        learning_rate=config.learning_rate,
        lr_schedule=config.lr_schedule,
        num_epochs=config.num_epochs,
        lora_rank=config.lora_rank,
        evaluator_builders=[],
        infrequent_evaluator_builders=[],
        save_every=config.save_every,
        eval_every=config.eval_every,
        infrequent_eval_every=config.infrequent_eval_every,
        wandb_project=config.wandb_project,
        wandb_name=wandb_name,
    )
    
    # Print config
    print("\n" + "="*60)
    print("Training Configuration")
    print("="*60)
    print(f"  Dataset type: {config.dataset_type}")
    print(f"  Subsample: {config.subsample_fraction*100:.0f}%")
    print(f"  Model: {config.model_name}")
    print(f"  LoRA rank: {config.lora_rank}")
    print(f"  Learning rate: {config.learning_rate}")
    print(f"  Batch size: {config.batch_size}")
    print(f"  Max length: {config.max_length}")
    print(f"  Epochs: {config.num_epochs}")
    print(f"  Log path: {log_path}")
    print(f"  WandB: {config.wandb_project}/{wandb_name}")
    print("="*60 + "\n")
    
    # Save training config
    os.makedirs(log_path, exist_ok=True)
    config_dict = {
        "dataset_type": config.dataset_type,
        "subsample_fraction": config.subsample_fraction,
        "model_name": config.model_name,
        "lora_rank": config.lora_rank,
        "learning_rate": config.learning_rate,
        "lr_schedule": config.lr_schedule,
        "num_epochs": config.num_epochs,
        "batch_size": config.batch_size,
        "max_length": config.max_length,
        "test_size": config.test_size,
        "shuffle_seed": config.shuffle_seed,
        "timestamp": datetime.now().isoformat(),
    }
    with open(os.path.join(log_path, "training_config.json"), 'w') as f:
        json.dump(config_dict, f, indent=2)
    
    # Run training
    await train.main(train_config)
    
    print(f"\n✓ Training completed. Checkpoints saved to: {log_path}")
    return log_path


def main():
    parser = argparse.ArgumentParser(
        description="Train Qwen3-8B on VeriThoughts datasets using Tinker"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        choices=["reasoning", "instruction"],
        required=True,
        help="Dataset type to train on"
    )
    parser.add_argument(
        "--subsample",
        type=float,
        default=1.0,
        help="Fraction of dataset to use (e.g., 0.25 for 25%%)"
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=DEFAULT_TRAINING_CONFIG["learning_rate"],
        help="Learning rate"
    )
    parser.add_argument(
        "--lora-rank",
        type=int,
        default=DEFAULT_TRAINING_CONFIG["lora_rank"],
        help="LoRA rank"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_TRAINING_CONFIG["batch_size"],
        help="Batch size"
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=DEFAULT_TRAINING_CONFIG["max_length"],
        help="Maximum sequence length"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=DEFAULT_TRAINING_CONFIG["num_epochs"],
        help="Number of training epochs"
    )
    parser.add_argument(
        "--log-path",
        type=str,
        default=None,
        help="Custom log path"
    )
    parser.add_argument(
        "--load-checkpoint",
        type=str,
        default=None,
        help="Path to checkpoint to resume from"
    )
    parser.add_argument(
        "--dataset-path",
        type=str,
        default=None,
        help="Custom dataset path (override auto-generated path)"
    )
    parser.add_argument(
        "--wandb-project",
        type=str,
        default=WANDB_PROJECT,
        help="WandB project name"
    )
    parser.add_argument(
        "--wandb-name",
        type=str,
        default=None,
        help="WandB run name"
    )
    parser.add_argument(
        "--test-size",
        type=int,
        default=500,
        help="Number of examples for test set"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed"
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing log directory"
    )
    
    args = parser.parse_args()
    
    # Create directories
    create_directories()
    
    # Build config
    config = TrainingConfig(
        dataset_type=args.dataset,
        subsample_fraction=args.subsample,
        learning_rate=args.learning_rate,
        lora_rank=args.lora_rank,
        batch_size=args.batch_size,
        max_length=args.max_length,
        num_epochs=args.epochs,
        log_path=args.log_path,
        load_checkpoint_path=args.load_checkpoint,
        dataset_path=args.dataset_path,
        wandb_project=args.wandb_project,
        wandb_name=args.wandb_name,
        test_size=args.test_size,
        shuffle_seed=args.seed,
        behavior_if_log_dir_exists="delete" if args.overwrite else "ask",
    )
    
    # Run training
    asyncio.run(run_training(config))


if __name__ == "__main__":
    main()

