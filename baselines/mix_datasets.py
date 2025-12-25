#!/usr/bin/env python3
"""
Mix VeriThoughts and corruption datasets for training.

This script combines VeriThoughts data with corruption data in various ratios.
Supports both reasoning and instruction variants.

Usage:
    # Mix 50% VeriThoughts reasoning + 50% corruption reasoning
    python mix_datasets.py \
        --verithoughts reasoning \
        --corruption corruption_reasoning \
        --ratio 0.5 \
        --output mixed_reasoning_50pct.jsonl
    
    # Mix 80% VeriThoughts instruction + 20% corruption instruction
    python mix_datasets.py \
        --verithoughts instruction \
        --corruption corruption_instruction \
        --ratio 0.2 \
        --output mixed_instruction_20pct.jsonl
"""

import json
import os
import argparse
import random
from typing import List, Dict
from tqdm import tqdm

from config import DATASETS_DIR, create_directories


def load_jsonl(file_path: str) -> List[Dict]:
    """Load data from a JSONL file."""
    data = []
    with open(file_path, 'r') as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    return data


def mix_datasets(
    verithoughts_file: str,
    corruption_file: str,
    corruption_ratio: float,
    output_file: str,
    seed: int = 42,
    shuffle: bool = True,
) -> str:
    """
    Mix VeriThoughts and corruption datasets.
    
    Args:
        verithoughts_file: Path to formatted VeriThoughts JSONL file
        corruption_file: Path to formatted corruption JSONL file
        corruption_ratio: Fraction of data that should be corruption (0.0 to 1.0)
        output_file: Path to save mixed dataset
        seed: Random seed for shuffling
        shuffle: Whether to shuffle the final dataset
        
    Returns:
        Path to the saved mixed dataset
    """
    # Load datasets
    print(f"Loading VeriThoughts data from {verithoughts_file}...")
    verithoughts_data = load_jsonl(verithoughts_file)
    print(f"Loaded {len(verithoughts_data)} VeriThoughts examples")
    
    print(f"Loading corruption data from {corruption_file}...")
    corruption_data = load_jsonl(corruption_file)
    print(f"Loaded {len(corruption_data)} corruption examples")
    
    # Calculate how many examples from each dataset
    # We want corruption_ratio of the total to be corruption data
    # If we have V VeriThoughts and C corruption examples:
    # We want: C_used / (V_used + C_used) = corruption_ratio
    # Solving: C_used = corruption_ratio * (V_used + C_used)
    #          C_used = corruption_ratio * V_used / (1 - corruption_ratio)
    
    if corruption_ratio == 0.0:
        # Only VeriThoughts
        num_corruption = 0
        num_verithoughts = len(verithoughts_data)
    elif corruption_ratio == 1.0:
        # Only corruption
        num_corruption = len(corruption_data)
        num_verithoughts = 0
    else:
        # Calculate based on available data
        # Use all available corruption data if possible
        max_corruption = len(corruption_data)
        # Calculate how many VeriThoughts we'd need for this ratio
        num_verithoughts_for_ratio = int(max_corruption * (1 - corruption_ratio) / corruption_ratio)
        
        if num_verithoughts_for_ratio <= len(verithoughts_data):
            # We have enough VeriThoughts data
            num_corruption = max_corruption
            num_verithoughts = num_verithoughts_for_ratio
        else:
            # We don't have enough VeriThoughts, use all of it
            num_verithoughts = len(verithoughts_data)
            num_corruption = int(num_verithoughts * corruption_ratio / (1 - corruption_ratio))
            num_corruption = min(num_corruption, max_corruption)
    
    print(f"\nMixing datasets:")
    print(f"  VeriThoughts: {num_verithoughts} examples")
    print(f"  Corruption: {num_corruption} examples")
    print(f"  Total: {num_verithoughts + num_corruption} examples")
    print(f"  Actual corruption ratio: {num_corruption / (num_verithoughts + num_corruption):.2%}")
    
    # Sample from each dataset
    random.seed(seed)
    sampled_verithoughts = random.sample(verithoughts_data, num_verithoughts)
    sampled_corruption = random.sample(corruption_data, num_corruption)
    
    # Combine
    mixed_data = sampled_verithoughts + sampled_corruption
    
    # Shuffle if requested
    if shuffle:
        random.shuffle(mixed_data)
    
    # Save
    os.makedirs(os.path.dirname(output_file) or '.', exist_ok=True)
    print(f"\nSaving mixed dataset to {output_file}...")
    with open(output_file, 'w') as f:
        for item in mixed_data:
            f.write(json.dumps(item) + "\n")
    
    print(f"✓ Saved {len(mixed_data)} examples to {output_file}")
    
    return output_file


def main():
    parser = argparse.ArgumentParser(description="Mix VeriThoughts and corruption datasets")
    parser.add_argument(
        "--verithoughts",
        type=str,
        required=True,
        help="VeriThoughts dataset type: 'reasoning' or 'instruction'"
    )
    parser.add_argument(
        "--corruption",
        type=str,
        required=True,
        help="Corruption dataset type: 'corruption_reasoning' or 'corruption_instruction'"
    )
    parser.add_argument(
        "--ratio",
        type=float,
        required=True,
        help="Ratio of corruption data (0.0 to 1.0). 0.5 means 50%% corruption, 50%% VeriThoughts"
    )
    parser.add_argument(
        "--subsample",
        type=float,
        default=1.0,
        help="Subsample fraction for VeriThoughts dataset (default: 1.0)"
    )
    parser.add_argument(
        "--corruption-subsample",
        type=float,
        default=1.0,
        help="Subsample fraction for corruption dataset (default: 1.0)"
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output JSONL file path"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for shuffling/sampling"
    )
    parser.add_argument(
        "--no-shuffle",
        dest="shuffle",
        action="store_false",
        help="Don't shuffle the final mixed dataset"
    )
    parser.add_argument(
        "--datasets-dir",
        type=str,
        default=None,
        help="Directory containing formatted datasets (default: from config)"
    )
    
    args = parser.parse_args()
    
    if args.datasets_dir is None:
        args.datasets_dir = DATASETS_DIR
    
    # Build file paths
    fraction_str = f"_{int(args.subsample*100)}pct" if args.subsample < 1.0 else ""
    verithoughts_file = os.path.join(
        args.datasets_dir,
        f"verithoughts_{args.verithoughts}{fraction_str}_formatted.jsonl"
    )
    
    corruption_fraction_str = f"_{int(args.corruption_subsample*100)}pct" if args.corruption_subsample < 1.0 else ""
    corruption_file = os.path.join(
        args.datasets_dir,
        f"{args.corruption}{corruption_fraction_str}_formatted.jsonl"
    )
    
    # Check files exist
    if not os.path.exists(verithoughts_file):
        raise FileNotFoundError(f"VeriThoughts file not found: {verithoughts_file}")
    if not os.path.exists(corruption_file):
        raise FileNotFoundError(f"Corruption file not found: {corruption_file}")
    
    # Create directories
    create_directories()
    
    # Mix datasets
    print("="*60)
    print("Mixing Datasets")
    print("="*60)
    output_path = mix_datasets(
        verithoughts_file=verithoughts_file,
        corruption_file=corruption_file,
        corruption_ratio=args.ratio,
        output_file=args.output,
        seed=args.seed,
        shuffle=args.shuffle,
    )
    print(f"\n✓ Mixed dataset saved to: {output_path}")


if __name__ == "__main__":
    main()

