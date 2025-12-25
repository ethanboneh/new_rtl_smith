#!/usr/bin/env python3
"""
Prepare and format corruption datasets for training with Tinker.

This script:
1. Loads corruption data from JSONL files (from llm_corruption pipeline)
2. Formats them into the conversation format expected by Tinker
3. Supports both with and without reasoning variants
4. Saves formatted datasets as JSONL files

FORMAT:
- User message: Bug description/instruction to fix the corrupted code
- Assistant message: 
  - With reasoning: <think>reasoning_trace</think> followed by [BEGIN]clean_code[DONE]
  - Without reasoning: [BEGIN]clean_code[DONE] only
"""

import json
import os
import argparse
from typing import Optional, List
from tqdm import tqdm

from config import DATASETS_DIR, create_directories


def format_corruption_example_with_reasoning(corruption_entry: dict) -> dict:
    """
    Format a corruption example with reasoning into conversation format.
    
    Input: corruption_entry with corrupted_code, clean_code, issue_description, reasoning_trace
    Output: {"messages": [...]} with <think>...</think> + [BEGIN]...[DONE]
    """
    corrupted_code = corruption_entry.get("corrupted_code", "")
    clean_code = corruption_entry.get("clean_code", "")
    issue_description = corruption_entry.get("issue_description", "")
    reasoning_trace = corruption_entry.get("reasoning_trace", "")
    
    # Build user message - instruction to fix the bug
    # Use issue_description if available, otherwise create a generic prompt
    if issue_description:
        user_content = f"Fix the following buggy SystemVerilog code:\n\n{corrupted_code}\n\nIssue: {issue_description}"
    else:
        user_content = f"Fix the following buggy SystemVerilog code:\n\n{corrupted_code}"
    
    # Build assistant response with reasoning and fixed code
    if reasoning_trace:
        assistant_content = f"<think>\n{reasoning_trace}\n</think>\n[BEGIN]\n{clean_code}\n[DONE]"
    else:
        # Fallback if no reasoning trace
        assistant_content = f"[BEGIN]\n{clean_code}\n[DONE]"
    
    messages = [
        {"role": "user", "content": user_content},
        {"role": "assistant", "content": assistant_content}
    ]
    
    return {"messages": messages}


def format_corruption_example_without_reasoning(corruption_entry: dict) -> dict:
    """
    Format a corruption example without reasoning into conversation format.
    
    Input: corruption_entry with corrupted_code, clean_code, issue_description
    Output: {"messages": [...]} with [BEGIN]...[DONE] only
    """
    corrupted_code = corruption_entry.get("corrupted_code", "")
    clean_code = corruption_entry.get("clean_code", "")
    issue_description = corruption_entry.get("issue_description", "")
    
    # Build user message - instruction to fix the bug
    if issue_description:
        user_content = f"Fix the following buggy SystemVerilog code:\n\n{corrupted_code}\n\nIssue: {issue_description}"
    else:
        user_content = f"Fix the following buggy SystemVerilog code:\n\n{corrupted_code}"
    
    # Build assistant response with just the fixed code
    assistant_content = f"[BEGIN]\n{clean_code}\n[DONE]"
    
    messages = [
        {"role": "user", "content": user_content},
        {"role": "assistant", "content": assistant_content}
    ]
    
    return {"messages": messages}


def prepare_corruption_dataset(
    corruption_file: str,
    include_reasoning: bool = True,
    subsample_fraction: float = 1.0,
    seed: int = 42,
    output_dir: Optional[str] = None,
) -> str:
    """
    Prepare a corruption dataset for training.
    
    Args:
        corruption_file: Path to JSONL file with corruption data
        include_reasoning: Whether to include reasoning traces
        subsample_fraction: Fraction of dataset to use
        seed: Random seed for shuffling/subsampling
        output_dir: Directory to save formatted dataset
        
    Returns:
        Path to the saved JSONL file
    """
    if output_dir is None:
        output_dir = DATASETS_DIR
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Load corruption data
    print(f"Loading corruption data from {corruption_file}...")
    corruption_data = []
    with open(corruption_file, 'r') as f:
        for line in f:
            if line.strip():
                corruption_data.append(json.loads(line))
    
    original_size = len(corruption_data)
    print(f"Original dataset size: {original_size}")
    
    # Shuffle dataset
    import random
    random.seed(seed)
    random.shuffle(corruption_data)
    
    # Subsample if needed
    if subsample_fraction < 1.0:
        num_samples = int(len(corruption_data) * subsample_fraction)
        corruption_data = corruption_data[:num_samples]
        print(f"Subsampled to {len(corruption_data)} examples ({subsample_fraction*100:.0f}%)")
    
    # Choose formatting function
    if include_reasoning:
        format_fn = format_corruption_example_with_reasoning
        dataset_type = "corruption_reasoning"
    else:
        format_fn = format_corruption_example_without_reasoning
        dataset_type = "corruption_instruction"
    
    # Format all examples
    print("Formatting examples...")
    formatted_data = []
    for entry in tqdm(corruption_data, desc="Formatting"):
        try:
            formatted = format_fn(entry)
            formatted_data.append(formatted)
        except Exception as e:
            print(f"Error formatting entry: {e}")
            continue
    
    print(f"Successfully formatted {len(formatted_data)} examples")
    
    # Build output filename
    fraction_str = f"_{int(subsample_fraction*100)}pct" if subsample_fraction < 1.0 else ""
    filename = f"{dataset_type}{fraction_str}_formatted.jsonl"
    output_path = os.path.join(output_dir, filename)
    
    # Save as JSONL
    print(f"Saving to {output_path}...")
    with open(output_path, 'w') as f:
        for item in formatted_data:
            f.write(json.dumps(item) + "\n")
    
    print(f"✓ Saved {len(formatted_data)} formatted examples to {output_path}")
    
    # Print a sample for verification
    print("\n--- Sample formatted example ---")
    sample = formatted_data[0]
    print(f"User (first 500 chars):\n{sample['messages'][0]['content'][:500]}...")
    print(f"\nAssistant (first 500 chars):\n{sample['messages'][1]['content'][:500]}...")
    print("--- End sample ---\n")
    
    return output_path


def main():
    parser = argparse.ArgumentParser(description="Prepare corruption datasets for training")
    parser.add_argument(
        "--corruption-file",
        type=str,
        required=True,
        help="Path to JSONL file with corruption data (from llm_corruption pipeline)"
    )
    parser.add_argument(
        "--include-reasoning",
        action="store_true",
        default=True,
        help="Include reasoning traces in the formatted data (default: True)"
    )
    parser.add_argument(
        "--no-reasoning",
        dest="include_reasoning",
        action="store_false",
        help="Exclude reasoning traces (instruction-only format)"
    )
    parser.add_argument(
        "--subsample",
        type=float,
        nargs="+",
        default=[1.0],
        help="Subsample fractions (e.g., 0.1 0.25 0.5 0.75 1.0)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for shuffling/subsampling"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory (default: from config)"
    )
    
    args = parser.parse_args()
    
    # Create directories
    create_directories()
    
    # Prepare datasets for each subsample fraction
    for fraction in args.subsample:
        print(f"\n{'='*60}")
        reasoning_str = "with reasoning" if args.include_reasoning else "without reasoning"
        print(f"Preparing corruption dataset {reasoning_str} (fraction={fraction})")
        print('='*60)
        output_path = prepare_corruption_dataset(
            corruption_file=args.corruption_file,
            include_reasoning=args.include_reasoning,
            subsample_fraction=fraction,
            seed=args.seed,
            output_dir=args.output_dir,
        )
        print(f"Output: {output_path}\n")


if __name__ == "__main__":
    main()

