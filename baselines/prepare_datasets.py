#!/usr/bin/env python3
"""
Prepare and format VeriThoughts datasets for training with Tinker.

This script:
1. Downloads datasets from HuggingFace
2. Converts them to the conversation format expected by Tinker
3. Saves formatted datasets as JSONL files

FORMAT:
- User message: The instruction/spec (simple, no extra system prompts)
- Assistant message: 
  - Reasoning: <think>...</think> followed by [BEGIN]...[DONE]
  - Instruction: [BEGIN]...[DONE] only

The [BEGIN]/[DONE] markers are extracted by our custom model factory during CVDP evaluation.
"""

import json
import os
import re
import argparse
from typing import Optional

from datasets import load_dataset, Dataset
from tqdm import tqdm

from config import (
    HF_DATASETS,
    DATASETS_DIR,
    create_directories,
)


def extract_code_from_output(output: str) -> str:
    """
    Extract the code portion from VeriThoughts output.
    The datasets use CODE BEGIN...CODE END format.
    """
    code_match = re.search(
        r'CODE\s*BEGIN\s*(.*?)\s*CODE\s*END',
        output,
        re.DOTALL | re.IGNORECASE
    )
    if code_match:
        return code_match.group(1).strip()
    return output.strip()


def extract_reasoning_from_output(output: str) -> str:
    """
    Extract the reasoning portion (before CODE BEGIN) from VeriThoughts output.
    """
    # Find where CODE BEGIN starts
    match = re.search(r'CODE\s*BEGIN', output, re.IGNORECASE)
    if match:
        reasoning = output[:match.start()].strip()
        return reasoning
    return ""


def format_reasoning_example(row: dict) -> dict:
    """
    Format a reasoning dataset example into conversation format.
    
    Input: instruction, input, output (with <think> reasoning + CODE BEGIN...CODE END)
    Output: {"messages": [...]} with <think>...</think> + [BEGIN]...[DONE]
    """
    instruction = row["instruction"]
    input_text = row.get("input", "")
    output = row["output"]
    
    # Build user message
    user_content = instruction.strip()
    if input_text and input_text.strip():
        user_content += "\n\n" + input_text.strip()
    
    # Extract reasoning and code from output
    reasoning = extract_reasoning_from_output(output)
    code = extract_code_from_output(output)
    
    # Build assistant response with [BEGIN]/[DONE] markers
    if reasoning:
        # Keep <think> tags if present, otherwise wrap reasoning
        if '<think>' in reasoning.lower():
            assistant_content = f"{reasoning}\n[BEGIN]\n{code}\n[DONE]"
        else:
            assistant_content = f"<think>\n{reasoning}\n</think>\n[BEGIN]\n{code}\n[DONE]"
    else:
        assistant_content = f"[BEGIN]\n{code}\n[DONE]"
    
    messages = [
        {"role": "user", "content": user_content},
        {"role": "assistant", "content": assistant_content}
    ]
    
    return {"messages": messages}


def format_instruction_example(row: dict) -> dict:
    """
    Format an instruction dataset example into conversation format.
    
    Input: instruction, input, output (CODE BEGIN...CODE END only)
    Output: {"messages": [...]} with [BEGIN]...[DONE]
    """
    instruction = row["instruction"]
    input_text = row.get("input", "")
    output = row["output"]
    
    # Build user message
    user_content = instruction.strip()
    if input_text and input_text.strip():
        user_content += "\n\n" + input_text.strip()
    
    # Extract code and wrap with [BEGIN]/[DONE]
    code = extract_code_from_output(output)
    assistant_content = f"[BEGIN]\n{code}\n[DONE]"
    
    messages = [
        {"role": "user", "content": user_content},
        {"role": "assistant", "content": assistant_content}
    ]
    
    return {"messages": messages}


def prepare_dataset(
    dataset_type: str,
    subsample_fraction: float = 1.0,
    seed: int = 42,
    output_dir: Optional[str] = None,
) -> str:
    """
    Prepare a dataset for training.
    
    Args:
        dataset_type: "reasoning" or "instruction"
        subsample_fraction: Fraction of dataset to use (for subsampling experiments)
        seed: Random seed for shuffling/subsampling
        output_dir: Directory to save formatted dataset
        
    Returns:
        Path to the saved JSONL file
    """
    if output_dir is None:
        output_dir = DATASETS_DIR
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Load dataset from HuggingFace
    hf_dataset_name = HF_DATASETS[dataset_type]
    print(f"Loading dataset: {hf_dataset_name}")
    dataset = load_dataset(hf_dataset_name, split="train")
    original_size = len(dataset)
    print(f"Original dataset size: {original_size}")
    
    # Shuffle dataset
    dataset = dataset.shuffle(seed=seed)
    
    # Subsample if needed
    if subsample_fraction < 1.0:
        num_samples = int(len(dataset) * subsample_fraction)
        dataset = dataset.select(range(num_samples))
        print(f"Subsampled to {len(dataset)} examples ({subsample_fraction*100:.0f}%)")
    
    # Choose formatting function
    if dataset_type == "reasoning":
        format_fn = format_reasoning_example
    else:
        format_fn = format_instruction_example
    
    # Format all examples
    print("Formatting examples...")
    formatted_data = []
    for row in tqdm(dataset, desc="Formatting"):
        try:
            formatted = format_fn(row)
            formatted_data.append(formatted)
        except Exception as e:
            print(f"Error formatting row: {e}")
            continue
    
    print(f"Successfully formatted {len(formatted_data)} examples")
    
    # Build output filename
    fraction_str = f"_{int(subsample_fraction*100)}pct" if subsample_fraction < 1.0 else ""
    filename = f"verithoughts_{dataset_type}{fraction_str}_formatted.jsonl"
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
    print(f"User (first 300 chars):\n{sample['messages'][0]['content'][:300]}...")
    print(f"\nAssistant (first 500 chars):\n{sample['messages'][1]['content'][:500]}...")
    print("--- End sample ---\n")
    
    return output_path


def main():
    parser = argparse.ArgumentParser(description="Prepare VeriThoughts datasets for training")
    parser.add_argument(
        "--dataset", 
        type=str, 
        choices=["reasoning", "instruction", "both"],
        default="both",
        help="Which dataset to prepare"
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
    
    # Determine which datasets to prepare
    if args.dataset == "both":
        datasets_to_prepare = ["reasoning", "instruction"]
    else:
        datasets_to_prepare = [args.dataset]
    
    # Prepare datasets
    for dataset_type in datasets_to_prepare:
        for fraction in args.subsample:
            print(f"\n{'='*60}")
            print(f"Preparing {dataset_type} dataset (fraction={fraction})")
            print('='*60)
            output_path = prepare_dataset(
                dataset_type=dataset_type,
                subsample_fraction=fraction,
                seed=args.seed,
                output_dir=args.output_dir,
            )
            print(f"Output: {output_path}\n")


if __name__ == "__main__":
    main()
