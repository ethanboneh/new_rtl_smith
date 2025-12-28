#!/usr/bin/env python3
"""
Prepare and format datasets for training with Tinker.

Supports:
1. VeriThoughts datasets (from HuggingFace) - reasoning and instruction
2. RTLLM dataset (local) - spec-to-code generation
3. RTLLM + Corruption dataset - combines RTLLM with bug-fixing examples

This script:
1. Downloads/loads datasets
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
from typing import Optional, List, Dict
from pathlib import Path

from datasets import load_dataset, Dataset
from tqdm import tqdm

from config import (
    HF_DATASETS,
    LOCAL_DATASETS,
    DATASETS_DIR,
    RTLLM_PATH,
    CORRUPTION_DATASETS,
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
    Output: {"messages": [...]} with separate reasoning and code messages
    
    IMPORTANT: We split reasoning and code into SEPARATE assistant messages so that
    we can use LAST_ASSISTANT_MESSAGE training and only compute loss on the code,
    not on the reasoning trace.
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
    
    messages = [{"role": "user", "content": user_content}]
    
    # Split into separate assistant messages for proper training:
    # 1. Reasoning message (NOT trained on with LAST_ASSISTANT_MESSAGE)
    # 2. Code message (trained on)
    if reasoning:
        # Keep <think> tags if present, otherwise wrap reasoning
        if '<think>' in reasoning.lower():
            reasoning_content = reasoning
        else:
            reasoning_content = f"<think>\n{reasoning}\n</think>"
        messages.append({"role": "assistant", "content": reasoning_content})
    
    # Code message - this is what we train on
    code_content = f"[BEGIN]\n{code}\n[DONE]"
    messages.append({"role": "assistant", "content": code_content})
    
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


# ==============================================================================
# RTLLM Dataset Loading
# ==============================================================================

def load_rtllm_dataset(rtllm_path: str = RTLLM_PATH) -> List[Dict]:
    """
    Load RTLLM dataset from local directory structure.
    
    RTLLM structure:
    - Category/Subcategory/design_name/
        - design_description.txt (the spec/prompt)
        - verified_<design_name>.v (the golden code)
        - testbench.v (optional)
    
    Returns:
        List of dicts with 'description' and 'code' keys
    """
    data = []
    rtllm_root = Path(rtllm_path)
    
    # Categories to scan (skip _chatgpt* and _pic folders)
    categories = ["Arithmetic", "Control", "Memory", "Miscellaneous"]
    
    for category in categories:
        category_path = rtllm_root / category
        if not category_path.exists():
            continue
            
        # Walk through all subdirectories
        for design_dir in category_path.rglob("*"):
            if not design_dir.is_dir():
                continue
                
            # Look for design_description.txt
            desc_file = design_dir / "design_description.txt"
            if not desc_file.exists():
                continue
            
            # Find the verified_*.v file
            verified_files = list(design_dir.glob("verified_*.v"))
            if not verified_files:
                continue
            
            # Read description and code
            try:
                description = desc_file.read_text().strip()
                code = verified_files[0].read_text().strip()
                
                # Extract design name from path
                design_name = design_dir.name
                subcategory = design_dir.parent.name
                
                data.append({
                    "description": description,
                    "code": code,
                    "design_name": design_name,
                    "category": category,
                    "subcategory": subcategory,
                })
            except Exception as e:
                print(f"Error reading {design_dir}: {e}")
                continue
    
    print(f"Loaded {len(data)} RTLLM examples")
    return data


def format_rtllm_example(row: dict) -> dict:
    """
    Format an RTLLM example into conversation format.
    
    Input: description, code, (optional) reasoning
    Output: {"messages": [...]} with optional reasoning + [BEGIN]...[DONE]
    
    If reasoning is present, splits into separate messages for LAST_ASSISTANT_MESSAGE training.
    """
    description = row["description"]
    code = row["code"]
    reasoning = row.get("reasoning", "")  # May be present if generated synthetically
    
    # The description already contains the full prompt
    user_content = description.strip()
    
    messages = [{"role": "user", "content": user_content}]
    
    # If reasoning is present, add it as a separate assistant message
    if reasoning:
        reasoning_content = f"<think>\n{reasoning}\n</think>"
        messages.append({"role": "assistant", "content": reasoning_content})
    
    # Code message (trained on with LAST_ASSISTANT_MESSAGE)
    code_content = f"[BEGIN]\n{code}\n[DONE]"
    messages.append({"role": "assistant", "content": code_content})
    
    return {"messages": messages}


# ==============================================================================
# Corruption Dataset Loading
# ==============================================================================

def load_corruption_dataset(corruption_path: str) -> List[Dict]:
    """
    Load corruption dataset from JSONL file.
    
    The corruption dataset contains:
    - clean_code: The correct code
    - corrupted_code: Code with a bug
    - issue_description: Description of the bug
    - reasoning_trace: How to debug and fix
    - rtllm_info.description: The original spec
    
    Returns:
        List of dicts ready for formatting
    """
    data = []
    
    if not os.path.exists(corruption_path):
        print(f"Warning: Corruption dataset not found: {corruption_path}")
        return data
    
    with open(corruption_path, 'r') as f:
        for line in f:
            if not line.strip():
                continue
            try:
                item = json.loads(line)
                data.append(item)
            except json.JSONDecodeError as e:
                print(f"Error parsing line: {e}")
                continue
    
    print(f"Loaded {len(data)} corruption examples from {corruption_path}")
    return data


def format_corruption_example_spec_to_code(row: dict) -> dict:
    """
    Format a corruption example as spec-to-code (ignore the corruption, just use clean code).
    
    This extracts the original RTLLM spec->code pair from the corruption dataset.
    """
    rtllm_info = row.get("rtllm_info", {})
    description = rtllm_info.get("description", "")
    clean_code = row.get("clean_code", "")
    
    if not description or not clean_code:
        return None
    
    user_content = description.strip()
    assistant_content = f"[BEGIN]\n{clean_code}\n[DONE]"
    
    messages = [
        {"role": "user", "content": user_content},
        {"role": "assistant", "content": assistant_content}
    ]
    
    return {"messages": messages}


def format_corruption_example_debugging(row: dict) -> dict:
    """
    Format a corruption example as a debugging task with reasoning.
    
    User provides: buggy code + issue description
    Assistant provides: reasoning (separate message) + code (trained on)
    
    IMPORTANT: We split reasoning and code into SEPARATE assistant messages so that
    we can use LAST_ASSISTANT_MESSAGE training and only compute loss on the code.
    """
    rtllm_info = row.get("rtllm_info", {})
    description = rtllm_info.get("description", "")
    corrupted_code = row.get("corrupted_code", "")
    clean_code = row.get("clean_code", "")
    issue_description = row.get("issue_description", "")
    reasoning_trace = row.get("reasoning_trace", "")
    
    if not corrupted_code or not clean_code:
        return None
    
    # Build user message: spec + buggy code + issue
    user_content = f"""The following Verilog code has a bug. Please fix it.

## Original Specification:
{description}

## Buggy Code:
```verilog
{corrupted_code}
```

## Issue Description:
{issue_description}

Please provide the corrected code."""
    
    messages = [{"role": "user", "content": user_content}]
    
    # Split into separate assistant messages for proper training:
    # 1. Reasoning message (NOT trained on with LAST_ASSISTANT_MESSAGE)
    if reasoning_trace:
        reasoning_content = f"<think>\n{reasoning_trace}\n</think>"
        messages.append({"role": "assistant", "content": reasoning_content})
    
    # 2. Code message (trained on)
    code_content = f"[BEGIN]\n{clean_code}\n[DONE]"
    messages.append({"role": "assistant", "content": code_content})
    
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
        dataset_type: "reasoning", "instruction", "rtllm", or "rtllm_corruption"
        subsample_fraction: Fraction of dataset to use (for subsampling experiments)
        seed: Random seed for shuffling/subsampling
        output_dir: Directory to save formatted dataset
        
    Returns:
        Path to the saved JSONL file
    """
    import random
    random.seed(seed)
    
    if output_dir is None:
        output_dir = DATASETS_DIR
    
    os.makedirs(output_dir, exist_ok=True)
    
    formatted_data = []
    
    # Handle different dataset types
    if dataset_type in HF_DATASETS:
        # VeriThoughts datasets from HuggingFace
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
        for row in tqdm(dataset, desc="Formatting"):
            try:
                formatted = format_fn(row)
                formatted_data.append(formatted)
            except Exception as e:
                print(f"Error formatting row: {e}")
                continue
        
        # Build output filename
        fraction_str = f"_{int(subsample_fraction*100)}pct" if subsample_fraction < 1.0 else ""
        filename = f"verithoughts_{dataset_type}{fraction_str}_formatted.jsonl"
    
    elif dataset_type == "rtllm":
        # Pure RTLLM dataset (spec -> code)
        print(f"Loading RTLLM dataset from {RTLLM_PATH}")
        rtllm_data = load_rtllm_dataset(RTLLM_PATH)
        
        # Shuffle
        random.shuffle(rtllm_data)
        
        # Subsample if needed
        if subsample_fraction < 1.0:
            num_samples = int(len(rtllm_data) * subsample_fraction)
            rtllm_data = rtllm_data[:num_samples]
            print(f"Subsampled to {len(rtllm_data)} examples ({subsample_fraction*100:.0f}%)")
        
        # Format examples
        print("Formatting RTLLM examples...")
        for row in tqdm(rtllm_data, desc="Formatting"):
            try:
                formatted = format_rtllm_example(row)
                formatted_data.append(formatted)
            except Exception as e:
                print(f"Error formatting row: {e}")
                continue
        
        # Build output filename
        fraction_str = f"_{int(subsample_fraction*100)}pct" if subsample_fraction < 1.0 else ""
        filename = f"rtllm{fraction_str}_formatted.jsonl"
    
    elif dataset_type == "rtllm_reasoning":
        # RTLLM with synthetic reasoning traces
        reasoning_path = os.path.join(DATASETS_DIR, "rtllm_with_reasoning.jsonl")
        if not os.path.exists(reasoning_path):
            raise FileNotFoundError(
                f"RTLLM with reasoning not found: {reasoning_path}\n"
                f"Generate it first with: python generate_rtllm_reasoning.py"
            )
        
        print(f"Loading RTLLM with reasoning from {reasoning_path}")
        rtllm_data = []
        with open(reasoning_path, 'r') as f:
            for line in f:
                if line.strip():
                    rtllm_data.append(json.loads(line))
        print(f"Loaded {len(rtllm_data)} examples with reasoning")
        
        # Shuffle
        random.shuffle(rtllm_data)
        
        # Subsample if needed
        if subsample_fraction < 1.0:
            num_samples = int(len(rtllm_data) * subsample_fraction)
            rtllm_data = rtllm_data[:num_samples]
            print(f"Subsampled to {len(rtllm_data)} examples ({subsample_fraction*100:.0f}%)")
        
        # Format examples (will include reasoning)
        print("Formatting RTLLM examples with reasoning...")
        for row in tqdm(rtllm_data, desc="Formatting"):
            try:
                formatted = format_rtllm_example(row)
                formatted_data.append(formatted)
            except Exception as e:
                print(f"Error formatting row: {e}")
                continue
        
        # Build output filename
        fraction_str = f"_{int(subsample_fraction*100)}pct" if subsample_fraction < 1.0 else ""
        filename = f"rtllm_reasoning{fraction_str}_formatted.jsonl"
    
    elif dataset_type == "rtllm_corruption":
        # RTLLM + Corruption dataset (combines spec->code with debugging tasks)
        print("Loading RTLLM + Corruption dataset...")
        
        # Load pure RTLLM data
        rtllm_data = load_rtllm_dataset(RTLLM_PATH)
        print(f"  RTLLM: {len(rtllm_data)} examples")
        
        # Load corruption data
        corruption_path = CORRUPTION_DATASETS.get("rtllm_from_spec")
        corruption_data = load_corruption_dataset(corruption_path) if corruption_path else []
        print(f"  Corruptions: {len(corruption_data)} examples")
        
        # Format RTLLM examples (spec -> code)
        print("Formatting RTLLM examples...")
        for row in tqdm(rtllm_data, desc="RTLLM"):
            try:
                formatted = format_rtllm_example(row)
                formatted_data.append(formatted)
            except Exception as e:
                print(f"Error formatting RTLLM row: {e}")
                continue
        
        # Format corruption examples as debugging tasks (with reasoning)
        print("Formatting corruption examples (debugging tasks)...")
        for row in tqdm(corruption_data, desc="Corruption"):
            try:
                formatted = format_corruption_example_debugging(row)
                if formatted:
                    formatted_data.append(formatted)
            except Exception as e:
                print(f"Error formatting corruption row: {e}")
                continue
        
        # Shuffle combined dataset
        random.shuffle(formatted_data)
        
        # Subsample if needed
        if subsample_fraction < 1.0:
            num_samples = int(len(formatted_data) * subsample_fraction)
            formatted_data = formatted_data[:num_samples]
            print(f"Subsampled to {len(formatted_data)} examples ({subsample_fraction*100:.0f}%)")
        
        # Build output filename
        fraction_str = f"_{int(subsample_fraction*100)}pct" if subsample_fraction < 1.0 else ""
        filename = f"rtllm_corruption{fraction_str}_formatted.jsonl"
    
    else:
        raise ValueError(f"Unknown dataset type: {dataset_type}")
    
    print(f"Successfully formatted {len(formatted_data)} examples")
    
    output_path = os.path.join(output_dir, filename)
    
    # Save as JSONL
    print(f"Saving to {output_path}...")
    with open(output_path, 'w') as f:
        for item in formatted_data:
            f.write(json.dumps(item) + "\n")
    
    print(f"✓ Saved {len(formatted_data)} formatted examples to {output_path}")
    
    # Print a sample for verification
    if formatted_data:
        print("\n--- Sample formatted example ---")
        sample = formatted_data[0]
        print(f"User (first 300 chars):\n{sample['messages'][0]['content'][:300]}...")
        print(f"\nAssistant (first 500 chars):\n{sample['messages'][1]['content'][:500]}...")
        print("--- End sample ---\n")
    
    return output_path


def main():
    parser = argparse.ArgumentParser(description="Prepare datasets for training")
    parser.add_argument(
        "--dataset", 
        type=str, 
        choices=["reasoning", "instruction", "rtllm", "rtllm_reasoning", "rtllm_corruption", "verithoughts", "all"],
        default="verithoughts",
        help="Which dataset to prepare. 'verithoughts' = reasoning+instruction, 'all' = everything, 'rtllm_reasoning' = RTLLM with synthetic reasoning"
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
    if args.dataset == "verithoughts":
        datasets_to_prepare = ["reasoning", "instruction"]
    elif args.dataset == "all":
        datasets_to_prepare = ["reasoning", "instruction", "rtllm", "rtllm_corruption"]
    else:
        datasets_to_prepare = [args.dataset]
    
    # Prepare datasets
    for dataset_type in datasets_to_prepare:
        for fraction in args.subsample:
            print(f"\n{'='*60}")
            print(f"Preparing {dataset_type} dataset (fraction={fraction})")
            print('='*60)
            try:
                output_path = prepare_dataset(
                    dataset_type=dataset_type,
                    subsample_fraction=fraction,
                    seed=args.seed,
                    output_dir=args.output_dir,
                )
                print(f"Output: {output_path}\n")
            except Exception as e:
                print(f"Error preparing {dataset_type}: {e}\n")
                continue


if __name__ == "__main__":
    main()
