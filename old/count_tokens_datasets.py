#!/usr/bin/env python3
"""
Count tokens in different dataset formats for comparison.

Counts tokens in:
1. 500 samples of MG-Verilog (spec-to-rtl format)
2. 500 samples of train corruptions without reasoning trace
3. 500 samples of train corruptions with reasoning trace
"""

import json
from pathlib import Path
from datasets import Dataset
from transformers import AutoTokenizer
import re

# Model and paths
model_name = "Qwen/Qwen2.5-Coder-3B"
mg_verilog_path = "/matx/u/ethanboneh/mg_verilog_dataset.arrow"
corruptions_path = "/afs/cs.stanford.edu/u/ethanboneh/data/rtl_smith_corruptions/train_corrupt_with_spec_headers.jsonl"

# ==== Prompt Templates (matching sft_mg_verilog.py) ====
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

# ==== Utilities (matching sft_mg_verilog.py) ====
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

# ==== Formatting functions ====
def format_mg_verilog_example(ex):
    """Format MG-Verilog example (spec-to-rtl)."""
    # Handle both raw and already-formatted datasets
    if "description" in ex and "code" in ex:
        desc = ex["description"]
        code = ex["code"]
    elif "input" in ex and "output" in ex:
        # Already formatted - extract from input/output
        # This shouldn't happen, but handle it gracefully
        raise ValueError("Example already formatted - need raw dataset with 'description' and 'code' columns")
    else:
        raise ValueError(f"Unexpected dataset structure. Available keys: {list(ex.keys())}")
    
    # Check if description is a dict (expected) or string (needs parsing)
    if isinstance(desc, str):
        # Try to parse as JSON
        try:
            import json
            desc = json.loads(desc)
        except (json.JSONDecodeError, TypeError):
            raise ValueError(f"Description is a string but not valid JSON. Type: {type(desc)}, First 100 chars: {str(desc)[:100]}")
    elif not isinstance(desc, dict):
        raise ValueError(f"Description is not a dict or string. Type: {type(desc)}")
    
    prompt_part, module_header = process_prompt(desc)
    
    # Combine module header with code
    if code.strip().startswith("module"):
        full_code = code.strip()
    else:
        if not module_header.endswith(";"):
            full_code = module_header + ";\n" + code.strip()
        else:
            full_code = module_header + "\n" + code.strip()
    
    # Build user prompt (matching sft_mg_verilog.py format_example)
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

def format_bugfix_example(ex, include_reasoning_trace=False):
    """Format bugfix example (with or without reasoning trace)."""
    spec = ex.get("spec", "").strip()
    issue_description = ex.get("issue_description", "").strip()
    corrupted_code = ex.get("corrupted_code", "").strip()
    clean_code = ex.get("clean_code", "").strip()
    reasoning_trace = ex.get("reasoning_trace", "").strip()
    
    # Build user prompt
    user_content = spec
    
    if issue_description:
        user_content += "\n\nIssue Description:\n"
        user_content += issue_description.strip() + "\n"
    
    if include_reasoning_trace and reasoning_trace:
        user_content += "\n\nReasoning Trace:\n"
        user_content += reasoning_trace.strip() + "\n"
    
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

def count_tokens(text, tokenizer):
    """Count tokens in text."""
    tokens = tokenizer.encode(text, add_special_tokens=False)
    return len(tokens)

def count_dataset_tokens(examples, format_fn, tokenizer, max_samples=500):
    """Count tokens in a dataset."""
    # Handle both Dataset and list inputs
    is_dataset = hasattr(examples, 'column_names') and hasattr(examples, '__getitem__') and hasattr(examples, '__len__')
    
    if is_dataset:
        # It's a Dataset object - access by index
        num_examples = len(examples)
        if max_samples and max_samples < num_examples:
            num_examples = max_samples
        
        indices = range(num_examples)
    else:
        # It's a list
        if max_samples and max_samples < len(examples):
            examples = examples[:max_samples]
        indices = range(len(examples))
    
    total_input_tokens = 0
    total_output_tokens = 0
    total_tokens = 0
    successful_count = 0
    
    for i in indices:
        try:
            if is_dataset:
                ex = examples[i]
            else:
                ex = examples[i]
            
            formatted = format_fn(ex)
            input_text = formatted["input"]
            output_text = formatted["output"]
            
            input_tokens = count_tokens(input_text, tokenizer)
            output_tokens = count_tokens(output_text, tokenizer)
            
            total_input_tokens += input_tokens
            total_output_tokens += output_tokens
            total_tokens += input_tokens + output_tokens
            successful_count += 1
        except Exception as e:
            print(f"Warning: Failed to format example {i}: {e}")
            continue
    
    num_examples = successful_count
    avg_input_tokens = total_input_tokens / num_examples if num_examples > 0 else 0
    avg_output_tokens = total_output_tokens / num_examples if num_examples > 0 else 0
    avg_total_tokens = total_tokens / num_examples if num_examples > 0 else 0
    
    return {
        "num_examples": num_examples,
        "total_input_tokens": total_input_tokens,
        "total_output_tokens": total_output_tokens,
        "total_tokens": total_tokens,
        "avg_input_tokens": avg_input_tokens,
        "avg_output_tokens": avg_output_tokens,
        "avg_total_tokens": avg_total_tokens,
    }

def main():
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    
    print("\n" + "="*80)
    print("1. Counting tokens in MG-Verilog dataset (spec-to-rtl format)")
    print("="*80)
    print(f"Loading dataset from {mg_verilog_path}...")
    mg_dataset = Dataset.from_file(mg_verilog_path)
    print(f"Loaded {len(mg_dataset)} examples")
    print(f"Dataset columns: {mg_dataset.column_names}")
    
    # Debug: check first example structure
    if len(mg_dataset) > 0:
        first_ex = mg_dataset[0]
        print(f"First example type: {type(first_ex)}")
        if isinstance(first_ex, dict):
            print(f"First example keys: {list(first_ex.keys())}")
            # Try to access description to see if it exists
            if "description" in first_ex:
                desc_type = type(first_ex["description"])
                print(f"Description type: {desc_type}")
                if isinstance(first_ex["description"], dict):
                    print(f"Description keys: {list(first_ex['description'].keys())[:5]}")
        else:
            print(f"First example is not a dict: {first_ex}")
    
    mg_stats = count_dataset_tokens(mg_dataset, format_mg_verilog_example, tokenizer, max_samples=500)
    print(f"\nMG-Verilog (500 samples):")
    print(f"  Total input tokens: {mg_stats['total_input_tokens']:,}")
    print(f"  Total output tokens: {mg_stats['total_output_tokens']:,}")
    print(f"  Total tokens: {mg_stats['total_tokens']:,}")
    print(f"  Avg input tokens per example: {mg_stats['avg_input_tokens']:.1f}")
    print(f"  Avg output tokens per example: {mg_stats['avg_output_tokens']:.1f}")
    print(f"  Avg total tokens per example: {mg_stats['avg_total_tokens']:.1f}")
    
    print("\n" + "="*80)
    print("2. Counting tokens in corruptions dataset (bugfix without reasoning)")
    print("="*80)
    print(f"Loading dataset from {corruptions_path}...")
    corruptions_examples = []
    with open(corruptions_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                ex = json.loads(line)
                corruptions_examples.append(ex)
            except json.JSONDecodeError:
                continue
    print(f"Loaded {len(corruptions_examples)} examples")
    
    bugfix_no_reasoning_stats = count_dataset_tokens(
        corruptions_examples,
        lambda ex: format_bugfix_example(ex, include_reasoning_trace=False),
        tokenizer,
        max_samples=500
    )
    print(f"\nBugfix without reasoning (500 samples):")
    print(f"  Total input tokens: {bugfix_no_reasoning_stats['total_input_tokens']:,}")
    print(f"  Total output tokens: {bugfix_no_reasoning_stats['total_output_tokens']:,}")
    print(f"  Total tokens: {bugfix_no_reasoning_stats['total_tokens']:,}")
    print(f"  Avg input tokens per example: {bugfix_no_reasoning_stats['avg_input_tokens']:.1f}")
    print(f"  Avg output tokens per example: {bugfix_no_reasoning_stats['avg_output_tokens']:.1f}")
    print(f"  Avg total tokens per example: {bugfix_no_reasoning_stats['avg_total_tokens']:.1f}")
    
    print("\n" + "="*80)
    print("3. Counting tokens in corruptions dataset (bugfix with reasoning)")
    print("="*80)
    
    bugfix_with_reasoning_stats = count_dataset_tokens(
        corruptions_examples,
        lambda ex: format_bugfix_example(ex, include_reasoning_trace=True),
        tokenizer,
        max_samples=500
    )
    print(f"\nBugfix with reasoning (500 samples):")
    print(f"  Total input tokens: {bugfix_with_reasoning_stats['total_input_tokens']:,}")
    print(f"  Total output tokens: {bugfix_with_reasoning_stats['total_output_tokens']:,}")
    print(f"  Total tokens: {bugfix_with_reasoning_stats['total_tokens']:,}")
    print(f"  Avg input tokens per example: {bugfix_with_reasoning_stats['avg_input_tokens']:.1f}")
    print(f"  Avg output tokens per example: {bugfix_with_reasoning_stats['avg_output_tokens']:.1f}")
    print(f"  Avg total tokens per example: {bugfix_with_reasoning_stats['avg_total_tokens']:.1f}")
    
    print("\n" + "="*80)
    print("Summary Comparison (500 samples each)")
    print("="*80)
    print(f"{'Dataset':<30} {'Input Tokens':>15} {'Output Tokens':>15} {'Total Tokens':>15}")
    print("-" * 80)
    print(f"{'MG-Verilog (spec-to-rtl)':<30} {mg_stats['total_input_tokens']:>15,} {mg_stats['total_output_tokens']:>15,} {mg_stats['total_tokens']:>15,}")
    print(f"{'Bugfix (no reasoning)':<30} {bugfix_no_reasoning_stats['total_input_tokens']:>15,} {bugfix_no_reasoning_stats['total_output_tokens']:>15,} {bugfix_no_reasoning_stats['total_tokens']:>15,}")
    print(f"{'Bugfix (with reasoning)':<30} {bugfix_with_reasoning_stats['total_input_tokens']:>15,} {bugfix_with_reasoning_stats['total_output_tokens']:>15,} {bugfix_with_reasoning_stats['total_tokens']:>15,}")
    
    print("\n" + "="*80)
    print("Average Tokens Per Example")
    print("="*80)
    print(f"{'Dataset':<30} {'Input':>12} {'Output':>12} {'Total':>12}")
    print("-" * 80)
    print(f"{'MG-Verilog (spec-to-rtl)':<30} {mg_stats['avg_input_tokens']:>12.1f} {mg_stats['avg_output_tokens']:>12.1f} {mg_stats['avg_total_tokens']:>12.1f}")
    print(f"{'Bugfix (no reasoning)':<30} {bugfix_no_reasoning_stats['avg_input_tokens']:>12.1f} {bugfix_no_reasoning_stats['avg_output_tokens']:>12.1f} {bugfix_no_reasoning_stats['avg_total_tokens']:>12.1f}")
    print(f"{'Bugfix (with reasoning)':<30} {bugfix_with_reasoning_stats['avg_input_tokens']:>12.1f} {bugfix_with_reasoning_stats['avg_output_tokens']:>12.1f} {bugfix_with_reasoning_stats['avg_total_tokens']:>12.1f}")

if __name__ == "__main__":
    main()

