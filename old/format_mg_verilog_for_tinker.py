#!/usr/bin/env python3
"""
Format MG-Verilog dataset using the same formatting as sft_mg_verilog.py
and save it for use with tinker.
"""

import re
import os
from datasets import Dataset

# ==== Constants ====
dataset_path = "/matx/u/ethanboneh/mg_verilog_dataset.arrow"
output_dir = "/matx/u/ethanboneh"
output_filename = "mg_verilog_formatted"

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
    
    # Return prompt and completion as strings
    return {
        "input": user_content,
        "output": "[BEGIN]\n" + full_code.strip() + "\n[DONE]"
    }

# ======================================================================
# Main
# ======================================================================

def main():
    print("Loading dataset...")
    dataset = Dataset.from_file(dataset_path)
    print(f"Dataset size before formatting: {len(dataset)}")
    
    print("Formatting dataset...")
    dataset = dataset.map(
        format_example,
        remove_columns=dataset.column_names,
        num_proc=8,
    )
    print(f"Dataset size after formatting: {len(dataset)}")
    
    print(f"Dataset columns: {dataset.column_names}")
    print(f"\nExample formatted input (first 500 chars):")
    print(dataset[0]['input'][:500])
    print(f"\nExample formatted output (first 500 chars):")
    print(dataset[0]['output'][:500])
    
    # Save in multiple formats for compatibility
    print(f"\nSaving formatted dataset to {output_dir}...")
    os.makedirs(output_dir, exist_ok=True)
    
    # Save as Arrow format (efficient, preserves types)
    arrow_path = os.path.join(output_dir, f"{output_filename}.arrow")
    dataset.save_to_disk(arrow_path)
    print(f"✓ Saved Arrow format: {arrow_path}")
    
    # Save as JSONL (human-readable, compatible with many tools)
    jsonl_path = os.path.join(output_dir, f"{output_filename}.jsonl")
    dataset.to_json(jsonl_path)
    print(f"✓ Saved JSONL format: {jsonl_path}")
    
    # Save as Parquet (efficient, widely supported)
    parquet_path = os.path.join(output_dir, f"{output_filename}.parquet")
    dataset.to_parquet(parquet_path)
    print(f"✓ Saved Parquet format: {parquet_path}")
    
    print(f"\n✓ Done! Formatted dataset saved to {output_dir}")
    print(f"  - Arrow: {arrow_path}")
    print(f"  - JSONL: {jsonl_path}")
    print(f"  - Parquet: {parquet_path}")

if __name__ == "__main__":
    main()

