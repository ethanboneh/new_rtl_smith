#!/usr/bin/env python3
"""
Main pipeline for generating LLM-based procedural modifications.

This script:
1. Loads VeriThoughts data
2. Uses LLM to corrupt code
3. Verifies corruption with linter
4. Generates issue descriptions
5. Generates reasoning traces
6. Saves results
"""

import argparse
import json
import os
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional
from datetime import datetime

# Add scripts directory to path
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, script_dir)
# Add baselines directory to path for config
baselines_dir = os.path.join(script_dir, '..', '..')
sys.path.insert(0, baselines_dir)

from llm_client import LLMClient
from linter import VerilatorLinter

try:
    from datasets import load_dataset
    from config import HF_DATASETS
    HF_AVAILABLE = True
except ImportError:
    HF_AVAILABLE = False
    print("Warning: 'datasets' package not available. Install with: pip install datasets")


def load_verithoughts_from_hf(dataset_type: str = "reasoning", split: str = "train") -> List[Dict]:
    """
    Load VeriThoughts dataset from HuggingFace.
    
    Args:
        dataset_type: "reasoning" or "instruction"
        split: Dataset split (default: "train")
    """
    if not HF_AVAILABLE:
        raise ImportError("'datasets' package required. Install with: pip install datasets")
    
    hf_dataset_name = HF_DATASETS.get(dataset_type)
    if not hf_dataset_name:
        raise ValueError(f"Unknown dataset type: {dataset_type}. Must be 'reasoning' or 'instruction'")
    
    print(f"Loading dataset from HuggingFace: {hf_dataset_name}")
    dataset = load_dataset(hf_dataset_name, split=split)
    
    # Convert to list of dicts
    data = [dict(row) for row in dataset]
    print(f"Loaded {len(data)} examples")
    
    return data


def load_verithoughts_from_jsonl(data_path: str) -> List[Dict]:
    """
    Load VeriThoughts dataset from local JSONL file.
    
    Expected format: JSONL file with entries containing 'code' or similar fields.
    """
    data = []
    with open(data_path, 'r') as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    return data


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


def extract_code_from_entry(entry: Dict) -> Optional[str]:
    """
    Extract SystemVerilog code from a VeriThoughts entry.
    
    VeriThoughts format has:
    - 'output': Contains "CODE BEGIN...CODE END" wrapped code
    - 'instruction': The prompt/instruction
    - 'input': Optional additional input
    """
    # VeriThoughts format: output field contains CODE BEGIN...CODE END
    if 'output' in entry:
        output = entry['output']
        if isinstance(output, str):
            code = extract_code_from_output(output)
            if code:
                return code
    
    # Fallback: try other common field names
    if 'code' in entry:
        return entry['code']
    elif 'response' in entry:
        return extract_code_from_output(entry['response'])
    
    return None


def extract_spec_from_entry(entry: Dict) -> Optional[str]:
    """
    Extract the specification/instruction from a VeriThoughts entry.
    
    VeriThoughts format has:
    - 'instruction': The prompt/specification describing what to build
    - 'input': Optional additional input
    """
    spec_parts = []
    
    if 'instruction' in entry:
        spec_parts.append(entry['instruction'])
    
    if 'input' in entry and entry['input']:
        spec_parts.append(f"\nAdditional Input:\n{entry['input']}")
    
    return '\n'.join(spec_parts) if spec_parts else None


def process_entry_from_spec(
    entry: Dict,
    llm_client: LLMClient,
    linter: Optional[VerilatorLinter],
    max_retries: int = 3,
    skip_linter: bool = False
) -> Optional[Dict]:
    """
    Generate buggy code from spec only (no gold code given to LLM).
    
    This creates more diverse/nuanced bugs since the LLM writes code
    from scratch rather than modifying existing code.
    
    Returns:
        Dictionary with corruption data or None if failed
    """
    clean_code = extract_code_from_entry(entry)
    if not clean_code:
        print(f"  Warning: Could not extract gold code from entry")
        return None
    
    spec_description = extract_spec_from_entry(entry)
    if not spec_description:
        print(f"  Warning: Could not extract spec from entry")
        return None
    
    for attempt in range(max_retries):
        try:
            print(f"  Attempt {attempt + 1}/{max_retries}: Generating buggy code from spec...")
            
            feedback = ""
            if attempt > 0:
                feedback = (
                    "\n\nPREVIOUS ATTEMPT FAILED. "
                    "Make sure to:\n"
                    "1. Match the module interface EXACTLY as specified\n"
                    "2. Introduce a subtle but real bug\n"
                    "3. Do not include any comments\n"
                    "4. Ensure the code compiles\n"
                )
            
            # Generate buggy code from spec
            buggy_result = llm_client.generate_buggy_from_spec(
                spec_description,
                feedback=feedback
            )
            buggy_code = buggy_result['buggy_code']
            
            if not buggy_code:
                print(f"    Failed: No buggy code generated")
                continue
            
            # Linter verification (optional)
            lint_result = None
            if not skip_linter and linter is not None:
                print(f"    Verifying with linter...")
                buggy_lint = linter.lint_code(buggy_code)
                lint_result = {
                    'buggy_violated_rules': buggy_lint['violated_rules'],
                    'buggy_has_errors': buggy_lint['has_errors'],
                }
            elif skip_linter:
                print(f"    Skipping linter verification...")
                lint_result = {
                    'buggy_violated_rules': set(),
                    'buggy_has_errors': False,
                }
            
            # Generate issue description
            print(f"    Generating issue description...")
            issue_description = llm_client.generate_issue_description(
                clean_code,
                buggy_code
            )
            
            # Generate reasoning trace
            print(f"    Generating reasoning trace...")
            reasoning_trace = llm_client.generate_reasoning_trace(
                clean_code,
                buggy_code,
                issue_description
            )
            
            # Success!
            return {
                'original_entry': entry,
                'generation_mode': 'from_spec',
                'clean_code': clean_code,
                'corrupted_code': buggy_code,
                'corruption_explanation': f"Bug Type: {buggy_result['bug_type']}\n{buggy_result['bug_description']}",
                'bug_type': buggy_result['bug_type'],
                'bug_description': buggy_result['bug_description'],
                'issue_description': issue_description,
                'reasoning_trace': reasoning_trace,
                'lint_result': {
                    'buggy_violated_rules': list(lint_result['buggy_violated_rules']) if lint_result else [],
                    'buggy_has_errors': lint_result['buggy_has_errors'] if lint_result else False,
                    'linter_skipped': skip_linter,
                },
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            print(f"    Error: {e}")
            import traceback
            traceback.print_exc()
            if attempt == max_retries - 1:
                print(f"  Failed after {max_retries} attempts")
                return None
    
    return None


def process_entry(
    entry: Dict,
    llm_client: LLMClient,
    linter: Optional[VerilatorLinter],
    max_retries: int = 3,
    require_lint_violations: bool = True,
    skip_linter: bool = False
) -> Optional[Dict]:
    """
    Process a single VeriThoughts entry to generate corruption.
    
    Returns:
        Dictionary with corruption data or None if failed
    """
    clean_code = extract_code_from_entry(entry)
    if not clean_code:
        print(f"  Warning: Could not extract code from entry")
        return None
    
    # Try to corrupt the code
    for attempt in range(max_retries):
        try:
            print(f"  Attempt {attempt + 1}/{max_retries}: Generating corruption...")
            
            # Generate corruption with feedback on previous attempts
            feedback = ""
            if attempt > 0:
                # Provide feedback about what went wrong
                if linter is not None:
                    clean_lint = linter.lint_code(clean_code)
                    feedback = (
                        f"\n\nPREVIOUS ATTEMPT FAILED: The corruption did not introduce new lint violations. "
                        f"Clean code has violations: {list(clean_lint['violated_rules'])}. "
                        f"You MUST introduce bugs that trigger Verilator warnings like LATCH, BLKSEQ, WIDTH, or MULTIDRIVEN. "
                        f"Try: (1) Change always_ff to always_comb and remove else clauses, "
                        f"(2) Use blocking = in always_ff blocks, (3) Drive same signal from multiple always blocks, "
                        f"or (4) Create width mismatches in assignments.\n"
                    )
                else:
                    # Generic feedback when linter is skipped
                    feedback = (
                        f"\n\nPREVIOUS ATTEMPT FAILED: The corruption did not generate valid code or the code extraction failed. "
                        f"Please ensure the output is valid SystemVerilog code wrapped in a code block (```). "
                        f"Try introducing bugs that violate RTL design principles like: "
                        f"(1) Change always_ff to always_comb and remove else clauses, "
                        f"(2) Use blocking = in always_ff blocks, (3) Drive same signal from multiple always blocks, "
                        f"or (4) Create width mismatches in assignments.\n"
                    )
            
            corruption_result = llm_client.corrupt_code(clean_code, feedback=feedback)
            corrupted_code = corruption_result['corrupted_code']
            
            if not corrupted_code:
                print(f"    Failed: No corrupted code generated")
                continue
            
            # Verify with linter (if enabled)
            lint_result = None
            if not skip_linter and linter is not None:
                print(f"    Verifying with linter...")
                lint_result = linter.verify_corruption(
                    clean_code,
                    corrupted_code,
                    require_new_violations=require_lint_violations
                )
                
                if not lint_result['is_valid']:
                    print(f"    Failed: Corruption does not introduce new lint violations")
                    print(f"    Clean rules: {lint_result['clean_violated_rules']}")
                    print(f"    Corrupt rules: {lint_result['corrupt_violated_rules']}")
                    print(f"    New rules: {lint_result['new_violated_rules']}")
                    continue
            elif skip_linter:
                print(f"    Skipping linter verification...")
                # Create a dummy lint result
                lint_result = {
                    'new_violated_rules': set(),
                    'corrupt_violated_rules': set(),
                    'clean_has_errors': False,
                    'corrupt_has_errors': False,
                }
            
            # Generate issue description
            print(f"    Generating issue description...")
            issue_description = llm_client.generate_issue_description(
                clean_code,
                corrupted_code
            )
            
            # Generate reasoning trace
            print(f"    Generating reasoning trace...")
            reasoning_trace = llm_client.generate_reasoning_trace(
                clean_code,
                corrupted_code,
                issue_description
            )
            
            # Success!
            return {
                'original_entry': entry,
                'clean_code': clean_code,
                'corrupted_code': corrupted_code,
                'corruption_explanation': corruption_result['explanation'],
                'issue_description': issue_description,
                'reasoning_trace': reasoning_trace,
                'lint_result': {
                    'new_violated_rules': list(lint_result['new_violated_rules']) if lint_result else [],
                    'all_corrupt_rules': list(lint_result['corrupt_violated_rules']) if lint_result else [],
                    'clean_has_errors': lint_result['clean_has_errors'] if lint_result else False,
                    'corrupt_has_errors': lint_result['corrupt_has_errors'] if lint_result else False,
                    'linter_skipped': skip_linter,
                },
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            print(f"    Error: {e}")
            if attempt == max_retries - 1:
                print(f"  Failed after {max_retries} attempts")
                return None
    
    return None


def main():
    parser = argparse.ArgumentParser(
        description="Generate LLM-based procedural modifications for VeriThoughts data. "
                    "Can load from HuggingFace datasets or local JSONL files."
    )
    parser.add_argument(
        "--input",
        type=str,
        default=None,
        help="Path to VeriThoughts JSONL file (optional if using --dataset-type)"
    )
    parser.add_argument(
        "--dataset-type",
        type=str,
        choices=["reasoning", "instruction"],
        default=None,
        help="Load from HuggingFace dataset (reasoning or instruction)"
    )
    parser.add_argument(
        "--split",
        type=str,
        default="train",
        help="Dataset split (default: train)"
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output JSONL file path"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="o3-mini",
        help="LLM model name (default: o3-mini)"
    )
    parser.add_argument(
        "--api-key",
        type=str,
        default=None,
        help="OpenAI API key (default: from OPENAI_API_KEY env var)"
    )
    parser.add_argument(
        "--base-url",
        type=str,
        default=None,
        help="API base URL (default: OpenAI)"
    )
    parser.add_argument(
        "--max-entries",
        type=int,
        default=None,
        help="Maximum number of entries to process"
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=3,
        help="Maximum retries per entry"
    )
    parser.add_argument(
        "--require-lint",
        action="store_true",
        default=True,
        help="Require new lint violations (default: True)"
    )
    parser.add_argument(
        "--no-require-lint",
        dest="require_lint",
        action="store_false",
        help="Don't require new lint violations"
    )
    parser.add_argument(
        "--skip-linter",
        action="store_true",
        default=False,
        help="Skip linter verification entirely (faster, but no lint validation)"
    )
    parser.add_argument(
        "--verilator-bin",
        type=str,
        default=None,
        help="Path to verilator binary (default: 'verilator' from PATH)"
    )
    parser.add_argument(
        "--corruptions-per-sample",
        type=int,
        default=1,
        help="Number of different corruptions to generate per input sample (default: 1)"
    )
    parser.add_argument(
        "--from-spec",
        action="store_true",
        default=False,
        help="Generate buggy code from spec only (no gold code given to LLM). "
             "This creates more diverse/nuanced bugs since LLM writes from scratch."
    )
    
    args = parser.parse_args()
    
    # Initialize clients
    print("Initializing LLM client...")
    llm_client = LLMClient(
        model=args.model,
        api_key=args.api_key,
        base_url=args.base_url
    )
    
    # Initialize linter (if not skipping)
    linter = None
    if not args.skip_linter:
        print("Initializing linter...")
        linter = VerilatorLinter(verilator_bin=args.verilator_bin)
    else:
        print("Skipping linter initialization (--skip-linter enabled)")
    
    # Load data
    if args.dataset_type:
        # Load from HuggingFace
        data = load_verithoughts_from_hf(
            dataset_type=args.dataset_type,
            split=args.split
        )
    elif args.input:
        # Load from local JSONL file
        print(f"Loading data from {args.input}...")
        data = load_verithoughts_from_jsonl(args.input)
        print(f"Loaded {len(data)} entries")
    else:
        parser.error("Must provide either --input (JSONL file) or --dataset-type (HuggingFace)")
    
    if args.max_entries:
        data = data[:args.max_entries]
        print(f"Processing first {len(data)} entries")
    
    # Print mode info
    if args.from_spec:
        print("\n*** FROM-SPEC MODE: Generating buggy code from specification only ***")
        print("    (LLM will not see gold code, creating more diverse bugs)\n")
    
    # Process entries
    results = []
    successful = 0
    failed = 0
    corruptions_per_sample = args.corruptions_per_sample
    
    for i, entry in enumerate(data):
        print(f"\nProcessing entry {i + 1}/{len(data)} (generating up to {corruptions_per_sample} corruptions)...")
        
        entry_successes = 0
        for corruption_idx in range(corruptions_per_sample):
            if corruptions_per_sample > 1:
                print(f"  Corruption {corruption_idx + 1}/{corruptions_per_sample}:")
            
            # Choose processing function based on mode
            if args.from_spec:
                result = process_entry_from_spec(
                    entry,
                    llm_client,
                    linter,
                    max_retries=args.max_retries,
                    skip_linter=args.skip_linter
                )
            else:
                result = process_entry(
                    entry,
                    llm_client,
                    linter,
                    max_retries=args.max_retries,
                    require_lint_violations=args.require_lint,
                    skip_linter=args.skip_linter
                )
            
            if result:
                # Add corruption index to result for tracking
                result['corruption_index'] = corruption_idx
                results.append(result)
                entry_successes += 1
                
                if args.from_spec:
                    bug_type = result.get('bug_type', 'unknown')
                    print(f"    ✓ Success! Bug type: {bug_type}")
                elif args.skip_linter:
                    print(f"    ✓ Success! (linter skipped)")
                else:
                    print(f"    ✓ Success! Violated rules: {result['lint_result']['new_violated_rules']}")
            else:
                print(f"    ✗ Failed")
        
        if entry_successes > 0:
            successful += 1
        else:
            failed += 1
        
        print(f"  Entry result: {entry_successes}/{corruptions_per_sample} corruptions generated")
    
    # Save results
    print(f"\nSaving results to {args.output}...")
    os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)
    
    with open(args.output, 'w') as f:
        for result in results:
            f.write(json.dumps(result) + '\n')
    
    print(f"\nSummary:")
    print(f"  Mode: {'from-spec' if args.from_spec else 'corrupt-gold'}")
    print(f"  Total input entries: {len(data)}")
    print(f"  Corruptions per sample: {corruptions_per_sample}")
    print(f"  Entries with at least one success: {successful}")
    print(f"  Entries with all failures: {failed}")
    print(f"  Total corruptions generated: {len(results)}")
    print(f"  Entry success rate: {successful/len(data)*100:.1f}%")
    if len(data) * corruptions_per_sample > 0:
        print(f"  Corruption success rate: {len(results)/(len(data)*corruptions_per_sample)*100:.1f}%")
    
    # Show bug type distribution for from-spec mode
    if args.from_spec and results:
        print(f"\nBug type distribution:")
        bug_types = {}
        for r in results:
            bt = r.get('bug_type', 'unknown')
            bug_types[bt] = bug_types.get(bt, 0) + 1
        for bt, count in sorted(bug_types.items(), key=lambda x: -x[1]):
            print(f"    {bt}: {count}")
    
    print(f"\nResults saved to: {args.output}")


if __name__ == "__main__":
    main()

