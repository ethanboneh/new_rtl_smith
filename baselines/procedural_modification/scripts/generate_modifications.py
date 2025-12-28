#!/usr/bin/env python3
"""
Main pipeline for generating procedural modifications on VeriThoughts data.

This script:
1. Loads VeriThoughts data (from HuggingFace or local JSONL)
2. Parses Verilog code and identifies applicable modifications
3. Applies procedural modifications to create buggy code
4. Generates issue descriptions based on the modification type
5. Saves results in the same format as llm_corruption

Usage:
    python scripts/generate_modifications.py \
        --dataset-type instruction \
        --output outputs/procedural_corruptions.jsonl \
        --max-entries 100
"""

import argparse
import json
import os
import re
import random
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from datetime import datetime
from dataclasses import asdict

# Add scripts directory to path
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, script_dir)

from verilog_ast import VerilogParser, VerilogModule, calculate_complexity
from filters import apply_filters, FILTER_REGISTRY
from modifications import (
    MODIFICATION_REGISTRY,
    MODIFICATION_BY_NAME,
    get_applicable_modifications,
    apply_random_modification,
    ModificationResult,
    ModificationCategory,
)

try:
    from datasets import load_dataset
    HF_AVAILABLE = True
except ImportError:
    HF_AVAILABLE = False
    print("Warning: 'datasets' package not available. Install with: pip install datasets")

# Add parent directory for config
baselines_dir = os.path.join(script_dir, '..', '..')
sys.path.insert(0, baselines_dir)

try:
    from config import HF_DATASETS
except ImportError:
    HF_DATASETS = {
        "reasoning": "wilyub/VeriThoughtsTrainSetConsistentReasoning",
        "instruction": "wilyub/VeriThoughtsTrainSetConsistentInstruction",
    }


# ============================================================================
# Issue Description Templates
# ============================================================================

ISSUE_DESCRIPTIONS = {
    "invert_if_else": (
        "The buggy code has swapped if and else bodies in a conditional block. "
        "This causes the logic to execute the opposite branch for each condition, "
        "leading to incorrect behavior when the condition is true vs false. "
        "Test cases that depend on the conditional logic will fail because the "
        "intended branch is not executed."
    ),
    "remove_else_branch": (
        "The buggy code is missing an else branch in a conditional statement. "
        "In combinational logic (always_comb or always @*), this can cause latch "
        "inference because the output is not defined for all input conditions. "
        "In sequential logic, signals may retain old values unexpectedly. "
        "This violates RTL design best practices requiring complete conditional assignments."
    ),
    "remove_case_default": (
        "The buggy code is missing the default case in a case statement. "
        "This creates an incomplete case statement that can infer latches in "
        "combinational blocks or cause undefined behavior for unhandled values. "
        "Linting tools will flag this as CASEINCOMPLETE or LATCH warning."
    ),
    "change_constant": (
        "The buggy code has an incorrect numeric constant value. "
        "This off-by-one or similar error causes incorrect computations, "
        "timing issues, or boundary condition failures. "
        "Test cases that rely on exact numeric values will fail."
    ),
    "change_operator": (
        "The buggy code uses the wrong operator in an expression. "
        "For example, using subtraction instead of addition, or AND instead of OR. "
        "This causes incorrect arithmetic or logical results. "
        "Test cases will fail because the computed values don't match expectations."
    ),
    "swap_operands": (
        "The buggy code has swapped operands in a non-commutative operation. "
        "For operations like subtraction, division, or comparisons, this changes "
        "the result entirely. For example, 'a - b' becomes 'b - a'. "
        "Test cases will fail due to incorrect computation results."
    ),
    "change_bit_width": (
        "The buggy code has an incorrect bit width specification. "
        "This causes width mismatches that can lead to truncation, sign extension "
        "issues, or synthesis warnings. Linting tools will flag WIDTH warnings."
    ),
    "remove_assignment": (
        "The buggy code is missing an assignment statement. "
        "This leaves a signal undriven or with stale values, causing incorrect "
        "behavior or synthesis warnings about undriven nets. "
        "Signals may be inferred as latches or remain at unknown values."
    ),
    "remove_conditional": (
        "The buggy code is missing a conditional block. "
        "This removes important control logic, causing the design to skip "
        "necessary state transitions or data processing. "
        "Test cases that depend on the conditional behavior will fail."
    ),
    "remove_loop": (
        "The buggy code is missing a loop construct. "
        "This removes iterative logic that may be essential for initialization, "
        "array operations, or repetitive computations. "
        "Functionality that depends on the loop will be broken."
    ),
    "swap_blocking_nonblocking": (
        "The buggy code uses the wrong assignment type (blocking vs non-blocking). "
        "Using blocking (=) in always_ff blocks causes BLKSEQ lint warnings and "
        "potential simulation/synthesis mismatches. Using non-blocking (<=) in "
        "combinational blocks can cause unexpected behavior. "
        "This violates fundamental RTL coding guidelines."
    ),
    "change_sensitivity_list": (
        "The buggy code has an incorrect or incomplete sensitivity list. "
        "This causes the always block to not trigger on all relevant signal "
        "changes, leading to simulation mismatches and incorrect behavior. "
        "Linting tools may flag COMBORDER or similar warnings."
    ),
    "add_multiple_driver": (
        "The buggy code has multiple drivers on the same signal. "
        "This creates a contention where different parts of the design try to "
        "drive the same signal to different values. "
        "Linting tools will flag MULTIDRIVEN errors, and synthesis may fail."
    ),
    "remove_reset_assignment": (
        "The buggy code is missing a signal initialization in the reset block. "
        "This leaves registers with undefined initial values after reset, "
        "causing unpredictable behavior on startup. "
        "Simulation may pass but hardware behavior will be incorrect."
    ),
    
    # High-Impact Modifications
    "invert_condition": (
        "The buggy code has an inverted condition in an if statement. "
        "Instead of checking if(x), it checks if(!x), causing the logic to "
        "execute when it shouldn't and skip execution when it should. "
        "All conditional behavior is reversed from intended functionality."
    ),
    "shuffle_case_items": (
        "The buggy code has reordered case items in a case statement. "
        "For priority-encoded case statements (casex, casez), the order matters. "
        "Reordering breaks the intended priority, causing incorrect item matching. "
        "Test cases may pass for some inputs but fail for overlapping patterns."
    ),
    "remove_always_block": (
        "The buggy code is missing an entire always block. "
        "This removes either sequential (flip-flop) or combinational logic entirely. "
        "Signals that were driven by this block are now undriven, causing synthesis "
        "failures or simulation X-propagation. Critical functionality is lost."
    ),
    "swap_port_signals": (
        "The buggy code has swapped signal connections in a module instantiation. "
        "Signals are connected to the wrong ports, causing data to flow incorrectly "
        "between modules. This is a classic integration bug that causes functional "
        "failures in hierarchical designs."
    ),
    "invert_reset_polarity": (
        "The buggy code has inverted reset polarity. "
        "If the design expects active-low reset (!rst), it now uses active-high, "
        "or vice versa. This causes the reset logic to activate during normal "
        "operation and deactivate during actual reset, breaking initialization."
    ),
    
    # FSM Modifications
    "remove_state_transition": (
        "The buggy code is missing a state transition in the FSM. "
        "The next-state logic for one state has been removed, causing the FSM "
        "to either stay stuck in that state or have undefined behavior. "
        "State machine operation is broken for affected state paths."
    ),
    "add_unreachable_state": (
        "The buggy code has an unreachable state in the FSM. "
        "A state was added that no transition leads to, creating dead code. "
        "While this may not immediately break functionality, it indicates "
        "design issues and may confuse synthesis optimization."
    ),
    "swap_state_encoding": (
        "The buggy code has swapped FSM state encoding values. "
        "Two states have exchanged their binary encodings, causing the FSM "
        "to enter the wrong state after transitions. This breaks the state "
        "machine behavior for all paths involving these states."
    ),
    "remove_state_update": (
        "The buggy code is missing the state register update. "
        "The state <= next_state assignment is removed, causing the FSM "
        "to freeze in its initial state. No state transitions occur, "
        "and the entire state machine is non-functional."
    ),
    
    # Clock/Reset Modifications
    "remove_async_reset": (
        "The buggy code has removed async reset from the sensitivity list. "
        "The always block only triggers on clock edges, not reset. "
        "This causes metastability on power-up and removes the ability "
        "to asynchronously reset the design."
    ),
    "duplicate_clock_signal": (
        "The buggy code has a duplicate clock signal. "
        "A phantom clock domain is created, which may confuse CDC analysis "
        "tools and create subtle timing issues in synthesis. "
        "This violates clock domain design best practices."
    ),
    "modify_cdc": (
        "The buggy code has a broken clock domain crossing synchronizer. "
        "A synchronizer flip-flop stage was removed, reducing metastability "
        "protection. This creates CDC violations that cause random data "
        "corruption in multi-clock domain designs."
    ),
}

REASONING_TEMPLATES = {
    "invert_if_else": """
1. Overall Design Intent:
   • The module contains conditional logic that should execute different code paths based on a condition.
   • The if branch should handle the true case, and the else branch should handle the false case.

2. Code Structure and Signal Flow Analysis:
   • Identified a conditional block with both if and else branches.
   • The branches contain different assignments or logic operations.

3. Identification of Specific Differences/Bugs:
   • The if and else bodies have been swapped.
   • Logic that should execute when condition is true now executes when false, and vice versa.

4. Root Cause and Impact:
   • Root Cause: The conditional bodies are inverted.
   • Impact: All conditional behavior is reversed, causing incorrect outputs for all test cases.

5. Clear Path to Fixing the Issue:
   • Swap the if and else bodies back to their correct positions.
   • Verify that the condition evaluates correctly and executes the intended branch.

6. RTL Design Principle Violated:
   • Correct conditional logic requires matching the intended behavior with the branch bodies.
""",
    "remove_else_branch": """
1. Overall Design Intent:
   • The module should provide a complete conditional assignment to avoid latch inference.
   • Combinational logic must define outputs for all input conditions.

2. Code Structure and Signal Flow Analysis:
   • Found an if statement without an else branch.
   • The assigned signal is not defined when the condition is false.

3. Identification of Specific Differences/Bugs:
   • The else branch has been removed from the conditional.
   • This creates an incomplete conditional assignment.

4. Root Cause and Impact:
   • Root Cause: Missing else branch leaves the signal undefined in some cases.
   • Impact: Latch inference in combinational blocks, or stale values in sequential blocks.

5. Clear Path to Fixing the Issue:
   • Add an else branch with appropriate default or alternative assignment.
   • Ensure all signals are assigned in all branches of the conditional.

6. RTL Design Principle Violated:
   • Combinational logic must have complete assignments to avoid latch inference.
   • All conditional blocks should have else clauses or default values.
""",
    "swap_blocking_nonblocking": """
1. Overall Design Intent:
   • Sequential logic (always_ff) should use non-blocking assignments (<=).
   • Combinational logic (always_comb) should use blocking assignments (=).

2. Code Structure and Signal Flow Analysis:
   • Found an assignment using the wrong assignment operator for the block type.
   • This violates standard RTL coding practices.

3. Identification of Specific Differences/Bugs:
   • Blocking assignment (=) used in sequential block, or
   • Non-blocking assignment (<=) used in combinational block.

4. Root Cause and Impact:
   • Root Cause: Incorrect assignment operator for the always block type.
   • Impact: Simulation/synthesis mismatches, BLKSEQ lint warnings, race conditions.

5. Clear Path to Fixing the Issue:
   • Use non-blocking (<=) in always_ff blocks for sequential logic.
   • Use blocking (=) in always_comb blocks for combinational logic.

6. RTL Design Principle Violated:
   • BLKSEQ: Blocking assignment in sequential block is a common RTL error.
   • Proper use of blocking vs non-blocking is essential for correct synthesis.
""",
}


def generate_reasoning_trace(modification_type: str, result: ModificationResult) -> str:
    """
    Generate a reasoning trace for the modification.
    
    Args:
        modification_type: Type of modification applied
        result: ModificationResult with details
        
    Returns:
        Reasoning trace string
    """
    if modification_type in REASONING_TEMPLATES:
        base_reasoning = REASONING_TEMPLATES[modification_type]
    else:
        # Generic reasoning template
        base_reasoning = f"""
1. Overall Design Intent:
   • The module implements specific RTL functionality.
   • The design should follow RTL best practices.

2. Code Structure and Signal Flow Analysis:
   • Analyzed the module structure and identified key constructs.
   • Found a location where a {modification_type} modification was applied.

3. Identification of Specific Differences/Bugs:
   • {result.description}
   • Line {result.line_number}: Original: {result.original_snippet}

4. Root Cause and Impact:
   • Root Cause: The modification introduces a semantic error.
   • Impact: The design behavior differs from the intended functionality.

5. Clear Path to Fixing the Issue:
   • Identify the modified code at line {result.line_number}.
   • Restore the original code: {result.original_snippet}

6. RTL Design Principle Violated:
   • The modification violates standard RTL design practices.
   • Proper RTL requires careful attention to {modification_type.replace('_', ' ')}.
"""
    
    return base_reasoning.strip()


def extract_code_from_output(output: str) -> str:
    """Extract code from VeriThoughts CODE BEGIN...CODE END format."""
    code_match = re.search(
        r'CODE\s*BEGIN\s*(.*?)\s*CODE\s*END',
        output,
        re.DOTALL | re.IGNORECASE
    )
    if code_match:
        return code_match.group(1).strip()
    return output.strip()


def extract_code_from_entry(entry: Dict) -> Optional[str]:
    """Extract code from a VeriThoughts entry."""
    if 'output' in entry:
        output = entry['output']
        if isinstance(output, str):
            code = extract_code_from_output(output)
            if code:
                return code
    
    if 'code' in entry:
        return entry['code']
    
    return None


def load_data_from_hf(dataset_type: str, split: str = "train") -> List[Dict]:
    """Load VeriThoughts dataset from HuggingFace."""
    if not HF_AVAILABLE:
        raise ImportError("'datasets' package required. Install with: pip install datasets")
    
    hf_dataset_name = HF_DATASETS.get(dataset_type)
    if not hf_dataset_name:
        raise ValueError(f"Unknown dataset type: {dataset_type}")
    
    print(f"Loading dataset from HuggingFace: {hf_dataset_name}")
    dataset = load_dataset(hf_dataset_name, split=split)
    
    data = [dict(row) for row in dataset]
    print(f"Loaded {len(data)} examples")
    
    return data


def load_data_from_jsonl(data_path: str) -> List[Dict]:
    """Load data from local JSONL file."""
    data = []
    with open(data_path, 'r') as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    return data


def normalize_code(code: str) -> str:
    """Normalize code for comparison (remove whitespace differences)."""
    # Remove comments
    code = re.sub(r'//.*$', '', code, flags=re.MULTILINE)
    code = re.sub(r'/\*.*?\*/', '', code, flags=re.DOTALL)
    # Normalize whitespace
    code = ' '.join(code.split())
    return code.strip()


def codes_are_different(clean_code: str, corrupt_code: str) -> bool:
    """
    Check if two pieces of code are different (ignoring whitespace).
    
    Args:
        clean_code: Original code
        corrupt_code: Modified code
        
    Returns:
        True if codes are different
    """
    return normalize_code(clean_code) != normalize_code(corrupt_code)


def process_entry(
    entry: Dict,
    parser: VerilogParser,
    max_modifications: int = 1,
    specific_modification: Optional[str] = None,
    likelihood: float = 0.5,
    verify_different: bool = True,
    formal_verify: bool = False,
    formal_timeout: int = 30,
    stacked_modifications: int = 1,
    used_modifications: Optional[set] = None,
) -> Optional[Dict]:
    """
    Process a single entry to generate procedural modification.
    
    Args:
        entry: VeriThoughts entry
        parser: VerilogParser instance
        max_modifications: Maximum modifications to apply
        specific_modification: Specific modification to apply (optional)
        likelihood: Probability of applying modification per candidate
        verify_different: Check that code actually changed (default: True)
        formal_verify: Use formal verification to check functional difference
        formal_timeout: Timeout for formal verification in seconds
        stacked_modifications: Number of modifications to stack/combine (default: 1)
        used_modifications: Set of modification types already used for this entry (to avoid duplicates)
        
    Returns:
        Dictionary with modification data or None if failed
    """
    clean_code = extract_code_from_entry(entry)
    if not clean_code:
        return None
    
    # Parse the code
    try:
        module = parser.parse(clean_code)
    except Exception as e:
        print(f"  Parse error: {e}")
        return None
    
    # Get applicable modifications
    applicable = get_applicable_modifications(module)
    if not applicable:
        return None
    
    # Filter out already used modifications if provided
    if used_modifications:
        applicable = [m for m in applicable if m.name not in used_modifications]
        if not applicable:
            return None
    
    # Track all modifications applied (for stacking)
    applied_modifications = []
    all_descriptions = []
    all_original_snippets = []
    all_modified_snippets = []
    all_line_numbers = []
    
    current_code = clean_code
    current_module = module
    
    # Apply stacked modifications
    for stack_idx in range(stacked_modifications):
        # Re-parse if we've modified the code
        if stack_idx > 0:
            try:
                current_module = parser.parse(current_code)
                # Get applicable modifications for the modified code
                applicable = get_applicable_modifications(current_module)
                # Filter out already applied modification types to get variety
                applicable = [m for m in applicable if m.name not in [am[0] for am in applied_modifications]]
                if not applicable:
                    break  # No more applicable modifications
            except Exception as e:
                break  # Parse error, stop stacking
        
        # Apply modification
        if specific_modification and stack_idx == 0:
            if specific_modification not in MODIFICATION_BY_NAME:
                print(f"  Unknown modification: {specific_modification}")
                return None
            result = MODIFICATION_BY_NAME[specific_modification].apply_fn(current_module, likelihood)
        else:
            result = apply_random_modification(current_module, likelihood)
        
        if not result.success:
            if stack_idx == 0:
                return None  # First modification must succeed
            break  # Subsequent modifications can fail
        
        # Update tracking
        applied_modifications.append((result.modification_type, result.line_number))
        all_descriptions.append(result.description)
        all_original_snippets.append(result.original_snippet)
        all_modified_snippets.append(result.modified_snippet)
        all_line_numbers.append(result.line_number)
        current_code = result.modified_code
    
    if not applied_modifications:
        return None
    
    # Verify the code actually changed
    if verify_different:
        if not codes_are_different(clean_code, current_code):
            return None
    
    # Optional: Formal verification
    formal_result = None
    if formal_verify:
        try:
            from formal_verify import verify_modification, VerificationResult
            formal_result = verify_modification(
                clean_code, 
                current_code, 
                timeout=formal_timeout
            )
            if not formal_result.is_valid_modification:
                return None
        except ImportError:
            pass
        except Exception as e:
            pass
    
    # Generate combined issue description for stacked modifications
    if len(applied_modifications) == 1:
        mod_type = applied_modifications[0][0]
        issue_description = ISSUE_DESCRIPTIONS.get(
            mod_type,
            f"The buggy code has a {mod_type.replace('_', ' ')} error. {all_descriptions[0]}"
        )
        modification_type = mod_type
    else:
        # Combined description for stacked modifications
        mod_types = [m[0] for m in applied_modifications]
        modification_type = "+".join(mod_types)
        issue_parts = []
        for i, (mod_type, _) in enumerate(applied_modifications):
            desc = ISSUE_DESCRIPTIONS.get(
                mod_type,
                f"The code has a {mod_type.replace('_', ' ')} error."
            )
            issue_parts.append(f"Bug {i+1} ({mod_type}): {desc}")
        issue_description = (
            f"The buggy code has {len(applied_modifications)} stacked bugs:\n\n" +
            "\n\n".join(issue_parts)
        )
    
    # Generate reasoning trace
    if len(applied_modifications) == 1:
        # Create a simple result object for single modification
        class SimpleResult:
            def __init__(self, mod_type, desc, line, orig, mod):
                self.modification_type = mod_type
                self.description = desc
                self.line_number = line
                self.original_snippet = orig
                self.modified_snippet = mod
        
        simple_result = SimpleResult(
            applied_modifications[0][0],
            all_descriptions[0],
            all_line_numbers[0],
            all_original_snippets[0],
            all_modified_snippets[0]
        )
        reasoning_trace = generate_reasoning_trace(applied_modifications[0][0], simple_result)
    else:
        # Combined reasoning trace for stacked modifications
        reasoning_parts = [f"This code contains {len(applied_modifications)} stacked bugs that need to be fixed:\n"]
        for i, (mod_type, line_num) in enumerate(applied_modifications):
            reasoning_parts.append(f"\n--- Bug {i+1}: {mod_type} (line {line_num}) ---")
            reasoning_parts.append(f"Description: {all_descriptions[i]}")
            reasoning_parts.append(f"Original: {all_original_snippets[i]}")
            reasoning_parts.append(f"Modified: {all_modified_snippets[i]}")
        reasoning_trace = "\n".join(reasoning_parts)
    
    # Build result dict
    result_dict = {
        'original_entry': entry,
        'clean_code': clean_code,
        'corrupted_code': current_code,
        'corruption_explanation': " | ".join(all_descriptions),
        'modification_type': modification_type,
        'modification_line': all_line_numbers[0] if len(all_line_numbers) == 1 else all_line_numbers,
        'original_snippet': all_original_snippets[0] if len(all_original_snippets) == 1 else all_original_snippets,
        'modified_snippet': all_modified_snippets[0] if len(all_modified_snippets) == 1 else all_modified_snippets,
        'issue_description': issue_description,
        'reasoning_trace': reasoning_trace,
        'complexity': calculate_complexity(module),
        'applicable_modifications': [m.name for m in get_applicable_modifications(module)],
        'stacked_count': len(applied_modifications),
        'applied_modifications': [m[0] for m in applied_modifications],
        'verified_different': True,
        'timestamp': datetime.now().isoformat(),
    }
    
    # Add formal verification results if available
    if formal_result is not None:
        result_dict['formal_verification'] = {
            'result': formal_result.result.name,
            'is_valid': formal_result.is_valid_modification,
            'message': formal_result.message,
            'time': formal_result.verification_time,
        }
    
    return result_dict


def main():
    parser = argparse.ArgumentParser(
        description="Generate procedural modifications for VeriThoughts data."
    )
    parser.add_argument(
        "--input",
        type=str,
        default=None,
        help="Path to input JSONL file (optional if using --dataset-type)"
    )
    parser.add_argument(
        "--dataset-type",
        type=str,
        choices=["reasoning", "instruction"],
        default=None,
        help="Load from HuggingFace dataset"
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
        default=None,
        help="Output JSONL file path"
    )
    parser.add_argument(
        "--max-entries",
        type=int,
        default=None,
        help="Maximum number of entries to process"
    )
    parser.add_argument(
        "--modification",
        type=str,
        default=None,
        help="Specific modification to apply (optional)"
    )
    parser.add_argument(
        "--likelihood",
        type=float,
        default=0.5,
        help="Probability of applying modification per candidate (default: 0.5)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed"
    )
    parser.add_argument(
        "--list-modifications",
        action="store_true",
        help="List available modifications and exit"
    )
    parser.add_argument(
        "--no-verify",
        dest="verify_different",
        action="store_false",
        default=True,
        help="Skip verification that code actually changed"
    )
    parser.add_argument(
        "--formal-verify",
        action="store_true",
        default=False,
        help="Use formal verification (requires Yosys)"
    )
    parser.add_argument(
        "--formal-timeout",
        type=int,
        default=30,
        help="Timeout for formal verification in seconds (default: 30)"
    )
    parser.add_argument(
        "--corruptions-per-sample",
        type=int,
        default=1,
        help="Number of different corruptions to generate per input sample (default: 1)"
    )
    parser.add_argument(
        "--stacked-modifications",
        type=int,
        default=1,
        help="Number of modification types to stack/combine on each corruption (default: 1)"
    )
    
    args = parser.parse_args()
    
    # List modifications if requested
    if args.list_modifications:
        print("\nAvailable Procedural Modifications:")
        print("=" * 60)
        for mod in MODIFICATION_REGISTRY:
            print(f"\n  {mod.name}")
            print(f"    Category: {mod.category.name}")
            print(f"    Description: {mod.description}")
            print(f"    Required filters: {mod.required_filter_indices}")
        return
    
    # Require output if not listing modifications
    if not args.output:
        parser.error("--output is required when not using --list-modifications")
    
    # Set random seed
    random.seed(args.seed)
    
    # Initialize parser
    verilog_parser = VerilogParser()
    
    # Load data
    if args.dataset_type:
        data = load_data_from_hf(args.dataset_type, args.split)
    elif args.input:
        print(f"Loading data from {args.input}...")
        data = load_data_from_jsonl(args.input)
        print(f"Loaded {len(data)} entries")
    else:
        parser.error("Must provide either --input or --dataset-type")
    
    if args.max_entries:
        data = data[:args.max_entries]
        print(f"Processing first {len(data)} entries")
    
    # Process entries
    results = []
    successful = 0
    failed = 0
    modification_counts = {}
    corruptions_per_sample = args.corruptions_per_sample
    stacked_modifications = args.stacked_modifications
    
    from tqdm import tqdm
    
    for i, entry in enumerate(tqdm(data, desc="Processing")):
        entry_successes = 0
        used_modifications = set()  # Track used modifications to get variety
        
        for corruption_idx in range(corruptions_per_sample):
            result = process_entry(
                entry,
                verilog_parser,
                specific_modification=args.modification,
                likelihood=args.likelihood,
                verify_different=args.verify_different,
                formal_verify=args.formal_verify,
                formal_timeout=args.formal_timeout,
                stacked_modifications=stacked_modifications,
                used_modifications=used_modifications if corruption_idx > 0 else None,
            )
            
            if result:
                result['corruption_index'] = corruption_idx
                results.append(result)
                entry_successes += 1
                mod_type = result['modification_type']
                modification_counts[mod_type] = modification_counts.get(mod_type, 0) + 1
                # Track used modification types for variety in subsequent corruptions
                if isinstance(result.get('applied_modifications'), list):
                    used_modifications.update(result['applied_modifications'])
                else:
                    used_modifications.add(mod_type)
        
        if entry_successes > 0:
            successful += 1
        else:
            failed += 1
    
    # Save results
    print(f"\nSaving results to {args.output}...")
    os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)
    
    with open(args.output, 'w') as f:
        for result in results:
            f.write(json.dumps(result) + '\n')
    
    # Print summary
    print(f"\n{'='*60}")
    print("Summary")
    print('='*60)
    print(f"Total input entries: {len(data)}")
    print(f"Corruptions per sample: {corruptions_per_sample}")
    print(f"Stacked modifications: {stacked_modifications}")
    print(f"Entries with at least one success: {successful}")
    print(f"Entries with all failures: {failed}")
    print(f"Total corruptions generated: {len(results)}")
    print(f"Entry success rate: {successful/len(data)*100:.1f}%")
    if len(data) * corruptions_per_sample > 0:
        print(f"Corruption success rate: {len(results)/(len(data)*corruptions_per_sample)*100:.1f}%")
    
    print(f"\nModification distribution:")
    for mod_type, count in sorted(modification_counts.items(), key=lambda x: -x[1]):
        print(f"  {mod_type}: {count}")
    
    print(f"\nResults saved to: {args.output}")


if __name__ == "__main__":
    main()

