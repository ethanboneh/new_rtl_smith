#!/usr/bin/env python3
"""
Generate procedural modifications for RTLLM dataset.

RTLLM Structure:
    /RTLLM/
        Arithmetic/
            Adder/
                adder_8bit/
                    verified_adder_8bit.v  # Golden design
                    testbench.v            # Testbench
                    design_description.txt # Description
                    makefile
        Control/
        Memory/
        Miscellaneous/

This script:
1. Loads RTLLM designs from the directory structure
2. Applies procedural modifications to create bugs
3. Optionally verifies corruptions using testbenches
4. Saves results in the same format as VeriThoughts corruptions

Usage:
    python scripts/generate_rtllm_corruptions.py \
        --rtllm-path /path/to/RTLLM \
        --output outputs/rtllm_corruptions.jsonl \
        --verify-testbench \
        --max-entries 10
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

from verilog_ast import VerilogParser, calculate_complexity
from filters import apply_filters, FILTER_REGISTRY
from modifications import (
    MODIFICATION_REGISTRY,
    MODIFICATION_BY_NAME,
    get_applicable_modifications,
    apply_random_modification,
    ModificationResult,
)
from testbench_runner import (
    run_testbench,
    verify_corruption,
    find_available_simulator,
    TestbenchResult,
)

# Import issue descriptions from main script
from generate_modifications import ISSUE_DESCRIPTIONS, generate_reasoning_trace


# ============================================================================
# RTLLM Data Loading
# ============================================================================

RTLLM_CATEGORIES = ["Arithmetic", "Control", "Memory", "Miscellaneous"]


def find_rtllm_designs(rtllm_path: str) -> List[Dict]:
    """
    Find all RTLLM designs in the directory structure.
    
    Returns list of dicts with:
        - category: e.g., "Arithmetic"
        - subcategory: e.g., "Adder"
        - name: e.g., "adder_8bit"
        - design_path: Path to verified_*.v
        - testbench_path: Path to testbench.v
        - description_path: Path to design_description.txt
        - module_name: Inferred module name
    """
    rtllm_path = Path(rtllm_path)
    designs = []
    
    for category in RTLLM_CATEGORIES:
        category_path = rtllm_path / category
        if not category_path.exists():
            continue
        
        # Iterate through subcategories (e.g., Adder, Multiplier)
        for subcategory_path in category_path.iterdir():
            if not subcategory_path.is_dir():
                continue
            
            subcategory = subcategory_path.name
            
            # Iterate through individual designs
            for design_path in subcategory_path.iterdir():
                if not design_path.is_dir():
                    continue
                
                design_name = design_path.name
                
                # Find the verified design file
                verified_files = list(design_path.glob("verified_*.v"))
                if not verified_files:
                    continue
                
                verified_file = verified_files[0]
                testbench_file = design_path / "testbench.v"
                description_file = design_path / "design_description.txt"
                
                # Infer module name from verified file
                # verified_adder_8bit.v -> adder_8bit
                module_name = verified_file.stem.replace("verified_", "")
                
                designs.append({
                    "category": category,
                    "subcategory": subcategory,
                    "name": design_name,
                    "design_path": str(verified_file),
                    "testbench_path": str(testbench_file) if testbench_file.exists() else None,
                    "description_path": str(description_file) if description_file.exists() else None,
                    "module_name": module_name,
                })
    
    return designs


def load_rtllm_design(design_info: Dict) -> Dict:
    """
    Load a single RTLLM design with its code and testbench.
    
    Returns dict with:
        - All fields from design_info
        - code: The Verilog code
        - testbench: The testbench code (or None)
        - description: The design description (or None)
    """
    result = design_info.copy()
    
    # Load design code
    with open(design_info["design_path"], 'r') as f:
        code = f.read()
    
    # Rename module from verified_* to * for testbench compatibility
    # e.g., "module verified_adder_8bit" -> "module adder_8bit"
    verified_module_name = f"verified_{design_info['module_name']}"
    code = re.sub(
        rf'\bmodule\s+{verified_module_name}\b',
        f"module {design_info['module_name']}",
        code
    )
    result["code"] = code
    
    # Load testbench if available
    if design_info["testbench_path"]:
        with open(design_info["testbench_path"], 'r') as f:
            result["testbench"] = f.read()
    else:
        result["testbench"] = None
    
    # Load description if available
    if design_info["description_path"]:
        with open(design_info["description_path"], 'r') as f:
            result["description"] = f.read()
    else:
        result["description"] = None
    
    return result


def normalize_code(code: str) -> str:
    """Normalize code for comparison."""
    code = re.sub(r'//.*$', '', code, flags=re.MULTILINE)
    code = re.sub(r'/\*.*?\*/', '', code, flags=re.DOTALL)
    code = ' '.join(code.split())
    return code.strip()


def codes_are_different(clean_code: str, corrupt_code: str) -> bool:
    """Check if two pieces of code are different."""
    return normalize_code(clean_code) != normalize_code(corrupt_code)


# ============================================================================
# Corruption Processing
# ============================================================================

def process_rtllm_entry(
    design: Dict,
    parser: VerilogParser,
    specific_modification: Optional[str] = None,
    likelihood: float = 0.5,
    verify_testbench: bool = False,
    require_testbench_fail: bool = True,
    stacked_modifications: int = 1,
    used_modifications: Optional[set] = None,
    timeout: int = 30,
) -> Optional[Dict]:
    """
    Process a single RTLLM design to generate procedural modification.
    
    Args:
        design: RTLLM design dict with code, testbench, etc.
        parser: VerilogParser instance
        specific_modification: Specific modification to apply
        likelihood: Probability of applying modification
        verify_testbench: Whether to verify with testbench
        require_testbench_fail: Only accept if testbench fails
        stacked_modifications: Number of modifications to stack
        used_modifications: Set of already used modifications
        timeout: Testbench timeout in seconds
    
    Returns:
        Dictionary with modification data or None if failed
    """
    clean_code = design["code"]
    
    # Parse the code
    try:
        module = parser.parse(clean_code)
    except Exception as e:
        return None
    
    # Get applicable modifications
    applicable = get_applicable_modifications(module)
    if not applicable:
        return None
    
    # Filter out already used modifications
    if used_modifications:
        applicable = [m for m in applicable if m.name not in used_modifications]
        if not applicable:
            return None
    
    # Track modifications applied
    applied_modifications = []
    all_descriptions = []
    all_original_snippets = []
    all_modified_snippets = []
    all_line_numbers = []
    
    current_code = clean_code
    current_module = module
    
    # Apply stacked modifications
    for stack_idx in range(stacked_modifications):
        if stack_idx > 0:
            try:
                current_module = parser.parse(current_code)
                applicable = get_applicable_modifications(current_module)
                applicable = [m for m in applicable if m.name not in [am[0] for am in applied_modifications]]
                if not applicable:
                    break
            except Exception:
                break
        
        # Apply modification
        if specific_modification and stack_idx == 0:
            if specific_modification not in MODIFICATION_BY_NAME:
                return None
            result = MODIFICATION_BY_NAME[specific_modification].apply_fn(current_module, likelihood)
        else:
            result = apply_random_modification(current_module, likelihood)
        
        if not result.success:
            if stack_idx == 0:
                return None
            break
        
        applied_modifications.append((result.modification_type, result.line_number))
        all_descriptions.append(result.description)
        all_original_snippets.append(result.original_snippet)
        all_modified_snippets.append(result.modified_snippet)
        all_line_numbers.append(result.line_number)
        current_code = result.modified_code
    
    if not applied_modifications:
        return None
    
    # Verify code actually changed
    if not codes_are_different(clean_code, current_code):
        return None
    
    # Testbench verification
    testbench_result = None
    clean_tb_result = None
    corruption_verified = None
    
    if verify_testbench and design.get("testbench"):
        clean_tb_result, corrupt_tb_result, is_valid = verify_corruption(
            clean_code,
            current_code,
            design["testbench"],
            design["module_name"],
            timeout=timeout
        )
        
        testbench_result = {
            "clean_passed": clean_tb_result.passed,
            "corrupt_passed": corrupt_tb_result.passed,
            "clean_errors": clean_tb_result.error_count,
            "corrupt_errors": corrupt_tb_result.error_count,
            "simulator": corrupt_tb_result.simulator,
            "is_valid_corruption": is_valid,
            "clean_compile_success": clean_tb_result.compile_success,
            "corrupt_compile_success": corrupt_tb_result.compile_success,
        }
        
        corruption_verified = is_valid
        
        # If we require testbench to fail and it doesn't, reject
        if require_testbench_fail and not is_valid:
            return None
    
    # Generate issue description
    if len(applied_modifications) == 1:
        mod_type = applied_modifications[0][0]
        issue_description = ISSUE_DESCRIPTIONS.get(
            mod_type,
            f"The buggy code has a {mod_type.replace('_', ' ')} error. {all_descriptions[0]}"
        )
        modification_type = mod_type
    else:
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
        reasoning_parts = [f"This code contains {len(applied_modifications)} stacked bugs that need to be fixed:\n"]
        for i, (mod_type, line_num) in enumerate(applied_modifications):
            reasoning_parts.append(f"\n--- Bug {i+1}: {mod_type} (line {line_num}) ---")
            reasoning_parts.append(f"Description: {all_descriptions[i]}")
            reasoning_parts.append(f"Original: {all_original_snippets[i]}")
            reasoning_parts.append(f"Modified: {all_modified_snippets[i]}")
        reasoning_trace = "\n".join(reasoning_parts)
    
    # Build result
    result_dict = {
        "source": "rtllm",
        "rtllm_info": {
            "category": design["category"],
            "subcategory": design["subcategory"],
            "design_name": design["name"],
            "module_name": design["module_name"],
            "description": design.get("description"),
        },
        "clean_code": clean_code,
        "corrupted_code": current_code,
        "corruption_explanation": " | ".join(all_descriptions),
        "modification_type": modification_type,
        "modification_line": all_line_numbers[0] if len(all_line_numbers) == 1 else all_line_numbers,
        "original_snippet": all_original_snippets[0] if len(all_original_snippets) == 1 else all_original_snippets,
        "modified_snippet": all_modified_snippets[0] if len(all_modified_snippets) == 1 else all_modified_snippets,
        "issue_description": issue_description,
        "reasoning_trace": reasoning_trace,
        "complexity": calculate_complexity(module),
        "applicable_modifications": [m.name for m in get_applicable_modifications(module)],
        "stacked_count": len(applied_modifications),
        "applied_modifications": [m[0] for m in applied_modifications],
        "verified_different": True,
        "testbench_verification": testbench_result,
        "corruption_verified_by_testbench": corruption_verified,
        "timestamp": datetime.now().isoformat(),
    }
    
    return result_dict


def main():
    parser = argparse.ArgumentParser(
        description="Generate procedural modifications for RTLLM dataset."
    )
    parser.add_argument(
        "--rtllm-path",
        type=str,
        default="/sailhome/ethanboneh/RTLLM",
        help="Path to RTLLM dataset directory"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output JSONL file path (required unless --list-designs)"
    )
    parser.add_argument(
        "--max-entries",
        type=int,
        default=None,
        help="Maximum number of designs to process"
    )
    parser.add_argument(
        "--modification",
        type=str,
        default=None,
        help="Specific modification to apply"
    )
    parser.add_argument(
        "--likelihood",
        type=float,
        default=0.5,
        help="Probability of applying modification (default: 0.5)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed"
    )
    parser.add_argument(
        "--corruptions-per-sample",
        type=int,
        default=1,
        help="Number of corruptions per design (default: 1)"
    )
    parser.add_argument(
        "--stacked-modifications",
        type=int,
        default=1,
        help="Number of modifications to stack (default: 1)"
    )
    parser.add_argument(
        "--verify-testbench",
        action="store_true",
        default=False,
        help="Verify corruptions with testbenches"
    )
    parser.add_argument(
        "--require-testbench-fail",
        action="store_true",
        default=False,
        help="Only keep corruptions where testbench fails (requires --verify-testbench)"
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=30,
        help="Testbench simulation timeout in seconds (default: 30)"
    )
    parser.add_argument(
        "--category",
        type=str,
        choices=RTLLM_CATEGORIES,
        default=None,
        help="Only process designs from specific category"
    )
    parser.add_argument(
        "--list-designs",
        action="store_true",
        help="List available designs and exit"
    )
    
    args = parser.parse_args()
    
    # Find designs
    print(f"Scanning RTLLM dataset at: {args.rtllm_path}")
    designs = find_rtllm_designs(args.rtllm_path)
    print(f"Found {len(designs)} designs")
    
    # Filter by category if specified
    if args.category:
        designs = [d for d in designs if d["category"] == args.category]
        print(f"Filtered to {len(designs)} designs in category: {args.category}")
    
    # List designs if requested
    if args.list_designs:
        print("\nAvailable RTLLM Designs:")
        print("=" * 60)
        for d in designs:
            has_tb = "✓" if d["testbench_path"] else "✗"
            print(f"  [{has_tb}] {d['category']}/{d['subcategory']}/{d['name']}")
        print(f"\n✓ = has testbench, ✗ = no testbench")
        print(f"Total: {len(designs)} designs")
        return
    
    # Require output if not listing
    if not args.output:
        parser.error("--output is required when not using --list-designs")
    
    # Check simulator availability
    if args.verify_testbench:
        simulator = find_available_simulator()
        if simulator:
            print(f"Using simulator: {simulator.value}")
        else:
            print("WARNING: No Verilog simulator found. Testbench verification will be skipped.")
            print("Install iverilog, verilator, or vcs for testbench verification.")
            args.verify_testbench = False
    
    # Set random seed
    random.seed(args.seed)
    
    # Initialize parser
    verilog_parser = VerilogParser()
    
    # Limit entries if specified
    if args.max_entries:
        designs = designs[:args.max_entries]
        print(f"Processing first {len(designs)} designs")
    
    # Process designs
    results = []
    successful = 0
    failed = 0
    modification_counts = {}
    verified_count = 0
    
    from tqdm import tqdm
    
    for design_info in tqdm(designs, desc="Processing"):
        # Load design
        try:
            design = load_rtllm_design(design_info)
        except Exception as e:
            print(f"\n  Error loading {design_info['name']}: {e}")
            failed += 1
            continue
        
        entry_successes = 0
        used_modifications = set()
        
        for corruption_idx in range(args.corruptions_per_sample):
            result = process_rtllm_entry(
                design,
                verilog_parser,
                specific_modification=args.modification,
                likelihood=args.likelihood,
                verify_testbench=args.verify_testbench,
                require_testbench_fail=args.require_testbench_fail,
                stacked_modifications=args.stacked_modifications,
                used_modifications=used_modifications if corruption_idx > 0 else None,
                timeout=args.timeout,
            )
            
            if result:
                result["corruption_index"] = corruption_idx
                results.append(result)
                entry_successes += 1
                
                mod_type = result["modification_type"]
                modification_counts[mod_type] = modification_counts.get(mod_type, 0) + 1
                
                # Track used modifications
                if isinstance(result.get("applied_modifications"), list):
                    used_modifications.update(result["applied_modifications"])
                else:
                    used_modifications.add(mod_type)
                
                # Count verified corruptions
                if result.get("corruption_verified_by_testbench"):
                    verified_count += 1
        
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
    print(f"Total RTLLM designs: {len(designs)}")
    print(f"Corruptions per sample: {args.corruptions_per_sample}")
    print(f"Stacked modifications: {args.stacked_modifications}")
    print(f"Designs with at least one success: {successful}")
    print(f"Designs with all failures: {failed}")
    print(f"Total corruptions generated: {len(results)}")
    if args.verify_testbench:
        print(f"Testbench-verified corruptions: {verified_count}")
    print(f"Entry success rate: {successful/len(designs)*100:.1f}%")
    
    print(f"\nModification distribution:")
    for mod_type, count in sorted(modification_counts.items(), key=lambda x: -x[1]):
        print(f"  {mod_type}: {count}")
    
    print(f"\nResults saved to: {args.output}")


if __name__ == "__main__":
    main()

