#!/usr/bin/env python3
"""
Generate LLM-based corruptions for RTLLM dataset.

RTLLM Structure:
    /RTLLM/
        Arithmetic/
            Adder/
                adder_8bit/
                    verified_adder_8bit.v  # Golden design
                    testbench.v            # Testbench
                    design_description.txt # Description
        Control/
        Memory/
        Miscellaneous/

This script:
1. Loads RTLLM designs from the directory structure
2. Uses LLM to corrupt the code
3. Optionally verifies corruptions using testbenches
4. Saves results in the same format as VeriThoughts corruptions

Usage:
    python scripts/generate_rtllm_corruptions.py \
        --rtllm-path /path/to/RTLLM \
        --output outputs/rtllm_corruptions.jsonl \
        --verify-testbench \
        --model gpt-4o-mini
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

from llm_client import LLMClient
from linter import VerilatorLinter
from testbench_runner import (
    run_testbench,
    verify_corruption,
    find_available_simulator,
    TestbenchResult,
)


# ============================================================================
# RTLLM Data Loading
# ============================================================================

RTLLM_CATEGORIES = ["Arithmetic", "Control", "Memory", "Miscellaneous"]


def find_rtllm_designs(rtllm_path: str) -> List[Dict]:
    """
    Find all RTLLM designs in the directory structure.
    """
    rtllm_path = Path(rtllm_path)
    designs = []
    
    for category in RTLLM_CATEGORIES:
        category_path = rtllm_path / category
        if not category_path.exists():
            continue
        
        for subcategory_path in category_path.iterdir():
            if not subcategory_path.is_dir():
                continue
            
            subcategory = subcategory_path.name
            
            for design_path in subcategory_path.iterdir():
                if not design_path.is_dir():
                    continue
                
                design_name = design_path.name
                
                verified_files = list(design_path.glob("verified_*.v"))
                if not verified_files:
                    continue
                
                verified_file = verified_files[0]
                testbench_file = design_path / "testbench.v"
                description_file = design_path / "design_description.txt"
                
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
    """Load a single RTLLM design with its code and testbench."""
    result = design_info.copy()
    
    with open(design_info["design_path"], 'r') as f:
        code = f.read()
    
    # Rename module from verified_* to * for testbench compatibility
    verified_module_name = f"verified_{design_info['module_name']}"
    code = re.sub(
        rf'\bmodule\s+{verified_module_name}\b',
        f"module {design_info['module_name']}",
        code
    )
    result["code"] = code
    
    if design_info["testbench_path"]:
        with open(design_info["testbench_path"], 'r') as f:
            result["testbench"] = f.read()
    else:
        result["testbench"] = None
    
    if design_info["description_path"]:
        with open(design_info["description_path"], 'r') as f:
            result["description"] = f.read()
    else:
        result["description"] = None
    
    return result


# ============================================================================
# Corruption Processing
# ============================================================================

def process_rtllm_entry_from_spec(
    design: Dict,
    llm_client: LLMClient,
    linter: Optional[VerilatorLinter],
    max_retries: int = 3,
    skip_linter: bool = False,
    verify_testbench: bool = False,
    require_testbench_fail: bool = True,
    timeout: int = 30,
) -> Optional[Dict]:
    """
    Generate buggy code directly from spec (no gold code provided to LLM).
    
    This creates more diverse/nuanced bugs since the LLM writes code
    from scratch rather than modifying existing code.
    """
    clean_code = design["code"]  # We still have this for verification
    spec_description = design.get("description")
    
    if not spec_description:
        print(f"    Failed: No spec description available for from-spec mode")
        return None
    
    for attempt in range(max_retries):
        try:
            print(f"  Attempt {attempt + 1}/{max_retries}: Generating buggy code from spec...")
            
            # Generate feedback for retries
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
                # For from-spec mode, we check if buggy code has any lint issues
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
            
            # Testbench verification
            testbench_result = None
            corruption_verified = None
            
            if verify_testbench and design.get("testbench"):
                print(f"    Verifying with testbench...")
                clean_tb_result, buggy_tb_result, is_valid = verify_corruption(
                    clean_code,
                    buggy_code,
                    design["testbench"],
                    design["module_name"],
                    timeout=timeout
                )
                
                testbench_result = {
                    "clean_passed": clean_tb_result.passed,
                    "buggy_passed": buggy_tb_result.passed,
                    "clean_errors": clean_tb_result.error_count,
                    "buggy_errors": buggy_tb_result.error_count,
                    "simulator": buggy_tb_result.simulator,
                    "is_valid_corruption": is_valid,
                    "clean_compile_success": clean_tb_result.compile_success,
                    "buggy_compile_success": buggy_tb_result.compile_success,
                }
                
                corruption_verified = is_valid
                
                if require_testbench_fail and not is_valid:
                    print(f"    Failed: Testbench did not fail (clean={clean_tb_result.passed}, buggy={buggy_tb_result.passed})")
                    continue
            
            # Generate issue description (comparing buggy to gold)
            print(f"    Generating issue description...")
            issue_description = llm_client.generate_issue_description(
                clean_code,  # Gold code
                buggy_code   # Buggy code
            )
            
            # Generate reasoning trace (from buggy to gold)
            print(f"    Generating reasoning trace...")
            reasoning_trace = llm_client.generate_reasoning_trace(
                clean_code,  # Gold code (the fix target)
                buggy_code,  # Buggy code (the starting point)
                issue_description
            )
            
            # Success!
            return {
                'source': 'rtllm',
                'generation_mode': 'from_spec',
                'rtllm_info': {
                    'category': design['category'],
                    'subcategory': design['subcategory'],
                    'design_name': design['name'],
                    'module_name': design['module_name'],
                    'description': spec_description,
                },
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
                'testbench_verification': testbench_result,
                'corruption_verified_by_testbench': corruption_verified,
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            print(f"    Error: {e}")
            import traceback
            traceback.print_exc()
            if attempt == max_retries - 1:
                return None
    
    return None


def process_rtllm_entry(
    design: Dict,
    llm_client: LLMClient,
    linter: Optional[VerilatorLinter],
    max_retries: int = 3,
    require_lint_violations: bool = True,
    skip_linter: bool = False,
    verify_testbench: bool = False,
    require_testbench_fail: bool = True,
    timeout: int = 30,
) -> Optional[Dict]:
    """
    Process a single RTLLM design to generate LLM corruption.
    """
    clean_code = design["code"]
    
    for attempt in range(max_retries):
        try:
            print(f"  Attempt {attempt + 1}/{max_retries}: Generating corruption...")
            
            # Generate corruption with feedback
            feedback = ""
            if attempt > 0:
                if linter is not None and not skip_linter:
                    clean_lint = linter.lint_code(clean_code)
                    feedback = (
                        f"\n\nPREVIOUS ATTEMPT FAILED. "
                        f"Clean code violations: {list(clean_lint['violated_rules'])}. "
                        f"Try: (1) Remove else clauses from always_comb, "
                        f"(2) Use blocking = in always_ff, "
                        f"(3) Add multiple drivers, "
                        f"(4) Create width mismatches.\n"
                    )
                else:
                    feedback = "\n\nPREVIOUS ATTEMPT FAILED. Try a different bug type.\n"
            
            corruption_result = llm_client.corrupt_code(clean_code, feedback=feedback)
            corrupted_code = corruption_result['corrupted_code']
            
            if not corrupted_code:
                print(f"    Failed: No corrupted code generated")
                continue
            
            # Linter verification
            lint_result = None
            if not skip_linter and linter is not None:
                print(f"    Verifying with linter...")
                lint_result = linter.verify_corruption(
                    clean_code,
                    corrupted_code,
                    require_new_violations=require_lint_violations
                )
                
                if not lint_result['is_valid']:
                    print(f"    Failed: No new lint violations")
                    continue
            elif skip_linter:
                print(f"    Skipping linter verification...")
                lint_result = {
                    'new_violated_rules': set(),
                    'corrupt_violated_rules': set(),
                    'clean_has_errors': False,
                    'corrupt_has_errors': False,
                }
            
            # Testbench verification
            testbench_result = None
            corruption_verified = None
            
            if verify_testbench and design.get("testbench"):
                print(f"    Verifying with testbench...")
                clean_tb_result, corrupt_tb_result, is_valid = verify_corruption(
                    clean_code,
                    corrupted_code,
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
                
                if require_testbench_fail and not is_valid:
                    print(f"    Failed: Testbench did not fail (clean={clean_tb_result.passed}, corrupt={corrupt_tb_result.passed})")
                    continue
            
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
                'source': 'rtllm',
                'rtllm_info': {
                    'category': design['category'],
                    'subcategory': design['subcategory'],
                    'design_name': design['name'],
                    'module_name': design['module_name'],
                    'description': design.get('description'),
                },
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
                'testbench_verification': testbench_result,
                'corruption_verified_by_testbench': corruption_verified,
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            print(f"    Error: {e}")
            if attempt == max_retries - 1:
                return None
    
    return None


def main():
    parser = argparse.ArgumentParser(
        description="Generate LLM-based corruptions for RTLLM dataset."
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
        "--model",
        type=str,
        default="gpt-4o-mini",
        help="LLM model name (default: gpt-4o-mini)"
    )
    parser.add_argument(
        "--api-key",
        type=str,
        default=None,
        help="OpenAI API key"
    )
    parser.add_argument(
        "--base-url",
        type=str,
        default=None,
        help="API base URL"
    )
    parser.add_argument(
        "--max-entries",
        type=int,
        default=None,
        help="Maximum number of designs to process"
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=3,
        help="Maximum retries per entry"
    )
    parser.add_argument(
        "--corruptions-per-sample",
        type=int,
        default=1,
        help="Number of corruptions per design (default: 1)"
    )
    parser.add_argument(
        "--require-lint",
        action="store_true",
        default=True,
        help="Require new lint violations"
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
        help="Skip linter verification"
    )
    parser.add_argument(
        "--verilator-bin",
        type=str,
        default=None,
        help="Path to verilator binary"
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
        help="Only keep corruptions where testbench fails"
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=30,
        help="Testbench simulation timeout"
    )
    parser.add_argument(
        "--category",
        type=str,
        choices=RTLLM_CATEGORIES,
        default=None,
        help="Only process specific category"
    )
    parser.add_argument(
        "--list-designs",
        action="store_true",
        help="List available designs and exit"
    )
    parser.add_argument(
        "--from-spec",
        action="store_true",
        default=False,
        help="Generate buggy code from spec only (no gold code given to LLM). "
             "This creates more diverse/nuanced bugs since LLM writes from scratch."
    )
    
    args = parser.parse_args()
    
    # Find designs
    print(f"Scanning RTLLM dataset at: {args.rtllm_path}")
    designs = find_rtllm_designs(args.rtllm_path)
    print(f"Found {len(designs)} designs")
    
    if args.category:
        designs = [d for d in designs if d["category"] == args.category]
        print(f"Filtered to {len(designs)} designs in: {args.category}")
    
    if args.list_designs:
        print("\nAvailable RTLLM Designs:")
        print("=" * 60)
        for d in designs:
            has_tb = "✓" if d["testbench_path"] else "✗"
            print(f"  [{has_tb}] {d['category']}/{d['subcategory']}/{d['name']}")
        print(f"\nTotal: {len(designs)} designs")
        return
    
    # Require output if not listing
    if not args.output:
        parser.error("--output is required when not using --list-designs")
    
    # Initialize LLM client
    print("Initializing LLM client...")
    llm_client = LLMClient(
        model=args.model,
        api_key=args.api_key,
        base_url=args.base_url
    )
    
    # Initialize linter
    linter = None
    if not args.skip_linter:
        print("Initializing linter...")
        try:
            linter = VerilatorLinter(verilator_bin=args.verilator_bin)
        except Exception as e:
            print(f"Warning: Could not initialize linter: {e}")
            print("Continuing without linter verification...")
            args.skip_linter = True
    
    # Check simulator for testbench verification
    if args.verify_testbench:
        simulator = find_available_simulator()
        if simulator:
            print(f"Using simulator: {simulator.value}")
        else:
            print("WARNING: No simulator found. Testbench verification disabled.")
            args.verify_testbench = False
    
    # Limit entries
    if args.max_entries:
        designs = designs[:args.max_entries]
        print(f"Processing first {len(designs)} designs")
    
    # Print mode info
    if args.from_spec:
        print("\n*** FROM-SPEC MODE: Generating buggy code from specification only ***")
        print("    (LLM will not see gold code, creating more diverse bugs)\n")
    
    # Process designs
    results = []
    successful = 0
    failed = 0
    verified_count = 0
    corruptions_per_sample = args.corruptions_per_sample
    
    for i, design_info in enumerate(designs):
        print(f"\nProcessing design {i + 1}/{len(designs)}: {design_info['name']} "
              f"(generating up to {corruptions_per_sample} corruptions)...")
        
        # Load design
        try:
            design = load_rtllm_design(design_info)
        except Exception as e:
            print(f"  Error loading design: {e}")
            failed += 1
            continue
        
        # Check for spec description in from-spec mode
        if args.from_spec and not design.get("description"):
            print(f"  Skipping: No spec description available for from-spec mode")
            failed += 1
            continue
        
        entry_successes = 0
        for corruption_idx in range(corruptions_per_sample):
            if corruptions_per_sample > 1:
                print(f"  Corruption {corruption_idx + 1}/{corruptions_per_sample}:")
            
            # Choose processing function based on mode
            if args.from_spec:
                result = process_rtllm_entry_from_spec(
                    design,
                    llm_client,
                    linter,
                    max_retries=args.max_retries,
                    skip_linter=args.skip_linter,
                    verify_testbench=args.verify_testbench,
                    require_testbench_fail=args.require_testbench_fail,
                    timeout=args.timeout,
                )
            else:
                result = process_rtllm_entry(
                    design,
                    llm_client,
                    linter,
                    max_retries=args.max_retries,
                    require_lint_violations=args.require_lint,
                    skip_linter=args.skip_linter,
                    verify_testbench=args.verify_testbench,
                    require_testbench_fail=args.require_testbench_fail,
                    timeout=args.timeout,
                )
            
            if result:
                result['corruption_index'] = corruption_idx
                results.append(result)
                entry_successes += 1
                
                if result.get('corruption_verified_by_testbench'):
                    verified_count += 1
                
                if args.from_spec:
                    bug_type = result.get('bug_type', 'unknown')
                    print(f"    ✓ Success! Bug type: {bug_type}")
                elif args.skip_linter:
                    print(f"    ✓ Success! (linter skipped)")
                else:
                    print(f"    ✓ Success! Violated: {result['lint_result']['new_violated_rules']}")
            else:
                print(f"    ✗ Failed")
        
        if entry_successes > 0:
            successful += 1
        else:
            failed += 1
        
        print(f"  Entry result: {entry_successes}/{corruptions_per_sample} corruptions")
    
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
    print(f"Mode: {'from-spec' if args.from_spec else 'corrupt-gold'}")
    print(f"Total RTLLM designs: {len(designs)}")
    print(f"Corruptions per sample: {corruptions_per_sample}")
    print(f"Designs with at least one success: {successful}")
    print(f"Designs with all failures: {failed}")
    print(f"Total corruptions generated: {len(results)}")
    if args.verify_testbench:
        print(f"Testbench-verified corruptions: {verified_count}")
    print(f"Entry success rate: {successful/len(designs)*100:.1f}%")
    if len(designs) * corruptions_per_sample > 0:
        print(f"Corruption success rate: {len(results)/(len(designs)*corruptions_per_sample)*100:.1f}%")
    
    # Show bug type distribution for from-spec mode
    if args.from_spec and results:
        print(f"\nBug type distribution:")
        bug_types = {}
        for r in results:
            bt = r.get('bug_type', 'unknown')
            bug_types[bt] = bug_types.get(bt, 0) + 1
        for bt, count in sorted(bug_types.items(), key=lambda x: -x[1]):
            print(f"  {bt}: {count}")
    
    print(f"\nResults saved to: {args.output}")


if __name__ == "__main__":
    main()

