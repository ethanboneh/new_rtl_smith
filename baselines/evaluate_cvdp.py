#!/usr/bin/env python3
"""
Evaluate models on the CVDP benchmark.

This script wraps the CVDP benchmark runner and provides a convenient interface
for running evaluations with different model configurations.

Performance Optimizations:
- Auto-detects optimal thread count based on CPU cores (defaults to CPU count - 1)
- Supports resuming from existing results (--resume flag)
- Optimized model factory with cached tokenizer/renderer
- Better resource utilization for parallel test execution

Usage:
    # Evaluate base Qwen3-8B (auto-detects threads)
    python evaluate_cvdp.py --model base
    
    # Evaluate with custom thread count
    python evaluate_cvdp.py --model base --threads 8
    
    # Evaluate fine-tuned models
    python evaluate_cvdp.py --model reasoning --checkpoint-path tinker://...
    python evaluate_cvdp.py --model instruction --checkpoint-path tinker://...
    
    # Run multiple samples for pass@k
    python evaluate_cvdp.py --model base --num-samples 5
    
    # Resume from existing results
    python evaluate_cvdp.py --model base --output-dir /path/to/results --resume
"""

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime
from typing import Optional
import multiprocessing

from config import (
    CVDP_BENCHMARK_PATH,
    CVDP_DATASET_PATH,
    RESULTS_DIR,
    MODEL_NAME,
    create_directories,
)


def get_optimal_thread_count(requested: Optional[int] = None) -> int:
    """
    Get optimal thread count based on available CPU cores.
    
    Args:
        requested: User-requested thread count, or None for auto
        
    Returns:
        Optimal thread count (defaults to CPU count - 1, min 1, max 16)
    """
    if requested is not None:
        return max(1, min(requested, 32))  # Cap at 32 for safety
    
    # Auto-detect: use all cores minus 1 (leave one for system)
    cpu_count = multiprocessing.cpu_count()
    optimal = max(1, cpu_count - 1)
    # Cap at 16 for reasonable memory usage
    return min(optimal, 16)


def run_cvdp_evaluation(
    model_type: str = "base",
    checkpoint_path: Optional[str] = None,
    num_samples: int = 1,
    output_dir: Optional[str] = None,
    threads: Optional[int] = None,
    timeout: int = 3600,
    resume: bool = False,
) -> str:
    """
    Run CVDP evaluation.
    
    Args:
        model_type: One of "base", "reasoning", "instruction"
        checkpoint_path: Path to fine-tuned model checkpoint (tinker://...)
        num_samples: Number of samples for pass@k evaluation
        output_dir: Directory for results
        threads: Number of parallel threads (None for auto-detect)
        timeout: Timeout in seconds
        resume: If True, resume from existing raw_result.json if available
        
    Returns:
        Path to the results directory
    """
    # Auto-detect optimal thread count if not specified
    if threads is None:
        threads = get_optimal_thread_count()
        print(f"Auto-detected optimal thread count: {threads}")
    
    # Set up output directory
    if output_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = os.path.join(RESULTS_DIR, f"cvdp_{model_type}_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)
    
    # Check for existing results if resuming
    raw_result_path = os.path.join(output_dir, "raw_result.json")
    if resume and os.path.exists(raw_result_path):
        print(f"Found existing raw_result.json at {raw_result_path}")
        print("CVDP benchmark will resume from existing results")
    
    # Set environment variables for model path
    env = os.environ.copy()
    if checkpoint_path:
        if model_type == "reasoning":
            env["TINKER_REASONING_MODEL_PATH"] = checkpoint_path
        elif model_type == "instruction":
            env["TINKER_INSTRUCTION_MODEL_PATH"] = checkpoint_path
    
    # Build the model name for the factory
    model_name_map = {
        "base": "tinker-qwen3-8b-base",
        "reasoning": "tinker-qwen3-8b-reasoning",
        "instruction": "tinker-qwen3-8b-instruction",
    }
    model_name = model_name_map.get(model_type, "tinker-qwen3-8b")
    
    # Path to our custom model factory
    factory_path = os.path.join(os.path.dirname(__file__), "model_factory_qwen.py")
    
    # Build command
    if num_samples > 1:
        # Use run_samples.py for pass@k evaluation
        script = os.path.join(CVDP_BENCHMARK_PATH, "run_samples.py")
        cmd = [
            sys.executable, script,
            "-f", CVDP_DATASET_PATH,
            "-l",
            "-m", model_name,
            "-c", factory_path,
            "-n", str(num_samples),
            "-k", "1",
            "-p", output_dir,
            "-t", str(threads),
            "-o",  # Enable host mode (no Docker)
        ]
    else:
        # Use run_benchmark.py for single evaluation
        script = os.path.join(CVDP_BENCHMARK_PATH, "run_benchmark.py")
        cmd = [
            sys.executable, script,
            "-f", CVDP_DATASET_PATH,
            "-l",
            "-m", model_name,
            "-c", factory_path,
            "-p", output_dir,
            "-t", str(threads),
            "-o",  # Enable host mode (no Docker)
        ]
    
    print(f"Running CVDP evaluation:")
    print(f"  Model type: {model_type}")
    print(f"  Checkpoint: {checkpoint_path or 'base model'}")
    print(f"  Samples: {num_samples}")
    print(f"  Output: {output_dir}")
    print(f"  Command: {' '.join(cmd)}")
    print()
    
    # Run the evaluation
    try:
        result = subprocess.run(
            cmd,
            cwd=CVDP_BENCHMARK_PATH,
            env=env,
            timeout=timeout,
            check=True,
        )
        print(f"\n✓ Evaluation completed successfully")
    except subprocess.TimeoutExpired:
        print(f"\n✗ Evaluation timed out after {timeout} seconds")
    except subprocess.CalledProcessError as e:
        print(f"\n✗ Evaluation failed with return code {e.returncode}")
    
    # Save evaluation config
    config = {
        "model_type": model_type,
        "model_name": model_name,
        "checkpoint_path": checkpoint_path,
        "num_samples": num_samples,
        "dataset": CVDP_DATASET_PATH,
        "timestamp": datetime.now().isoformat(),
    }
    config_path = os.path.join(output_dir, "eval_config.json")
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    
    return output_dir


def parse_results(results_dir: str) -> dict:
    """
    Parse evaluation results from the results directory.
    
    Args:
        results_dir: Path to results directory
        
    Returns:
        Dictionary with parsed results
    """
    results = {}
    
    # Try to load report.json
    report_path = os.path.join(results_dir, "report.json")
    if os.path.exists(report_path):
        with open(report_path, 'r') as f:
            results["report"] = json.load(f)
    
    # Try to load composite_report.json (for multi-sample runs)
    composite_path = os.path.join(results_dir, "composite_report.json")
    if os.path.exists(composite_path):
        with open(composite_path, 'r') as f:
            results["composite_report"] = json.load(f)
    
    return results


def print_summary(results: dict):
    """Print a summary of evaluation results."""
    if "report" in results:
        report = results["report"]
        print("\n" + "="*60)
        print("CVDP Evaluation Summary")
        print("="*60)
        
        # Extract key metrics
        if "summary" in report:
            summary = report["summary"]
            print(f"Total problems: {summary.get('total', 'N/A')}")
            print(f"Passed: {summary.get('passed', 'N/A')}")
            print(f"Failed: {summary.get('failed', 'N/A')}")
            if "pass_rate" in summary:
                print(f"Pass rate: {summary['pass_rate']*100:.1f}%")
    
    if "composite_report" in results:
        composite = results["composite_report"]
        print("\nPass@k Results:")
        if "pass_at_k" in composite:
            for k, rate in composite["pass_at_k"].items():
                print(f"  Pass@{k}: {rate*100:.1f}%")


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate models on CVDP benchmark"
    )
    parser.add_argument(
        "--model",
        type=str,
        choices=["base", "reasoning", "instruction"],
        default="base",
        help="Model type to evaluate"
    )
    parser.add_argument(
        "--checkpoint-path",
        type=str,
        default=None,
        help="Path to fine-tuned model checkpoint (tinker://...)"
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=1,
        help="Number of samples for pass@k evaluation"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory for results"
    )
    parser.add_argument(
        "--threads",
        type=int,
        default=None,
        help="Number of parallel threads (default: auto-detect based on CPU cores)"
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=3600,
        help="Timeout in seconds"
    )
    parser.add_argument(
        "--parse-only",
        type=str,
        default=None,
        help="Only parse results from existing directory (skip evaluation)"
    )
    parser.add_argument(
        "--host",
        action="store_true",
        default=True,  # Default to host mode (no Docker)
        help="Run tests on host without Docker (requires iverilog, cocotb, pytest)"
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume evaluation from existing raw_result.json if available"
    )
    
    args = parser.parse_args()
    
    # Create directories
    create_directories()
    
    if args.parse_only:
        # Just parse existing results
        results = parse_results(args.parse_only)
        print_summary(results)
    else:
        # Run evaluation
        results_dir = run_cvdp_evaluation(
            model_type=args.model,
            checkpoint_path=args.checkpoint_path,
            num_samples=args.num_samples,
            output_dir=args.output_dir,
            threads=args.threads,
            timeout=args.timeout,
            resume=args.resume,
        )
        
        # Parse and display results
        results = parse_results(results_dir)
        print_summary(results)
        
        print(f"\nResults saved to: {results_dir}")


if __name__ == "__main__":
    main()

