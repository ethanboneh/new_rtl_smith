#!/usr/bin/env python3
"""
Master script to run all baseline experiments.

This script orchestrates:
1. Dataset preparation
2. Base model evaluation
3. Fine-tuning on both datasets
4. Subsampling experiments
5. Evaluation of fine-tuned models

Usage:
    # Run all experiments
    python run_experiments.py --all
    
    # Run specific experiments
    python run_experiments.py --prepare-data
    python run_experiments.py --eval-base
    python run_experiments.py --train-reasoning
    python run_experiments.py --train-instruction
    python run_experiments.py --subsample-experiments
"""

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime
from typing import List, Optional

from config import (
    DATA_DIR,
    DATASETS_DIR,
    CHECKPOINTS_DIR,
    RESULTS_DIR,
    SUBSAMPLE_FRACTIONS,
    WANDB_PROJECT,
    create_directories,
)


class ExperimentRunner:
    """Orchestrates running baseline experiments."""
    
    def __init__(
        self,
        dry_run: bool = False,
        verbose: bool = True,
    ):
        self.dry_run = dry_run
        self.verbose = verbose
        self.results = {}
        
        # Track experiment timestamps
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
    def log(self, message: str):
        """Print a log message."""
        if self.verbose:
            print(f"[{datetime.now().strftime('%H:%M:%S')}] {message}")
    
    def run_command(self, cmd: List[str], description: str) -> bool:
        """Run a command and return success status."""
        self.log(f"Running: {description}")
        if self.verbose:
            print(f"  Command: {' '.join(cmd)}")
        
        if self.dry_run:
            print("  [DRY RUN - skipping]")
            return True
        
        try:
            result = subprocess.run(
                cmd,
                cwd=os.path.dirname(__file__),
                check=True,
            )
            self.log(f"✓ Completed: {description}")
            return True
        except subprocess.CalledProcessError as e:
            self.log(f"✗ Failed: {description} (exit code {e.returncode})")
            return False
    
    def prepare_datasets(self, subsamples: Optional[List[float]] = None):
        """Prepare all datasets."""
        self.log("="*60)
        self.log("STEP 1: Preparing Datasets")
        self.log("="*60)
        
        if subsamples is None:
            subsamples = SUBSAMPLE_FRACTIONS
        
        # Prepare full datasets first
        for dataset_type in ["reasoning", "instruction"]:
            cmd = [
                sys.executable, "prepare_datasets.py",
                "--dataset", dataset_type,
                "--subsample"] + [str(s) for s in subsamples]
            
            success = self.run_command(
                cmd,
                f"Prepare {dataset_type} dataset (subsamples: {subsamples})"
            )
            self.results[f"prepare_{dataset_type}"] = success
    
    def evaluate_base_model(self, num_samples: int = 5):
        """Evaluate the base Qwen3-8B model on CVDP."""
        self.log("="*60)
        self.log("STEP 2: Evaluating Base Model")
        self.log("="*60)
        
        output_dir = os.path.join(RESULTS_DIR, f"base_model_{self.timestamp}")
        
        cmd = [
            sys.executable, "evaluate_cvdp.py",
            "--model", "base",
            "--num-samples", str(num_samples),
            "--output-dir", output_dir,
        ]
        
        success = self.run_command(cmd, "Evaluate base Qwen3-8B on CVDP")
        self.results["eval_base"] = {
            "success": success,
            "output_dir": output_dir,
        }
        
        return output_dir
    
    def train_model(
        self,
        dataset_type: str,
        subsample_fraction: float = 1.0,
    ) -> Optional[str]:
        """Train a model on the specified dataset."""
        self.log(f"Training on {dataset_type} ({subsample_fraction*100:.0f}%)")
        
        fraction_str = f"_{int(subsample_fraction*100)}pct" if subsample_fraction < 1.0 else ""
        
        cmd = [
            sys.executable, "train_tinker.py",
            "--dataset", dataset_type,
            "--subsample", str(subsample_fraction),
            "--overwrite",
        ]
        
        success = self.run_command(
            cmd,
            f"Train on {dataset_type}{fraction_str}"
        )
        
        # Store result
        key = f"train_{dataset_type}{fraction_str}"
        self.results[key] = {"success": success}
        
        if success:
            # Return expected checkpoint path
            # (actual path determined by train_tinker.py)
            return os.path.join(CHECKPOINTS_DIR, f"{dataset_type}{fraction_str}_*")
        return None
    
    def train_all_full(self):
        """Train on both full datasets."""
        self.log("="*60)
        self.log("STEP 3: Training on Full Datasets")
        self.log("="*60)
        
        for dataset_type in ["reasoning", "instruction"]:
            self.train_model(dataset_type, subsample_fraction=1.0)
    
    def run_subsampling_experiments(self, fractions: Optional[List[float]] = None):
        """Run subsampling experiments to check data saturation."""
        self.log("="*60)
        self.log("STEP 4: Subsampling Experiments")
        self.log("="*60)
        
        if fractions is None:
            fractions = SUBSAMPLE_FRACTIONS
        
        for dataset_type in ["reasoning", "instruction"]:
            for fraction in fractions:
                if fraction == 1.0:
                    continue  # Skip full dataset (already trained)
                self.train_model(dataset_type, subsample_fraction=fraction)
    
    def evaluate_finetuned_models(self, num_samples: int = 5):
        """Evaluate fine-tuned models on CVDP."""
        self.log("="*60)
        self.log("STEP 5: Evaluating Fine-tuned Models")
        self.log("="*60)
        
        # This would need the actual checkpoint paths from training
        # For now, we document the expected usage
        self.log("NOTE: Fine-tuned model evaluation requires checkpoint paths")
        self.log("Use 'evaluate_cvdp.py --model reasoning/instruction --checkpoint-path <path>'")
        
        # Example commands for documentation
        self.results["eval_finetuned_docs"] = {
            "reasoning_cmd": (
                "python evaluate_cvdp.py --model reasoning "
                "--checkpoint-path tinker://path/to/reasoning/checkpoint"
            ),
            "instruction_cmd": (
                "python evaluate_cvdp.py --model instruction "
                "--checkpoint-path tinker://path/to/instruction/checkpoint"
            ),
        }
    
    def generate_summary(self) -> str:
        """Generate a summary of all experiment results."""
        summary = []
        summary.append("="*60)
        summary.append("EXPERIMENT SUMMARY")
        summary.append("="*60)
        summary.append(f"Timestamp: {self.timestamp}")
        summary.append("")
        
        for key, value in self.results.items():
            if isinstance(value, dict):
                status = "✓" if value.get("success", False) else "✗"
                summary.append(f"{status} {key}")
                if "output_dir" in value:
                    summary.append(f"    Output: {value['output_dir']}")
            elif isinstance(value, bool):
                status = "✓" if value else "✗"
                summary.append(f"{status} {key}")
        
        summary.append("")
        summary.append("="*60)
        
        return "\n".join(summary)
    
    def save_results(self, output_path: Optional[str] = None):
        """Save experiment results to JSON."""
        if output_path is None:
            output_path = os.path.join(RESULTS_DIR, f"experiment_results_{self.timestamp}.json")
        
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        results_data = {
            "timestamp": self.timestamp,
            "results": self.results,
        }
        
        with open(output_path, 'w') as f:
            json.dump(results_data, f, indent=2, default=str)
        
        self.log(f"Results saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Run baseline experiments for spec-to-Verilog generation"
    )
    
    # Experiment selection
    parser.add_argument("--all", action="store_true", help="Run all experiments")
    parser.add_argument("--prepare-data", action="store_true", help="Prepare datasets")
    parser.add_argument("--eval-base", action="store_true", help="Evaluate base model")
    parser.add_argument("--train-reasoning", action="store_true", help="Train on reasoning dataset")
    parser.add_argument("--train-instruction", action="store_true", help="Train on instruction dataset")
    parser.add_argument("--subsample-experiments", action="store_true", help="Run subsampling experiments")
    parser.add_argument("--eval-finetuned", action="store_true", help="Evaluate fine-tuned models")
    
    # Options
    parser.add_argument("--dry-run", action="store_true", help="Print commands without executing")
    parser.add_argument("--num-samples", type=int, default=5, help="Number of samples for evaluation")
    parser.add_argument(
        "--subsamples",
        type=float,
        nargs="+",
        default=SUBSAMPLE_FRACTIONS,
        help="Subsample fractions for experiments"
    )
    parser.add_argument("--quiet", action="store_true", help="Reduce verbosity")
    
    args = parser.parse_args()
    
    # Create directories
    create_directories()
    
    # Initialize runner
    runner = ExperimentRunner(
        dry_run=args.dry_run,
        verbose=not args.quiet,
    )
    
    # Determine which experiments to run
    if args.all:
        args.prepare_data = True
        args.eval_base = True
        args.train_reasoning = True
        args.train_instruction = True
        args.subsample_experiments = True
    
    # Run selected experiments
    if args.prepare_data:
        runner.prepare_datasets(subsamples=args.subsamples)
    
    if args.eval_base:
        runner.evaluate_base_model(num_samples=args.num_samples)
    
    if args.train_reasoning:
        runner.train_model("reasoning", subsample_fraction=1.0)
    
    if args.train_instruction:
        runner.train_model("instruction", subsample_fraction=1.0)
    
    if args.subsample_experiments:
        runner.run_subsampling_experiments(fractions=args.subsamples)
    
    if args.eval_finetuned:
        runner.evaluate_finetuned_models(num_samples=args.num_samples)
    
    # Print and save summary
    print("\n" + runner.generate_summary())
    runner.save_results()


if __name__ == "__main__":
    main()

