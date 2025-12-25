#!/bin/bash
# Run all baseline experiments
#
# This is the master script that runs the complete experimental pipeline:
# 1. Prepare all datasets
# 2. Evaluate base model
# 3. Train on full datasets
# 4. Run subsampling experiments
#
# Usage:
#   ./scripts/run_all.sh
#
# Prerequisites:
#   Run ./scripts/setup.sh first to install dependencies

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASELINES_DIR="$(dirname "$SCRIPT_DIR")"

cd "$BASELINES_DIR"

# Load environment variables from .env file
if [ -f ".env" ]; then
    echo "Loading environment variables from .env..."
    set -a  # Export all variables
    source .env
    set +a
fi

# Check for required dependencies
check_dependency() {
    python -c "import $1" 2>/dev/null || {
        echo "ERROR: Missing dependency '$1'"
        echo "Please run './scripts/setup.sh' first to install all dependencies."
        exit 1
    }
}

echo "Checking dependencies..."
check_dependency "chz"
check_dependency "tinker"
check_dependency "nltk"
check_dependency "datasets"

# Check for TINKER_API_KEY
if [ -z "$TINKER_API_KEY" ]; then
    echo "WARNING: TINKER_API_KEY environment variable is not set."
    echo "Training and evaluation may fail without a valid API key."
    echo ""
fi

echo "============================================================"
echo "RTL-Smith Baseline Experiments"
echo "============================================================"
echo "Start time: $(date)"
echo ""

# Create directories
python -c "from config import create_directories; create_directories()"

# Step 1: Prepare all datasets
echo ""
echo "============================================================"
echo "STEP 1: Preparing Datasets"
echo "============================================================"
"$SCRIPT_DIR/prepare_all_data.sh"

# Step 2: Evaluate base model
echo ""
echo "============================================================"
echo "STEP 2: Evaluating Base Model"
echo "============================================================"
"$SCRIPT_DIR/eval_base.sh" --num-samples 5

# Step 3: Train on full datasets
echo ""
echo "============================================================"
echo "STEP 3: Training on Full Datasets"
echo "============================================================"
echo ""
echo ">>> Training on reasoning dataset..."
"$SCRIPT_DIR/train_reasoning.sh"

echo ""
echo ">>> Training on instruction dataset..."
"$SCRIPT_DIR/train_instruction.sh"

# Step 4: Subsampling experiments (optional - comment out if not needed)
echo ""
echo "============================================================"
echo "STEP 4: Subsampling Experiments"
echo "============================================================"
"$SCRIPT_DIR/run_subsample_experiments.sh"

echo ""
echo "============================================================"
echo "ALL EXPERIMENTS COMPLETE!"
echo "============================================================"
echo "End time: $(date)"
echo ""
echo "Results are stored in: /matx/u/ethanboneh/baselines_data/"
echo "  - datasets/     : Formatted training data"
echo "  - checkpoints/  : Model checkpoints"
echo "  - results/      : Evaluation results"
echo ""
echo "Next steps:"
echo "1. Check WandB dashboard for training metrics"
echo "2. Run evaluate_cvdp.py with fine-tuned model checkpoints"
echo "3. Compare results across different configurations"