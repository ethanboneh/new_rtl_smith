#!/bin/bash
# Run subsampling experiments to check data saturation
#
# This script trains models on various fractions of the dataset
# to determine if we can achieve similar performance with less data.
#
# Usage:
#   ./scripts/run_subsample_experiments.sh [--dataset DATASET]

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASELINES_DIR="$(dirname "$SCRIPT_DIR")"

cd "$BASELINES_DIR"

# Load environment variables from .env file
if [ -f ".env" ]; then
    set -a && source .env && set +a
fi

# Parse arguments
DATASET="both"
while [[ $# -gt 0 ]]; do
    case $1 in
        --dataset)
            DATASET="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Subsample fractions to test
FRACTIONS="0.1 0.25 0.5 0.75"

echo "=============================================="
echo "Running Subsampling Experiments"
echo "=============================================="
echo "Dataset: $DATASET"
echo "Fractions: $FRACTIONS"
echo ""

# Determine which datasets to run
if [[ "$DATASET" == "both" ]]; then
    DATASETS="reasoning instruction"
else
    DATASETS="$DATASET"
fi

# Prepare all subsampled datasets first
echo ">>> Preparing datasets..."
for ds in $DATASETS; do
    python prepare_datasets.py --dataset "$ds" --subsample $FRACTIONS
done

# Run training for each combination
echo ""
echo ">>> Running training experiments..."
for ds in $DATASETS; do
    for frac in $FRACTIONS; do
        echo ""
        echo "========================================"
        echo "Training: $ds at ${frac} fraction"
        echo "========================================"
        
        python train_tinker.py \
            --dataset "$ds" \
            --subsample "$frac" \
            --learning-rate 5e-4 \
            --lora-rank 64 \
            --batch-size 64 \
            --max-length 8192 \
            --epochs 1 \
            --wandb-project rtl-smith-baselines \
            --overwrite
    done
done

echo ""
echo "=============================================="
echo "Subsampling experiments complete!"
echo "=============================================="
echo ""
echo "Next steps:"
echo "1. Review WandB dashboard for training curves"
echo "2. Evaluate each checkpoint on CVDP"
echo "3. Compare pass@k across different fractions"

