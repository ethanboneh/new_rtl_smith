#!/bin/bash
# Train Qwen3-8B on RTLLM dataset (spec-to-code)
#
# This script trains on the RTLLM benchmark dataset which contains
# natural language specifications paired with verified Verilog implementations.
#
# Usage:
#   ./scripts/train_rtllm.sh [--subsample FRACTION]

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASELINES_DIR="$(dirname "$SCRIPT_DIR")"

cd "$BASELINES_DIR"

# Load environment variables from .env file
if [ -f ".env" ]; then
    set -a && source .env && set +a
fi

# Parse arguments
SUBSAMPLE="1.0"
while [[ $# -gt 0 ]]; do
    case $1 in
        --subsample)
            SUBSAMPLE="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

echo "=============================================="
echo "Training on RTLLM Dataset"
echo "=============================================="
echo "Subsample fraction: $SUBSAMPLE"
echo "Model: Qwen/Qwen3-8B"
echo "WandB project: rtl-smith-baselines"
echo ""

# Ensure dataset exists
FRACTION_STR=""
if [[ "$SUBSAMPLE" != "1.0" ]]; then
    FRACTION_PCT=$(python -c "print(int(float('$SUBSAMPLE') * 100))")
    FRACTION_STR="_${FRACTION_PCT}pct"
fi
DATASET_PATH="/matx/u/ethanboneh/baselines_data/datasets/rtllm${FRACTION_STR}_formatted.jsonl"

if [[ ! -f "$DATASET_PATH" ]]; then
    echo "Dataset not found: $DATASET_PATH"
    echo "Running dataset preparation first..."
    python prepare_datasets.py --dataset rtllm --subsample "$SUBSAMPLE"
fi

# Run training
# Note: RTLLM has ~50 examples, so we use smaller batch size and more epochs
python train_tinker.py \
    --dataset rtllm \
    --subsample "$SUBSAMPLE" \
    --learning-rate 5e-4 \
    --lora-rank 64 \
    --batch-size 8 \
    --max-length 8192 \
    --epochs 3 \
    --test-size 5 \
    --wandb-project rtl-smith-baselines \
    --overwrite

echo ""
echo "=============================================="
echo "Training complete!"
echo "=============================================="

