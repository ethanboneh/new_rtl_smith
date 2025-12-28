#!/bin/bash
# Train Qwen3-8B on RTLLM + Corruption dataset
#
# This script trains on a combined dataset that includes:
# 1. RTLLM spec-to-code examples (for generation)
# 2. Corruption/debugging examples (buggy code -> fixed code with reasoning)
#
# This trains the model for both code generation AND debugging capabilities.
#
# Usage:
#   ./scripts/train_rtllm_corruption.sh [--subsample FRACTION]

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
echo "Training on RTLLM + Corruption Dataset"
echo "=============================================="
echo "Subsample fraction: $SUBSAMPLE"
echo "Model: Qwen/Qwen3-8B"
echo "WandB project: rtl-smith-baselines"
echo ""
echo "This dataset combines:"
echo "  - RTLLM spec-to-code examples"
echo "  - Corruption/debugging examples (with reasoning traces)"
echo ""

# Ensure dataset exists
FRACTION_STR=""
if [[ "$SUBSAMPLE" != "1.0" ]]; then
    FRACTION_PCT=$(python -c "print(int(float('$SUBSAMPLE') * 100))")
    FRACTION_STR="_${FRACTION_PCT}pct"
fi
DATASET_PATH="/matx/u/ethanboneh/baselines_data/datasets/rtllm_corruption${FRACTION_STR}_formatted.jsonl"

if [[ ! -f "$DATASET_PATH" ]]; then
    echo "Dataset not found: $DATASET_PATH"
    echo "Running dataset preparation first..."
    python prepare_datasets.py --dataset rtllm_corruption --subsample "$SUBSAMPLE"
fi

# Run training
# Note: Combined dataset is larger but still relatively small
# Batch size will auto-adjust if needed
python train_tinker.py \
    --dataset rtllm_corruption \
    --subsample "$SUBSAMPLE" \
    --learning-rate 5e-4 \
    --lora-rank 64 \
    --batch-size 16 \
    --max-length 8192 \
    --epochs 2 \
    --test-size 10 \
    --wandb-project rtl-smith-baselines \
    --overwrite

echo ""
echo "=============================================="
echo "Training complete!"
echo "=============================================="

