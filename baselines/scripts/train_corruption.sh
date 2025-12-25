#!/bin/bash
# Train Qwen3-8B on mixed VeriThoughts + Corruption datasets
#
# This script trains on a mixed dataset combining VeriThoughts with
# corruption data for bug-fixing tasks.
#
# Usage:
#   ./scripts/train_corruption.sh \
#       --verithoughts reasoning \
#       --corruption-ratio 0.5 \
#       --include-reasoning
#
#   ./scripts/train_corruption.sh \
#       --verithoughts instruction \
#       --corruption-ratio 0.2 \
#       --no-reasoning

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASELINES_DIR="$(dirname "$SCRIPT_DIR")"

cd "$BASELINES_DIR"

# Load environment variables from .env file
if [ -f ".env" ]; then
    set -a && source .env && set +a
fi

# Parse arguments
VERITHOUGHTS_TYPE="reasoning"
CORRUPTION_RATIO="0.5"
INCLUDE_REASONING=true
CORRUPTION_FILE=""
SUBSAMPLE="1.0"
CORRUPTION_SUBSAMPLE="1.0"

while [[ $# -gt 0 ]]; do
    case $1 in
        --verithoughts)
            VERITHOUGHTS_TYPE="$2"
            shift 2
            ;;
        --corruption-ratio)
            CORRUPTION_RATIO="$2"
            shift 2
            ;;
        --include-reasoning)
            INCLUDE_REASONING=true
            shift
            ;;
        --no-reasoning)
            INCLUDE_REASONING=false
            shift
            ;;
        --corruption-file)
            CORRUPTION_FILE="$2"
            shift 2
            ;;
        --subsample)
            SUBSAMPLE="$2"
            shift 2
            ;;
        --corruption-subsample)
            CORRUPTION_SUBSAMPLE="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Determine corruption dataset type
if [ "$INCLUDE_REASONING" = true ]; then
    CORRUPTION_TYPE="corruption_reasoning"
else
    CORRUPTION_TYPE="corruption_instruction"
fi

echo "=============================================="
echo "Training on Mixed VeriThoughts + Corruption Dataset"
echo "=============================================="
echo "VeriThoughts type: $VERITHOUGHTS_TYPE"
echo "Corruption type: $CORRUPTION_TYPE"
echo "Corruption ratio: $CORRUPTION_RATIO"
echo "Subsample: $SUBSAMPLE"
echo "Model: Qwen/Qwen3-8B"
echo "WandB project: rtl-smith-baselines"
echo ""

# Step 1: Prepare corruption dataset if needed
if [ -n "$CORRUPTION_FILE" ]; then
    echo ">>> Preparing corruption dataset..."
    REASONING_FLAG=""
    if [ "$INCLUDE_REASONING" = false ]; then
        REASONING_FLAG="--no-reasoning"
    fi
    
    python prepare_corruption_datasets.py \
        --corruption-file "$CORRUPTION_FILE" \
        $REASONING_FLAG \
        --subsample "$CORRUPTION_SUBSAMPLE"
fi

# Step 2: Mix datasets
echo ""
echo ">>> Mixing datasets..."
FRACTION_STR=""
if [[ "$SUBSAMPLE" != "1.0" ]]; then
    FRACTION_PCT=$(python -c "print(int(float('$SUBSAMPLE') * 100))")
    FRACTION_STR="_${FRACTION_PCT}pct"
fi

CORRUPTION_FRACTION_STR=""
if [[ "$CORRUPTION_SUBSAMPLE" != "1.0" ]]; then
    CORRUPTION_FRACTION_PCT=$(python -c "print(int(float('$CORRUPTION_SUBSAMPLE') * 100))")
    CORRUPTION_FRACTION_STR="_${CORRUPTION_FRACTION_PCT}pct"
fi

RATIO_PCT=$(python -c "print(int(float('$CORRUPTION_RATIO') * 100))")
MIXED_DATASET_NAME="mixed_${VERITHOUGHTS_TYPE}_${CORRUPTION_TYPE}_${RATIO_PCT}pct${FRACTION_STR}.jsonl"
MIXED_DATASET_PATH="/matx/u/ethanboneh/baselines_data/datasets/${MIXED_DATASET_NAME}"

# Check if mixed dataset already exists
if [[ ! -f "$MIXED_DATASET_PATH" ]]; then
    python mix_datasets.py \
        --verithoughts "$VERITHOUGHTS_TYPE" \
        --corruption "$CORRUPTION_TYPE" \
        --ratio "$CORRUPTION_RATIO" \
        --subsample "$SUBSAMPLE" \
        --corruption-subsample "$CORRUPTION_SUBSAMPLE" \
        --output "$MIXED_DATASET_PATH"
else
    echo "Mixed dataset already exists: $MIXED_DATASET_PATH"
fi

# Step 3: Ensure VeriThoughts dataset exists
VERITHOUGHTS_DATASET_PATH="/matx/u/ethanboneh/baselines_data/datasets/verithoughts_${VERITHOUGHTS_TYPE}${FRACTION_STR}_formatted.jsonl"
if [[ ! -f "$VERITHOUGHTS_DATASET_PATH" ]]; then
    echo "VeriThoughts dataset not found: $VERITHOUGHTS_DATASET_PATH"
    echo "Running dataset preparation first..."
    python prepare_datasets.py --dataset "$VERITHOUGHTS_TYPE" --subsample "$SUBSAMPLE"
fi

# Step 4: Run training
echo ""
echo ">>> Running training..."
python train_tinker.py \
    --dataset-path "$MIXED_DATASET_PATH" \
    --learning-rate 5e-4 \
    --lora-rank 64 \
    --batch-size 64 \
    --max-length 8192 \
    --epochs 1 \
    --wandb-project rtl-smith-baselines \
    --overwrite

echo ""
echo "=============================================="
echo "Training complete!"
echo "=============================================="

