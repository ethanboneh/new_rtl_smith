#!/bin/bash
# Evaluate base Qwen3-8B on CVDP benchmark
#
# This script runs the base model evaluation on CVDP before any fine-tuning.
# Uses multiple samples for pass@k evaluation.
#
# Usage:
#   ./scripts/eval_base.sh [--num-samples N]

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASELINES_DIR="$(dirname "$SCRIPT_DIR")"

cd "$BASELINES_DIR"

# Load environment variables from .env file
if [ -f ".env" ]; then
    set -a && source .env && set +a
fi

# Parse arguments
NUM_SAMPLES=5
while [[ $# -gt 0 ]]; do
    case $1 in
        --num-samples)
            NUM_SAMPLES="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTPUT_DIR="/matx/u/ethanboneh/baselines_data/results/base_model_${TIMESTAMP}"

echo "=============================================="
echo "Evaluating Base Qwen3-8B on CVDP"
echo "=============================================="
echo "Number of samples: $NUM_SAMPLES"
echo "Output directory: $OUTPUT_DIR"
echo ""

python evaluate_cvdp.py \
    --model base \
    --num-samples "$NUM_SAMPLES" \
    --output-dir "$OUTPUT_DIR" \
    --threads 4

echo ""
echo "=============================================="
echo "Evaluation complete!"
echo "Results saved to: $OUTPUT_DIR"
echo "=============================================="

