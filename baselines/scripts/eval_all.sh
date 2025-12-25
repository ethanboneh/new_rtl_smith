#!/bin/bash
# Run all evaluations: base model + all fine-tuned models
#
# This script evaluates:
# 1. Base Qwen3-8B
# 2. All instruction-tuned models (10%, 25%, 50%, 75%, 100%)
# 3. All reasoning-tuned models (10%, 25%, 50%, 75%, 100%)
#
# Usage:
#   ./scripts/eval_all.sh [num_samples]

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
NUM_SAMPLES=${1:-5}

echo "=========================================="
echo "Running All CVDP Evaluations"
echo "=========================================="
echo "Samples per model: $NUM_SAMPLES"
echo ""

# 1. Evaluate base model
echo "[1/11] Evaluating base model..."
"$SCRIPT_DIR/eval_base.sh" --num-samples "$NUM_SAMPLES"

# 2. Evaluate instruction models
for fraction in 100pct 75pct 50pct 25pct 10pct; do
    echo ""
    echo "[Instruction $fraction] Evaluating..."
    "$SCRIPT_DIR/eval_finetuned.sh" instruction "$fraction" "$NUM_SAMPLES"
done

# 3. Evaluate reasoning models
for fraction in 100pct 75pct 50pct 25pct 10pct; do
    echo ""
    echo "[Reasoning $fraction] Evaluating..."
    "$SCRIPT_DIR/eval_finetuned.sh" reasoning "$fraction" "$NUM_SAMPLES"
done

echo ""
echo "=========================================="
echo "All Evaluations Complete!"
echo "=========================================="
echo ""
echo "Results saved to /matx/u/ethanboneh/baselines_data/results/"
echo ""
echo "To compare results, check:"
echo "  - base_model_*/report.json"
echo "  - instruction_*_finetuned/report.json"
echo "  - reasoning_*_finetuned/report.json"

