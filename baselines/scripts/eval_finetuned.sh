#!/bin/bash
# Evaluate fine-tuned models on CVDP
# This script automatically uses the correct sampler_weights paths

set -e

# Source environment
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/.."

if [ -f .env ]; then
    set -a
    source .env
    set +a
fi

# Checkpoint sampler paths (use these, NOT the state_path/weights paths!)
declare -A INSTRUCTION_CHECKPOINTS=(
    ["100pct"]="tinker://459fe071-8fe6-5c07-b2ea-197719fc44f9:train:0/sampler_weights/final"
    ["75pct"]="tinker://0183c8eb-8692-5c5e-9ee0-a13ea60263eb:train:0/sampler_weights/final"
    ["50pct"]="tinker://ea2767bc-0296-5737-af80-86636c06d7e0:train:0/sampler_weights/final"
    ["25pct"]="tinker://4740acb4-db5e-5b4c-9903-ebb4ab967e31:train:0/sampler_weights/final"
    ["10pct"]="tinker://b35209b7-6dac-58af-9e91-03c7cb5985b8:train:0/sampler_weights/final"
)

declare -A REASONING_CHECKPOINTS=(
    ["100pct"]="tinker://656c516a-ea20-5b39-81e8-89c41723d9ab:train:0/sampler_weights/final"
    ["75pct"]="tinker://d651045f-aa56-5475-8662-4c58ba368f8d:train:0/sampler_weights/final"
    ["50pct"]="tinker://ceec7820-da05-593a-a33b-d83e36944ee0:train:0/sampler_weights/final"
    ["25pct"]="tinker://778e3c90-98c4-5af2-8d60-d110bf66977e:train:0/sampler_weights/final"
    ["10pct"]="tinker://0cf339ba-d92e-53d4-a7ef-bd829649932c:train:0/sampler_weights/final"
)

usage() {
    echo "Usage: $0 <model_type> <fraction> [num_samples]"
    echo ""
    echo "Arguments:"
    echo "  model_type    : 'instruction' or 'reasoning'"
    echo "  fraction      : '10pct', '25pct', '50pct', '75pct', or '100pct'"
    echo "  num_samples   : Number of samples for pass@k (default: 5)"
    echo ""
    echo "Example:"
    echo "  $0 instruction 100pct 5"
    echo "  $0 reasoning 50pct 3"
    echo ""
    echo "Available checkpoints:"
    echo "  Instruction: ${!INSTRUCTION_CHECKPOINTS[@]}"
    echo "  Reasoning:   ${!REASONING_CHECKPOINTS[@]}"
}

if [ $# -lt 2 ]; then
    usage
    exit 1
fi

MODEL_TYPE=$1
FRACTION=$2
NUM_SAMPLES=${3:-5}

# Get the correct checkpoint path
if [ "$MODEL_TYPE" == "instruction" ]; then
    CHECKPOINT="${INSTRUCTION_CHECKPOINTS[$FRACTION]}"
    export TINKER_INSTRUCTION_MODEL_PATH="$CHECKPOINT"
elif [ "$MODEL_TYPE" == "reasoning" ]; then
    CHECKPOINT="${REASONING_CHECKPOINTS[$FRACTION]}"
    export TINKER_REASONING_MODEL_PATH="$CHECKPOINT"
else
    echo "Error: model_type must be 'instruction' or 'reasoning'"
    usage
    exit 1
fi

if [ -z "$CHECKPOINT" ]; then
    echo "Error: No checkpoint found for fraction '$FRACTION'"
    echo "Available fractions: 10pct, 25pct, 50pct, 75pct, 100pct"
    exit 1
fi

OUTPUT_DIR="/matx/u/ethanboneh/baselines_data/results/${MODEL_TYPE}_${FRACTION}_finetuned"

echo "=========================================="
echo "Evaluating Fine-tuned Model"
echo "=========================================="
echo "Model type:  $MODEL_TYPE"
echo "Fraction:    $FRACTION"
echo "Checkpoint:  $CHECKPOINT"
echo "Samples:     $NUM_SAMPLES"
echo "Output:      $OUTPUT_DIR"
echo "=========================================="
echo ""

python evaluate_cvdp.py \
    --model "$MODEL_TYPE" \
    --num-samples "$NUM_SAMPLES" \
    --output-dir "$OUTPUT_DIR"

echo ""
echo "✓ Evaluation complete!"
echo "Results saved to: $OUTPUT_DIR"

