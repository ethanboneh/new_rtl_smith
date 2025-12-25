#!/bin/bash
# Prepare all datasets for baseline experiments
#
# This script prepares both reasoning and instruction datasets
# at various subsample fractions for the saturation experiments.
#
# Usage:
#   ./scripts/prepare_all_data.sh

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASELINES_DIR="$(dirname "$SCRIPT_DIR")"

cd "$BASELINES_DIR"

echo "=============================================="
echo "Preparing VeriThoughts Datasets"
echo "=============================================="
echo "Output directory: /matx/u/ethanboneh/baselines_data/datasets"
echo ""

# Prepare full datasets
echo ">>> Preparing full datasets..."
python prepare_datasets.py --dataset both --subsample 1.0

# Prepare subsampled datasets for saturation experiments
echo ""
echo ">>> Preparing subsampled datasets..."
python prepare_datasets.py --dataset both --subsample 0.1 0.25 0.5 0.75

echo ""
echo "=============================================="
echo "Dataset preparation complete!"
echo "=============================================="

