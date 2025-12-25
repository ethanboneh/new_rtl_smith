#!/bin/bash
# Setup script for RTL-Smith baseline experiments
#
# This script installs all required dependencies for:
# - Tinker training
# - CVDP benchmark evaluation
# - Dataset preparation
#
# Usage:
#   ./scripts/setup.sh

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASELINES_DIR="$(dirname "$SCRIPT_DIR")"
PROJECT_ROOT="$(dirname "$BASELINES_DIR")"

echo "=============================================="
echo "RTL-Smith Baselines Setup"
echo "=============================================="
echo "Project root: $PROJECT_ROOT"
echo ""

# Check for TINKER_API_KEY
if [ -z "$TINKER_API_KEY" ]; then
    echo "WARNING: TINKER_API_KEY environment variable is not set."
    echo "You will need to set it before running training or evaluation."
    echo "  export TINKER_API_KEY=sk-..."
    echo ""
fi

# Install baselines requirements
echo ">>> Installing baselines requirements..."
pip install -r "$BASELINES_DIR/requirements.txt"

# Install tinker-cookbook
echo ""
echo ">>> Installing tinker-cookbook..."
pip install -e "$PROJECT_ROOT/tinker-cookbook[wandb]"

# Install CVDP benchmark requirements
echo ""
echo ">>> Installing CVDP benchmark requirements..."
pip install -r "$PROJECT_ROOT/cvdp_benchmark/requirements.txt"

# Download NLTK data (required for CVDP)
echo ""
echo ">>> Downloading NLTK data..."
python -c "import nltk; nltk.download('punkt', quiet=True); nltk.download('punkt_tab', quiet=True)"

# Create data directories
echo ""
echo ">>> Creating data directories..."
python -c "from config import create_directories; create_directories()"

echo ""
echo "=============================================="
echo "Setup complete!"
echo "=============================================="
echo ""
echo "Next steps:"
echo "1. Set TINKER_API_KEY if not already set:"
echo "   export TINKER_API_KEY=sk-..."
echo ""
echo "2. Run experiments:"
echo "   ./scripts/run_all.sh"
echo ""
echo "Or run individual steps:"
echo "   ./scripts/prepare_all_data.sh"
echo "   ./scripts/eval_base.sh"
echo "   ./scripts/train_reasoning.sh"

