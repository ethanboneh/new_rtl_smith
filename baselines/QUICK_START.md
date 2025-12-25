# Quick Start Commands

## Evaluate Base Qwen3-8B (No Fine-tuning)

### Option 1: Using Python script (recommended)
```bash
cd /afs/cs.stanford.edu/u/ethanboneh/new_rtl_smith/baselines

# Single sample evaluation
python evaluate_cvdp.py --model base

# Pass@5 evaluation (5 samples per problem)
python evaluate_cvdp.py --model base --num-samples 5

# With custom output directory
python evaluate_cvdp.py --model base --num-samples 5 --output-dir /path/to/results

# With more threads for faster evaluation
python evaluate_cvdp.py --model base --num-samples 5 --threads 8
```

### Option 2: Using shell script
```bash
cd /afs/cs.stanford.edu/u/ethanboneh/new_rtl_smith/baselines

# Default: 5 samples
./scripts/eval_base.sh

# Custom number of samples
./scripts/eval_base.sh --num-samples 10
```

## What This Does

- Uses **base Qwen3-8B** model (no fine-tuning)
- Evaluates on **CVDP benchmark** (303 problems, non-commercial)
- Uses your **custom model factory** (`model_factory_qwen.py`)
- Runs test harnesses in **Docker containers** (requires Docker permissions)
- Generates **pass@k** metrics

## Prerequisites

1. ✅ Tinker API key set: `export TINKER_API_KEY=sk-...`
2. ✅ Docker working: `docker ps` (or fix permissions first)
3. ✅ CVDP benchmark installed (already done)

## Expected Output

Results will be saved to:
```
/matx/u/ethanboneh/baselines_data/results/cvdp_base_<timestamp>/
```

Contains:
- `report.json` - Detailed results
- `composite_report.json` - Pass@k metrics (if num-samples > 1)
- `eval_config.json` - Configuration used

