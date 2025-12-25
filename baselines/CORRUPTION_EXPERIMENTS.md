# Corruption Experiments Guide

This guide explains how to run experiments mixing VeriThoughts training data with corruption (bug-fixing) data.

## Overview

The corruption experiments allow you to:
1. Format corruption data from the `llm_corruption` pipeline into training format
2. Mix VeriThoughts data with corruption data in various ratios
3. Train models on the mixed datasets with or without reasoning traces

## Workflow

### Step 1: Generate Corruption Data

First, generate corruption data using the LLM corruption pipeline:

```bash
cd baselines/llm_corruption
python scripts/generate_corruptions.py \
    --dataset-type instruction \
    --output outputs/corruptions.jsonl \
    --model o3-mini \
    --skip-linter \
    --max-entries 1000
```

### Step 2: Prepare Corruption Dataset

Format the corruption data for training. You can create two variants:

**With reasoning:**
```bash
cd baselines
python prepare_corruption_datasets.py \
    --corruption-file llm_corruption/outputs/corruptions.jsonl \
    --include-reasoning \
    --subsample 1.0
```

**Without reasoning (instruction-only):**
```bash
python prepare_corruption_datasets.py \
    --corruption-file llm_corruption/outputs/corruptions.jsonl \
    --no-reasoning \
    --subsample 1.0
```

This creates formatted datasets in `/matx/u/ethanboneh/baselines_data/datasets/`:
- `corruption_reasoning_formatted.jsonl` (with reasoning)
- `corruption_instruction_formatted.jsonl` (without reasoning)

### Step 3: Mix Datasets

Mix VeriThoughts and corruption data in your desired ratio:

```bash
# Mix 50% VeriThoughts reasoning + 50% corruption reasoning
python mix_datasets.py \
    --verithoughts reasoning \
    --corruption corruption_reasoning \
    --ratio 0.5 \
    --output /matx/u/ethanboneh/baselines_data/datasets/mixed_reasoning_50pct.jsonl

# Mix 80% VeriThoughts instruction + 20% corruption instruction
python mix_datasets.py \
    --verithoughts instruction \
    --corruption corruption_instruction \
    --ratio 0.2 \
    --output /matx/u/ethanboneh/baselines_data/datasets/mixed_instruction_20pct.jsonl
```

### Step 4: Train on Mixed Dataset

Train using the `train_corruption.sh` script:

```bash
# Train on mixed reasoning dataset (50% corruption)
./scripts/train_corruption.sh \
    --verithoughts reasoning \
    --corruption-ratio 0.5 \
    --include-reasoning \
    --corruption-file llm_corruption/outputs/corruptions.jsonl

# Train on mixed instruction dataset (20% corruption)
./scripts/train_corruption.sh \
    --verithoughts instruction \
    --corruption-ratio 0.2 \
    --no-reasoning \
    --corruption-file llm_corruption/outputs/corruptions.jsonl
```

Or train directly using `train_tinker.py` with a custom dataset path:

```bash
python train_tinker.py \
    --dataset-path /matx/u/ethanboneh/baselines_data/datasets/mixed_reasoning_50pct.jsonl \
    --learning-rate 5e-4 \
    --lora-rank 64 \
    --batch-size 64 \
    --max-length 8192 \
    --epochs 1 \
    --wandb-project rtl-smith-baselines
```

## Quick Start Example

Complete workflow for a quick experiment:

```bash
# 1. Generate corruption data (if not already done)
cd baselines/llm_corruption
python scripts/generate_corruptions.py \
    --dataset-type instruction \
    --output outputs/corruptions.jsonl \
    --model o3-mini \
    --skip-linter \
    --max-entries 100

# 2. Prepare corruption dataset (with reasoning)
cd ../baselines
python prepare_corruption_datasets.py \
    --corruption-file llm_corruption/outputs/corruptions.jsonl \
    --include-reasoning

# 3. Ensure VeriThoughts dataset exists
python prepare_datasets.py --dataset reasoning --subsample 1.0

# 4. Mix datasets (50% corruption)
python mix_datasets.py \
    --verithoughts reasoning \
    --corruption corruption_reasoning \
    --ratio 0.5 \
    --output /matx/u/ethanboneh/baselines_data/datasets/mixed_reasoning_50pct.jsonl

# 5. Train
./scripts/train_corruption.sh \
    --verithoughts reasoning \
    --corruption-ratio 0.5 \
    --include-reasoning \
    --corruption-file llm_corruption/outputs/corruptions.jsonl
```

## Script Options

### `prepare_corruption_datasets.py`

- `--corruption-file`: Path to corruption JSONL file (required)
- `--include-reasoning`: Include reasoning traces (default)
- `--no-reasoning`: Exclude reasoning traces (instruction-only)
- `--subsample`: Subsample fractions (e.g., `0.1 0.25 0.5 1.0`)
- `--seed`: Random seed for shuffling
- `--output-dir`: Output directory (default: from config)

### `mix_datasets.py`

- `--verithoughts`: VeriThoughts type (`reasoning` or `instruction`)
- `--corruption`: Corruption type (`corruption_reasoning` or `corruption_instruction`)
- `--ratio`: Corruption ratio (0.0 to 1.0, e.g., 0.5 = 50% corruption)
- `--subsample`: Subsample fraction for VeriThoughts
- `--corruption-subsample`: Subsample fraction for corruption data
- `--output`: Output JSONL file path (required)
- `--seed`: Random seed
- `--no-shuffle`: Don't shuffle final dataset

### `train_corruption.sh`

- `--verithoughts`: VeriThoughts type (`reasoning` or `instruction`)
- `--corruption-ratio`: Corruption ratio (0.0 to 1.0)
- `--include-reasoning`: Include reasoning traces
- `--no-reasoning`: Exclude reasoning traces
- `--corruption-file`: Path to corruption JSONL (will prepare if needed)
- `--subsample`: Subsample fraction for VeriThoughts
- `--corruption-subsample`: Subsample fraction for corruption

## Data Format

### Corruption Dataset Format

The corruption data is formatted as:

**With reasoning:**
```json
{
  "messages": [
    {
      "role": "user",
      "content": "Fix the following buggy SystemVerilog code:\n\n[corrupted_code]\n\nIssue: [issue_description]"
    },
    {
      "role": "assistant",
      "content": "<think>\n[reasoning_trace]\n</think>\n[BEGIN]\n[clean_code]\n[DONE]"
    }
  ]
}
```

**Without reasoning:**
```json
{
  "messages": [
    {
      "role": "user",
      "content": "Fix the following buggy SystemVerilog code:\n\n[corrupted_code]\n\nIssue: [issue_description]"
    },
    {
      "role": "assistant",
      "content": "[BEGIN]\n[clean_code]\n[DONE]"
    }
  ]
}
```

## Notes

- Existing scripts (`prepare_all_data.sh`, `train_instruction.sh`, `train_reasoning.sh`) continue to work unchanged
- The corruption experiments are additive - they don't modify existing functionality
- All formatted datasets are saved in `/matx/u/ethanboneh/baselines_data/datasets/`
- Training checkpoints are saved in `/matx/u/ethanboneh/baselines_data/checkpoints/`

