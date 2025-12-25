# RTL-Smith Baseline Experiments

This directory contains baseline experiments for spec-to-Verilog generation using Qwen3-8B and the CVDP benchmark.

## Overview

We evaluate Qwen3-8B on the CVDP (Code Verification and Debug Problem) benchmark:
1. **Base model evaluation**: Evaluate vanilla Qwen3-8B
2. **Fine-tuning on VeriThoughts Reasoning**: Train on dataset with chain-of-thought reasoning
3. **Fine-tuning on VeriThoughts Instruction**: Train on dataset with code-only output
4. **Subsampling experiments**: Check data saturation by training on subsets

### Datasets

| Dataset | Source | Output Format | Size |
|---------|--------|---------------|------|
| VeriThoughts Reasoning | `wilyub/VeriThoughtsTrainSetConsistentReasoning` | `<think>...</think>` + code | ~10k |
| VeriThoughts Instruction | `wilyub/VeriThoughtsTrainSetConsistentInstruction` | Code only | ~10k |

### Benchmark

- **CVDP v1.0.2**: Non-agentic code generation benchmark (no commercial problems)
- **Dataset**: `cvdp_v1.0.2_nonagentic_code_generation_no_commercial.jsonl` (303 problems)
- **Metric**: Pass@k evaluation

## Project Structure

```
baselines/
├── config.py                 # Configuration constants and paths
├── prepare_datasets.py       # Format datasets for training
├── train_tinker.py          # Training script using Tinker
├── evaluate_cvdp.py         # CVDP benchmark evaluation
├── model_factory_qwen.py    # Custom model factory for CVDP
├── run_experiments.py       # Master experiment orchestration
├── scripts/                 # Shell scripts for running experiments
│   ├── prepare_all_data.sh
│   ├── train_reasoning.sh
│   ├── train_instruction.sh
│   ├── eval_base.sh
│   ├── run_subsample_experiments.sh
│   └── run_all.sh
└── README.md
```

## Quick Start

### Prerequisites

1. Tinker API key set as `TINKER_API_KEY` environment variable
2. Python 3.11+

### Setup

Run the setup script to install all dependencies:

```bash
cd baselines
./scripts/setup.sh
```

Or install manually:

```bash
# Install baselines requirements
pip install -r requirements.txt

# Install tinker-cookbook with wandb support
pip install -e ../tinker-cookbook[wandb]

# Install CVDP benchmark requirements
pip install -r ../cvdp_benchmark/requirements.txt

# Download NLTK data
python -c "import nltk; nltk.download('punkt'); nltk.download('punkt_tab')"
```

### Running All Experiments

```bash
# Run the complete experimental pipeline
./scripts/run_all.sh
```

Or run individual steps:

```bash
# 1. Prepare datasets
./scripts/prepare_all_data.sh

# 2. Evaluate base model
./scripts/eval_base.sh

# 3. Train on datasets
./scripts/train_reasoning.sh
./scripts/train_instruction.sh

# 4. Subsampling experiments
./scripts/run_subsample_experiments.sh
```

## Detailed Usage

### Dataset Preparation

Format VeriThoughts datasets for training:

```bash
# Prepare both datasets (full and subsampled)
python prepare_datasets.py --dataset both --subsample 1.0 0.5 0.25 0.1

# Prepare only reasoning dataset
python prepare_datasets.py --dataset reasoning --subsample 1.0
```

Output format (JSONL with messages):
```json
{"messages": [
  {"role": "user", "content": "You are a Verilog RTL designer...\n\nQuestion:\n..."},
  {"role": "assistant", "content": "[BEGIN]\nmodule...\n[DONE]"}
]}
```

### Training

Train Qwen3-8B using Tinker:

```bash
# Train on reasoning dataset (full)
python train_tinker.py --dataset reasoning

# Train on instruction dataset with custom hyperparameters
python train_tinker.py --dataset instruction \
    --learning-rate 1e-4 \
    --lora-rank 128 \
    --batch-size 32

# Train on subsampled dataset
python train_tinker.py --dataset reasoning --subsample 0.25
```

Training hyperparameters (defaults):
- Learning rate: 5e-4
- LoRA rank: 64
- Batch size: 64
- Max sequence length: 8192
- Epochs: 1

### Evaluation

Evaluate models on CVDP:

```bash
# Evaluate base model with pass@5
python evaluate_cvdp.py --model base --num-samples 5

# Evaluate fine-tuned model
python evaluate_cvdp.py --model reasoning \
    --checkpoint-path tinker://path/to/checkpoint \
    --num-samples 5
```

### Subsampling Experiments

Test data saturation by training on different fractions:

```bash
# Run all subsample fractions (10%, 25%, 50%, 75%)
./scripts/run_subsample_experiments.sh

# Or manually for specific fractions
python train_tinker.py --dataset reasoning --subsample 0.1
python train_tinker.py --dataset reasoning --subsample 0.25
python train_tinker.py --dataset reasoning --subsample 0.5
python train_tinker.py --dataset reasoning --subsample 0.75
```

## Data Storage

All data is stored in `/matx/u/ethanboneh/baselines_data/`:

```
/matx/u/ethanboneh/baselines_data/
├── datasets/           # Formatted training data
│   ├── verithoughts_reasoning_formatted.jsonl
│   ├── verithoughts_reasoning_25pct_formatted.jsonl
│   ├── verithoughts_instruction_formatted.jsonl
│   └── ...
├── checkpoints/        # Model checkpoints from training
│   └── reasoning_qwen3_8b_lr5e-4_rank64_*/
└── results/           # CVDP evaluation results
    └── base_model_*/
```

## WandB Logging

Training metrics are logged to WandB:
- Project: `rtl-smith-baselines`
- Metrics: train_mean_nll, test/nll, learning_rate, progress

View at: https://wandb.ai/[your-username]/rtl-smith-baselines

## Format Consistency

### Training Data Format

1. **User message**: The instruction/spec from the dataset (simple, no extra prompts)

2. **Assistant response**:
   - **Reasoning dataset**: `<think>...</think>` reasoning followed by `[BEGIN]...[DONE]`
   - **Instruction dataset**: Just `[BEGIN]...[DONE]`

Example (reasoning):
```
<think>
Let me analyze this problem...
The module needs to implement a counter...
</think>
[BEGIN]
module counter(
    input clk, rst,
    output reg [7:0] count
);
    always @(posedge clk)
        if (rst) count <= 0;
        else count <= count + 1;
endmodule
[DONE]
```

### CVDP Evaluation

Our custom model factory (`model_factory_qwen.py`) handles code extraction:
- Extracts code from `[BEGIN]...[DONE]` markers
- Strips `<think>...</think>` reasoning traces
- Falls back to module detection or full response

This allows models fine-tuned with reasoning traces to be evaluated correctly on CVDP.

## Expected Results

Based on typical performance (actual results may vary):

| Model | Pass@1 | Pass@5 |
|-------|--------|--------|
| Base Qwen3-8B | ~X% | ~Y% |
| + Reasoning FT | ~X% | ~Y% |
| + Instruction FT | ~X% | ~Y% |

Subsampling experiments will show learning curves as a function of data fraction.

## Troubleshooting

### Dataset not found
```bash
# Run dataset preparation first
python prepare_datasets.py --dataset [reasoning|instruction] --subsample [fraction]
```

### Tinker API errors
```bash
# Verify API key is set
echo $TINKER_API_KEY

# Check connection
python -c "import tinker; print(tinker.ServiceClient())"
```

### Out of memory
Reduce batch size or max sequence length:
```bash
python train_tinker.py --dataset reasoning --batch-size 32 --max-length 4096
```

## References

- [CVDP Benchmark](../cvdp_benchmark/README.md)
- [Tinker Cookbook](../tinker-cookbook/README.md)
- [VeriThoughts Dataset](https://huggingface.co/datasets/wilyub/VeriThoughtsTrainSetConsistentReasoning)
