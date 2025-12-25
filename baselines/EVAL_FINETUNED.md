# Evaluating Fine-Tuned Models

## Finding Your Checkpoint Path

After training, Tinker saves checkpoints. You need to find the **sampler_weights** path (not the weights path).

### Method 1: Check checkpoints.jsonl

Look in your checkpoint directory:
```bash
cd /matx/u/ethanboneh/baselines_data/checkpoints/reasoning_Qwen3_8B_lr0.0005_rank64_20251208_164150/
cat checkpoints.jsonl | grep sampler_weights
```

This will show paths like:
```
tinker://xxx:train:0/sampler_weights/final
```

### Method 2: Use the Latest Checkpoint

The most recent checkpoint directory is:
```
reasoning_Qwen3_8B_lr0.0005_rank64_20251208_164150
```

## Commands to Evaluate

### For Reasoning Dataset (Full VeriThoughts)

```bash
cd /afs/cs.stanford.edu/u/ethanboneh/new_rtl_smith/baselines

# Set the checkpoint path as environment variable
export TINKER_REASONING_MODEL_PATH="tinker://xxx:train:0/sampler_weights/final"

# Or pass it directly
python evaluate_cvdp.py \
    --model reasoning \
    --checkpoint-path "tinker://xxx:train:0/sampler_weights/final" \
    --num-samples 5
```

### For Instruction Dataset

```bash
export TINKER_INSTRUCTION_MODEL_PATH="tinker://xxx:train:0/sampler_weights/final"

python evaluate_cvdp.py \
    --model instruction \
    --checkpoint-path "tinker://xxx:train:0/sampler_weights/final" \
    --num-samples 5
```

## Important Notes

1. **Use `sampler_weights`, NOT `weights`**:
   - ✅ Correct: `tinker://xxx:train:0/sampler_weights/final`
   - ❌ Wrong: `tinker://xxx:train:0/weights/final`

2. **Model type must match**:
   - If you trained on `reasoning` dataset, use `--model reasoning`
   - If you trained on `instruction` dataset, use `--model instruction`

3. **Host mode is enabled by default** (no Docker needed)

## Quick Check

To see what checkpoints you have:
```bash
ls -lht /matx/u/ethanboneh/baselines_data/checkpoints/ | head -10
```

