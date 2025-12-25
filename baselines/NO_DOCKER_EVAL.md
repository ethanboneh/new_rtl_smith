# Running Evaluation WITHOUT Docker

I've modified the CVDP benchmark code to support host mode (running without Docker). 

## Changes Made

1. **`cvdp_benchmark/src/repository.py`**: Added `log_run_host()` method that runs tests directly on host
2. **`cvdp_benchmark/src/argparse_common.py`**: Uncommented the `--host` flag
3. **`baselines/evaluate_cvdp.py`**: Automatically adds `-o` flag to enable host mode

## Command to Run Base Qwen3-8B Evaluation

```bash
cd /afs/cs.stanford.edu/u/ethanboneh/new_rtl_smith/baselines

# Pass@5 evaluation (recommended)
python evaluate_cvdp.py --model base --num-samples 5

# Single sample (faster)
python evaluate_cvdp.py --model base
```

## What This Does

- Uses your **locally installed tools** (iverilog, cocotb, pytest)
- Runs tests **directly on host** (no Docker containers)
- Evaluates **base Qwen3-8B** on the **non-commercial CVDP dataset**
- Results saved to `/matx/u/ethanboneh/baselines_data/results/cvdp_base_<timestamp>/`

## Requirements

✅ **iverilog** - Installed in your environment  
✅ **cocotb** - Installed in your environment  
✅ **pytest** - Installed in your environment  
✅ **Tinker API key** - Set as `TINKER_API_KEY`

## How It Works

The host mode:
1. Reads the `.env` file from the harness
2. Translates Docker paths (`/code/rtl/`) to host paths (`<repo>/rtl/`)
3. Sets up environment variables
4. Runs `pytest` directly on the test runner
5. Uses your local iverilog and cocotb installations

No Docker required! 🎉

