"""
Configuration constants for baseline experiments.
"""

import os

# ==============================================================================
# Paths
# ==============================================================================

# Data storage (large storage area)
DATA_DIR = "/matx/u/ethanboneh/baselines_data"
CHECKPOINTS_DIR = os.path.join(DATA_DIR, "checkpoints")
DATASETS_DIR = os.path.join(DATA_DIR, "datasets")
RESULTS_DIR = os.path.join(DATA_DIR, "results")

# Project paths
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
CVDP_BENCHMARK_PATH = os.path.join(PROJECT_ROOT, "..", "cvdp_benchmark")
TINKER_COOKBOOK_PATH = os.path.join(PROJECT_ROOT, "..", "tinker-cookbook")

# CVDP dataset
CVDP_DATASET_PATH = os.path.join(
    CVDP_BENCHMARK_PATH, 
    "example_dataset", 
    "cvdp_v1.0.2_nonagentic_code_generation_no_commercial.jsonl"
)

# RTLLM dataset (local)
RTLLM_PATH = "/sailhome/ethanboneh/RTLLM"

# Corruption datasets (generated)
CORRUPTION_DATASETS = {
    "rtllm_from_spec": os.path.join(PROJECT_ROOT, "llm_corruption", "outputs", "rtllm_from_spec_test.jsonl"),
    "rtllm_llm_corruptions": os.path.join(PROJECT_ROOT, "llm_corruption", "outputs", "rtllm_llm_corruptions.jsonl"),
    "procedural_corruptions": os.path.join(PROJECT_ROOT, "procedural_modification", "outputs", "procedural_corruptions.jsonl"),
}

# ==============================================================================
# Model Configuration
# ==============================================================================

MODEL_NAME = "Qwen/Qwen3-8B"
RENDERER_NAME = "qwen3"

# ==============================================================================
# Hugging Face Datasets
# ==============================================================================

HF_DATASETS = {
    "reasoning": "wilyub/VeriThoughtsTrainSetConsistentReasoning",
    "instruction": "wilyub/VeriThoughtsTrainSetConsistentInstruction",
}

# Local dataset types (not from HuggingFace)
LOCAL_DATASETS = {
    "rtllm": "rtllm",  # Pure RTLLM spec->code
    "rtllm_corruption": "rtllm_corruption",  # RTLLM + corruption data for debugging training
}

# ==============================================================================
# Training Hyperparameters
# ==============================================================================

DEFAULT_TRAINING_CONFIG = {
    "learning_rate": 5e-4,
    "lr_schedule": "linear",
    "num_epochs": 3,
    "lora_rank": 64,
    "batch_size": 64,
    "max_length": 16384,
    "save_every": 100,
    "eval_every": 50,
    "infrequent_eval_every": 500,
}

# ==============================================================================
# Subsampling Configuration
# ==============================================================================

# Fractions of data to use for subsampling experiments
SUBSAMPLE_FRACTIONS = [0.1, 0.25, 0.5, 0.75, 1.0]

# ==============================================================================
# WandB Configuration
# ==============================================================================

WANDB_PROJECT = "rtl-smith-baselines"
WANDB_ENTITY = None  # Set if using a team/organization

# ==============================================================================
# Prompt Templates
# ==============================================================================
# NOTE: The CVDP benchmark uses a different system prompt format.
# The system prompt in CVDP is generated dynamically based on category:
#   - Base: "You are a helpful assistant..." with folder structure info
#   - Category-specific guidance is added (e.g., for cid003: "Specification to RTL Translation")
#
# For training, we DO NOT include:
#   - Custom system prompts (like "You are a Verilog RTL designer...")
#   - Coding rules suffix  
#   - [BEGIN]/[DONE] markers
#
# The VeriThoughts datasets use CODE BEGIN/CODE END markers which we preserve.

# CVDP system prompt (from model_helpers.py)
CVDP_SYSTEM_PROMPT = """You are a helpful assistance.
Consider that you have a folder structure like the following:

    - rtl/*   : Contains files which are RTL code.
    - verif/* : Contains files which are used to verify the correctness of the RTL code.
    - docs/*  : Contains files used to document the project, like Block Guides, RTL Plans and Verification Plans.

When generating files, return the file name in the correct place at the folder structure.
"""

# Category-specific guidance (from model_helpers.py)
CVDP_CATEGORY_GUIDANCE = {
    2: "You are solving an 'RTL Code Completion' problem. To solve this problem correctly, you should only respond with the RTL code generated according to the requirements.",
    3: "You are solving a 'Specification to RTL Translation' problem. To solve this problem correctly, you should only respond with the RTL code translated from the specification.",
    4: "You are solving an 'RTL Code Modification' problem. To solve this problem correctly, you should only respond with the modified RTL code according to the requirements.",
    5: "You are solving a 'Specification to RTL Translation: Module Instantiation and Component Reuse' problem. To solve this problem correctly, you should only respond with the RTL code translated from the specification and with proper module instantiation and component reuse.",
    7: "You are solving an 'RTL Lint Improvement or Power-Performance Optimization' problem. To solve this problem correctly, you should only respond with improved RTL code to address lint issues or optimize for power/performance.",
    16: "You are solving an 'RTL Debugging and Bug Fixing' problem. To solve this problem correctly, you should only respond with the RTL code that is debugged and fixed to address the bug."
}

# For VeriThoughts datasets, the code is wrapped in these markers
VERITHOUGHTS_CODE_PREFIX = "CODE BEGIN"
VERITHOUGHTS_CODE_SUFFIX = "CODE END"


def create_directories():
    """Create necessary directories for data storage."""
    for dir_path in [DATA_DIR, CHECKPOINTS_DIR, DATASETS_DIR, RESULTS_DIR]:
        os.makedirs(dir_path, exist_ok=True)
    print(f"Created directories in {DATA_DIR}")


if __name__ == "__main__":
    create_directories()
