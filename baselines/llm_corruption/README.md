# LLM-Generated Procedural Modifications

This directory contains tools for generating procedural modifications to VeriThoughts data using language models. The system corrupts clean SystemVerilog code, verifies the corruptions with a linter, and generates issue descriptions and reasoning traces.

## Overview

The pipeline performs the following steps:

1. **Corruption Generation**: Uses an LLM (e.g., o3-mini) to corrupt clean VeriThoughts code
2. **Linter Verification**: Runs Verilator to verify that the corruption introduces new lint violations
3. **Issue Description**: Generates a detailed issue description for the bug
4. **Reasoning Trace**: Generates a step-by-step reasoning trace for fixing the bug

## Installation

### Prerequisites

1. **Verilator** (Required for linting)

   Ubuntu/Debian:
   ```bash
   sudo apt-get update
   sudo apt-get install verilator
   ```

   macOS:
   ```bash
   brew install verilator
   ```

   Or build from source:
   ```bash
   git clone https://github.com/verilator/verilator
   cd verilator
   autoconf
   ./configure
   make
   sudo make install
   ```

   Verify installation:
   ```bash
   verilator --version
   ```

2. **Python Dependencies**

   ```bash
   pip install openai pyyaml datasets
   ```

   Or install from requirements:
   ```bash
   pip install -r requirements.txt
   ```

3. **OpenAI API Key**

   Set your API key:
   ```bash
   export OPENAI_API_KEY="your-api-key-here"
   ```

   Or pass it via command line argument.

## Directory Structure

```
llm_corruption/
├── prompts/                    # YAML prompt templates
│   ├── corruption_prompt.yaml      # Prompt for corrupting code
│   ├── issue_description_prompt.yaml   # Prompt for issue descriptions
│   └── reasoning_trace_prompt.yaml    # Prompt for reasoning traces
├── scripts/                    # Python scripts
│   ├── linter.py                  # Verilator linter integration
│   ├── llm_client.py              # LLM client for API calls
│   └── generate_corruptions.py    # Main pipeline script
├── outputs/                    # Generated corruptions (created automatically)
├── requirements.txt            # Python dependencies
└── README.md                   # This file
```

## Usage

### Basic Usage

**Load from HuggingFace (recommended):**
```bash
python scripts/generate_corruptions.py \
    --dataset-type reasoning \
    --output outputs/corruptions.jsonl \
    --model o3-mini
```

**Or load from local JSONL file:**
```bash
python scripts/generate_corruptions.py \
    --input /path/to/verithoughts.jsonl \
    --output outputs/corruptions.jsonl \
    --model o3-mini
```

### Advanced Options

```bash
python scripts/generate_corruptions.py \
    --dataset-type reasoning \
    --output outputs/corruptions.jsonl \
    --model o3-mini \
    --max-entries 100 \
    --max-retries 5 \
    --require-lint \
    --api-key "your-key" \
    --verilator-bin /usr/local/bin/verilator
```

**Skip linter verification (faster, for testing):**
```bash
python scripts/generate_corruptions.py \
    --dataset-type instruction \
    --output outputs/corruptions.jsonl \
    --model o3-mini \
    --skip-linter \
    --max-entries 10
```

### Arguments

- `--input`: Path to VeriThoughts JSONL file (optional, use with local files)
- `--dataset-type`: Load from HuggingFace ("reasoning" or "instruction")
- `--split`: Dataset split (default: "train")
- `--output`: Output JSONL file path (required)
- `--model`: LLM model name (default: "o3-mini")
- `--api-key`: OpenAI API key (default: from OPENAI_API_KEY env var)
- `--base-url`: API base URL for custom endpoints
- `--max-entries`: Maximum number of entries to process
- `--max-retries`: Maximum retries per entry if lint verification fails (default: 3)
- `--require-lint`: Require new lint violations (default: True)
- `--no-require-lint`: Don't require new lint violations
- `--skip-linter`: Skip linter verification entirely (faster, but no lint validation)
- `--verilator-bin`: Path to verilator binary (default: "verilator" from PATH)

## Output Format

Each entry in the output JSONL file contains:

```json
{
  "original_entry": {...},           // Original VeriThoughts entry
  "clean_code": "...",                // Original clean SystemVerilog code
  "corrupted_code": "...",            // Corrupted code
  "corruption_explanation": "...",    // LLM explanation of the corruption
  "issue_description": "...",         // Generated issue description
  "reasoning_trace": "...",           // Generated reasoning trace
  "lint_result": {
    "new_violated_rules": ["LATCH", "BLKSEQ"],  // New lint rules violated
    "all_corrupt_rules": ["LATCH", "BLKSEQ", "WIDTH"],
    "clean_has_errors": false,
    "corrupt_has_errors": true
  },
  "timestamp": "2024-01-01T12:00:00"
}
```

## Linter Integration

The system uses Verilator to verify corruptions. It checks:

- **New Violations**: Ensures corrupt code introduces new lint violations
- **Rule Diversity**: Tracks which lint rules are violated (LATCH, BLKSEQ, WIDTH, etc.)
- **Error Detection**: Verifies that violations are detected

### Common Verilator Warning Codes

- `LATCH`: Latch inference (incomplete conditionals)
- `BLKSEQ`: Blocking assignment in sequential block
- `WIDTH`: Width mismatch
- `MULTIDRIVEN`: Multiple drivers for same signal
- `UNOPTFLAT`: Unoptimizable combinational loop
- `UNSIGNED`: Unsigned comparison warning
- `CASEINCOMPLETE`: Incomplete case statement

## Customization

### Modifying Prompts

Edit the YAML files in `prompts/` to customize the prompts:

- `corruption_prompt.yaml`: Modify bug types and guidance
- `issue_description_prompt.yaml`: Adjust issue description format
- `reasoning_trace_prompt.yaml`: Change reasoning trace structure

### Adding New Linters

To add support for other linters, modify `scripts/linter.py` to add new linter classes following the same interface.

## Troubleshooting

### Verilator Not Found

```
RuntimeError: Verilator not found. Please install Verilator...
```

Solution: Install Verilator (see Installation section) or provide path via `--verilator-bin`.

### API Key Issues

```
ValueError: API key not provided...
```

Solution: Set `OPENAI_API_KEY` environment variable or use `--api-key`.

### No Lint Violations

If corruptions don't introduce lint violations:

1. Check that the prompt emphasizes lint-focused bugs
2. Try increasing `--max-retries`
3. Use `--no-require-lint` to allow corruptions without violations (not recommended)

### Code Extraction Failures

If code cannot be extracted from VeriThoughts entries, modify `extract_code_from_entry()` in `generate_corruptions.py` to match your data format.

## Example Workflow

1. **Run Generation** (loads directly from HuggingFace):
   ```bash
   python scripts/generate_corruptions.py \
       --dataset-type reasoning \
       --output outputs/corruptions.jsonl \
       --max-entries 50
   ```

   Or use a local JSONL file:
   ```bash
   python scripts/generate_corruptions.py \
       --input data/verithoughts_train.jsonl \
       --output outputs/corruptions.jsonl \
       --max-entries 50
   ```

3. **Verify Results**:
   ```bash
   # Check output
   head -n 1 outputs/corruptions.jsonl | jq '.lint_result'
   ```

4. **Use in Training**: The output can be used for fine-tuning models on bug fixing tasks.

## Notes

- The system focuses on generating non-trivial bugs that violate fundamental RTL design principles
- Each corruption is verified to ensure it introduces new lint violations
- The reasoning traces are designed to be useful for training models
- Prompts can be customized to target specific types of bugs or lint rules

## Recommended Linter: Verilator

**Verilator** is recommended because:

1. **Free and Open Source**: No licensing costs
2. **Widely Used**: Industry-standard tool
3. **Comprehensive Rules**: Detects many RTL design issues
4. **Easy Integration**: Simple command-line interface
5. **Active Development**: Regularly updated

Alternative linters (if needed):
- **Verible** (Google): Open source, good for style checks
- **Spyglass** (Synopsys): Commercial, very comprehensive
- **JasperGold** (Cadence): Commercial, formal verification focus

