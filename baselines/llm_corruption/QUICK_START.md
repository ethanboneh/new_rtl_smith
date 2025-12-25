# Quick Start Guide

## Installation

1. **Install Verilator**:
   ```bash
   # Ubuntu/Debian
   sudo apt-get install verilator
   
   # macOS
   brew install verilator
   
   # Verify
   verilator --version
   ```

2. **Install Python dependencies**:
   ```bash
   pip install openai pyyaml
   ```

3. **Set API key**:
   ```bash
   export OPENAI_API_KEY="your-api-key-here"
   ```

## Quick Test

Test the linter:
```bash
python scripts/test_linter.py
```

## Generate Corruptions

**From HuggingFace (recommended):**
```bash
python scripts/generate_corruptions.py \
    --dataset-type reasoning \
    --output outputs/corruptions.jsonl \
    --model o3-mini \
    --max-entries 10
```

**Skip linter (faster, for testing):**
```bash
python scripts/generate_corruptions.py \
    --dataset-type instruction \
    --output outputs/corruptions.jsonl \
    --model o3-mini \
    --skip-linter \
    --max-entries 10
```

**From local JSONL file:**
```bash
python scripts/generate_corruptions.py \
    --input /path/to/verithoughts.jsonl \
    --output outputs/corruptions.jsonl \
    --model o3-mini \
    --max-entries 10
```

## Output

Check the results:
```bash
# View first entry
head -n 1 outputs/corruptions.jsonl | jq '.'

# Check lint violations
head -n 1 outputs/corruptions.jsonl | jq '.lint_result.new_violated_rules'
```

## Troubleshooting

- **Verilator not found**: Install Verilator (see above)
- **API errors**: Check your `OPENAI_API_KEY`
- **No violations**: Try increasing `--max-retries` or check prompt customization

