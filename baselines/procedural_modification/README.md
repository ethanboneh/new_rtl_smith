# Procedural Modification for Verilog

This directory contains tools for generating procedural modifications to Verilog/SystemVerilog code, following the SWE-smith approach adapted for RTL design.

## Overview

The procedural modification approach creates bugs by applying AST-level transformations to Verilog code. Unlike LLM-based corruption, this approach is **zero-cost** and **deterministic**, making it suitable for generating large amounts of training data.

## How It Works

The pipeline follows the SWE-smith methodology:

1. **Parse**: Convert Verilog source code into a structured AST representation
2. **Filter**: Apply criteria to identify code suitable for specific modifications
3. **Modify**: Apply a procedural transformation (e.g., remove else branch, swap operators)
4. **Generate**: Convert the modified AST back to source code

```
┌──────────────────┐     ┌──────────────────┐     ┌──────────────────┐
│  Verilog Code    │────▶│   Parse AST      │────▶│  Apply Filters   │
│  (VeriThoughts)  │     │  (verilog_ast)   │     │  (filters.py)    │
└──────────────────┘     └──────────────────┘     └──────────────────┘
                                                          │
                         ┌──────────────────┐     ┌───────▼──────────┐
                         │   Buggy Code     │◀────│ Apply Modification│
                         │   + Metadata     │     │ (modifications.py)│
                         └──────────────────┘     └──────────────────┘
```

## Installation

```bash
pip install pyverilog jinja2 datasets tqdm
```

Or install from requirements:

```bash
pip install -r requirements.txt
```

## Quick Start

### Generate Procedural Modifications

```bash
# From HuggingFace dataset
python scripts/generate_modifications.py \
    --dataset-type instruction \
    --output outputs/procedural_corruptions.jsonl \
    --max-entries 100

# From local JSONL file
python scripts/generate_modifications.py \
    --input /path/to/verithoughts.jsonl \
    --output outputs/procedural_corruptions.jsonl \
    --max-entries 100
```

### List Available Modifications

```bash
python scripts/generate_modifications.py --list-modifications
```

### Apply Specific Modification

```bash
python scripts/generate_modifications.py \
    --dataset-type instruction \
    --output outputs/operator_bugs.jsonl \
    --modification change_operator \
    --max-entries 100
```

### Run Tests

```bash
python tests/test_modifications.py
```

## Available Modifications

### Control Flow

| Modification | Description | Filter Requirements |
|-------------|-------------|---------------------|
| `invert_if_else` | Swap if and else bodies | has_module, has_if_else |
| `remove_else_branch` | Remove else branch (causes latch) | has_module, has_if_else |
| `remove_case_default` | Remove default case | has_module, has_case_statements |

### Expressions

| Modification | Description | Filter Requirements |
|-------------|-------------|---------------------|
| `change_constant` | Change numeric constant by ±1 | has_module, has_numeric_constants |
| `change_operator` | Change operator (e.g., + to -) | has_module, has_operators |
| `swap_operands` | Swap operands of binary operation | has_module, has_operators |
| `change_bit_width` | Modify bit width specification | has_module, has_width_specs |

### Removal

| Modification | Description | Filter Requirements |
|-------------|-------------|---------------------|
| `remove_assignment` | Remove an assignment statement | has_module, has_assignments |
| `remove_conditional` | Remove an if block | has_module, has_conditionals |
| `remove_loop` | Remove a loop block | has_module, has_loops |

### RTL-Specific

| Modification | Description | Filter Requirements |
|-------------|-------------|---------------------|
| `swap_blocking_nonblocking` | Swap = and <= | has_module, has_assignments |
| `change_sensitivity_list` | Modify always sensitivity | has_module, has_always_blocks |
| `add_multiple_driver` | Add conflicting driver | has_module, has_continuous_assignments |
| `remove_reset_assignment` | Remove reset initialization | has_module, has_sequential_logic |

## Filter Criteria

| Index | Name | Description |
|-------|------|-------------|
| 1 | filter_has_module | Is this a valid module definition |
| 2 | filter_has_always_blocks | Does the module contain always blocks |
| 3 | filter_has_sequential_logic | Does the module have always_ff/posedge |
| 4 | filter_has_combinational_logic | Does the module have always_comb/@* |
| 5 | filter_has_loops | Does the module contain for/while loops |
| 6 | filter_has_conditionals | Does the module contain if statements |
| 7 | filter_has_case_statements | Does the module contain case statements |
| 8 | filter_has_assignments | Does the module contain assignments |
| 9 | filter_has_blocking_assignments | Does the module contain = assignments |
| 10 | filter_has_nonblocking_assignments | Does the module contain <= assignments |
| 11 | filter_has_continuous_assignments | Does the module contain assign statements |
| 12 | filter_has_if_else | Does the module contain if-else blocks |
| 13 | filter_has_operators | Does the module contain binary operators |
| 14 | filter_has_arithmetic_operators | Does the module contain +,-,*,/,% |
| 15 | filter_has_logical_operators | Does the module contain &&, \|\| |
| 16 | filter_has_bitwise_operators | Does the module contain &,\|,^,<<,>> |
| 17 | filter_has_comparisons | Does the module contain ==, !=, <, > |
| 18 | filter_has_module_instances | Does the module instantiate other modules |
| 19 | filter_min_complexity | Is the module above minimum complexity |
| 20 | filter_max_complexity | Is the module below maximum complexity |
| 21 | filter_min_lines | Does the module have minimum lines |
| 22 | filter_max_lines | Is the module below maximum lines |
| 23 | filter_has_numeric_constants | Does the module contain numeric literals |
| 24 | filter_has_width_specs | Does the module contain [n:m] widths |
| 25 | filter_has_generate_blocks | Does the module contain generate blocks |

## Output Format

Each entry in the output JSONL file contains:

```json
{
  "original_entry": {...},
  "clean_code": "...",
  "corrupted_code": "...",
  "corruption_explanation": "Swapped if and else bodies",
  "modification_type": "invert_if_else",
  "modification_line": 15,
  "original_snippet": "if (!rst) begin...",
  "modified_snippet": "if (!rst) begin...",
  "issue_description": "The buggy code has swapped if and else bodies...",
  "reasoning_trace": "1. Overall Design Intent:...",
  "complexity": 8,
  "applicable_modifications": ["invert_if_else", "change_constant", ...],
  "timestamp": "2024-01-01T12:00:00"
}
```

## Integration with Training Pipeline

The output format is compatible with the `llm_corruption` pipeline. You can use the same `prepare_corruption_datasets.py` script:

```bash
# Prepare for training (with reasoning)
python ../prepare_corruption_datasets.py \
    --corruption-file procedural_modification/outputs/procedural_corruptions.jsonl \
    --include-reasoning

# Mix with VeriThoughts
python ../mix_datasets.py \
    --verithoughts reasoning \
    --corruption corruption_reasoning \
    --ratio 0.3 \
    --output /matx/u/ethanboneh/baselines_data/datasets/mixed_with_procedural.jsonl
```

## Advantages Over LLM-Based Corruption

| Aspect | Procedural | LLM-Based |
|--------|-----------|-----------|
| Cost | Zero (no API calls) | API costs |
| Speed | Very fast | Slow (API latency) |
| Determinism | Deterministic | Stochastic |
| Scalability | Highly scalable | Limited by budget |
| Bug types | Predefined patterns | Creative/diverse |
| Lint compliance | Controllable | May not trigger lints |

## Comparison to SWE-smith

This implementation adapts SWE-smith's approach for Verilog:

| SWE-smith (Python) | This Implementation (Verilog) |
|-------------------|-------------------------------|
| Function/Class AST | Module/Always block AST |
| Remove loops | Remove loops |
| Invert if/else | Invert if/else |
| Change operators | Change operators |
| Swap operands | Swap operands |
| Remove assignments | Remove assignments |
| - | Swap blocking/non-blocking |
| - | Change sensitivity list |
| - | Add multiple drivers |
| - | Remove reset assignments |

## Directory Structure

```
procedural_modification/
├── scripts/
│   ├── verilog_ast.py          # Verilog parser and AST utilities
│   ├── filters.py              # Filtering criteria
│   ├── modifications.py        # Procedural modification functions
│   └── generate_modifications.py  # Main pipeline script
├── tests/
│   └── test_modifications.py   # Unit tests
├── outputs/                    # Generated corruptions
├── requirements.txt            # Dependencies
└── README.md                   # This file
```

## Known Limitations

1. **Regex-based parsing**: Uses regex patterns rather than a full parser, which may miss some edge cases
2. **SystemVerilog support**: Limited support for advanced SystemVerilog constructs
3. **Semantic validity**: Modifications may create syntactically invalid code in some cases
4. **No lint verification**: Unlike llm_corruption, does not verify modifications trigger lint errors

## Contributing

To add a new modification:

1. Add the modification function in `modifications.py`
2. Add any new filter criteria in `filters.py`
3. Register the modification in `MODIFICATION_REGISTRY`
4. Add issue description template in `generate_modifications.py`
5. Add tests in `tests/test_modifications.py`

