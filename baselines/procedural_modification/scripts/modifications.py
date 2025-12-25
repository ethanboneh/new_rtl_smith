#!/usr/bin/env python3
"""
Procedural Modifications for Verilog/SystemVerilog code.

This module implements procedural modifications following the SWE-smith approach,
adapted for Verilog RTL code. Each modification takes a VerilogModule and applies
a fixed transformation to introduce bugs.

Modifications are categorized into:
- Control Flow: Modifications to conditionals, case statements
- Expressions: Changes to operators, constants, operands
- Removal: Removal of assignments, loops, conditionals
- RTL-specific: Blocking/non-blocking swaps, sensitivity list issues
"""

import random
import re
from dataclasses import dataclass, field
from typing import Callable, List, Optional, Tuple, Dict, Any
from enum import Enum, auto

from verilog_ast import VerilogModule, VerilogParser, ASTNode, NodeType


class ModificationCategory(Enum):
    """Categories of procedural modifications."""
    CONTROL_FLOW = auto()
    EXPRESSIONS = auto()
    REMOVAL = auto()
    RTL_SPECIFIC = auto()
    FSM = auto()
    CLOCK_RESET = auto()
    HIERARCHY = auto()
    TIMING = auto()


@dataclass
class ModificationResult:
    """Result of applying a procedural modification."""
    success: bool
    modified_code: str
    modification_type: str
    description: str
    original_snippet: str = ""
    modified_snippet: str = ""
    line_number: int = 0


@dataclass
class ProceduralModification:
    """Represents a procedural modification with its metadata."""
    name: str
    category: ModificationCategory
    description: str
    required_filter_indices: List[int]
    apply_fn: Callable[[VerilogModule, float], ModificationResult]
    likelihood: float = 0.5  # Default probability of applying per candidate


# ============================================================================
# Control Flow Modifications
# ============================================================================

def invert_if_else(module: VerilogModule, likelihood: float = 0.5) -> ModificationResult:
    """
    Invert the if-else bodies of a conditional.
    
    Example:
        if (cond) begin
            a <= 1;
        end else begin
            a <= 0;
        end
    
    Becomes:
        if (cond) begin
            a <= 0;  // Was in else
        end else begin
            a <= 1;  // Was in if
        end
    
    This is equivalent to SWE-smith's "Invert If/Else" modification.
    """
    # Find if-else blocks
    if_else_pattern = re.compile(
        r'(\bif\s*\([^)]+\)\s*begin)(.*?)(\bend\s*else\s*begin)(.*?)(\bend\b)',
        re.DOTALL
    )
    
    matches = list(if_else_pattern.finditer(module.source_code))
    if not matches:
        return ModificationResult(
            success=False,
            modified_code=module.source_code,
            modification_type="invert_if_else",
            description="No if-else blocks found"
        )
    
    # Select a random if-else block based on likelihood
    candidates = [m for m in matches if random.random() < likelihood]
    if not candidates:
        candidates = [random.choice(matches)]
    
    match = random.choice(candidates)
    
    if_header = match.group(1)
    if_body = match.group(2)
    else_header = match.group(3)
    else_body = match.group(4)
    end_keyword = match.group(5)
    
    # Swap the bodies
    original = match.group(0)
    modified = f"{if_header}{else_body}{else_header}{if_body}{end_keyword}"
    
    new_code = module.source_code[:match.start()] + modified + module.source_code[match.end():]
    
    return ModificationResult(
        success=True,
        modified_code=new_code,
        modification_type="invert_if_else",
        description="Swapped if and else bodies",
        original_snippet=original[:200],
        modified_snippet=modified[:200],
        line_number=module.source_code[:match.start()].count('\n') + 1
    )


def remove_else_branch(module: VerilogModule, likelihood: float = 0.5) -> ModificationResult:
    """
    Remove the else branch from an if-else statement.
    
    This can cause latch inference in combinational logic or incorrect behavior.
    """
    # Find if-else blocks
    if_else_pattern = re.compile(
        r'(\bif\s*\([^)]+\)\s*begin.*?\bend)\s*(else\s*begin.*?\bend)',
        re.DOTALL
    )
    
    matches = list(if_else_pattern.finditer(module.source_code))
    if not matches:
        return ModificationResult(
            success=False,
            modified_code=module.source_code,
            modification_type="remove_else_branch",
            description="No if-else blocks found"
        )
    
    candidates = [m for m in matches if random.random() < likelihood]
    if not candidates:
        candidates = [random.choice(matches)]
    
    match = random.choice(candidates)
    
    # Keep only the if part, remove else
    original = match.group(0)
    modified = match.group(1)
    
    new_code = module.source_code[:match.start()] + modified + module.source_code[match.end():]
    
    return ModificationResult(
        success=True,
        modified_code=new_code,
        modification_type="remove_else_branch",
        description="Removed else branch (may cause latch inference)",
        original_snippet=original[:200],
        modified_snippet=modified[:200],
        line_number=module.source_code[:match.start()].count('\n') + 1
    )


def remove_case_default(module: VerilogModule, likelihood: float = 0.5) -> ModificationResult:
    """
    Remove the default case from a case statement.
    
    This can cause incomplete case statement warnings and latch inference.
    """
    # Find case statements with default
    case_pattern = re.compile(
        r'(\bcase[xz]?\s*\([^)]+\).*?)(default\s*:.*?)(\bendcase\b)',
        re.DOTALL
    )
    
    matches = list(case_pattern.finditer(module.source_code))
    if not matches:
        return ModificationResult(
            success=False,
            modified_code=module.source_code,
            modification_type="remove_case_default",
            description="No case statements with default found"
        )
    
    candidates = [m for m in matches if random.random() < likelihood]
    if not candidates:
        candidates = [random.choice(matches)]
    
    match = random.choice(candidates)
    
    original = match.group(0)
    # Remove the default case
    modified = match.group(1) + match.group(3)
    
    new_code = module.source_code[:match.start()] + modified + module.source_code[match.end():]
    
    return ModificationResult(
        success=True,
        modified_code=new_code,
        modification_type="remove_case_default",
        description="Removed default case (may cause incomplete case warning)",
        original_snippet=original[:200],
        modified_snippet=modified[:200],
        line_number=module.source_code[:match.start()].count('\n') + 1
    )


# ============================================================================
# Expression Modifications
# ============================================================================

def change_constant(module: VerilogModule, likelihood: float = 0.5) -> ModificationResult:
    """
    Change a numeric constant by ±1.
    
    Example: 8'b0 becomes 8'b1, or counter + 1 becomes counter + 2
    
    This is equivalent to SWE-smith's "Change Constants" modification.
    """
    # Find numeric constants
    patterns = [
        # Sized binary/hex/decimal literals
        (re.compile(r"(\d+)'([bBhHdDoO])([0-9a-fA-F_]+)"), 'sized'),
        # Plain decimal numbers (not in bit widths)
        (re.compile(r"(?<!['\[])(\b\d+\b)(?!['\]:])"), 'decimal'),
    ]
    
    all_matches = []
    for pattern, pattern_type in patterns:
        for match in pattern.finditer(module.source_code):
            all_matches.append((match, pattern_type))
    
    if not all_matches:
        return ModificationResult(
            success=False,
            modified_code=module.source_code,
            modification_type="change_constant",
            description="No numeric constants found"
        )
    
    candidates = [m for m in all_matches if random.random() < likelihood]
    if not candidates:
        candidates = [random.choice(all_matches)]
    
    match, pattern_type = random.choice(candidates)
    
    original = match.group(0)
    
    if pattern_type == 'sized':
        # For sized literals, modify the value part
        width = match.group(1)
        base = match.group(2)
        value = match.group(3)
        
        # Parse and modify the value
        if base.lower() == 'b':
            # Binary - flip a bit
            if '1' in value:
                new_value = value.replace('1', '0', 1)
            else:
                new_value = value[:-1] + '1' if value else '1'
        elif base.lower() == 'h':
            # Hex - increment/decrement
            try:
                int_val = int(value.replace('_', ''), 16)
                new_int = int_val + random.choice([-1, 1])
                new_value = format(max(0, new_int), 'x')
            except ValueError:
                new_value = value
        else:
            # Decimal
            try:
                int_val = int(value.replace('_', ''))
                new_int = int_val + random.choice([-1, 1])
                new_value = str(max(0, new_int))
            except ValueError:
                new_value = value
        
        modified = f"{width}'{base}{new_value}"
    else:
        # Plain decimal
        try:
            int_val = int(original)
            new_int = int_val + random.choice([-1, 1])
            modified = str(max(0, new_int))
        except ValueError:
            modified = original
    
    new_code = module.source_code[:match.start()] + modified + module.source_code[match.end():]
    
    return ModificationResult(
        success=True,
        modified_code=new_code,
        modification_type="change_constant",
        description=f"Changed constant from {original} to {modified}",
        original_snippet=original,
        modified_snippet=modified,
        line_number=module.source_code[:match.start()].count('\n') + 1
    )


def change_operator(module: VerilogModule, likelihood: float = 0.5) -> ModificationResult:
    """
    Change an operator to a different one.
    
    Examples:
        + -> -
        && -> ||
        == -> !=
        < -> >
    
    This is equivalent to SWE-smith's "Change Operator" modification.
    """
    operator_swaps = {
        '+': '-',
        '-': '+',
        '*': '/',
        '/': '*',
        '&': '|',
        '|': '&',
        '^': '&',
        '&&': '||',
        '||': '&&',
        '==': '!=',
        '!=': '==',
        '<': '>',
        '>': '<',
        '<=': '>=',  # Careful: not non-blocking!
        '>=': '<=',
        '<<': '>>',
        '>>': '<<',
    }
    
    # Find operators in expressions (not in always blocks for <= vs >=)
    all_matches = []
    for op in operator_swaps.keys():
        # Escape special regex characters
        escaped_op = re.escape(op)
        
        # For <= and >=, we need to be careful not to match non-blocking assignments
        if op in ('<=', '>='):
            # Only match in comparison contexts (after a space or word char, before space)
            pattern = re.compile(rf'(\w)\s*{escaped_op}\s*(\w)')
        else:
            pattern = re.compile(rf'\s{escaped_op}\s')
        
        for match in pattern.finditer(module.source_code):
            all_matches.append((match, op))
    
    if not all_matches:
        return ModificationResult(
            success=False,
            modified_code=module.source_code,
            modification_type="change_operator",
            description="No suitable operators found"
        )
    
    candidates = [m for m in all_matches if random.random() < likelihood]
    if not candidates:
        candidates = [random.choice(all_matches)]
    
    match, op = random.choice(candidates)
    new_op = operator_swaps[op]
    
    original = match.group(0)
    modified = original.replace(op, new_op)
    
    new_code = module.source_code[:match.start()] + modified + module.source_code[match.end():]
    
    return ModificationResult(
        success=True,
        modified_code=new_code,
        modification_type="change_operator",
        description=f"Changed operator from '{op}' to '{new_op}'",
        original_snippet=original.strip(),
        modified_snippet=modified.strip(),
        line_number=module.source_code[:match.start()].count('\n') + 1
    )


def swap_operands(module: VerilogModule, likelihood: float = 0.5) -> ModificationResult:
    """
    Swap the operands of a binary operation.
    
    Example: a - b becomes b - a
    
    This is equivalent to SWE-smith's "Swap Operands" modification.
    """
    # Find binary operations with non-commutative operators
    pattern = re.compile(r'(\w+)\s*(-|/|%|<<|>>|<|>|<=|>=)\s*(\w+)')
    
    matches = list(pattern.finditer(module.source_code))
    if not matches:
        return ModificationResult(
            success=False,
            modified_code=module.source_code,
            modification_type="swap_operands",
            description="No suitable binary operations found"
        )
    
    candidates = [m for m in matches if random.random() < likelihood]
    if not candidates:
        candidates = [random.choice(matches)]
    
    match = random.choice(candidates)
    
    left = match.group(1)
    op = match.group(2)
    right = match.group(3)
    
    original = match.group(0)
    modified = f"{right} {op} {left}"
    
    new_code = module.source_code[:match.start()] + modified + module.source_code[match.end():]
    
    return ModificationResult(
        success=True,
        modified_code=new_code,
        modification_type="swap_operands",
        description=f"Swapped operands: '{left} {op} {right}' -> '{right} {op} {left}'",
        original_snippet=original,
        modified_snippet=modified,
        line_number=module.source_code[:match.start()].count('\n') + 1
    )


def change_bit_width(module: VerilogModule, likelihood: float = 0.5) -> ModificationResult:
    """
    Change a bit width specification.
    
    Example: [7:0] becomes [6:0] or [8:0]
    
    This can cause width mismatch errors.
    """
    pattern = re.compile(r'\[(\d+):(\d+)\]')
    
    matches = list(pattern.finditer(module.source_code))
    if not matches:
        return ModificationResult(
            success=False,
            modified_code=module.source_code,
            modification_type="change_bit_width",
            description="No bit width specifications found"
        )
    
    candidates = [m for m in matches if random.random() < likelihood]
    if not candidates:
        candidates = [random.choice(matches)]
    
    match = random.choice(candidates)
    
    msb = int(match.group(1))
    lsb = int(match.group(2))
    
    # Randomly change MSB or LSB
    if random.random() < 0.5:
        new_msb = msb + random.choice([-1, 1])
        new_lsb = lsb
    else:
        new_msb = msb
        new_lsb = lsb + random.choice([-1, 1])
    
    # Ensure valid range
    new_msb = max(0, new_msb)
    new_lsb = max(0, new_lsb)
    
    original = match.group(0)
    modified = f"[{new_msb}:{new_lsb}]"
    
    new_code = module.source_code[:match.start()] + modified + module.source_code[match.end():]
    
    return ModificationResult(
        success=True,
        modified_code=new_code,
        modification_type="change_bit_width",
        description=f"Changed bit width from {original} to {modified}",
        original_snippet=original,
        modified_snippet=modified,
        line_number=module.source_code[:match.start()].count('\n') + 1
    )


# ============================================================================
# Removal Modifications
# ============================================================================

def remove_assignment(module: VerilogModule, likelihood: float = 0.5) -> ModificationResult:
    """
    Remove an assignment statement.
    
    This can cause signals to be undriven or latches to be inferred.
    
    This is equivalent to SWE-smith's "Remove Assignments" modification.
    """
    # Find assignment statements
    patterns = [
        re.compile(r'^\s*assign\s+\w+.*?;\s*$', re.MULTILINE),
        re.compile(r'^\s*\w+\s*<=\s*[^;]+;\s*$', re.MULTILINE),
        re.compile(r'^\s*\w+\s*=\s*[^;]+;\s*$', re.MULTILINE),
    ]
    
    all_matches = []
    for pattern in patterns:
        all_matches.extend(pattern.finditer(module.source_code))
    
    if not all_matches:
        return ModificationResult(
            success=False,
            modified_code=module.source_code,
            modification_type="remove_assignment",
            description="No assignments found"
        )
    
    candidates = [m for m in all_matches if random.random() < likelihood]
    if not candidates:
        candidates = [random.choice(all_matches)]
    
    match = random.choice(candidates)
    
    original = match.group(0)
    
    # Remove the assignment (replace with empty or comment)
    new_code = module.source_code[:match.start()] + module.source_code[match.end():]
    
    return ModificationResult(
        success=True,
        modified_code=new_code,
        modification_type="remove_assignment",
        description="Removed an assignment statement",
        original_snippet=original.strip(),
        modified_snippet="(removed)",
        line_number=module.source_code[:match.start()].count('\n') + 1
    )


def remove_conditional(module: VerilogModule, likelihood: float = 0.5) -> ModificationResult:
    """
    Remove a conditional (if) block.
    
    This is equivalent to SWE-smith's "Remove Conditionals" modification.
    """
    # Find if blocks (not if-else to avoid complex restructuring)
    pattern = re.compile(
        r'\bif\s*\([^)]+\)\s*begin.*?\bend\b(?!\s*else)',
        re.DOTALL
    )
    
    matches = list(pattern.finditer(module.source_code))
    if not matches:
        return ModificationResult(
            success=False,
            modified_code=module.source_code,
            modification_type="remove_conditional",
            description="No suitable conditionals found"
        )
    
    candidates = [m for m in matches if random.random() < likelihood]
    if not candidates:
        candidates = [random.choice(matches)]
    
    match = random.choice(candidates)
    
    original = match.group(0)
    
    new_code = module.source_code[:match.start()] + module.source_code[match.end():]
    
    return ModificationResult(
        success=True,
        modified_code=new_code,
        modification_type="remove_conditional",
        description="Removed a conditional block",
        original_snippet=original[:200],
        modified_snippet="(removed)",
        line_number=module.source_code[:match.start()].count('\n') + 1
    )


def remove_loop(module: VerilogModule, likelihood: float = 0.5) -> ModificationResult:
    """
    Remove a loop (for/while) block.
    
    This is equivalent to SWE-smith's "Remove Loops" modification.
    """
    pattern = re.compile(
        r'\b(for|while)\s*\([^)]+\)\s*begin.*?\bend\b',
        re.DOTALL
    )
    
    matches = list(pattern.finditer(module.source_code))
    if not matches:
        return ModificationResult(
            success=False,
            modified_code=module.source_code,
            modification_type="remove_loop",
            description="No loops found"
        )
    
    candidates = [m for m in matches if random.random() < likelihood]
    if not candidates:
        candidates = [random.choice(matches)]
    
    match = random.choice(candidates)
    
    original = match.group(0)
    
    new_code = module.source_code[:match.start()] + module.source_code[match.end():]
    
    return ModificationResult(
        success=True,
        modified_code=new_code,
        modification_type="remove_loop",
        description="Removed a loop block",
        original_snippet=original[:200],
        modified_snippet="(removed)",
        line_number=module.source_code[:match.start()].count('\n') + 1
    )


# ============================================================================
# RTL-Specific Modifications
# ============================================================================

def swap_blocking_nonblocking(module: VerilogModule, likelihood: float = 0.5) -> ModificationResult:
    """
    Swap blocking (=) and non-blocking (<=) assignments.
    
    This is a common RTL bug that can cause simulation/synthesis mismatches.
    - Blocking in always_ff -> causes BLKSEQ warning
    - Non-blocking in always_comb -> causes issues
    """
    # Find assignments in always blocks
    # We'll swap = to <= or <= to =
    
    # Find non-blocking assignments
    nb_pattern = re.compile(r'(\w+(?:\s*\[[^\]]+\])?)\s*<=\s*([^;]+);')
    # Find blocking assignments (not continuous assign)
    b_pattern = re.compile(r'(?<!assign\s)(\w+(?:\s*\[[^\]]+\])?)\s*=(?!=)\s*([^;<=]+);')
    
    all_matches = []
    for match in nb_pattern.finditer(module.source_code):
        all_matches.append((match, 'nb_to_b'))
    for match in b_pattern.finditer(module.source_code):
        # Skip if it looks like a continuous assign
        if not module.source_code[:match.start()].rstrip().endswith('assign'):
            all_matches.append((match, 'b_to_nb'))
    
    if not all_matches:
        return ModificationResult(
            success=False,
            modified_code=module.source_code,
            modification_type="swap_blocking_nonblocking",
            description="No suitable assignments found"
        )
    
    candidates = [m for m in all_matches if random.random() < likelihood]
    if not candidates:
        candidates = [random.choice(all_matches)]
    
    match, swap_type = random.choice(candidates)
    
    original = match.group(0)
    lhs = match.group(1)
    rhs = match.group(2)
    
    if swap_type == 'nb_to_b':
        modified = f"{lhs} = {rhs};"
        desc = "Changed non-blocking (<=) to blocking (=)"
    else:
        modified = f"{lhs} <= {rhs};"
        desc = "Changed blocking (=) to non-blocking (<=)"
    
    new_code = module.source_code[:match.start()] + modified + module.source_code[match.end():]
    
    return ModificationResult(
        success=True,
        modified_code=new_code,
        modification_type="swap_blocking_nonblocking",
        description=desc,
        original_snippet=original,
        modified_snippet=modified,
        line_number=module.source_code[:match.start()].count('\n') + 1
    )


def change_sensitivity_list(module: VerilogModule, likelihood: float = 0.5) -> ModificationResult:
    """
    Modify the sensitivity list of an always block.
    
    This can cause missing triggers or simulation issues.
    """
    # Find always blocks with sensitivity lists
    pattern = re.compile(r'always\s*@\s*\(([^)]+)\)')
    
    matches = list(pattern.finditer(module.source_code))
    if not matches:
        return ModificationResult(
            success=False,
            modified_code=module.source_code,
            modification_type="change_sensitivity_list",
            description="No always blocks with sensitivity lists found"
        )
    
    candidates = [m for m in matches if random.random() < likelihood]
    if not candidates:
        candidates = [random.choice(matches)]
    
    match = random.choice(candidates)
    
    original = match.group(0)
    sens_list = match.group(1)
    
    # Various modifications to sensitivity list
    modifications = []
    
    # Remove 'or' separated items
    if ' or ' in sens_list:
        parts = sens_list.split(' or ')
        if len(parts) > 1:
            new_sens = ' or '.join(parts[:-1])  # Remove last item
            modifications.append(new_sens)
    
    # Remove posedge/negedge
    if 'posedge' in sens_list:
        modifications.append(sens_list.replace('posedge ', ''))
    if 'negedge' in sens_list:
        modifications.append(sens_list.replace('negedge ', ''))
    
    # Change * to explicit (partial)
    if sens_list.strip() == '*':
        modifications.append('a')  # Incomplete sensitivity
    
    if not modifications:
        return ModificationResult(
            success=False,
            modified_code=module.source_code,
            modification_type="change_sensitivity_list",
            description="Could not modify sensitivity list"
        )
    
    new_sens = random.choice(modifications)
    modified = f"always @({new_sens})"
    
    new_code = module.source_code[:match.start()] + modified + module.source_code[match.end():]
    
    return ModificationResult(
        success=True,
        modified_code=new_code,
        modification_type="change_sensitivity_list",
        description=f"Modified sensitivity list",
        original_snippet=original,
        modified_snippet=modified,
        line_number=module.source_code[:match.start()].count('\n') + 1
    )


def add_multiple_driver(module: VerilogModule, likelihood: float = 0.5) -> ModificationResult:
    """
    Duplicate an assignment to create multiple drivers.
    
    This causes MULTIDRIVEN lint errors.
    """
    # Find continuous assignments
    pattern = re.compile(r'(^\s*assign\s+(\w+)[^;]+;)', re.MULTILINE)
    
    matches = list(pattern.finditer(module.source_code))
    if not matches:
        return ModificationResult(
            success=False,
            modified_code=module.source_code,
            modification_type="add_multiple_driver",
            description="No continuous assignments found"
        )
    
    candidates = [m for m in matches if random.random() < likelihood]
    if not candidates:
        candidates = [random.choice(matches)]
    
    match = random.choice(candidates)
    
    original = match.group(1)
    signal_name = match.group(2)
    
    # Add a conflicting assignment
    conflicting = f"\nassign {signal_name} = 1'b0;  // Conflicting assignment"
    
    new_code = module.source_code[:match.end()] + conflicting + module.source_code[match.end():]
    
    return ModificationResult(
        success=True,
        modified_code=new_code,
        modification_type="add_multiple_driver",
        description=f"Added conflicting driver for signal '{signal_name}'",
        original_snippet=original.strip(),
        modified_snippet=f"{original.strip()}{conflicting}",
        line_number=module.source_code[:match.start()].count('\n') + 1
    )


def remove_reset_assignment(module: VerilogModule, likelihood: float = 0.5) -> ModificationResult:
    """
    Remove an assignment from a reset block.
    
    This causes uninitialized register warnings.
    """
    # Find reset blocks (if (!rst) or if (rst == 0) patterns)
    pattern = re.compile(
        r'if\s*\(\s*(!?\s*\w*rst\w*|rst\w*\s*==\s*[01\'bhd]+)\s*\)\s*begin(.*?)end',
        re.DOTALL | re.IGNORECASE
    )
    
    matches = list(pattern.finditer(module.source_code))
    if not matches:
        return ModificationResult(
            success=False,
            modified_code=module.source_code,
            modification_type="remove_reset_assignment",
            description="No reset blocks found"
        )
    
    candidates = [m for m in matches if random.random() < likelihood]
    if not candidates:
        candidates = [random.choice(matches)]
    
    match = random.choice(candidates)
    
    reset_body = match.group(2)
    
    # Find assignments in reset block
    assign_pattern = re.compile(r'\w+\s*<=\s*[^;]+;')
    assigns = list(assign_pattern.finditer(reset_body))
    
    if not assigns:
        return ModificationResult(
            success=False,
            modified_code=module.source_code,
            modification_type="remove_reset_assignment",
            description="No assignments in reset block"
        )
    
    # Remove one assignment
    assign_to_remove = random.choice(assigns)
    
    new_reset_body = reset_body[:assign_to_remove.start()] + reset_body[assign_to_remove.end():]
    
    original_block = match.group(0)
    new_block = original_block.replace(reset_body, new_reset_body)
    
    new_code = module.source_code[:match.start()] + new_block + module.source_code[match.end():]
    
    return ModificationResult(
        success=True,
        modified_code=new_code,
        modification_type="remove_reset_assignment",
        description="Removed assignment from reset block",
        original_snippet=assign_to_remove.group(0),
        modified_snippet="(removed)",
        line_number=module.source_code[:match.start()].count('\n') + 1
    )


# ============================================================================
# High-Impact Modifications
# ============================================================================

def invert_condition(module: VerilogModule, likelihood: float = 0.5) -> ModificationResult:
    """
    Invert the condition in an if statement.
    
    Example: if (x) -> if (!x)
    
    Different from invert_if_else which swaps bodies.
    """
    # Find if conditions
    pattern = re.compile(r'(\bif\s*\()([^)]+)(\)\s*begin)')
    
    matches = list(pattern.finditer(module.source_code))
    if not matches:
        return ModificationResult(
            success=False,
            modified_code=module.source_code,
            modification_type="invert_condition",
            description="No if statements found"
        )
    
    candidates = [m for m in matches if random.random() < likelihood]
    if not candidates:
        candidates = [random.choice(matches)]
    
    match = random.choice(candidates)
    
    condition = match.group(2).strip()
    
    # Invert the condition
    if condition.startswith('!'):
        # Remove negation
        new_condition = condition[1:].strip()
        if new_condition.startswith('(') and new_condition.endswith(')'):
            new_condition = new_condition[1:-1]
    else:
        # Add negation
        new_condition = f"!({condition})"
    
    original = match.group(0)
    modified = f"{match.group(1)}{new_condition}{match.group(3)}"
    
    new_code = module.source_code[:match.start()] + modified + module.source_code[match.end():]
    
    return ModificationResult(
        success=True,
        modified_code=new_code,
        modification_type="invert_condition",
        description=f"Inverted condition: '{condition}' -> '{new_condition}'",
        original_snippet=original,
        modified_snippet=modified,
        line_number=module.source_code[:match.start()].count('\n') + 1
    )


def shuffle_case_items(module: VerilogModule, likelihood: float = 0.5) -> ModificationResult:
    """
    Shuffle the order of case items in a case statement.
    
    This can break priority encoding and cause incorrect behavior.
    """
    # Find case statements
    case_pattern = re.compile(
        r'(\bcase[xz]?\s*\([^)]+\))(.*?)(\bendcase\b)',
        re.DOTALL
    )
    
    matches = list(case_pattern.finditer(module.source_code))
    if not matches:
        return ModificationResult(
            success=False,
            modified_code=module.source_code,
            modification_type="shuffle_case_items",
            description="No case statements found"
        )
    
    candidates = [m for m in matches if random.random() < likelihood]
    if not candidates:
        candidates = [random.choice(matches)]
    
    match = random.choice(candidates)
    
    case_header = match.group(1)
    case_body = match.group(2)
    case_end = match.group(3)
    
    # Parse case items (pattern: value: statement(s))
    # Split by lines that start with a case label
    item_pattern = re.compile(r'(\s*\S+\s*:.*?)(?=\s*\S+\s*:|$)', re.DOTALL)
    items = item_pattern.findall(case_body)
    
    if len(items) < 2:
        return ModificationResult(
            success=False,
            modified_code=module.source_code,
            modification_type="shuffle_case_items",
            description="Not enough case items to shuffle"
        )
    
    # Separate default from other items
    default_item = None
    other_items = []
    for item in items:
        if 'default' in item.lower():
            default_item = item
        else:
            other_items.append(item)
    
    if len(other_items) < 2:
        return ModificationResult(
            success=False,
            modified_code=module.source_code,
            modification_type="shuffle_case_items",
            description="Not enough non-default case items to shuffle"
        )
    
    # Shuffle non-default items
    random.shuffle(other_items)
    
    # Reconstruct (keep default at end if present)
    new_items = other_items
    if default_item:
        new_items.append(default_item)
    
    new_case_body = ''.join(new_items)
    
    original = match.group(0)
    modified = f"{case_header}{new_case_body}{case_end}"
    
    new_code = module.source_code[:match.start()] + modified + module.source_code[match.end():]
    
    return ModificationResult(
        success=True,
        modified_code=new_code,
        modification_type="shuffle_case_items",
        description="Shuffled case items order",
        original_snippet=original[:200],
        modified_snippet=modified[:200],
        line_number=module.source_code[:match.start()].count('\n') + 1
    )


def remove_always_block(module: VerilogModule, likelihood: float = 0.5) -> ModificationResult:
    """
    Remove an entire always block.
    
    This removes either sequential or combinational logic entirely.
    """
    # Find always blocks
    patterns = [
        re.compile(r'always_ff\s*@\s*\([^)]+\)\s*begin.*?\bend\b', re.DOTALL),
        re.compile(r'always_comb\s*begin.*?\bend\b', re.DOTALL),
        re.compile(r'always\s*@\s*\([^)]+\)\s*begin.*?\bend\b', re.DOTALL),
        re.compile(r'always\s*@\s*\*\s*begin.*?\bend\b', re.DOTALL),
    ]
    
    all_matches = []
    for pattern in patterns:
        all_matches.extend(pattern.finditer(module.source_code))
    
    if not all_matches:
        return ModificationResult(
            success=False,
            modified_code=module.source_code,
            modification_type="remove_always_block",
            description="No always blocks found"
        )
    
    candidates = [m for m in all_matches if random.random() < likelihood]
    if not candidates:
        candidates = [random.choice(all_matches)]
    
    match = random.choice(candidates)
    
    original = match.group(0)
    
    # Remove the always block
    new_code = module.source_code[:match.start()] + module.source_code[match.end():]
    
    return ModificationResult(
        success=True,
        modified_code=new_code,
        modification_type="remove_always_block",
        description="Removed an always block",
        original_snippet=original[:200],
        modified_snippet="(removed)",
        line_number=module.source_code[:match.start()].count('\n') + 1
    )


def swap_port_signals(module: VerilogModule, likelihood: float = 0.5) -> ModificationResult:
    """
    Swap signals in a module instantiation port map.
    
    This is a classic integration bug.
    """
    # Find module instantiations with port connections
    # Pattern: .port_name(signal_name)
    inst_pattern = re.compile(
        r'(\w+)\s+(?:#\s*\([^)]*\)\s+)?(\w+)\s*\((.*?)\);',
        re.DOTALL
    )
    
    matches = list(inst_pattern.finditer(module.source_code))
    
    # Filter out keywords
    keywords = {'module', 'endmodule', 'input', 'output', 'wire', 'reg', 'logic', 
                'always', 'assign', 'if', 'else', 'case', 'for', 'while', 'begin', 'end'}
    
    valid_matches = [m for m in matches if m.group(1).lower() not in keywords]
    
    if not valid_matches:
        return ModificationResult(
            success=False,
            modified_code=module.source_code,
            modification_type="swap_port_signals",
            description="No module instantiations found"
        )
    
    candidates = [m for m in valid_matches if random.random() < likelihood]
    if not candidates:
        candidates = [random.choice(valid_matches)]
    
    match = random.choice(candidates)
    
    port_connections = match.group(3)
    
    # Find named port connections: .port(signal)
    port_pattern = re.compile(r'\.(\w+)\s*\(([^)]+)\)')
    ports = list(port_pattern.finditer(port_connections))
    
    if len(ports) < 2:
        return ModificationResult(
            success=False,
            modified_code=module.source_code,
            modification_type="swap_port_signals",
            description="Not enough ports to swap"
        )
    
    # Select two ports to swap their signals
    port1, port2 = random.sample(ports, 2)
    
    port1_name = port1.group(1)
    port1_signal = port1.group(2)
    port2_name = port2.group(1)
    port2_signal = port2.group(2)
    
    # Swap the signals
    new_port_connections = port_connections
    # Replace in reverse order of position to avoid offset issues
    if port1.start() > port2.start():
        new_port_connections = (
            new_port_connections[:port1.start()] +
            f".{port1_name}({port2_signal})" +
            new_port_connections[port1.end():]
        )
        new_port_connections = (
            new_port_connections[:port2.start()] +
            f".{port2_name}({port1_signal})" +
            new_port_connections[port2.end():]
        )
    else:
        new_port_connections = (
            new_port_connections[:port2.start()] +
            f".{port2_name}({port1_signal})" +
            new_port_connections[port2.end():]
        )
        new_port_connections = (
            new_port_connections[:port1.start()] +
            f".{port1_name}({port2_signal})" +
            new_port_connections[port1.end():]
        )
    
    # Rebuild the instantiation
    module_type = match.group(1)
    inst_name = match.group(2)
    
    original = match.group(0)
    modified = f"{module_type} {inst_name}({new_port_connections});"
    
    new_code = module.source_code[:match.start()] + modified + module.source_code[match.end():]
    
    return ModificationResult(
        success=True,
        modified_code=new_code,
        modification_type="swap_port_signals",
        description=f"Swapped signals: .{port1_name}({port1_signal}) <-> .{port2_name}({port2_signal})",
        original_snippet=original[:200],
        modified_snippet=modified[:200],
        line_number=module.source_code[:match.start()].count('\n') + 1
    )


def invert_reset_polarity(module: VerilogModule, likelihood: float = 0.5) -> ModificationResult:
    """
    Invert the reset polarity.
    
    Changes negedge rst to posedge rst, or inverts reset condition check.
    """
    modifications_made = []
    new_code = module.source_code
    
    # Option 1: Change sensitivity list
    sens_pattern = re.compile(r'(always\s*@\s*\([^)]*)(negedge)(\s+\w*rst\w*)([^)]*\))', re.IGNORECASE)
    match = sens_pattern.search(new_code)
    if match and random.random() < likelihood:
        original = match.group(0)
        modified = f"{match.group(1)}posedge{match.group(3)}{match.group(4)}"
        new_code = new_code[:match.start()] + modified + new_code[match.end():]
        modifications_made.append(("sensitivity", original, modified))
    
    # Option 2: Invert reset condition
    if not modifications_made:
        cond_pattern = re.compile(r'if\s*\(\s*(!)\s*(\w*rst\w*)\s*\)', re.IGNORECASE)
        match = cond_pattern.search(new_code)
        if match:
            original = match.group(0)
            modified = f"if ({match.group(2)})"
            new_code = new_code[:match.start()] + modified + new_code[match.end():]
            modifications_made.append(("condition", original, modified))
        else:
            # Try the opposite: if (rst) -> if (!rst)
            cond_pattern2 = re.compile(r'if\s*\(\s*(\w*rst\w*)\s*\)(?!\s*begin\s*//)', re.IGNORECASE)
            match = cond_pattern2.search(new_code)
            if match:
                original = match.group(0)
                modified = f"if (!{match.group(1)})"
                new_code = new_code[:match.start()] + modified + new_code[match.end():]
                modifications_made.append(("condition", original, modified))
    
    if not modifications_made:
        return ModificationResult(
            success=False,
            modified_code=module.source_code,
            modification_type="invert_reset_polarity",
            description="No reset polarity to invert"
        )
    
    mod_type, original, modified = modifications_made[0]
    return ModificationResult(
        success=True,
        modified_code=new_code,
        modification_type="invert_reset_polarity",
        description=f"Inverted reset polarity in {mod_type}",
        original_snippet=original,
        modified_snippet=modified,
        line_number=module.source_code.find(original) // len(module.source_code.split('\n')[0]) + 1
    )


# ============================================================================
# FSM Modifications
# ============================================================================

def remove_state_transition(module: VerilogModule, likelihood: float = 0.5) -> ModificationResult:
    """
    Remove a state transition from FSM next-state logic.
    
    This breaks FSM operation by removing a branch.
    """
    # Look for FSM patterns in case statements
    # Common pattern: state_name: next_state = OTHER_STATE;
    fsm_pattern = re.compile(
        r'(\b\w+\s*:\s*)(next_state|state_next|nxt_state|state_n)\s*[<=]=\s*\w+\s*;',
        re.IGNORECASE
    )
    
    matches = list(fsm_pattern.finditer(module.source_code))
    
    if not matches:
        # Try alternative pattern: case item with state assignment
        fsm_pattern2 = re.compile(
            r'(\b[A-Z_][A-Z0-9_]*\s*:\s*begin.*?)(state|next_state)\s*[<=]=\s*\w+;(.*?end)',
            re.DOTALL | re.IGNORECASE
        )
        matches = list(fsm_pattern2.finditer(module.source_code))
    
    if not matches:
        return ModificationResult(
            success=False,
            modified_code=module.source_code,
            modification_type="remove_state_transition",
            description="No FSM state transitions found"
        )
    
    candidates = [m for m in matches if random.random() < likelihood]
    if not candidates:
        candidates = [random.choice(matches)]
    
    match = random.choice(candidates)
    original = match.group(0)
    
    # Remove the entire state case item or just the assignment
    new_code = module.source_code[:match.start()] + module.source_code[match.end():]
    
    return ModificationResult(
        success=True,
        modified_code=new_code,
        modification_type="remove_state_transition",
        description="Removed FSM state transition",
        original_snippet=original[:150],
        modified_snippet="(removed)",
        line_number=module.source_code[:match.start()].count('\n') + 1
    )


def add_unreachable_state(module: VerilogModule, likelihood: float = 0.5) -> ModificationResult:
    """
    Add an unreachable state to an FSM.
    
    This adds dead code that will never be executed.
    """
    # Find case statement for FSM
    case_pattern = re.compile(
        r'(case[xz]?\s*\(\s*(?:state|current_state|cs)\s*\))(.*?)(endcase)',
        re.DOTALL | re.IGNORECASE
    )
    
    match = case_pattern.search(module.source_code)
    
    if not match:
        return ModificationResult(
            success=False,
            modified_code=module.source_code,
            modification_type="add_unreachable_state",
            description="No FSM case statement found"
        )
    
    case_header = match.group(1)
    case_body = match.group(2)
    case_end = match.group(3)
    
    # Add unreachable state before endcase
    unreachable_state = """
        UNREACHABLE_STATE: begin
            // This state should never be reached
        end
"""
    
    modified_body = case_body + unreachable_state
    
    original = match.group(0)
    modified = f"{case_header}{modified_body}{case_end}"
    
    new_code = module.source_code[:match.start()] + modified + module.source_code[match.end():]
    
    return ModificationResult(
        success=True,
        modified_code=new_code,
        modification_type="add_unreachable_state",
        description="Added unreachable state to FSM",
        original_snippet=original[:150],
        modified_snippet=modified[:200],
        line_number=module.source_code[:match.start()].count('\n') + 1
    )


def swap_state_encoding(module: VerilogModule, likelihood: float = 0.5) -> ModificationResult:
    """
    Swap the encoding values of two FSM states.
    
    This breaks implicitly assumed state ordering.
    """
    # Find state parameter/localparam definitions
    state_pattern = re.compile(
        r'(localparam|parameter)\s+(\w+)\s*=\s*(\d+\'[bhd]\w+|\d+)',
        re.IGNORECASE
    )
    
    matches = list(state_pattern.finditer(module.source_code))
    
    # Filter to only state-like names
    state_matches = []
    for m in matches:
        name = m.group(2).upper()
        if any(kw in name for kw in ['STATE', 'IDLE', 'INIT', 'DONE', 'WAIT', 'READ', 'WRITE', 'S0', 'S1', 'S2']):
            state_matches.append(m)
    
    if len(state_matches) < 2:
        return ModificationResult(
            success=False,
            modified_code=module.source_code,
            modification_type="swap_state_encoding",
            description="Not enough FSM states to swap"
        )
    
    # Select two states to swap
    state1, state2 = random.sample(state_matches, 2)
    
    name1, val1 = state1.group(2), state1.group(3)
    name2, val2 = state2.group(2), state2.group(3)
    
    # Swap values in the source
    new_code = module.source_code
    
    # Create replacement strings
    orig1 = state1.group(0)
    orig2 = state2.group(0)
    new1 = f"{state1.group(1)} {name1} = {val2}"
    new2 = f"{state2.group(1)} {name2} = {val1}"
    
    # Replace (handle order to avoid offset issues)
    if state1.start() > state2.start():
        new_code = new_code[:state1.start()] + new1 + new_code[state1.end():]
        new_code = new_code[:state2.start()] + new2 + new_code[state2.end():]
    else:
        new_code = new_code[:state2.start()] + new2 + new_code[state2.end():]
        new_code = new_code[:state1.start()] + new1 + new_code[state1.end():]
    
    return ModificationResult(
        success=True,
        modified_code=new_code,
        modification_type="swap_state_encoding",
        description=f"Swapped state encodings: {name1}={val1}<->{name2}={val2}",
        original_snippet=f"{orig1}, {orig2}",
        modified_snippet=f"{new1}, {new2}",
        line_number=module.source_code[:state1.start()].count('\n') + 1
    )


def remove_state_update(module: VerilogModule, likelihood: float = 0.5) -> ModificationResult:
    """
    Remove the state register update (state <= next_state).
    
    This freezes the FSM in its initial state.
    """
    # Find state register update patterns
    patterns = [
        re.compile(r'(state|current_state|cs)\s*<=\s*(next_state|state_next|ns)\s*;', re.IGNORECASE),
        re.compile(r'(state|current_state|cs)\s*<=\s*\w+\s*;', re.IGNORECASE),
    ]
    
    all_matches = []
    for pattern in patterns:
        all_matches.extend(pattern.finditer(module.source_code))
    
    if not all_matches:
        return ModificationResult(
            success=False,
            modified_code=module.source_code,
            modification_type="remove_state_update",
            description="No state register update found"
        )
    
    candidates = [m for m in all_matches if random.random() < likelihood]
    if not candidates:
        candidates = [random.choice(all_matches)]
    
    match = random.choice(candidates)
    original = match.group(0)
    
    # Remove the state update
    new_code = module.source_code[:match.start()] + module.source_code[match.end():]
    
    return ModificationResult(
        success=True,
        modified_code=new_code,
        modification_type="remove_state_update",
        description="Removed state register update (FSM frozen)",
        original_snippet=original,
        modified_snippet="(removed)",
        line_number=module.source_code[:match.start()].count('\n') + 1
    )


# ============================================================================
# Clock/Reset Modifications
# ============================================================================

def remove_async_reset(module: VerilogModule, likelihood: float = 0.5) -> ModificationResult:
    """
    Remove async reset from sensitivity list, keeping only sync reset.
    
    This can cause metastability on power-up.
    """
    # Find always blocks with async reset in sensitivity
    pattern = re.compile(
        r'(always(?:_ff)?\s*@\s*\(\s*posedge\s+\w+)\s+or\s+(?:pos|neg)edge\s+\w*rst\w*(\s*\))',
        re.IGNORECASE
    )
    
    matches = list(pattern.finditer(module.source_code))
    
    if not matches:
        return ModificationResult(
            success=False,
            modified_code=module.source_code,
            modification_type="remove_async_reset",
            description="No async reset in sensitivity list found"
        )
    
    candidates = [m for m in matches if random.random() < likelihood]
    if not candidates:
        candidates = [random.choice(matches)]
    
    match = random.choice(candidates)
    
    original = match.group(0)
    # Remove the async reset part
    modified = f"{match.group(1)}{match.group(2)}"
    
    new_code = module.source_code[:match.start()] + modified + module.source_code[match.end():]
    
    return ModificationResult(
        success=True,
        modified_code=new_code,
        modification_type="remove_async_reset",
        description="Removed async reset from sensitivity list",
        original_snippet=original,
        modified_snippet=modified,
        line_number=module.source_code[:match.start()].count('\n') + 1
    )


def duplicate_clock_signal(module: VerilogModule, likelihood: float = 0.5) -> ModificationResult:
    """
    Add a duplicate clock reference, creating phantom domain issues.
    
    This can cause CDC-like issues in synthesis.
    """
    # Find clock usage in sensitivity list
    pattern = re.compile(r'(posedge|negedge)\s+(\w*clk\w*)', re.IGNORECASE)
    
    matches = list(pattern.finditer(module.source_code))
    
    if not matches:
        return ModificationResult(
            success=False,
            modified_code=module.source_code,
            modification_type="duplicate_clock_signal",
            description="No clock signals found"
        )
    
    match = matches[0]
    clk_name = match.group(2)
    
    # Find the module port list and add a fake clock
    port_pattern = re.compile(r'(module\s+\w+\s*(?:#\s*\([^)]*\))?\s*\()([^)]+)(\))')
    port_match = port_pattern.search(module.source_code)
    
    if not port_match:
        return ModificationResult(
            success=False,
            modified_code=module.source_code,
            modification_type="duplicate_clock_signal",
            description="Could not find module port list"
        )
    
    # Add duplicate clock wire declaration after ports
    fake_clk = f"clk_dup"
    new_wire = f"\n    wire {fake_clk};\n    assign {fake_clk} = {clk_name};"
    
    # Find endmodule and insert before it
    endmodule_pos = module.source_code.rfind('endmodule')
    if endmodule_pos == -1:
        return ModificationResult(
            success=False,
            modified_code=module.source_code,
            modification_type="duplicate_clock_signal",
            description="Could not find endmodule"
        )
    
    new_code = module.source_code[:endmodule_pos] + new_wire + "\n" + module.source_code[endmodule_pos:]
    
    return ModificationResult(
        success=True,
        modified_code=new_code,
        modification_type="duplicate_clock_signal",
        description=f"Added duplicate clock signal {fake_clk}",
        original_snippet="(none)",
        modified_snippet=new_wire.strip(),
        line_number=endmodule_pos // 50 + 1
    )


def modify_cdc(module: VerilogModule, likelihood: float = 0.5) -> ModificationResult:
    """
    Modify clock domain crossing by removing synchronizer.
    
    This breaks CDC safety by removing flip-flop stages.
    """
    # Look for synchronizer patterns: signal_sync, signal_meta, signal_d1/d2
    sync_patterns = [
        re.compile(r'(\w+_sync)\s*<=\s*(\w+_meta)\s*;', re.IGNORECASE),
        re.compile(r'(\w+_d2)\s*<=\s*(\w+_d1)\s*;', re.IGNORECASE),
        re.compile(r'(\w+_ff2)\s*<=\s*(\w+_ff1)\s*;', re.IGNORECASE),
        re.compile(r'(\w+)\[1\]\s*<=\s*\1\[0\]\s*;'),  # sync_reg[1] <= sync_reg[0]
    ]
    
    all_matches = []
    for pattern in sync_patterns:
        all_matches.extend(pattern.finditer(module.source_code))
    
    if not all_matches:
        # Try to find any two-stage register pattern
        two_stage = re.compile(r'(\w+)\s*<=\s*(\w+)\s*;.*?\1', re.DOTALL)
        all_matches = list(two_stage.finditer(module.source_code))
    
    if not all_matches:
        return ModificationResult(
            success=False,
            modified_code=module.source_code,
            modification_type="modify_cdc",
            description="No CDC synchronizer patterns found"
        )
    
    candidates = [m for m in all_matches if random.random() < likelihood]
    if not candidates:
        candidates = [random.choice(all_matches)]
    
    match = random.choice(candidates)
    original = match.group(0)
    
    # Remove the synchronizer stage
    new_code = module.source_code[:match.start()] + module.source_code[match.end():]
    
    return ModificationResult(
        success=True,
        modified_code=new_code,
        modification_type="modify_cdc",
        description="Removed CDC synchronizer stage",
        original_snippet=original,
        modified_snippet="(removed)",
        line_number=module.source_code[:match.start()].count('\n') + 1
    )


# ============================================================================
# Modification Registry
# ============================================================================

MODIFICATION_REGISTRY: List[ProceduralModification] = [
    # Control Flow
    ProceduralModification(
        name="invert_if_else",
        category=ModificationCategory.CONTROL_FLOW,
        description="Swap the if and else bodies",
        required_filter_indices=[1, 12],  # has_module, has_if_else
        apply_fn=invert_if_else,
    ),
    ProceduralModification(
        name="remove_else_branch",
        category=ModificationCategory.CONTROL_FLOW,
        description="Remove the else branch (may cause latch)",
        required_filter_indices=[1, 12],
        apply_fn=remove_else_branch,
    ),
    ProceduralModification(
        name="remove_case_default",
        category=ModificationCategory.CONTROL_FLOW,
        description="Remove case default (incomplete case)",
        required_filter_indices=[1, 7],  # has_module, has_case_statements
        apply_fn=remove_case_default,
    ),
    
    # Expressions
    ProceduralModification(
        name="change_constant",
        category=ModificationCategory.EXPRESSIONS,
        description="Change a numeric constant by ±1",
        required_filter_indices=[1, 23],  # has_module, has_numeric_constants
        apply_fn=change_constant,
    ),
    ProceduralModification(
        name="change_operator",
        category=ModificationCategory.EXPRESSIONS,
        description="Change operator (e.g., + to -)",
        required_filter_indices=[1, 13],  # has_module, has_operators
        apply_fn=change_operator,
    ),
    ProceduralModification(
        name="swap_operands",
        category=ModificationCategory.EXPRESSIONS,
        description="Swap operands of binary operation",
        required_filter_indices=[1, 13],
        apply_fn=swap_operands,
    ),
    ProceduralModification(
        name="change_bit_width",
        category=ModificationCategory.EXPRESSIONS,
        description="Change bit width specification",
        required_filter_indices=[1, 24],  # has_module, has_width_specs
        apply_fn=change_bit_width,
    ),
    
    # Removal
    ProceduralModification(
        name="remove_assignment",
        category=ModificationCategory.REMOVAL,
        description="Remove an assignment statement",
        required_filter_indices=[1, 8],  # has_module, has_assignments
        apply_fn=remove_assignment,
    ),
    ProceduralModification(
        name="remove_conditional",
        category=ModificationCategory.REMOVAL,
        description="Remove an if block",
        required_filter_indices=[1, 6],  # has_module, has_conditionals
        apply_fn=remove_conditional,
    ),
    ProceduralModification(
        name="remove_loop",
        category=ModificationCategory.REMOVAL,
        description="Remove a loop block",
        required_filter_indices=[1, 5],  # has_module, has_loops
        apply_fn=remove_loop,
    ),
    
    # RTL-Specific
    ProceduralModification(
        name="swap_blocking_nonblocking",
        category=ModificationCategory.RTL_SPECIFIC,
        description="Swap blocking/non-blocking assignments",
        required_filter_indices=[1, 8],  # has_module, has_assignments
        apply_fn=swap_blocking_nonblocking,
    ),
    ProceduralModification(
        name="change_sensitivity_list",
        category=ModificationCategory.RTL_SPECIFIC,
        description="Modify always block sensitivity list",
        required_filter_indices=[1, 2],  # has_module, has_always_blocks
        apply_fn=change_sensitivity_list,
    ),
    ProceduralModification(
        name="add_multiple_driver",
        category=ModificationCategory.RTL_SPECIFIC,
        description="Add conflicting driver (MULTIDRIVEN)",
        required_filter_indices=[1, 11],  # has_module, has_continuous_assignments
        apply_fn=add_multiple_driver,
    ),
    ProceduralModification(
        name="remove_reset_assignment",
        category=ModificationCategory.RTL_SPECIFIC,
        description="Remove assignment from reset block",
        required_filter_indices=[1, 3],  # has_module, has_sequential_logic
        apply_fn=remove_reset_assignment,
    ),
    
    # High-Impact Modifications
    ProceduralModification(
        name="invert_condition",
        category=ModificationCategory.CONTROL_FLOW,
        description="Invert if condition (if(x) -> if(!x))",
        required_filter_indices=[1, 6],  # has_module, has_conditionals
        apply_fn=invert_condition,
    ),
    ProceduralModification(
        name="shuffle_case_items",
        category=ModificationCategory.CONTROL_FLOW,
        description="Shuffle case item order (breaks priority)",
        required_filter_indices=[1, 7],  # has_module, has_case_statements
        apply_fn=shuffle_case_items,
    ),
    ProceduralModification(
        name="remove_always_block",
        category=ModificationCategory.REMOVAL,
        description="Remove entire always block",
        required_filter_indices=[1, 2],  # has_module, has_always_blocks
        apply_fn=remove_always_block,
    ),
    ProceduralModification(
        name="swap_port_signals",
        category=ModificationCategory.HIERARCHY,
        description="Swap signals in module port map",
        required_filter_indices=[1, 18],  # has_module, has_module_instances
        apply_fn=swap_port_signals,
    ),
    ProceduralModification(
        name="invert_reset_polarity",
        category=ModificationCategory.CLOCK_RESET,
        description="Invert reset polarity (negedge<->posedge)",
        required_filter_indices=[1, 28],  # has_module, has_reset_logic
        apply_fn=invert_reset_polarity,
    ),
    
    # FSM Modifications
    ProceduralModification(
        name="remove_state_transition",
        category=ModificationCategory.FSM,
        description="Remove FSM state transition",
        required_filter_indices=[1, 27],  # has_module, has_fsm
        apply_fn=remove_state_transition,
    ),
    ProceduralModification(
        name="add_unreachable_state",
        category=ModificationCategory.FSM,
        description="Add unreachable state to FSM",
        required_filter_indices=[1, 27],  # has_module, has_fsm
        apply_fn=add_unreachable_state,
    ),
    ProceduralModification(
        name="swap_state_encoding",
        category=ModificationCategory.FSM,
        description="Swap FSM state encoding values",
        required_filter_indices=[1, 27],  # has_module, has_fsm
        apply_fn=swap_state_encoding,
    ),
    ProceduralModification(
        name="remove_state_update",
        category=ModificationCategory.FSM,
        description="Remove state register update (freeze FSM)",
        required_filter_indices=[1, 27],  # has_module, has_fsm
        apply_fn=remove_state_update,
    ),
    
    # Clock/Reset Modifications
    ProceduralModification(
        name="remove_async_reset",
        category=ModificationCategory.CLOCK_RESET,
        description="Remove async reset from sensitivity list",
        required_filter_indices=[1, 29],  # has_module, has_async_reset
        apply_fn=remove_async_reset,
    ),
    ProceduralModification(
        name="duplicate_clock_signal",
        category=ModificationCategory.CLOCK_RESET,
        description="Add duplicate clock signal",
        required_filter_indices=[1, 3],  # has_module, has_sequential_logic
        apply_fn=duplicate_clock_signal,
    ),
    ProceduralModification(
        name="modify_cdc",
        category=ModificationCategory.TIMING,
        description="Remove CDC synchronizer stage",
        required_filter_indices=[1, 3],  # has_module, has_sequential_logic
        apply_fn=modify_cdc,
    ),
]

# Create lookup by name
MODIFICATION_BY_NAME = {m.name: m for m in MODIFICATION_REGISTRY}


def get_applicable_modifications(module: VerilogModule) -> List[ProceduralModification]:
    """
    Get all modifications that can be applied to a module.
    
    Args:
        module: Parsed VerilogModule
        
    Returns:
        List of applicable ProceduralModification objects
    """
    from filters import apply_filters
    
    applicable = []
    for mod in MODIFICATION_REGISTRY:
        if apply_filters(module, mod.required_filter_indices):
            applicable.append(mod)
    
    return applicable


def apply_random_modification(
    module: VerilogModule,
    likelihood: float = 0.5,
    modification_name: Optional[str] = None,
) -> ModificationResult:
    """
    Apply a random (or specified) modification to a module.
    
    Args:
        module: Parsed VerilogModule
        likelihood: Probability of applying modification per candidate
        modification_name: Specific modification to apply (optional)
        
    Returns:
        ModificationResult with the modified code
    """
    if modification_name:
        if modification_name not in MODIFICATION_BY_NAME:
            return ModificationResult(
                success=False,
                modified_code=module.source_code,
                modification_type=modification_name,
                description=f"Unknown modification: {modification_name}"
            )
        mod = MODIFICATION_BY_NAME[modification_name]
        return mod.apply_fn(module, likelihood)
    
    # Get applicable modifications
    applicable = get_applicable_modifications(module)
    
    if not applicable:
        return ModificationResult(
            success=False,
            modified_code=module.source_code,
            modification_type="none",
            description="No applicable modifications found"
        )
    
    # Select and apply random modification
    mod = random.choice(applicable)
    return mod.apply_fn(module, likelihood)


if __name__ == "__main__":
    # Test the modifications
    test_code = """
module test_module (
    input wire clk,
    input wire rst,
    input wire [7:0] data_in,
    output reg [7:0] data_out
);

    reg [7:0] counter;
    wire [7:0] sum;
    
    assign sum = data_in + counter;
    
    always_ff @(posedge clk or negedge rst) begin
        if (!rst) begin
            counter <= 8'b0;
            data_out <= 8'b0;
        end else begin
            counter <= counter + 1;
            data_out <= sum;
        end
    end
    
    always_comb begin
        if (counter > 100) begin
            // Do something
        end
    end

endmodule
"""
    
    parser = VerilogParser()
    module = parser.parse(test_code)
    
    print("Available Modifications:")
    print("=" * 60)
    
    applicable = get_applicable_modifications(module)
    for mod in applicable:
        print(f"  - {mod.name}: {mod.description}")
    
    print("\nApplying random modification...")
    result = apply_random_modification(module)
    
    print(f"\nResult: {'Success' if result.success else 'Failed'}")
    print(f"Type: {result.modification_type}")
    print(f"Description: {result.description}")
    print(f"Line: {result.line_number}")
    print(f"\nOriginal snippet:\n{result.original_snippet}")
    print(f"\nModified snippet:\n{result.modified_snippet}")

