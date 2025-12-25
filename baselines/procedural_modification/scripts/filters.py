#!/usr/bin/env python3
"""
Filtering criteria for Verilog AST nodes.

This module implements filters similar to SWE-smith's criteria for identifying
Verilog code that can be modified with specific procedural modifications.

Each filter function takes a VerilogModule and returns True if the module
passes the filter criteria.
"""

from typing import Callable, List, Optional
from dataclasses import dataclass

from verilog_ast import VerilogModule, NodeType, calculate_complexity


@dataclass
class FilterCriteria:
    """Represents a filter criterion with its index and description."""
    index: int
    name: str
    description: str
    filter_fn: Callable[[VerilogModule], bool]


def filter_has_module(module: VerilogModule) -> bool:
    """
    Filter 1: Check if the code contains a valid module definition.
    
    Equivalent to SWE-smith's filter_functions - checks if we have a module.
    """
    return module.name != "unknown" and len(module.source_code.strip()) > 0


def filter_has_always_blocks(module: VerilogModule) -> bool:
    """
    Filter 2: Check if the module contains always blocks.
    
    Always blocks are the primary procedural blocks in Verilog.
    """
    return len(module.always_blocks) > 0


def filter_has_sequential_logic(module: VerilogModule) -> bool:
    """
    Filter 3: Check if the module has sequential logic (always_ff or clocked always).
    
    Sequential logic uses posedge/negedge for clock synchronization.
    """
    for node in module.always_blocks:
        if node.node_type == NodeType.ALWAYS_FF:
            return True
        # Check for posedge/negedge in regular always blocks
        if 'posedge' in node.content or 'negedge' in node.content:
            return True
    return False


def filter_has_combinational_logic(module: VerilogModule) -> bool:
    """
    Filter 4: Check if the module has combinational logic (always_comb or always @*).
    """
    for node in module.always_blocks:
        if node.node_type == NodeType.ALWAYS_COMB:
            return True
        # Check for always @* or always @(*)
        if '@*' in node.content or '@(*)' in node.content:
            return True
    return False


def filter_has_loops(module: VerilogModule) -> bool:
    """
    Filter 5: Check if the module contains for or while loops.
    
    Equivalent to SWE-smith's filter_loops.
    """
    return len(module.loops) > 0


def filter_has_conditionals(module: VerilogModule) -> bool:
    """
    Filter 6: Check if the module contains if statements.
    
    Equivalent to SWE-smith's filter_conditionals.
    """
    return len(module.if_statements) > 0


def filter_has_case_statements(module: VerilogModule) -> bool:
    """
    Filter 7: Check if the module contains case statements.
    
    Case statements are common in FSMs and decoders.
    """
    return len(module.case_statements) > 0


def filter_has_assignments(module: VerilogModule) -> bool:
    """
    Filter 8: Check if the module contains assignments.
    
    Equivalent to SWE-smith's filter_assignments.
    """
    return len(module.assignments) > 0


def filter_has_blocking_assignments(module: VerilogModule) -> bool:
    """
    Filter 9: Check if the module contains blocking assignments (=).
    """
    return any(
        node.node_type == NodeType.BLOCKING_ASSIGNMENT
        for node in module.assignments
    )


def filter_has_nonblocking_assignments(module: VerilogModule) -> bool:
    """
    Filter 10: Check if the module contains non-blocking assignments (<=).
    """
    return any(
        node.node_type == NodeType.NONBLOCKING_ASSIGNMENT
        for node in module.assignments
    )


def filter_has_continuous_assignments(module: VerilogModule) -> bool:
    """
    Filter 11: Check if the module contains continuous assignments (assign).
    """
    return any(
        node.node_type == NodeType.CONTINUOUS_ASSIGNMENT
        for node in module.assignments
    )


def filter_has_if_else(module: VerilogModule) -> bool:
    """
    Filter 12: Check if the module contains if-else blocks.
    
    Equivalent to SWE-smith's filter_if_else.
    """
    return any(
        node.attributes.get('has_else', False)
        for node in module.if_statements
    )


def filter_has_operators(module: VerilogModule) -> bool:
    """
    Filter 13: Check if the module contains binary or boolean operators.
    
    Equivalent to SWE-smith's filter_operators.
    """
    return len(module.operators) > 0


def filter_has_arithmetic_operators(module: VerilogModule) -> bool:
    """
    Filter 14: Check if the module contains arithmetic operators (+, -, *, /, %).
    """
    arithmetic_ops = {'+', '-', '*', '/', '%'}
    return any(
        node.attributes.get('operator', '') in arithmetic_ops
        for node in module.operators
        if node.node_type == NodeType.BINARY_OPERATOR
    )


def filter_has_logical_operators(module: VerilogModule) -> bool:
    """
    Filter 15: Check if the module contains logical operators (&&, ||, !).
    """
    logical_ops = {'&&', '||'}
    return any(
        node.attributes.get('operator', '') in logical_ops
        for node in module.operators
        if node.node_type == NodeType.BINARY_OPERATOR
    )


def filter_has_bitwise_operators(module: VerilogModule) -> bool:
    """
    Filter 16: Check if the module contains bitwise operators (&, |, ^, ~, <<, >>).
    """
    bitwise_ops = {'&', '|', '^', '~', '<<', '>>'}
    return any(
        node.attributes.get('operator', '') in bitwise_ops
        for node in module.operators
        if node.node_type == NodeType.BINARY_OPERATOR
    )


def filter_has_comparisons(module: VerilogModule) -> bool:
    """
    Filter 17: Check if the module contains comparison operators.
    """
    return any(
        node.node_type == NodeType.COMPARISON
        for node in module.operators
    )


def filter_has_module_instances(module: VerilogModule) -> bool:
    """
    Filter 18: Check if the module instantiates other modules.
    """
    return len(module.instances) > 0


def filter_min_complexity(module: VerilogModule, min_complexity: int = 3) -> bool:
    """
    Filter 19: Check if the module meets minimum complexity threshold.
    
    Equivalent to SWE-smith's filter_min_complexity.
    Filters out trivially simple modules.
    """
    return calculate_complexity(module) >= min_complexity


def filter_max_complexity(module: VerilogModule, max_complexity: int = 50) -> bool:
    """
    Filter 20: Check if the module is below maximum complexity threshold.
    
    Equivalent to SWE-smith's filter_max_complexity.
    Filters out extremely complex modules that are hard to modify.
    """
    return calculate_complexity(module) <= max_complexity


def filter_min_lines(module: VerilogModule, min_lines: int = 10) -> bool:
    """
    Filter 21: Check if the module has at least a minimum number of lines.
    """
    return len(module.lines) >= min_lines


def filter_max_lines(module: VerilogModule, max_lines: int = 500) -> bool:
    """
    Filter 22: Check if the module is below maximum line count.
    """
    return len(module.lines) <= max_lines


def filter_has_numeric_constants(module: VerilogModule) -> bool:
    """
    Filter 23: Check if the module contains numeric constants.
    
    Used for constant modification procedures.
    """
    import re
    # Look for Verilog numeric literals
    patterns = [
        r"\d+'[bBhHdDoO][0-9a-fA-FxXzZ_]+",  # Sized literals like 8'b0, 16'hFF
        r"\d+",  # Plain decimal numbers
    ]
    
    for pattern in patterns:
        if re.search(pattern, module.source_code):
            return True
    return False


def filter_has_width_specs(module: VerilogModule) -> bool:
    """
    Filter 24: Check if the module contains bit width specifications [n:m].
    """
    import re
    return bool(re.search(r'\[\s*\d+\s*:\s*\d+\s*\]', module.source_code))


def filter_has_generate_blocks(module: VerilogModule) -> bool:
    """
    Filter 25: Check if the module contains generate blocks.
    """
    return 'generate' in module.source_code.lower()


def filter_has_ports(module: VerilogModule) -> bool:
    """
    Filter 26: Check if the module has input/output ports.
    """
    import re
    return bool(re.search(r'\b(input|output|inout)\b', module.source_code))


def filter_has_fsm(module: VerilogModule) -> bool:
    """
    Filter 27: Check if the module contains FSM patterns.
    
    Looks for:
    - State register declarations (state, next_state, current_state)
    - Enum/parameter state definitions
    - Case statements with state-like patterns
    """
    import re
    source = module.source_code.lower()
    
    # Look for state-related signal names
    state_patterns = [
        r'\breg\s+.*\bstate\b',
        r'\blogic\s+.*\bstate\b',
        r'\b(current_state|next_state|state_reg|state_next)\b',
        r'\bparameter\s+\w+\s*=.*state',
        r'\benum\b.*\bstate\b',
        r'\blocalparam\s+.*\b(IDLE|INIT|DONE|WAIT|READ|WRITE)\b',
    ]
    
    for pattern in state_patterns:
        if re.search(pattern, source, re.IGNORECASE):
            return True
    
    # Check for case statements with state-like items
    if len(module.case_statements) > 0:
        for node in module.case_statements:
            if re.search(r'\b(IDLE|INIT|DONE|WAIT|READ|WRITE|S\d+|STATE)', node.content, re.IGNORECASE):
                return True
    
    return False


def filter_has_reset_logic(module: VerilogModule) -> bool:
    """
    Filter 28: Check if the module has explicit reset logic.
    """
    import re
    # Look for reset patterns
    patterns = [
        r'\brst\b',
        r'\breset\b',
        r'\brstn?\b',
        r'\bareset\b',
        r'\bsrst\b',
        r'negedge\s+\w*rst',
        r'posedge\s+\w*rst',
        r'if\s*\(\s*!?\s*\w*rst',
    ]
    
    for pattern in patterns:
        if re.search(pattern, module.source_code, re.IGNORECASE):
            return True
    return False


def filter_has_async_reset(module: VerilogModule) -> bool:
    """
    Filter 29: Check if the module has asynchronous reset.
    """
    import re
    # Async reset appears in sensitivity list: always @(posedge clk or negedge rst)
    return bool(re.search(
        r'always\s*@\s*\([^)]*\b(posedge|negedge)\s+\w*rst\w*[^)]*\)',
        module.source_code,
        re.IGNORECASE
    ))


def filter_has_multiple_clocks(module: VerilogModule) -> bool:
    """
    Filter 30: Check if the module has multiple clock domains (CDC candidate).
    """
    import re
    # Find all clock references in sensitivity lists
    clock_pattern = r'(posedge|negedge)\s+(\w+)'
    matches = re.findall(clock_pattern, module.source_code)
    
    # Extract unique clock names (excluding reset signals)
    clocks = set()
    for edge, signal in matches:
        sig_lower = signal.lower()
        if not any(rst in sig_lower for rst in ['rst', 'reset']):
            clocks.add(signal)
    
    return len(clocks) > 1


def filter_has_parameters(module: VerilogModule) -> bool:
    """
    Filter 31: Check if the module uses parameters or localparams.
    """
    import re
    return bool(re.search(r'\b(parameter|localparam)\b', module.source_code, re.IGNORECASE))


# ============================================================================
# Filter Registry
# ============================================================================

# All available filters with their indices and descriptions
FILTER_REGISTRY: List[FilterCriteria] = [
    FilterCriteria(1, "filter_has_module", "Is this a valid module definition", filter_has_module),
    FilterCriteria(2, "filter_has_always_blocks", "Does the module contain always blocks", filter_has_always_blocks),
    FilterCriteria(3, "filter_has_sequential_logic", "Does the module have sequential logic (always_ff, posedge)", filter_has_sequential_logic),
    FilterCriteria(4, "filter_has_combinational_logic", "Does the module have combinational logic (always_comb, @*)", filter_has_combinational_logic),
    FilterCriteria(5, "filter_has_loops", "Does the module contain for/while loops", filter_has_loops),
    FilterCriteria(6, "filter_has_conditionals", "Does the module contain if statements", filter_has_conditionals),
    FilterCriteria(7, "filter_has_case_statements", "Does the module contain case statements", filter_has_case_statements),
    FilterCriteria(8, "filter_has_assignments", "Does the module contain assignments", filter_has_assignments),
    FilterCriteria(9, "filter_has_blocking_assignments", "Does the module contain blocking assignments (=)", filter_has_blocking_assignments),
    FilterCriteria(10, "filter_has_nonblocking_assignments", "Does the module contain non-blocking assignments (<=)", filter_has_nonblocking_assignments),
    FilterCriteria(11, "filter_has_continuous_assignments", "Does the module contain continuous assignments (assign)", filter_has_continuous_assignments),
    FilterCriteria(12, "filter_has_if_else", "Does the module contain if-else blocks", filter_has_if_else),
    FilterCriteria(13, "filter_has_operators", "Does the module contain binary/boolean operators", filter_has_operators),
    FilterCriteria(14, "filter_has_arithmetic_operators", "Does the module contain arithmetic operators (+,-,*,/,%)", filter_has_arithmetic_operators),
    FilterCriteria(15, "filter_has_logical_operators", "Does the module contain logical operators (&&, ||)", filter_has_logical_operators),
    FilterCriteria(16, "filter_has_bitwise_operators", "Does the module contain bitwise operators (&,|,^,<<,>>)", filter_has_bitwise_operators),
    FilterCriteria(17, "filter_has_comparisons", "Does the module contain comparison operators", filter_has_comparisons),
    FilterCriteria(18, "filter_has_module_instances", "Does the module instantiate other modules", filter_has_module_instances),
    FilterCriteria(19, "filter_min_complexity", "Is the module above minimum complexity (default: 3)", filter_min_complexity),
    FilterCriteria(20, "filter_max_complexity", "Is the module below maximum complexity (default: 50)", filter_max_complexity),
    FilterCriteria(21, "filter_min_lines", "Does the module have minimum lines (default: 10)", filter_min_lines),
    FilterCriteria(22, "filter_max_lines", "Is the module below maximum lines (default: 500)", filter_max_lines),
    FilterCriteria(23, "filter_has_numeric_constants", "Does the module contain numeric constants", filter_has_numeric_constants),
    FilterCriteria(24, "filter_has_width_specs", "Does the module contain bit width specifications [n:m]", filter_has_width_specs),
    FilterCriteria(25, "filter_has_generate_blocks", "Does the module contain generate blocks", filter_has_generate_blocks),
    FilterCriteria(26, "filter_has_ports", "Does the module have input/output ports", filter_has_ports),
    FilterCriteria(27, "filter_has_fsm", "Does the module contain FSM patterns", filter_has_fsm),
    FilterCriteria(28, "filter_has_reset_logic", "Does the module have explicit reset logic", filter_has_reset_logic),
    FilterCriteria(29, "filter_has_async_reset", "Does the module have asynchronous reset", filter_has_async_reset),
    FilterCriteria(30, "filter_has_multiple_clocks", "Does the module have multiple clock domains (CDC)", filter_has_multiple_clocks),
    FilterCriteria(31, "filter_has_parameters", "Does the module use parameters or localparams", filter_has_parameters),
]

# Create lookup by name
FILTER_BY_NAME = {f.name: f for f in FILTER_REGISTRY}
FILTER_BY_INDEX = {f.index: f for f in FILTER_REGISTRY}


def apply_filters(module: VerilogModule, filter_indices: List[int]) -> bool:
    """
    Apply multiple filters to a module.
    
    Args:
        module: Parsed VerilogModule
        filter_indices: List of filter indices to apply
        
    Returns:
        True if module passes ALL filters
    """
    for idx in filter_indices:
        if idx not in FILTER_BY_INDEX:
            raise ValueError(f"Unknown filter index: {idx}")
        
        filter_criteria = FILTER_BY_INDEX[idx]
        if not filter_criteria.filter_fn(module):
            return False
    
    return True


def get_applicable_modifications(module: VerilogModule) -> List[str]:
    """
    Determine which procedural modifications can be applied to a module.
    
    Args:
        module: Parsed VerilogModule
        
    Returns:
        List of applicable modification names
    """
    # Import here to avoid circular dependency
    from modifications import MODIFICATION_REGISTRY
    
    applicable = []
    for mod in MODIFICATION_REGISTRY:
        if apply_filters(module, mod.required_filter_indices):
            applicable.append(mod.name)
    
    return applicable


if __name__ == "__main__":
    # Test the filters
    from verilog_ast import VerilogParser
    
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
    
    print("Filter Results:")
    print("=" * 60)
    for criteria in FILTER_REGISTRY:
        result = criteria.filter_fn(module)
        status = "✓" if result else "✗"
        print(f"  {status} {criteria.name}: {criteria.description}")

