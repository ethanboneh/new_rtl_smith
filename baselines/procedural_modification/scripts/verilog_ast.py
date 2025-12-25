#!/usr/bin/env python3
"""
Verilog AST Parser and Utilities.

This module provides functionality to parse Verilog/SystemVerilog code into an AST
using pyverilog and utilities to traverse/modify the AST.

Note: pyverilog has limitations with SystemVerilog-specific constructs.
This module includes fallback regex-based parsing for common constructs.
"""

import os
import re
import tempfile
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any, Tuple, Set
from enum import Enum, auto


class NodeType(Enum):
    """Types of AST nodes we care about for modifications."""
    MODULE = auto()
    ALWAYS_BLOCK = auto()
    ALWAYS_FF = auto()
    ALWAYS_COMB = auto()
    IF_STATEMENT = auto()
    CASE_STATEMENT = auto()
    FOR_LOOP = auto()
    WHILE_LOOP = auto()
    ASSIGNMENT = auto()
    BLOCKING_ASSIGNMENT = auto()
    NONBLOCKING_ASSIGNMENT = auto()
    CONTINUOUS_ASSIGNMENT = auto()
    BINARY_OPERATOR = auto()
    UNARY_OPERATOR = auto()
    COMPARISON = auto()
    WIRE_DECL = auto()
    REG_DECL = auto()
    LOGIC_DECL = auto()
    PARAMETER = auto()
    PORT = auto()
    INSTANCE = auto()
    FUNCTION = auto()
    TASK = auto()
    BEGIN_END_BLOCK = auto()
    GENERATE_BLOCK = auto()


@dataclass
class ASTNode:
    """Represents a node in the Verilog AST."""
    node_type: NodeType
    start_line: int
    end_line: int
    start_col: int = 0
    end_col: int = 0
    content: str = ""
    children: List['ASTNode'] = field(default_factory=list)
    parent: Optional['ASTNode'] = None
    attributes: Dict[str, Any] = field(default_factory=dict)
    
    def __hash__(self):
        return hash((self.node_type, self.start_line, self.end_line, self.content[:50]))


@dataclass
class VerilogModule:
    """Represents a parsed Verilog module."""
    name: str
    source_code: str
    lines: List[str]
    nodes: List[ASTNode]
    
    # Categorized nodes for easy filtering
    always_blocks: List[ASTNode] = field(default_factory=list)
    if_statements: List[ASTNode] = field(default_factory=list)
    case_statements: List[ASTNode] = field(default_factory=list)
    loops: List[ASTNode] = field(default_factory=list)
    assignments: List[ASTNode] = field(default_factory=list)
    operators: List[ASTNode] = field(default_factory=list)
    declarations: List[ASTNode] = field(default_factory=list)
    instances: List[ASTNode] = field(default_factory=list)


class VerilogParser:
    """
    Parser for Verilog/SystemVerilog code.
    
    Uses regex-based parsing to identify key constructs that can be modified.
    This approach is more robust than pyverilog for SystemVerilog constructs.
    """
    
    def __init__(self):
        """Initialize the parser with regex patterns."""
        # Patterns for different constructs
        self.patterns = {
            # Module definition
            'module': re.compile(
                r'^\s*module\s+(\w+)\s*(?:#\s*\([^)]*\))?\s*\([^)]*\)\s*;',
                re.MULTILINE | re.DOTALL
            ),
            
            # Always blocks (various types)
            'always_ff': re.compile(
                r'always_ff\s*@\s*\([^)]+\)\s*begin',
                re.MULTILINE
            ),
            'always_comb': re.compile(
                r'always_comb\s*begin',
                re.MULTILINE
            ),
            'always_latch': re.compile(
                r'always_latch\s*begin',
                re.MULTILINE
            ),
            'always_at': re.compile(
                r'always\s*@\s*\([^)]+\)\s*begin',
                re.MULTILINE
            ),
            'always_star': re.compile(
                r'always\s*@\s*\*\s*begin',
                re.MULTILINE
            ),
            
            # Control flow
            'if_statement': re.compile(
                r'\bif\s*\([^)]+\)\s*begin',
                re.MULTILINE
            ),
            'if_else': re.compile(
                r'\bif\s*\([^)]+\)\s*begin.*?\bend\s*else\s*begin',
                re.MULTILINE | re.DOTALL
            ),
            'case_statement': re.compile(
                r'\bcase[xz]?\s*\([^)]+\)',
                re.MULTILINE
            ),
            
            # Loops
            'for_loop': re.compile(
                r'\bfor\s*\([^)]+\)\s*begin',
                re.MULTILINE
            ),
            'while_loop': re.compile(
                r'\bwhile\s*\([^)]+\)\s*begin',
                re.MULTILINE
            ),
            
            # Assignments
            'blocking_assign': re.compile(
                r'(\w+(?:\s*\[[^\]]+\])?)\s*=\s*([^;]+);',
                re.MULTILINE
            ),
            'nonblocking_assign': re.compile(
                r'(\w+(?:\s*\[[^\]]+\])?)\s*<=\s*([^;]+);',
                re.MULTILINE
            ),
            'continuous_assign': re.compile(
                r'^\s*assign\s+(\w+(?:\s*\[[^\]]+\])?)\s*=\s*([^;]+);',
                re.MULTILINE
            ),
            
            # Binary operators
            'binary_op': re.compile(
                r'(\w+)\s*([+\-*/%&|^]|<<|>>|&&|\|\|)\s*(\w+)',
                re.MULTILINE
            ),
            'comparison_op': re.compile(
                r'(\w+)\s*(==|!=|===|!==|<|>|<=|>=)\s*(\w+)',
                re.MULTILINE
            ),
            
            # Declarations
            'wire_decl': re.compile(
                r'^\s*wire\s+(?:signed\s+)?(?:\[[^\]]+\]\s+)?(\w+)',
                re.MULTILINE
            ),
            'reg_decl': re.compile(
                r'^\s*reg\s+(?:signed\s+)?(?:\[[^\]]+\]\s+)?(\w+)',
                re.MULTILINE
            ),
            'logic_decl': re.compile(
                r'^\s*logic\s+(?:signed\s+)?(?:\[[^\]]+\]\s+)?(\w+)',
                re.MULTILINE
            ),
            
            # Module instances
            'instance': re.compile(
                r'^\s*(\w+)\s+(?:#\s*\([^)]*\)\s+)?(\w+)\s*\(',
                re.MULTILINE
            ),
            
            # Parameters
            'parameter': re.compile(
                r'^\s*(?:localparam|parameter)\s+(?:\w+\s+)?(\w+)\s*=\s*([^;,]+)',
                re.MULTILINE
            ),
        }
    
    def parse(self, source_code: str) -> VerilogModule:
        """
        Parse Verilog source code into structured representation.
        
        Args:
            source_code: Verilog/SystemVerilog source code
            
        Returns:
            VerilogModule containing parsed AST nodes
        """
        lines = source_code.split('\n')
        nodes = []
        
        # Extract module name
        module_match = self.patterns['module'].search(source_code)
        module_name = module_match.group(1) if module_match else "unknown"
        
        # Parse always blocks
        always_blocks = self._parse_always_blocks(source_code, lines)
        nodes.extend(always_blocks)
        
        # Parse if statements
        if_statements = self._parse_if_statements(source_code, lines)
        nodes.extend(if_statements)
        
        # Parse case statements
        case_statements = self._parse_case_statements(source_code, lines)
        nodes.extend(case_statements)
        
        # Parse loops
        loops = self._parse_loops(source_code, lines)
        nodes.extend(loops)
        
        # Parse assignments
        assignments = self._parse_assignments(source_code, lines)
        nodes.extend(assignments)
        
        # Parse operators
        operators = self._parse_operators(source_code, lines)
        nodes.extend(operators)
        
        # Parse declarations
        declarations = self._parse_declarations(source_code, lines)
        nodes.extend(declarations)
        
        # Parse instances
        instances = self._parse_instances(source_code, lines)
        nodes.extend(instances)
        
        return VerilogModule(
            name=module_name,
            source_code=source_code,
            lines=lines,
            nodes=nodes,
            always_blocks=always_blocks,
            if_statements=if_statements,
            case_statements=case_statements,
            loops=loops,
            assignments=assignments,
            operators=operators,
            declarations=declarations,
            instances=instances,
        )
    
    def _get_line_number(self, source: str, pos: int) -> int:
        """Get the line number for a position in the source."""
        return source[:pos].count('\n') + 1
    
    def _find_matching_end(self, source: str, start_pos: int) -> int:
        """
        Find the matching 'end' for a 'begin' block.
        
        Args:
            source: Source code
            start_pos: Position of 'begin'
            
        Returns:
            Position of matching 'end'
        """
        depth = 0
        i = start_pos
        in_string = False
        string_char = None
        
        while i < len(source):
            char = source[i]
            
            # Handle strings
            if char in '"\'':
                if not in_string:
                    in_string = True
                    string_char = char
                elif char == string_char:
                    in_string = False
                i += 1
                continue
            
            if in_string:
                i += 1
                continue
            
            # Check for keywords
            if source[i:i+5] == 'begin':
                depth += 1
                i += 5
            elif source[i:i+3] == 'end' and (i + 3 >= len(source) or not source[i+3].isalnum()):
                depth -= 1
                if depth == 0:
                    return i + 3
                i += 3
            else:
                i += 1
        
        return len(source)
    
    def _parse_always_blocks(self, source: str, lines: List[str]) -> List[ASTNode]:
        """Parse always blocks from source code."""
        nodes = []
        
        # Different always block types
        patterns = [
            (self.patterns['always_ff'], NodeType.ALWAYS_FF),
            (self.patterns['always_comb'], NodeType.ALWAYS_COMB),
            (self.patterns['always_latch'], NodeType.ALWAYS_BLOCK),
            (self.patterns['always_at'], NodeType.ALWAYS_BLOCK),
            (self.patterns['always_star'], NodeType.ALWAYS_BLOCK),
        ]
        
        for pattern, node_type in patterns:
            for match in pattern.finditer(source):
                start_pos = match.start()
                # Find the 'begin' and matching 'end'
                begin_pos = match.end() - 5  # Position of 'begin'
                end_pos = self._find_matching_end(source, begin_pos)
                
                start_line = self._get_line_number(source, start_pos)
                end_line = self._get_line_number(source, end_pos)
                
                content = source[start_pos:end_pos]
                
                node = ASTNode(
                    node_type=node_type,
                    start_line=start_line,
                    end_line=end_line,
                    content=content,
                    attributes={'match': match.group()}
                )
                nodes.append(node)
        
        return nodes
    
    def _parse_if_statements(self, source: str, lines: List[str]) -> List[ASTNode]:
        """Parse if statements from source code."""
        nodes = []
        
        for match in self.patterns['if_statement'].finditer(source):
            start_pos = match.start()
            begin_pos = match.end() - 5  # Position of 'begin'
            end_pos = self._find_matching_end(source, begin_pos)
            
            start_line = self._get_line_number(source, start_pos)
            end_line = self._get_line_number(source, end_pos)
            
            content = source[start_pos:end_pos]
            
            # Check if this has an else clause
            # Look for 'else' pattern in the source after this if statement
            # We need to check if 'else' follows the first 'end' of this if block
            has_else = False
            
            # Check within the content for 'else begin' pattern
            if 'else' in content:
                # Count if there's an 'else begin' that's part of this if
                has_else = bool(re.search(r'\bend\s+else\b', content))
            
            # Also check right after the end
            remaining = source[end_pos:end_pos+50].strip()
            if remaining.startswith('else'):
                has_else = True
            
            node = ASTNode(
                node_type=NodeType.IF_STATEMENT,
                start_line=start_line,
                end_line=end_line,
                content=content,
                attributes={'has_else': has_else}
            )
            nodes.append(node)
        
        return nodes
    
    def _parse_case_statements(self, source: str, lines: List[str]) -> List[ASTNode]:
        """Parse case statements from source code."""
        nodes = []
        
        for match in self.patterns['case_statement'].finditer(source):
            start_pos = match.start()
            
            # Find matching endcase
            endcase_match = re.search(r'\bendcase\b', source[start_pos:])
            if endcase_match:
                end_pos = start_pos + endcase_match.end()
            else:
                continue
            
            start_line = self._get_line_number(source, start_pos)
            end_line = self._get_line_number(source, end_pos)
            
            content = source[start_pos:end_pos]
            
            node = ASTNode(
                node_type=NodeType.CASE_STATEMENT,
                start_line=start_line,
                end_line=end_line,
                content=content,
            )
            nodes.append(node)
        
        return nodes
    
    def _parse_loops(self, source: str, lines: List[str]) -> List[ASTNode]:
        """Parse for and while loops from source code."""
        nodes = []
        
        for pattern, node_type in [(self.patterns['for_loop'], NodeType.FOR_LOOP),
                                    (self.patterns['while_loop'], NodeType.WHILE_LOOP)]:
            for match in pattern.finditer(source):
                start_pos = match.start()
                begin_pos = match.end() - 5  # Position of 'begin'
                end_pos = self._find_matching_end(source, begin_pos)
                
                start_line = self._get_line_number(source, start_pos)
                end_line = self._get_line_number(source, end_pos)
                
                content = source[start_pos:end_pos]
                
                node = ASTNode(
                    node_type=node_type,
                    start_line=start_line,
                    end_line=end_line,
                    content=content,
                )
                nodes.append(node)
        
        return nodes
    
    def _parse_assignments(self, source: str, lines: List[str]) -> List[ASTNode]:
        """Parse assignments from source code."""
        nodes = []
        
        # Continuous assignments
        for match in self.patterns['continuous_assign'].finditer(source):
            start_pos = match.start()
            end_pos = match.end()
            
            start_line = self._get_line_number(source, start_pos)
            
            node = ASTNode(
                node_type=NodeType.CONTINUOUS_ASSIGNMENT,
                start_line=start_line,
                end_line=start_line,
                content=match.group(),
                attributes={
                    'lhs': match.group(1),
                    'rhs': match.group(2),
                }
            )
            nodes.append(node)
        
        # Non-blocking assignments (must check before blocking to avoid false matches)
        for match in self.patterns['nonblocking_assign'].finditer(source):
            start_pos = match.start()
            start_line = self._get_line_number(source, start_pos)
            
            node = ASTNode(
                node_type=NodeType.NONBLOCKING_ASSIGNMENT,
                start_line=start_line,
                end_line=start_line,
                content=match.group(),
                attributes={
                    'lhs': match.group(1),
                    'rhs': match.group(2),
                }
            )
            nodes.append(node)
        
        # Blocking assignments (exclude continuous assigns and non-blocking)
        blocking_pattern = re.compile(r'(?<!assign\s)(\w+(?:\s*\[[^\]]+\])?)\s*=(?!=)\s*([^;<=]+);', re.MULTILINE)
        for match in blocking_pattern.finditer(source):
            # Skip if this is part of a non-blocking assignment
            start_pos = match.start()
            if source[max(0, start_pos-2):start_pos+1].strip().endswith('<='):
                continue
                
            start_line = self._get_line_number(source, start_pos)
            
            node = ASTNode(
                node_type=NodeType.BLOCKING_ASSIGNMENT,
                start_line=start_line,
                end_line=start_line,
                content=match.group(),
                attributes={
                    'lhs': match.group(1),
                    'rhs': match.group(2),
                }
            )
            nodes.append(node)
        
        return nodes
    
    def _parse_operators(self, source: str, lines: List[str]) -> List[ASTNode]:
        """Parse binary and comparison operators from source code."""
        nodes = []
        
        # Binary operators
        for match in self.patterns['binary_op'].finditer(source):
            start_pos = match.start()
            start_line = self._get_line_number(source, start_pos)
            
            node = ASTNode(
                node_type=NodeType.BINARY_OPERATOR,
                start_line=start_line,
                end_line=start_line,
                content=match.group(),
                attributes={
                    'left': match.group(1),
                    'operator': match.group(2),
                    'right': match.group(3),
                }
            )
            nodes.append(node)
        
        # Comparison operators
        for match in self.patterns['comparison_op'].finditer(source):
            start_pos = match.start()
            start_line = self._get_line_number(source, start_pos)
            
            node = ASTNode(
                node_type=NodeType.COMPARISON,
                start_line=start_line,
                end_line=start_line,
                content=match.group(),
                attributes={
                    'left': match.group(1),
                    'operator': match.group(2),
                    'right': match.group(3),
                }
            )
            nodes.append(node)
        
        return nodes
    
    def _parse_declarations(self, source: str, lines: List[str]) -> List[ASTNode]:
        """Parse wire, reg, logic declarations from source code."""
        nodes = []
        
        for pattern, node_type in [
            (self.patterns['wire_decl'], NodeType.WIRE_DECL),
            (self.patterns['reg_decl'], NodeType.REG_DECL),
            (self.patterns['logic_decl'], NodeType.LOGIC_DECL),
        ]:
            for match in pattern.finditer(source):
                start_pos = match.start()
                start_line = self._get_line_number(source, start_pos)
                
                node = ASTNode(
                    node_type=node_type,
                    start_line=start_line,
                    end_line=start_line,
                    content=match.group(),
                    attributes={'name': match.group(1)}
                )
                nodes.append(node)
        
        return nodes
    
    def _parse_instances(self, source: str, lines: List[str]) -> List[ASTNode]:
        """Parse module instances from source code."""
        nodes = []
        
        # Common Verilog keywords that are not module names
        keywords = {
            'module', 'endmodule', 'input', 'output', 'inout', 'wire', 'reg',
            'logic', 'always', 'assign', 'begin', 'end', 'if', 'else', 'case',
            'endcase', 'for', 'while', 'parameter', 'localparam', 'function',
            'endfunction', 'task', 'endtask', 'initial', 'posedge', 'negedge',
            'or', 'and', 'not', 'xor', 'nand', 'nor', 'xnor', 'buf', 'integer',
            'real', 'time', 'genvar', 'generate', 'endgenerate',
        }
        
        for match in self.patterns['instance'].finditer(source):
            module_type = match.group(1)
            instance_name = match.group(2)
            
            # Skip if module_type is a keyword
            if module_type.lower() in keywords:
                continue
            
            start_pos = match.start()
            
            # Find the closing );
            remaining = source[start_pos:]
            depth = 0
            end_offset = 0
            for i, char in enumerate(remaining):
                if char == '(':
                    depth += 1
                elif char == ')':
                    depth -= 1
                    if depth == 0:
                        end_offset = i + 2  # Include );
                        break
            
            start_line = self._get_line_number(source, start_pos)
            end_line = self._get_line_number(source, start_pos + end_offset)
            
            content = source[start_pos:start_pos + end_offset]
            
            node = ASTNode(
                node_type=NodeType.INSTANCE,
                start_line=start_line,
                end_line=end_line,
                content=content,
                attributes={
                    'module_type': module_type,
                    'instance_name': instance_name,
                }
            )
            nodes.append(node)
        
        return nodes


def calculate_complexity(module: VerilogModule) -> int:
    """
    Calculate complexity score for a Verilog module.
    
    Complexity = number of:
    - Conditional blocks (if, case)
    - Loops (for, while)
    - Boolean operators
    - Comparison operators
    - Always blocks
    
    Args:
        module: Parsed VerilogModule
        
    Returns:
        Complexity score
    """
    complexity = 0
    
    complexity += len(module.if_statements)
    complexity += len(module.case_statements)
    complexity += len(module.loops)
    complexity += len(module.always_blocks)
    
    # Count operators
    for node in module.operators:
        if node.node_type in (NodeType.BINARY_OPERATOR, NodeType.COMPARISON):
            complexity += 1
    
    return complexity


if __name__ == "__main__":
    # Test the parser
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
    
    print(f"Module name: {module.name}")
    print(f"Always blocks: {len(module.always_blocks)}")
    print(f"If statements: {len(module.if_statements)}")
    print(f"Assignments: {len(module.assignments)}")
    print(f"Operators: {len(module.operators)}")
    print(f"Complexity: {calculate_complexity(module)}")

