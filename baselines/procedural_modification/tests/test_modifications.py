#!/usr/bin/env python3
"""
Tests for procedural modifications.

Run with: python -m pytest tests/test_modifications.py -v
Or: python tests/test_modifications.py
"""

import sys
import os

# Add scripts directory to path
script_dir = os.path.join(os.path.dirname(__file__), '..', 'scripts')
sys.path.insert(0, script_dir)

from verilog_ast import VerilogParser, VerilogModule, calculate_complexity
from filters import (
    filter_has_module,
    filter_has_always_blocks,
    filter_has_conditionals,
    filter_has_if_else,
    filter_has_assignments,
    filter_has_operators,
    filter_has_numeric_constants,
    apply_filters,
)
from modifications import (
    invert_if_else,
    remove_else_branch,
    change_constant,
    change_operator,
    swap_operands,
    remove_assignment,
    swap_blocking_nonblocking,
    get_applicable_modifications,
    apply_random_modification,
)


# Test Verilog code samples
SIMPLE_MODULE = """
module simple_adder (
    input wire [7:0] a,
    input wire [7:0] b,
    output wire [7:0] sum
);
    assign sum = a + b;
endmodule
"""

SEQUENTIAL_MODULE = """
module counter (
    input wire clk,
    input wire rst,
    output reg [7:0] count
);
    always_ff @(posedge clk or negedge rst) begin
        if (!rst) begin
            count <= 8'b0;
        end else begin
            count <= count + 1;
        end
    end
endmodule
"""

COMBINATIONAL_MODULE = """
module mux4 (
    input wire [1:0] sel,
    input wire [7:0] a, b, c, d,
    output reg [7:0] out
);
    always_comb begin
        case (sel)
            2'b00: out = a;
            2'b01: out = b;
            2'b10: out = c;
            2'b11: out = d;
            default: out = 8'b0;
        endcase
    end
endmodule
"""

COMPLEX_MODULE = """
module processor (
    input wire clk,
    input wire rst,
    input wire [7:0] data_in,
    input wire valid,
    output reg [7:0] data_out,
    output reg done
);
    reg [7:0] buffer;
    reg [3:0] state;
    wire [7:0] processed;
    
    assign processed = data_in + buffer;
    
    always_ff @(posedge clk or negedge rst) begin
        if (!rst) begin
            buffer <= 8'b0;
            state <= 4'b0;
            data_out <= 8'b0;
            done <= 1'b0;
        end else begin
            if (valid) begin
                buffer <= data_in;
                state <= state + 1;
            end
            
            if (state > 4) begin
                data_out <= processed;
                done <= 1'b1;
            end else begin
                done <= 1'b0;
            end
        end
    end
    
    always_comb begin
        if (state == 0) begin
            // Initial state
        end
    end
endmodule
"""


class TestVerilogParser:
    """Tests for Verilog parser."""
    
    def test_parse_simple_module(self):
        """Test parsing a simple module."""
        parser = VerilogParser()
        module = parser.parse(SIMPLE_MODULE)
        
        assert module.name == "simple_adder"
        assert len(module.assignments) > 0
        
    def test_parse_sequential_module(self):
        """Test parsing a sequential module."""
        parser = VerilogParser()
        module = parser.parse(SEQUENTIAL_MODULE)
        
        assert module.name == "counter"
        assert len(module.always_blocks) > 0
        assert len(module.if_statements) > 0
        
    def test_parse_combinational_module(self):
        """Test parsing a combinational module."""
        parser = VerilogParser()
        module = parser.parse(COMBINATIONAL_MODULE)
        
        assert module.name == "mux4"
        assert len(module.case_statements) > 0
        
    def test_parse_complex_module(self):
        """Test parsing a complex module."""
        parser = VerilogParser()
        module = parser.parse(COMPLEX_MODULE)
        
        assert module.name == "processor"
        assert len(module.always_blocks) >= 2
        assert len(module.if_statements) > 0
        assert len(module.assignments) > 0


class TestFilters:
    """Tests for filtering functions."""
    
    def test_filter_has_module(self):
        """Test module filter."""
        parser = VerilogParser()
        module = parser.parse(SIMPLE_MODULE)
        
        assert filter_has_module(module) == True
        
    def test_filter_has_always_blocks(self):
        """Test always block filter."""
        parser = VerilogParser()
        
        seq_module = parser.parse(SEQUENTIAL_MODULE)
        simple_module = parser.parse(SIMPLE_MODULE)
        
        assert filter_has_always_blocks(seq_module) == True
        assert filter_has_always_blocks(simple_module) == False
        
    def test_filter_has_conditionals(self):
        """Test conditional filter."""
        parser = VerilogParser()
        module = parser.parse(SEQUENTIAL_MODULE)
        
        assert filter_has_conditionals(module) == True
        
    def test_filter_has_if_else(self):
        """Test if-else filter."""
        parser = VerilogParser()
        module = parser.parse(SEQUENTIAL_MODULE)
        
        assert filter_has_if_else(module) == True
        
    def test_filter_has_operators(self):
        """Test operator filter."""
        parser = VerilogParser()
        module = parser.parse(SIMPLE_MODULE)
        
        assert filter_has_operators(module) == True
        
    def test_apply_multiple_filters(self):
        """Test applying multiple filters."""
        parser = VerilogParser()
        module = parser.parse(SEQUENTIAL_MODULE)
        
        # Module has: always blocks, conditionals, assignments
        assert apply_filters(module, [1, 2, 6, 8]) == True


class TestModifications:
    """Tests for modification functions."""
    
    def test_invert_if_else(self):
        """Test if-else inversion."""
        parser = VerilogParser()
        module = parser.parse(SEQUENTIAL_MODULE)
        
        result = invert_if_else(module, likelihood=1.0)
        
        # Should succeed since module has if-else
        assert result.success == True
        assert result.modification_type == "invert_if_else"
        assert result.modified_code != module.source_code
        
    def test_remove_else_branch(self):
        """Test else branch removal."""
        parser = VerilogParser()
        module = parser.parse(SEQUENTIAL_MODULE)
        
        result = remove_else_branch(module, likelihood=1.0)
        
        assert result.success == True
        assert result.modification_type == "remove_else_branch"
        # Check that 'else' is removed
        if result.success:
            # The else should be reduced
            assert result.modified_code.count('else') < module.source_code.count('else')
        
    def test_change_constant(self):
        """Test constant modification."""
        parser = VerilogParser()
        module = parser.parse(SEQUENTIAL_MODULE)
        
        result = change_constant(module, likelihood=1.0)
        
        assert result.success == True
        assert result.modification_type == "change_constant"
        
    def test_change_operator(self):
        """Test operator modification."""
        parser = VerilogParser()
        module = parser.parse(SIMPLE_MODULE)
        
        result = change_operator(module, likelihood=1.0)
        
        # May or may not succeed depending on operator patterns
        if result.success:
            assert result.modification_type == "change_operator"
            
    def test_swap_blocking_nonblocking(self):
        """Test blocking/non-blocking swap."""
        parser = VerilogParser()
        module = parser.parse(SEQUENTIAL_MODULE)
        
        result = swap_blocking_nonblocking(module, likelihood=1.0)
        
        assert result.success == True
        assert result.modification_type == "swap_blocking_nonblocking"
        
    def test_get_applicable_modifications(self):
        """Test getting applicable modifications."""
        parser = VerilogParser()
        module = parser.parse(COMPLEX_MODULE)
        
        applicable = get_applicable_modifications(module)
        
        assert len(applicable) > 0
        # Complex module should have many applicable modifications
        
    def test_apply_random_modification(self):
        """Test applying a random modification."""
        parser = VerilogParser()
        module = parser.parse(COMPLEX_MODULE)
        
        result = apply_random_modification(module, likelihood=1.0)
        
        assert result.success == True
        assert result.modified_code != module.source_code


class TestComplexity:
    """Tests for complexity calculation."""
    
    def test_simple_module_complexity(self):
        """Test complexity of simple module."""
        parser = VerilogParser()
        module = parser.parse(SIMPLE_MODULE)
        
        complexity = calculate_complexity(module)
        
        # Simple module should have low complexity
        assert complexity < 5
        
    def test_complex_module_complexity(self):
        """Test complexity of complex module."""
        parser = VerilogParser()
        module = parser.parse(COMPLEX_MODULE)
        
        complexity = calculate_complexity(module)
        
        # Complex module should have higher complexity
        assert complexity > 5


def run_tests():
    """Run all tests and print results."""
    import traceback
    
    test_classes = [
        TestVerilogParser,
        TestFilters,
        TestModifications,
        TestComplexity,
    ]
    
    total_tests = 0
    passed_tests = 0
    failed_tests = []
    
    for test_class in test_classes:
        print(f"\n{'='*60}")
        print(f"Running {test_class.__name__}")
        print('='*60)
        
        instance = test_class()
        
        for method_name in dir(instance):
            if method_name.startswith('test_'):
                total_tests += 1
                try:
                    getattr(instance, method_name)()
                    print(f"  ✓ {method_name}")
                    passed_tests += 1
                except Exception as e:
                    print(f"  ✗ {method_name}")
                    print(f"    Error: {e}")
                    failed_tests.append((method_name, str(e)))
    
    print(f"\n{'='*60}")
    print("Test Summary")
    print('='*60)
    print(f"Total: {total_tests}")
    print(f"Passed: {passed_tests}")
    print(f"Failed: {len(failed_tests)}")
    
    if failed_tests:
        print(f"\nFailed tests:")
        for name, error in failed_tests:
            print(f"  - {name}: {error}")
    
    return len(failed_tests) == 0


if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1)

