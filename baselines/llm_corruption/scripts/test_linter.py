#!/usr/bin/env python3
"""
Test script for the Verilator linter integration.

This script tests the linter with sample SystemVerilog code.
"""

import sys
import os

# Add scripts directory to path
sys.path.insert(0, os.path.dirname(__file__))

from linter import VerilatorLinter


# Sample clean code
CLEAN_CODE = """
module counter (
    input logic clk,
    input logic rst,
    output logic [7:0] count
);
    always_ff @(posedge clk or posedge rst) begin
        if (rst) begin
            count <= 8'b0;
        end else begin
            count <= count + 1;
        end
    end
endmodule
"""

# Sample corrupt code (introduces latch)
CORRUPT_CODE = """
module counter (
    input logic clk,
    input logic rst,
    output logic [7:0] count
);
    always_comb begin
        if (rst) begin
            count = 8'b0;
        end
        // Missing else - will infer latch!
    end
endmodule
"""


def main():
    print("Testing Verilator Linter Integration\n")
    print("=" * 60)
    
    try:
        linter = VerilatorLinter()
    except RuntimeError as e:
        print(f"ERROR: {e}")
        print("\nPlease install Verilator:")
        print("  Ubuntu/Debian: sudo apt-get install verilator")
        print("  macOS: brew install verilator")
        sys.exit(1)
    
    print("\n1. Testing clean code...")
    clean_result = linter.lint_code(CLEAN_CODE, module_name="counter")
    print(f"   Errors: {len(clean_result['errors'])}")
    print(f"   Warnings: {len(clean_result['warnings'])}")
    print(f"   Violated rules: {clean_result['violated_rules']}")
    
    print("\n2. Testing corrupt code...")
    corrupt_result = linter.lint_code(CORRUPT_CODE, module_name="counter")
    print(f"   Errors: {len(corrupt_result['errors'])}")
    print(f"   Warnings: {len(corrupt_result['warnings'])}")
    print(f"   Violated rules: {corrupt_result['violated_rules']}")
    
    if corrupt_result['warnings']:
        print("\n   Warnings:")
        for warn in corrupt_result['warnings']:
            print(f"     - {warn}")
    
    print("\n3. Verifying corruption...")
    verification = linter.verify_corruption(CLEAN_CODE, CORRUPT_CODE)
    print(f"   Is valid: {verification['is_valid']}")
    print(f"   New violated rules: {verification['new_violated_rules']}")
    
    if verification['is_valid']:
        print("\n✓ Linter integration working correctly!")
    else:
        print("\n⚠ Corruption did not introduce new violations")
        print("  (This might be expected depending on Verilator version)")


if __name__ == "__main__":
    main()

