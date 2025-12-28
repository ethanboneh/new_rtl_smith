#!/usr/bin/env python3
"""
Testbench runner for verifying Verilog corruptions.

Supports multiple simulators:
- iverilog (Icarus Verilog) - free, open source
- verilator - free, open source, fast
- vcs - commercial (Synopsys)

The runner compiles and simulates the design with its testbench,
then parses the output to determine pass/fail status.
"""

import os
import re
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Optional, Tuple


class SimulatorType(Enum):
    IVERILOG = "iverilog"
    VERILATOR = "verilator"
    VCS = "vcs"


@dataclass
class TestbenchResult:
    """Result of running a testbench."""
    passed: bool
    error_count: int
    total_tests: int
    output: str
    simulator: str
    compile_success: bool
    runtime_error: bool
    error_message: Optional[str] = None


def find_available_simulator() -> Optional[SimulatorType]:
    """Find an available Verilog simulator."""
    # Check in order of preference
    if shutil.which("iverilog"):
        return SimulatorType.IVERILOG
    if shutil.which("verilator"):
        return SimulatorType.VERILATOR
    if shutil.which("vcs"):
        return SimulatorType.VCS
    return None


def run_iverilog(
    design_code: str,
    testbench_code: str,
    module_name: str,
    timeout: int = 30,
    work_dir: Optional[str] = None,
) -> TestbenchResult:
    """
    Run testbench using Icarus Verilog.
    
    Args:
        design_code: The Verilog design code
        testbench_code: The testbench code
        module_name: Name of the design module (used for filename)
        timeout: Simulation timeout in seconds
        work_dir: Working directory (temp dir if None)
    
    Returns:
        TestbenchResult with pass/fail status and details
    """
    cleanup = work_dir is None
    if work_dir is None:
        work_dir = tempfile.mkdtemp(prefix="rtllm_test_")
    
    try:
        work_path = Path(work_dir)
        
        # Write design file
        design_file = work_path / f"{module_name}.v"
        design_file.write_text(design_code)
        
        # Write testbench file
        tb_file = work_path / "testbench.v"
        tb_file.write_text(testbench_code)
        
        # Compile with iverilog
        output_file = work_path / "sim.out"
        compile_cmd = [
            "iverilog",
            "-o", str(output_file),
            "-g2012",  # SystemVerilog 2012 support
            str(design_file),
            str(tb_file)
        ]
        
        try:
            compile_result = subprocess.run(
                compile_cmd,
                capture_output=True,
                text=True,
                timeout=timeout,
                cwd=work_dir
            )
        except subprocess.TimeoutExpired:
            return TestbenchResult(
                passed=False,
                error_count=0,
                total_tests=0,
                output="",
                simulator="iverilog",
                compile_success=False,
                runtime_error=True,
                error_message="Compilation timed out"
            )
        
        if compile_result.returncode != 0:
            return TestbenchResult(
                passed=False,
                error_count=0,
                total_tests=0,
                output=compile_result.stderr,
                simulator="iverilog",
                compile_success=False,
                runtime_error=False,
                error_message=f"Compilation failed: {compile_result.stderr}"
            )
        
        # Run simulation
        try:
            sim_result = subprocess.run(
                ["vvp", str(output_file)],
                capture_output=True,
                text=True,
                timeout=timeout,
                cwd=work_dir
            )
        except subprocess.TimeoutExpired:
            return TestbenchResult(
                passed=False,
                error_count=0,
                total_tests=0,
                output="",
                simulator="iverilog",
                compile_success=True,
                runtime_error=True,
                error_message="Simulation timed out"
            )
        
        output = sim_result.stdout + sim_result.stderr
        
        # Parse results - look for RTLLM testbench patterns
        return parse_testbench_output(output, "iverilog")
        
    finally:
        if cleanup and work_dir:
            shutil.rmtree(work_dir, ignore_errors=True)


def run_verilator(
    design_code: str,
    testbench_code: str,
    module_name: str,
    timeout: int = 30,
    work_dir: Optional[str] = None,
) -> TestbenchResult:
    """
    Run testbench using Verilator.
    
    Note: Verilator requires C++ testbenches, so this creates a simple wrapper.
    For RTLLM-style Verilog testbenches, iverilog is preferred.
    """
    cleanup = work_dir is None
    if work_dir is None:
        work_dir = tempfile.mkdtemp(prefix="rtllm_test_")
    
    try:
        work_path = Path(work_dir)
        
        # Write design file
        design_file = work_path / f"{module_name}.v"
        design_file.write_text(design_code)
        
        # Write testbench file
        tb_file = work_path / "testbench.v"
        tb_file.write_text(testbench_code)
        
        # Try to use verilator with --binary option (newer versions)
        # Fall back to lint-only mode
        compile_cmd = [
            "verilator",
            "--binary",
            "-j", "0",
            "--timing",
            "-Wno-fatal",
            "-o", "sim",
            str(design_file),
            str(tb_file)
        ]
        
        try:
            compile_result = subprocess.run(
                compile_cmd,
                capture_output=True,
                text=True,
                timeout=timeout,
                cwd=work_dir
            )
        except subprocess.TimeoutExpired:
            return TestbenchResult(
                passed=False,
                error_count=0,
                total_tests=0,
                output="",
                simulator="verilator",
                compile_success=False,
                runtime_error=True,
                error_message="Compilation timed out"
            )
        
        if compile_result.returncode != 0:
            # Verilator may not support the testbench style
            return TestbenchResult(
                passed=False,
                error_count=0,
                total_tests=0,
                output=compile_result.stderr,
                simulator="verilator",
                compile_success=False,
                runtime_error=False,
                error_message=f"Verilator compilation failed (may not support this testbench style)"
            )
        
        # Run simulation
        sim_exe = work_path / "obj_dir" / "sim"
        if not sim_exe.exists():
            sim_exe = work_path / "sim"
        
        if not sim_exe.exists():
            return TestbenchResult(
                passed=False,
                error_count=0,
                total_tests=0,
                output="",
                simulator="verilator",
                compile_success=False,
                runtime_error=False,
                error_message="Simulation executable not found"
            )
        
        try:
            sim_result = subprocess.run(
                [str(sim_exe)],
                capture_output=True,
                text=True,
                timeout=timeout,
                cwd=work_dir
            )
        except subprocess.TimeoutExpired:
            return TestbenchResult(
                passed=False,
                error_count=0,
                total_tests=0,
                output="",
                simulator="verilator",
                compile_success=True,
                runtime_error=True,
                error_message="Simulation timed out"
            )
        
        output = sim_result.stdout + sim_result.stderr
        return parse_testbench_output(output, "verilator")
        
    finally:
        if cleanup and work_dir:
            shutil.rmtree(work_dir, ignore_errors=True)


def run_vcs(
    design_code: str,
    testbench_code: str,
    module_name: str,
    timeout: int = 60,
    work_dir: Optional[str] = None,
) -> TestbenchResult:
    """Run testbench using VCS (Synopsys)."""
    cleanup = work_dir is None
    if work_dir is None:
        work_dir = tempfile.mkdtemp(prefix="rtllm_test_")
    
    try:
        work_path = Path(work_dir)
        
        # Write design file
        design_file = work_path / f"{module_name}.v"
        design_file.write_text(design_code)
        
        # Write testbench file
        tb_file = work_path / "testbench.v"
        tb_file.write_text(testbench_code)
        
        # Compile with VCS
        compile_cmd = [
            "vcs",
            "-sverilog",
            "+v2k",
            "-timescale=1ns/1ns",
            "-o", "simv",
            str(design_file),
            str(tb_file)
        ]
        
        try:
            compile_result = subprocess.run(
                compile_cmd,
                capture_output=True,
                text=True,
                timeout=timeout,
                cwd=work_dir
            )
        except subprocess.TimeoutExpired:
            return TestbenchResult(
                passed=False,
                error_count=0,
                total_tests=0,
                output="",
                simulator="vcs",
                compile_success=False,
                runtime_error=True,
                error_message="Compilation timed out"
            )
        
        if compile_result.returncode != 0:
            return TestbenchResult(
                passed=False,
                error_count=0,
                total_tests=0,
                output=compile_result.stderr,
                simulator="vcs",
                compile_success=False,
                runtime_error=False,
                error_message=f"VCS compilation failed: {compile_result.stderr}"
            )
        
        # Run simulation
        try:
            sim_result = subprocess.run(
                ["./simv"],
                capture_output=True,
                text=True,
                timeout=timeout,
                cwd=work_dir
            )
        except subprocess.TimeoutExpired:
            return TestbenchResult(
                passed=False,
                error_count=0,
                total_tests=0,
                output="",
                simulator="vcs",
                compile_success=True,
                runtime_error=True,
                error_message="Simulation timed out"
            )
        
        output = sim_result.stdout + sim_result.stderr
        return parse_testbench_output(output, "vcs")
        
    finally:
        if cleanup and work_dir:
            shutil.rmtree(work_dir, ignore_errors=True)


def parse_testbench_output(output: str, simulator: str) -> TestbenchResult:
    """
    Parse testbench output to determine pass/fail status.
    
    RTLLM testbenches typically output:
    - "===========Your Design Passed===========" for pass
    - "===========Test completed with X /Y failures===========" for fail
    """
    # Check for pass message
    if "Your Design Passed" in output:
        return TestbenchResult(
            passed=True,
            error_count=0,
            total_tests=100,  # RTLLM default
            output=output,
            simulator=simulator,
            compile_success=True,
            runtime_error=False
        )
    
    # Check for failure message with count
    fail_match = re.search(r'Test completed with (\d+)\s*/\s*(\d+) failures', output)
    if fail_match:
        error_count = int(fail_match.group(1))
        total_tests = int(fail_match.group(2))
        return TestbenchResult(
            passed=False,
            error_count=error_count,
            total_tests=total_tests,
            output=output,
            simulator=simulator,
            compile_success=True,
            runtime_error=False
        )
    
    # Check for generic failure patterns
    if "FAILED" in output.upper() or "ERROR" in output.upper():
        return TestbenchResult(
            passed=False,
            error_count=1,
            total_tests=1,
            output=output,
            simulator=simulator,
            compile_success=True,
            runtime_error=False
        )
    
    # Check for generic pass patterns
    if "PASSED" in output.upper() or "SUCCESS" in output.upper():
        return TestbenchResult(
            passed=True,
            error_count=0,
            total_tests=1,
            output=output,
            simulator=simulator,
            compile_success=True,
            runtime_error=False
        )
    
    # Unknown - simulation completed but unclear result
    return TestbenchResult(
        passed=False,
        error_count=0,
        total_tests=0,
        output=output,
        simulator=simulator,
        compile_success=True,
        runtime_error=False,
        error_message="Could not determine pass/fail from output"
    )


def run_testbench(
    design_code: str,
    testbench_code: str,
    module_name: str,
    simulator: Optional[SimulatorType] = None,
    timeout: int = 30,
    work_dir: Optional[str] = None,
) -> TestbenchResult:
    """
    Run a testbench against a design using the specified or available simulator.
    
    Args:
        design_code: The Verilog design code to test
        testbench_code: The testbench code
        module_name: Name of the design module
        simulator: Specific simulator to use (auto-detect if None)
        timeout: Simulation timeout in seconds
        work_dir: Working directory (temp dir if None)
    
    Returns:
        TestbenchResult with pass/fail status and details
    """
    if simulator is None:
        simulator = find_available_simulator()
    
    if simulator is None:
        return TestbenchResult(
            passed=False,
            error_count=0,
            total_tests=0,
            output="",
            simulator="none",
            compile_success=False,
            runtime_error=False,
            error_message="No Verilog simulator found. Install iverilog, verilator, or vcs."
        )
    
    if simulator == SimulatorType.IVERILOG:
        return run_iverilog(design_code, testbench_code, module_name, timeout, work_dir)
    elif simulator == SimulatorType.VERILATOR:
        return run_verilator(design_code, testbench_code, module_name, timeout, work_dir)
    elif simulator == SimulatorType.VCS:
        return run_vcs(design_code, testbench_code, module_name, timeout, work_dir)
    else:
        return TestbenchResult(
            passed=False,
            error_count=0,
            total_tests=0,
            output="",
            simulator=str(simulator),
            compile_success=False,
            runtime_error=False,
            error_message=f"Unknown simulator: {simulator}"
        )


def verify_corruption(
    clean_code: str,
    corrupt_code: str,
    testbench_code: str,
    module_name: str,
    simulator: Optional[SimulatorType] = None,
    timeout: int = 30,
) -> Tuple[TestbenchResult, TestbenchResult, bool]:
    """
    Verify that a corruption is valid by checking:
    1. Clean code passes the testbench
    2. Corrupt code fails the testbench
    
    Args:
        clean_code: Original correct code
        corrupt_code: Corrupted code
        testbench_code: Testbench code
        module_name: Module name
        simulator: Simulator to use
        timeout: Simulation timeout
    
    Returns:
        Tuple of (clean_result, corrupt_result, is_valid_corruption)
        is_valid_corruption is True if clean passes and corrupt fails
    """
    # Test clean code
    clean_result = run_testbench(
        clean_code, testbench_code, module_name, simulator, timeout
    )
    
    # Test corrupt code
    corrupt_result = run_testbench(
        corrupt_code, testbench_code, module_name, simulator, timeout
    )
    
    # Valid corruption: clean passes, corrupt fails
    is_valid = clean_result.passed and not corrupt_result.passed
    
    return clean_result, corrupt_result, is_valid


if __name__ == "__main__":
    # Test the module
    print(f"Available simulator: {find_available_simulator()}")
    
    # Simple test
    design = """
module adder(input [7:0] a, b, output [7:0] sum);
    assign sum = a + b;
endmodule
"""
    
    testbench = """
module testbench;
    reg [7:0] a, b;
    wire [7:0] sum;
    integer error = 0;
    integer i;
    
    adder uut(.a(a), .b(b), .sum(sum));
    
    initial begin
        for (i = 0; i < 10; i = i + 1) begin
            a = $random;
            b = $random;
            #10;
            if (sum !== a + b) error = error + 1;
        end
        if (error == 0)
            $display("===========Your Design Passed===========");
        else
            $display("===========Test completed with %d /10 failures===========", error);
        $finish;
    end
endmodule
"""
    
    result = run_testbench(design, testbench, "adder")
    print(f"Result: passed={result.passed}, errors={result.error_count}")
    print(f"Output: {result.output[:200] if result.output else 'N/A'}")

