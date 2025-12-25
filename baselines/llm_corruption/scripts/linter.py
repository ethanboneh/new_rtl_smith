#!/usr/bin/env python3
"""
Verilator linter integration for verifying corrupt code.

This script runs Verilator linting on SystemVerilog code and extracts
which lint rules are violated.
"""

import subprocess
import re
import json
import tempfile
import os
from typing import Dict, List, Optional, Tuple
from pathlib import Path


class VerilatorLinter:
    """Wrapper for Verilator linting functionality."""
    
    def __init__(self, verilator_bin: Optional[str] = None):
        """
        Initialize the Verilator linter.
        
        Args:
            verilator_bin: Path to verilator binary. If None, uses 'verilator' from PATH.
        """
        self.verilator_bin = verilator_bin or "verilator"
        self._check_verilator()
    
    def _check_verilator(self):
        """Check if Verilator is available."""
        try:
            result = subprocess.run(
                [self.verilator_bin, "--version"],
                capture_output=True,
                text=True,
                timeout=10
            )
            if result.returncode != 0:
                raise RuntimeError(f"Verilator not found or not working: {result.stderr}")
            print(f"Using Verilator: {result.stdout.strip()}")
        except FileNotFoundError:
            raise RuntimeError(
                f"Verilator not found. Please install Verilator:\n"
                f"  Ubuntu/Debian: sudo apt-get install verilator\n"
                f"  macOS: brew install verilator\n"
                f"  Or build from source: https://github.com/verilator/verilator"
            )
    
    def lint_code(
        self,
        code: str,
        module_name: Optional[str] = None,
        extra_flags: Optional[List[str]] = None
    ) -> Dict:
        """
        Lint SystemVerilog code and return results.
        
        Args:
            code: SystemVerilog code to lint
            module_name: Optional module name (extracted from code if not provided)
            extra_flags: Additional Verilator flags
            
        Returns:
            Dictionary with:
            - 'has_errors': bool
            - 'has_warnings': bool
            - 'errors': List of error messages
            - 'warnings': List of warning messages
            - 'violated_rules': Set of lint rule codes (e.g., 'WIDTH', 'LATCH', 'BLKSEQ')
            - 'exit_code': int
        """
        # Extract module name if not provided
        if module_name is None:
            match = re.search(r'module\s+(\w+)', code)
            module_name = match.group(1) if match else "test_module"
        
        # Create temporary file
        with tempfile.NamedTemporaryFile(
            mode='w',
            suffix='.sv',
            delete=False,
            prefix=f'{module_name}_'
        ) as f:
            f.write(code)
            temp_file = f.name
        
        try:
            # Build Verilator command
            # Note: Older Verilator versions (like 4.034) may not support all -Wwarn-* flags
            # We'll try with -Wall first, which should enable most warnings
            cmd = [
                self.verilator_bin,
                "--lint-only",
                "--sv",
                "-Wall",  # Enable all warnings
                "-Wno-fatal",  # Don't stop on warnings
                temp_file
            ]
            
            # Try to add specific warning flags (may not work in older versions)
            # These are optional and will be ignored if not supported
            try:
                # Test if verilator supports these flags by checking version
                version_result = subprocess.run(
                    [self.verilator_bin, "--version"],
                    capture_output=True,
                    text=True,
                    timeout=5
                )
                # If version is 5.0+, add specific warning flags
                if "5." in version_result.stdout or "6." in version_result.stdout:
                    cmd.insert(-1, "-Wwarn-BLKSEQ")
                    cmd.insert(-1, "-Wwarn-LATCH")
                    cmd.insert(-1, "-Wwarn-WIDTH")
                    cmd.insert(-1, "-Wwarn-MULTIDRIVEN")
            except:
                pass  # Ignore if version check fails
            
            if extra_flags:
                cmd.extend(extra_flags)
            
            # Run Verilator
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=30
            )
            
            # Parse output
            errors, warnings = self._parse_output(result.stdout, result.stderr)
            violated_rules = self._extract_rules(errors + warnings)
            
            # Debug: if no violations found but exit code suggests issues, log the raw output
            # Also check if there are any warnings/errors at all, even if we can't parse the codes
            if len(violated_rules) == 0:
                output_text = result.stdout + result.stderr
                # Check if there are any warning/error indicators even if we didn't parse them
                if "%Warning" in output_text or "%Error" in output_text:
                    # We have warnings but couldn't parse them - try a more lenient approach
                    # Extract any word after Warning- or Error- as a potential rule code
                    lenient_pattern = r'%((?:Error|Warning))-(\w+)'
                    for match in re.finditer(lenient_pattern, output_text):
                        code = match.group(2)
                        if code not in ['ALL', 'FATAL']:  # Skip meta-warnings
                            violated_rules.add(code)
                            if match.group(1) == "Error":
                                errors.append(f"{code}: (parsed from output)")
                            else:
                                warnings.append(f"{code}: (parsed from output)")
            
            return {
                'has_errors': len(errors) > 0,
                'has_warnings': len(warnings) > 0,
                'errors': errors,
                'warnings': warnings,
                'violated_rules': violated_rules,
                'exit_code': result.returncode,
                'stdout': result.stdout,
                'stderr': result.stderr
            }
        finally:
            # Clean up
            if os.path.exists(temp_file):
                os.unlink(temp_file)
    
    def _parse_output(self, stdout: str, stderr: str) -> Tuple[List[str], List[str]]:
        """Parse Verilator output into errors and warnings."""
        errors = []
        warnings = []
        
        # Combine stdout and stderr
        output = stdout + stderr
        
        # Debug: print output if no matches found
        if not output.strip():
            return errors, warnings
        
        # Pattern for Verilator messages
        # Format: %Warning-CODE: file:line: message
        # Format: %Error-CODE: file:line: message
        # Also handle: %Warning[CODE]: file:line: message (alternative format)
        patterns = [
            r'%((?:Error|Warning))-(\w+):\s+[^:]+:\d+:\s+(.+)',  # Standard format
            r'%((?:Error|Warning))\[(\w+)\]:\s+[^:]+:\d+:\s+(.+)',  # Alternative format
            r'%((?:Error|Warning))-(\w+)\s+[^:]+:\d+:\s+(.+)',  # Space after dash
        ]
        
        for pattern in patterns:
            for match in re.finditer(pattern, output, re.MULTILINE):
                msg_type = match.group(1)
                code = match.group(2)
                message = match.group(3).strip()
                
                full_message = f"{code}: {message}"
                
                if msg_type == "Error":
                    errors.append(full_message)
                else:
                    warnings.append(full_message)
        
        # Remove duplicates while preserving order
        errors = list(dict.fromkeys(errors))
        warnings = list(dict.fromkeys(warnings))
        
        return errors, warnings
    
    def _extract_rules(self, messages: List[str]) -> set:
        """Extract lint rule codes from messages."""
        rules = set()
        
        # Common Verilator warning/error codes
        # Format: CODE: message
        pattern = r'^(\w+):'
        
        for msg in messages:
            match = re.match(pattern, msg)
            if match:
                rules.add(match.group(1))
        
        return rules
    
    def verify_corruption(
        self,
        clean_code: str,
        corrupt_code: str,
        require_new_violations: bool = True
    ) -> Dict:
        """
        Verify that corrupt code introduces new lint violations.
        
        Args:
            clean_code: Original clean code
            corrupt_code: Corrupted code
            require_new_violations: If True, requires that corrupt code has violations
                                   that clean code doesn't have
            
        Returns:
            Dictionary with verification results
        """
        clean_result = self.lint_code(clean_code)
        corrupt_result = self.lint_code(corrupt_code)
        
        clean_rules = clean_result['violated_rules']
        corrupt_rules = corrupt_result['violated_rules']
        
        new_rules = corrupt_rules - clean_rules
        removed_rules = clean_rules - corrupt_rules
        
        is_valid = True
        if require_new_violations:
            # Require at least one new violation
            is_valid = len(new_rules) > 0
        
        return {
            'is_valid': is_valid,
            'clean_has_errors': clean_result['has_errors'],
            'clean_has_warnings': clean_result['has_warnings'],
            'corrupt_has_errors': corrupt_result['has_errors'],
            'corrupt_has_warnings': corrupt_result['has_warnings'],
            'clean_violated_rules': clean_rules,
            'corrupt_violated_rules': corrupt_rules,
            'new_violated_rules': new_rules,
            'removed_violated_rules': removed_rules,
            'clean_result': clean_result,
            'corrupt_result': corrupt_result
        }


def main():
    """Test the linter."""
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python linter.py <code_file.sv>")
        sys.exit(1)
    
    code_file = sys.argv[1]
    with open(code_file, 'r') as f:
        code = f.read()
    
    linter = VerilatorLinter()
    result = linter.lint_code(code)
    
    print("Lint Results:")
    print(f"  Errors: {len(result['errors'])}")
    print(f"  Warnings: {len(result['warnings'])}")
    print(f"  Violated Rules: {result['violated_rules']}")
    
    if result['errors']:
        print("\nErrors:")
        for err in result['errors']:
            print(f"  - {err}")
    
    if result['warnings']:
        print("\nWarnings:")
        for warn in result['warnings']:
            print(f"  - {warn}")


if __name__ == "__main__":
    main()

