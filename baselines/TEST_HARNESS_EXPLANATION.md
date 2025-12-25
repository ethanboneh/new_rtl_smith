# What the Test Harness Does (And Why Docker)

## What the Test Harness Actually Runs

The test harness uses **open-source verification tools** to test your generated RTL code:

### Tools Used:
1. **Icarus Verilog (iverilog)** - Open-source Verilog compiler/simulator
   - Compiles your RTL code
   - Runs simulation
   - Currently **NOT installed** on your system

2. **Cocotb** - Python-based verification framework
   - Writes testbenches in Python
   - Drives the simulation
   - Currently **NOT installed** on your system

3. **pytest** - Python testing framework
   - Runs the test suite
   - Reports pass/fail
   - Currently **NOT installed** on your system

### What Happens:
1. Your generated RTL code is placed in `/code/rtl/`
2. A testbench (written in Python using cocotb) is in `/src/test_*.py`
3. The Docker container runs:
   ```bash
   pytest -s /src/test_runner.py
   ```
4. This compiles your RTL with iverilog, runs the simulation, and checks if outputs match expected values

## Why Docker is Required

Docker packages all these tools together in a pre-built image (`ghcr.io/hdl/sim/osvb`):
- ✅ Icarus Verilog (iverilog)
- ✅ Cocotb Python framework
- ✅ pytest
- ✅ All dependencies and correct versions

**Without Docker**, you would need to:
- Install iverilog (and its dependencies)
- Install cocotb (Python package with C extensions)
- Install pytest
- Ensure all versions are compatible
- Set up the environment correctly

## Could You Run Without Docker?

**Theoretically yes, but practically no:**

1. **CVDP code is hardcoded for Docker**: The `repository.py` code always generates Docker commands - there's no "host mode" implementation (it's marked as TODO in the code)

2. **You'd need to install tools locally**:
   ```bash
   # Install iverilog
   sudo apt-get install iverilog  # or build from source
   
   # Install cocotb
   pip install cocotb
   
   # Install pytest
   pip install pytest
   ```

3. **You'd need to modify CVDP code** to bypass Docker and run commands directly

4. **Environment consistency**: Docker ensures everyone gets the same results regardless of their system

## The Docker Image

The image used is: `ghcr.io/hdl/sim/osvb` (Open Source Verification Build)
- This is a public image from GitHub Container Registry
- Contains all verification tools pre-configured
- No commercial licenses needed
- Free and open-source

## Summary

**Docker is required because:**
- The test harness needs iverilog, cocotb, and pytest
- These tools aren't installed on your system
- CVDP benchmark code is designed to use Docker for consistency
- Docker packages everything in a ready-to-use container

**It's NOT required for:**
- Commercial EDA tools (you're using open-source tools)
- License servers
- Special hardware

**Bottom line**: Docker is just a convenient way to package the open-source verification tools. The actual verification uses free, open-source software (iverilog + cocotb), not commercial tools.

