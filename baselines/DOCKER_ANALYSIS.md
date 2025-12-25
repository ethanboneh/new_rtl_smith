# Docker Requirement Analysis

## What the Test Harness Actually Does

The test harness runs **open-source verification tools**:
- **Icarus Verilog (iverilog)**: Compiles and simulates your RTL code
- **Cocotb**: Python-based testbench framework
- **pytest**: Runs the test suite

These tools are packaged in a Docker image (`ghcr.io/hdl/sim/osvb`) because:
1. They're not installed on your system
2. Docker ensures consistent, reproducible environments
3. CVDP code is designed to use Docker (no "host mode" implemented)

**Docker is just packaging free, open-source tools** - not commercial EDA software.

## Summary

**Docker IS required** for running CVDP benchmark evaluations, even for non-commercial datasets. However, you don't need commercial EDA tool licenses.

## What Docker is Used For

### 1. Test Harness Execution (Required)
- **All CVDP evaluations** (including non-commercial) use Docker to run the test harness
- The test harness runs in Docker containers to ensure:
  - Consistent environment across different systems
  - Isolation of test runs
  - Reproducible results

### 2. Commercial EDA Tools (NOT needed for your setup)
- Only required for datasets with:
  - Categories 12, 13, 14 (VERIF_EDA_CATEGORIES)
  - `__VERIF_EDA_IMAGE__` template variables
- Your dataset (`cvdp_v1.0.2_nonagentic_code_generation_no_commercial.jsonl`) does **NOT** require this
- Verified: `requires_commercial_eda_tools()` returns `False` for your dataset

### 3. Agentic Workflows (NOT used in baselines)
- Only needed if running agent-based evaluations
- Your baselines project uses non-agentic evaluation

## Your Specific Setup

✅ **Dataset**: `cvdp_v1.0.2_nonagentic_code_generation_no_commercial.jsonl`
- Does NOT require commercial EDA tools
- Does NOT require license networks
- Does NOT have `__VERIF_EDA_IMAGE__` templates

❌ **Docker Still Required**: For test harness execution

## Options to Fix Docker Access

### Option 1: Add User to Docker Group (Recommended)
```bash
# Add yourself to docker group
sudo usermod -aG docker $USER

# Log out and log back in (or restart your session)
# Then verify:
docker ps
```

**Pros**: Clean, secure, no sudo needed
**Cons**: Requires logout/login

### Option 2: Use Sudo (Quick Fix)
```bash
# Use sudo for docker commands
sudo docker ps

# The CVDP benchmark will need sudo access
# You may need to modify scripts or use sudo when running evaluations
```

**Pros**: Works immediately
**Cons**: Less secure, requires sudo for all docker commands

### Option 3: Podman (Alternative Container Runtime)
If Docker is problematic, you could potentially use Podman (Docker-compatible alternative), but this would require modifying the CVDP benchmark code to use `podman` instead of `docker`.

### Option 4: Wait for Host Mode (Not Available)
The CVDP benchmark has a `--host` flag mentioned in the code, but it's marked as "not currently implemented" in `argparse_common.py`. This would allow running without Docker, but it's not available yet.

## Verification

You can verify your dataset doesn't need commercial EDA tools:
```bash
cd /afs/cs.stanford.edu/u/ethanboneh/new_rtl_smith/cvdp_benchmark
python3 -c "from src.commercial_eda import requires_commercial_eda_tools; \
    result = requires_commercial_eda_tools('example_dataset/cvdp_v1.0.2_nonagentic_code_generation_no_commercial.jsonl'); \
    print('Requires commercial EDA:', result)"
# Output: Requires commercial EDA: False
```

## Recommendation

**Fix Docker permissions** using Option 1 (add to docker group). This is the cleanest solution and will work seamlessly with the CVDP benchmark.

The Docker requirement is minimal - it's just for running test harnesses in isolated containers, not for heavy commercial EDA tools.

