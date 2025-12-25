# Running CVDP Without Docker (Host Mode)

## Current Situation

You've installed iverilog, cocotb, and pytest, but the CVDP benchmark code is **hardcoded to use Docker**. The `--host` flag exists in the code but is **not implemented** (it's commented out).

## Option 1: Fix Docker Permissions (Easiest - Recommended)

This is the simplest solution:

```bash
# Add yourself to docker group
sudo usermod -aG docker $USER

# Log out and log back in (or restart your session)
# Then verify:
docker ps
```

**Why this is best**: No code changes needed, works immediately, maintains consistency with CVDP's design.

## Option 2: Modify CVDP Code for Host Mode

If you really want to avoid Docker, you need to modify the CVDP code. Here's what needs to change:

### Files to Modify

1. **`src/repository.py`** - Add host mode support to `log_docker()` method
2. **`src/argparse_common.py`** - Uncomment the `--host` flag
3. **`src/dataset_processor.py`** - Pass host flag through

### Implementation Steps

#### Step 1: Uncomment the host flag

In `src/argparse_common.py`, uncomment lines 43-45:
```python
parser.add_argument("-o", "--host", action="store_true",
                   help="Run tests on host instead of Docker")
```

#### Step 2: Add host mode to log_docker()

In `src/repository.py`, modify the `log_docker()` method around line 366:

```python
def log_docker(self, docker : str = "", cmd : str = "", service : str = "", logfile : str = "", 
              monitor_size=True):
    # If host mode, run directly without Docker
    if self.host:
        return self.log_run_host(docker, cmd, service, logfile, monitor_size)
    
    # ... existing Docker code ...
```

#### Step 3: Implement log_run_host()

Add a new method to run tests directly on the host:

```python
def log_run_host(self, docker_dir: str, cmd: str, service: str, logfile: str, monitor_size=True):
    """Run test harness directly on host without Docker."""
    # Parse docker-compose.yml to get environment variables
    docker_compose_path = os.path.join(docker_dir, 'docker-compose.yml')
    
    # Read .env file
    env_file = os.path.join(docker_dir, 'src', '.env')
    env_vars = {}
    if os.path.exists(env_file):
        with open(env_file, 'r') as f:
            for line in f:
                if '=' in line and not line.strip().startswith('#'):
                    key, value = line.strip().split('=', 1)
                    env_vars[key.strip()] = value.strip()
    
    # Set up environment
    import os
    test_env = os.environ.copy()
    test_env.update(env_vars)
    
    # Change to the rundir (where RTL files are)
    # The docker-compose.yml says working_dir is /code/rundir
    # In host mode, this would be <repo>/rundir
    repo_dir = os.path.dirname(os.path.dirname(docker_dir))
    rundir = os.path.join(repo_dir, 'rundir')
    
    # Build the command
    # From docker-compose: pytest -s -o cache_dir=/rundir/harness/.cache /src/test_runner.py -s
    # In host mode: pytest -s -o cache_dir=<rundir>/harness/.cache <docker_dir>/src/test_runner.py -s
    cache_dir = os.path.join(rundir, 'harness', '.cache')
    os.makedirs(cache_dir, exist_ok=True)
    
    test_runner = os.path.join(docker_dir, 'src', 'test_runner.py')
    pytest_cmd = f"pytest -s -o cache_dir={cache_dir} {test_runner} -s"
    
    # Run the command
    return self.log_run(
        pytest_cmd,
        kill=None,
        logfile=logfile,
        monitor_dir=rundir if monitor_size else None,
        monitor_kill_cmd=None
    )
```

#### Step 4: Handle environment variables

The test harness expects:
- `VERILOG_SOURCES` - path to RTL files
- `TOPLEVEL` - top-level module name
- `MODULE` - test module name
- `SIM` - simulator (icarus)
- `PYTHONPATH` - path to harness library

You'll need to adjust paths from Docker paths (`/code/rtl/...`) to host paths.

### Challenges

1. **Path translation**: Docker uses `/code/rtl/` but host uses `<repo>/rtl/`
2. **Environment setup**: Need to replicate Docker environment exactly
3. **Isolation**: Docker provides isolation - host mode doesn't
4. **Testing**: Need to verify it works correctly

## Option 3: Use Docker but with Fixed Permissions

This is still the recommended approach - just fix Docker permissions and use it as designed.

## Recommendation

**Go with Option 1** (fix Docker permissions). It's:
- ✅ Fastest (5 minutes)
- ✅ No code changes
- ✅ Maintains compatibility
- ✅ Works with all CVDP features
- ✅ Consistent with how CVDP is designed

The Docker requirement is minimal - it's just packaging open-source tools, not commercial software.


