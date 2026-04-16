---
name: capture-kernel-trace
description: >
  Capture GPU kernel ATT (Advanced Thread Trace) locally via rocprofv3. Use this
  whenever the user wants to profile a local test or benchmark, discover kernel names,
  write an input.yaml with kernel_include_regex, collect ui_output_agent_* ATT output,
  and save the trace in a stable local directory for later analysis. Usage:
  /capture-kernel-trace <test_script.py> [kernel_name_pattern]
tools: Bash,Read,Write,Edit,Grep,Glob
---

# Capture Kernel Trace

Capture rocprofv3 ATT traces on the local machine, then stage the trace output
in a stable local directory for later analysis.

## Arguments

| Argument | Required | Description |
|----------|----------|-------------|
| `<test_script>` | Yes | Python test/bench script to profile, e.g. `./tests/kernels/test_pa.py` |
| `[kernel_pattern]` | No | Kernel name regex. If omitted, discover via `--stats` first |

If no test script is provided, ask the user.

If the user already has a `ui_output_agent_*_dispatch_*` directory, skip this skill
and use `/kernel-trace-analysis --dir <path>` instead of recollecting a trace.

## Local Environment

Assume a local FlyDSL/AITER checkout unless the user says otherwise:

```
REPO_ROOT=/FlyDSL
AITER_ROOT=/aiter
LIB_DIR=/FlyDSL/build-fly-v23/python_packages/flydsl/_mlir/_mlir_libs
PYTHONPATH=/FlyDSL/python:/FlyDSL/tests:/aiter
TRACE_OUTPUT_DIR=/tmp/kernel_trace_output
TRACE_ROOT=/claude-stuff/trace_data
INPUT_YAML=/tmp/input_trace.yaml
```

Before running `rocprofv3`:
- export `LD_LIBRARY_PATH=$LIB_DIR${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}`
- export `PYTHONPATH=/FlyDSL/python:/FlyDSL/tests:/aiter${PYTHONPATH:+:$PYTHONPATH}`
- export `FLYDSL_DEBUG_ENABLE_DEBUG_INFO=1`
- `cd` to the repo root that contains the test script, usually `/FlyDSL`

Adjust these paths only if the user's local environment differs.

---

## Workflow

```
Step 1: Verify the local test script path and profiling environment
Step 2: Discover kernel names (if pattern not provided)
Step 3: Write input.yaml with kernel_include_regex
Step 4: Run rocprofv3 -i input.yaml locally
Step 5: Copy the newest ui_output_agent_* directory into a stable local trace dir
Step 6: Validate code.json and source mapping
```

---

## Step 1: Verify local setup

Make sure the script exists locally. If it is relative, resolve it from the repo root
you plan to run in.

```bash
ls ./tests/kernels/test_pa.py
mkdir -p /claude-stuff/trace_data
```

For FlyDSL tests, use:

```bash
export LD_LIBRARY_PATH=/FlyDSL/build-fly-v23/python_packages/flydsl/_mlir/_mlir_libs${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
export PYTHONPATH=/FlyDSL/python:/FlyDSL/tests:/aiter${PYTHONPATH:+:$PYTHONPATH}
export FLYDSL_DEBUG_ENABLE_DEBUG_INFO=1
cd /FlyDSL
```

---

## Step 2: Kernel discovery (if no pattern provided)

Run `rocprofv3` in stats mode locally:

```bash
cd /FlyDSL && rocprofv3 --stats --kernel-trace -f csv -o /tmp/discover -- python ./tests/kernels/test_pa.py 2>&1
```

Then inspect `/tmp/discover_kernel_stats.csv` and select the target kernel.
Prefer the FlyDSL kernel the user cares about. Typical names contain `pa_decode`,
`kernel_0`, or the function name from the test script.

If there are multiple plausible kernels, present a short ranked table and let the user choose.

---

## Step 3: Configure input.yaml

Write `/tmp/input_trace.yaml` locally with the target `kernel_include_regex`:

```yaml
jobs:
   -
       kernel_include_regex: <KERNEL_PATTERN>
       kernel_iteration_range: "[1, [2-4]]"
       output_file: out
       output_directory: /tmp/kernel_trace_output
       output_format: [csv]
       truncate_kernels: true
       sys_trace: true
       advanced_thread_trace: true
       att_target_cu: 1
       att_shader_engine_mask: "0xf"
       att_simd_select: "0xf"
       att_buffer_size: "0x6000000"
```

You can write it with a heredoc or equivalent local file write. Example:

```bash
tee /tmp/input_trace.yaml > /dev/null <<'YAMLEOF'
jobs:
   -
       kernel_include_regex: "<KERNEL_PATTERN>"
       kernel_iteration_range: "[1, [2-4]]"
       output_file: out
       output_directory: /tmp/kernel_trace_output
       output_format: [csv]
       truncate_kernels: true
       sys_trace: true
       advanced_thread_trace: true
       att_target_cu: 1
       att_shader_engine_mask: "0xf"
       att_simd_select: "0xf"
       att_buffer_size: "0x6000000"
YAMLEOF
```

Key configuration:
- `kernel_include_regex`: Exact name or regex from Step 2
- `kernel_iteration_range`: `"[1, [2-4]]"` skips warmup (iteration 0), traces iterations 2-4
- `att_target_cu: 1`: Single CU for manageable output
- `att_buffer_size: "0x6000000"`: 96MB per SE (increase to `0xC000000` if truncated)

---

## Step 4: Run rocprofv3 with ATT

Run locally from the repo that contains the test script:

```bash
cd /FlyDSL && rm -rf /tmp/kernel_trace_output && rocprofv3 -i /tmp/input_trace.yaml -- python ./tests/kernels/test_pa.py 2>&1
```

Important:
- Set `FLYDSL_DEBUG_ENABLE_DEBUG_INFO=1` so `code.json` contains source file:line annotations
- Allow 3-5 minutes for JIT compilation plus trace collection
- If the test only runs a few iterations, adjust `kernel_iteration_range`

---

## Step 5: Stage the trace output locally

Find the newest `ui_output_agent_*` directory under `/tmp/kernel_trace_output`, then copy it
into a stable local directory such as `/claude-stuff/trace_data/<timestamp>_<kernel_short_name>/`.

A simple local staging flow:
1. Create `LOCAL_TRACE_DIR`
2. Copy the newest `ui_output_agent_*` directory contents into it
3. Copy `out_kernel_trace.csv`
4. Copy any `stats_*.csv`

The staged directory should contain:
- `code.json`
- `occupancy.json`
- `filenames.json`
- `snapshots.json`
- `se*_*.json` / `wstates*.json`
- `out_kernel_trace.csv`
- optional `stats_*.csv`

---

## Step 6: Verify the staged trace

Quick validation:

```bash
python3 -c "
import json
with open('/path/to/trace/code.json') as f:
    data = json.load(f)
n = len(data.get('code', []))
has_src = sum(1 for row in data.get('code', []) if row[3])
print(f'Instructions: {n}, with source mapping: {has_src} ({100*has_src//max(n,1)}%)')
"
```

---

## Output

After capture, report:

1. **Trace location**: Local path to the staged trace directory
2. **Kernel info**: Name, VGPR/AGPR counts, grid size, duration (from `out_kernel_trace.csv`)
3. **Source mapping**: Whether debug info is present (% of instructions with source annotations)
4. **Instruction count**: Total instructions in `code.json`
5. **Next step**: Suggest running `/kernel-trace-analysis --dir <dispatch_dir>` on the staged trace

Example output:

```
Trace captured: /claude-stuff/trace_data/20260325_153000_pa_decode/
  Kernel: pa_decode_sw_kernel_0
  Duration: 208.3 us
  arch_vgpr=96, accum_vgpr=128, SGPR=80
  Instructions: 2692, source-mapped: 2105 (78%)

Run /kernel-trace-analysis --dir /claude-stuff/trace_data/20260325_153000_pa_decode to analyze bottlenecks.
```

---

## Error Handling

| Error | Fix |
|-------|-----|
| `rocprof-trace-decoder library path not found` | Install decoder: see kernel-trace-analysis skill Step 3 |
| `INVALID_SHADER_DATA` | aqlprofile/decoder version mismatch, update both |
| Empty `ui_output_agent_*` | `kernel_include_regex` did not match; re-check the kernel name from Step 2 |
| No source mapping in `code.json` | Ensure `FLYDSL_DEBUG_ENABLE_DEBUG_INFO=1` is set |
| Trace truncated (missing instructions) | Increase `att_buffer_size` to `0xC000000` (192MB) |
| `libmlir_float16_utils.so` missing | Fix `LD_LIBRARY_PATH` or rebuild/symlink the FlyDSL `_mlir` libs |
| `rocprofv3: command not found` | Install ROCm profiling tools or run on a machine/container that has `rocprofv3` |
| Test script not found | `cd` to the correct repo root or pass the correct script path |
| `kernel_iteration_range` mismatch | Test ran fewer iterations than expected; use `"[0, [1-2]]"` |

---

## Quick Reference

### One-liner: discover + capture + stage locally

```bash
# 1. Environment
export LD_LIBRARY_PATH=/FlyDSL/build-fly-v23/python_packages/flydsl/_mlir/_mlir_libs${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
export PYTHONPATH=/FlyDSL/python:/FlyDSL/tests:/aiter${PYTHONPATH:+:$PYTHONPATH}
export FLYDSL_DEBUG_ENABLE_DEBUG_INFO=1
cd /FlyDSL

# 2. Discover
rocprofv3 --stats --kernel-trace -f csv -o /tmp/disc -- python ./tests/kernels/test_pa.py 2>&1

# 3. Capture (after setting KERNEL_PATTERN)
tee /tmp/it.yaml > /dev/null <<'EOF'
jobs:
   -
       kernel_include_regex: "<KERNEL_PATTERN>"
       kernel_iteration_range: "[1, [2-4]]"
       output_file: out
       output_directory: /tmp/kernel_trace_output
       output_format: [csv]
       truncate_kernels: true
       sys_trace: true
       advanced_thread_trace: true
       att_target_cu: 1
       att_shader_engine_mask: "0xf"
       att_simd_select: "0xf"
       att_buffer_size: "0x6000000"
EOF
rm -rf /tmp/kernel_trace_output
rocprofv3 -i /tmp/it.yaml -- python ./tests/kernels/test_pa.py 2>&1

# 4. Stage
python3 -c "
from pathlib import Path
import shutil, time
out = Path('/tmp/kernel_trace_output')
ui_dirs = sorted(out.glob('ui_output_agent_*'), key=lambda p: p.stat().st_mtime, reverse=True)
if not ui_dirs:
    raise SystemExit('No ui_output_agent_* directories found')
latest = ui_dirs[0]
dst = Path('/claude-stuff/trace_data') / f'{time.strftime(\"%Y%m%d_%H%M%S\")}_{latest.name}'
dst.mkdir(parents=True, exist_ok=True)
shutil.copytree(latest, dst, dirs_exist_ok=True)
for pattern in ('out_kernel_trace.csv', 'stats_*.csv'):
    for src in out.glob(pattern):
        shutil.copy2(src, dst / src.name)
print(dst)
"
```
