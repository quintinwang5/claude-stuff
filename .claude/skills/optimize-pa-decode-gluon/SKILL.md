---
name: optimize-pa-decode-gluon
description: >
  Optimize the pa_decode_gluon paged attention decode implementation in
  aiter/ops/triton/gluon/pa_decode_gluon.py from the ROCm/aiter repo.
  Covers both Gluon kernel optimizations and Python API-level optimizations.
  Use when user wants to improve performance of the paged attention decode
  kernels on AMD MI300X (CDNA3) or MI350 (CDNA4) GPUs.
  Usage: /optimize-pa-decode-gluon
tools: Read,Edit,Bash,Grep,Glob,Agent
---

# Optimize pa_decode_gluon

Optimize the paged attention decode implementation in `aiter/ops/triton/gluon/pa_decode_gluon.py`.

## Source Reference

- **Repository**: https://github.com/ROCm/aiter
- **File**: `aiter/ops/triton/gluon/pa_decode_gluon.py`
- **Components**: 5 Gluon/Triton kernels + 2 wrapper functions + 1 Python API function

## Architecture Overview

The file implements a two-phase paged attention decode:

1. **Phase 1 - Partitioned Attention**: One of 3 kernel variants computes partial attention per context partition
   - `paged_attention_decode_v2_gluon_dot_kernel` - standard blocks (KV_BLOCK_SIZE=16/64)
   - `paged_attention_decode_v2_gluon_large_block_dot_kernel` - large blocks (KV_BLOCK_SIZE=1024)
   - `paged_attention_decode_sliding_window` - sliding window / prefix sharing (PS) path
2. **Phase 2 - Reduction**: One of 2 kernels reduces partial results across partitions
   - `paged_attention_decode_v2_reduce_kernel` - standard reduction
   - `paged_attention_decode_ps_reduce_kernel` - PS-aware reduction
3. **API Function**: `pa_decode_gluon()` - orchestrates the two phases

## Optimization Checklist

When this skill is invoked, systematically analyze and optimize the following areas. **Always read the latest source code first** - do not rely on cached/stale versions.

### Part 1: Gluon Kernel Optimizations

#### 1.1 Memory Access Patterns
- [ ] **Coalesced global loads**: Ensure `gl.load` / `gl.amd.cdna3.buffer_load` access patterns produce coalesced 128-byte transactions. Check that innermost dimension strides are 1 (element-contiguous).
- [ ] **Shared memory bank conflicts**: When `gl.allocate_shared_memory` or `gl.SwizzledSharedLayout` is used, verify swizzle parameters eliminate bank conflicts for the actual access pattern. Current: `SwizzledSharedLayout(KV_16B_ELEMENT_COUNT, 1, 16, order=[1,0])` - verify vec/perPhase/maxPhase are optimal for the query tensor shape.
- [ ] **Vectorized loads**: Confirm `size_per_thread` in BlockedLayout uses maximum vectorization (8 for bf16/fp16, 16 for fp8) in the contiguous dimension. If a layout has `size_per_thread=[1,1]` in the inner dim, consider restructuring.
- [ ] **Redundant loads**: Query tensor is loaded once and reused across the KV loop - good. Check if block_tables lookup can be hoisted or prefetched.

#### 1.2 Compute Efficiency
- [ ] **MFMA instruction selection**: Verify `QK_PV_MFMA_INSTR_SHAPE` matches the optimal MFMA instruction for the target arch. For CDNA3 fp8: `mfma_f32_16x16x32_fp8`, for CDNA3 bf16: `mfma_f32_16x16x16_bf16`. For CDNA4: check if 32x32 variants or new instructions give better throughput.
- [ ] **Warp-level parallelism**: Current config uses 4 warps per CTA (`warps_per_cta=[1,4]` for MFMA). For CDNA4 with higher register file, consider `warps_per_cta=[2,4]` or `[1,8]` if register pressure allows.
- [ ] **waves_per_eu tuning**: The wrapper sets `waves_per_eu` to 3 or 4 based on `QUERY_GROUP_SIZE_POW2`. Profile to determine if this is optimal. Lower values reduce register spilling but may underutilize the CU. **NOTE**: Autotune sweep across num_stages(1-4) x waves_per_eu(0-4) showed no meaningful difference for the `sliding_window_head_1` kernel - default Triton settings are already reasonable for this kernel.
- [ ] **Occupancy vs register pressure**: The large_block kernel has more live variables. Check if `maxnreg` pragma or register allocation hints could help. **CRITICAL WARNING - see Section 1.2.1 below**.
- [ ] **exp2 vs exp**: Already using `tl.math.exp2` with `LOG2_E` multiplier - this is optimal for AMD GPUs. Confirm no inadvertent `tl.exp` calls.

#### 1.2.1 VGPR Architecture on CDNA3 (CRITICAL)

On MI300X (gfx942), VGPRs are split into two **separate** register files:
- **arch_vgpr**: General-purpose vector registers (used by VALU, VMEM, LDS ops)
- **accum_vgpr (AGPR)**: Accumulator registers (used exclusively by MFMA result writeback)

**Occupancy** is determined by: `256 / max(arch_vgpr, accum_vgpr)` waves per SIMD (4 SIMDs per CU).

**WARNING: `maxnreg` on CDNA3 is DANGEROUS for MFMA-heavy kernels:**
- `maxnreg=64` forces the compiler to set `accum_vgpr=0`, which doubles occupancy (12.5% -> 25%)
- BUT this forces ALL MFMA results to be moved through arch_vgpr via `v_accvgpr_read` instructions
- The result is **massive register spilling** and MFMA pipeline stalls
- Real-world impact: GPU kernel time increased from ~149us to ~670us (4.5x SLOWER) even though occupancy doubled
- **End-to-end benchmark (triton.testing.do_bench) masked this regression** because CPU dispatch overhead (~211us) dominated the measurement, making the change appear neutral

**Rule**: Never use `maxnreg` to eliminate accum_vgpr usage on CDNA3. Accum_vgpr are essentially free for MFMA - they don't compete with arch_vgpr for occupancy. Only consider `maxnreg` to reduce arch_vgpr if it's the occupancy bottleneck.

To check VGPR allocation, query the rocprofv3 database:
```sql
SELECT ks.KernelName, ki.arch_vgpr_count, ki.accum_vgpr_count
FROM rocpd_kernel_dispatch kd
JOIN rocpd_info_kernel_symbol ks ON kd.kernel_symbol_id = ks.id
JOIN rocpd_info_kernel ki ON kd.kernel_id = ki.id
WHERE ks.KernelName LIKE '%paged_attention%'
LIMIT 5;
```

#### 1.3 Loop Structure & Prefetch
- [ ] **K-cache prefetch (double-buffer)**: The main KV loop issues 8× `global_load_dwordx4` for K-cache tiles, then hits `s_waitcnt vmcnt(0)` with **~33K stall cycles** before the first MFMA. The 126 instructions between load and consume are not enough to hide global memory latency. Apply double-buffering: pre-load the first iteration's K-cache before the loop, issue the next iteration's K-cache loads right after the swap, then compute MFMA on the current buffer while the next buffer is in flight.
  - **Register budget**: Trace shows VGPR=256, AGPR=55, total=311/512. There is ~201 regs headroom — enough for a second set of 8× dwordx4 buffers (~128 VGPRs). **NOTE**: arch_vgpr and accum_vgpr are separate register files on CDNA3 - see Section 1.2.1. Headroom must be calculated for each file independently.
  - **Pattern**: See `/prefetch-data-load` skill for the mechanical transformation.
- [ ] **V-value load hoisting**: V-value loads (8× `global_load_dwordx4` at source :1669) currently start after the softmax reduce phase. Their addresses can be computed before the reduce. **Hoist V-value load issuance into the softmax reduce barrier-wait region** (L606-L770) to overlap V-value fetch latency (~17K idle cycles) with barrier stalls (~96K stall cycles). This turns wasted barrier-wait time into useful prefetch.
- [ ] **Overlap value load with QK MFMA** in `paged_attention_decode_v2_gluon_dot_kernel`: Value is loaded after QK MFMA, so value load latency is fully exposed. Restructure to issue value load before or concurrently with QK MFMA (mirrors existing pattern in `paged_attention_decode_sliding_window`).
- [ ] **Loop unrolling**: `KV_COMPUTE_BLOCK_COUNT` is typically 1-4. Check if explicit unrolling (via `tl.static_range` or compile-time loop) helps.
- [ ] **num_stages**: Currently hardcoded to `num_stages=1`. If Triton supports multi-stage pipelining for Gluon kernels, try `num_stages=2` to overlap next iteration's loads with current compute. **NOTE**: Autotune sweep showed num_stages=1-4 had no meaningful impact on the sliding_window_head_1 kernel.

#### 1.3.1 Softmax Reduce Optimization (Highest Impact)
The softmax cross-wave reduce phase (source :189 and :291) is the **single largest bottleneck** at ~96K stall cycles (40.6% of all stalls). It currently performs two sequential reduce passes (max and sum) each with 3-4 barriers:

```
ds_bpermute → s_waitcnt lgkmcnt(0) → s_barrier → ds_swizzle(SWAP,16)
→ s_waitcnt → ds_write LDS → s_waitcnt → s_barrier → ds_read LDS → ...
```

Optimizations:
- [ ] **Merge max and sum reduce passes**: If online softmax computes max and sum in separate reduce trees, merge them into a single pass that reduces a `(max, sum)` tuple together. This halves the barrier count from ~7 to ~3-4.
- [ ] **Use continuous `ds_bpermute` tree reduce**: Replace the `ds_bpermute → barrier → ds_write LDS → barrier → ds_read LDS` pattern with consecutive `ds_bpermute` calls for each tree level. Each `ds_bpermute` is a cross-lane shuffle that doesn't require LDS write/read round-trips or inter-wave barriers.
- [ ] **Reduce LDS round-trip in reduce**: Each `ds_write → s_waitcnt → s_barrier → ds_read` sequence costs ~6-8K stall cycles. If the reduce can use only `ds_bpermute`/`ds_swizzle` (intra-wavefront), the LDS write/read + barrier steps can be eliminated entirely for within-wavefront stages.

#### 1.4 Data Type Optimizations
- [ ] **FP8 accumulation precision**: Currently accumulates in fp32, which is correct. However, the `probability_scale` path for per-token FP8 quantization does `value_scale_value * FP8_MAX_VALUE / (value_scale_max + 1e-8)` - the epsilon `1e-8` may be unnecessary if `value_scale_max > 0` is guaranteed. Removing it saves a floating-point add.
- [ ] **Output type conversion**: `attention_accumulator.to(OUTPUT_DTYPE)` happens after the loop. If intermediate results are only used in fp32 operations, this is fine. Verify no unnecessary intermediate conversions.
- [ ] **Cast placement**: `query_converted.to(COMPUTE_TYPE)` and `key_converted.to(COMPUTE_TYPE)` should happen as late as possible (right before MFMA) to minimize register pressure from wider types.

#### 1.5 Masking Optimizations
- [ ] **Compile-time mask elimination**: When `IS_CAUSAL=False` and the partition is fully within context length, the boundary mask is all-true. Add a fast path that skips `tl.where` for fully-valid partitions.
- [ ] **Mask sentinel value**: Currently uses `float(-3.4e38)` instead of `-inf` to avoid NaN. This is correct but slightly less precise. Verify this doesn't affect output quality for very long sequences.

#### 1.6 CDNA4-Specific Optimizations
- [ ] **New MFMA instructions**: CDNA4 (gfx950) may support larger MFMA shapes or new data types. Check if `mfma_f32_32x32x32` variants are available and beneficial.
- [ ] **Increased register file**: CDNA4 has more VGPRs per CU. Increase `KV_COMPUTE_BLOCK_SIZE` or tile sizes to exploit this.
- [ ] **Matrix core improvements**: CDNA4 may have improved fp8 throughput. Verify `MFMA_INSTR_K=32` is still optimal.

### Part 2: Python API Optimizations

#### 2.1 Tensor Allocation
- [ ] **Duplicate allocation**: `exp_sums`, `max_logits`, and `temporary_output` are conditionally allocated TWICE in `pa_decode_gluon()` - once before assertions and once after. Remove the first allocation block (lines before assertions) since the second one always executes.
- [ ] **Pre-allocation / caching**: These intermediate tensors are allocated every call. Consider accepting them as pre-allocated buffers (which the API already supports via optional params) and document this as the recommended usage pattern.
- [ ] **Memory pool**: Use `torch.cuda.caching_allocator` or a persistent buffer pool to avoid per-call allocation overhead for `exp_sums`, `max_logits`, `temporary_output`.

#### 2.2 Kernel Dispatch
- [ ] **Grid calculation**: `max_context_partition_num` is passed as an argument. For `one_shot` cases (1 partition), the reduction kernel is skipped - this is already optimized.
- [ ] **PS path short-circuit**: When `PS=True`, the sliding_window kernel is launched inside the wrapper and returns early. But the wrapper is still called through an extra function indirection. Consider inlining or using `@torch.compile` for the dispatch logic.
- [ ] **Stride computation**: Multiple `.stride()` calls on reshaped tensors. These are cheap but could be computed once and cached for hot paths.

#### 2.3 One-Shot Path
- [ ] **Direct output write**: When `one_shot=True`, the kernel writes directly to `output_5d`, skipping the reduction kernel. But `temporary_output` may still be allocated. Add early return before allocation when `one_shot=True`.
- [ ] **one_shot detection**: `one_shot = max_context_partition_num <= 1` is set twice. Clean up to compute once.

#### 2.4 Validation Overhead
- [ ] **Assert cost**: Multiple `assert` statements run on every call. In production, these should be compiled out (`python -O`) or guarded behind a debug flag.
- [ ] **Type checking**: `query.dtype in [...]` checks run every call. Consider a `@lru_cache` or one-time validation approach.

#### 2.5 Recommended Splits
- [ ] **get_recommended_splits**: Uses `torch.cuda.get_device_properties()` which may not be cached. Wrap in `@lru_cache` (partially done already for `get_cdna_version`).
- [ ] **Occupancy-aware splitting**: `get_occupancy()` returns hardcoded `2`. This should be dynamic based on kernel register usage and shared memory. Use Triton's occupancy calculator if available.

### Part 3: Integration-Level Optimizations

#### 3.1 Kernel Fusion
- [ ] **Fuse reduction into attention kernel**: For small `max_context_partition_num` (<=4), the reduction could be done within the attention kernel using cross-CTA synchronization or within the same CTA if partitions fit.
- [ ] **ONE_SHOT in sliding_window**: The `paged_attention_decode_sliding_window` kernel already supports `ONE_SHOT` mode which fuses attention+reduction. Verify this path is always taken when possible.

#### 3.2 Autotuning
- [ ] **Uncomment autotune**: The `paged_attention_decode_v2_gluon_dot_kernel` has a commented-out `@triton.autotune` decorator. Enable it with a focused config space to find optimal `waves_per_eu` and `num_stages` per hardware.
- [ ] **Config key**: Autotune key should include `HEAD_SIZE`, `KV_BLOCK_SIZE`, `QUERY_GROUP_SIZE`, and `CONTEXT_PARTITION_SIZE` for representative coverage.

## How To Apply

1. **Read the source**: Always start by reading the latest version of `pa_decode_gluon.py`.
2. **Profile first**: Before making changes, establish a baseline using `rocprofv3 --kernel-trace --stats` to measure **actual GPU kernel time**. Do NOT rely solely on end-to-end benchmarks (see Section "Benchmarking Pitfalls" below).
3. **Apply one optimization at a time**: Make a single change, benchmark, and verify correctness before moving to the next.
4. **Verify correctness**: Use the existing test suite (`test_pa_decode_gluon.py` or equivalent) to ensure output matches within tolerance. For focused testing on a specific head config, create a targeted test script (e.g., 14 tests for (4,1) heads only) rather than running the full 576-test suite which can OOM with large batch sizes.
5. **Benchmark**: Compare **GPU kernel duration** (us) via `rocprofv3 --kernel-trace`, not end-to-end wall-clock time. Also monitor VGPR counts (arch + accum) via rocprofv3 database queries.
6. **Clear Triton cache**: After any kernel source or compilation parameter change, clear the cache: `rm -rf ~/.triton/cache/*`. Stale cached kernels will mask your changes.

## Benchmarking Pitfalls

**CRITICAL: End-to-end benchmarks are misleading for this kernel.**

The Triton JIT dispatch overhead for `pa_decode_gluon` is ~211us per call (CPU-side). This includes:
- `specialize_impl` called ~30 times per launch to hash all 50+ kernel arguments (~105us)
- Python argument marshalling and grid computation (~106us)

For kernels with GPU time < 200us, this CPU overhead **dominates** the end-to-end measurement from `triton.testing.do_bench`. A kernel change that doubles GPU time from 149us to 670us may appear neutral or even beneficial in end-to-end measurement because `do_bench` measures wall-clock time where 211us of CPU overhead masks the GPU regression.

**Always measure GPU kernel time separately:**

```bash
# Method 1: rocprofv3 kernel trace (recommended)
rocprofv3 --kernel-trace --stats -- python bench_target.py
# Parse the results.db SQLite database:
sqlite3 results.db "
  SELECT ks.KernelName,
         COUNT(*) as calls,
         ROUND(AVG(kd.end - kd.start)/1000.0, 1) as avg_us,
         ROUND(MIN(kd.end - kd.start)/1000.0, 1) as min_us
  FROM rocpd_kernel_dispatch kd
  JOIN rocpd_info_kernel_symbol ks ON kd.kernel_symbol_id = ks.id
  WHERE ks.KernelName LIKE '%paged_attention%'
  GROUP BY ks.KernelName
  ORDER BY avg_us DESC;
"

# Method 2: Check VGPR allocation
sqlite3 results.db "
  SELECT ks.KernelName, ki.arch_vgpr_count, ki.accum_vgpr_count
  FROM rocpd_kernel_dispatch kd
  JOIN rocpd_info_kernel_symbol ks ON kd.kernel_symbol_id = ks.id
  JOIN rocpd_info_kernel ki ON kd.kernel_id = ki.id
  WHERE ks.KernelName LIKE '%paged_attention%'
  LIMIT 5;
"
```

## Kernel Dispatch Path Reference

Understanding which kernel gets dispatched for a given configuration:

| Condition | Kernel Function | Notes |
|-----------|----------------|-------|
| `PS=True` or `SLIDING_WINDOW>0`, `num_kv_heads=1` | `paged_attention_decode_sliding_window_head_1` (line ~897) | Single KV-head specialization |
| `PS=True` or `SLIDING_WINDOW>0`, `num_kv_heads>1` | `paged_attention_decode_sliding_window` | General sliding window |
| `KV_BLOCK_SIZE=1024` | `paged_attention_decode_v2_gluon_large_block_dot_kernel` | Large block variant |
| Default (no PS, no SW, block<1024) | `paged_attention_decode_v2_gluon_dot_kernel` | Standard kernel |

The PS path is launched via `_paged_attention_decode_v2_with_dot_kernel_reshape_wrapper` (line ~4295). Kernel launch parameters end at line ~4440 (`ONE_SHOT=ONE_SHOT,`).

**ISA trace invalidation**: Any compilation parameter change (maxnreg, num_stages, waves_per_eu) produces entirely different ISA. All ISA-level optimization tasks (instruction scheduling, LDS patterns, VGPR layout) must be re-analyzed after any such change.

## Quick Wins (Highest Impact, Lowest Risk)

### Kernel-Level (requires re-profiling after each change)
1. **Merge softmax max/sum reduce passes** — reduces ~7 barriers to ~3-4, saving ~40K+ stall cycles (estimated 15-20% kernel speedup). Modifies reduce logic only.
2. **K-cache prefetch double-buffering** — eliminates ~33K stall at `s_waitcnt vmcnt(0)` before MFMA. Register headroom is sufficient (201 spare regs). See `/prefetch-data-load`.
3. **Hoist V-value loads into barrier-wait region** — overlaps ~17K idle with ~96K barrier stall, nearly free latency hiding.
4. **Overlap value load with QK MFMA** in `dot_kernel` — mirrors existing pattern in sliding_window kernel.

### Caller-Level (no kernel changes, validated improvements)
5. **Pre-allocate temp buffers** — pass `exp_sums`, `max_logits`, `temporary_output` as pre-allocated tensors to `pa_decode_gluon()` instead of letting it allocate per call. **Measured: 15.2% end-to-end latency reduction** (184.8us -> 156.8us for BS=128). The API already supports these as optional parameters.
6. **Use ONE_SHOT mode for small batch sizes** — when `max_context_partition_num <= 1`, set `one_shot=True` to skip the reduction kernel. **Measured: 38.5% end-to-end reduction** for BS=4 (176us -> 108.3us). Automatically detected by the API but callers should set `max_context_partition_num` correctly.
7. **Fix duplicate tensor allocation** in `pa_decode_gluon()` API — pure cleanup, zero risk.
8. **Reduce Triton JIT dispatch overhead** — specialize_impl is called ~30 times per launch. Consider caching argument hashes or using `@triton.autotune` with a single config to skip specialization.

## Trace-Based Evidence (MI308 gfx942, dispatch 4025)

Reference trace: `~/Documents/ui_output_agent_12034_dispatch_4025/`

| Region | Stall Cycles | % of Total Stall | Root Cause |
|--------|-------------|------------------|------------|
| Softmax reduce (:189 + :291) | 96,112 | 40.6% | 7 barriers + serial LDS reduce |
| QK pre-MFMA wait (L397 vmcnt) | 32,740 | 13.8% | K-cache load latency exposed |
| Prologue lgkmcnt waits | 17,960 | 7.6% | Kernel arg scalar loads |
| K-cache global_load stalls | 38,036 | 16.1% | TA pressure from 8 concurrent loads |
| V-value region idle | 17,212 idle | — | Address computation bubbles |

MFMA utilization: **1.4%** (32 MFMA instructions, 10,280 / 735,416 total cycles) — severely memory/sync bound.

## Remote Workflow (SSH + Docker)

When the kernel runs on a remote MI300X host inside a Docker container:

1. **Write scripts locally** (e.g., `apply_config.py`, `test_correctness.py`, `profile_overhead.py`)
2. **Transfer via scp + docker cp**: `scp script.py host:/tmp/ && ssh host 'docker cp /tmp/script.py container:/path/'`
3. **Run via docker exec**: `ssh host 'docker exec container python /path/script.py'`
4. **Do NOT try inline Python in SSH**: Nested quoting (bash -> ssh -> docker exec -> python) fails with syntax errors. Always use script files.
5. **Repos in container**: `/opt/triton` (branch: `rocm-maxnreg-support-v35`), `/opt/aiter`, plus the working copy at `/mnt/sixifang/aiter`
6. **Backup before modifying**: `cp pa_decode_gluon.py pa_decode_gluon.py.bak` and restore between experiments

## Output

After optimization, report:
- Which optimizations were applied
- Before/after **GPU kernel duration** (from rocprofv3, not end-to-end)
- VGPR allocation (arch + accum) before/after
- Any correctness concerns or trade-offs
- Remaining optimization opportunities
