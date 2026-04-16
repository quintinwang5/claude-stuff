# kernel-opt

This is an autonomous optimization program, in the style of `autoresearch/program.md`, for a single kernel. The goal is to continuously reduce the target kernel's GPU execution time without breaking correctness; any attempt that regresses performance must be rolled back, and only performance-improving attempts may be kept and committed.

## Fixed Target

For this run, use the following command as the single source of truth for both the **unit test and benchmark**:

```bash
python tests/kernels/test_pa.py
```

Throughout the entire optimization process, **do not modify this benchmark command** unless a human explicitly asks you to switch targets.

### Fixed Metrics

- Correctness gate: the script must exit successfully, and `.csv` must contain `err_flydsl_ps == 0`
- Script-side benchmark metric: `us_flydsl_ps` in the `.csv`, lower is better
- Final keep/discard metric: the average GPU kernel time from `rocprofv3` produced by `/kernel-trace-analysis`, lower is better
- Noise rule: changes smaller than `1%` are considered noise by default, unless the same-direction improvement appears twice in a row
- If the benchmark output contains multiple configs, do not look only at a single overall average; you must break results down by config dimensions, and at minimum prioritize any actual `block_size`, `mtp`, `head_size`, plus any other columns that directly affect dispatch

`us_flydsl_ps` is used for quick regression checks; when deciding whether to keep a modification, prioritize `rocprofv3` GPU kernel time. Do not keep a modification just because Python-side timing looks faster if the actual GPU kernel got slower.

When the benchmark returns many configs, do not collapse all rows into a single number. You must first inspect the config patterns and identify which combinations naturally belong to the same class, such as `block_size=1024`, `mtp=1`, `head_size=128`; all later optimization, regression judgment, and kernel-splitting decisions must be analyzed by config class.

## Setup

Prepare as follows before starting:
0. Use `/FlyDSL` as the root directory
1. First choose a run tag, for example `apr16-pa-ps`
2. Create a dedicated worktree or dedicated branch from the current `HEAD` to avoid polluting the existing workspace. Example:

```bash
git worktree add ../kernel-opt-<tag> -b kernel-opt/<tag> HEAD
```

3. Work in the dedicated worktree; do not experiment directly in the original dirty workspace
4. Read first:
   - `tests/kernels/test_pa.py`
   - The kernel implementation file ultimately dispatched by that script
   - Any helper files that directly affect launch config / tile / prefetch / LDS / wait strategy
5. Confirm that both `rocprofv3` and `rocm-smi` are available in `PATH`; if either is unavailable, stop and inform the human
6. If `performance.md` does not exist, create it
7. Record the current commit as:

```bash
BEST_COMMIT=$(git rev-parse HEAD)
```

`BEST_COMMIT` means "the current latest known-correct and best-performing commit." All later failed or regressive attempts must return to this point before continuing.

Also maintain a regression counter:

```bash
CONSECUTIVE_REGRESSIONS=0
```

Here, a "performance regression" specifically means: the candidate passes correctness and benchmark, but compared with the current `BEST_COMMIT`, the target kernel's average GPU time in `rocprofv3` gets worse beyond the noise threshold (`> 1%`). Only this case should increment the regression counter by `+1`.

Also maintain a recent regression summary list:

```bash
RECENT_REGRESSIONS=[]  # rolling window, max len = 5
```

Whenever a clear performance regression occurs, append a summary to `RECENT_REGRESSIONS` that includes at least:

- Date/time
- Baseline commit: current `BEST_COMMIT`
- Candidate commit or working tree state
- Hypothesis for this round
- Modified files
- `us_flydsl_ps`: best -> candidate
- `rocprofv3` target kernel average GPU time: best -> candidate
- Percentage change relative to best
- Main regression points observed in this round's trace

"Consecutive" is defined as follows:

- Clear performance regression: `CONSECUTIVE_REGRESSIONS += 1`
- Performance improvement is kept: `CONSECUTIVE_REGRESSIONS = 0`, and clear `RECENT_REGRESSIONS`
- Tie within noise: `CONSECUTIVE_REGRESSIONS = 0`, and clear `RECENT_REGRESSIONS`
- Benchmark failure / missing metrics: do not count as a "performance regression" and do not write to `RECENT_REGRESSIONS`; because no comparable performance regression was formed, also reset `CONSECUTIVE_REGRESSIONS = 0` by default
- When correctness fails and `.csv` shows `err_flydsl_ps != 0`, do not discard the round immediately; first enter a subflow of "up to 3 kernel repair attempts to restore accuracy." Only if all 3 repairs still fail should the optimization proposal be treated as failed, and `CONSECUTIVE_REGRESSIONS` reset to `0`

## GPU Availability Gate

Any test, benchmark, correctness retest, or `/kernel-trace-analysis` run that will actually occupy a GPU must pass the following GPU gate before it starts:

1. First use `rocm-smi` to inspect all GPUs and look for an idle GPU
2. Define "idle" based on the utilization / memory / process information visible in `rocm-smi`; prefer a card with no active compute process, near-idle GPU utilization, and low memory usage or memory usage that is stable and not growing
3. If multiple idle GPUs are found, prefer the most idle one and bind subsequent commands to that GPU, for example by setting `HIP_VISIBLE_DEVICES=<gpu_id>`
4. If there is currently no idle GPU, do not preempt; wait `10` minutes, then re-check with `rocm-smi`
5. Wait at most `1` hour total, meaning retry every `10` minutes after the initial check; if an idle GPU appears at any point during this period, continue the task
6. If there is still no available GPU after a full `1` hour, this is not a normal benchmark failure; it is a stop condition for the entire optimization task: abort immediately and stop trying new optimization ideas

Do not run tests on a busy GPU, or benchmark noise may become distorted and you may interfere with other people's jobs.

## First Run: Baseline

The first round must be the baseline with unmodified code.

1. First execute `GPU Availability Gate`, obtain an idle GPU, then run the fixed benchmark on that GPU and write the full output to the log:

```bash
python tests/kernels/test_pa.py
```

2. Confirm:
   - Process exit code is `0`
   - A `.csv` was generated
   - `err_flydsl_ps == 0`

3. Read baseline metrics from the `.csv`:

```bash
python - <<'PY'
import pandas as pd
df = pd.read_csv("<filename>.csv")
if not (df["err_flydsl_ps"] == 0).all():
    raise SystemExit("correctness failed")
print(f"baseline_us_asm_fp8={df['us_flydsl_ps'].mean():.3f}")
PY
```

4. Execute `GPU Availability Gate` again, then run the same benchmark command with:

```text
/kernel-trace-analysis python tests/kernels/test_pa.py
```

5. Record from the trace report:
   - Hottest kernel name
   - Average GPU time of that kernel
   - arch VGPR / accum VGPR / SGPR / LDS
   - Top bottlenecks

6. Write the baseline into `performance.md` with status `baseline`

## Config-Aware Strategy

The benchmark `.csv` may contain many configs. During autonomous optimization, you must first understand the config structure before deciding on an optimization strategy:

1. First inspect which config columns actually exist in the `.csv`; do not assume the fields are fixed
2. Prioritize grouping by actual `block_size`, `mtp`, and `head_size`; if other columns explain performance stratification better, include them in the config classification as well
3. Record baseline correctness and performance for each config class; do not record only an overall average
4. Every optimization round must state clearly whether it is a "global optimization" or an optimization specific to one config class
5. If a hypothesis only makes sense for a certain config class, it is allowed to optimize only that class, but the target config class must be explicitly recorded
6. Non-target configs must not be heavily sacrificed for gains on the target configs; do not use "one subset improved a lot" to hide "another subset clearly regressed"

A "config class" means a group of inputs that goes through the same launch / tile / kernel decision path and shows similar performance behavior. Classification must be based on real columns present in benchmark output, not on arbitrary guesswork.

## What You Can Change

You may modify:

- The kernel implementation itself
- Runtime dispatch logic so that different config classes use different kernels
- Launch config
- Tile / wave / split / partition choices
- Prefetch / double-buffer / software pipelining
- LDS layout / swizzle / padding
- Instruction scheduling before/after wait/barrier
- Vectorization width
- Local code that affects occupancy or memory-latency hiding
- A specialized kernel added specifically for one config class

You may not modify:

- The fixed benchmark command
- Benchmark shape
- Secretly reducing the amount of work or bypassing correctness just to "run faster"
- Introducing new dependencies
- Keeping a modification that makes GPU kernel time worse

If you must modify the benchmark script itself, it may only be for:

- Fixing the benchmark harness
- Making the metrics more stable and easier to parse
- Fixing an obviously wrong correctness gate

Do not modify the benchmark just to make the numbers look better.

## Correctness Repair Rule

If a candidate round produces a `.csv` and `err_flydsl_ps != 0`, that means the optimization broke accuracy. Do not immediately abandon the current idea; you must first try to repair the kernel:

1. Keep the current optimization hypothesis unchanged, and allow only the minimal repairs required to make accuracy pass again
2. Before each repair attempt, execute `GPU Availability Gate`, then rerun the fixed benchmark and recheck `err_flydsl_ps` in the `.csv`
3. A single optimization proposal is allowed at most `3` consecutive repair attempts
4. As soon as one repair restores `err_flydsl_ps == 0`, return to the normal performance evaluation flow and continue comparing the candidate against `BEST_COMMIT`
5. If `err_flydsl_ps` is still not `0` after `3` repairs, abandon this optimization proposal: revert to `BEST_COMMIT`, do not commit, do not write to `performance.md`, and continue with a new optimization idea

This `3`-repair limit is a local limit for a single optimization proposal and does not count toward `CONSECUTIVE_REGRESSIONS`.

## Kernel Split Rule

If benchmark results show clear config divergence, for example one config class improves significantly while another class clearly and reproducibly regresses, do not force all configs to share the same optimized version. In this case, prefer splitting:

1. Keep the current known-best general-purpose kernel as the default path
2. Add a specialized kernel for the config class that clearly benefits
3. In runtime dispatch, choose kernels based on config conditions, for example actual `block_size`, `mtp`, `head_size`, or other real configuration columns
4. Route the benefiting configs to the new kernel, while keeping the other configs on the old best kernel
5. After splitting, rerun the fixed benchmark, verify correctness for all configs, and compare performance again across config classes

Only if "after dispatch splitting, the benefiting configs are clearly faster and the other configs are not slowed down" may this kind of new-kernel approach be kept.

## Optimization Loop

Starting from the baseline, enter an infinite loop. Do not stop unless manually interrupted by a human.

LOOP FOREVER:

1. Start from `BEST_COMMIT`
2. First execute `GPU Availability Gate`, then run or rerun `/kernel-trace-analysis` to make sure the bottleneck profile is up to date
3. Read the latest benchmark `.csv`, and by config dimension re-check which config classes are hottest, slowest, or most prone to regressions; at minimum inspect `block_size`, `mtp`, and `head_size`
4. Choose exactly **one** bottleneck-driven hypothesis for this round, and state clearly whether it is a "global optimization" or an optimization specific to one config class
5. Write down the hypothesis and target config class in one sentence before editing, for example:
   - Launch next-iteration loads earlier to reduce exposed `s_waitcnt vmcnt(0)`
   - Reduce LDS write/read round-trips to lower `lgkmcnt` and barrier stall
   - Reduce arch VGPR pressure to improve occupancy
   - Insert independent work between MFMA/VALU to fill idle bubbles
6. Only modify the minimal set of files needed to support this hypothesis
7. Set `FIX_ATTEMPTS=0`, first execute `GPU Availability Gate`, then rerun the fixed benchmark and save the output as `candidate.log`
8. If the benchmark fails or metrics are missing:
   - Discard this attempt
   - Revert to `BEST_COMMIT`
   - Reset `CONSECUTIVE_REGRESSIONS` to `0`
   - Clear `RECENT_REGRESSIONS`
   - Continue to the next idea
9. If the benchmark succeeds, first read the `.csv` by config class; do not look only at the total average
10. If any config class has `err_flydsl_ps != 0`:
   - Enter the correctness repair subflow
   - Repair the kernel while keeping the current optimization direction unchanged, and increment `FIX_ATTEMPTS += 1`
   - Before each repair attempt, execute `GPU Availability Gate`, rerun the fixed benchmark, and recheck `err_flydsl_ps`
   - Proceed to the next step only after all config classes are restored to `err_flydsl_ps == 0`
   - If failure still remains after `FIX_ATTEMPTS >= 3`:
     - Abandon this optimization proposal
     - Revert to `BEST_COMMIT`
     - Do not commit
     - Do not write to `performance.md`
     - Reset `CONSECUTIVE_REGRESSIONS` to `0`
     - Clear `RECENT_REGRESSIONS`
     - Continue to the next idea
11. First execute `GPU Availability Gate`, then run `/kernel-trace-analysis` again on the candidate
12. When comparing the candidate against the current best, you must look at both the overall result and the per-config result:
   - All config classes must still have `err_flydsl_ps == 0`
   - The target config class must genuinely improve
   - Non-target config classes must not show clear and reproducible severe regressions
   - `us_flydsl_ps` must not improve by sacrificing other config classes
   - The target kernel's average GPU time in `rocprofv3` must improve; if kernels are split, confirm that the new dispatch makes the benefiting configs faster without slowing down the others
13. If the candidate is better:
   - Append the result to `performance.md`
   - Run `/format-code`
   - After formatting, first execute `GPU Availability Gate`, then rerun the fixed benchmark
   - If the formatted result is still better, commit the code and `performance.md`
   - Update `BEST_COMMIT` to the new commit
   - Reset `CONSECUTIVE_REGRESSIONS` to `0`
   - Clear `RECENT_REGRESSIONS`
14. If the candidate shows a divergence where "some config classes improve a lot but other config classes regress severely":
   - Do not keep this single-kernel version
   - If this divergence is stable and explainable, follow `Kernel Split Rule` to switch to a specialized kernel + dispatch, then reevaluate
   - If you have not split yet, first revert to `BEST_COMMIT`
15. If the candidate is tied or worse:
   - Revert all modifications back to `BEST_COMMIT`
   - Do not commit
   - Do not write to `performance.md`
   - If it is a tie within noise, reset `CONSECUTIVE_REGRESSIONS` to `0` and clear `RECENT_REGRESSIONS`
   - If it is a measurable performance regression rather than a tie within noise, increment `CONSECUTIVE_REGRESSIONS += 1` and append this round's summary to `RECENT_REGRESSIONS`
   - If `CONSECUTIVE_REGRESSIONS >= 5`, stop optimization and write the stop reason plus the latest 5 regression summaries into `performance.md`
   - Otherwise continue to the next round

Because the entire experiment runs in a dedicated worktree / dedicated branch, failed or regressive attempts may reset the experiment branch directly back to `BEST_COMMIT`. Do not do this in a shared workspace that contains unrelated changes.

## Using `/kernel-trace-analysis`

Every analysis round runs the same fixed command:

```text
/kernel-trace-analysis python tests/kernels/test_pa.py
```

When reading the report, prioritize:

- High-cycle `s_waitcnt` / `s_barrier`
- High-cycle `buffer_load_*` or other VMEM stalls
- `ds_read_*` / `ds_write_*` / `lgkmcnt` pressure
- Idle bubbles around MFMA/VALU
- Whether arch VGPR or accum VGPR is the occupancy limiter

If there are multiple kernels:

- Optimize the one with the highest GPU time share first
- If both the main kernel and the reduce kernel are hot, prioritize the larger one
- Only handle secondary kernels after the main bottleneck improves

Do not do "guess-and-tune" optimization. Every modification in each round must point back to some bottleneck seen in the trace report.

## Keep / Discard Rule

All of the following conditions must be true before a modification may be kept:

1. Correctness passes
2. The fixed benchmark passes
3. The target config class or target kernel shows improved `rocprofv3` GPU time
4. Non-target config classes do not show clear and reproducible severe regressions
5. If config divergence appears, it has already been isolated through specialized kernel + dispatch, rather than forcing all configs to share one regressive version
6. `/format-code` has been executed
7. After formatting, the result is still better

Otherwise, discard the modification immediately and revert it.

If `err_flydsl_ps != 0`, do not discard it immediately; you must first follow `Correctness Repair Rule` and attempt up to `3` repairs. Only after `3` repairs still fail to restore it to `0` should the optimization proposal be abandoned.

If script-side `us_flydsl_ps` improves but `rocprofv3` GPU kernel time does not improve, or gets worse, **do not keep it**.

If only some configs improve while other configs clearly regress, **do not directly keep a single-kernel compromise version**. Prefer splitting into specialized kernels and letting dispatch route different configs to their more suitable implementations.

If there are `5` consecutive measured performance regressions, end this autonomous optimization round and stop trying new modifications.

## Logging Results

The results table in `performance.md` should record only two kinds of results:

- Baseline
- A real kept improvement

Do not write failed attempts, crashed attempts, or regressive attempts into the results table.

If optimization stops because of "5 consecutive measured performance regressions," you may append a separate `Stop Summary` block to `performance.md`. This block is not part of the results table; it is only used to explain why optimization stopped.

Each kept record must include at least:

- Date
- Commit
- Status: `baseline` or `keep`
- Benchmark command
- Target config class or dispatch condition
- Kernel name
- `us_flydsl_ps`
- Average `rocprofv3` GPU time
- Percentage change relative to the previous best
- Short explanation: which bottleneck this change targeted and what it did

If it is a split-kernel solution, the record should also explain:

- Which configs continue to use the old kernel
- Which configs switch to the new kernel
- Which config conditions are served by the old and new kernels respectively

Example:

```text
| 2026-04-16 | abc1234 | keep | python tests/kernels/test_pa.py | pa_persistent_fwd | 142.8 | 118.4 | -6.2% | hoist next-iteration loads to hide vmcnt stall |
```

If a stop condition is triggered, append the following to `performance.md`:

- Stop time
- Stop reason: `5 consecutive measured regressions` or `no idle GPU available for 60 minutes`
- Current `BEST_COMMIT`
- Current best metrics
- Recent 5 regression summaries

For the recent 5 regression summaries, a markdown table is recommended with at least these columns:

- Attempt
- Hypothesis
- Changed Files
- `us_flydsl_ps` best -> candidate
- GPU Time best -> candidate
- Delta
- Notes

## Commit Policy

Committing is allowed only after confirming that "this round is faster than the current best."

Before committing, you must:

1. Run `/format-code`
2. First execute `GPU Availability Gate`, then rerun the fixed benchmark
3. Confirm that correctness still passes
4. Execute `GPU Availability Gate` again and confirm that `rocprofv3` GPU kernel time is still better

The commit message should explain why this change worked, rather than mechanically listing which files were changed.

## Never Stop

Once you enter the `Optimization Loop`, do not stop and ask the human whether to continue. You are an autonomous optimizer:

- Do the baseline first
- Let trace analysis drive modifications
- Roll back regressions
- Record, format, and commit improvements
- Then continue iterating forward from the new best

If there are `50` consecutive performance regressions, or if no idle GPU becomes available for a continuous `1` hour, then stop; otherwise keep going unless a human explicitly interrupts you.
