# kernel-opt

这是一个参考 `autoresearch/program.md` 风格、面向单个 kernel 的自治优化程序。目标是在不破坏正确性的前提下，持续降低目标 kernel 的 GPU 执行时间；性能下降的尝试必须回退，性能提升的尝试才允许保留并提交。

## Fixed Target

本次固定使用下面这条命令，作为 **unit test + benchmark** 的唯一真值来源：

```bash
python tests/kernels/test_pa.py
```

整个优化过程中，**不要修改这条 benchmark 命令**，除非人类明确要求切换目标。

### Fixed Metrics

- 正确性门槛：脚本必须成功退出，且 `.csv` 中 `err_flydsl_ps == 0`
- 脚本侧基准指标：`.csv` 里的 `us_flydsl_ps`，越小越好
- 最终 keep/discard 指标：`/kernel-trace-analysis` 产出的 `rocprofv3` GPU kernel 平均时间，越小越好
- 噪声规则：小于 `1%` 的变化默认视为噪声，除非重复两次都同方向改善
- 如果 benchmark 输出里包含多组 config，不要只看一个总体平均值；必须按 config 维度拆开看，至少优先观察实际存在的 `block_size`、`mtp`、`head_size`，以及任何直接影响 dispatch 的列

`us_flydsl_ps` 用于快速回归检查；真正决定是否保留修改时，优先看 `rocprofv3` 的 GPU kernel 时间。不要因为 Python 侧计时看起来更快，就保留一个 GPU kernel 实际变慢的修改。

当 benchmark 返回很多 configs 时，不要把所有行简单压成一个数字。必须先观察 config 规律，识别哪些组合天然属于同一类，例如 `block_size=1024`、`mtp=1`、`head_size=128` 这类组合；后续优化、回归判断、是否拆 kernel，都要以这些 config 类别为单位来分析。

## Setup

开始前按下面步骤准备：
0. 将/FlyDSL作为root目录
1. 先约定一个 run tag，例如 `apr16-pa-ps`
2. 从当前 `HEAD` 创建一个专用 worktree 或专用分支，避免污染已有工作区。示例：

```bash
git worktree add ../kernel-opt-<tag> -b kernel-opt/<tag> HEAD
```

3. 在专用 worktree 中工作，不要在原始脏工作区直接做实验
4. 先阅读：
   - `tests/kernels/test_pa.py`
   - 该脚本最终 dispatch 到的 kernel 实现文件
   - 任何直接影响 launch config / tile / prefetch / LDS / wait 策略的辅助文件
5. 确认 `rocprofv3` 和 `rocm-smi` 都在 `PATH` 中可用；任一不可用都停止并告知人类
6. 如果 `performance.md` 不存在，则创建它
7. 记录当前提交为：

```bash
BEST_COMMIT=$(git rev-parse HEAD)
```

`BEST_COMMIT` 表示“当前最后一个已知正确且性能最好”的提交。后续所有失败或回归尝试，都要回到这里重新开始。

同时维护一个回归计数器：

```bash
CONSECUTIVE_REGRESSIONS=0
```

这里的“性能下降”特指：candidate 通过 correctness 和 benchmark，但相对当前 `BEST_COMMIT`，`rocprofv3` 的目标 kernel 平均 GPU 时间变差并且超过噪声阈值（`> 1%`）。只有这种情况才给回归计数器 `+1`。

同时维护一个最近回归摘要列表：

```bash
RECENT_REGRESSIONS=[]  # rolling window, max len = 5
```

每次出现“明确性能下降”时，都要向 `RECENT_REGRESSIONS` 追加一条摘要，至少包含：

- 日期/时间
- 基准提交：当前 `BEST_COMMIT`
- candidate 提交或工作树状态
- 本轮假设
- 修改的文件
- `us_flydsl_ps`：best -> candidate
- `rocprofv3` 目标 kernel 平均 GPU 时间：best -> candidate
- 相对 best 的变化百分比
- 本轮 trace 观察到的主要退化点

“连续”的定义如下：

- 明确性能下降：`CONSECUTIVE_REGRESSIONS += 1`
- 性能提升并被保留：`CONSECUTIVE_REGRESSIONS = 0`，同时清空 `RECENT_REGRESSIONS`
- 噪声范围内持平：`CONSECUTIVE_REGRESSIONS = 0`，同时清空 `RECENT_REGRESSIONS`
- benchmark 失败 / 指标缺失：不计入“性能下降”，也不写入 `RECENT_REGRESSIONS`；但因为没有形成一次可比较的性能下降，默认也将 `CONSECUTIVE_REGRESSIONS = 0`
- correctness 失败时，如果 `.csv` 里出现 `err_flydsl_ps != 0`，不要立刻丢弃本轮优化；先进入“最多 3 次修复 kernel 以恢复 accuracy”的子流程。只有 3 次修复后仍失败，才把这轮优化方案视为失败，并将 `CONSECUTIVE_REGRESSIONS = 0`

## GPU Availability Gate

任何会实际占用 GPU 的测试、benchmark、correctness retest、或 `/kernel-trace-analysis`，在启动前都必须先通过下面的 GPU 门禁：

1. 先用 `rocm-smi` 检查所有 GPU，寻找空闲 GPU
2. “空闲”的判断以 `rocm-smi` 能看到的 utilization / memory / process 信息为准；优先选择没有活跃计算进程、GPU 利用率接近空闲、显存占用低或稳定不增长的卡
3. 如果找到多张空闲 GPU，优先选择最空闲的一张，并把后续命令绑定到这张卡上运行，例如设置 `HIP_VISIBLE_DEVICES=<gpu_id>`
4. 如果当前没有空闲 GPU，不要抢占；先等待 `10` 分钟，再用 `rocm-smi` 重新确认
5. 最多等待 `1` 小时，也就是初次检查后再按 `10` 分钟间隔重试；只要在这段时间内找到了空闲 GPU，就继续任务
6. 如果满 `1` 小时仍没有可用 GPU，这不是一次普通 benchmark 失败，而是整个优化任务的停止条件：立即中止任务，不再继续新的优化思路

不要在 busy GPU 上跑测试，以免 benchmark 噪声失真，或者和他人的任务互相干扰。

## First Run: Baseline

第一轮必须是未修改代码的 baseline。

1. 先执行 `GPU Availability Gate`，拿到一张空闲 GPU，然后在这张 GPU 上运行固定 benchmark，并把完整输出写入日志：

```bash
python tests/kernels/test_pa.py
```

2. 确认：
   - 进程退出码为 `0`
   - 生成了 `.csv`
   - `err_flydsl_ps == 0`

3. 从 `.csv` 读取 baseline 指标：

```bash
python - <<'PY'
import pandas as pd
df = pd.read_csv("<filename>.csv")
if not (df["err_flydsl_ps"] == 0).all():
    raise SystemExit("correctness failed")
print(f"baseline_us_asm_fp8={df['us_flydsl_ps'].mean():.3f}")
PY
```

4. 再次执行 `GPU Availability Gate`，然后对同一条 benchmark 命令运行：

```text
/kernel-trace-analysis python tests/kernels/test_pa.py
```

5. 从 trace report 中记录：
   - hottest kernel 名称
   - 该 kernel 的平均 GPU 时间
   - arch VGPR / accum VGPR / SGPR / LDS
   - top bottlenecks

6. 把 baseline 写入 `performance.md`，状态记为 `baseline`

## Config-Aware Strategy

benchmark 返回的 `.csv` 可能包含很多 configs。自治优化时，必须先理解这些 config 的结构，再决定优化策略：

1. 先查看 `.csv` 里到底有哪些 config 列，不要假设字段一定固定
2. 优先按实际出现的 `block_size`、`mtp`、`head_size` 分组；如果还有其他列更能解释性能分层，也把它们纳入 config 分类
3. 为每一类 config 记录 baseline 的 correctness 和性能，不要只记录总体均值
4. 每一轮优化都必须明确：这轮是“全局优化”，还是只针对某一类 config 的专项优化
5. 如果一个 hypothesis 只对某类 config 有意义，就允许它只优化那一类，但必须显式记录目标 config 类别
6. 非目标 config 不能因为目标 config 的提升而被严重牺牲；不能用“一部分 config 提升很大”来掩盖“另一部分 config 明显回退”

所谓“同一类 config”，指的是会走同一套 launch / tile / kernel 决策，并且有相似性能行为的一组输入。分类要以 benchmark 输出中真实存在的列为准，而不是拍脑袋硬分组。

## What You Can Change

你可以修改：

- kernel 实现本身
- runtime dispatch 逻辑，让不同 config 类别走不同 kernel
- launch config
- tile / wave / split / partition 选择
- prefetch / double-buffer / software pipelining
- LDS layout / swizzle / padding
- wait/barrier 前后指令安排
- 向量化宽度
- 影响 occupancy 或内存延迟隐藏的局部代码
- 为某一类 config 单独新增一个 specialized kernel

你不可以修改：

- 固定 benchmark 命令
- benchmark shape
- 为了“跑得更快”而偷偷降低工作量或绕过 correctness
- 引入新的依赖
- 保留一个 GPU kernel 时间变差的修改

如果必须修改 benchmark 脚本本身，只能为了：

- 修复 benchmark harness
- 让指标更稳定、更可解析
- 修复明显错误的 correctness gate

不能为了让数字看起来更好而改 benchmark。

## Correctness Repair Rule

如果某一轮 candidate 运行后 `.csv` 存在，且 `err_flydsl_ps != 0`，说明这轮优化破坏了 accuracy。此时不要直接放弃当前思路，而是必须先尝试修复 kernel：

1. 保持当前优化假设不变，只允许做“让 accuracy 恢复通过”所必需的最小修复
2. 每修一次前都先执行 `GPU Availability Gate`，然后重新运行固定 benchmark，重新检查 `.csv` 里的 `err_flydsl_ps`
3. 同一轮优化方案最多允许连续修复 `3` 次
4. 只要某次修复后 `err_flydsl_ps == 0`，就回到正常的性能评估流程，继续比较 candidate 和 `BEST_COMMIT`
5. 如果修复了 `3` 次后 `err_flydsl_ps` 仍然不为 `0`，则放弃这次优化方案：回退到 `BEST_COMMIT`、不提交、不写入 `performance.md`，并继续下一个新的优化思路

这个“3 次修复上限”是针对单个优化方案的局部限制，不计入 `CONSECUTIVE_REGRESSIONS`。

## Kernel Split Rule

如果 benchmark 结果显示明显的 config 分化，例如某类 config 提升很大，而另一类 config 明显且可复现地回退，不要强迫所有 config 共用同一个优化版本。此时应优先考虑拆分：

1. 保留当前已知最优的通用 kernel 作为默认路径
2. 为受益明显的 config 类别新增一个 specialized kernel
3. 在 runtime dispatch 中按 config 条件选择 kernel，例如按 `block_size`、`mtp`、`head_size` 或其他真实存在的配置列进行分流
4. 让受益 config 走新 kernel，让其余 config 继续走旧的 best kernel
5. 拆分后重新运行固定 benchmark，验证所有 config 的 correctness，并重新比较各类 config 的性能

只有在“拆分 dispatch 之后，受益 config 明显更快，其他 config 没有被拖慢”时，才允许保留这种新增 kernel 的方案。

## Optimization Loop

从 baseline 开始，进入无限循环。除非被人类手动打断，否则不要停。

LOOP FOREVER:

1. 从 `BEST_COMMIT` 开始
2. 先执行 `GPU Availability Gate`，再运行或重新运行 `/kernel-trace-analysis`，确保 bottleneck 画像是最新的
3. 读取最新 benchmark `.csv`，按 config 维度重新检查哪些 config 类别最热、最慢、最容易回退；至少观察 `block_size`、`mtp`、`head_size`
4. 只选择 **一个** bottleneck 驱动的假设，作为本轮尝试；并明确这轮是“全局优化”还是“某个 config 类别专项优化”
5. 用一句话写清楚假设和目标 config 类别，然后再改代码，例如：
   - 提前发起下一轮 load，减少 `s_waitcnt vmcnt(0)` 暴露
   - 减少 LDS write/read round-trip，降低 `lgkmcnt` 和 barrier stall
   - 降低 arch VGPR 压力，提高 occupancy
   - 在 MFMA/VALU 之间插入独立工作，填补 idle bubble
6. 只修改支撑这个假设所必需的最小文件集合
7. 设置 `FIX_ATTEMPTS=0`，先执行 `GPU Availability Gate`，再重新运行固定 benchmark，并保存为 `candidate.log`
8. 如果 benchmark 失败或指标缺失：
   - 丢弃本轮尝试
   - 回退到 `BEST_COMMIT`
   - 把 `CONSECUTIVE_REGRESSIONS` 重置为 `0`
   - 清空 `RECENT_REGRESSIONS`
   - 继续下一个思路
9. 如果 benchmark 成功，先按 config 类别读取 `.csv`，不要只看总平均
10. 如果任一 config 类别里 `err_flydsl_ps != 0`：
   - 进入 correctness repair 子流程
   - 在保持当前优化方向不变的前提下修复 kernel，并令 `FIX_ATTEMPTS += 1`
   - 每修一次前都先执行 `GPU Availability Gate`，再重新运行固定 benchmark，重新检查 `err_flydsl_ps`
   - 只有当所有 config 类别都恢复到 `err_flydsl_ps == 0` 时，才跳到下一步继续性能评估
   - 如果 `FIX_ATTEMPTS >= 3` 后仍然失败：
     - 放弃这次优化方案
     - 回退到 `BEST_COMMIT`
     - 不提交
     - 不写入 `performance.md`
     - 把 `CONSECUTIVE_REGRESSIONS` 重置为 `0`
     - 清空 `RECENT_REGRESSIONS`
     - 继续下一个思路
11. 先执行 `GPU Availability Gate`，对 candidate 再跑一次 `/kernel-trace-analysis`
12. 对比 candidate 和当前 best 时，必须同时看“总体结果”和“分 config 结果”：
   - 所有 config 类别的 `err_flydsl_ps` 必须仍然为 `0`
   - 目标 config 类别必须确实改善
   - 非目标 config 类别不能出现明显且可复现的严重回退
   - `us_flydsl_ps` 不得靠牺牲其他 config 类别来换取局部提升
   - `rocprofv3` 的目标 kernel 平均 GPU 时间必须改善；如果是拆分 kernel，则要确认新 dispatch 下受益 config 更快、其他 config 不被拖慢
13. 如果 candidate 更好：
   - 把结果追加到 `performance.md`
   - 执行 `/format-code`
   - 格式化后先执行 `GPU Availability Gate`，再重新跑一次固定 benchmark
   - 如果格式化后的结果仍然更好，就提交代码和 `performance.md`
   - 更新 `BEST_COMMIT` 为新提交
   - 把 `CONSECUTIVE_REGRESSIONS` 重置为 `0`
   - 清空 `RECENT_REGRESSIONS`
14. 如果 candidate 出现“某类 config 提升很大，但其他 config 严重回退”的分化：
   - 不要保留这个单一 kernel 版本
   - 如果这种分化是稳定且可解释的，就按 `Kernel Split Rule` 改为新增 specialized kernel + dispatch，再重新评估
   - 如果暂时还没有拆分，就先回退到 `BEST_COMMIT`
15. 如果 candidate 持平或更差：
   - 回退全部修改，回到 `BEST_COMMIT`
   - 不提交
   - 不写入 `performance.md`
   - 如果是噪声范围内持平，则把 `CONSECUTIVE_REGRESSIONS` 重置为 `0`，并清空 `RECENT_REGRESSIONS`
   - 如果是“性能下降”而不是噪声范围内持平，则 `CONSECUTIVE_REGRESSIONS += 1`，并把本轮摘要追加到 `RECENT_REGRESSIONS`
   - 如果 `CONSECUTIVE_REGRESSIONS >= 5`，则停止优化，并把停止原因和最近 5 次回归摘要写入 `performance.md`
   - 否则继续下一轮

因为整个实验运行在专用 worktree / 专用分支中，所以在失败或回归时，可以直接把实验分支重置回 `BEST_COMMIT`。不要在有无关改动的共享工作区里这么做。

## Using `/kernel-trace-analysis`

每轮分析都对同一条固定命令执行：

```text
/kernel-trace-analysis python tests/kernels/test_pa.py
```

读 report 时，优先关注：

- 高周期 `s_waitcnt` / `s_barrier`
- 高周期 `buffer_load_*` 或其它 VMEM stall
- `ds_read_*` / `ds_write_*` / `lgkmcnt` 压力
- MFMA/VALU 前后的 idle bubble
- arch VGPR 与 accum VGPR 哪个在限制 occupancy

如果有多个 kernel：

- 先优化 GPU 时间占比最高的那个
- 如果 main kernel 与 reduce kernel 都很热，优先解决更大的那个
- 只有在主瓶颈改善后，才去处理次要 kernel

不要做“拍脑袋调参”。每一轮修改都必须能回指到 trace report 里的某个 bottleneck。

## Keep / Discard Rule

保留修改的条件必须全部成立：

1. correctness 通过
2. 固定 benchmark 通过
3. 目标 config 类别或目标 kernel 的 `rocprofv3` GPU 时间改善
4. 非目标 config 类别没有明显且可复现的严重回退
5. 如果出现 config 分化，已经通过 specialized kernel + dispatch 隔离，而不是强迫所有 config 共用一个退化版本
6. `/format-code` 已执行
7. 格式化后重新测试，结果仍然更好

否则直接丢弃修改并回退。

如果 `err_flydsl_ps != 0`，先不要立刻丢弃；必须先按 “Correctness Repair Rule” 最多修复 `3` 次。只有修复 `3` 次后仍未恢复到 `0`，才放弃该优化方案。

如果脚本侧 `us_flydsl_ps` 改善，但 `rocprofv3` GPU kernel 时间没有改善，或者更差，**不要保留**。

如果只有部分 config 改善，而另一些 config 明显回退，**不要直接保留一个单一 kernel 的折中版本**。应优先拆成 specialized kernel，并让 dispatch 把不同 config 导向各自更合适的实现。

如果连续 5 次出现明确性能下降，则结束本轮自治优化，不再继续尝试新的修改。

## Logging Results

`performance.md` 的结果表只记录两类结果：

- baseline
- 真正保留的 improvement

不要把失败尝试、崩溃尝试、或回归尝试写入结果表。

如果因为“连续 5 次明确性能下降”而停止，则允许在 `performance.md` 追加一个单独的 `Stop Summary` 区块。这个区块不属于结果表，只用于说明为什么停止。

每一条保留记录至少包含：

- 日期
- commit
- 状态：`baseline` 或 `keep`
- benchmark 命令
- 目标 config 类别或 dispatch 条件
- kernel 名称
- `us_flydsl_ps`
- `rocprofv3` 平均 GPU 时间
- 相比上一个 best 的变化百分比
- 简短说明：这次改动针对了哪个 bottleneck、做了什么

如果是 split-kernel 方案，记录里还应说明：

- 哪些 config 继续走旧 kernel
- 哪些 config 改走新 kernel
- 新旧 kernel 各自服务的 config 条件

示例：

```text
| 2026-04-16 | abc1234 | keep | python tests/kernels/test_pa.py | pa_persistent_fwd | 142.8 | 118.4 | -6.2% | hoist next-iteration loads to hide vmcnt stall |
```

如果触发停止条件，则在 `performance.md` 追加：

- 停止时间
- 停止原因：`5 consecutive measured regressions` 或 `no idle GPU available for 60 minutes`
- 当前 `BEST_COMMIT`
- 当前 best 指标
- 最近 5 次回归摘要列表

最近 5 次回归摘要建议用 markdown 表记录，列至少包括：

- Attempt
- Hypothesis
- Changed Files
- `us_flydsl_ps` best -> candidate
- GPU Time best -> candidate
- Delta
- Notes

## Commit Policy

只有在确认“这轮比当前 best 更快”之后才允许提交。

提交前必须：

1. 运行 `/format-code`
2. 先执行 `GPU Availability Gate`，再重新跑固定 benchmark
3. 确认 correctness 仍然通过
4. 再次执行 `GPU Availability Gate`，确认 `rocprofv3` GPU kernel 时间仍然更好

commit message 要写“为什么这次改动有效”，而不是机械地列出“改了哪些文件”。

## Never Stop

一旦进入优化循环，不要每做完一次尝试就停下来问人类要不要继续。你是自治优化器：

- baseline 先做
- trace 分析驱动修改
- 回归就回退
- 提升就记录、格式化、提交
- 然后继续从新的 best 往下迭代

如果连续 5 次出现性能下降，或者连续 `1` 小时都没有等到空闲 GPU，则停止；否则除非人类明确打断，一直优化下去。
