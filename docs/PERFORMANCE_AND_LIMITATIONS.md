# Performance and Limitations / 性能、口径与局限

> 本文全部为**历史冻结结论**（引用 `evidence/**`，sha256 见 [`EVIDENCE_MAP.md`](EVIDENCE_MAP.md)）。
> 本轮（2026-09-29）**没有**运行任何 GPU 实验、没有重新编译、没有重跑 P3 矩阵。

## 0. 公共条件与测量口径

- 硬件：单块 **RTX 5080（SM120，84 个 SM，L2 64 MB），WSL2**，驱动 610.62，**没有锁定 GPU 时钟**；
- 软件与负载：torch 2.9.1+cu130、triton 3.5.1、flash-attn 2.8.3、Qwen3-0.6B、bf16、H=1024、28 层；
- **不同口径的数字不能直接比较**：

| 记号 | 口径 |
|---|---|
| `dev` | nsys 测得的 kernel device duration（热 L2，boost clock） |
| `ncu` | Nsight Compute，默认锁 base clock 并在每个 pass 前 flush cache（冷 DRAM） |
| `ev-eager` / `ev-graph` | CUDA events 测得的 stream 时间 / 次调用；eager 口径包含 host 分发 |
| `host` | CPU 侧时间（NVTX 或 `perf_counter`） |
| `step` / `E2E` | 引擎单步或单批的墙钟时间 |

## 1. 不能借用的数字

作者此前的 CUDA Kernel Lab C6 记录过"unfused → fused 下降 **52%**"（NSYS：10465 → 5024 ns；FP16、K=4096、
N=512、torch 2.11）。这是 C6 **自己**的"两个 kernel 对一个 kernel"的比较：

- nano 的 baseline **本来就是单个融合 kernel**，所以 NV5 是**融合对融合**；
- 两者的契约（C6-R 对 N-C）、dtype、H、环境也都不同；
- → 这个数字**不能**当作 NV5 或 nano 的提速。

## 2. P2：kernel 层（VERIFIED 测量）

**graph 模式下的 `dev`**（nsys，5 个样本，每组 2000 个 kernel）：

| N | baseline | Triton | CUDA |
|---|---|---|---|
| 1 / 8 / 64 | 704 / 768 / 896 ns | +18% / +8% / +11% | +27% / +21% / +21% |
| 1379 | 3904 ns | **−11.5%** | **−14.8%** |
| 4096 | 9376 ns | −5.5% | −7.2% |
| 16384 | 162049 ns | +0.1% | +0.4% |

**`ncu`（冷 cache、base clock）下排序相反**：N=1379 时 baseline 为 **9.60 µs**、Triton **10.85 µs**、CUDA **10.43 µs**。
不 flush cache 的对照 NCU 测得 L2 命中率 99.5–99.9%，说明 microbenchmark 处于 L2 常驻状态。
为什么中等 N 下两种口径排序相反——可能解释是"L2 常驻时更少更宽的指令占优；DRAM 受限时 baseline 更高的占用率占优"——
属于 **HYPOTHESIS**，没有被隔离验证。

**SASS 与计数器**（VERIFIED）：

- 候选为 `LDG.E.128` / `STG.E.128`（每请求 16 个 sector），baseline 为 `LDG.E.64` / `STG.E.64`（8 个 sector）；
- 动态指令减少 **25%（CUDA）到 37%（Triton）**；
- **DRAM 读取字节数三者相同**——逻辑 load 次数减少 ≠ DRAM 流量下降。

**小 N 时 baseline 更快**：baseline 每行 256 线程，候选 128 线程，占用率分别约 16% 与 8%。
用 latency 掩盖能力来解释是 **HYPOTHESIS**；本项目**没有**做改变线程数的对照实验，也没有为追速度去改 kernel。

## 3. P2：host 路径与 E2E

**E2E**：真实引擎、按轨迹 teacher-forced、64 个 token；每种模式 **2 轮**，每轮 1 次 warm-up + 3 次测量；`step` 口径。

| 模式 · 负载 | baseline（r1/r2） | Triton 相对 baseline | CUDA 相对 baseline | baseline 两轮差 |
|---|---|---|---|---|
| graph · bs8 decode | 2.992 / 2.960 ms | −1.3% | −1.4% | 1.1% |
| graph · bs64 decode | 4.649 / 4.603 ms | +0.9% | +0.2% | 1.0% |
| eager · bs8 decode | 23.18 / 22.23 ms | −5.3% | −14.1% | 4.2% |
| graph 进程中的 bs8 prefill（prefill 本身走 eager） | 27.36 / 27.96 ms | −12.5% | −20.1% | 2.2% |

- **graph decode：不可观测**——差异与 baseline 自身两轮间的波动同量级，而且方向不一致。
- **Amdahl**（DERIVED，每个后端 1 次 nsys）：fused 站点只占 step 墙钟的 **1.2–1.45%**，即使这些 kernel 完全不耗时，
  上限也只有约 **1.5%**。实测候选在 bs8 decode 中反而多花 16.7 µs（Triton）与 21.4 µs（CUDA）。
- **eager 模式的收益来自 host，不是 kernel**：GPU busy/wall 约 0.09；插桩测得每步 fused 调用内的 host 时间为
  **6039 / 4624 / 2256 µs**（baseline / Triton / CUDA），由此预测的节省量与 E2E 实测差值吻合。
- **DeviceContext 现象**（VERIFIED）：nano 在 `model_runner.py` L38 留下一个 TorchFunctionMode，
  在该状态下 baseline 每次调用的 host 时间为 91–116 µs，去掉后为 61–64 µs；剩余约 20 µs 差距的原因 **UNVERIFIED**。

## 4. P3：预注册的 Hybrid 实验

- **假设来源**：这是看过 P2 数据**之后**提出的。阈值只来自 P2：上限 N ≤ 4096、下限 N ≥ 1024，其中 1024–1378 为插值。
- **计划**：在**任何性能测量之前**锁定并设为只读。
- **矩阵**：8 轮 × 4 个角色（B baseline、C cuda、H hybrid、B2 第二个 baseline 作空对照）；
  顺序为 4×4 循环拉丁方加逆序方；32 个进程共用一个编译缓存；每进程每负载 1 次 warm-up + 5 次重复。
- **统计**：同轮内配对 `d = (X − B)/B`（负值更好），对 8 个配对值计算 mean 的 95% percentile bootstrap CI（10 000 次重采样）。
- **主判据**：bs8_T8 与 bs8_T16 的完整单批耗时必须**同时**满足
  P1：CI 上界 < 0；P2：至少 7/8 轮为负；P3：`|mean| > 2·SD(d_null)`。

| 负载 | d_H mean | CI95 | 为负轮数 | 2·SD(d_null) | P1 / P2 / P3 |
|---|---|---|---|---|---|
| **bs8_T8** | **−3.34%** | [−5.94, −0.93] | **6/8** | **13.10%**（SD = 6.55%） | ✔ / ✘ / ✘ |
| **bs8_T16** | −1.37% | [−3.54, +1.06] | 6/8 | 4.74% | ✘ / ✘ / ✘ |
| bs8_T64（次要） | −1.52% | [−3.71, +0.32] | 6/8 | — | 只报告 |
| bs64_T8 / T64（负对照） | +0.31% / −0.10% | 都包含 0 | 4/8 | — | 无回归 |

**判定：`NV5_P3_HYBRID_COMPLETE_NO_STABLE_GAIN`——预注册的稳定加速不成立。**

- **第 8 轮的处理**：第 8 轮中有 3 个进程开始时 host 负载偏高，按协议**没有删除**。事后去掉第 8 轮的敏感性分析
  （只报告）不改变判定；而且在同样的 7 轮上，连**空对照**的 CI 都排除了 0，说明小样本下单看 CI 并不可靠。
- **纯 CUDA 后端**：C 在 bs8_T8 上为 −4.67%（8/8 轮为负），但它**不在**预注册假设内，**不作为结论**。
- **Hybrid 在 prefill step 上**：−3.1% 到 −9.3%；**在 decode 上**：中位数变化在 ±0.9% 以内，保持 baseline 量级。

**归因**（每后端 1 次插桩 nsys，含 NVTX 开销，只报告）：

| 后端 | prefill 中 56 个 kernel 的 device 时间 | 每次 fused 调用的 host 时间 |
|---|---|---|
| baseline | 234.0 µs | 84.5 µs |
| cuda | 202.8 µs | 37.2 µs |
| hybrid | 201.9 µs | 51.4 µs |

- Hybrid 与 CUDA 执行的是**同一个** kernel；
- Hybrid 每次调用约多 **14 µs** 的 Python 策略判断，这是 **DERIVED**，不是一次单独隔离的因果证明。
  因此它只拿到 CUDA 后端约 **74%** 的 host 节省；
- 剩下的收益又淹没在 bs8 短负载约 **6.7%** 的进程间 CV 中。

## 5. 三个独立的结论

| 结论 | 状态 |
|---|---|
| 集成与验证是否成功 | **成功**（VERIFIED） |
| 局部正向数据 | **存在，但有条件**：graph 中等 N 的 `dev`、eager 下的 host 优势、P3 的 prefill step |
| 预注册的 E2E 稳定收益 | **不成立** |

## 6. 局限

- 只在单块 RTX 5080、WSL2 上测过，**没有锁定时钟**；P2 的 E2E 只有 **2 轮**；P3 的 host 归因来自每个后端 **1 次**插桩运行。
- P2 E2E 只有 2 轮是硬限制：不要把 −14.1% 当成稳定结论。
- N-C 契约只在两套工具链上验证；M-2 的归因只在 ≤256 token 的短上下文上闭合。
- **HYPOTHESIS**：decode 尺寸下 baseline 更快的机制；NCU 与 nsys 排序相反的机制。
- **UNVERIFIED**：真实模型 decode 中候选单个 kernel 比 microbenchmark 更慢的原因；P3 第 8 轮 host 负载偏高的原因。
- 没有覆盖：其它 GPU、dtype、H、模型、TP>1、在线 serving、长上下文。
- **为什么止损**：kernel 的 Amdahl 上限只有约 1.5%，可观测的收益来自 host 路径（DeviceContext、Dynamo guard、
  prefill 不走 graph），这些都超出了"RMSNorm 后端"的范围。
