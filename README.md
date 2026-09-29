# NV5 Kernel Integration — 在 nano-vLLM 上做的 fused residual-add + RMSNorm 后端研究

> **一句话**：在固定版本的开源 [nano-vLLM](https://github.com/GeeeekExplorer/nano-vllm) 之上，给 Qwen3 的
> **fused residual-add + RMSNorm** 路径加了一套可切换后端分发，按 nano **实际运行的数值契约（N-C）**自己写了
> Triton 与 CUDA kernel，再用算子级 / 逐调用 / 真实模型三层门禁验证正确性，最后用 profiler 和一个**预注册的 E2E
> 实验**回答"它到底能不能更快"。

> **定位（请先读这一段）**：这是上游 nano-vLLM 的**衍生研究实现**，不是自研推理引擎。调度器、KV Cache、
> Attention/FlashAttention、CUDA Graph 的捕获与 replay、采样全部来自上游，版权归原作者
> （MIT，`Copyright (c) 2025 Xingkai Yu`，见 [`LICENSE`](LICENSE) 与
> [`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md)）。NV5 只改了 RMSNorm 的**分发入口**
> （`layernorm.py`，+7 / −1 行），并新增 `nanovllm/layers/nv5_rmsnorm/` 与 `nv5/tests/`。
> 本仓库**不包含** nano-vLLM 的完整源码，也不包含它的 Git 历史。

---

## 结论先行（三个不能含糊的事实）

1. **正确性与集成是成功的（VERIFIED）**：Triton 与 CUDA 两个 kernel 在 442 个算子用例、3584 次冻结的真实激活
   调用、56 个真实模型的 fused 站点路由、以及 36 个 CUDA Graph 的节点枚举上全部通过。
2. **在 nano 默认的 CUDA Graph decode 上，不存在已证实的稳定端到端加速**。fused 站点只占单步墙钟的
   1.2–1.45%（Amdahl，DERIVED），graph decode 的 E2E 差异落在噪声之内。
3. **P3 预注册的 Hybrid 实验判定为 `NV5_P3_HYBRID_COMPLETE_NO_STABLE_GAIN`**。最好看的数字是
   bs8_T8 的 −3.34%（bootstrap CI `[−5.94%, −0.93%]`），但 8 轮里只有 **6/8** 轮为负，而空对照的 SD 是
   **6.55%**（2·SD = 13.10%）。按**测量之前就定好**的三条判据，它**不满足**稳定增益标准。
   → **−3.34% 不能被写成"Hybrid 带来 3.3% 的端到端加速"。**

---

## 1. 首屏架构：NV5 改了什么

```
上游 Qwen3-0.6B（28 层，bf16，H=1024）                ← 上游，NV5 未改
  └─ Qwen3Model.forward
       ├─ RMS-only 调用（57 次：input_layernorm@L0、q_norm/k_norm、…）  → 永远走上游 rms_forward
       └─ fused 调用（56 次：residual + RMSNorm）                        ← NV5 唯一介入点
            └─ RMSNorm.forward(x, residual)
                 │
                 ├─ NV5_RMSNORM_BACKEND 未设置 / "" / "baseline"（默认）→ 上游 @torch.compile add_rms_forward
                 │                                                       （Inductor 生成的单个融合 kernel）
                 ├─ "triton"   → NV5 Triton kernel（N-C 契约）            ← 自研
                 ├─ "cuda"     → NV5 CUDA kernel（N-C 契约，导入时 JIT）  ← 自研
                 └─ "hybrid"   → 逐调用路由策略：满足条件时走上面的 CUDA kernel，
                                 其余情况（含全部 decode 与 graph 捕获）走上游 baseline   ← 自研（策略）
```

**Hybrid 不是第四个 kernel**。它只是在"同一个 CUDA kernel"与"上游 baseline"之间做逐调用选择的一条
`bool use_cuda(x, residual, weight)` 策略（[`hybrid_backend.py`](src_snapshot/nanovllm/layers/nv5_rmsnorm/hybrid_backend.py)）。

分发在 `RMSNorm.__init__` 里**只做一次**，`forward` 期间不读环境变量；任何未知名都抛
`NV5BackendError`，**不做静默回退**。

---

## 2. 我真正新增 / 修改的代码（逐文件）

镜像位于 [`src_snapshot/`](src_snapshot/)，是从权威 NV5 功能 tree `db8dc57d` **逐字节提取**的冻结副本
（SHA256 见 [`integration/SOURCE_MANIFEST.tsv`](integration/SOURCE_MANIFEST.tsv)）。

| 文件 | 归属 | 说明 |
|---|---|---|
| [`nanovllm/layers/layernorm.py`](src_snapshot/nanovllm/layers/layernorm.py) | **上游文件 + NV5 局部修改** | 只加：1 行 import、2 行构造绑定、1 个 `forward` 分发分支（+7/−1）。上游 `rms_forward` / `add_rms_forward` 函数体与 `@torch.compile` **逐字未动** |
| [`nv5_rmsnorm/__init__.py`](src_snapshot/nanovllm/layers/nv5_rmsnorm/__init__.py) | NV5 原创 | 后端 registry、环境变量选择、策略接口、fail-loud 语义 |
| [`nv5_rmsnorm/triton_backend.py`](src_snapshot/nanovllm/layers/nv5_rmsnorm/triton_backend.py) | NV5 原创 | Triton kernel（每行一个 program，整行驻留寄存器） |
| [`nv5_rmsnorm/cuda_backend.py`](src_snapshot/nanovllm/layers/nv5_rmsnorm/cuda_backend.py) | NV5 原创 | `torch.utils.cpp_extension` JIT 加载的 Python wrapper |
| [`nv5_rmsnorm/csrc/nv5_fused_add_rmsnorm.cu`](src_snapshot/nanovllm/layers/nv5_rmsnorm/csrc/nv5_fused_add_rmsnorm.cu) | NV5 原创（重写） | CUDA kernel：向量路径（`uint4`，`s` 缓存在寄存器）+ 标量路径。结构借鉴作者此前的 CUDA Lab C6 思路，**没有复制 C6 的代码或二进制**，且契约不同（N-C ≠ C6-R） |
| [`nv5_rmsnorm/hybrid_backend.py`](src_snapshot/nanovllm/layers/nv5_rmsnorm/hybrid_backend.py) | NV5 原创 | Hybrid 路由策略（不是新 kernel） |
| [`nv5_rmsnorm/_common.py`](src_snapshot/nanovllm/layers/nv5_rmsnorm/_common.py) | NV5 原创 | 共享输入契约校验（拒绝 fp32） |
| [`nv5_rmsnorm/reference.py`](src_snapshot/nanovllm/layers/nv5_rmsnorm/reference.py) | NV5 原创 | 纯 torch 的 N-C 正对照与 C6-R 负对照（仅测试用） |
| [`nv5_rmsnorm/testing.py`](src_snapshot/nanovllm/layers/nv5_rmsnorm/testing.py) | NV5 原创 | 把 reference 绑定到已建好模型的测试入口 |
| [`nv5/tests/test_dispatch.py`](src_snapshot/nv5/tests/test_dispatch.py) | NV5 原创 | 分发语义、reference 语义（历史 33 项） |
| [`nv5/tests/test_opd_baseline_transparency.py`](src_snapshot/nv5/tests/test_opd_baseline_transparency.py) | NV5 原创 | OP-D：默认路径与上游 `bb823b3e` 逐位相同（历史 148 项） |
| [`nv5/tests/test_backends.py`](src_snapshot/nv5/tests/test_backends.py) | NV5 原创 | Triton/CUDA 数值、内存语义、fail-loud、graph replay（历史 74 项） |
| [`nv5/tests/test_hybrid.py`](src_snapshot/nv5/tests/test_hybrid.py) | NV5 原创 | 策略真值表、模块路径、错误传播、图捕获节点（历史 54 项） |

`src_snapshot/` 是**供代码审阅的冻结镜像**，不是可以直接 `import nanovllm` 的完整 Python 包。
唯一权威的可执行集成方式是下面的"固定上游 + 精确补丁"。

---

## 3. 怎么重建（最短可操作流程，纯 CPU 即可）

```bash
git clone https://github.com/CiaranCw/nv5-kernel-integration.git    # 私有仓库
cd nv5-kernel-integration

# 在一个全新的空目录里，重建“固定上游 + NV5 补丁”的功能 tree
./integration/bootstrap.sh /tmp/nv5-rebuild

# 静态验真（镜像 SHA256、补丁 SHA256、重建 tree）
python3 integration/verify_snapshot.py --repo /tmp/nv5-rebuild
```

`bootstrap.sh` 会：从**官方** `https://github.com/GeeeekExplorer/nano-vllm.git` 取固定的
`bb823b3e06983d71485a8e1f23715ebd87d98ef8`（**不会**自动跟进上游 main）→ 校验 commit 与 tree SHA →
校验补丁 SHA256 与路径注入 → `git apply --check` → 应用 → `git add -A` + `git write-tree` →
断言结果等于权威 NV5 功能 tree：

```
upstream tree   5ec2b4b8034b26f5cc9af73c13d520a2aa8774f0   (bb823b3e^{tree})
patch  sha256   96a87f5088eadff9b2b66248fb48f674ad40812510658822a21cdf075e26ce86
rebuilt tree    05480eb86b23065ac412f1325c587bb129abd2da   ==  db8dc57d^{tree}  ✅
```

目标目录已存在且非空时脚本**直接失败**，绝不 `rm -rf`；上游不可达 / SHA 不符 / 补丁被拒 / tree 不一致都非零退出。

**如果读者机器上没有 GPU、模型权重或冻结张量**：本节的两步静态验真仍可完整执行；但历史 GPU 实验
（M-1' 重放、nsys/ncu、E2E 矩阵）**无法直接重放**，见 [`docs/REPRODUCIBILITY.md`](docs/REPRODUCIBILITY.md)。

---

## 4. 数值契约：为什么不能直接用现成 kernel

源码字面是 **N-E**（两次 bf16 舍入），但 `@torch.compile` 之后 Inductor 把中间那次舍入**消掉了**；
实际运行的是 **N-C**（统计与归一化都用未舍入的 fp32 和，最后只舍入一次）：

```
N-C:  residual_out = rnd_T(s)                  s = fp32(x) + fp32(residual)
      inv          = rsqrt( sum(s·s)/H + eps )  ← 未舍入的 s
      y            = rnd_T( (s·inv) · fp32(w) ) ← 乘 weight 前不舍入
```

我此前在**另一个独立项目** CUDA Kernel Lab C6 里用的是 **C6-R**（用已舍入的 residual 参与统计），
与 nano 不兼容：两者 `residual_out` 完全相同，**y 在 F1 随机输入下约 23% 的 bf16 元素不同**，
中点输入下约 76%，而 C6 原有的 bf16 容差 1e-1 根本区分不出来。
→ 所以 NV5 的 kernel 是**照着 N-C 重写的**，不能拿 C6 的数字来描述 NV5。

契约用 **8-oracle 因子设计**唯一确定（N-C = (U,U,N)）：baseline 唯一匹配 N-C，其余 7 个 oracle 的最差
不一致比例为 24–78%。详见 [`docs/NUMERICAL_CONTRACT_AND_TESTS.md`](docs/NUMERICAL_CONTRACT_AND_TESTS.md)。

---

## 5. 性能：口径分开看，别混算

| 记号 | 口径 |
|---|---|
| `dev` | nsys 测得的 kernel device duration（热 L2、boost clock） |
| `ncu` | Nsight Compute，默认锁 base clock 且每 pass 前 flush cache（冷 DRAM） |
| `ev-eager` / `ev-graph` | CUDA events 的 stream 时间 / 次调用 |
| `host` | CPU 侧时间 |
| `step` / `E2E` | 引擎单步或单批墙钟 |

**kernel 层（P2，graph 模式 `dev`）**：中等 N 有收益，小 N 反而更慢。

| N | baseline | Triton | CUDA |
|---|---|---|---|
| 1 / 8 / 64 | 704 / 768 / 896 ns | +18% / +8% / +11% | +27% / +21% / +21% |
| 1379 | 3904 ns | **−11.5%** | **−14.8%** |
| 4096 | 9376 ns | −5.5% | −7.2% |
| 16384 | 162049 ns | +0.1% | +0.4% |

`ncu` 冷 cache 下排序**相反**（N=1379：baseline 9.60 µs、Triton 10.85 µs、CUDA 10.43 µs）——机制属 **HYPOTHESIS**。

**端到端（P2，2 轮）**：

| 模式 · 负载 | baseline | Triton | CUDA | baseline 两轮差 |
|---|---|---|---|---|
| graph · bs8 decode | 2.992 / 2.960 ms | −1.3% | −1.4% | 1.1% |
| graph · bs64 decode | 4.649 / 4.603 ms | +0.9% | +0.2% | 1.0% |
| eager · bs8 decode | 23.18 / 22.23 ms | −5.3% | −14.1% | 4.2% |
| graph 进程中的 bs8 prefill | 27.36 / 27.96 ms | −12.5% | −20.1% | 2.2% |

- **graph decode：不可观测**（差异与 baseline 自身两轮波动同量级且方向不一致）。
- **eager 的收益来自 host，不是 kernel**：插桩测得每步 fused 调用内的 host 时间为
  6039 / 4624 / 2256 µs（baseline / Triton / CUDA）。
- **Amdahl（DERIVED）**：fused 站点占 step 墙钟 1.2–1.45%，即使这些 kernel 完全免费，上限也只有约 1.5%。

**P3 预注册 Hybrid 实验**：8 轮 × 4 角色（B / C / H / B2 空对照）、4×4 循环拉丁方加逆序方、32 进程共享编译缓存、
每负载 1 次 warm-up + 5 次重复；判据为"CI 上界 < 0"且"≥7/8 轮为负"且"|mean| > 2·SD(d_null)"**同时**满足。

| 负载 | d_H mean | CI95 | 为负轮数 | 2·SD(d_null) | P1/P2/P3 |
|---|---|---|---|---|---|
| **bs8_T8** | **−3.34%** | [−5.94, −0.93] | **6/8** | **13.10%**（SD 6.55%） | ✔ / ✘ / ✘ |
| **bs8_T16** | −1.37% | [−3.54, +1.06] | 6/8 | 4.74% | ✘ / ✘ / ✘ |
| bs8_T64（次要） | −1.52% | [−3.71, +0.32] | 6/8 | — | 只报告 |
| bs64_T8 / T64（负对照） | +0.31% / −0.10% | 都包含 0 | 4/8 | — | 无回归 |

判定：**`NV5_P3_HYBRID_COMPLETE_NO_STABLE_GAIN`**。

### 不能借用的数字

CUDA Kernel Lab C6 自己记录过 "unfused → fused 下降 52%"（NSYS：10465 → 5024 ns；FP16、K=4096、N=512、
torch 2.11）。那是 C6 **自己的**"两个 kernel 对一个 kernel"。nano 的 baseline **本来就是单个融合 kernel**，
NV5 比的是**融合对融合**，两者契约、dtype、H 也都不同 → **这个 52% 不能写成 NV5 或 nano 的加速。**

---

## 6. 门禁历史（失败全部保留，不改写）

| 阶段 | 门禁 | 结果 |
|---|---|---|
| P1A | OP-D / M-0 | PASS（148/148；真实模型 logits 512/512 逐位相同） |
| P1A | **M-CAL**（用最终 logits 区分契约） | 第一轮 INVALID → Amendment 1 → 第二轮 **`M_CAL_NONDISCRIMINATIVE`**。第 0 层 wrong/ref 相差 557.65 倍，3–4 层后饱和，最终 KL 5.8e-4 对 5.9e-4 → **最终 token/logits 相同不是契约证明** |
| P1A-R1 | M-1'（3584 次冻结的真实调用） | 正对照 **3584/3584 通过**；C6-R 负对照 **3584/3584 全部被检出**（最小不一致比例 0.187 = 阈值的 187 倍） |
| P1B | OP-R/Y/B/W/A/M/F/G、M-1'、真实路由、36 个 graph 节点、M-3 | Triton 与 CUDA **全部 PASS**（442 个算子用例） |
| P1B | **M-2**（eager 对 graph 逐位相同） | **FAIL**（baseline 也 FAIL）。根因：eager 的 `block_table` 是 1 列、graph 的是 16 列补零，FlashAttention 因此选了不同的 split-KV 方案；最小复现 10.2% 元素不同，强制 `num_splits=1` 后逐位相同 |
| P2 | M-2 归因 | eager 补零到 16 列后与 graph 在 **63/63** 步上逐位相同（只在受测短上下文负载上） |
| P2 | **M-2'**（失败后单独登记） | PASS。**原 M-2 FAIL 不改写** |
| P3 | G1–G7 | PASS（G4：prefill 56 次→CUDA、decode 3528 次→baseline；G6：hybrid 的 36 个 graph 节点与 baseline 完全相同，候选节点 0） |
| P3 | **G8**（字面要求 best_config 集合不变） | **FAIL**（13 → 14，新增条目是采样 kernel 首次编译产物）→ **在任何性能测量之前**登记 Amendment 1 定义 G8' → **G8' PASS**。历史两行都保留 |

每条关键数字都能在 [`docs/EVIDENCE_MAP.md`](docs/EVIDENCE_MAP.md) 里查到证据文件路径与 SHA256。

---

## 7. 局限（请一并阅读）

- 只在**单块 RTX 5080（SM120）、WSL2** 上测过，**没有锁定 GPU 时钟**；P2 的 E2E 只有 **2 轮**；P3 的 host 归因来自每个后端 **1 次**插桩运行。
- N-C 契约只在两套工具链上独立验证（torch 2.11/triton 3.6 与 torch 2.9.1/triton 3.5.1），**不能推广成所有 PyTorch 版本的定律**。
- M-2 的归因只在 **≤256 token 的短上下文**上闭合；没有修复上游或 FlashAttention 的任何问题。
- **HYPOTHESIS**（机制未经隔离验证）：小 N 时 baseline 更快的机制（256 线程/行 vs 128 线程/行，占用率 16% vs 8%）；NCU 与 nsys 排序相反的机制。
- **UNVERIFIED**：真实模型 decode 中候选 kernel 比 microbenchmark 更慢的原因；P3 第 8 轮 host 负载偏高的原因。
- 未覆盖：其他 GPU / dtype / H / 模型、TP>1、在线 serving、长上下文。
- **约 14 µs/调用的 Hybrid 额外 host 时间是 DERIVED**（由一次插桩运行的实测值推导），不是单独隔离的因果证明。
- 831 MB 冻结张量、1.5 GB 模型权重、5.9 GB venv、约 19 GB 编译缓存与原始 profiler 产物**都在作者本地**，不在本仓库。

---

## 8. 导航

| 文档 | 内容 |
|---|---|
| [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) | 真实调用链、代码归属、逐文件导读、CUDA Graph 捕获与 replay |
| [`docs/NUMERICAL_CONTRACT_AND_TESTS.md`](docs/NUMERICAL_CONTRACT_AND_TESTS.md) | N-E / N-C / C6-R 三个契约、8-oracle 设计、门禁定义与失败史 |
| [`docs/PERFORMANCE_AND_LIMITATIONS.md`](docs/PERFORMANCE_AND_LIMITATIONS.md) | P2 kernel/host/E2E/Amdahl、P3 预注册实验、测量口径与局限 |
| [`docs/REPRODUCIBILITY.md`](docs/REPRODUCIBILITY.md) | 版本锁定、本机可跑什么、哪些只在本地 |
| [`docs/PROVENANCE.md`](docs/PROVENANCE.md) | 溯源：上游 pin、旧完整仓的关系、本轮独立验证了什么 |
| [`docs/EVIDENCE_MAP.md`](docs/EVIDENCE_MAP.md) | 结论 → 证据文件 → SHA256，以及访问限制 |
| [`docs/INTERVIEW_GUIDE.md`](docs/INTERVIEW_GUIDE.md) | 90 秒 / 3 分钟讲稿、常见追问、"能说 / 不能说" |
| [`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md) | 上游 MIT 归属、自研部分的许可边界、C6 与模型的关系 |

脚本：[`integration/bootstrap.sh`](integration/bootstrap.sh)（重建）、
[`integration/verify_snapshot.py`](integration/verify_snapshot.py)（静态验真）、
[`scripts/check_docs.py`](scripts/check_docs.py)（文档/证据一致性）、
[`scripts/scan_secrets.py`](scripts/scan_secrets.py)（敏感信息与大文件扫描）。

---

## 9. 溯源与访问限制

- 上游：`GeeeekExplorer/nano-vllm` @ `bb823b3e06983d71485a8e1f23715ebd87d98ef8`（MIT，`Copyright (c) 2025 Xingkai Yu`）。
- **完整的 nano-vLLM 集成工程在另一个私有仓库**（`CiaranCw/nv5-nanovllm-kernel-integration`，`main = a67ee1e2c6c907d33be0e39833ebba04c517998b`）。本仓库的 Git 历史是**从自己的根提交开始**的独立历史，**不导入**上游或旧功能提交。
- 该私有仓库与 CUDA Kernel Lab（`CiaranCw/cuda-kernel-lab`）**对无权限读者不可用**；本仓库不承诺可以直接访问它们。无权限读者可以用本仓库的 `bootstrap.sh` 从**公开上游**完成等效重建。
- 证据标签：**VERIFIED**（直接测得）/ **DERIVED**（由已验证数据计算）/ **HYPOTHESIS**（机制未隔离验证）/ **UNVERIFIED**（未确认）。
