# Numerical Contract and Tests / 数值契约与门禁

> 除注明外，本文所有结果均为**历史冻结结论**（引用 `evidence/**`，sha256 见
> [`EVIDENCE_MAP.md`](EVIDENCE_MAP.md)）。本轮（2026-09-29）**没有**重跑任何 GPU 实验。

## 1. 三个契约

记号：T 为 bf16 / fp16，`rnd_T` 为 round-to-nearest-even，`s = fp32(x) + fp32(residual)`，按行计算。

| 契约 | residual_out | RMS 统计量 | 归一化输入 | 乘 weight 前是否舍入 | 来源 |
|---|---|---|---|---|---|
| **N-E**：nano 源码的字面语义 | `rnd_T(s)` | 未舍入的 s | s | **是**（两次舍入） | 上游 `layernorm.py` L34-40 逐行执行 |
| **N-C**：nano 实际运行的语义（**NV5 的目标**） | `rnd_T(s)` | **未舍入的 s** | **s** | **否**（单次舍入） | `@torch.compile` → Inductor |
| **C6-R**：CUDA Kernel Lab 的舍入残差契约 | `rnd_T(s)` | `rnd_T(s)` | `rnd_T(s)` | 否 | 作者此前的独立项目 C6 |

```
N-C:  residual_out = rnd_T(s)
      inv          = rsqrt( sum(s·s)/H + eps )     # s 未舍入
      y            = rnd_T( (s·inv) · fp32(w) )    # 只在最后舍入一次
```

**为什么源码的第二次舍入会消失**（VERIFIED，torch 2.9.1 / triton 3.5.1）：Inductor 生成的代码把源码里的
`x.to(orig_dtype)` 降级成 `tmp16 = tmp15.to(tl.float32)`，也就是没有舍入。在同一段函数体上：

- 默认编译与 eager 相比，bf16 有 **26.1%** 的元素不同；
- 设置 `emulate_precision_casts=True` 后，差异变为 **0**。

该结论**依赖工具链**：在 torch 2.11 / triton 3.6（P0）与 torch 2.9.1 / triton 3.5.1（P0.5）上各自独立测得 N-C；
其它版本 **UNVERIFIED**。不能把它当成"所有 PyTorch 版本的定律"。

**C6-R ≠ N-C**：两者 `residual_out` 完全相同，差别只在 `y` 上。F1 随机输入下约 **23%** 的 bf16 元素不同，
中点输入下约 **76%** 不同；C6 原有的 tolerance（bf16 为 1e-1）**无法区分**这两种契约。
→ NV5 不能直接复用 C6 kernel，也不能用 C6 的数字来描述 NV5。

**fp32 的上游缺陷**：fp32 下 `x.float()` 返回输入本身，于是输入被原地修改，返回的 `y` 与 `residual_out` 是
同一个张量。因此 fp32 不能当 oracle；NV5 的后端**拒绝 fp32**，而 baseline **保持上游行为不变**（OP-D 会验证这一点）。

## 2. baseline 的形态（P0.5，VERIFIED）

- decode N=1–512 与 prefill N=28–16384，每次调用都只有 **1 个** kernel：
  `triton_per_fused__to_copy_add_mean_mul_pow_rsqrt_0`，**0 个 graph break**。
- 测量方法：`torch.profiler` 在该环境拿不到 GPU 事件，因此把一次调用捕获进 `CUDAGraph(keep_graph=True)`，
  再用 driver API 枚举节点。
- 这意味着 NV5 比的是**融合对融合**，没有"消除 launch"或"融合带来的收益"可以宣称。

## 3. 判别契约的方法

**8-oracle 因子设计**：用三个二元舍入边界（统计量来源、归一化来源、乘 weight 前是否舍入）组合出 8 个 oracle，
N-C = (U, U, N)。判定规则在测量**之前**固定：每个输入族都满足 ≤1 ULP 且 ≤1e-3 才算 MATCH，并且必须
**恰好一个** oracle MATCH。

结果：baseline **唯一匹配 N-C**；最差值为 bf16 **1.14e-5** / 1 ULP、fp16 **9.16e-5** / 1 ULP；
其余 7 个 oracle 的最差不一致比例为 **24–78%**。

| 输入族 | 隔离的边界 |
|---|---|
| F2：中点输入，w=1 | 归一化来源 |
| F4：残差为 0 | 乘 weight 前是否舍入 |
| F5：同向偏移 +0.75 ulp | 统计量来源 |

**算子门禁**（P0.5 固定，之后从未修改）：

| 门禁 | 判据 |
|---|---|
| OP-R | residual 输出与参考逐位相同（0 bit 不同） |
| OP-Y | 相对 N-C oracle：≤1 ULP 且不一致比例 ≤1e-3 |
| OP-B | 相对 baseline：≤1 ULP 且 ≤2e-3 |
| OP-W | 8-oracle 语义 witness 中唯一归类为 N-C |
| OP-A | 真实激活上满足 OP-R/Y，并唯一归类为 N-C |
| OP-M | 内存语义：不修改输入，输出是新的连续张量 |
| OP-F | 非法输入 fail-loud |
| OP-G | 单算子的 graph replay 与 eager 逐位相同，1 个节点，捕获期间不 JIT |
| OP-D | 默认路径与上游逐位相同 |

**为什么正确的 N-C 实现也会差 1 ULP**：`sum(s²)` 的归约顺序与 `rsqrt` 实现不同，当 `y` 恰好靠近 bf16 舍入
边界时会翻转最后一位。baseline **自身**相对 oracle 就有这种噪声：M-1' 的 3584 次调用里有 **219 次**存在 1 ULP 差异；
Triton 与 CUDA 相对 baseline 的最差差异为 **3.66e-4 / 1 ULP**。

## 4. 门禁历史（按发现顺序；历史失败**全部保留**）

| 阶段 | 门禁 | 结果 | 要点 |
|---|---|---|---|
| P1A | OP-D、M-0 | PASS | 148/148；真实模型 logits **512/512** 逐位相同（eager 与 graph） |
| P1A | **M-CAL**（用最终 logits 区分正确与错误契约） | 第一轮 INVALID → Amendment 1（只改协议：共享编译缓存 + V5）→ 第二轮 **`M_CAL_NONDISCRIMINATIVE`** | 第 0 层时 ref 与 wrong 相差约 **557.65** 倍，但 3–4 层后都饱和；最终 KL 为 **5.8e-4 对 5.9e-4**。**最终 logits 或 token 相同都不是契约证明** |
| P1A-R1 | M-1'（失败后预注册的 V2） | 正对照 **3584/3584 通过**；C6-R 负对照 **3584/3584 全部失败** | 3584 = 56 个 site × 64 步（prefill 56 次 @1379 行；decode 3528 次 @8 行）。最小不一致比例 **0.187**，是阈值的 **187 倍**。冻结的真实激活为 831 MB |
| P1B | OP-R/Y/B/W/A/M/F/G、M-1'、真实路由、36 个 graph 的节点、M-3 | Triton 与 CUDA **全部 PASS** | 442 个算子用例；eager 下 3584 次候选调用、**0 次** baseline 执行；每个 graph 中 56 个候选节点 |
| P1B | **M-2**：同一后端 eager 对 graph 逐位相同 | **FAIL**（**baseline 也 FAIL**） | eager 的 block table 是 1 列，graph 的是 16 列并补零，FlashAttention 因此选择了不同的 split-KV 方案。最小复现中 **10.2%** 的元素不同，强制 `num_splits=1` 后逐位相同 |
| P2 | M-2 归因 | 把 eager 补零到 16 列后，与 graph 在 **63/63** 步上逐位相同 | 只在原来的短上下文（≤256 token）负载上验证 |
| P2 | **M-2'**（看到失败之后单独登记） | PASS | 要求同一路径可重复、两条路径路由都正确、旧门禁全部回归；**原 M-2 FAIL 不改写** |
| P3 | G1–G7 | PASS | G4：**3584/3584**（prefill 56 次→CUDA，decode 3528 次→baseline）；G6：hybrid 的 36 个 graph 节点与 baseline 完全相同，候选节点为 **0** |
| P3 | **G8**：V5 按字面要求 best_config 集合不变 | **FAIL**（13 → 14）→ Amendment 1 定义 G8' → **PASS** | 新增条目是采样 kernel（xnumel=151936）首次编译的产物，与 RMSNorm 无关；这份修订案登记在**任何性能测量之前** |

## 5. 三层验证的关系（也是"为什么最终 logits 不够"的答案）

1. **算子级**（OP-R/Y/B/W/A/M/F/G）：442 个用例，覆盖 dtype、形状、内存语义、fail-loud、graph replay；
2. **逐调用**（M-1'）：3584 次**冻结的真实激活**，正对照与负对照都标定过——这是真正有判别力的一层；
3. **真实模型**：56 个 fused 站点的路由计数，加上用 driver API 枚举 36 个 CUDA Graph 的节点
   （因为 replay 不经过 Python，计数器看不到）。

M-CAL 的失败说明：只看最终 logits / token 会**同时放过**正确契约与错误契约，因此它没有被用来给候选"放行"。
