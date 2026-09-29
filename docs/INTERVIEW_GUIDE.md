# Interview Guide / 面试讲解与边界

> 技术简报，不是简历成品。所有数字均来自 [`EVIDENCE_MAP.md`](EVIDENCE_MAP.md) 中的冻结证据；
> 历史 GPU 结果本轮**未重跑**，引用时应说明"这是当时的测量，条件见 PERFORMANCE_AND_LIMITATIONS"。

## 主线：C6 独立优化 → 发现 nano 已融合 → 数值契约适配 → 双后端与分发 → 有判别力的正确性测试 → 模式差异最小复现 → P2/P3 性能取舍

1. **起点**：我先在独立的 CUDA Kernel Lab 里做过 C1–C6，其中 C6 把 unfused 的 residual-add + RMSNorm 融合成一个
   kernel，在它自己的实验里 unfused→fused 下降约 52%。
2. **发现**：转到 nano-vLLM 后发现它的 baseline **本来就是单个融合 kernel**（Inductor 生成的
   `triton_per_fused__to_copy_add_mean_mul_pow_rsqrt_0`，0 个 graph break）。所以这里没有"两合一"的空间。
3. **契约**：nano 源码字面上有两次 bf16 舍入（N-E），但 `@torch.compile` 之后 Inductor 把中间那次消掉了，
   实际运行的是 **N-C**（统计与归一化都用未舍入的 fp32 和，最后只舍入一次）。我此前 C6 用的是 C6-R
   （已舍入 residual 参与统计），与 nano 不兼容——约 23% 的 bf16 元素在最后一位不同，而 C6 的容差区分不了。
   于是我照着 N-C **重写**了 kernel。
4. **实现**：Triton 版（每行一个 program）+ CUDA 版（`uint4` 向量路径把 `s` 缓存在寄存器；标量路径重算 `s`，
   **绝不读回舍入后的 residual**），再加一个只在 `RMSNorm.forward` 做局部分发的入口。
5. **验证**：最终 logits 做门禁是**无效的**（M-CAL = `M_CAL_NONDISCRIMINATIVE`：第 0 层差 557.65 倍，3–4 层后饱和，
   KL 5.8e-4 对 5.9e-4）。改成用 8-oracle 因子设计唯一确定契约，再冻结 **3584 次真实调用**做逐调用验证：
   正对照 3584/3584 通过，C6-R 负对照 3584/3584 全部被检出。
6. **模式差异最小复现**：eager 对 graph 逐位比较的 M-2 失败了，但**baseline 也失败**。我定位到 graph 路径把
   `block_table` 补零到 16 列，FlashAttention 因此选了不同的 split-KV 方案；补零后的 eager 在 63/63 步上与 graph
   逐位相同。**我没有修改 Attention**，原 FAIL 保留，另登记了 M-2'。
7. **性能取舍**：kernel 层中等 N（1379–4096）快 5–15%，小 N 慢 8–27%，NCU 冷 cache 下排序相反。
   端到端上，nano 默认的 graph decode **观测不到收益**（Amdahl 上限约 1.5%）；eager 下 CUDA 快约 14%，
   但来源是 host 分发开销而非 kernel。预注册的 Hybrid 实验判为 **NO_STABLE_GAIN**，我据此止损。

## 90 秒版

> 我在开源 nano-vLLM 上做了一个衍生研究：把 Qwen3 的 fused residual-add + RMSNorm 做成可切换后端，
> 用 Triton 和 CUDA 各写了一个 kernel。第一个发现是，nano 源码字面上有两次 bf16 舍入，但 `torch.compile`
> 之后 Inductor 把中间那次消掉了——实际契约是"统计与归一化都用未舍入的 fp32 和，最后只舍入一次"。
> 我用 8 个 oracle 的因子设计把这个契约唯一确定下来，并照它实现 kernel。第二个发现是，用最终 logits
> 做门禁是无效的：正确契约和错误契约的 KL 几乎一样，所以我改成冻结 3584 次真实调用做逐调用验证，
> 负对照在 3584 次调用上全部被检出。性能上，baseline 本来就是一个融合核；在 nano 默认的 CUDA Graph decode
> 下，这个算子只占单步的约 1.5%，**没有可观测的端到端收益**。我又做了一个预注册的 Hybrid 实验，
> 只在 prefill 用 CUDA，按事先定好的判据**也没有达到稳定增益**。

## 3 分钟版

1. **背景（30s）**：nano-vLLM 是上游的轻量推理引擎，调度、KV Cache、FlashAttention、CUDA Graph 都是它原有的。
   我只改了 RMSNorm 的分发入口（`layernorm.py`，+7/−1 行），新增了 `nv5_rmsnorm/` 和测试。
2. **数值契约（45s）**：源码 N-E（两次舍入）→ 编译后 N-C（一次舍入）；C6 的 C6-R 与 nano 不兼容；
   在两个 torch 版本上都用 8-oracle witness 唯一确定了 N-C；CUDA kernel 借鉴 C6 结构但重写了第二阶段。
3. **验证（45s）**：算子级 442 个用例；逐调用 3584 次冻结真实激活（正负对照都标定）；真实模型 56 个 site 的路由计数
   加 36 个 CUDA Graph 的节点枚举（replay 不经过 Python，所以必须用 driver API）。旧 logits 门禁判为
   NONDISCRIMINATIVE，记录保留。M-2 失败但 baseline 同样失败，最小复现后保留失败、另登记 M-2'。
4. **性能（45s）**：kernel 层 graph 中等 N 快 5–15%，小 N 慢 8–27%；SASS 显示 128-bit 访存、指令少 25–37%，
   但 DRAM 字节不变。端到端：graph decode 观测不到；eager 下 CUDA 快约 14%，来源是 host。
5. **P3 预注册实验（15s）**：8 轮拉丁方带空对照，三条判据要同时满足；结果 bs8_T8 为 −3.34%、CI [−5.94, −0.93]，
   但只有 6/8 轮为负、空对照 SD 6.55% → **没有稳定增益**，就此止损。

## 典型追问

| 问题 | 回答要点 |
|---|---|
| 为什么不直接用 C6 或 vLLM 的 kernel？ | 契约不同（C6-R 对 N-C），约 23% 元素在最后一位不同；C6 原有 tolerance 区分不了 |
| 1 ULP 的差异为什么能接受？ | 归约顺序与 `rsqrt` 实现差异；baseline 自身相对 oracle 也有同样噪声（3584 次中 219 次）；阈值与错误契约相差 200 倍以上 |
| 为什么最终 token 相同不能证明正确？ | bf16 下 1 ULP 扰动经 3–4 层饱和；同契约与错误契约的 KL 几乎相等（5.8e-4 对 5.9e-4） |
| CUDA Graph 下怎么证明走的是哪个 kernel？ | replay 不执行 Python，用 `cuGraphGetNodes` / `cuGraphKernelNodeGetParams` / `cuFuncGetName` 枚举节点 |
| M-2 失败是你的问题吗？ | 不是。baseline 同样失败；最小复现只改了 FA 的 `block_table` 宽度；我没有修改 Attention，也只在受测负载上成立 |
| 为什么 decode 时 baseline 更快？ | 事实：小 N 时 baseline 每行 256 线程、候选 128 线程，占用率 16% 对 8%。用 latency 掩盖解释是 **HYPOTHESIS**，没做对照实验 |
| −3.34% 不算加速吗？ | 预注册三条判据只满足一条：2 轮为正、空对照 SD 6.55%。按事先定的规则就是没有稳定增益，不能事后改口径 |
| 你改了 nano 的调度器 / KV Cache / Attention 吗？ | **没有**。那些全部来自上游；我只改了 RMSNorm 的分发入口 |
| 下一步做什么？ | 不再打磨 RMSNorm（Amdahl 上限约 1.5%）。可观测收益在 host 路径：DeviceContext、Dynamo guard、prefill 不走 graph，需要新计划 |

## 数字隔离：可以说 / 不能说

| 可以说 | 不能说 |
|---|---|
| "C6 在它自己的实验里 unfused→fused 下降 52%（FP16、K=4096、torch 2.11）" | "我让 nano 快了 52%"：nano 的 baseline 本来就是融合 kernel |
| "graph 中等 N、热 L2 时，kernel 的 device 时间快 5–15%" | "自研 kernel 比 Inductor 快 15%"：有条件，小 N 更慢 |
| "eager 路径下 CUDA 后端让 bs8 decode 的 step 缩短约 14%（P2，2 轮），来源是 host" | "推理加速 14%"：nano 默认的 graph decode 上观测不到 |
| "P3 预注册实验：bs8_T8 −3.34%，CI [−5.94, −0.93]，6/8 轮，未达稳定增益标准" | "Hybrid 带来 3.3% 的端到端加速" |
| "定位并最小复现了 graph decode 中 FlashAttention split-KV 的选择差异" | "修复了 nano / vLLM / FlashAttention 的问题"：没有修复，只在受测负载上成立 |
| "在 nano-vLLM 上做的衍生研究实现，单 GPU、离线、研究性质" | "自研推理引擎"、"生产上线"、"适用于所有 FlashAttention 场景" |
| "N-C 契约在 torch 2.11/triton 3.6 与 torch 2.9.1/triton 3.5.1 上独立验证过" | "所有 PyTorch 版本都是这个行为"：其它版本 UNVERIFIED |

## 讲项目时不要做的事

- 不要说"我把 nano 的两个 kernel 融合成一个"——baseline 本来就是 1 个融合 kernel；
- 不要说"我实现了调度器 / KV Cache / Attention / CUDA Graph"——全部来自上游；
- 不要把 C6 的 52% 说成 NV5 的成绩；
- 不要把 −3.34% 说成已证实的加速；
- 不要隐藏 M-CAL 无效、M-2 FAIL、G8 FAIL 这些历史——它们是这个项目最有说服力的部分之一。
