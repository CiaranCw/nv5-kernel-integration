# Architecture / 真实执行路径与代码导读

本文中的行号以 NV5 功能 tree `db8dc57d` 与上游 `bb823b3e` 为准（两者一致，因为文档提交没有改源码）。

## 1. 真实调用链（上游代码，NV5 未改）

```
LLMEngine.step
 └─ ModelRunner.run(seqs, is_prefill)                        nanovllm/engine/model_runner.py L214-220
     ├─ prefill: prepare_prefill → set_context(True, …)       L129-170（L169）
     ├─ decode : prepare_decode  → set_context(False, …)      L172-188（L186）
     ├─ run_model                                             L195-212
     │   ├─ prefill、enforce_eager 或 bs>512 → self.model(...)          直接执行 Python 前向
     │   └─ decode 且 bs≤512 → 写入 graph_vars（block_tables 为 16 列静态缓冲）→ graph.replay()
     └─ sampler → .tolist() → reset_context()                 L218-219

启动时：warmup_model → run(…, True)，以 16384 行做一次 prefill        L91-101
        capture_cudagraph：对 36 个 bs（1,2,4,8,16,32,…,512）          L222-257
            set_context(False) → 1 次 warm-up → 捕获 1 次前向 → reset_context()

Qwen3Model.forward（nanovllm/models/qwen3.py L173-183）
  第 0 层 input_layernorm(h)                → RMS-only
  第 1–27 层 input_layernorm(h, residual)   → fused（27 次）
  每层 q_norm / k_norm（3-D 非连续视图）     → RMS-only（56 次）
  每层 post_attention_layernorm             → fused（28 次）
  最终 norm(h, residual)                    → fused（1 次）
  ⇒ Qwen3-0.6B 每次前向：fused 56 次、RMS-only 57 次（共 113 个 RMSNorm 模块）
```

### CUDA Graph 的关键事实（决定了 Hybrid 的形状）

- Python 里的路由判断**只在捕获时执行**：捕获时 `is_prefill=False` 且 stream 处于 capturing 状态；
- replay **不会**再执行 Python 前向，因此 decode graph 里的 kernel 在捕获时就固定了；
- 想证明 decode 到底走了哪个 kernel，只能**枚举 graph 节点**（`cuGraphGetNodes` 等 driver API），
  Python 计数器看不到 replay。这也是 `nv5/tests/test_hybrid.py` 里 `graph_kernel_names()` 存在的原因。

## 2. NV5 的分发入口

```
RMSNorm.forward(x, residual)
  ├─ residual is None                        → 上游 rms_forward        （RMS-only 永远不被替换）
  ├─ _nv5_impl is None                       → 上游 add_rms_forward    （baseline）
  ├─ _nv5_route 存在且返回 False              → 上游 add_rms_forward    （Hybrid 的策略路由）
  └─ 否则                                    → _nv5_impl(x, residual, weight, eps)
```

- 绑定在 `RMSNorm.__init__` 里**只做一次**（`layernorm.py` L17-18），`forward` 期间不读环境变量；
- `NV5_RMSNORM_BACKEND` 未设置 / `""` / `baseline` → 走上游，**并且不会 import 任何候选模块**
  （因此默认路径不依赖 Triton 编译或 nvcc）；
- 其它任何值 → 抛 `NV5BackendError`，**无静默回退**；
- 被选中的 impl 内部抛错 → 直接向上传播，**不降级**到 baseline。

## 3. 逐文件导读

### 3.1 `nanovllm/layers/layernorm.py`（唯一被修改的上游文件）

- L4：`from nanovllm.layers import nv5_rmsnorm`
- L17：`self._nv5_backend, self._nv5_impl = nv5_rmsnorm.resolve()`
- L18：`self._nv5_route = nv5_rmsnorm.route_policy(self._nv5_backend)`（只有 `hybrid` 会拿到逐调用策略）
- L20–44：`rms_forward` / `add_rms_forward` 与上游**逐字相同**
- L46–56：`forward` 的三段判断（见 §2）

### 3.2 `nv5_rmsnorm/__init__.py`（registry 与选择语义）

- L35–39：保留名 `triton` / `cuda` / `hybrid`；`_MODULES` 提供懒加载路径；`POLICY_BACKENDS = ("hybrid",)`
- L71–81：`selected_backend()`——空值或 `baseline` 返回 baseline；测试专用名与未知名都抛错
- L84–95：`resolve()`——首次选择时 `importlib.import_module`，由模块导入时自行 `register_backend`；
  导入后仍未注册则抛 `NV5BackendNotImplementedError`，**不回退**
- L62–68：`route_policy()`——策略后端没注册策略就报错

### 3.3 `nv5_rmsnorm/_common.py`（共享输入契约）

`check_inputs`：CUDA 张量、bf16/fp16（**拒绝 fp32**）、三者 dtype 一致、2-D `[N,H]`、weight 为 `[H]`、
全部连续、同一设备。**不要求** 16 字节对齐——CUDA 后端遇到未对齐输入会自己走标量路径。

### 3.4 `nv5_rmsnorm/triton_backend.py`

- kernel L19–32：每行一个 program，整行驻留寄存器
  1. 载入 x、r 并转 fp32，`s = x + r`
  2. store `residual_out = rnd_T(s)`
  3. `inv = libdevice.rsqrt(sum(s*s)/H + eps)`（用未舍入的 `s`）
  4. store `rnd_T((s*inv)*w)`
- `BLOCK = next_pow2(H)`，`num_warps = clamp(BLOCK/256, 1, 16)`（H=1024 时为 4 warps）
- N 只体现在 grid 中，不是 kernel 参数 → 不同 N 不会触发重编译

### 3.5 `nv5_rmsnorm/cuda_backend.py` + `csrc/nv5_fused_add_rmsnorm.cu`

- `cuda_backend.py` L16–18：**导入时**用 `torch.utils.cpp_extension.load(...)` JIT 编译，
  即发生在 RMSNorm 构造阶段，早于 warm-up 与 graph 捕获（捕获期间不能 JIT）
- `.cu` 结构：
  - `block_sum`（L30–47）：warp 内 `__shfl_down_sync` + `__shared__` 暂存各 warp 部分和
  - **向量路径**（L55–97）：`uint4` 一次读 8 个 bf16，fp32 的 `s` 缓存在寄存器数组 `s[ITERS][8]`
  - **标量路径**（L99–116）：phase 2 重新读 x、r 再算一次 `s`，与 phase 1 逐位相同，
    **绝不读回舍入后的 `residual_out`**
  - `dispatch`（L127–147）：H=1024 时 `kv = 128` → `BLOCK=128, ITERS=1`
  - host 侧（L151–174）：`TORCH_CHECK`、使用 `getCurrentCUDAStream()`（这是能被 graph 捕获的前提）、`empty_like` 分配输出
- 编译选项：`-O3 -std=c++17 --expt-relaxed-constexpr`，目标 sm_120，**没有** fast-math

### 3.6 `nv5_rmsnorm/hybrid_backend.py`（策略，不是新 kernel）

`use_cuda(x, residual, weight)` 全部满足才返回 True：

1. `get_context().is_prefill` 为 True，且**不在** CUDA Graph 捕获中（`torch.cuda.is_current_stream_capturing()`）
2. 2-D、bf16、`H == 1024`、`1024 ≤ N ≤ 4096`
3. residual 与 weight 的 dtype / shape 匹配、同一设备、连续
4. x / residual / weight 的 `data_ptr()` 均 16 字节对齐

注册的 impl 与 `cuda` 后端是**同一个函数对象**。返回 False 是有意的策略路由，不是出错回退。

### 3.7 `nv5_rmsnorm/reference.py` + `testing.py`（仅测试）

- `reference_nc`：用 fp64 归约实现 N-C，作为**正对照**（不受 baseline 归约顺序影响）
- `reference_wrong_c6r`：先把 `s` 舍入再做统计与归一化，作为**负对照**
- 只能通过 `testing.bind_test_backend(model, name)` 绑定，且需要 `NV5_RMSNORM_ENABLE_TEST_BACKENDS=1`；
  通过环境变量**无法**选中它们

## 4. 测试矩阵（历史记录，本轮未重跑）

| 文件 | 历史检查项 | 内容 |
|---|---|---|
| `nv5/tests/test_dispatch.py` | 33 | 选择语义、测试专用绑定、reference 语义 |
| `nv5/tests/test_opd_baseline_transparency.py` | 148 | OP-D：默认路径与上游 `layernorm.py` 逐位相同，含 RMS-only 与 fp32 的别名行为 |
| `nv5/tests/test_backends.py` | 74 | Triton/CUDA 的数值、内存语义、fail-loud、graph replay 与 eager 逐位相同 |
| `nv5/tests/test_hybrid.py` | 54 | 策略真值表、模块路径、错误传播、prefill 上下文捕获只得到 baseline 节点 |

- 这些测试需要 CUDA GPU 与对应依赖，**不需要**模型或任何私有数据；
- `test_opd_baseline_transparency.py` 运行时执行 `git show bb823b3e:nanovllm/layers/layernorm.py`，
  因此要在**带有完整上游历史的 clone** 里运行——也就是 `bootstrap.sh` 重建出来的那棵树；
- 历史通过记录全部在 RTX 5080 上取得（见 [`REPRODUCIBILITY.md`](REPRODUCIBILITY.md)）。

## 5. 一处上游现象（理解 P2 host 开销与 M-2 的关键）

- `model_runner.py` L38：`torch.set_default_device("cpu")` 会在 TorchFunctionMode 栈上留下一个
  `DeviceContext`。在这种状态下 baseline 每次调用的 host 时间为 91–116 µs，去掉该 mode 后为 61–64 µs
  （VERIFIED；剩余约 20 µs 差距的原因 **UNVERIFIED**）。
- `block_table` 的两种宽度：eager 用 `prepare_block_tables`（宽度 = 本批最长序列块数），
  graph 用静态缓冲 `zeros(max_bs, ceil(max_model_len/256))` 并补零。这个宽度差异正是**原 M-2 FAIL 的根因**
  （详见 [`NUMERICAL_CONTRACT_AND_TESTS.md`](NUMERICAL_CONTRACT_AND_TESTS.md)）。

NV5 **没有修改**以上任何一处上游行为。
