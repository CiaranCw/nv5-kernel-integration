# Evidence Map / 证据索引

`evidence/**` 中的每个文件都是旧完整集成仓中冻结副本的**逐字节拷贝**
（源：`docs/nv5/evidence/**`，原提交 `a67ee1e2c6c907d33be0e39833ebba04c517998b`）。
**本轮（2026-09-29）已独立重算全部 24 个文件的 sha256，与本表记录逐字一致**；也可用
`python3 scripts/check_docs.py` 重新校验（含关键字符串检查）。

标签：**VERIFIED**（直接测得）/ **DERIVED**（由已验证数据计算）/ **HYPOTHESIS**（机制未隔离验证）/ **UNVERIFIED**（未确认）。

| ID | 标签 | 阶段 | 结论 | 本仓库中的证据 | 关键字符串 | sha256 |
|---|---|---|---|---|---|---|
| E01 | VERIFIED | P0.5 | baseline 每次调用 1 个 Inductor kernel | [`p05/evidence/final_lock/baseline_kernel_profile.txt`](../evidence/p05/evidence/final_lock/baseline_kernel_profile.txt) | `triton_per_fused__to_copy_add_mean_mul_pow_rsqrt_0` | `9f933a955917b48491d3064ee021c2706ed2b5936958fd94f66cc94445bef5bf` |
| E02 | VERIFIED | P0.5 | 按输入族隔离舍入边界：各族下 N-C(UUN) 误差最小（bf16 最差 1.14e-5），其余 oracle 在各自判别族上的最差比例为 2.4e-1 到 7.8e-1 | [`p05/evidence/contract_boundary_isolation.txt`](../evidence/p05/evidence/contract_boundary_isolation.txt) | `N-C(UUN)` | `e1e6a8c0c5a0bb9956885edb905b264ca5c402b854b7d4112186aa7500208b22` |
| E03 | VERIFIED | P1A | **M-CAL = `M_CAL_NONDISCRIMINATIVE`**（永久保留，不是 PASS） | [`p1a/evidence/mcal/r2/mcal_result.json`](../evidence/p1a/evidence/mcal/r2/mcal_result.json) | `M_CAL_NONDISCRIMINATIVE` | `786bb93fa778d7ebc9fb1f822014ff86a0f547246a718b7820b1201eb4ab02f4` |
| E04 | VERIFIED | P1A | 逐层发散：第 0 层 hidden 的 wrong/ref 为 557.65，之后饱和 | [`p1a/evidence/mcal/r2/layer_divergence_summary.txt`](../evidence/p1a/evidence/mcal/r2/layer_divergence_summary.txt) | `557.65` | `6b2abd06eb14892f9418573039c7fa82da7c3b78d83e1017179c56dbaa4281c2` |
| E05 | VERIFIED | P1A-R1 | M-1' 标定：正对照 3584/3584 通过，负对照全部失败 | [`p1a_r1/evidence/m1/summary.json`](../evidence/p1a_r1/evidence/m1/summary.json) | `3584` | `11ac490cfe6b0b47616e6793ca3a01bb642de7df506783962b44e1c5ee024c9e` |
| E06 | VERIFIED | P1A-R1 | OP-W：正对照唯一 (U,U,N)，负对照唯一 (R,R,N) | [`p1a_r1/evidence/m1/op_w_witness.json`](../evidence/p1a_r1/evidence/m1/op_w_witness.json) | `R,R,N` | `536d91c87e5838447ef1cbcf4cbc76036b1b7ae7cab7ba0e94e9bef8445c3823` |
| E07 | VERIFIED | P1B | CUDA 的 OP 门禁（442 个用例） | [`p1b/evidence/op_gates/cuda/op_gates_summary.json`](../evidence/p1b/evidence/op_gates/cuda/op_gates_summary.json) | `OP_R` | `d1cb79372f177191d36b8f9f8eaad4dd96da9b88d25b654609131cd254cb5d29` |
| E08 | VERIFIED | P1B | Triton 的 OP 门禁（442 个用例） | [`p1b/evidence/op_gates/triton/op_gates_summary.json`](../evidence/p1b/evidence/op_gates/triton/op_gates_summary.json) | `OP_R` | `2efaf714c1da92ab2467c27f586ecce3386f2b12dc1c4eabbd4ab24553734545` |
| E09 | VERIFIED | P1B | M-2 最小复现：block table 1 列对 16 列，`num_splits=1` 后逐位相同 | [`p1b/evidence/model/m2_min_repro_fa_block_table.json`](../evidence/p1b/evidence/model/m2_min_repro_fa_block_table.json) | `num_splits` | `98e5f0c67275a6b9867648d4dd3a33ff3950dd67b9115b0445d7265e18583da6` |
| E10 | VERIFIED | P2 | 补零后的 eager 与 graph 在 63/63 步上逐位相同（CUDA 后端） | [`p2/evidence/m2_attribution/padded_eager_cuda.json`](../evidence/p2/evidence/m2_attribution/padded_eager_cuda.json) | `63` | `2ebca8755bb3c2a9325e6520a6491f8c17f98e885881d9d1c2a998f251de7aee` |
| E11 | VERIFIED | P2 | M-2' 结果（**原 M-2 FAIL 未被改写**） | [`p2/evidence/m2prime/m2prime_results.json`](../evidence/p2/evidence/m2prime/m2prime_results.json) | `PASS` | `174813de85c2958e1d36d63c74f77f4ab124ede057412d84f16a97a0421fd90f` |
| E12 | VERIFIED | P2 | nsys 算子层的 device 时长与 host 时间 | [`p2/evidence/nsys_op/nsys_op_summary.txt`](../evidence/p2/evidence/nsys_op/nsys_op_summary.txt) | `graph&#124;N=1379&#124;cuda` | `146a3a520b205ba0c1361b446398933b1b954d847790e3935bf7f32f691e3ff3` |
| E13 | VERIFIED | P2 | NCU 计数器（base clock、flush cache）：排序与 nsys 相反 | [`p2/evidence/ncu/ncu_summary.txt`](../evidence/p2/evidence/ncu/ncu_summary.txt) | `baseline_N1379` | `a5fa021d7fab860d6773cebf86c08ffad1c1e36effe461c84a6185722ee80259` |
| E14 | VERIFIED | P2 | SASS 全局访存宽度（LDG.E.128 对 LDG.E.64） | [`p2/evidence/sass/sass_ldst_summary.txt`](../evidence/p2/evidence/sass/sass_ldst_summary.txt) | `LDG.128` | `0da951b9be72908f3a14ae29b245471bf5db1fc8356d1644019ae9aeb7267861` |
| E15 | VERIFIED | P2 | 真实引擎 E2E（eager 与 graph，**仅 2 轮**） | [`p2/evidence/e2e/e2e_summary.txt`](../evidence/p2/evidence/e2e/e2e_summary.txt) | `graph bs8  decode_ms_median` | `495d21d527e6472fa88b34d3b388186c64638fdff3813f21f55c5be041fd2f03` |
| E16 | DERIVED | P2 | Amdahl：fused 站点占 step 1.2–1.45% | [`p2/evidence/amdahl/amdahl_summary.txt`](../evidence/p2/evidence/amdahl/amdahl_summary.txt) | `baseline_graph_bs8` | `dc930e48cb64b3b981bb2d66b524c3c58dfb417c44baf1203d8a96aabfc9ec9d` |
| E17 | VERIFIED | P2 | DeviceContext 对 host 开销的影响（baseline） | [`p2/evidence/host_overhead/baseline.json`](../evidence/p2/evidence/host_overhead/baseline.json) | `torch_function_stack_len_nano_state` | `5354fcd865affec26944aeeb91b78b000f4df8208f76fe85fb8ba7e4e1c16eb0` |
| E18 | VERIFIED | P3 | G4 M-1'-H：prefill 56 次→cuda，decode 3528 次→baseline | [`p3/evidence/gates/G4_m1h_summary.json`](../evidence/p3/evidence/gates/G4_m1h_summary.json) | `3528` | `7fbf3b9bc1c087eeab00d9f90fa24f617c049b96bab54d0b5daec620745d43c0` |
| E19 | VERIFIED | P3 | G6/G7/**G8'**：路由、graph 节点、确定性、V5（**原 G8 字面 FAIL 保留**） | [`p3/evidence/gates/G6_G7_G8_impact.json`](../evidence/p3/evidence/gates/G6_G7_G8_impact.json) | `"PASS": true` | `ce478f7f8be5df86fd3bd24bffd18a8deeb8c37d5f74de4609d1c248df1abbf4` |
| E20 | VERIFIED | P3 | 预注册矩阵分析结果：**`NV5_P3_HYBRID_COMPLETE_NO_STABLE_GAIN`** | [`p3/evidence/perf_matrix/e2e_analysis.json`](../evidence/p3/evidence/perf_matrix/e2e_analysis.json) | `NV5_P3_HYBRID_COMPLETE_NO_STABLE_GAIN` | `58d932d756d1c22fff22350762d9a74a3b187442cbb2d658a0d80893978165a6` |
| E21 | VERIFIED | P3 | 逐轮原始值与配对统计表（bs8_T8 −3.34%、6/8、SD 6.55%） | [`p3/evidence/perf_matrix/perf_tables.md`](../evidence/p3/evidence/perf_matrix/perf_tables.md) | `bs8_T8` | `17f77bbcc786a216bcbe9e29fe2576789ccf50a22af5b5289c21609c6334448c` |
| E22 | VERIFIED | P3 | 事后敏感性分析（只报告，不改变判定） | [`p3/evidence/perf_matrix/sensitivity_excl_r8_POSTHOC.txt`](../evidence/p3/evidence/perf_matrix/sensitivity_excl_r8_POSTHOC.txt) | `POST-HOC` | `bd272a93e1c7e80ae768d77bdd6ce3b6c7a024d1bbed9f18c5172f02a410b84b` |
| E23 | DERIVED | P3 | host 与 device 归因（**每后端仅 1 次**插桩运行）；~14 µs/调用的 Hybrid 额外 host 时间由此推导 | [`p3/evidence/attribution/attribution_summary.txt`](../evidence/p3/evidence/attribution/attribution_summary.txt) | `hybrid_graph_bs8_calls` | `d7c2d1e3491051ffd0c49a01c2268a004a638ca76af665ee2b19fde90e1ef05d` |
| E24 | VERIFIED | P0.5 | 依赖锁（**脱敏副本**：本地 `file://` 路径已换成官方 release URL + sha256） | [`requirements.lock.sanitized.txt`](../evidence/requirements.lock.sanitized.txt) | `torch==2.9.1+cu130` | `722158c16b24e545a08a1c8058ccd897ab414b64872dabb64efe7dd8f96bffc6` |

## 只保存在作者本地、不进入本仓库的数据

| 内容 | 本地位置（`<NV5_WORKSPACE>` 为作者的 NV5 工作空间） | 大小 | 校验 |
|---|---|---|---|
| M-1' 冻结的真实激活（66 个文件，只读） | `<NV5_WORKSPACE>/.cache/p1a_r1/capture/` | **831 MB** | 逐文件清单 `p1a_r1/evidence/m1/capture_checksums.txt`，其 sha256 为 `eb401a8def41c0a5f07807bc4678d52e9fa163d12c4db92cd99442015e2fa722`；`m1_frozen_reference_data.json` 的 sha256 为 `bdb4a88836be9d33788940fd9192634c1fa5cfcfb83430c609abf5533367e063` |
| 模型权重 | `<NV5_WORKSPACE>/models/Qwen3-0.6B` | 1.5 GB | `model.safetensors` sha256 `f47f71177f32bcd101b7573ec9171e6a57f4f4d31148d38e382306f42996874b` |
| venv | `<NV5_WORKSPACE>/.venv` | 5.9 GB | 锁文件 E24 |
| 编译缓存、运行张量、原始 profiler 二进制（`.nsys-rep` / `.ncu-repz` / sqlite） | `<NV5_WORKSPACE>/.cache/` | 约 19 GB | 解析后的文本已放入 `evidence/` |
| P3 锁定计划 / Amendment 1 | `<NV5_WORKSPACE>/p3/P3_PLAN_LOCKED.md`、`AMENDMENT_1_G8_V5_DEFINITION.md` | 小 | `f31fd9305354f2488dea3ac48519959d87c4773abca0a5fa9bee27444bc7e275` / `459a1cfd06fbbe82a50579a288197ecbe1514aa6aef0d7421bef17224badf09d` |
| P2 的 M-2' 修订案 | `<NV5_WORKSPACE>/p2/M2_AMENDMENT.md` | 小 | `0d0f5819d3b1396afcc9c869703e39c6262f490ec05cb41e6e05d08b33326db0` |

**这些路径是作者本机的绝对路径**，读者无法访问；本仓库不承诺它们可以被下载。

## 访问限制

- 旧完整集成仓 `CiaranCw/nv5-nanovllm-kernel-integration` 与 CUDA Kernel Lab `CiaranCw/cuda-kernel-lab`
  均为**私有仓库**，对无权限读者不可用。
- 本仓库的 `evidence/**` 是上述结论的**派生文本证据**，足以独立核对数字，但不足以重放实验。
- 想要等效重建可执行代码，请使用 [`../integration/bootstrap.sh`](../integration/bootstrap.sh) 从**公开上游**重建。
