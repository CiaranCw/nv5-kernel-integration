# Reproducibility / 复现：能做什么、不能做什么

## 1. 固定版本

| 项目 | 值 |
|---|---|
| 上游 nano-vLLM | `bb823b3e06983d71485a8e1f23715ebd87d98ef8`（tree `5ec2b4b8034b26f5cc9af73c13d520a2aa8774f0`） |
| NV5 功能 tree | `db8dc57d9d16a827c8d35cbc63e33dad8d36b671`（tree `05480eb86b23065ac412f1325c587bb129abd2da`） |
| 功能补丁 | `integration/patches/nv5-functional.patch`，sha256 `96a87f5088eadff9b2b66248fb48f674ad40812510658822a21cdf075e26ce86` |
| Python / 依赖 | Python 3.12.13；依赖锁见 [`evidence/requirements.lock.sanitized.txt`](../evidence/requirements.lock.sanitized.txt)（57 个包，本地 `file://` 路径已换成官方 release URL + sha256） |
| 关键版本 | torch 2.9.1+cu130、triton 3.5.1、flash-attn 2.8.3（官方 `cu13torch2.9cxx11abiTRUE-cp312` wheel）、transformers 5.13.1、nvidia-cuda-nvcc 13.0.88、ninja 1.13.2；主机编译器 gcc-13 |
| 模型 | `Qwen/Qwen3-0.6B` @ `c1899de289a04d12100db370d81485cdf75e47ca`；`model.safetensors` sha256 `f47f71177f32bcd101b7573ec9171e6a57f4f4d31148d38e382306f42996874b` |
| GPU（历史） | RTX 5080（SM120），驱动 610.62，WSL2；CUDA 扩展为 sm_120 编译 |

## 2. 在任何机器上都能做（纯 CPU，无 GPU / 无模型）

```bash
git clone https://github.com/CiaranCw/nv5-kernel-integration.git
cd nv5-kernel-integration

# 1) 在全新空目录重建“固定上游 + 补丁”
./integration/bootstrap.sh /tmp/nv5-rebuild

# 2) 静态验真：镜像 SHA256 / 补丁 SHA256 / 重建 tree
python3 integration/verify_snapshot.py --repo /tmp/nv5-rebuild

# 3) 文档与证据一致性、敏感信息扫描
python3 scripts/check_docs.py
python3 scripts/scan_secrets.py
```

`bootstrap.sh` 只依赖 git 与 sha256sum，从**官方**上游取固定 SHA（可通过 `--upstream-url` 换成离线镜像，
但脚本会打印警告）。目标目录已存在且非空时失败，绝不 `rm -rf`；上游不可达 / SHA 不符 / 补丁被拒 / tree 不一致
都非零退出。

期望输出（本轮实测）：

```
rebuilt tree  05480eb86b23065ac412f1325c587bb129abd2da  ==  NV5 P3 db8dc57d^{tree}
```

## 3. 有 GPU 时可以额外跑的：仓库内的 4 个单测

4 个测试只需要 CUDA GPU 与上述依赖，**不需要模型或私有数据**。在 `bootstrap.sh` 重建出的那棵树里
（`test_opd_baseline_transparency.py` 需要完整上游历史，所以它必须在重建树中、而不能在 `src_snapshot/` 里运行）：

```bash
cd /tmp/nv5-rebuild
cp -r <本仓库>/nv5/tests /tmp/nv5-rebuild/nv5     # 补丁已包含 nv5/tests，此步通常不需要
export PYTHONDONTWRITEBYTECODE=1
unset NV5_RMSNORM_BACKEND
for t in test_dispatch test_opd_baseline_transparency test_backends test_hybrid; do
  python nv5/tests/$t.py | tail -n 1
done
# 历史结果（RTX 5080）：33/33、148/148、74/74、54/54
```

注意：

- CUDA 后端在**导入时**用 `torch.utils.cpp_extension` 编译，需要 CUDA 13 的 nvcc 与 ninja；
- 在其它 GPU / 其它 torch 版本上运行，得到的是**新的**结果，**不能**代表 RTX 5080 上的历史结果；
- 这些历史通过记录本轮**没有**重跑（见 [`PROVENANCE.md`](PROVENANCE.md) §5）。

## 4. 从 clone **无法直接复现**的部分

| 内容 | 位置 | 大小 |
|---|---|---|
| M-1' 冻结的真实激活（66 个文件，只读） | 作者本地 `<NV5_WORKSPACE>/.cache/p1a_r1/capture/` | **831 MB** |
| 模型权重 | `<NV5_WORKSPACE>/models/Qwen3-0.6B` | 1.5 GB |
| venv | `<NV5_WORKSPACE>/.venv` | 5.9 GB |
| 编译缓存、运行张量、原始 profiler 输出（`.nsys-rep`、`.ncu-repz`、sqlite） | `<NV5_WORKSPACE>/.cache/` | 约 19 GB |
| 各阶段探针与门禁脚本（在 nano 仓库之外，写死了本地绝对路径） | `<NV5_WORKSPACE>/{p05,p1a,p1a_r1,p1b,p2,p3}/probes/` | 小 |
| P3 锁定计划 / Amendment 1 / P2 的 M-2' 修订案 | `<NV5_WORKSPACE>/p3/P3_PLAN_LOCKED.md` 等 | 小 |

因此 **M-1' 重放、真实模型的路由与 graph 节点门禁、M-0/M-2/M-2'、microbenchmark、nsys/ncu 以及 E2E 矩阵**，
都**需要**本地的模型、冻结数据和工作空间脚本；只有本仓库的 clone **无法**复现它们。
解析后的文本结论已放入 [`evidence/`](../evidence/)，但它们是对原始二进制的**派生摘要**，不是原始产物本身。

## 5. 重跑时的约定（如果未来要重做）

- 新建编译缓存，并在同一组实验内共享（V5）；**不要**复用或删除历史缓存；
- 新结果写成**新的**记录，不覆盖历史数据；
- P3 的矩阵驱动写死了 `.cache/p3/perf_compile`，该目录已存在，重跑时要复制脚本并改用新缓存路径；
- 不要让脚本依赖第三方网盘 / 临时文件中转站；本仓库的重建只依赖 git 与官方上游 URL。
