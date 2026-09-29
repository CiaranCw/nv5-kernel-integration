# Provenance / 溯源与本轮独立验证

本文回答三个问题：这份代码从哪来、本仓库与"旧完整集成仓"是什么关系、以及**本轮（2026-09-29）实际独立验证了什么**。

## 1. 固定身份

| 项目 | 值 |
|---|---|
| 上游 nano-vLLM | `https://github.com/GeeeekExplorer/nano-vllm.git` @ `bb823b3e06983d71485a8e1f23715ebd87d98ef8`（tree `5ec2b4b8034b26f5cc9af73c13d520a2aa8774f0`） |
| NV5 功能权威版本 | `db8dc57d9d16a827c8d35cbc63e33dad8d36b671`（tree `05480eb86b23065ac412f1325c587bb129abd2da`） |
| NV5 功能提交链 | `bb823b3e` → `1adb5682`（P1A 分发框架）→ `f040b482`（P1B Triton/CUDA）→ `db8dc57d`（P3 Hybrid） |
| 旧完整仓文档提交 | `a67ee1e2c6c907d33be0e39833ebba04c517998b`（**docs-only**，不含任何源码改动） |
| 本仓库 Git 历史 | 从自己的根提交开始，**不导入**上游或旧功能提交 |
| 功能补丁 | `integration/patches/nv5-functional.patch` = `git diff --binary --full-index bb823b3e..db8dc57d`，sha256 `96a87f5088eadff9b2b66248fb48f674ad40812510658822a21cdf075e26ce86` |

## 2. 与旧完整集成仓的关系

完整的 nano-vLLM 集成工程仍保存在作者的**另一个私有仓库**
`https://github.com/CiaranCw/nv5-nanovllm-kernel-integration.git`（`main = a67ee1e2c6c907d33be0e39833ebba04c517998b`），
本地工作空间记为 `<NV5_WORKSPACE>/nano-vllm`（作者本机的绝对路径不在本仓库中披露）。它对无权限读者不可用；
本仓库**不承诺**可以直接访问它。

两者的分工：

| 内容 | 旧完整仓 | 本仓库 |
|---|---|---|
| nano-vLLM 完整源码 + 上游 Git 历史 | ✅ | ❌（不含） |
| NV5 功能源码（`nv5_rmsnorm/` 与 `nv5/tests/`） | ✅（原位） | ✅（**逐字节镜像** + 精确补丁） |
| 被修改的 `layernorm.py` | ✅ | ✅（镜像 + 补丁） |
| 冻结文档与证据 | ✅ | ✅（精选小体积证据，逐字节复制） |
| 831 MB 冻结张量 / 模型 / venv / 原始 profiler 产物 | ✅（本地） | ❌（不进仓库） |

**唯一权威的可执行集成方式**是"固定上游 commit + 精确补丁"。`src_snapshot/` 是供代码审阅的冻结镜像，
不是能在独立根目录直接 `import nanovllm` 的完整 Python 引擎包。

## 3. 代码归属（不可混淆的三类）

| 类别 | 内容 |
|---|---|
| **original upstream** | 调度、`ModelRunner`、Qwen3 模型、Attention/FlashAttention 调用、KV Cache、CUDA Graph 捕获与 replay、采样、`nanovllm/utils/context.py`、以及 `RMSNorm.rms_forward` / `add_rms_forward` 的函数体与 `@torch.compile`。**NV5 一行未改** |
| **modified upstream** | `nanovllm/layers/layernorm.py`（+7 / −1：import、2 行构造绑定、1 个分发分支） |
| **NV5 原创** | `nv5_rmsnorm/` 下 8 个文件与 `nv5/tests/` 下 4 个文件 |

`.cu` 文件的归属需要说明清楚：它**重写**自作者此前的 CUDA Kernel Lab C6 的设计思想（每行一个 block、
warp shuffle 块归约、16 字节向量访问），**没有复制 C6 的任何代码或二进制**；契约也改成了 N-C。
CUDA Kernel Lab 是**独立的私有项目**，本任务**没有**给它添加任何双向链接（草稿见审阅包中的
`CUDA_LAB_RECIPROCAL_LINK_DRAFT.md`，不随仓库发布）。

## 4. 本轮（2026-09-29）实际独立验证的项目

下列每一项都是本轮用真实 Git 对象 / 文件哈希**重新执行**的结果，不是引用旧结论：

| # | 验证 | 命令 | 结果 |
|---|---|---|---|
| 1 | 提交链与父子关系 | `git cat-file commit` 逐个核对 parent | `bb823b3e → 1adb5682 → f040b482 → db8dc57d → a67ee1e` ✅ |
| 2 | docs-only 提交不含源码 | `git diff --stat db8dc57d a67ee1e` | 仅 `README.md`、`.gitignore`、`docs/nv5/**` ✅ |
| 3 | 功能改动范围 | `git diff --name-status bb823b3e db8dc57d` | 13 个路径：1 个 `M` + 12 个 `A` ✅ |
| 4 | 上游基线可达且等于其 `main` HEAD | `git ls-remote` | `bb823b3e…` ✅ |
| 5 | **补丁精确重建 P3 tree** | `integration/bootstrap.sh` → `git write-tree` | `05480eb86b23065ac412f1325c587bb129abd2da` ✅ |
| 6 | 镜像文件逐字节等于 P3 blob | `integration/verify_snapshot.py --repo …` | 13/13 sha256 一致 ✅ |
| 7 | 冻结证据 SHA256 | 重算 `evidence/**` | 24/24 与 `EVIDENCE_MAP` 记录一致 ✅ |
| 8 | 历史审阅包 SHA256 | `sha256sum` | 学习包 / 最终审阅包均与预期一致 ✅ |
| 9 | 文件模式与行尾 | `git ls-files -s`、CRLF 统计 | `layernorm.py` 100755 保留；无 CRLF ✅ |
| 10 | 负向测试 | 已有非空目录 / 篡改补丁 / 路径注入 / 上游不可达 | 分别 exit 2 / 3 / 3 / 4，且未删除任何既有文件 ✅ |
| 11 | 远程二次验证 | 从新远程 clone 后再跑 bootstrap + verify | 见 `SOURCE_PATCH_REBUILD_AUDIT.md` |
| 12 | 旧仓未变更 | `git status`、`git rev-parse HEAD`、远程 `ls-remote` | 见 `PRE_POST_STATE.md` |

## 5. 本轮**没有**重跑的项目（引用既有冻结证据）

本轮**没有**运行任何 GPU 实验、没有重新编译 CUDA 扩展、没有重跑 P3 的 32 进程矩阵。
下面这些结论全部**引用**旧冻结证据（`evidence/**`，sha256 见 [`EVIDENCE_MAP.md`](EVIDENCE_MAP.md)），
在本仓库中标注为历史结论，**不是**本轮的新测量：

- N-C 契约的确定、8-oracle witness、442 个算子门禁；
- M-1' 的 3584 次冻结调用重放，以及 C6-R 负对照；
- M-0 / M-CAL / M-2 / M-2' / M-3、36 个 CUDA Graph 的节点枚举；
- P2 的 nsys / ncu / SASS / E2E / Amdahl / host 归因；
- P3 的 G1–G8 门禁与 8 轮预注册性能矩阵。

历史运行环境与本轮不同（历史为单块 RTX 5080 + WSL2 + torch 2.9.1+cu130），且本轮未执行 GPU 流程，
因此这些数字**只能**按历史口径引用，不能声称被本轮复现。

## 6. 为什么可以相信"镜像没有被动过手脚"

1. 镜像来自 `git show db8dc57d:<path>`，即 Git 对象本身；
2. `SOURCE_MANIFEST.tsv` 记录每个文件的 `p3_sha256` 与该路径在 P3 tree 中的 **blob SHA**；
3. 补丁重建出的 tree 与权威 `db8dc57d^{tree}` **相等**——这是一个覆盖全树的整体哈希，
   任何单字节改动（含行尾、权限、遗漏文件）都会让它不等；
4. `verify_snapshot.py` 再逐项比对镜像 ↔ 重建树 ↔ manifest 三者。
