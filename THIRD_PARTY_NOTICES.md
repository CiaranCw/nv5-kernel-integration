# Third-party notices / 第三方声明

## 1. Upstream project: nano-vLLM

- Project: <https://github.com/GeeeekExplorer/nano-vllm>
- Pinned commit used by NV5: `bb823b3e06983d71485a8e1f23715ebd87d98ef8`
- License: **MIT**, `Copyright (c) 2025 Xingkai Yu`
- The complete, unmodified license text is kept verbatim in [`LICENSE`](LICENSE) and
  [`LICENSES/nano-vllm-MIT.txt`](LICENSES/nano-vllm-MIT.txt).

### What NV5 takes from upstream

NV5 does **not** redistribute the nano-vLLM repository. This showcase repository contains only:

| Item | Origin |
|---|---|
| `src_snapshot/nanovllm/layers/layernorm.py` | **modified copy of the upstream file** (upstream blob `71bf4198…` → NV5 blob `09a75c93…`). The upstream `rms_forward` / `add_rms_forward` bodies and their `@torch.compile` decorators are untouched; NV5 adds an import, two constructor bindings and one dispatch branch in `forward` (+7 / −1 lines). |
| `integration/patches/nv5-functional.patch` | a `git diff --binary --full-index` between upstream `bb823b3e…` and NV5 `db8dc57d…`. It contains upstream context lines and therefore upstream-copyrighted material. |
| `LICENSE`, `LICENSES/nano-vllm-MIT.txt` | verbatim upstream MIT text. |
| `evidence/requirements.lock.sanitized.txt` | a sanitized copy of the local dependency lock (local `file://` paths replaced by the official release URL + sha256). Third-party packages keep their own licenses; nothing here grants or alters them. |

Everything else in this repository — `src_snapshot/nanovllm/layers/nv5_rmsnorm/**`,
`src_snapshot/nv5/tests/**`, `docs/**`, `integration/bootstrap.sh`,
`integration/verify_snapshot.py`, `integration/SOURCE_MANIFEST.tsv`, `scripts/**` — is authored by the
NV5 author.

## 2. NV5-authored files: no additional license is granted here

The NV5-authored files (all 12 new files listed as `nv5_original` in
[`integration/SOURCE_MANIFEST.tsv`](integration/SOURCE_MANIFEST.tsv), plus the documentation and scripts
in this repository) are **not** covered by the upstream MIT grant. This repository intentionally does
**not** invent a new license for them: they remain under the exclusive copyright of the NV5 author, and
no redistribution rights are granted by this document. Contact the repository owner if you need terms.

The upstream MIT license continues to apply to the upstream material it covers — in particular to
`src_snapshot/nanovllm/layers/layernorm.py` and to the upstream context inside the patch — and that
grant is preserved in full by the verbatim `LICENSE` file.

## 3. Relationship to the CUDA Kernel Lab (C1–C6)

The CUDA kernel in `src_snapshot/nanovllm/layers/nv5_rmsnorm/csrc/nv5_fused_add_rmsnorm.cu` reuses a
*design idea* from the author's earlier, separate CUDA Kernel Lab work (one block per row, warp-shuffle
block reduction, 16-byte vector access). **No C6 source file, build artifact or binary is copied**, and
the numerical contract is different (see [`docs/NUMERICAL_CONTRACT_AND_TESTS.md`](docs/NUMERICAL_CONTRACT_AND_TESTS.md):
NV5 implements N-C, whereas C6 used the rounded-residual C6-R contract).

The CUDA Kernel Lab is a **separate private project**. No bidirectional link has been added to it by this
task; a draft proposal lives in `CUDA_LAB_RECIPROCAL_LINK_DRAFT.md`, which is **not** part of this
repository and is only delivered inside the review bundle for the owner's future approval.

## 4. Models and data

`Qwen/Qwen3-0.6B` (Hugging Face, `Qwen/Qwen3-0.6B` @ `c1899de289a04d12100db370d81485cdf75e47ca`) was used
for the historical experiments. **No model weights, frozen activations, or measurement binaries are
included in this repository**; see [`docs/REPRODUCIBILITY.md`](docs/REPRODUCIBILITY.md).

## 5. Trademarks

"NVIDIA", "RTX", "Nsight Compute", "Nsight Systems" and "CUDA" are trademarks of NVIDIA Corporation.
"PyTorch" and "Triton" belong to their respective owners. Mention here is descriptive only.
