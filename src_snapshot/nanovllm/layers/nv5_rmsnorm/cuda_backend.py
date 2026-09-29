"""CUDA candidate for the fused residual-add + RMSNorm path, N-C contract (see csrc/nv5_fused_add_rmsnorm.cu).

The extension is JIT-built with torch.utils.cpp_extension at import time, i.e. when an RMSNorm is
constructed with NV5_RMSNORM_BACKEND=cuda, which is before model warm-up and CUDA-graph capture.
Build output goes to TORCH_EXTENSIONS_DIR; CUDA_HOME must point to a CUDA 13 toolkit matching torch.
"""
from pathlib import Path

import torch
from torch.utils.cpp_extension import load

from nanovllm.layers.nv5_rmsnorm import register_backend
from nanovllm.layers.nv5_rmsnorm._common import check_inputs

_SRC = Path(__file__).resolve().parent / "csrc" / "nv5_fused_add_rmsnorm.cu"
_ext = load(name="nv5_fused_add_rmsnorm_nc", sources=[str(_SRC)],
            extra_cflags=["-O3", "-std=c++17"],
            extra_cuda_cflags=["-O3", "-std=c++17", "--expt-relaxed-constexpr"], verbose=False)


def fused_add_rms_norm(x: torch.Tensor, residual: torch.Tensor, weight: torch.Tensor, eps: float):
    check_inputs(x, residual, weight)
    y, residual_out = _ext.fused_add_rms_norm_nc(x, residual, weight, float(eps))
    return y, residual_out


register_backend("cuda", fused_add_rms_norm)
