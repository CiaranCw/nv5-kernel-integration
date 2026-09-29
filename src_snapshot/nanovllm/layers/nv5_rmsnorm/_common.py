"""Input contract shared by the NV5 fused residual-add + RMSNorm candidate backends."""
import torch

from nanovllm.layers.nv5_rmsnorm import NV5BackendError

SUPPORTED_DTYPES = (torch.bfloat16, torch.float16)


def check_inputs(x: torch.Tensor, residual: torch.Tensor, weight: torch.Tensor) -> None:
    """Raise NV5BackendError for anything outside the supported fused path (no silent fallback)."""
    if not (x.is_cuda and residual.is_cuda and weight.is_cuda):
        raise NV5BackendError("x, residual and weight must be CUDA tensors")
    if x.dtype not in SUPPORTED_DTYPES:
        raise NV5BackendError(f"unsupported dtype {x.dtype}; only bf16/fp16 are supported (fp32 is rejected)")
    if residual.dtype != x.dtype or weight.dtype != x.dtype:
        raise NV5BackendError(f"dtype mismatch: x={x.dtype} residual={residual.dtype} weight={weight.dtype}")
    if x.dim() != 2 or residual.shape != x.shape or weight.dim() != 1 or weight.shape[0] != x.shape[1]:
        raise NV5BackendError(f"shape mismatch: expected x=[N,H], residual=[N,H], weight=[H]; got "
                              f"{tuple(x.shape)}, {tuple(residual.shape)}, {tuple(weight.shape)}")
    if not (x.is_contiguous() and residual.is_contiguous() and weight.is_contiguous()):
        raise NV5BackendError("x, residual and weight must be contiguous")
    if residual.device != x.device or weight.device != x.device:
        raise NV5BackendError("x, residual and weight must be on the same device")
