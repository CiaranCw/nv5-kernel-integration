"""Pure-torch contract references for the fused residual-add + RMSNorm path (test-only).

Both use an fp64 reduction so that they are independent of the baseline's reduction order.
T is the input dtype (bf16 / fp16); fp32 input is rejected.

reference_nc          N-C  (U,U,N): statistic and normalization use the unrounded fp32 sum,
                                    single final rounding after an fp32 weight multiply.
reference_wrong_c6r   C6-R (R,R,N): statistic and normalization use round_T(sum) (deliberately
                                    wrong for nano; used only as a calibration negative control).
"""
import torch

_DTYPES = (torch.bfloat16, torch.float16)


def _check(x: torch.Tensor, residual: torch.Tensor, weight: torch.Tensor) -> None:
    if x.dtype not in _DTYPES:
        raise TypeError(f"reference backends support bf16/fp16 only, got {x.dtype}")
    if residual.dtype != x.dtype or weight.dtype != x.dtype:
        raise TypeError(f"dtype mismatch: x={x.dtype} residual={residual.dtype} weight={weight.dtype}")
    if residual.shape != x.shape or weight.shape != x.shape[-1:]:
        raise ValueError(f"shape mismatch: x={tuple(x.shape)} residual={tuple(residual.shape)} weight={tuple(weight.shape)}")


def _normalize(src64: torch.Tensor, weight: torch.Tensor, eps: float, dtype: torch.dtype) -> torch.Tensor:
    ms = (src64 * src64).mean(dim=-1, keepdim=True)
    n32 = (src64 * torch.rsqrt(ms + eps)).float()
    return (n32 * weight.float()).to(dtype)


def reference_nc(x: torch.Tensor, residual: torch.Tensor, weight: torch.Tensor, eps: float):
    _check(x, residual, weight)
    s32 = x.float() + residual.float()
    return _normalize(s32.double(), weight, eps, x.dtype), s32.to(x.dtype)


def reference_wrong_c6r(x: torch.Tensor, residual: torch.Tensor, weight: torch.Tensor, eps: float):
    _check(x, residual, weight)
    r_t = (x.float() + residual.float()).to(x.dtype)
    return _normalize(r_t.double(), weight, eps, x.dtype), r_t
