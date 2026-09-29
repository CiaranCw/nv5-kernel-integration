"""Triton candidate for the fused residual-add + RMSNorm path, N-C contract.

    s   = fp32(x) + fp32(residual)
    residual_out = round_T(s)
    inv = rsqrt(sum(s * s) / H + eps)        statistic and normalization use the unrounded s
    y   = round_T((s * inv) * fp32(weight))  no rounding to T before the weight multiply

One program per row; the whole row is held in registers (BLOCK = next_pow2(H), masked tail).
"""
import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice

from nanovllm.layers.nv5_rmsnorm import register_backend
from nanovllm.layers.nv5_rmsnorm._common import check_inputs


@triton.jit
def _nv5_fused_add_rmsnorm_nc(x_ptr, r_ptr, w_ptr, y_ptr, ro_ptr, eps, H: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0).to(tl.int64)
    offs = tl.arange(0, BLOCK)
    mask = offs < H
    base = row * H
    x = tl.load(x_ptr + base + offs, mask=mask, other=0.0).to(tl.float32)
    r = tl.load(r_ptr + base + offs, mask=mask, other=0.0).to(tl.float32)
    s = x + r
    tl.store(ro_ptr + base + offs, s.to(ro_ptr.dtype.element_ty), mask=mask)
    inv = libdevice.rsqrt(tl.sum(s * s, axis=0) / H + eps)
    w = tl.load(w_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    y = (s * inv) * w
    tl.store(y_ptr + base + offs, y.to(y_ptr.dtype.element_ty), mask=mask)


def _num_warps(block: int) -> int:
    return max(1, min(16, block // 256))


def fused_add_rms_norm(x: torch.Tensor, residual: torch.Tensor, weight: torch.Tensor, eps: float):
    check_inputs(x, residual, weight)
    n, h = x.shape
    y = torch.empty_like(x)
    residual_out = torch.empty_like(x)
    if n == 0:
        return y, residual_out
    block = triton.next_power_of_2(h)
    _nv5_fused_add_rmsnorm_nc[(n,)](x, residual, weight, y, residual_out, eps, H=h, BLOCK=block,
                                    num_warps=_num_warps(block))
    return y, residual_out


register_backend("triton", fused_add_rms_norm)
