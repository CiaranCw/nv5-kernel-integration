"""Workload-aware hybrid for the fused residual-add + RMSNorm path.

Runs the CUDA N-C kernel (cuda_backend) only inside the envelope verified in NV5-P2: prefill, bf16, H=1024,
1024 <= N <= 4096 rows, 2-D contiguous, 16-byte aligned. Every other call, in particular every decode step and
anything executed while a CUDA graph is being captured, runs the upstream compiled add_rms_forward.

The phase signal is nano's own step context (nanovllm.utils.context), the same one Attention reads:
ModelRunner.prepare_prefill sets is_prefill=True, prepare_decode and capture_cudagraph set it False, and run()
resets it after each step. Graph replay does not execute Python, so decode graphs contain whatever was chosen
at capture time, which is always the upstream path.
"""
import torch

from nanovllm.layers.nv5_rmsnorm import register_backend, register_policy
from nanovllm.layers.nv5_rmsnorm.cuda_backend import fused_add_rms_norm
from nanovllm.utils.context import get_context

CUDA_DTYPE = torch.bfloat16
CUDA_HIDDEN = 1024
CUDA_MIN_ROWS = 1024
CUDA_MAX_ROWS = 4096


def use_cuda(x: torch.Tensor, residual: torch.Tensor, weight: torch.Tensor) -> bool:
    if not get_context().is_prefill or torch.cuda.is_current_stream_capturing():
        return False
    if x.dim() != 2 or x.dtype != CUDA_DTYPE or x.shape[1] != CUDA_HIDDEN or not CUDA_MIN_ROWS <= x.shape[0] <= CUDA_MAX_ROWS:
        return False
    if residual.dtype != CUDA_DTYPE or weight.dtype != CUDA_DTYPE or residual.shape != x.shape or weight.shape != (CUDA_HIDDEN,):
        return False
    if not (x.is_cuda and residual.device == x.device and weight.device == x.device):
        return False
    if not (x.is_contiguous() and residual.is_contiguous() and weight.is_contiguous()):
        return False
    return x.data_ptr() % 16 == 0 and residual.data_ptr() % 16 == 0 and weight.data_ptr() % 16 == 0


register_backend("hybrid", fused_add_rms_norm)
register_policy("hybrid", use_cuda)
