"""Test-only entry point for binding reference backends onto an already-built model.

Never used by the inference path. Requires NV5_RMSNORM_ENABLE_TEST_BACKENDS=1 as an explicit switch.
"""
import os

from torch import nn

from nanovllm.layers.layernorm import RMSNorm
from nanovllm.layers.nv5_rmsnorm import BASELINE, NV5BackendError
from nanovllm.layers.nv5_rmsnorm.reference import reference_nc, reference_wrong_c6r

ENV_ENABLE = "NV5_RMSNORM_ENABLE_TEST_BACKENDS"
TEST_BACKENDS = {"reference_nc": reference_nc, "reference_wrong_c6r": reference_wrong_c6r}


def bind_test_backend(model: nn.Module, name: str) -> int:
    """Rebind the fused-path impl of every RMSNorm in `model`; returns the number of modules bound."""
    if os.environ.get(ENV_ENABLE) != "1":
        raise NV5BackendError(f"test backends are disabled; set {ENV_ENABLE}=1 in a test harness")
    if name == BASELINE:
        impl = None
    elif name in TEST_BACKENDS:
        impl = TEST_BACKENDS[name]
    else:
        raise NV5BackendError(f"unknown test backend {name!r}; expected one of {[BASELINE, *TEST_BACKENDS]}")
    n = 0
    for m in model.modules():
        if isinstance(m, RMSNorm):
            m._nv5_backend, m._nv5_impl, m._nv5_route = name, impl, None
            n += 1
    return n
