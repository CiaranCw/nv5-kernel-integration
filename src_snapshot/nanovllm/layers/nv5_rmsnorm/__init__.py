"""Backend selection for the fused residual-add + RMSNorm path (RMSNorm.forward with a residual).

Only the residual branch is dispatched; RMS-only calls (layer-0 input_layernorm, q_norm/k_norm)
always run the upstream implementation. Selection happens once, at RMSNorm construction; a policy backend
additionally chooses per call between its impl and the upstream path.

NV5_RMSNORM_BACKEND:
    unset / "" / "baseline"  upstream compiled add_rms_forward (default; no candidate module is imported)
    "triton" / "cuda"        N-C candidate; its module is imported on first selection and registers itself
    "hybrid"                 per-call route policy: the CUDA N-C kernel for the verified prefill envelope,
                             upstream add_rms_forward otherwise (see hybrid_backend.py)
    anything else            raise NV5BackendError (no silent fallback)

A route policy returning False is an intentional choice of the upstream path, not an error fallback;
errors raised by the selected impl propagate.

Test-only reference backends are not selectable here; see nanovllm.layers.nv5_rmsnorm.testing.
"""
import importlib
import os
from typing import Callable

import torch

ENV_BACKEND = "NV5_RMSNORM_BACKEND"
BASELINE = "baseline"
TEST_ONLY_BACKENDS = ("reference_nc", "reference_wrong_c6r")

# fn(x, residual, weight, eps) -> (y, residual_out); outputs are new tensors, inputs are not modified.
FusedAddRMSNorm = Callable[[torch.Tensor, torch.Tensor, torch.Tensor, float], tuple[torch.Tensor, torch.Tensor]]

# fn(x, residual, weight) -> True to run the backend impl, False to run upstream add_rms_forward.
RoutePolicy = Callable[[torch.Tensor, torch.Tensor, torch.Tensor], bool]

_REGISTRY: dict[str, FusedAddRMSNorm | None] = {"triton": None, "cuda": None, "hybrid": None}
_MODULES = {"triton": "nanovllm.layers.nv5_rmsnorm.triton_backend", "cuda": "nanovllm.layers.nv5_rmsnorm.cuda_backend",
            "hybrid": "nanovllm.layers.nv5_rmsnorm.hybrid_backend"}
POLICY_BACKENDS = ("hybrid",)
_POLICIES: dict[str, RoutePolicy] = {}


class NV5BackendError(RuntimeError):
    pass


class NV5BackendNotImplementedError(NV5BackendError, NotImplementedError):
    pass


def register_backend(name: str, fn: FusedAddRMSNorm) -> None:
    if name not in _REGISTRY:
        raise NV5BackendError(f"cannot register {name!r}; reserved backend names are {sorted(_REGISTRY)}")
    _REGISTRY[name] = fn


def register_policy(name: str, fn: RoutePolicy) -> None:
    if name not in POLICY_BACKENDS:
        raise NV5BackendError(f"cannot register a route policy for {name!r}; policy backends are {list(POLICY_BACKENDS)}")
    _POLICIES[name] = fn


def route_policy(name: str) -> RoutePolicy | None:
    """Return the per-call route policy of a policy backend, None for every other backend."""
    if name not in POLICY_BACKENDS:
        return None
    if name not in _POLICIES:
        raise NV5BackendNotImplementedError(f"{ENV_BACKEND}={name!r} has no registered route policy")
    return _POLICIES[name]


def selected_backend() -> str:
    name = os.environ.get(ENV_BACKEND, "")
    if name in ("", BASELINE):
        return BASELINE
    if name in _REGISTRY:
        return name
    if name in TEST_ONLY_BACKENDS:
        raise NV5BackendError(
            f"{ENV_BACKEND}={name!r} is a test-only backend and cannot be selected through the environment; "
            f"use nanovllm.layers.nv5_rmsnorm.testing.bind_test_backend")
    raise NV5BackendError(f"unknown {ENV_BACKEND}={name!r}; expected one of {[BASELINE, *_REGISTRY]}")


def resolve() -> tuple[str, FusedAddRMSNorm | None]:
    """Return (backend_name, impl). impl is None for the upstream baseline path."""
    name = selected_backend()
    if name == BASELINE:
        return name, None
    if _REGISTRY[name] is None:
        importlib.import_module(_MODULES[name])
    fn = _REGISTRY[name]
    if fn is None:
        raise NV5BackendNotImplementedError(
            f"{ENV_BACKEND}={name!r} is reserved but not implemented; no fallback to {BASELINE!r} is performed")
    return name, fn
