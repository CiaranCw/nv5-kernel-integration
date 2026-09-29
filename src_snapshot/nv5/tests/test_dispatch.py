#!/usr/bin/env python
"""NV5 dispatch framework and reference-backend unit tests.

Run: source env/activate.sh && python nv5/tests/test_dispatch.py   (from the nano-vllm root)
"""
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch  # noqa: E402

from nanovllm.layers import nv5_rmsnorm  # noqa: E402
from nanovllm.layers.layernorm import RMSNorm  # noqa: E402
from nanovllm.layers.nv5_rmsnorm import NV5BackendError, NV5BackendNotImplementedError  # noqa: E402
from nanovllm.layers.nv5_rmsnorm import testing as nv5_testing  # noqa: E402
from nanovllm.layers.nv5_rmsnorm.reference import reference_nc, reference_wrong_c6r  # noqa: E402

RESULTS = []


def check(name, cond, detail=""):
    RESULTS.append((name, bool(cond)))
    print(f"{'PASS' if cond else 'FAIL'}  {name}  {detail}")


def with_env(**kw):
    class Ctx:
        def __enter__(self):
            self.old = {k: os.environ.get(k) for k in kw}
            for k, v in kw.items():
                if v is None:
                    os.environ.pop(k, None)
                else:
                    os.environ[k] = v

        def __exit__(self, *a):
            for k, v in self.old.items():
                if v is None:
                    os.environ.pop(k, None)
                else:
                    os.environ[k] = v
    return Ctx()


def raises(exc, fn):
    try:
        fn()
    except exc as e:
        return True, repr(e)
    except Exception as e:  # noqa: BLE001
        return False, f"wrong exception {e!r}"
    return False, "no exception"


def ordered(t):
    i = t.contiguous().view(torch.int16).to(torch.int32) & 0xFFFF
    return torch.where(i >= 0x8000, -(i & 0x7FFF), i)


def test_selection():
    for val in (None, "", "baseline"):
        with with_env(NV5_RMSNORM_BACKEND=val):
            m = RMSNorm(64)
            check(f"env={val!r} -> baseline", m._nv5_backend == "baseline" and m._nv5_impl is None)
    for val in ("Baseline", "foo", "reference", " baseline"):
        with with_env(NV5_RMSNORM_BACKEND=val):
            ok, d = raises(NV5BackendError, lambda: RMSNorm(64))
            check(f"env={val!r} unknown -> NV5BackendError", ok and "unknown" in d, d)
    for val in nv5_rmsnorm.TEST_ONLY_BACKENDS:
        with with_env(NV5_RMSNORM_BACKEND=val):
            ok, d = raises(NV5BackendError, lambda: RMSNorm(64))
            check(f"env={val!r} test-only -> NV5BackendError", ok and "test-only" in d, d)
    for val in ("triton", "cuda"):
        for fb in (None, "1"):
            with with_env(NV5_RMSNORM_BACKEND=val, NV5_RMSNORM_ALLOW_FALLBACK=fb):
                m = RMSNorm(64)
                check(f"env={val!r} ALLOW_FALLBACK={fb!r} -> registered candidate bound",
                      m._nv5_backend == val and m._nv5_impl is not None and m._nv5_impl is nv5_rmsnorm._REGISTRY[val])
    saved = dict(nv5_rmsnorm._REGISTRY), dict(nv5_rmsnorm._MODULES)
    try:
        nv5_rmsnorm._REGISTRY["triton"] = None
        nv5_rmsnorm._MODULES["triton"] = "nanovllm.layers.nv5_rmsnorm._common"  # imports but never registers
        with with_env(NV5_RMSNORM_BACKEND="triton", NV5_RMSNORM_ALLOW_FALLBACK="1"):
            ok, d = raises(NV5BackendNotImplementedError, lambda: RMSNorm(64))
            check("reserved backend whose module does not register -> NotImplemented (no fallback)", ok, d)
    finally:
        nv5_rmsnorm._REGISTRY.update(saved[0])
        nv5_rmsnorm._MODULES.update(saved[1])
    ok, d = raises(NV5BackendError, lambda: nv5_rmsnorm.register_backend("reference_nc", reference_nc))
    check("register_backend rejects non-reserved names", ok, d)


def test_test_only_binding():
    model = torch.nn.Sequential(RMSNorm(64), RMSNorm(64))
    with with_env(NV5_RMSNORM_ENABLE_TEST_BACKENDS=None):
        ok, d = raises(NV5BackendError, lambda: nv5_testing.bind_test_backend(model, "reference_nc"))
        check("bind_test_backend requires explicit enable switch", ok, d)
    with with_env(NV5_RMSNORM_ENABLE_TEST_BACKENDS="1"):
        n = nv5_testing.bind_test_backend(model, "reference_nc")
        check("bind reference_nc", n == 2 and all(m._nv5_impl is reference_nc for m in model))
        ok, d = raises(NV5BackendError, lambda: nv5_testing.bind_test_backend(model, "triton"))
        check("bind rejects non-test names", ok, d)
        nv5_testing.bind_test_backend(model, "baseline")
        check("bind baseline restores upstream path", all(m._nv5_impl is None for m in model))


def test_reference_semantics():
    g = torch.Generator(device="cuda").manual_seed(1)
    for dt in (torch.bfloat16, torch.float16):
        x = torch.randn(64, 1024, device="cuda", generator=g).to(dt)
        r = (4 * torch.randn(64, 1024, device="cuda", generator=g)).to(dt)
        w = (1 + 0.25 * torch.randn(1024, device="cuda", generator=g)).to(dt)
        x0, r0, w0 = x.clone(), r.clone(), w.clone()
        m = RMSNorm(1024).to(device="cuda", dtype=dt)
        with torch.no_grad():
            m.weight.copy_(w)
        with torch.inference_mode():
            yb, rob = m(x, r)
        for name, fn in (("reference_nc", reference_nc), ("reference_wrong_c6r", reference_wrong_c6r)):
            y, ro = fn(x, r, w, 1e-6)
            ptrs = {x.data_ptr(), r.data_ptr(), w.data_ptr()}
            check(f"{name} {dt} memory semantics",
                  torch.equal(x, x0) and torch.equal(r, r0) and torch.equal(w, w0)
                  and y.data_ptr() not in ptrs and ro.data_ptr() not in ptrs and y.data_ptr() != ro.data_ptr()
                  and y.is_contiguous() and ro.is_contiguous() and y.dtype == dt and ro.dtype == dt)
            check(f"{name} {dt} residual_out == round_T(fp32 sum) bitwise",
                  torch.equal(ro, (x.float() + r.float()).to(dt)))
            d = (ordered(y) - ordered(yb)).abs()
            frac, mu = (d > 0).float().mean().item(), int(d.max())
            if name == "reference_nc":
                check(f"{name} {dt} vs baseline within OP-Y noise (ulp<=1, frac<=1e-3)", mu <= 1 and frac <= 1e-3,
                      f"frac={frac:.2e} max_ulp={mu}")
            else:
                check(f"{name} {dt} differs from baseline (negative control, frac>0.1)", frac > 0.1,
                      f"frac={frac:.2e} max_ulp={mu}")
        ok, d = raises(TypeError, lambda: reference_nc(x.float(), r.float(), w.float(), 1e-6))
        check(f"reference rejects fp32 ({dt} case)", ok, d)


def main():
    test_selection()
    test_test_only_binding()
    test_reference_semantics()
    n_fail = sum(not ok for _, ok in RESULTS)
    print(f"TOTAL {len(RESULTS)} checks, {len(RESULTS) - n_fail} PASS, {n_fail} FAIL")
    return 0 if n_fail == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
