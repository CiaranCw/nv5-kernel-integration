#!/usr/bin/env python
"""Functional tests for the N-C candidate backends (triton, cuda).

Checks per backend: residual_out bitwise equal to round_T(fp32 x + fp32 r); y within 1 ULP / 1e-3 of the
fp64-statistic N-C reference (reference_nc); memory semantics; fail-loud input validation; CUDA-graph
replay bitwise equal to eager. The full pre-registered gate matrix lives outside the repository.

Run: source env/activate.sh && python nv5/tests/test_backends.py [triton|cuda ...]   (from the nano-vllm root)
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch  # noqa: E402

from nanovllm.layers import nv5_rmsnorm  # noqa: E402
from nanovllm.layers.nv5_rmsnorm import NV5BackendError  # noqa: E402
from nanovllm.layers.nv5_rmsnorm.reference import reference_nc  # noqa: E402

RESULTS = []


def check(name, cond, detail=""):
    RESULTS.append((name, bool(cond)))
    print(f"{'PASS' if cond else 'FAIL'}  {name}  {detail}")


def ordered(t):
    i = t.contiguous().view(torch.int16).to(torch.int32) & 0xFFFF
    return torch.where(i >= 0x8000, -(i & 0x7FFF), i)


def load(name):
    if nv5_rmsnorm._REGISTRY[name] is None:
        __import__(nv5_rmsnorm._MODULES[name])
    return nv5_rmsnorm._REGISTRY[name]


def raises(fn):
    try:
        fn()
    except NV5BackendError as e:
        return True, repr(e)[:120]
    except Exception as e:  # noqa: BLE001
        return False, f"wrong exception {e!r}"[:120]
    return False, "no exception"


def test_backend(name):
    fn = load(name)
    g = torch.Generator(device="cuda").manual_seed(7)
    for dt in (torch.bfloat16, torch.float16):
        for n, h in ((1, 1024), (8, 1024), (1379, 1024), (33, 1000), (16, 4096)):
            x = torch.randn(n, h, device="cuda", generator=g).to(dt)
            r = (4 * torch.randn(n, h, device="cuda", generator=g)).to(dt)
            w = (1 + 0.25 * torch.randn(h, device="cuda", generator=g)).to(dt)
            x0, r0, w0 = x.clone(), r.clone(), w.clone()
            y, ro = fn(x, r, w, 1e-6)
            yr, _ = reference_nc(x, r, w, 1e-6)
            d = (ordered(y) - ordered(yr)).abs()
            tag = f"{name} {str(dt)[6:]} N={n} H={h}"
            check(f"{tag} residual_out bitwise", torch.equal(ro, (x.float() + r.float()).to(dt)))
            check(f"{tag} y vs N-C reference", int(d.max()) <= 1 and (d > 0).float().mean().item() <= 1e-3,
                  f"frac={(d > 0).float().mean().item():.1e} max_ulp={int(d.max())}")
            ptrs = {x.data_ptr(), r.data_ptr(), w.data_ptr()}
            check(f"{tag} memory semantics", torch.equal(x, x0) and torch.equal(r, r0) and torch.equal(w, w0)
                  and y.data_ptr() not in ptrs and ro.data_ptr() not in ptrs and y.data_ptr() != ro.data_ptr()
                  and y.is_contiguous() and ro.is_contiguous() and y.dtype == ro.dtype == dt and y.shape == x.shape)
    x = torch.randn(8, 1024, device="cuda", dtype=torch.bfloat16)
    w = torch.ones(1024, device="cuda", dtype=torch.bfloat16)
    bad = {"fp32": (x.float(), x.float(), w.float()), "dtype mismatch": (x, x.half(), w),
           "non-contiguous x": (torch.randn(1024, 8, device="cuda", dtype=torch.bfloat16).t(), x, w),
           "3-D x": (x.view(2, 4, 1024), x.view(2, 4, 1024), w), "weight size": (x, x, w[:512]),
           "cpu tensors": (x.cpu(), x.cpu(), w.cpu())}
    for what, args in bad.items():
        ok, d = raises(lambda: fn(*args, 1e-6))
        check(f"{name} fail-loud: {what}", ok, d)
    r = torch.randn_like(x)
    y_e, ro_e = fn(x, r, w, 1e-6)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        y_g, ro_g = fn(x, r, w, 1e-6)
    graph.replay()
    torch.cuda.synchronize()
    check(f"{name} CUDA graph replay bitwise equal to eager", torch.equal(y_g, y_e) and torch.equal(ro_g, ro_e))


def main():
    for name in sys.argv[1:] or ["triton", "cuda"]:
        test_backend(name)
    n_fail = sum(not ok for _, ok in RESULTS)
    print(f"TOTAL {len(RESULTS)} checks, {len(RESULTS) - n_fail} PASS, {n_fail} FAIL")
    return 0 if n_fail == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
