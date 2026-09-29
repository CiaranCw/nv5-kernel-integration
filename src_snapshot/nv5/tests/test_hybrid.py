#!/usr/bin/env python
"""Tests for the workload-aware hybrid backend (NV5_RMSNORM_BACKEND=hybrid).

Checks: selection and policy registration; the route-policy truth table (phase, rows, dtype, H, layout,
alignment, CUDA-graph capture); module-level execution path and numerics (CUDA path bitwise equal to the cuda
backend and within OP-R/OP-B of the baseline, upstream path bitwise equal to a baseline module); RMS-only calls
untouched; errors of the selected CUDA path propagate; a capture under a prefill context records no candidate node.

Run: source env/activate.sh && python nv5/tests/test_hybrid.py   (from the nano-vllm root)
"""
import collections
import ctypes
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch  # noqa: E402

from nanovllm.layers import nv5_rmsnorm  # noqa: E402
from nanovllm.layers.layernorm import RMSNorm  # noqa: E402
from nanovllm.layers.nv5_rmsnorm import NV5BackendError, NV5BackendNotImplementedError  # noqa: E402
from nanovllm.utils.context import reset_context, set_context  # noqa: E402

RESULTS = []
EPS = 1e-6


def check(name, cond, detail=""):
    RESULTS.append((name, bool(cond)))
    print(f"{'PASS' if cond else 'FAIL'}  {name}  {detail}")


def ordered(t):
    i = t.contiguous().view(torch.int16).to(torch.int32) & 0xFFFF
    return torch.where(i >= 0x8000, -(i & 0x7FFF), i)


def make(backend, h=1024, dt=torch.bfloat16, w=None):
    old = os.environ.get("NV5_RMSNORM_BACKEND")
    os.environ["NV5_RMSNORM_BACKEND"] = backend
    try:
        m = RMSNorm(h, eps=EPS).to(device="cuda", dtype=dt)
    finally:
        if old is None:
            os.environ.pop("NV5_RMSNORM_BACKEND")
        else:
            os.environ["NV5_RMSNORM_BACKEND"] = old
    if w is not None:
        with torch.no_grad():
            m.weight.copy_(w)
    return m


def inputs(n, h=1024, dt=torch.bfloat16, seed=0):
    g = torch.Generator(device="cuda").manual_seed(seed)
    x = torch.randn(n, h, device="cuda", generator=g).to(dt)
    r = (4 * torch.randn(n, h, device="cuda", generator=g)).to(dt)
    w = (1 + 0.25 * torch.randn(h, device="cuda", generator=g)).to(dt)
    return x, r, w


def prefill_ctx():
    set_context(True)


def test_selection():
    from nanovllm.layers.nv5_rmsnorm import hybrid_backend
    m = make("hybrid")
    check("hybrid selected: backend id, impl is the cuda backend function, route policy bound",
          m._nv5_backend == "hybrid" and m._nv5_impl is nv5_rmsnorm._REGISTRY["cuda"] is nv5_rmsnorm._REGISTRY["hybrid"]
          and m._nv5_route is hybrid_backend.use_cuda)
    for b in ("baseline", "triton", "cuda"):
        mb = make(b)
        check(f"{b}: no route policy", mb._nv5_route is None and nv5_rmsnorm.route_policy(b) is None)
    try:
        nv5_rmsnorm.register_policy("cuda", hybrid_backend.use_cuda)
        ok, d = False, "no exception"
    except NV5BackendError as e:
        ok, d = True, repr(e)
    check("register_policy rejects non-policy backends", ok, d)
    saved = nv5_rmsnorm._POLICIES.pop("hybrid")
    try:
        nv5_rmsnorm.route_policy("hybrid")
        ok, d = False, "no exception"
    except NV5BackendNotImplementedError as e:
        ok, d = True, repr(e)
    finally:
        nv5_rmsnorm._POLICIES["hybrid"] = saved
    check("hybrid without a registered policy fails loud", ok, d)


def test_policy_table():
    from nanovllm.layers.nv5_rmsnorm.hybrid_backend import use_cuda
    w = torch.ones(1024, device="cuda", dtype=torch.bfloat16)
    prefill_ctx()
    for n in (1024, 1379, 2048, 4096):
        x = torch.empty(n, 1024, device="cuda", dtype=torch.bfloat16)
        check(f"prefill bf16 H=1024 N={n} -> cuda", use_cuda(x, torch.empty_like(x), w))
    for n in (1, 8, 64, 512, 1023, 4097, 11032, 16384):
        x = torch.empty(n, 1024, device="cuda", dtype=torch.bfloat16)
        check(f"prefill bf16 H=1024 N={n} -> baseline", not use_cuda(x, torch.empty_like(x), w))
    x = torch.empty(1379, 1024, device="cuda", dtype=torch.bfloat16)
    cases = {
        "fp16": (x.half(), x.half(), w.half()),
        "fp32": (x.float(), x.float(), w.float()),
        "H=896": (x[:, :896].contiguous(), x[:, :896].contiguous(), w[:896].contiguous()),
        "H=2048": (torch.empty(1379, 2048, device="cuda", dtype=torch.bfloat16),) * 2 + (torch.ones(2048, device="cuda", dtype=torch.bfloat16),),
        "non-contiguous x": (torch.empty(2048, 1379, device="cuda", dtype=torch.bfloat16).t()[:, :1024], x, w),
        "misaligned x (2-byte offset)": (torch.empty(1379 * 1024 + 1, device="cuda", dtype=torch.bfloat16)[1:].view(1379, 1024), x, w),
        "residual shape mismatch": (x, x[:1378], w),
        "weight dtype fp32": (x, x, w.float()),
        "3-D x": (x.view(1379, 8, 128), x.view(1379, 8, 128), w),
        "cpu tensors": (x.cpu(), x.cpu(), w.cpu()),
    }
    for what, args in cases.items():
        check(f"prefill N=1379 {what} -> baseline", not use_cuda(*args))
    set_context(False)
    for n in (8, 512, 1379, 4096):
        xd = torch.empty(n, 1024, device="cuda", dtype=torch.bfloat16)
        check(f"decode context N={n} -> baseline", not use_cuda(xd, torch.empty_like(xd), w))
    reset_context()
    check("reset (default) context N=1379 -> baseline", not use_cuda(x, torch.empty_like(x), w))


def test_module_paths():
    calls = collections.Counter()
    x, r, w = inputs(1379, seed=1)
    xs, rs, _ = inputs(8, seed=2)
    hy, base = make("hybrid", w=w), make("baseline", w=w)
    real = hy._nv5_impl

    def counted(*a):
        calls["cuda"] += 1
        return real(*a)
    hy._nv5_impl = counted
    cuda_fn = nv5_rmsnorm._REGISTRY["cuda"]
    with torch.inference_mode():
        for ctx, xx, rr, exp in (("prefill", x, r, "cuda"), ("decode", x, r, "baseline"),
                                 ("prefill", xs, rs, "baseline"), ("decode", xs, rs, "baseline")):
            set_context(ctx == "prefill")
            x0, r0 = xx.clone(), rr.clone()
            before = calls["cuda"]
            y, ro = hy(xx, rr)
            ran = "cuda" if calls["cuda"] == before + 1 else "baseline"
            yb, rob = base(xx, rr)
            tag = f"module {ctx} N={xx.shape[0]}"
            check(f"{tag} executed path == {exp}", ran == exp, f"ran={ran}")
            check(f"{tag} inputs unchanged", torch.equal(xx, x0) and torch.equal(rr, r0))
            if exp == "cuda":
                yc, roc = cuda_fn(xx, rr, w, EPS)
                d = (ordered(y) - ordered(yb)).abs()
                check(f"{tag} bitwise equal to cuda backend", torch.equal(y, yc) and torch.equal(ro, roc))
                check(f"{tag} OP-R residual_out == round_T(fp32 sum)", torch.equal(ro, (xx.float() + rr.float()).to(xx.dtype)))
                check(f"{tag} OP-B vs baseline (<=1 ULP, <=2e-3)", int(d.max()) <= 1 and (d > 0).float().mean().item() <= 2e-3,
                      f"frac={(d > 0).float().mean().item():.1e} max_ulp={int(d.max())}")
            else:
                check(f"{tag} bitwise equal to baseline module", torch.equal(y, yb) and torch.equal(ro, rob))
        prefill_ctx()
        before = calls["cuda"]
        check("RMS-only (no residual) under prefill context: bitwise equal to baseline, impl not called",
              torch.equal(hy(x), base(x)) and calls["cuda"] == before)
        xq = torch.randn(1379, 16, 128, device="cuda", dtype=torch.bfloat16)
        hq, bq = make("hybrid", h=128), make("baseline", h=128)
        check("RMS-only 3-D q/k layout under prefill context: bitwise equal to baseline", torch.equal(hq(xq), bq(xq)))
    reset_context()


def test_errors_propagate():
    x, r, w = inputs(1379, seed=3)
    hy = make("hybrid", w=w)

    def boom(*a):
        raise NV5BackendError("selected CUDA path failed (test)")
    hy._nv5_impl = boom
    with torch.inference_mode():
        prefill_ctx()
        try:
            hy(x, r)
            ok, d = False, "no exception (silent fallback?)"
        except NV5BackendError as e:
            ok, d = True, repr(e)
        check("error in the selected CUDA path propagates (no fallback)", ok, d)
        set_context(False)
        y, _ = hy(x, r)
        check("decode context never calls the CUDA impl", y.shape == x.shape)
    reset_context()


def graph_kernel_names(graph):
    cu = ctypes.CDLL("libcuda.so.1")

    class KNP(ctypes.Structure):
        _fields_ = [("func", ctypes.c_void_p)] + [(f, ctypes.c_uint) for f in ("gx", "gy", "gz", "bx", "by", "bz", "smem")] + \
                   [("kp", ctypes.c_void_p), ("extra", ctypes.c_void_p), ("kern", ctypes.c_void_p), ("ctx", ctypes.c_void_p)]
    raw = ctypes.c_void_p(graph.raw_cuda_graph())
    n = ctypes.c_size_t(0)
    cu.cuGraphGetNodes(raw, None, ctypes.byref(n))
    arr = (ctypes.c_void_p * n.value)()
    cu.cuGraphGetNodes(raw, arr, ctypes.byref(n))
    names = []
    for nd in arr:
        t = ctypes.c_int(-1)
        cu.cuGraphNodeGetType(ctypes.c_void_p(nd), ctypes.byref(t))
        if t.value != 0:
            names.append(f"<type {t.value}>")
            continue
        p = KNP()
        cu.cuGraphKernelNodeGetParams_v2(ctypes.c_void_p(nd), ctypes.byref(p))
        nm = ctypes.c_char_p()
        if p.func:
            cu.cuFuncGetName(ctypes.byref(nm), ctypes.c_void_p(p.func))
        else:
            cu.cuKernelGetName(ctypes.byref(nm), ctypes.c_void_p(p.kern))
        names.append((nm.value or b"?").decode())
    return names


def test_graph_capture_under_prefill_context():
    x, r, w = inputs(1379, seed=4)
    hy, base = make("hybrid", w=w), make("baseline", w=w)
    calls = collections.Counter()
    real = hy._nv5_impl

    def counted(*a):
        calls["cuda"] += 1
        return real(*a)
    hy._nv5_impl = counted
    with torch.inference_mode():
        set_context(False)
        yb, rob = base(x, r)
        hy(x, r)
        torch.cuda.synchronize()
        prefill_ctx()
        graph = torch.cuda.CUDAGraph(keep_graph=True)
        with torch.cuda.graph(graph):
            y, ro = hy(x, r)
        graph.replay()
        torch.cuda.synchronize()
    reset_context()
    names = graph_kernel_names(graph)
    check("capture under prefill context: CUDA impl not called", calls["cuda"] == 0, f"calls={calls['cuda']}")
    check("capture under prefill context: 0 candidate nodes, 1 Inductor node",
          sum("nv5_fused_add_rmsnorm" in s for s in names) == 0 and len(names) == 1 and names[0].startswith("triton_"), str(names))
    check("capture under prefill context: replay bitwise equal to baseline eager", torch.equal(y, yb) and torch.equal(ro, rob))


def main():
    test_selection()
    test_policy_table()
    test_module_paths()
    test_errors_propagate()
    test_graph_capture_under_prefill_context()
    n_fail = sum(not ok for _, ok in RESULTS)
    print(f"TOTAL {len(RESULTS)} checks, {len(RESULTS) - n_fail} PASS, {n_fail} FAIL")
    return 0 if n_fail == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
