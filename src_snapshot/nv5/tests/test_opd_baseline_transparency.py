#!/usr/bin/env python
"""OP-D: with NV5_RMSNORM_BACKEND unset, the modified RMSNorm must be bit-identical to upstream bb823b3e.

The upstream layernorm.py is extracted with `git show <sha>:nanovllm/layers/layernorm.py` into $TMPDIR
(no checkout). Covers P0.5 input families F1-F5 and shapes, bf16/fp16, RMS-only 2D and non-contiguous
3D (q_norm layout), memory semantics, and fp32 (upstream alias behaviour must be preserved as-is).

Run: source env/activate.sh && python nv5/tests/test_opd_baseline_transparency.py   (from nano-vllm root)
"""
import importlib.util
import os
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
UPSTREAM_SHA = "bb823b3e06983d71485a8e1f23715ebd87d98ef8"
assert not os.environ.get("NV5_RMSNORM_BACKEND"), "OP-D must run with NV5_RMSNORM_BACKEND unset"

import torch  # noqa: E402

# Many dtype/shape combinations in one process: a hit on the recompile limit would silently fall back to
# eager for BOTH classes and make them trivially equal, so the limit is raised and compiled use is checked.
torch._dynamo.config.cache_size_limit = 256

from nanovllm.layers.layernorm import RMSNorm as ModRMSNorm  # noqa: E402

EPS = 1e-6
RESULTS = []


def check(name, cond, detail=""):
    RESULTS.append((name, bool(cond)))
    if not cond or os.environ.get("NV5_VERBOSE"):
        print(f"{'PASS' if cond else 'FAIL'}  {name}  {detail}")


def load_upstream():
    src = subprocess.run(["git", "-C", str(ROOT), "show", f"{UPSTREAM_SHA}:nanovllm/layers/layernorm.py"],
                         check=True, capture_output=True, text=True).stdout
    d = Path(tempfile.mkdtemp(prefix="nv5_opd_"))
    p = d / "upstream_layernorm.py"
    p.write_text(src)
    spec = importlib.util.spec_from_file_location("nv5_upstream_layernorm", p)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.RMSNorm, p


def make_inputs(family, dt, N, K, g):
    p = 10 if dt == torch.float16 else 7
    rnd_w = lambda: (1.0 + 0.25 * torch.randn(K, device="cuda", generator=g)).to(dt)  # noqa: E731
    if family == "F1_random":
        return (torch.randn(N, K, device="cuda", generator=g).to(dt),
                (4 * torch.randn(N, K, device="cuda", generator=g)).to(dt), rnd_w())
    if family in ("F2_midpoint_w1", "F3_midpoint_randw"):
        vals = torch.tensor([(2 * m + 1) * 2.0 ** (-(p + 1)) for m in range(4)], dtype=torch.float64, device="cuda")
        r = vals[torch.arange(K, device="cuda") % 4].to(dt).unsqueeze(0).expand(N, K).contiguous()
        w = torch.ones(K, dtype=dt, device="cuda") if family == "F2_midpoint_w1" else rnd_w()
        return torch.ones(N, K, dtype=dt, device="cuda"), r, w
    if family == "F4_zero_residual_randw":
        x = (2 * torch.randn(N, K, device="cuda", generator=g)).to(dt)
        return x, torch.zeros_like(x), rnd_w()
    if family == "F5_biased_up":
        mant = torch.randint(0, 2 ** p, (N, K), device="cuda", generator=g).double()
        sign = torch.where(torch.rand(N, K, device="cuda", generator=g) < 0.5, -1.0, 1.0).double()
        scale = 2.0 ** torch.randint(-1, 2, (N, K), device="cuda", generator=g).double()
        return (sign * (1 + mant * 2.0 ** -p) * scale).to(dt), (sign * 0.75 * 2.0 ** -p * scale).to(dt), rnd_w()
    raise ValueError(family)


SHAPES = {"F1_random": [(1, 1024), (4, 896), (64, 1024), (512, 1024), (2048, 1024), (7, 4096), (512, 4096)],
          "F2_midpoint_w1": [(4, 1024), (16, 4096)],
          "F3_midpoint_randw": [(4, 1024), (16, 4096)],
          "F4_zero_residual_randw": [(1, 1024), (64, 1024), (512, 4096)],
          "F5_biased_up": [(1, 1024), (64, 1024), (512, 1024), (16, 4096)]}


def pair(K, dt, w, Up):
    a, b = Up(K, eps=EPS).to(device="cuda", dtype=dt), ModRMSNorm(K, eps=EPS).to(device="cuda", dtype=dt)
    with torch.no_grad():
        a.weight.copy_(w)
        b.weight.copy_(w)
    assert b._nv5_backend == "baseline" and b._nv5_impl is None
    return a, b


def mem_ok(ins, outs):
    ptrs = {t.data_ptr() for t in ins}
    return all(o.data_ptr() not in ptrs for o in outs) and len({o.data_ptr() for o in outs}) == len(outs)


def fused_cases(Up):
    for dt in (torch.bfloat16, torch.float16):
        g = torch.Generator(device="cuda").manual_seed(20260928)
        for fam, shapes in SHAPES.items():
            for N, K in shapes:
                x, r, w = make_inputs(fam, dt, N, K, g)
                a, b = pair(K, dt, w, Up)
                x0, r0 = x.clone(), r.clone()
                with torch.inference_mode():
                    ya, ra = a(x, r)
                    yb, rb = b(x, r)
                    ye, _ = ModRMSNorm.add_rms_forward._torchdynamo_orig_callable(b, x, r)
                tag = f"fused {str(dt)[6:]} {fam} N={N} K={K}"
                check(f"{tag} y bitwise", torch.equal(ya.view(torch.int16), yb.view(torch.int16)))
                check(f"{tag} residual_out bitwise", torch.equal(ra.view(torch.int16), rb.view(torch.int16)))
                check(f"{tag} memory semantics", torch.equal(x, x0) and torch.equal(r, r0)
                      and mem_ok((x, r), (yb, rb)) and mem_ok((x, r), (ya, ra))
                      and yb.is_contiguous() and rb.is_contiguous() and yb.dtype == ya.dtype == dt)
                if fam in ("F1_random", "F4_zero_residual_randw"):
                    frac = (yb != ye).float().mean().item()
                    check(f"{tag} compiled path in use (differs from eager body)", frac > 0.1, f"frac={frac:.3f}")


def rms_only_cases(Up):
    for dt in (torch.bfloat16, torch.float16):
        g = torch.Generator(device="cuda").manual_seed(7)
        for shape in ((1, 1024), (512, 1024), (28, 16, 128), (28, 8, 128)):
            K = shape[-1]
            if len(shape) == 3:
                big = torch.randn(shape[0], 32 * K, device="cuda", generator=g).to(dt)
                x = big[:, : shape[1] * K].view(*shape)
            else:
                x = (2 * torch.randn(*shape, device="cuda", generator=g)).to(dt)
            w = (1 + 0.25 * torch.randn(K, device="cuda", generator=g)).to(dt)
            a, b = pair(K, dt, w, Up)
            x0 = x.clone()
            with torch.inference_mode():
                ya, yb = a(x), b(x)
            tag = f"rms-only {str(dt)[6:]} shape={shape} contiguous={x.is_contiguous()}"
            check(f"{tag} y bitwise", torch.equal(ya.view(torch.int16), yb.view(torch.int16)))
            check(f"{tag} memory semantics", torch.equal(x, x0) and mem_ok((x,), (yb,)) and yb.shape == x.shape)


def fp32_cases(Up):
    g = torch.Generator(device="cuda").manual_seed(3)
    w = (1 + 0.25 * torch.randn(1024, device="cuda", generator=g))
    xs = torch.randn(4, 1024, device="cuda", generator=g)
    rs = torch.randn(4, 1024, device="cuda", generator=g)
    obs = []
    for cls in (Up, ModRMSNorm):
        m = cls(1024, eps=EPS).to(device="cuda", dtype=torch.float32)
        with torch.no_grad():
            m.weight.copy_(w)
        x, r = xs.clone(), rs.clone()
        with torch.inference_mode():
            y, ro = m(x, r)
        xq = xs.clone()
        with torch.inference_mode():
            yq = m(xq)
        obs.append(dict(y=y.clone(), ro=ro.clone(), x_after=x.clone(), r_after=r.clone(), yq=yq.clone(), xq_after=xq.clone(),
                        alias=(y.data_ptr() == x.data_ptr(), ro.data_ptr() == x.data_ptr(), yq.data_ptr() == xq.data_ptr())))
    u, m = obs
    check("fp32 add_rms outputs bitwise equal to upstream", torch.equal(u["y"], m["y"]) and torch.equal(u["ro"], m["ro"]))
    check("fp32 add_rms input mutation identical to upstream", torch.equal(u["x_after"], m["x_after"]) and torch.equal(u["r_after"], m["r_after"]))
    check("fp32 rms-only output/mutation identical to upstream", torch.equal(u["yq"], m["yq"]) and torch.equal(u["xq_after"], m["xq_after"]))
    check("fp32 alias pattern identical to upstream (known upstream behaviour kept)", u["alias"] == m["alias"], f"alias={m['alias']}")


def main():
    Up, p = load_upstream()
    print(f"upstream layernorm extracted from {UPSTREAM_SHA} -> {p}")
    print(f"modified layernorm: {ModRMSNorm.__module__} ({Path(sys.modules[ModRMSNorm.__module__].__file__)})")
    print(f"torch {torch.__version__}, inductor cache {os.environ.get('TORCHINDUCTOR_CACHE_DIR')}")
    fused_cases(Up)
    rms_only_cases(Up)
    fp32_cases(Up)
    import torch._dynamo.utils as du
    print(f"dynamo: stats={dict(du.counters['stats'])} graph_break={dict(du.counters['graph_break'])}")
    n_fail = sum(not ok for _, ok in RESULTS)
    print(f"OP-D TOTAL {len(RESULTS)} checks, {len(RESULTS) - n_fail} PASS, {n_fail} FAIL")
    return 0 if n_fail == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
