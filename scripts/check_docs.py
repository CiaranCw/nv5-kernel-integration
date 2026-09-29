#!/usr/bin/env python3
"""Static, CPU-only consistency check for the NV5 standalone showcase docs (stdlib only).

Checks
  - relative Markdown links in README.md and docs/*.md resolve;
  - no absolute local home paths and no obvious secrets inside the docs;
  - EVIDENCE_MAP sha256 + key string for all 24 evidence files;
  - headline P3 numbers stated in the docs match evidence/p3/evidence/perf_matrix/e2e_analysis.json;
  - a few cited source lines in src_snapshot/ still contain the cited code;
  - the upstream MIT notice is preserved.

Usage (from anywhere):  python3 scripts/check_docs.py
"""
from __future__ import annotations

import hashlib
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "docs"
EVIDENCE = ROOT / "evidence"
fails: list[str] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    print(("PASS " if ok else "FAIL ") + name + (f"  [{detail}]" if detail and not ok else ""))
    if not ok:
        fails.append(name)


md_files = [ROOT / "README.md", ROOT / "THIRD_PARTY_NOTICES.md"] + sorted(DOCS.glob("*.md"))
for md in md_files:
    text = md.read_text(encoding="utf-8")
    links = [l for l in re.findall(r"\]\(([^)\s]+)\)", text)
             if not l.startswith(("http://", "https://", "#", "mailto:"))]
    bad = [l for l in links if not (md.parent / l.split("#")[0].split(":")[0]).exists()]
    check(f"links resolve: {md.relative_to(ROOT)} ({len(links)} relative links)", not bad, ", ".join(bad))
    check(f"no absolute home paths: {md.relative_to(ROOT)}",
          not re.search(r"/home/[a-z]|/Users/[a-z]|C:\\\\?Users", text))
    check(f"no private e-mail address: {md.relative_to(ROOT)}",
          not re.search(r"[0-9]{5,}@qq\.com", text))

rows = [r for r in (DOCS / "EVIDENCE_MAP.md").read_text(encoding="utf-8").splitlines()
        if re.match(r"\| E\d\d ", r)]
for r in rows:
    cells = [c.strip() for c in r.strip("|").split("|")]
    # the link label itself contains "evidence/", so take the target and split on its first "evidence/"
    rel = re.search(r"\]\(([^)]+)\)", cells[4]).group(1).split("evidence/", 1)[1]
    token = cells[5].strip("`").replace("&#124;", "|")
    digest = cells[6].strip("`")
    p = EVIDENCE / rel
    ok = p.exists() and hashlib.sha256(p.read_bytes()).hexdigest() == digest \
        and token in p.read_text(errors="replace")
    check(f"evidence {cells[0]} sha256 + key string: {rel}", ok)
check("evidence map has 24 rows", len(rows) == 24, str(len(rows)))

a = json.loads((EVIDENCE / "p3/evidence/perf_matrix/e2e_analysis.json").read_text())
t8, t16 = a["paired"]["bs8_T8|total_ms"], a["paired"]["bs8_T16|total_ms"]
facts = {
    "bs8_T8 mean d_H": (f"{100 * t8['d_H']['mean']:.2f}", "-3.34"),
    "bs8_T8 CI95": (f"[{100 * t8['d_H']['ci95'][0]:.2f}, {100 * t8['d_H']['ci95'][1]:.2f}]", "[-5.94, -0.93]"),
    "bs8_T8 negative rounds": (f"{t8['d_H']['n_neg']}/8", "6/8"),
    "bs8_T8 null SD": (f"{100 * t8['d_null']['sd']:.2f}", "6.55"),
    "bs8_T16 mean d_H": (f"{100 * t16['d_H']['mean']:.2f}", "-1.37"),
    "overall status": (a["overall"]["suggested_status"], "NV5_P3_HYBRID_COMPLETE_NO_STABLE_GAIN"),
}
for k, (got, want) in facts.items():
    check(f"analysis json: {k} = {want}", got == want, got)

perf = (DOCS / "PERFORMANCE_AND_LIMITATIONS.md").read_text(encoding="utf-8")
for s in ("−3.34%", "[−5.94, −0.93]", "6/8", "6.55%", "−1.37%", "NO_STABLE_GAIN"):
    check(f"PERFORMANCE_AND_LIMITATIONS.md states {s}", s in perf)

readme = (ROOT / "README.md").read_text(encoding="utf-8")
for s in ("NO_STABLE_GAIN", "6/8", "6.55%", "M_CAL_NONDISCRIMINATIVE", "M-2", "G8", "1.2–1.45%"):
    check(f"README.md mentions {s}", s in readme)
check("README does not claim a stable E2E gain",
      "稳定端到端加速" in readme and "不存在已证实的稳定端到端加速" in readme)

CITES = [
    ("nanovllm/layers/layernorm.py", "self._nv5_backend, self._nv5_impl = nv5_rmsnorm.resolve()"),
    ("nanovllm/layers/layernorm.py", "self._nv5_route = nv5_rmsnorm.route_policy(self._nv5_backend)"),
    ("nanovllm/layers/layernorm.py", "return self._nv5_impl(x, residual, self.weight, self.eps)"),
    ("nanovllm/layers/layernorm.py", "@torch.compile"),
    ("nanovllm/layers/nv5_rmsnorm/__init__.py", "importlib.import_module(_MODULES[name])"),
    ("nanovllm/layers/nv5_rmsnorm/hybrid_backend.py",
     "if not get_context().is_prefill or torch.cuda.is_current_stream_capturing():"),
    ("nanovllm/layers/nv5_rmsnorm/cuda_backend.py", '_ext = load(name="nv5_fused_add_rmsnorm_nc"'),
    ("nanovllm/layers/nv5_rmsnorm/triton_backend.py", "inv = libdevice.rsqrt(tl.sum(s * s, axis=0) / H + eps)"),
    ("nanovllm/layers/nv5_rmsnorm/csrc/nv5_fused_add_rmsnorm.cu", "reinterpret_cast<const uint4*>(x + base)"),
    ("nv5/tests/test_opd_baseline_transparency.py",
     'UPSTREAM_SHA = "bb823b3e06983d71485a8e1f23715ebd87d98ef8"'),
]
for f, code in CITES:
    p = ROOT / "src_snapshot" / f
    check(f"cited source still contains the quoted code: {f}", p.is_file() and code in p.read_text(encoding="utf-8"))

lic = (ROOT / "LICENSE").read_text(encoding="utf-8")
check("upstream MIT notice preserved (Copyright (c) 2025 Xingkai Yu)",
      "MIT License" in lic and "Copyright (c) 2025 Xingkai Yu" in lic)
notices = (ROOT / "THIRD_PARTY_NOTICES.md").read_text(encoding="utf-8")
check("THIRD_PARTY_NOTICES states the NV5 files are not covered by the upstream MIT grant",
      "not** covered by the upstream MIT grant" in notices)
check("THIRD_PARTY_NOTICES marks layernorm.py as a modified upstream file",
      "modified copy of the upstream file" in notices)
check("THIRD_PARTY_NOTICES states no C6 code or binary was copied",
      "No C6 source file, build artifact or binary is copied" in notices)

print(f"\n{'ALL PASS' if not fails else str(len(fails)) + ' FAIL'}")
sys.exit(1 if fails else 0)
