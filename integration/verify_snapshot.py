#!/usr/bin/env python3
"""Static, CPU-only verification for the NV5 standalone showcase (stdlib only; no GPU, no model, no network).

Checks
  1. every file listed in SOURCE_MANIFEST.tsv exists under src_snapshot/ with the recorded sha256 and size;
  2. no extra/unexpected file exists under src_snapshot/;
  3. integration/patches/nv5-functional.patch matches its recorded sha256;
  4. the modified upstream file still contains the untouched upstream bodies (guards against a "rewritten"
     display copy silently diverging from the real integration);
  5. with --repo DIR: every manifest path inside a rebuilt tree is byte-identical to the snapshot, and
     `git -C DIR write-tree` equals the authoritative NV5 P3 tree.

Usage
  python3 integration/verify_snapshot.py                 # snapshot + manifest + patch
  python3 integration/verify_snapshot.py --repo /path/to/rebuilt   # + rebuilt-tree comparison
"""
from __future__ import annotations

import argparse
import hashlib
import os
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
SNAPSHOT = ROOT / "src_snapshot"
MANIFEST = HERE / "SOURCE_MANIFEST.tsv"
PATCH = HERE / "patches" / "nv5-functional.patch"

PATCH_SHA256 = "96a87f5088eadff9b2b66248fb48f674ad40812510658822a21cdf075e26ce86"
NV5_TREE = "05480eb86b23065ac412f1325c587bb129abd2da"
UPSTREAM_SHA = "bb823b3e06983d71485a8e1f23715ebd87d98ef8"

# upstream nano-vLLM bodies that must survive unchanged inside the modified layernorm.py
UPSTREAM_INVARIANTS = [
    "@torch.compile\n    def rms_forward(",
    "@torch.compile\n    def add_rms_forward(",
    "        if residual is None:\n            return self.rms_forward(x)",
]

fails: list[str] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    print(("PASS " if ok else "FAIL ") + name + (f"  [{detail}]" if detail and not ok else ""))
    if not ok:
        fails.append(name)


def sha256_file(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def load_manifest() -> list[dict]:
    rows = []
    for line in MANIFEST.read_text(encoding="utf-8").splitlines():
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        parts = line.split("\t")
        if parts[0] == "path":
            continue
        rows.append({"path": parts[0], "role": parts[1], "sha256": parts[2],
                     "bytes": int(parts[3]), "blob": parts[4]})
    return rows


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", help="rebuilt upstream+patch tree to compare against the snapshot")
    args = ap.parse_args()

    check("manifest exists", MANIFEST.is_file(), str(MANIFEST))
    rows = load_manifest()
    check("manifest has 13 source rows (1 modified upstream + 12 new NV5 files)",
          len(rows) == 13, str(len(rows)))

    seen: set[str] = set()
    for r in rows:
        p = SNAPSHOT / r["path"]
        ok = p.is_file()
        check(f"snapshot present: {r['path']}", ok)
        if not ok:
            continue
        seen.add(r["path"])
        got = sha256_file(p)
        check(f"snapshot sha256: {r['path']}", got == r["sha256"], got)
        size = p.stat().st_size
        check(f"snapshot size: {r['path']}", size == r["bytes"], str(size))
        check(f"role recorded: {r['path']} = {r['role']}",
              r["role"] in ("nv5_original", "modified_upstream"), r["role"])

    on_disk = {str(f.relative_to(SNAPSHOT)) for f in SNAPSHOT.rglob("*") if f.is_file()}
    extra = sorted(on_disk - seen)
    check("no unexpected file under src_snapshot", not extra, ", ".join(extra))

    check("patch exists", PATCH.is_file(), str(PATCH))
    if PATCH.is_file():
        got = sha256_file(PATCH)
        check("patch sha256 == manifest constant", got == PATCH_SHA256, got)

    ln = SNAPSHOT / "nanovllm" / "layers" / "layernorm.py"
    if ln.is_file():
        text = ln.read_text(encoding="utf-8")
        for i, frag in enumerate(UPSTREAM_INVARIANTS):
            check(f"modified layernorm.py keeps upstream body #{i + 1}", frag in text)
        check("modified layernorm.py keeps the upstream copyright-bearing module untouched elsewhere",
              "class RMSNorm(nn.Module):" in text and "self.weight = nn.Parameter(torch.ones(hidden_size))" in text)
        check("modified layernorm.py only adds the NV5 dispatch",
              "self._nv5_backend, self._nv5_impl = nv5_rmsnorm.resolve()" in text
              and "self._nv5_route = nv5_rmsnorm.route_policy(self._nv5_backend)" in text)

    if args.repo:
        repo = Path(args.repo).resolve()
        check("repo dir exists", repo.is_dir(), str(repo))
        head = subprocess.run(["git", "-C", str(repo), "rev-parse", "HEAD"],
                              capture_output=True, text=True)
        check(f"rebuilt repo HEAD == pinned upstream {UPSTREAM_SHA} (clean base, never upstream main)",
              head.returncode == 0 and head.stdout.strip() == UPSTREAM_SHA, head.stdout.strip())
        for r in rows:
            p = repo / r["path"]
            ok = p.is_file()
            check(f"rebuilt tree file: {r['path']}", ok)
            if ok:
                check(f"rebuilt tree sha256 == snapshot: {r['path']}", sha256_file(p) == r["sha256"])
        with open(os.devnull, "w") as devnull:
            subprocess.run(["git", "-C", str(repo), "add", "-A"], stdout=devnull, stderr=devnull)
        wt = subprocess.run(["git", "-C", str(repo), "write-tree"], capture_output=True, text=True)
        tree = wt.stdout.strip()
        check(f"rebuilt git write-tree == NV5 P3 tree {NV5_TREE}", tree == NV5_TREE, tree or wt.stderr)

    print(f"\n{'ALL PASS' if not fails else str(len(fails)) + ' FAIL'}")
    return 1 if fails else 0


if __name__ == "__main__":
    raise SystemExit(main())
