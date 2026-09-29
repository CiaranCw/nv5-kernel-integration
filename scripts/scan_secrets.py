#!/usr/bin/env python3
"""Pre-publication scan for the NV5 standalone showcase (stdlib only, CPU-only, read-only).

Looks for, across every text file that would be committed:
  - credentials (GitHub tokens, AWS keys, private keys, generic 'token/password/secret = ...');
  - private e-mail addresses (QQ-style numeric addresses);
  - absolute machine paths (/home/<user>/..., /Users/<...>, C:\\Users\\...);
  - local file:// URLs and private model paths;
  - oversized files (would bloat the repository);
  - non-UTF-8 / binary files where text is expected.

Usage:  python3 scripts/scan_secrets.py [--root DIR] [--max-bytes N]
Exit code 0 = clean, 1 = findings.
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

DEFAULT_MAX_BYTES = 400_000

PATTERNS = [
    ("github token", re.compile(r"\b(gho|ghu|ghs|ghr|ghp)_[A-Za-z0-9]{16,}\b")),
    ("github fine-grained pat", re.compile(r"\bgithub_pat_[A-Za-z0-9_]{20,}\b")),
    ("aws access key id", re.compile(r"\bAKIA[0-9A-Z]{16}\b")),
    ("private key block", re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----")),
    ("generic secret assignment",
     re.compile(r"(?i)\b(api[_-]?key|access[_-]?token|secret[_-]?key|password|passwd|private[_-]?token)\b\s*[:=]\s*['\"][^'\"]{8,}['\"]")),
    ("qq-style numeric e-mail", re.compile(r"\b[0-9]{5,}@qq\.com\b")),
    ("absolute home path", re.compile(r"/home/[A-Za-z0-9_.-]+/")),
    ("macOS home path", re.compile(r"/Users/[A-Za-z0-9_.-]+/")),
    ("windows user path", re.compile(r"[A-Za-z]:\\\\?Users\\\\?")),
    # "file://" alone is used in the docs to *describe* removed local paths; only absolute forms matter
    ("local file url", re.compile(r"file:///")),
]

TEXT_SUFFIXES = {".md", ".py", ".sh", ".tsv", ".txt", ".json", ".cu", ".patch", ".gitignore", ""}
# the scanner's own pattern table literally contains the examples it looks for, so it skips itself
SELF = Path(__file__).resolve()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=str(Path(__file__).resolve().parents[1]))
    ap.add_argument("--max-bytes", type=int, default=DEFAULT_MAX_BYTES)
    args = ap.parse_args()

    root = Path(args.root).resolve()
    findings: list[str] = []
    scanned = 0

    for p in sorted(f for f in root.rglob("*") if f.is_file()):
        rel = str(p.relative_to(root))
        if rel.startswith(".git/") or rel == ".git" or p.resolve() == SELF:
            continue
        size = p.stat().st_size
        scanned += 1
        if size > args.max_bytes:
            findings.append(f"OVERSIZE {rel} ({size} bytes > {args.max_bytes})")
        if p.suffix not in TEXT_SUFFIXES and p.name not in ("LICENSE", ".gitignore"):
            findings.append(f"UNEXPECTED-FILETYPE {rel}")
            continue
        try:
            text = p.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            findings.append(f"NON-UTF8 {rel}")
            continue
        for label, rx in PATTERNS:
            for m in rx.finditer(text):
                line = text.count("\n", 0, m.start()) + 1
                snippet = m.group(0)[:60].replace("\n", " ")
                findings.append(f"{label.upper()} {rel}:{line} -> {snippet}")

    print(f"scanned {scanned} files under {root}")
    if findings:
        print(f"\n{len(findings)} FINDING(S):")
        for f in findings:
            print("  " + f)
        print("\nReview each finding before publishing. Note: 'local file url' is expected to hit only")
        print("the sanitized lock file comment if that comment itself is quoted, and 'absolute home path'")
        print("must never appear in docs or committed artefacts.")
        return 1
    print("CLEAN: no credential, private e-mail, absolute machine path, or oversized file found.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
