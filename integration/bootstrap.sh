#!/usr/bin/env bash
# NV5 standalone showcase — rebuild the authoritative NV5 functional tree in an isolated directory.
#
#   upstream nano-vLLM @ bb823b3e  +  integration/patches/nv5-functional.patch  ==  NV5 P3 tree db8dc57d
#
# This script NEVER deletes, overwrites or "repairs" an existing target directory: if the target already
# exists and is non-empty, it exits non-zero. It only creates a brand new Git work tree.
#
# Usage:
#   ./integration/bootstrap.sh <new-empty-target-dir>
#   ./integration/bootstrap.sh --upstream-url <git-url-or-local-path> <new-empty-target-dir>
#   ./integration/bootstrap.sh --patch <path-to-patch> <new-empty-target-dir>
#
# Exit codes: 0 ok | 2 usage/target problem | 3 patch rejected by validation | 4 upstream unreachable or
#             SHA/tree mismatch | 5 patch did not apply | 6 rebuilt tree != expected NV5 tree
set -Eeuo pipefail

# ---- fixed, never auto-updated to upstream main ------------------------------------------------
UPSTREAM_URL_DEFAULT="https://github.com/GeeeekExplorer/nano-vllm.git"
UPSTREAM_SHA="bb823b3e06983d71485a8e1f23715ebd87d98ef8"
UPSTREAM_TREE="5ec2b4b8034b26f5cc9af73c13d520a2aa8774f0"
NV5_TREE="05480eb86b23065ac412f1325c587bb129abd2da"
PATCH_SHA256="96a87f5088eadff9b2b66248fb48f674ad40812510658822a21cdf075e26ce86"

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
UPSTREAM_URL="$UPSTREAM_URL_DEFAULT"
PATCH_FILE="$SCRIPT_DIR/patches/nv5-functional.patch"
TARGET=""

die() { printf 'bootstrap: ERROR: %s\n' "$*" >&2; exit "${DIE_CODE:-2}"; }
log() { printf 'bootstrap: %s\n' "$*"; }

while [ $# -gt 0 ]; do
  case "$1" in
    -u|--upstream-url) UPSTREAM_URL="${2:?--upstream-url needs a value}"; shift 2 ;;
    -p|--patch)        PATCH_FILE="${2:?--patch needs a value}"; shift 2 ;;
    -h|--help)         sed -n '2,20p' "$0"; exit 0 ;;
    --)                shift; break ;;
    -*)                die "unknown option $1" ;;
    *)                 [ -n "$TARGET" ] && die "unexpected extra argument: $1"; TARGET="$1"; shift ;;
  esac
done
while [ $# -gt 0 ]; do [ -n "$TARGET" ] && die "unexpected extra argument: $1"; TARGET="$1"; shift; done

[ -n "$TARGET" ] || die "missing target directory (must be a NEW, empty path)"

# ---- 1. target must not exist or be empty; never rm -rf ---------------------------------------
if [ -e "$TARGET" ]; then
  [ -d "$TARGET" ] || die "target exists and is not a directory: $TARGET"
  if [ -n "$(ls -A "$TARGET" 2>/dev/null)" ]; then
    die "target directory already exists and is NOT empty: $TARGET
      bootstrap.sh refuses to touch existing content. Choose a different, empty path."
  fi
  log "target exists but is empty, reusing it: $TARGET"
fi
mkdir -p "$TARGET"
TARGET_ABS="$(cd -- "$TARGET" && pwd)"

# ---- 2. patch validation (sha256 + no path traversal) -----------------------------------------
[ -f "$PATCH_FILE" ] || die "patch not found: $PATCH_FILE"
command -v git >/dev/null || die "git is required"

if command -v sha256sum >/dev/null; then
  GOT="$(sha256sum "$PATCH_FILE" | awk '{print $1}')"
elif command -v shasum >/dev/null; then
  GOT="$(shasum -a 256 "$PATCH_FILE" | awk '{print $1}')"
else
  DIE_CODE=3 die "no sha256 tool found (sha256sum or shasum)"
fi
[ "$GOT" = "$PATCH_SHA256" ] || DIE_CODE=3 die "patch sha256 mismatch
      expected $PATCH_SHA256
      actual   $GOT
      file     $PATCH_FILE"
log "patch sha256 OK  $PATCH_SHA256"

# reject absolute or escaping paths before letting git apply touch anything
# ("/dev/null" is the legitimate git-diff placeholder for created files)
BADPATHS="$(sed -nE 's/^\+\+\+ //p' "$PATCH_FILE" \
            | grep -vE '^/dev/null$' \
            | grep -E '(^/|(^|/)\.\.(/|$))' || true)"
if [ -n "$BADPATHS" ]; then
  DIE_CODE=3 die "patch contains an absolute or path-traversal target; refusing to apply:
$BADPATHS"
fi
log "patch path-traversal scan OK"

# ---- 3. fetch the FIXED upstream commit (never main) -------------------------------------------
if [ "$UPSTREAM_URL" != "$UPSTREAM_URL_DEFAULT" ]; then
  log "WARNING: non-default upstream source in use: $UPSTREAM_URL"
  log "         (the official pinned source is $UPSTREAM_URL_DEFAULT)"
fi

git init -q "$TARGET_ABS"
git -C "$TARGET_ABS" remote add origin "$UPSTREAM_URL"
log "fetching $UPSTREAM_SHA from $UPSTREAM_URL"
export GIT_TERMINAL_PROMPT=0
if ! git -C "$TARGET_ABS" fetch --depth 1 origin "$UPSTREAM_SHA" >/dev/null 2>&1; then
  log "fetch-by-sha failed; retrying with a full fetch of refs/heads/*"
  git -C "$TARGET_ABS" fetch origin '+refs/heads/*:refs/remotes/origin/*' >/dev/null 2>&1 \
    || DIE_CODE=4 die "cannot fetch from $UPSTREAM_URL"
fi
git -C "$TARGET_ABS" checkout -q FETCH_HEAD 2>/dev/null \
  || git -C "$TARGET_ABS" checkout -q "$UPSTREAM_SHA" 2>/dev/null \
  || DIE_CODE=4 die "commit $UPSTREAM_SHA not available from $UPSTREAM_URL"

HEAD_SHA="$(git -C "$TARGET_ABS" rev-parse HEAD)"
[ "$HEAD_SHA" = "$UPSTREAM_SHA" ] || DIE_CODE=4 die "upstream HEAD $HEAD_SHA != pinned $UPSTREAM_SHA"
HEAD_TREE="$(git -C "$TARGET_ABS" rev-parse 'HEAD^{tree}')"
[ "$HEAD_TREE" = "$UPSTREAM_TREE" ] || DIE_CODE=4 die "upstream tree $HEAD_TREE != expected $UPSTREAM_TREE"
log "upstream pinned OK  commit=$HEAD_SHA  tree=$HEAD_TREE"

# ---- 4. apply the patch ------------------------------------------------------------------------
git -C "$TARGET_ABS" apply --check "$PATCH_FILE" \
  || DIE_CODE=5 die "git apply --check rejected the patch (upstream work tree does not match the patch context)"
log "git apply --check OK"
git -C "$TARGET_ABS" apply "$PATCH_FILE" \
  || DIE_CODE=5 die "git apply failed to apply the patch"
log "patch applied"

# ---- 5. rebuild the tree and compare with the authoritative NV5 P3 tree ------------------------
git -C "$TARGET_ABS" add -A
REBUILT="$(git -C "$TARGET_ABS" write-tree)"
log "rebuilt tree   $REBUILT"
log "expected tree  $NV5_TREE"
[ "$REBUILT" = "$NV5_TREE" ] || DIE_CODE=6 die "rebuilt tree does not equal the authoritative NV5 P3 tree
      This is a hard failure: the patch does not exactly reconstruct db8dc57d^{tree}.
      Diagnose line endings / file modes / missing paths before trusting any other result."

# everything the patch produced must be staged and nothing may be left over
# (note: index vs HEAD legitimately differs - that difference IS the NV5 change set)
if ! git -C "$TARGET_ABS" diff --quiet; then
  DIE_CODE=6 die "unstaged modifications remain after 'git add -A'"
fi
LEFTOVER="$(git -C "$TARGET_ABS" ls-files --others --exclude-standard)"
if [ -n "$LEFTOVER" ]; then
  DIE_CODE=6 die "untracked leftovers after 'git add -A': $LEFTOVER"
fi
log "index matches work tree; no leftovers"

cat <<EOF

bootstrap: SUCCESS
  target        : $TARGET_ABS
  upstream      : $UPSTREAM_URL @ $UPSTREAM_SHA (tree $UPSTREAM_TREE)
  patch         : $PATCH_FILE (sha256 $PATCH_SHA256)
  rebuilt tree  : $REBUILT  ==  NV5 P3 db8dc57d^{tree}
  next          : python3 $SCRIPT_DIR/verify_snapshot.py --repo "$TARGET_ABS"
EOF
