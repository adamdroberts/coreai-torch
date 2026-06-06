#!/usr/bin/env bash
# Copyright 2026 Apple Inc.
#
# Use of this source code is governed by a BSD-3-clause license that can
# be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

# Wrap in braces so bash reads the entire script before executing.
# This prevents breakage if `git checkout` changes this file on disk.
{
set -euo pipefail

DOCS_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$DOCS_DIR/.." && pwd)"
STAGING="$(mktemp -d)"

# --- Parse args ---
REMOTE="origin"
while [[ $# -gt 0 ]]; do
    case "$1" in
        --remote) REMOTE="$2"; shift 2 ;;
        *) echo "Usage: $0 [--remote <name>]" >&2; exit 1 ;;
    esac
done

echo "Deploying to remote '$REMOTE'"

# --- Pre-flight: required tools ---
for cmd in sphinx-build ghp-import; do
    if ! command -v "$cmd" &>/dev/null; then
        echo "Error: $cmd not found. Install docs deps: uv sync --extra docs" >&2
        exit 1
    fi
done

# --- Save original branch (needed by cleanup) ---
original_ref="$(git -C "$REPO_ROOT" rev-parse --abbrev-ref HEAD)"
[ "$original_ref" = "HEAD" ] && original_ref="$(git -C "$REPO_ROOT" rev-parse HEAD)"

cleanup() {
    git -C "$REPO_ROOT" checkout "$original_ref" 2>/dev/null || true
    rm -rf "$STAGING"
}
trap cleanup EXIT

# Abort if the working tree is dirty.
if ! git -C "$REPO_ROOT" diff --quiet || ! git -C "$REPO_ROOT" diff --cached --quiet; then
    echo "Error: working tree is dirty. Commit or stash changes before deploying." >&2
    exit 1
fi

# Build and stage a version.
#   $1 = version label (used as subdir name and version_match)
#   $2 = git ref to build from (branch, tag, or HEAD)
stage_version() {
    local version="$1" ref="$2"
    echo "=== Building $version (from $ref) ==="
    git checkout "$ref"
    rm -rf "$DOCS_DIR/_build"
    VERSION_MATCH="$version" \
        sphinx-build -b html "$DOCS_DIR" "$DOCS_DIR/_build/html"
    cp -r "$DOCS_DIR/_build/html" "$STAGING/$version"
}

# --- Main ---

# Stage /main/ from main branch
stage_version "main" "main"

# Copy root index page and root versions.json verbatim.
cp "$DOCS_DIR/_static/index.html" "$STAGING/"
cp "$DOCS_DIR/_static/versions.json" "$STAGING/versions.json"

# Push to the target remote
echo "=== Deploying all versions to '$REMOTE' ==="
ghp-import -n -p -f -r "$REMOTE" "$STAGING"
echo "Done."

exit
}
