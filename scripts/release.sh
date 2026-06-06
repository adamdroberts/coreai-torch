#!/usr/bin/env bash
# Copyright 2026 Apple Inc.
#
# Use of this source code is governed by a BSD-3-clause license that can
# be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

# Release script for coreai-torch.
#
# This script is designed to run from main AFTER the release branch PR has been
# merged. The release branch should have already:
#   - Pinned dependencies (e.g. coreai)
#   - Bumped the version in coreai_torch/__version__.py
#   - Updated docs version metadata
#   - Passed all tests, doc tutorials, doc build, and lint
#
# This script performs the fail-safe post-merge steps:
#   1. Pre-flight checks (clean tree, on main, version not yet tagged)
#   2. Lint (optional sanity check)
#   3. Tests (optional sanity check)
#   4. Git tag + push
#   5. Build wheel + sdist from tagged commit
#   6. Publish to Artifactory via uv publish
#   7. Deploy docs
#
# Usage:
#   UV_PUBLISH_USERNAME="user" UV_PUBLISH_PASSWORD="token" ./scripts/release.sh
#   ./scripts/release.sh --skip-publish    # tag + build + docs only
#   ./scripts/release.sh --dry-run         # preview without executing
#   ./scripts/release.sh --help
{
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"

# --- Defaults ---
YES=false
DRY_RUN=false
SKIP_LINT=false
SKIP_TESTS=false
SKIP_DOCS=false
SKIP_PUBLISH=false
COMMIT_SHA=""
PUBLISH_URL="${UV_PUBLISH_URL:-https://upload.pypi.org/legacy/}"

# --- Colors (only when outputting to a terminal) ---
if [[ -t 1 ]]; then
    RED='\033[0;31m'
    GREEN='\033[0;32m'
    YELLOW='\033[0;33m'
    BOLD='\033[1m'
    RESET='\033[0m'
else
    RED='' GREEN='' YELLOW='' BOLD='' RESET=''
fi

info()    { printf "${GREEN}[INFO]${RESET} %s\n" "$*"; }
warn()    { printf "${YELLOW}[WARN]${RESET} %s\n" "$*"; }
error()   { printf "${RED}[ERROR]${RESET} %s\n" "$*" >&2; }
step()    { printf "\n${BOLD}=== [%s] %s ===${RESET}\n" "$1" "$2"; }

confirm() {
    if $YES; then return 0; fi
    printf "${BOLD}%s [y/N] ${RESET}" "$1"
    read -r answer
    case "$answer" in
        [yY]|[yY][eE][sS]) return 0 ;;
        *) return 1 ;;
    esac
}

usage() {
    cat <<'USAGE'
Usage: ./scripts/release.sh [OPTIONS]

Post-merge release script for coreai-torch.
Reads version from coreai_torch/__version__.py automatically.

Flow: pre-flight → tag → checkout tag → build → publish → docs deploy

Options:
  -y, --yes             Skip all confirmation prompts
  --dry-run             Print what would happen without executing
  --skip-lint           Skip the lint check
  --skip-tests          Skip the test suite
  --skip-docs           Skip doc notebook tests
  --skip-publish        Skip publishing to Artifactory
  --commit SHA          Tag a specific commit instead of HEAD (e.g. merge commit SHA)
  --publish-url URL     Override UV_PUBLISH_URL
  -h, --help            Show this help message

Environment variables:
  UV_PUBLISH_URL        Override package index URL (default: PyPI)
  UV_PUBLISH_TOKEN      Auth token for Artifactory (token-based auth)
  UV_PUBLISH_USERNAME   Username for Artifactory (basic auth, alternative to token)
  UV_PUBLISH_PASSWORD   Password for Artifactory (basic auth, alternative to token)

Examples:
  # Full release (publishes to PyPI by default)
  UV_PUBLISH_USERNAME="user" UV_PUBLISH_PASSWORD="token" ./scripts/release.sh

  # Full release with token auth
  UV_PUBLISH_TOKEN="..." ./scripts/release.sh

  # Tag + docs only (skip publish)
  ./scripts/release.sh --skip-publish

  # Preview what would happen
  ./scripts/release.sh --dry-run
USAGE
}

# --- Parse arguments ---
while [[ $# -gt 0 ]]; do
    case "$1" in
        -y|--yes)           YES=true; shift ;;
        --dry-run)          DRY_RUN=true; shift ;;
        --skip-lint)        SKIP_LINT=true; shift ;;
        --skip-tests)       SKIP_TESTS=true; shift ;;
        --skip-docs)        SKIP_DOCS=true; shift ;;
        --skip-publish)     SKIP_PUBLISH=true; shift ;;
        --commit)           COMMIT_SHA="$2"; shift 2 ;;
        --publish-url)      PUBLISH_URL="$2"; shift 2 ;;
        -h|--help)          usage; exit 0 ;;
        *)                  error "Unknown option: $1"; usage; exit 1 ;;
    esac
done

# --- Read version ---
VERSION_FILE="$REPO_ROOT/coreai_torch/__version__.py"
VERSION="$(sed -n 's/.*__version__[[:space:]]*=[[:space:]]*"\([^"]*\)".*/\1/p' "$VERSION_FILE")"
if [[ -z "$VERSION" ]]; then
    error "Could not read version from $VERSION_FILE"
    exit 1
fi
TAG="v${VERSION}"

run() {
    if $DRY_RUN; then
        info "[dry-run] $*"
    else
        "$@"
    fi
}

# =========================================================================
# Step 1: Pre-flight checks
# =========================================================================
step "1/7" "Pre-flight checks"

info "Version: $VERSION (tag: $TAG)"
info "Repository: $REPO_ROOT"

# Clean working tree
if ! git -C "$REPO_ROOT" diff --quiet || ! git -C "$REPO_ROOT" diff --cached --quiet; then
    error "Working tree is dirty. Commit or stash changes before releasing."
    exit 1
fi
info "Working tree is clean"

# Check branch
BRANCH="$(git -C "$REPO_ROOT" rev-parse --abbrev-ref HEAD)"
if [[ "$BRANCH" != "main" ]]; then
    warn "Not on main branch (currently on '$BRANCH')"
    if ! $YES; then
        confirm "Continue anyway?" || exit 1
    fi
else
    info "On main branch"
fi

# Check tag doesn't exist
if git -C "$REPO_ROOT" tag -l "$TAG" | grep -q "$TAG"; then
    error "Tag $TAG already exists. Has this version already been released?"
    exit 1
fi
info "Tag $TAG does not exist yet"

# Check uv is available
if ! command -v uv &>/dev/null; then
    error "uv is not installed. Install it: https://docs.astral.sh/uv/"
    exit 1
fi
info "uv is available"

# Check publish credentials (if publishing)
if ! $SKIP_PUBLISH; then
    if [[ -z "${UV_PUBLISH_TOKEN:-}" ]] && [[ -z "${UV_PUBLISH_USERNAME:-}" || -z "${UV_PUBLISH_PASSWORD:-}" ]]; then
        error "No publish credentials found. Set UV_PUBLISH_TOKEN or UV_PUBLISH_USERNAME+UV_PUBLISH_PASSWORD."
        exit 1
    fi
    info "Publish URL: $PUBLISH_URL"
    info "Publish credentials: configured"
fi

# =========================================================================
# Step 2: Lint (optional sanity check)
# =========================================================================
if $SKIP_LINT; then
    step "2/7" "Lint (skipped)"
else
    step "2/7" "Lint"
    run uv run ruff check "$REPO_ROOT"
    run uv run ruff format --check "$REPO_ROOT"
    info "Lint passed"
fi

# =========================================================================
# Step 3: Tests (optional sanity check)
# =========================================================================
if $SKIP_TESTS; then
    step "3/7" "Tests (skipped)"
else
    step "3/7" "Tests"
    run uv run pytest "$REPO_ROOT/tests/" -n auto
    info "Tests passed"
fi

# =========================================================================
# Step 4: Git tag + push
# =========================================================================
step "4/7" "Git tag"

if [[ -n "$COMMIT_SHA" ]]; then
    # Verify the commit exists
    if ! git -C "$REPO_ROOT" cat-file -t "$COMMIT_SHA" &>/dev/null; then
        error "Commit $COMMIT_SHA does not exist"
        exit 1
    fi
    info "Tagging specific commit: $COMMIT_SHA"
fi

if confirm "Create and push tag $TAG?"; then
    run git -C "$REPO_ROOT" tag "$TAG" ${COMMIT_SHA:+"$COMMIT_SHA"}
    run git -C "$REPO_ROOT" push origin "$TAG"
    info "Tag $TAG created and pushed"
else
    warn "Tagging skipped — cannot proceed with build from tag"
    if ! $DRY_RUN; then
        error "Build requires a tagged commit. Aborting."
        exit 1
    fi
fi

# =========================================================================
# Step 5: Build from tagged commit
# =========================================================================
step "5/7" "Build (from $TAG)"

# Checkout the tag for a clean, tagged build
ORIGINAL_REF="$(git -C "$REPO_ROOT" rev-parse --abbrev-ref HEAD)"
[ "$ORIGINAL_REF" = "HEAD" ] && ORIGINAL_REF="$(git -C "$REPO_ROOT" rev-parse HEAD)"

restore_branch() {
    git -C "$REPO_ROOT" checkout "$ORIGINAL_REF" 2>/dev/null || true
}
trap restore_branch EXIT

run git -C "$REPO_ROOT" checkout "$TAG"
run rm -rf "$REPO_ROOT/dist"
run uv build --directory "$REPO_ROOT"

if ! $DRY_RUN; then
    info "Built artifacts:"
    ls -lh "$REPO_ROOT/dist/"
fi

# Return to original branch
run git -C "$REPO_ROOT" checkout "$ORIGINAL_REF"
trap - EXIT

# =========================================================================
# Step 6: Publish
# =========================================================================
if $SKIP_PUBLISH; then
    step "6/7" "Publish (skipped)"
else
    step "6/7" "Publish"
    info "Target: $PUBLISH_URL"

    if ! $DRY_RUN; then
        ls -1 "$REPO_ROOT/dist/"
    fi

    if confirm "Publish coreai-torch $VERSION to Artifactory?"; then
        run uv publish --publish-url "$PUBLISH_URL" "$REPO_ROOT/dist/"*
        info "Published successfully"
    else
        warn "Publish skipped by user"
    fi
fi

# =========================================================================
# Step 7: Deploy docs
# =========================================================================
if $SKIP_DOCS; then
    step "7/7" "Doc deploy (skipped)"
else
    step "7/7" "Deploy docs"
    if confirm "Deploy documentation?"; then
        run uv run "$REPO_ROOT/docs/deploy.sh"
        run uv run "$REPO_ROOT/docs/deploy.sh" --remote pie
        info "Docs deployed to origin and pie"
    else
        warn "Doc deploy skipped by user"
    fi
fi

# =========================================================================
# Done
# =========================================================================
printf "\n%s%sRelease %s complete.%s\n" "$GREEN" "$BOLD" "$VERSION" "$RESET"
info "Tag: $TAG"
if ! $SKIP_PUBLISH; then
    info "Package: coreai-torch==$VERSION"
fi

exit
}
