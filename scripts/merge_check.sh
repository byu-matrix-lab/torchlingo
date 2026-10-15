#!/usr/bin/env bash
#
# Build a pull request merged with its base branch, in a scratch worktree, and run preflight on
# the result: what main would be if the PR merged now.
#
# WHY THIS EXISTS. A PR's green checks say it passed against the base as it was when they ran.
# They are not re-run when the base moves, and GitHub still shows MERGEABLE / CLEAN, so PR #17
# carried ten green ticks while main had grown a gate its new module failed (CLAUDE.md, "A
# green PR is green against the base it last saw"). The remedy is to build the merge result
# before merging. On 2026-10-14 that was done by hand for fifteen PRs in a row. It caught a
# conflict in the generated roadmap on PR #278 and one in CHANGELOG.md on PR #280, both before
# the merge. Every time the steps were the same, and getting them wrong was easy:
#
#   - a branch already checked out in another worktree cannot be checked out again, and the
#     hand-written fallback then ran on the wrong commit. So this fetches GitHub's own
#     pull/N/head ref and checks it out detached, which also works for a PR from a fork;
#   - piping preflight through `tail` hid its exit code, so a failure read as a pass. So the
#     log goes to a file, the exit code is kept, and the verdict is the last line printed.
#
# Git LFS files are not downloaded, the same as in CI, so the checkout is quick and the tests
# that need LFS data skip exactly as they do there. Preflight itself makes the scratch tree
# test its own src/, not the main checkout's installed copy.
#
# Usage:
#   scripts/merge_check.sh PR            preflight on PR merged with its base
#   scripts/merge_check.sh PR --all      ... including notebook execution (minutes)
#   scripts/merge_check.sh PR --keep     ... and keep the scratch worktree afterwards
#
# Exit status, and the last line printed:
#   0  CLEAN           merges without conflict and passes preflight
#   1  PREFLIGHT-FAIL  merges, but preflight fails on the result (the log says where)
#   2  CONFLICT        does not merge; the conflicting files are listed
#   3  ERROR           bad arguments, or the PR or its base could not be fetched
#
# Nothing is pushed and no branch is created: the merge exists only in the scratch worktree,
# which is removed at the end unless --keep is given or the removal fails. The log is kept.
set -uo pipefail

usage() {
  echo "usage: scripts/merge_check.sh PR [--all] [--keep]   (see the header for more)" >&2
  exit 3
}

PR=""
PREFLIGHT_ARGS=()
KEEP=0
for arg in "$@"; do
  case "$arg" in
    --all) PREFLIGHT_ARGS+=("--all") ;;
    --keep) KEEP=1 ;;
    -h | --help) usage ;;
    *[!0-9]*) echo "not a PR number: $arg" >&2; usage ;;
    *) PR="$arg" ;;
  esac
done
[[ -n "$PR" ]] || usage

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT" || exit 3

if ! BASE="$(gh pr view "$PR" --json baseRefName -q .baseRefName 2>&1)"; then
  echo "could not read PR $PR: $BASE" >&2
  echo "ERROR"
  exit 3
fi
# Two fetches, not one: with two refs in one fetch, FETCH_HEAD names the first, and the first
# version of this script merged main with itself and reported CLEAN.
if ! git fetch -q origin "$BASE"; then
  echo "could not fetch $BASE from origin" >&2
  echo "ERROR"
  exit 3
fi
BASE_SHA="$(git rev-parse "origin/$BASE")"
if ! git fetch -q origin "pull/$PR/head"; then
  echo "could not fetch pull/$PR/head from origin" >&2
  echo "ERROR"
  exit 3
fi
HEAD_SHA="$(git rev-parse FETCH_HEAD)"
if [[ "$HEAD_SHA" == "$BASE_SHA" ]]; then
  echo "PR $PR's head is $BASE itself; there is nothing to merge" >&2
  echo "ERROR"
  exit 3
fi

TMP_ROOT="${TMPDIR:-/tmp}"
SCRATCH="$(mktemp -d "${TMP_ROOT%/}/merge-check-$PR.XXXXXX")"
TREE="$SCRATCH/tree"
LOG="$SCRATCH/preflight.log"
echo "PR $PR: head $(git rev-parse --short "$HEAD_SHA"), merged with $BASE at $(git rev-parse --short "$BASE_SHA")"
echo "scratch: $SCRATCH"

if ! GIT_LFS_SKIP_SMUDGE=1 git worktree add -q --detach "$TREE" "$HEAD_SHA"; then
  echo "could not create the scratch worktree" >&2
  echo "ERROR"
  exit 3
fi

cleanup() {
  if [[ $KEEP -eq 1 ]]; then
    echo "kept: $TREE  (remove with: git worktree remove $TREE)"
  elif ! git worktree remove "$TREE" 2>/dev/null; then
    echo "left in place, not clean: $TREE  (look, then: git worktree remove --force $TREE)"
  fi
}

cd "$TREE" || exit 3
if ! GIT_LFS_SKIP_SMUDGE=1 git -c user.name=merge-check -c user.email=merge-check@localhost \
  merge -q --no-edit "$BASE_SHA" > "$SCRATCH/merge.log" 2>&1; then
  echo "conflicts:"
  git diff --name-only --diff-filter=U | sed 's/^/  /'
  git merge --abort
  cd "$REPO_ROOT" && cleanup
  echo "CONFLICT"
  exit 2
fi

# The ${a[@]+...} form, because bash 3.2 (macOS's /bin/bash) calls an empty array unbound.
scripts/preflight.sh ${PREFLIGHT_ARGS[@]+"${PREFLIGHT_ARGS[@]}"} > "$LOG" 2>&1
STATUS=$?
grep -E '^  [a-z]' "$LOG"
cd "$REPO_ROOT" && cleanup
echo "log: $LOG"
if [[ $STATUS -eq 0 ]]; then
  echo "CLEAN"
  exit 0
fi
echo "PREFLIGHT-FAIL"
exit 1
