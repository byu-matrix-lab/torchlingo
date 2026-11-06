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
# THE INSTALL CHECK. When preflight passes, every notebook the PR adds or changes that has an
# install cell then runs through `student_path.sh --colab --install-only`, from the merged tree,
# each in an environment of its own (a shared environment exercises only the first install).
# PR #329's COMET cell read a helper defined in the setup cell, and that check caught it only
# after promotion, on `course` 6405a02 (fixed in PR #331): student_path.yml runs it on `course`,
# which is too late. Task #273. About 35 s a notebook; it needs uv and the network. It installs
# TorchLingo from the merged commit, not from `course`, so one PR can carry a library change and
# the notebook that uses it. Until Task #311 it installed from `course`, and A14's COMET fix took
# two PRs and two landings, the function first and the notebook after its promotion.
# student_path.yml still tests what `course` serves.
#
# SKIPPING WHAT ALREADY PASSED. Preflight, and each notebook's install check, are skipped if
# this exact tree passed them within six hours (scripts/tree_passes.sh). That covers running
# preflight on a branch up to date with main, and retrying a landing after a GitHub error.
# --fresh runs everything. A PR touching only notes/ runs `preflight.sh --notes`, seconds rather
# than minutes, since it changes no code (Task #312).
#
# Usage:
#   scripts/merge_check.sh PR              preflight on PR merged with its base, then the
#                                          install check on the notebooks the PR changes
#   scripts/merge_check.sh PR --all        ... with notebook execution in preflight (minutes)
#   scripts/merge_check.sh PR --keep       ... and keep the scratch worktree afterwards
#   scripts/merge_check.sh PR --no-student ... skipping the install check, e.g. with no uv
#   scripts/merge_check.sh PR --fresh      ... running everything, whatever already passed
#
# Exit status, and the last line printed:
#   0  CLEAN           merges without conflict, passes preflight and the install check
#   1  PREFLIGHT-FAIL  merges, but preflight fails on the result (the log says where)
#   2  CONFLICT        does not merge; the conflicting files are listed
#   3  ERROR           bad arguments, or the PR or its base could not be fetched
#   4  STUDENT-FAIL    passes preflight, but a changed notebook's install check fails
#
# Nothing is pushed and no branch is created: the merge exists only in the scratch worktree,
# which is removed at the end unless --keep is given or the removal fails. The log is kept.
set -uo pipefail

usage() {
  echo "usage: scripts/merge_check.sh PR [--all] [--keep] [--no-student] [--fresh]   (see the header for more)" >&2
  exit 3
}

PR=""
PREFLIGHT_ARGS=()
PREFLIGHT_KIND=preflight
KEEP=0
STUDENT=1
FRESH=0
for arg in "$@"; do
  case "$arg" in
    --all) PREFLIGHT_ARGS+=("--all"); PREFLIGHT_KIND=preflight-all ;;
    --keep) KEEP=1 ;;
    --no-student) STUDENT=0 ;;
    --fresh) FRESH=1 ;;
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

# shellcheck source=scripts/tree_passes.sh
source scripts/tree_passes.sh
MERGED_TREE="$(tree_of_head)"

already_passed() {
  [[ $FRESH -eq 0 ]] && tree_passed_recently "$MERGED_TREE" "$1"
}

NOTES_ONLY=0
if [[ $FRESH -eq 0 && ${#PREFLIGHT_ARGS[@]} -eq 0 &&
  -z "$(git diff --name-only "$BASE_SHA...$HEAD_SHA" -- . ':(exclude)notes/')" ]]; then
  NOTES_ONLY=1
fi

if already_passed "$PREFLIGHT_KIND"; then
  echo "  preflight         skipped: this exact tree passed it recently (--fresh to run it)"
elif [[ $NOTES_ONLY -eq 1 ]]; then
  # Nothing here can change what the tests see except the roadmap, which is the notebook
  # metadata tests' input. Not recorded as a pass: the full preflight did not run.
  scripts/preflight.sh --notes > "$LOG" 2>&1
  STATUS=$?
  grep -E '^  [a-z]' "$LOG"
  if [[ $STATUS -ne 0 ]]; then
    cd "$REPO_ROOT" && cleanup
    echo "log: $LOG"
    echo "PREFLIGHT-FAIL"
    exit 1
  fi
else
  # The ${a[@]+...} form, because bash 3.2 (macOS's /bin/bash) calls an empty array unbound.
  scripts/preflight.sh ${PREFLIGHT_ARGS[@]+"${PREFLIGHT_ARGS[@]}"} > "$LOG" 2>&1
  STATUS=$?
  grep -E '^  [a-z]' "$LOG"
  if [[ $STATUS -ne 0 ]]; then
    cd "$REPO_ROOT" && cleanup
    echo "log: $LOG"
    echo "PREFLIGHT-FAIL"
    exit 1
  fi
  record_tree_pass "$MERGED_TREE" "$PREFLIGHT_KIND"
fi

# The install check (see the header): only notebooks the PR itself adds or changes, and only
# those with an install cell, since student_path.sh skips the rest after building each an
# environment.
STUDENT_STATUS=0
if [[ $STUDENT -eq 1 ]]; then
  NOTEBOOKS=()
  while IFS= read -r nb; do
    grep -q 'pip install' "$nb" && NOTEBOOKS+=("$nb")
  done < <(git diff --name-only --diff-filter=ACMR "$BASE_SHA...$HEAD_SHA" -- '*.ipynb')
  if [[ ${#NOTEBOOKS[@]} -eq 0 ]]; then
    echo "  install check     none: no notebook the PR changes has an install cell"
  elif ! command -v uv >/dev/null; then
    echo "  install check     needs uv; install it, or pass --no-student to skip the check"
    STUDENT_STATUS=1
  else
    # An export of the merged commit, so that building it cannot leave the worktree unclean.
    EXPORT="$SCRATCH/merged-commit"
    mkdir "$EXPORT" && git archive HEAD | tar -x -C "$EXPORT"
    for nb in "${NOTEBOOKS[@]}"; do
      STEM="$(basename "$nb" .ipynb)"
      if already_passed "install-$STEM"; then
        echo "  install check     skipped: passed recently on this tree  $nb"
        continue
      fi
      STUDENT_LOG="$SCRATCH/student-$STEM.log"
      if TORCHLINGO_SOURCE="$EXPORT" STUDENT_DIR="$SCRATCH/student-$STEM" \
        scripts/student_path.sh --colab --install-only "$nb" > "$STUDENT_LOG" 2>&1; then
        echo "  install check     ok    $nb"
        record_tree_pass "$MERGED_TREE" "install-$STEM"
      else
        echo "  install check     FAIL  $nb  (log: $STUDENT_LOG)"
        STUDENT_STATUS=1
      fi
    done
  fi
fi

cd "$REPO_ROOT" && cleanup
echo "log: $LOG"
if [[ $STUDENT_STATUS -ne 0 ]]; then
  echo "STUDENT-FAIL"
  exit 4
fi
echo "CLEAN"
exit 0
