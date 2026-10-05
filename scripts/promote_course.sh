#!/usr/bin/env bash
#
# Move the `course` branch, which every course notebook installs TorchLingo from, to a commit
# of `main` whose checks have all passed.
#
# WHY THIS EXISTS. The notebooks' install cells read
#     %pip install "torchlingo @ git+https://github.com/byu-matrix-lab/torchlingo@course"
# so `course` is what twenty-four students run. Pointing them at `main` would let one red
# merge break every badge at once; pointing them at PyPI made a notebook wait for a release
# before it could use library code merged beside it. `course` is the middle: it moves only
# here, only forward, and only to a commit CI has passed. Task #196, Eric 2026-10-05.
#
# WHAT IT CHECKS, before pushing anything:
#   - the commit is on origin/main;
#   - GitHub has check runs for it, every one has completed, and none failed. The
#     student-path job is left out: it tests what `course` already holds, not this commit;
#   - the move is a fast-forward of origin/course (git refuses anything else, and this script
#     never forces).
#
# Usage:
#   scripts/promote_course.sh            # promote origin/main's head
#   scripts/promote_course.sh <commit>   # promote an earlier commit of main
#
# Each push to `course` starts the student-path workflow, which runs every notebook from its
# own install cell: watch it with `gh run list --workflow student_path.yml`.

set -euo pipefail

REPO="byu-matrix-lab/torchlingo"
cd "$(dirname "$0")/.."

git fetch --quiet origin main
target="$(git rev-parse "${1:-origin/main}^{commit}")"
short="$(git rev-parse --short "$target")"

if ! git merge-base --is-ancestor "$target" origin/main; then
  echo "REFUSED: $short is not on origin/main." >&2
  exit 1
fi

runs="$(gh api "repos/$REPO/commits/$target/check-runs?per_page=100" \
  --jq '.check_runs[] | select(.name | startswith("Notebooks from") | not)
        | "\(.status) \(.conclusion // "-") \(.name)"')"

if [ -z "$runs" ]; then
  echo "REFUSED: GitHub has no check runs for $short yet." >&2
  exit 1
fi

pending="$(grep -v '^completed ' <<<"$runs" || true)"
failed="$(grep '^completed ' <<<"$runs" | grep -Ev '^completed (success|skipped|neutral) ' || true)"
if [ -n "$pending" ] || [ -n "$failed" ]; then
  echo "REFUSED: not every check on $short has passed." >&2
  [ -n "$pending" ] && printf 'still running:\n%s\n' "$pending" >&2
  [ -n "$failed" ] && printf 'failed:\n%s\n' "$failed" >&2
  exit 1
fi
echo "$(wc -l <<<"$runs" | tr -d ' ') checks passed on $short."

if git fetch --quiet origin course 2>/dev/null; then
  if [ "$(git rev-parse origin/course)" = "$target" ]; then
    echo "course is already at $short; nothing to do."
    exit 0
  fi
  echo "course: $(git rev-parse --short origin/course) -> $short"
  git log --oneline origin/course.."$target"
else
  echo "course does not exist yet; creating it at $short."
fi

git push origin "$target:refs/heads/course"
echo "Promoted. Students' next install gets $short."
