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
#   scripts/promote_course.sh                    # promote origin/main's head
#   scripts/promote_course.sh <commit>           # promote an earlier commit of main
#   scripts/promote_course.sh --await [<commit>] # wait for its checks first, then promote
#
# --await polls every 30 seconds for up to 90 minutes, and treats a failed call to GitHub as
# "not yet" rather than a reason to stop: on 2026-10-05 a network blip killed a hand-written
# wait loop, and runners were scarce enough that checks took an hour to start.
#
# A job GitHub cancels because no runner ever picked it up is reported as that, with the
# command that re-runs it; it is not a failure of the code, but it is not a pass either, so the
# promotion is still refused.
#
# Each push to `course` starts the student-path workflow, which runs every notebook from its
# own install cell: watch it with `gh run list --workflow student_path.yml`.

set -euo pipefail

REPO="byu-matrix-lab/torchlingo"
cd "$(dirname "$0")/.."

await=0
if [ "${1:-}" = "--await" ]; then
  await=1
  shift
fi

git fetch --quiet origin main
target="$(git rev-parse "${1:-origin/main}^{commit}")"
short="$(git rev-parse --short "$target")"

if ! git merge-base --is-ancestor "$target" origin/main; then
  echo "REFUSED: $short is not on origin/main." >&2
  exit 1
fi

# One line per check run: status, conclusion, check-run id, details URL, name. The student-path
# job is left out: it tests what `course` already holds, not this commit. Prints nothing if
# GitHub cannot be reached.
check_runs() {
  gh api "repos/$REPO/commits/$target/check-runs?per_page=100" \
    --jq '.check_runs[] | select(.name | startswith("Notebooks from") | not)
          | "\(.status) \(.conclusion // "-") \(.id) \(.details_url) \(.name)"' 2>/dev/null || true
}

runs="$(check_runs)"
if [ "$await" = 1 ]; then
  for _ in $(seq 1 180); do
    if [ -n "$runs" ] && ! grep -qv '^completed ' <<<"$runs"; then
      break
    fi
    sleep 30
    runs="$(check_runs)"
  done
fi

if [ -z "$runs" ]; then
  echo "REFUSED: GitHub has no check runs for $short yet, or could not be reached." >&2
  exit 1
fi

pending="$(grep -v '^completed ' <<<"$runs" || true)"
failed="$(grep '^completed ' <<<"$runs" | grep -Ev '^completed (success|skipped|neutral) ' || true)"
if [ -n "$pending" ] || [ -n "$failed" ]; then
  echo "REFUSED: not every check on $short has passed." >&2
  [ -n "$pending" ] && printf 'still running:\n%s\n' "$(cut -d' ' -f1,5- <<<"$pending")" >&2
  if [ -n "$failed" ]; then
    echo "failed:" >&2
    stranded_runs=""
    while read -r status conclusion id url name; do
      reason=""
      if [ "$conclusion" = "cancelled" ] && gh api "repos/$REPO/check-runs/$id/annotations" \
          --jq '.[].message' 2>/dev/null | grep -q "not acquired by Runner"; then
        reason="  (GitHub never gave it a runner; the code was not run)"
        stranded_runs="$stranded_runs $(sed -E 's#.*/actions/runs/([0-9]+)/.*#\1#' <<<"$url")"
      fi
      echo "$status $conclusion $name$reason" >&2
    done <<<"$failed"
    for run in $(tr ' ' '\n' <<<"$stranded_runs" | sort -u); do
      echo "Re-run the stranded jobs with:  gh run rerun $run --failed" >&2
    done
  fi
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
