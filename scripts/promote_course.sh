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
#     student-path job is left out: it tests what `course` already holds, not this commit.
#     When the commit is a PR's merge with exactly the tree of that PR's head, the PR's
#     checks are used, so promoting does not wait for main to re-run them;
#   - the tests_and_build workflow run for those checks has completed and passed, so a check
#     list read before every job has started cannot pass for a finished one;
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
# wait loop, and runners were scarce enough that checks took an hour to start. Each poll
# prints a timestamped line saying what it is waiting on, or the error GitHub returned:
# on 2026-10-15 the wait ran 17 minutes after every check had passed, and said nothing.
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

# Whose checks count. A squash merge of a PR that was up to date with main has exactly the tree
# the PR's own checks passed on, so those checks are this commit's checks and waiting for main's
# run of the same jobs adds about five minutes and no information (PR #295, 2026-10-15). Any
# other commit, or a PR whose tree differs, is judged by its own checks on main.
checked="$target"
via=""
pr="$(gh api "repos/$REPO/commits/$target/pulls" --jq '.[0] | "\(.number) \(.head.sha)"' 2>/dev/null || true)"
if [ -n "$pr" ] && git fetch --quiet origin "refs/pull/${pr%% *}/head" 2>/dev/null \
    && [ "$(git rev-parse "${pr#* }^{tree}")" = "$(git rev-parse "$target^{tree}")" ]; then
  checked="${pr#* }"
  via=" (PR #${pr%% *}'s checks, on the same tree at $(git rev-parse --short "$checked"))"
fi

# Sets runs to one line per check run: status, conclusion, check-run id, details URL, name. The
# student-path job is left out: it tests what `course` already holds, not this commit. If
# GitHub cannot be reached, runs is empty and api_error says why.
#
# Plus one line for the tests_and_build workflow run itself, "queued" until it exists. Check
# runs appear only as their jobs start, so "every check run passed" can hold of the first job
# to finish alone: on 2026-10-16 this promoted 5af98f6 on "1 checks passed", with the tests not
# yet begun. The workflow run exists from the push and is complete only when every job is.
fetch_runs() {
  local out wf
  if out="$(gh api "repos/$REPO/commits/$checked/check-runs?per_page=100" \
      --jq '.check_runs[] | select(.name | startswith("Notebooks from") | not)
            | "\(.status) \(.conclusion // "-") \(.id) \(.details_url) \(.name)"' 2>&1)" \
      && wf="$(gh api "repos/$REPO/actions/workflows/tests_and_build.yml/runs?head_sha=$checked" \
      --jq '.workflow_runs[0] // empty | "\(.status) \(.conclusion // "-") - \(.html_url) workflow tests_and_build"' 2>&1)"; then
    runs="$out"$'\n'"${wf:-queued - - - workflow tests_and_build (not started)}"
    runs="${runs#$'\n'}"
    api_error=""
  else
    runs=""
    api_error="$(tail -n 1 <<<"${wf:-$out}")"
  fi
}

fetch_runs
if [ "$await" = 1 ]; then
  echo "Waiting for the checks on $short$via, every 30 seconds." >&2
  for _ in $(seq 1 180); do
    if [ -n "$runs" ] && ! grep -qv '^completed ' <<<"$runs"; then
      break
    fi
    if [ -n "$api_error" ]; then
      echo "$(date -u +%H:%M:%SZ)  GitHub call failed, trying again: $api_error" >&2
    elif [ -z "$runs" ]; then
      echo "$(date -u +%H:%M:%SZ)  no check runs yet" >&2
    else
      echo "$(date -u +%H:%M:%SZ)  $(grep -c '^completed ' <<<"$runs") of $(wc -l <<<"$runs" | tr -d ' ') checks done; waiting on: $(grep -v '^completed ' <<<"$runs" | cut -d' ' -f5- | paste -sd, -)" >&2
    fi
    sleep 30
    fetch_runs
  done
fi

if [ -z "$runs" ]; then
  echo "REFUSED: GitHub has no check runs for $short yet, or could not be reached.${api_error:+ $api_error}" >&2
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
    # A push to main cancels the run on the commit before it, so an older commit of main may
    # have no finished run. Its successor holds it, and is the one to promote.
    if grep -q '^completed cancelled .* workflow tests_and_build' <<<"$failed" &&
        [ "$target" != "$(git rev-parse origin/main)" ]; then
      echo "A later push to main superseded this run. Promote main's head instead:" >&2
      echo "  scripts/promote_course.sh --await" >&2
    fi
  fi
  exit 1
fi
echo "$(wc -l <<<"$runs" | tr -d ' ') checks passed on $short$via."

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
