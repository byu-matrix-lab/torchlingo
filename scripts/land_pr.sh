#!/usr/bin/env bash
#
# Land a pull request: check it against main, merge it, confirm nothing else closed, clean up
# its branch and worktree, and optionally promote it to `course` and wait for the student path.
#
# WHY THIS EXISTS. PRs #295 to #302 each landed by the same eight steps, typed by hand every
# time, and CLAUDE.md ("Pull requests") gives the reason for each one:
#
#   1. scripts/merge_check.sh PR      a green PR is green against the base it last saw
#   2. record the open-PR set         PRs #26 and #84 closed themselves on another's merge
#   3. gh pr merge, no --delete-branch   deleting the branch on merge is how #11 and #12 died
#   4. compare the open-PR set        ... which is how #84 was caught and reopened at once
#   5. remove the PR's worktree       only if it holds nothing uncommitted
#   6. delete the branch, local and remote, as a separate step
#   7. scripts/promote_course.sh --await   with --promote
#   8. wait for the student-path run on the promoted commit   with --promote
#
# Typed from memory they got skipped or reordered, and step 8 was the one nobody did: the
# student-path run failed on every push to `course` from 2026-10-14 until PR #299. Task #245.
#
# Usage:
#   scripts/land_pr.sh PR                     steps 1-6
#   scripts/land_pr.sh PR --promote           ... then 7 and 8
#   scripts/land_pr.sh PR --promote --no-wait ... then 7 only; prints how to check step 8
#   scripts/land_pr.sh PR --admin [--promote] merge with --admin, past required reviews
#   scripts/land_pr.sh PR PR ... [flags]      several, in the order given; --promote applies
#                                             to the last only
#
# --admin is passed only on Eric's explicit OK for that PR, or for a PR that touches only
# notes/ (CLAUDE.md memory: notes-only PRs may self-merge). This script never decides that.
# Given several PRs, --admin applies to every one, so the OK must cover them all.
#
# SEVERAL PRS. Each is landed as it would be alone, steps 1-6, so each merge_check runs against
# the main the previous merge left (CLAUDE.md: "When several PRs merge in a row"). Only the last
# is promoted, with one student-path run for the batch (Eric, 2026-10-19). The first failure
# stops the batch, and the PRs after it are left open. The stop names every PR the batch already
# merged that `course` does not have, with the command to promote them, since a stop before the
# last PR leaves them on main where no student gets them.
#
# --no-wait, because by step 8 the promotion has happened: the wait verifies, it does not
# gate. On 2026-10-22 it was ten of the batch's thirty minutes.
#
# Stops at the first step that fails and says which; every later step is left undone, so
# nothing is cleaned up after a merge that went wrong. Prints a timestamped line per step, and
# the full output of steps 1, 7 and 8 goes to a log, named at the start. Run it from any
# checkout or worktree of the repository: it acts on the main checkout, so it can remove the
# worktree it was started from.
set -uo pipefail

REPO="byu-matrix-lab/torchlingo"

usage() {
  echo "usage: scripts/land_pr.sh PR [PR ...] [--admin] [--promote [--no-wait]]   (see the header for more)" >&2
  exit 2
}

PRS=()
ADMIN=""   # a flag, not an array: an empty array is "unbound" under set -u in macOS's bash 3.2
PROMOTE=0
WAIT=1
for arg in "$@"; do
  case "$arg" in
    --admin) ADMIN="--admin" ;;
    --promote) PROMOTE=1 ;;
    --no-wait) WAIT=0 ;;
    -h | --help) usage ;;
    *[!0-9]*) echo "not a PR number: $arg" >&2; usage ;;
    *) PRS+=("$arg") ;;
  esac
done
[[ ${#PRS[@]} -gt 0 ]] || usage
[[ $WAIT -eq 1 || $PROMOTE -eq 1 ]] || { echo "--no-wait goes with --promote" >&2; usage; }

# The main checkout, wherever this was started: a worktree's common git dir is the main one's.
MAIN="$(cd "$(dirname "$(git rev-parse --path-format=absolute --git-common-dir)")" && pwd)" || exit 2
cd "$MAIN" || exit 2

# What a batch stopped at PR $1 left behind. A PR landed before the stop, or by it, is on main
# but not on course, since only the last is promoted, and its fix may be one students need.
# So name each one, and the command that promotes them.
batch_stopped() {
  local n info merged="" unpromoted=""
  echo "STOPPED: the batch stopped at PR #$1; the PRs after it are still open" >&2
  git fetch --quiet origin course 2>/dev/null
  for n in "${PRS[@]}"; do
    info="$(gh pr view "$n" --repo "$REPO" --json state,mergeCommit \
      --jq '"\(.state) \(.mergeCommit.oid // "-")"' 2>&1)" ||
      { echo "  PR #$n: could not tell whether it merged: $info" >&2; continue; }
    if [[ "${info%% *}" == MERGED ]]; then
      merged="${merged:+$merged,} PR #$n"
      git merge-base --is-ancestor "${info#* }" origin/course 2>/dev/null ||
        unpromoted="${unpromoted:+$unpromoted,} PR #$n"
    fi
    [[ "$n" == "$1" ]] && break
  done
  [[ -n "$merged" ]] && echo "merged to main:$merged" >&2
  if [[ -n "$unpromoted" ]]; then
    echo "not on course, so students do not have them yet:$unpromoted" >&2
    echo "  promote them with:  scripts/promote_course.sh --await" >&2
  fi
}

# Several PRs: land each by running this script on it alone, promoting with the last. The main
# checkout's copy, since landing the first may remove the worktree this was started from.
if [[ ${#PRS[@]} -gt 1 ]]; then
  last="${PRS[${#PRS[@]}-1]}"
  for n in "${PRS[@]}"; do
    flags=()
    [[ -n "$ADMIN" ]] && flags+=("$ADMIN")
    if [[ "$n" == "$last" && $PROMOTE -eq 1 ]]; then
      flags+=("--promote")
      [[ $WAIT -eq 0 ]] && flags+=("--no-wait")
    fi
    "$MAIN/scripts/land_pr.sh" "$n" ${flags[@]+"${flags[@]}"} || {
      batch_stopped "$n"
      exit 1
    }
  done
  exit 0
fi
PR="${PRS[0]}"
LOG="${TMPDIR:-/tmp}/land-pr-$PR-$(date +%Y%m%d-%H%M%S).log"

step() { echo "$(date -u +%H:%M:%SZ)  $*"; }
stop() { echo "STOPPED: $*" >&2; [[ -s "$LOG" ]] && echo "log: $LOG" >&2; exit 1; }
# Sorted as text, not numerically: comm needs that order.
open_prs() { gh pr list --repo "$REPO" --state open --limit 200 --json number --jq '.[].number' | sort; }

step "landing PR #$PR; log: $LOG"
info="$(gh pr view "$PR" --repo "$REPO" \
  --json state,baseRefName,headRefName,headRefOid,isCrossRepository,title \
  --jq '"\(.state) \(.baseRefName) \(.headRefName) \(.headRefOid) \(.isCrossRepository)"')" ||
  stop "could not read PR #$PR from GitHub"
read -r state base branch head_sha cross <<<"$info"
[[ "$state" == OPEN ]] || stop "PR #$PR is $state, not open"
[[ "$base" == main ]] || stop "PR #$PR targets $base; every PR targets main (CLAUDE.md)"

step "1/8 merge_check.sh $PR"
"$MAIN/scripts/merge_check.sh" "$PR" >>"$LOG" 2>&1
verdict=$?
[[ $verdict -eq 0 ]] || stop "merge_check.sh ended $(tail -n 1 "$LOG"), not CLEAN"

step "2/8 recording the open PRs"
before="$(open_prs)" || stop "could not list the open PRs"

step "3/8 merging PR #$PR${ADMIN:+ (--admin)}"
gh pr merge "$PR" --repo "$REPO" --squash $ADMIN || stop "gh pr merge failed"
merged="$(gh pr view "$PR" --repo "$REPO" --json state,mergeCommit --jq '"\(.state) \(.mergeCommit.oid)"')"
[[ "${merged%% *}" == MERGED ]] || stop "PR #$PR is ${merged%% *} after the merge"
sha="${merged#* }"
step "    merged as ${sha:0:7}"

step "4/8 comparing the open PRs"
after="$(open_prs)" || stop "could not list the open PRs after the merge"
gone="$(comm -23 <(grep -vx "$PR" <<<"$before") <(echo "$after"))"
if [[ -n "$gone" ]]; then
  for n in $gone; do echo "PR #$n closed during this merge: gh pr reopen $n --repo $REPO" >&2; done
  stop "other PRs closed; reopen them before deleting anything"
fi
step "    open PRs: ${after//$'\n'/ }${after:+ }(as before, less PR #$PR)"

git fetch --quiet origin
if [[ "$cross" == true ]]; then
  step "5-6/8 PR #$PR is from a fork; no branch or worktree here to clean up"
else
  step "5/8 the worktree on $branch"
  worktree="$(git worktree list --porcelain |
    awk -v b="refs/heads/$branch" '/^worktree /{w=substr($0,10)} $0=="branch " b {print w}')"
  keep=""
  if [[ -z "$worktree" ]]; then
    step "    none"
  elif [[ "$worktree" == "$MAIN" ]]; then
    keep="it is checked out in the main checkout"
  elif [[ -n "$(git -C "$worktree" status --porcelain)" ]]; then
    keep="$worktree has uncommitted changes"
  else
    git worktree remove "$worktree" || stop "could not remove $worktree"
    step "    removed $worktree"
  fi

  step "6/8 the branch $branch"
  if [[ -n "$keep" ]]; then
    step "    kept, because $keep"
  elif git show-ref --verify --quiet "refs/heads/$branch" &&
       [[ "$(git rev-parse "refs/heads/$branch")" != "$head_sha" ]]; then
    # A local commit the PR never had would be lost; it was not reviewed or merged.
    step "    local $branch kept: it is not at the PR's head ${head_sha:0:7}"
  else
    git show-ref --verify --quiet "refs/heads/$branch" && git branch -D "$branch" >/dev/null &&
      step "    deleted the local branch"
    if [[ -n "$(gh pr list --repo "$REPO" --state open --base "$branch" --json number --jq '.[].number')" ]]; then
      step "    remote $branch kept: an open PR targets it"
    elif git push --quiet origin --delete "$branch" 2>>"$LOG"; then
      step "    deleted origin/$branch"
    else
      step "    origin/$branch was not deleted (already gone?); see the log"
    fi
  fi
fi

if [[ "$(git branch --show-current)" == main ]]; then
  git merge --quiet --ff-only origin/main 2>>"$LOG" && step "    main checkout fast-forwarded to $(git rev-parse --short HEAD)"
fi

if [[ $PROMOTE -eq 0 ]]; then
  step "done: PR #$PR landed as ${sha:0:7}; not promoted (--promote to do so)"
  exit 0
fi

step "7/8 promote_course.sh --await ${sha:0:7}"
"$MAIN/scripts/promote_course.sh" --await "$sha" >>"$LOG" 2>&1 ||
  stop "promote_course.sh refused: $(tail -n 1 "$LOG")"
step "    $(tail -n 1 "$LOG")"

if [[ $WAIT -eq 0 ]]; then
  step "done: PR #$PR landed as ${sha:0:7} and promoted; the student-path run was not waited for."
  echo "Check it with:  gh run list --repo $REPO --workflow student_path.yml --branch course --commit $sha"
  exit 0
fi

# The push to course starts this run. Polled every 30 seconds for up to 60 minutes; a failed
# call is "not yet", except an authentication failure, which stops at once.
step "8/8 waiting for the student-path run on ${sha:0:7}"
for _ in $(seq 1 120); do
  if out="$(gh run list --repo "$REPO" --workflow student_path.yml --branch course --limit 10 \
      --json headSha,status,conclusion,url \
      --jq ".[] | select(.headSha == \"$sha\") | \"\(.status) \(.conclusion) \(.url)\"" 2>&1)"; then
    read -r run_status conclusion url <<<"$(head -n 1 <<<"$out")"
    if [[ "${run_status:-}" == completed ]]; then
      [[ "$conclusion" == success ]] || stop "the student-path run on ${sha:0:7} ended $conclusion: $url"
      step "done: PR #$PR landed as ${sha:0:7}, promoted, and the student path passed: $url"
      exit 0
    fi
    echo "$(date -u +%H:%M:%SZ)  student path: ${run_status:-not started}" >>"$LOG"
  elif grep -qiE "401|bad credentials|authentication|gh auth login" <<<"$out"; then
    stop "GitHub refused the credentials: $out"
  else
    echo "$(date -u +%H:%M:%SZ)  GitHub call failed, trying again: $out" >>"$LOG"
  fi
  sleep 30
done
stop "the student-path run on ${sha:0:7} did not finish within 60 minutes"
