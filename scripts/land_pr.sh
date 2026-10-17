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
#   scripts/land_pr.sh PR --admin [--promote] merge with --admin, past required reviews
#
# --admin is passed only on Eric's explicit OK for that PR, or for a PR that touches only
# notes/ (CLAUDE.md memory: notes-only PRs may self-merge). This script never decides that.
#
# Stops at the first step that fails and says which; every later step is left undone, so
# nothing is cleaned up after a merge that went wrong. Prints a timestamped line per step, and
# the full output of steps 1, 7 and 8 goes to a log, named at the start. Run it from any
# checkout or worktree of the repository: it acts on the main checkout, so it can remove the
# worktree it was started from.
set -uo pipefail

REPO="byu-matrix-lab/torchlingo"

usage() {
  echo "usage: scripts/land_pr.sh PR [--admin] [--promote]   (see the header for more)" >&2
  exit 2
}

PR=""
ADMIN=""   # a flag, not an array: an empty array is "unbound" under set -u in macOS's bash 3.2
PROMOTE=0
for arg in "$@"; do
  case "$arg" in
    --admin) ADMIN="--admin" ;;
    --promote) PROMOTE=1 ;;
    -h | --help) usage ;;
    *[!0-9]*) echo "not a PR number: $arg" >&2; usage ;;
    *) PR="$arg" ;;
  esac
done
[[ -n "$PR" ]] || usage

# The main checkout, wherever this was started: a worktree's common git dir is the main one's.
MAIN="$(cd "$(dirname "$(git rev-parse --path-format=absolute --git-common-dir)")" && pwd)" || exit 2
cd "$MAIN" || exit 2
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
