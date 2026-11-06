#!/usr/bin/env bash
#
# Remember which exact trees passed a check, so the same code is not checked twice.
#
# Sourced by preflight.sh and merge_check.sh; it does nothing when run. A pass is recorded by
# git tree SHA, which names the contents of every tracked file, so a PR's merge with main that
# changes nothing beyond what preflight already saw is the same tree, and a one-character
# change anywhere is a different one. The record is a file in the shared git directory, so every
# worktree sees it, and it is never pushed.
#
# WHY THIS EXISTS. On Nov 5 GitHub errors at the merge step cost PR #380 two more full merge
# checks, against an unmoved main, and each landing repeated a preflight run minutes earlier.
#
# WHY PASSES EXPIRE. The tree is not all a check depends on: the install check reads Colab's
# published versions, fresh each run, and the tests run on this machine's packages. A pass older
# than TREE_PASS_MINUTES counts for nothing, so the check runs again. `merge_check.sh --fresh`
# ignores passes altogether.

TREE_PASS_MINUTES="${TREE_PASS_MINUTES:-360}"

_tree_pass_file() {
  local dir
  dir="$(git rev-parse --path-format=absolute --git-common-dir)/torchlingo-passes" || return 1
  mkdir -p "$dir" && printf '%s/%s-%s' "$dir" "$2" "$1"
}

# The tree of HEAD, the commit a pass is about.
tree_of_head() {
  git rev-parse 'HEAD^{tree}'
}

# record_tree_pass TREE KIND: TREE passed the check called KIND, now.
record_tree_pass() {
  local file
  file="$(_tree_pass_file "$1" "$2")" && : > "$file"
}

# tree_passed_recently TREE KIND: true if TREE passed KIND within TREE_PASS_MINUTES.
tree_passed_recently() {
  local file
  file="$(_tree_pass_file "$1" "$2")" || return 1
  [[ -e "$file" && -n "$(find "$file" -mmin "-$TREE_PASS_MINUTES" 2>/dev/null)" ]]
}
