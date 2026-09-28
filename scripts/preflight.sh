#!/usr/bin/env bash
#
# Run exactly what CI runs, locally, fastest gate first.
#
# WHY THIS EXISTS. CI gates seven separate things across four jobs. Running them from
# memory means running most of them, and the one that gets forgotten is the one that fails:
# on 2026-09-28 a pull request burned a full CI cycle on `render_report.py --check`, which
# takes 0.4 seconds here, because a generated report had been hand-edited. A round trip
# through GitHub costs four to six minutes; this costs about thirty-five seconds.
#
# ORDERING IS THE POINT. Gates are cheapest first, and the script stops at the first
# failure, so a formatting slip is reported in two seconds instead of after the test suite.
#
# Usage:
#   scripts/preflight.sh            everything except notebook execution
#   scripts/preflight.sh --all      also execute the runnable notebooks (slow: minutes)
#   scripts/preflight.sh --quick    the four static gates only (~3 seconds)
#
set -uo pipefail

cd "$(dirname "$0")/.."

if [[ -d .venv && -z "${VIRTUAL_ENV:-}" ]]; then
  # shellcheck disable=SC1091
  source .venv/bin/activate
fi

MODE="${1:-}"
FAILED=0
STARTED=$(date +%s)

run() {
  local label="$1"
  shift
  printf '  %-34s' "$label"
  local began output status
  began=$(date +%s)
  output=$("$@" 2>&1)
  status=$?
  local took=$(( $(date +%s) - began ))
  if [[ $status -eq 0 ]]; then
    printf 'ok    %3ds\n' "$took"
    return 0
  fi
  printf 'FAIL  %3ds\n\n' "$took"
  echo "$output" | tail -40
  echo
  FAILED=1
  return 1
}

echo "preflight: the gates CI applies, cheapest first"
echo

# --- static gates, about three seconds in total ------------------------------------------
run "ruff lint" ruff check src tests || exit 1
run "ruff format" ruff format --check src tests || exit 1
run "reports match their JSON" python scripts/render_report.py --check || exit 1
run "notebook metadata + purpose" python scripts/notebook_meta.py --check || exit 1

if [[ "$MODE" == "--quick" ]]; then
  echo
  echo "quick mode: static gates only, $(( $(date +%s) - STARTED ))s"
  exit 0
fi

# --- the test suite, both runners, plus doctests -----------------------------------------
# CI runs pytest AND `unittest discover`, because a test that only one runner collects has
# slipped through before. Doctests are a separate gate again: a module can pass its tests and
# still ship an example that raises.
run "test suite (pytest)" python -m pytest -q --no-header || true
run "test suite (unittest)" python -m unittest discover tests || true
run "docstring examples" python -m pytest -q --no-header --doctest-modules src/torchlingo || true

# --- the docs build, which is strict and catches nav and link errors ----------------------
run "docs build (strict)" mkdocs build --strict --config-file docs/mkdocs.yml -d /tmp/preflight-site || true

# --- notebook execution, minutes rather than seconds, so opt in --------------------------
if [[ "$MODE" == "--all" ]]; then
  run "execute notebooks" python scripts/execute_notebooks.py || true
else
  printf '  %-34s%s\n' "execute notebooks" "skipped  (--all to include)"
fi

echo
TOTAL=$(( $(date +%s) - STARTED ))
if [[ $FAILED -eq 0 ]]; then
  echo "preflight passed in ${TOTAL}s. Safe to push."
  exit 0
fi
echo "preflight FAILED in ${TOTAL}s. Fix the above before pushing."
exit 1
