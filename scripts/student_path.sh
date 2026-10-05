#!/usr/bin/env bash
#
# Run notebooks the way a Colab student does, as far as a laptop can.
#
# WHY THIS EXISTS. scripts/execute_notebooks.py runs notebooks against the repository's own
# editable install, with data/ linked in from the checkout. A student has neither: they open
# a Colab badge, get a runtime with no torchlingo, and depend on the notebook's own install
# cell (from the repository's `course` branch since 2026-10-05; from PyPI before that) and on
# whatever it downloads. On 2026-09-28 this harness found two
# failures CI could not see: tutorials 4 and 7 read data/ that a pip install does not ship,
# and tutorial 8 loads a checkpoint that tutorial 3 saved in a different Colab runtime.
#
# WHAT IT DOES
#   - a FRESH virtual environment per run, holding only the notebook runner. A reused one
#     already had torchlingo installed, so the install cell under test did nothing at all --
#     which is how a run that "passed" once proved less than it appeared to;
#   - a fake `google.colab` module, inserted before the first cell, so every Colab-only
#     branch executes: the install, the downloads, `drive.mount`;
#   - `/content/drive` rewritten to a scratch directory, since it cannot be created off Colab.
#
# Notebooks named in ONE invocation share the environment, so only the first exercises its
# install cell; the rest find torchlingo already there. To test a notebook's install, give it
# a run of its own.
#
# WHAT IT CANNOT DO. Google's Drive authorization dialog, a GPU, and Colab's own network. Of
# Colab's preinstalled packages it provides only numpy, pandas and matplotlib; a notebook
# relying on any other must install it. A PASS here means "runs from a clean install of what
# the notebook's install cell names", not "runs in Colab".
#
# Usage:
#   scripts/student_path.sh NOTEBOOK...
#   scripts/student_path.sh docs/docs/course/lecture-07-toy-model.ipynb
#   STUDENT_DIR=/some/dir scripts/student_path.sh ...    # keep the environment and outputs there
#
# Each notebook's executed copy is written to $STUDENT_DIR/run-<stem>-<time>/out.ipynb.
# Nothing is ever deleted: every run gets new directories.
#
set -uo pipefail

REPO="$(cd "$(dirname "$0")/.." && pwd)"
STAMP="$(date +%Y%m%d-%H%M%S)"
STUDENT_DIR="${STUDENT_DIR:-${TMPDIR:-/tmp}/torchlingo-student-$STAMP}"

if [ $# -eq 0 ]; then
  sed -n '2,30p' "$0" | sed 's/^# \{0,1\}//'
  exit 2
fi

mkdir -p "$STUDENT_DIR"
cd "$STUDENT_DIR" || exit 1

if [ ! -x venv/bin/python ]; then
  python3 -m venv venv
  # The runner, and the few packages every Colab runtime already has. Not torchlingo: the
  # notebooks install that themselves, and that is what is under test. Without the Colab
  # baseline, Lecture 5 failed on `import numpy`, which it never installs because Colab has it.
  # torchlingo depends on all three, so this cannot hide a notebook that forgot to install it.
  venv/bin/pip install --quiet nbclient nbformat ipykernel numpy pandas matplotlib
fi
if venv/bin/python -c "import torchlingo" 2>/dev/null; then
  echo "NOTE: torchlingo is already in $STUDENT_DIR/venv, so install cells will not be exercised."
fi

# The kernel inherits this, so a notebook's `!pip install` finds the venv's pip, as it finds
# the runtime's only pip in Colab. Without it `!pip` reached the system pip and installed
# outside the venv, and Lectures 3, 4, 5 and 12 failed at their next import although they run
# in Colab. `%pip` targets the kernel's interpreter either way.
export VIRTUAL_ENV="$STUDENT_DIR/venv"
export PATH="$VIRTUAL_ENV/bin:$PATH"

status=0
for path in "$@"; do
  stem="$(basename "$path" .ipynb)"
  run="run-$stem-$STAMP"
  mkdir "$run"
  cp "$REPO/$path" "$run/in.ipynb" 2>/dev/null || cp "$path" "$run/in.ipynb" || exit 1

  venv/bin/python - "$run" <<'EOF' || status=1
import os
import sys
import time

import nbformat
from nbclient import NotebookClient

run = sys.argv[1]
nb = nbformat.read(f"{run}/in.ipynb", as_version=4)

# Pretend to be Colab before any cell runs, including a Drive that "mounts" successfully.
# The fake must survive a real `import google.colab` (torchlingo.colab.is_colab does one), so
# it hangs off a `google` package, and mounting creates MyDrive as the real one does.
# /content cannot be created off Colab, so the mount point is a scratch directory: rewritten in
# notebook source that names it, and given to torchlingo.colab.setup through its variable.
fake_drive = os.path.abspath(f"{run}/fake-content-drive")
nb.cells.insert(0, nbformat.v4.new_code_cell(
    "import os, sys, types\n"
    "from pathlib import Path\n"
    f"os.environ['TORCHLINGO_DRIVE_MOUNT'] = {fake_drive!r}\n"
    "def _mount(path, **kw):\n"
    "    (Path(path) / 'MyDrive').mkdir(parents=True, exist_ok=True)\n"
    "    print(f'(fake) mounted {path}')\n"
    "colab = types.ModuleType('google.colab')\n"
    "colab.drive = types.SimpleNamespace(mount=_mount)\n"
    "google = sys.modules.get('google') or types.ModuleType('google')\n"
    "google.colab = colab\n"
    "sys.modules['google'] = google\n"
    "sys.modules['google.colab'] = colab"
))
for cell in nb.cells:
    cell.source = cell.source.replace("/content/drive", fake_drive)

start = time.time()
try:
    NotebookClient(nb, timeout=900, kernel_name="python3",
                   resources={"metadata": {"path": run}}).execute()
    result = "PASS"
except Exception as exc:  # report and carry on to the next notebook
    result = f"FAIL: {type(exc).__name__}: {str(exc)[-600:]}"
nbformat.write(nb, f"{run}/out.ipynb")
print(f"{os.path.basename(run)}: {result}  ({time.time() - start:.0f}s)", flush=True)
sys.exit(0 if result == "PASS" else 1)
EOF
done

echo "torchlingo installed by the notebooks: $(venv/bin/pip show torchlingo 2>/dev/null | awk '/^Version/ {print $2}')"
echo "outputs: $STUDENT_DIR"
exit $status
