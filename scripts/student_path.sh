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
# WHAT IT CANNOT DO. Google's Drive authorization dialog, a GPU, and Colab's own network. By
# default it provides only numpy, pandas and matplotlib of Colab's preinstalled packages; a
# notebook relying on any other must install it. A PASS here means "runs from a clean install
# of what the notebook's install cell names", not "runs in Colab". --colab narrows the gap.
#
# --colab builds the environment as Colab's GPU runtime has it: its Python minor version and
# the packages it ships that an install can collide with, at Colab's versions, read from
# googlecolab/backend-info (Google publishes the runtime's `pip freeze` there). Both COMET
# breaks of October 2026 needed this to reproduce: unbabel-comet's numpy<2 pin built numpy
# 1.26 from source on Python 3.13, and COMET would not load on Colab's transformers 5. Needs uv.
#
# --install-only runs only the install cells, then imports every module the notebook imports,
# then, for a few packages whose import proves little, a one-line smoke test (COMET loads a
# model and scores a pair: the transformers-5 break raised there, after a clean import). It
# needs no GPU and no student answers, so it covers the notebooks a full run must skip.
#
# Usage:
#   scripts/student_path.sh [--colab] [--install-only] NOTEBOOK...
#   scripts/student_path.sh docs/docs/course/lecture-07-toy-model.ipynb
#   scripts/student_path.sh --colab --install-only docs/docs/course/lecture-11-comet-on-a6.ipynb
#   STUDENT_DIR=/some/dir scripts/student_path.sh ...    # keep the environment and outputs there
#
# Each notebook's executed copy is written to $STUDENT_DIR/run-<stem>-<time>/out.ipynb.
# Nothing is ever deleted: every run gets new directories.
#
set -uo pipefail

REPO="$(cd "$(dirname "$0")/.." && pwd)"
STAMP="$(date +%Y%m%d-%H%M%S)"
STUDENT_DIR="${STUDENT_DIR:-${TMPDIR:-/tmp}/torchlingo-student-$STAMP}"

COLAB=0
INSTALL_ONLY=0
while [ $# -gt 0 ]; do
  case "$1" in
    --colab) COLAB=1; shift ;;
    --install-only) INSTALL_ONLY=1; shift ;;
    --*) echo "unknown option: $1" >&2; exit 2 ;;
    *) break ;;
  esac
done

if [ $# -eq 0 ]; then
  sed -n '2,45p' "$0" | sed 's/^# \{0,1\}//'
  exit 2
fi

mkdir -p "$STUDENT_DIR"
cd "$STUDENT_DIR" || exit 1

# What Colab's GPU runtime ships, of the packages a course install can collide with. Not all
# ~650: many are Linux- or CUDA-only, and these are the ones COMET, transformers and
# torchlingo's own requirements touch. The by-hand rebuild that found the transformers-5 break
# used this set.
COLAB_PACKAGES="numpy pandas scipy scikit-learn matplotlib torch transformers huggingface_hub
tokenizers safetensors protobuf sentencepiece setuptools tensorboard nltk"
BACKEND_INFO="https://raw.githubusercontent.com/googlecolab/backend-info/main"

if [ ! -x venv/bin/python ] && [ "$COLAB" = 1 ]; then
  command -v uv >/dev/null || { echo "--colab needs uv, to get Colab's Python" >&2; exit 2; }
  curl -sfL --retry 2 "$BACKEND_INFO/os-info-gpu.txt" -o colab-os-info.txt &&
    curl -sfL --retry 2 "$BACKEND_INFO/pip-freeze.gpu.txt" -o colab-pip-freeze.txt ||
    { echo "could not fetch Colab's package list from $BACKEND_INFO" >&2; exit 2; }
  # "Python 3.13.15" -> 3.13: uv may not have Colab's exact patch release, and a patch
  # release does not change which wheels exist.
  python_version="$(awk '/^Python / {split($2, v, "."); print v[1] "." v[2]}' colab-os-info.txt)"
  # Pins, from the freeze: `torch==2.11.0+cu130` loses its CUDA tag (there is no GPU here, and
  # the CPU build has the same API), and `torch @ https://.../torch-2.11.0%2Bcpu-...whl` is read
  # for its version.
  for name in $COLAB_PACKAGES; do
    grep -iE "^${name//_/[-_]}( @ |==)" colab-pip-freeze.txt | head -1
  done | sed -E 's/^([^ =]+) @ .*\/[^/-]+-([0-9][^%-]*)(%2B|-).*/\1==\2/; s/\+[a-z0-9.]+$//' \
    > colab-pins.txt
  # When Google last published the list, so a failure says which Colab image it was tested
  # against. Best effort: an unauthenticated API call can be rate-limited, and the run does
  # not need it.
  freeze_date="$(curl -sf "https://api.github.com/repos/googlecolab/backend-info/commits?path=pip-freeze.gpu.txt&per_page=1" |
    grep -m1 -o '"date": *"[^"]*"' | cut -d'"' -f4)"
  echo "Colab's runtime (backend-info, published ${freeze_date:-on an unknown date}):" \
       "Python $python_version, $(wc -l < colab-pins.txt | tr -d ' ') pins:" \
       "$(grep -E '^(numpy|torch|transformers)==' colab-pins.txt | tr '\n' ' ')"
  uv venv --quiet --seed -p "$python_version" venv || exit 1
  # The runner, then Colab's pins, with pip: what the notebooks' own installs use, and it reads
  # PIP_EXTRA_INDEX_URL, which CI uses for the CPU build of torch.
  venv/bin/python -m pip install --quiet nbclient nbformat ipykernel -r colab-pins.txt || exit 1
elif [ ! -x venv/bin/python ]; then
  python3 -m venv venv
  # The runner, and the few packages every Colab runtime already has. Not torchlingo: the
  # notebooks install that themselves, and that is what is under test. Without the Colab
  # baseline, Lecture 5 failed on `import numpy`, which it never installs because Colab has it.
  # torchlingo depends on all three, so this cannot hide a notebook that forgot to install it.
  venv/bin/pip install --quiet nbclient nbformat ipykernel numpy pandas matplotlib
fi
export INSTALL_ONLY
if venv/bin/python -c "import torchlingo" 2>/dev/null; then
  echo "NOTE: torchlingo is already in $STUDENT_DIR/venv, so a first install is not exercised;"
  echo "      the install cell's forced reinstall still replaces it with the course branch."
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

# A smoke test for packages whose import proves little: COMET imported cleanly on
# transformers 5 and raised only when loading a model.
SMOKE = {
    "comet": (
        "from comet import download_model, load_from_checkpoint\n"
        "model = load_from_checkpoint(download_model('Unbabel/wmt22-comet-da'))\n"
        "out = model.predict([{'src': 'Pode ser entregue?', 'mt': 'Can it be delivered?',\n"
        "                      'ref': 'Can it be delivered?'}],\n"
        "                    batch_size=1, gpus=0, num_workers=2, progress_bar=False)\n"
        "print(f'COMET scored one pair: {out.system_score:.3f}')"
    ),
}

if os.environ.get("INSTALL_ONLY") == "1":
    import ast

    def python_lines(source):
        # IPython magics and shell escapes are not Python. Each becomes a `pass` at its own
        # indent, so the rest parses even when a magic was all of an `if` block's body.
        return "\n".join(
            line[: len(line) - len(line.lstrip())] + "pass"
            if line.lstrip().startswith(("%", "!")) else line
            for line in source.splitlines())

    modules = set()
    for cell in nb.cells[1:]:
        if cell.cell_type != "code":
            continue
        try:
            tree = ast.parse(python_lines(cell.source))
        except SyntaxError:  # a `blanks` cell; its imports are elsewhere or not yet written
            continue
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                modules.update(alias.name.split(".")[0] for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                modules.add(node.module.split(".")[0])
    modules.discard("google")   # faked above

    installs = [cell for cell in nb.cells[1:]
                if cell.cell_type == "code" and "pip install" in cell.source]
    if not installs:
        print(f"{os.path.basename(run)}: SKIP  (no install cell)", flush=True)
        sys.exit(0)
    for cell in installs:
        # An install cell may go on to use what it installed on the student's files, as
        # Lecture 13's `!comet-score` does; only the installing is under test here.
        cell.source = "\n".join(line for line in cell.source.splitlines()
                                if not line.lstrip().startswith("!") or "pip" in line)
    checks = [nbformat.v4.new_code_cell(
        "import importlib\n"
        "failed = []\n"
        f"for name in {sorted(modules)!r}:\n"
        "    try:\n"
        "        importlib.import_module(name)\n"
        "    except Exception as exc:\n"
        "        failed.append(f'{name}: {type(exc).__name__}: {exc}')\n"
        "assert not failed, 'imports failed after the install cells:\\n' + '\\n'.join(failed)\n"
        f"print('imported', {len(modules)}, 'modules')"
    )]
    if os.path.exists("colab-pins.txt"):
        # What the install replaced of Colab's own packages. Replacing numpy or torch is a
        # failure even when every import still works here: Colab's compiled packages were built
        # against its numpy, and on 2026-10-15 unbabel-comet's numpy<2 downgrade broke them in
        # Colab while PyPI's wheels on a laptop carried on importing.
        pins = dict(line.strip().split("==") for line in open("colab-pins.txt") if "==" in line)
        checks.append(nbformat.v4.new_code_cell(
            "from importlib.metadata import version\n"
            f"pins = {pins!r}\n"
            "changed = {name: (want, version(name).split('+')[0]) for name, want in pins.items()\n"
            "           if version(name).split('+')[0] != want}\n"
            "for name, (want, got) in sorted(changed.items()):\n"
            "    print(f'the install replaced Colab\\'s {name} {want} with {got}')\n"
            "fatal = sorted(set(changed) & {'numpy', 'torch'})\n"
            "assert not fatal, f'the install replaced Colab\\'s {fatal}, which its other packages are built against'\n"
            "print(f'{len(pins) - len(changed)} of {len(pins)} of Colab\\'s packages left as Colab ships them')"
        ))
    checks += [nbformat.v4.new_code_cell(SMOKE[name]) for name in sorted(modules & set(SMOKE))]
    nb.cells = nb.cells[:1] + installs + checks

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
