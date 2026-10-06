# Notebook Setup

One call that prepares a notebook: the device, Google Drive, and the data files a
`pip install` does not ship.

## Why it exists

Every notebook needs the same plumbing before it can teach anything: is this Colab, is
there a GPU, is Drive mounted so work survives a disconnect, and are the data files here.
Written out, that is twenty to sixty lines at the top of every notebook, and a student reading
them learns nothing about translation. `setup` does it and prints what it did.

It behaves the same in Colab and in a local clone. Outside Colab, Drive is the current
directory and files already on disk are left alone, so a notebook needs no branch of its own.

## Quick Start

The install stays a cell of its own, because nothing in the library can run before the
library is installed:

```python
# Install TorchLingo in Colab. Nothing to uncomment. Pip's "you may need to restart the
# kernel" does not apply to a first install; if the install fails, the import below stops here.
import sys
if "google.colab" in sys.modules:
    %pip install --quiet "torchlingo @ git+https://github.com/byu-matrix-lab/torchlingo@course"
    # pip keeps a TorchLingo already in this session when its version number matches, even if
    # it is older code; this swaps in the course branch's, leaving the dependencies alone.
    %pip install --quiet --force-reinstall --no-deps "torchlingo @ git+https://github.com/byu-matrix-lab/torchlingo@course"
import torchlingo
```

The second `%pip` line matters more than it looks. The `course` branch and the PyPI release
can carry the same version number, and pip treats an installed TorchLingo of that number as
already satisfying the first line, so a session that installed TorchLingo earlier (another
notebook, or before `course` moved) would keep its older code. Reinstalling TorchLingo alone,
without its dependencies, takes a few seconds and leaves torch untouched.

The last line is the point: `%pip` reports a failure but does not stop the cell, so without
the import a failed install would surface one cell later as a confusing `ModuleNotFoundError`.

The course notebooks install from the repository's `course` branch rather than from PyPI, so
a notebook and the library code it calls reach students together, without waiting for a
release. `course` only ever moves to a commit of `main` whose checks have passed. Outside the
course, `pip install torchlingo` installs the latest release as usual.

Because `course` moves between releases, the version number alone no longer says which code is
installed, so `setup()` prints `torchlingo.version_label()`: the release, plus the branch and
commit for an install from GitHub (`0.2.5 (course @ 11aa496)`). Quote that line when reporting a
problem.

Then:

```python
from torchlingo.colab import setup

env = setup(gpu=True, drive=True, data=["data/pretrained/model.pt"])
# TorchLingo 0.2.5 (course @ 11aa496) | PyTorch 2.x | device Tesla T4 | Colab
# Drive: /content/drive/MyDrive
#   data/pretrained/model.pt    10.2 MB

out_dir = env.drive_dir / "CS479"      # survives a disconnect
device = env.device                    # cuda when there is a GPU, else cpu
```

- **`gpu=True`** stops with instructions when there is no GPU, rather than let a training
  run start on the CPU and take days.
- **`drive=True`** mounts Google Drive in Colab and returns `MyDrive`; elsewhere it returns
  the current directory.
- **`data=[...]`** downloads repository files that are missing, choosing GitHub's ordinary
  or Git LFS host by itself. A clone made without `git lfs pull` has pointer files in place
  of the real ones; those are replaced too.

## API

::: torchlingo.colab
    options:
      show_source: true
