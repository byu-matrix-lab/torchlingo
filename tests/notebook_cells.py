"""Run a Colab-only notebook's code cells outside Colab, for tests.

``scripts/execute_notebooks.py`` skips every notebook that declares ``colab``: they need a GPU and
a Drive holding a student's own files, and CI has neither. So until a test like this one exists,
nothing executes them. Lecture 9's Part B shipped with a ``NameError`` that its first run found
in minutes.

This runs chosen cells in-process, from the notebook file itself, so the test exercises whatever
the notebook says today. Three things are faked, and nothing else:

- **Drive** is a scratch directory. Outside Colab, ``setup(drive=True)`` treats the current
  directory as ``MyDrive``, so :func:`colab_faked` changes into it.
- **The GPU requirement** is lifted. ``setup(gpu=True)`` stops when CUDA is absent; here it runs
  on whatever device there is.
- **Sizes** a test cannot afford (epochs, model dimensions) are swapped by :func:`substitute`,
  which fails if a swap no longer matches the notebook. A swap that silently stopped applying
  would turn a quick test into an hours-long one, or test something the notebook no longer
  says.
"""

import contextlib
import io
import json
import os
from pathlib import Path

import torchlingo.colab

COURSE = Path(__file__).resolve().parents[1] / "docs" / "docs" / "course"


def code_cells(notebook: Path) -> list[tuple[int, str]]:
    """Return ``(index, source)`` for every code cell, indexed as in the notebook."""
    cells = json.loads(Path(notebook).read_text(encoding="utf-8"))["cells"]
    return [
        (i, "".join(c["source"]))
        for i, c in enumerate(cells)
        if c["cell_type"] == "code"
    ]


def cell_containing(cells: list[tuple[int, str]], marker: str) -> tuple[int, str]:
    """The one code cell whose source contains ``marker``.

    Cells are picked by what they say rather than by position, so adding a cell elsewhere does
    not silently point a test at the wrong one.

    Raises:
        LookupError: If no cell, or more than one, contains the marker.
    """
    found = [cell for cell in cells if marker in cell[1]]
    if len(found) != 1:
        raise LookupError(
            f"{len(found)} code cells contain {marker!r}; expected exactly 1"
        )
    return found[0]


def substitute(
    cells: list[tuple[int, str]], replacements: dict[str, str]
) -> list[tuple[int, str]]:
    """Apply every replacement to every cell.

    Raises:
        LookupError: If some replacement matches none of the cells.
    """
    unmatched = [old for old in replacements if not any(old in src for _, src in cells)]
    if unmatched:
        raise LookupError(f"not in these cells any more: {unmatched}")
    out = []
    for i, src in cells:
        for old, new in replacements.items():
            src = src.replace(old, new)
        out.append((i, src))
    return out


@contextlib.contextmanager
def colab_faked(drive_dir: Path):
    """Run with ``drive_dir`` as Drive and no GPU required; restore both afterwards."""
    real_setup = torchlingo.colab.setup

    def setup_without_gpu(**kwargs):
        return real_setup(**{**kwargs, "gpu": False})

    previous = Path.cwd()
    torchlingo.colab.setup = setup_without_gpu
    os.chdir(drive_dir)
    try:
        yield
    finally:
        os.chdir(previous)
        torchlingo.colab.setup = real_setup


def run_cells(
    cells: list[tuple[int, str]], namespace: dict | None = None
) -> tuple[dict, str]:
    """Execute cells in order in one namespace, as a kernel would.

    Returns:
        tuple[dict, str]: The namespace afterwards, and everything the cells printed.
        A traceback names the failing cell as ``cell-<index>``.
    """
    namespace = {} if namespace is None else namespace
    printed = io.StringIO()
    with contextlib.redirect_stdout(printed):
        for i, src in cells:
            # The repository's own notebook, run as a kernel runs it: the point of the module.
            exec(compile(src, f"cell-{i}", "exec"), namespace)  # noqa: S102
    return namespace, printed.getvalue()
