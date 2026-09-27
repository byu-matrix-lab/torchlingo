"""Execute the tutorial notebooks to prove they still run.

The docs site does **not** run these. `docs/mkdocs.yml` configures mkdocs-jupyter
with `execute: false` and `allow_errors: true`, so a tutorial can rot completely
-- wrong results, or an outright exception -- and neither the docs build nor a
reader will surface it. That is not hypothetical: tutorial 2 once shipped
producing empty translations for every phrase while printing a training curve,
and tutorial 3 raised `NameError` on a clean run. Both went unnoticed.

A broken tutorial is worse in a teaching library than anywhere else, because a
student cannot tell "the tutorial is broken" from "I did it wrong."

Two details this script exists to get right:

1. **Run in a scratch directory.** The notebooks write `data/` and
   `checkpoints/` relative to their own location, and those paths are only
   gitignored at the repo root. Executing in place litters the docs tree.
2. **Run in filename order, in one directory.** Tutorial 3 loads the checkpoint
   tutorial 2 saves, so the order matters and the two must share a working
   directory.
3. **Skip, do not fail, when the data is not there.** The corpus and the
   pretrained checkpoint live in Git LFS and CI checks out without it, so those
   tutorials are skipped in CI and exercised in a normal clone. See
   ``REQUIREMENTS``.

Run:
    python scripts/execute_notebooks.py

Exits non-zero if any notebook fails, so CI can gate on it.
"""

import argparse
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

TUTORIALS = Path("docs/docs/tutorials")
TIMEOUT_SECONDS = 900

# Which notebooks cannot run without a Git LFS artifact.
#
# CI checks out without LFS on purpose, so these files are pointers there and
# the notebooks that need them are skipped rather than failed. A developer with
# a normal clone runs all of them.
#
# Tutorial 3 lists the corpus even though it never names it: it loads the
# checkpoint tutorial 2 trains, so it inherits tutorial 2's inputs.
REQUIREMENTS = {
    "02-train-tiny-model.ipynb": ["data/example.tsv"],
    "03-inference-and-beamsearch.ipynb": ["data/example.tsv"],
    # Part 8 loads the pretrained checkpoint to show cross-attention on a
    # Transformer, so this tutorial now needs the same artifacts tutorial 5 does.
    "04-attention-and-alignment.ipynb": ["data/example.tsv", "data/pretrained/model.pt"],
    "05-real-translations.ipynb": ["data/example.tsv", "data/pretrained/model.pt"],
}


def execute(notebook: Path, workdir: Path, timeout: int) -> tuple[bool, str]:
    """Run one notebook in ``workdir`` and report whether it succeeded.

    Args:
        notebook (Path): The notebook to execute, already copied into workdir.
        workdir (Path): Directory to execute in; also where the notebook writes
            any files it creates.
        timeout (int): Per-notebook timeout in seconds.

    Returns:
        tuple[bool, str]: Success flag, and captured output when it failed.
    """
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "jupyter",
            "nbconvert",
            "--to",
            "notebook",
            "--execute",
            "--inplace",
            f"--ExecutePreprocessor.timeout={timeout}",
            notebook.name,
        ],
        cwd=workdir,
        capture_output=True,
        text=True,
        check=False,
    )
    return result.returncode == 0, result.stdout + result.stderr


def is_available(path: Path) -> bool:
    """Report whether a file is really present, not just an LFS pointer.

    CI checks out without LFS on purpose, to keep it fast and off the bandwidth
    quota. Large artifacts therefore arrive as ~130-byte pointer files. They
    exist, they are readable, and feeding one to pandas or torch produces a
    baffling parse error rather than a useful message -- so callers check.

    Args:
        path (Path): File to test.

    Returns:
        bool: True if the real content is present.
    """
    if not path.exists():
        return False
    with path.open("rb") as handle:
        return not handle.read(40).startswith(b"version https://git-lfs")


def missing_requirements(notebook: Path) -> list[str]:
    """List the artifacts this notebook needs that have not been fetched.

    Args:
        notebook (Path): The notebook, identified by filename.

    Returns:
        list[str]: Repo-relative paths that are absent or still LFS pointers.
    """
    return [
        name
        for name in REQUIREMENTS.get(notebook.name, [])
        if not is_available(Path(name))
    ]


def link_repo_data(workdir: Path) -> None:
    """Expose the repository's read-only data inside the scratch directory.

    Some tutorials read shipped files -- the corpus, the pretrained model --
    while others *write* into ``data/`` as they go. So ``data/`` itself is a real
    directory in the scratch, and only the read-only members are symlinked into
    it. Linking the whole directory instead would let a notebook write into the
    repository, which is exactly what running in a scratch directory is meant to
    prevent.

    Args:
        workdir (Path): The scratch directory notebooks execute in.
    """
    source = Path("data").resolve()
    if not source.is_dir():
        return
    target = workdir / "data"
    target.mkdir(exist_ok=True)
    for name in ("example.tsv", "pretrained", "multilingual_example"):
        member = source / name
        if member.exists():
            (target / name).symlink_to(member)


def main() -> int:
    """Execute every tutorial notebook in order and summarize."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--tutorials", type=Path, default=TUTORIALS)
    parser.add_argument("--timeout", type=int, default=TIMEOUT_SECONDS)
    args = parser.parse_args()

    notebooks = sorted(args.tutorials.glob("*.ipynb"))
    if not notebooks:
        print(f"No notebooks found under {args.tutorials}", file=sys.stderr)
        return 1

    failures = []
    skipped = []
    # One shared scratch directory: tutorial 3 needs the checkpoint tutorial 2
    # writes, so they cannot be isolated from each other.
    with tempfile.TemporaryDirectory(prefix="torchlingo-notebooks-") as tmp:
        workdir = Path(tmp)
        for notebook in notebooks:
            shutil.copy(notebook, workdir / notebook.name)
        link_repo_data(workdir)

        for notebook in notebooks:
            absent = missing_requirements(notebook)
            if absent:
                skipped.append(notebook.name)
                print(
                    f"  SKIP  {notebook.name} (needs {', '.join(absent)})", flush=True
                )
                continue
            print(f"executing {notebook.name} ...", flush=True)
            ok, output = execute(workdir / notebook.name, workdir, args.timeout)
            if ok:
                print(f"  PASS  {notebook.name}", flush=True)
            else:
                failures.append(notebook.name)
                print(f"  FAIL  {notebook.name}", flush=True)
                print(output, file=sys.stderr, flush=True)

    print()
    ran = len(notebooks) - len(skipped)
    print(f"{ran - len(failures)}/{ran} notebooks executed cleanly")
    if skipped:
        print(
            f"{len(skipped)} skipped for unfetched data: " + ", ".join(skipped),
            flush=True,
        )
    if failures:
        print("failed: " + ", ".join(failures), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
