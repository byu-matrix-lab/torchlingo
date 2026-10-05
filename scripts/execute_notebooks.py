"""Execute the tutorial notebooks to prove they still run.

The docs site does **not** run these. `docs/mkdocs.yml` configures mkdocs-jupyter
with `execute: false` and `allow_errors: true`, so a tutorial can rot completely
-- wrong results, or an outright exception -- and neither the docs build nor a
reader will surface it. That is not hypothetical: tutorial 3 once shipped
producing empty translations for every phrase while printing a training curve,
and tutorial 8 raised `NameError` on a clean run. Both went unnoticed.

A broken tutorial is worse in a teaching library than anywhere else, because a
student cannot tell "the tutorial is broken" from "I did it wrong."

Two details this script exists to get right:

1. **Run in a scratch directory.** The notebooks write `data/` and
   `checkpoints/` relative to their own location, and those paths are only
   gitignored at the repo root. Executing in place litters the docs tree.
2. **Run in filename order, in one directory.** Tutorial 8 loads the checkpoint
   tutorial 3 saves, so the order matters and the two must share a working
   directory.
3. **Skip, do not fail, when the data is not there.** The corpus and the
   pretrained checkpoint live in Git LFS and CI checks out without it, so those
   tutorials are skipped in CI and exercised in a normal clone. Each notebook
   declares what it needs in its own ``torchlingo`` metadata -- see
   ``scripts/notebook_meta.py``.

Run:
    python scripts/execute_notebooks.py
    python scripts/execute_notebooks.py --as-student   # PyPI install, Colab faked

``--as-student`` runs each notebook through ``scripts/student_path.sh`` instead:
a fresh environment *per notebook*, the notebook's own install cell pulling
from PyPI, no ``data/`` linked in. It tests what is released, not what is
checked out, so it runs on a schedule rather than on pull requests.

Exits non-zero if any notebook fails, so CI can gate on it.
"""

import argparse
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

from notebook_meta import is_redirect, read_meta

TUTORIALS = Path("docs/docs/tutorials")
COURSE = Path("docs/docs/course")
TIMEOUT_SECONDS = 900
STUDENT_PATH = Path(__file__).resolve().parent / "student_path.sh"

# What the student path can supply. It installs packages and downloads, and it fakes
# `google.colab` well enough for a notebook that mounts Drive only to write into it. It has
# no GPU, no real Drive holding a student's own files, and no token -- so a notebook that
# declares `colab` (in the course, that means one of those), `hf-token` or `blanks` is skipped.
STUDENT_CANNOT = {"blanks", "colab", "hf-token"}

# Failures already filed. Reported, not counted, so a scheduled run stays green on news it
# already has; an entry that starts passing is reported too, so it gets removed.
KNOWN_STUDENT_FAILURES = {
    "08-inference-and-beamsearch.ipynb": "Task #166: loads a checkpoint tutorial 3 "
    "saved in a different runtime",
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

    Read from the notebook's own ``torchlingo.needs`` metadata rather than a table here.
    A table in this file was a second place to declare the same fact, and it had already
    drifted once: Part 8 of tutorial 4 started loading the pretrained checkpoint, and the
    table had to be remembered separately or CI would fail on a missing LFS artifact
    instead of skipping.

    Args:
        notebook (Path): The notebook to inspect. Must be the repository copy, not the
            scratch copy, since only the former is the source of truth.

    Returns:
        list[str]: Repo-relative paths that are absent or still LFS pointers.
    """
    needs = read_meta(notebook).get("needs", [])
    return [name for name in needs if not is_available(Path(name))]


def missing_capabilities(notebook: Path) -> list[str]:
    """List the environment capabilities this notebook declares and CI does not have.

    Distinct from :func:`missing_requirements`, which is about *files in this repository*.
    A notebook can be missing no file and still be unrunnable here because it installs a
    package, downloads a model, mounts Google Drive or wants a HuggingFace token. Those are
    capabilities, not paths, which is why declaring them needed a second field.

    Every declared capability counts as missing, because none of them is available in the
    test job and none should be added to it silently: Eric's standing decision is that CI
    stays lightweight, so a notebook that wants more than a plain environment is skipped
    rather than accommodated.

    Args:
        notebook (Path): The notebook to inspect.

    Returns:
        list[str]: The capabilities it declared, sorted; empty when it needs none.
    """
    return sorted(read_meta(notebook).get("requires", []))


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


def run_as_student(notebook: Path) -> tuple[bool, str]:
    """Run one notebook through ``student_path.sh``, in an environment of its own.

    One environment per notebook, because a shared one exercises only the first
    notebook's install cell: every later one finds torchlingo already there.

    Args:
        notebook (Path): Repository path of the notebook.

    Returns:
        tuple[bool, str]: Success flag, and the harness's output.
    """
    student_dir = tempfile.mkdtemp(prefix=f"torchlingo-student-{notebook.stem}-")
    result = subprocess.run(
        ["bash", str(STUDENT_PATH), str(notebook)],
        env={**os.environ, "STUDENT_DIR": student_dir},
        capture_output=True,
        text=True,
        check=False,
    )
    return result.returncode == 0, result.stdout + result.stderr


def main_as_student(notebooks: list[Path]) -> int:
    """Run every notebook the student path can supply, and summarize."""
    failures, skipped, known = [], [], []
    for notebook in notebooks:
        unavailable = sorted(set(missing_capabilities(notebook)) & STUDENT_CANNOT)
        if unavailable:
            skipped.append(notebook.name)
            print(f"  SKIP  {notebook.name} (requires {', '.join(unavailable)})")
            continue
        print(f"as a student: {notebook.name} ...", flush=True)
        ok, output = run_as_student(notebook)
        reason = KNOWN_STUDENT_FAILURES.get(notebook.name)
        if reason and not ok:
            known.append(notebook.name)
            print(f"  KNOWN {notebook.name} ({reason})", flush=True)
        elif reason:
            print(
                f"  PASS  {notebook.name}: now passes; remove it from "
                "KNOWN_STUDENT_FAILURES",
                flush=True,
            )
        elif ok:
            print(f"  PASS  {notebook.name}", flush=True)
        else:
            failures.append(notebook.name)
            print(f"  FAIL  {notebook.name}", flush=True)
            print(output, file=sys.stderr, flush=True)

    ran = len(notebooks) - len(skipped) - len(known)
    print(f"\n{ran - len(failures)}/{ran} notebooks ran cleanly from a PyPI install")
    if known:
        print(f"{len(known)} known failures, not counted: " + ", ".join(known))
    if failures:
        print("failed: " + ", ".join(failures), file=sys.stderr)
        return 1
    return 0


def main() -> int:
    """Execute every tutorial notebook in order and summarize."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--tutorials", type=Path, default=TUTORIALS)
    parser.add_argument("--course", type=Path, default=COURSE)
    parser.add_argument("--timeout", type=int, default=TIMEOUT_SECONDS)
    parser.add_argument(
        "--as-student",
        action="store_true",
        help="install from PyPI and fake Colab, one fresh environment per notebook",
    )
    args = parser.parse_args()

    # Both families, because the course notebooks are the ones students open in the room, on
    # a clock -- a broken one costs twenty minutes of class rather than a confusing evening.
    # They were executed by nothing at all until this was widened.
    #
    # Tutorials first and in name order, because tutorial 8 loads the checkpoint tutorial 3
    # writes. The course notebooks are independent of each other and of the tutorials.
    # Redirect stubs (Task #186) are left out: one Markdown cell, nothing to execute.
    notebooks = [
        path
        for path in sorted(args.tutorials.glob("*.ipynb"))
        + sorted(args.course.glob("*.ipynb"))
        if not is_redirect(path)
    ]
    if not notebooks:
        print(
            f"No notebooks found under {args.tutorials} or {args.course}",
            file=sys.stderr,
        )
        return 1
    if args.as_student:
        return main_as_student(notebooks)

    failures = []
    skipped = []
    # One shared scratch directory: tutorial 8 needs the checkpoint tutorial 3
    # writes, so they cannot be isolated from each other.
    with tempfile.TemporaryDirectory(prefix="torchlingo-notebooks-") as tmp:
        workdir = Path(tmp)
        for notebook in notebooks:
            shutil.copy(notebook, workdir / notebook.name)
        link_repo_data(workdir)

        for notebook in notebooks:
            unavailable = missing_capabilities(notebook)
            if unavailable:
                skipped.append(notebook.name)
                print(
                    f"  SKIP  {notebook.name} (requires {', '.join(unavailable)})",
                    flush=True,
                )
                continue
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
