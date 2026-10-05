"""Notebook setup in one call: the device, Google Drive, and the data files.

Every course notebook and tutorial used to open with twenty to sixty lines of the same
plumbing: detect Colab, check for a GPU, mount Drive, download the files a pip install does
not ship. None of it is what the notebook teaches, and a student reading it learns nothing
except that notebooks are long. :func:`setup` does all of it and says what it did, so the
notebook's first real cell can be about translation.

It runs the same way in Colab and in a local clone. Outside Colab, Drive is a local folder
and a data file already on disk is left alone, so the notebook needs no branch of its own.

Installing TorchLingo itself cannot live here, since this module is not importable until the
install has happened. That stays a short cell of its own at the top of each notebook.

Example:
    Skipped under ``--doctest-modules``: it downloads, and in Colab it opens Google's
    authorization dialog.

    >>> from torchlingo.colab import setup                        # doctest: +SKIP
    >>> env = setup(gpu=True, drive=True, data=["data/pretrained/model.pt"])  # doctest: +SKIP
    >>> out_dir = env.drive_dir / "CS479"                          # doctest: +SKIP
"""

from __future__ import annotations

import os
import urllib.request
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path

import torch

from .training_checkpoint import DRIVE_MOUNT_POINT, is_colab, mount_drive

REPOSITORY = "byu-matrix-lab/torchlingo"

# GitHub serves ordinary files from one host and Git LFS files from another. Asking the
# ordinary host for an LFS file returns a small pointer file instead of the content, which
# is how fetch_data tells them apart without a hard-coded list of which files are which.
_RAW_URL = "https://raw.githubusercontent.com/{repo}/{ref}/{path}"
_LFS_URL = "https://media.githubusercontent.com/media/{repo}/{ref}/{path}"
_LFS_POINTER = b"version https://git-lfs.github.com/spec/"


def _is_lfs_pointer(path: Path) -> bool:
    """Report whether a file on disk is a Git LFS pointer rather than its content.

    A clone made without ``git lfs pull`` has these in place of the real files.

    Args:
        path (Path): An existing file.

    Returns:
        bool: True if the file starts with the LFS pointer header.
    """
    with path.open("rb") as handle:
        return handle.read(len(_LFS_POINTER)) == _LFS_POINTER


def _download(url: str) -> bytes:
    """Fetch a URL's bytes.

    Args:
        url (str): The URL.

    Returns:
        bytes: The response body.
    """
    with urllib.request.urlopen(url, timeout=120) as response:
        return response.read()


def fetch_data(
    paths: Iterable[str | Path],
    *,
    root: str | Path = ".",
    ref: str = "main",
) -> list[Path]:
    """Make repository files available locally, downloading only what is missing.

    For a notebook opened from a Colab badge, where ``pip install torchlingo`` provides the
    library but none of the repository's ``data/``. A file that already exists is left
    alone, unless it is a Git LFS pointer, in which case the real content replaces it.

    Args:
        paths (Iterable[str | Path]): Repository-relative paths, such as
            ``"data/pretrained/model.pt"``.
        root (str | Path, optional): Where the paths are resolved and written. Defaults to
            the current directory, which in Colab is ``/content``.
        ref (str, optional): Git branch, tag or commit to fetch from. Defaults to ``"main"``.

    Returns:
        list[Path]: The local path of each file, in the order given.

    Raises:
        urllib.error.URLError: If a download fails, for example with no network.

    Examples:
        >>> import tempfile
        >>> from pathlib import Path
        >>> with tempfile.TemporaryDirectory() as tmp:
        ...     present = Path(tmp) / "data" / "notes.txt"
        ...     present.parent.mkdir()
        ...     _ = present.write_text("already here")
        ...     [p.name for p in fetch_data(["data/notes.txt"], root=tmp)]
        ['notes.txt']
    """
    local: list[Path] = []
    for path in paths:
        target = Path(root) / path
        if not target.exists() or _is_lfs_pointer(target):
            relative = Path(path).as_posix()
            content = _download(
                _RAW_URL.format(repo=REPOSITORY, ref=ref, path=relative)
            )
            if content.startswith(_LFS_POINTER):
                content = _download(
                    _LFS_URL.format(repo=REPOSITORY, ref=ref, path=relative)
                )
            target.parent.mkdir(parents=True, exist_ok=True)
            # Write beside the target and move it into place, so a download interrupted
            # halfway never leaves a truncated file that looks like a real one.
            partial = target.with_name(target.name + ".partial")
            partial.write_bytes(content)
            partial.replace(target)
        local.append(target)
    return local


@dataclass
class NotebookEnvironment:
    """What :func:`setup` found and did.

    Attributes:
        device (torch.device): ``cuda`` when a GPU is available, otherwise ``cpu``.
        in_colab (bool): Whether this is a Colab runtime.
        drive_dir (Path | None): Where to keep work that must outlive the runtime:
            ``MyDrive`` when Drive is mounted in Colab, the current directory elsewhere,
            and None when ``drive`` was not requested.
        files (list[Path]): Local paths of the data files requested.
    """

    device: torch.device
    in_colab: bool
    drive_dir: Path | None = None
    files: list[Path] = field(default_factory=list)


def setup(
    *,
    gpu: bool = False,
    drive: bool = False,
    data: Iterable[str | Path] = (),
) -> NotebookEnvironment:
    """Prepare a notebook's environment and print what was done.

    Args:
        gpu (bool, optional): Require a GPU. When True and none is available, stop with
            instructions rather than let a training run start on the CPU and take days.
        drive (bool, optional): Mount Google Drive in Colab, so files written under
            ``drive_dir`` survive a disconnect. Outside Colab, ``drive_dir`` is the current
            directory.
        data (Iterable[str | Path], optional): Repository-relative files the notebook reads,
            downloaded if missing. See :func:`fetch_data`.

    Returns:
        NotebookEnvironment: The device, the Drive directory and the data files.

    Raises:
        RuntimeError: If ``gpu`` is True and no CUDA device is available, or if ``drive``
            is True in Colab and Drive did not mount.

    Examples:
        >>> env = setup()
        TorchLingo ... | PyTorch ... | device cpu | not in Colab
        >>> env.device.type
        'cpu'
    """
    import torchlingo

    in_colab = is_colab()
    if gpu and not torch.cuda.is_available():
        raise RuntimeError(
            "No GPU. In Colab: Runtime > Change runtime type > GPU, then Runtime > Restart, "
            "then run the notebook again from the top. Training on a CPU would take days."
        )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    where = torch.cuda.get_device_name(0) if device.type == "cuda" else device.type
    print(
        f"TorchLingo {torchlingo.version_label()} | PyTorch {torch.__version__} | "
        f"device {where} | {'Colab' if in_colab else 'not in Colab'}"
    )

    drive_dir = None
    if drive:
        # An environment variable, so a test harness can point the mount somewhere
        # writable: /content cannot be created outside Colab.
        mount_point = Path(os.environ.get("TORCHLINGO_DRIVE_MOUNT", DRIVE_MOUNT_POINT))
        if in_colab:
            if not mount_drive(mount_point):
                raise RuntimeError(
                    "Google Drive did not mount. Run this cell again and approve access "
                    "when Google asks; without Drive, a disconnect loses your work."
                )
            drive_dir = mount_point / "MyDrive"
        else:
            drive_dir = Path(".")
        print(f"Drive: {drive_dir}")

    files = fetch_data(data)
    for path in files:
        print(f"  {path}  {path.stat().st_size / 1e6:6.1f} MB")

    return NotebookEnvironment(
        device=device, in_colab=in_colab, drive_dir=drive_dir, files=files
    )
