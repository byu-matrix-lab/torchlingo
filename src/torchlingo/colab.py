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
    >>> a8_dir = course_dir(env.drive_dir, "assignment8", needs=["train.tsv"])  # doctest: +SKIP
"""

from __future__ import annotations

import importlib.metadata
import json
import os
import urllib.request
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

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


COURSE_FOLDER = "CS479"


def course_dir(
    drive_dir: str | Path | None,
    assignment: str | None = None,
    *,
    needs: Iterable[str | Path] = (),
    hint: str = "",
    create: bool = False,
) -> Path:
    """Find a course folder on Drive, stopping with instructions if files are missing.

    The course notebooks keep each assignment's work in ``MyDrive/CS479/<assignment>``, and
    one notebook reads what an earlier one wrote there. When a file is missing, a student
    should see which files, in which folder, and what writes them, not a ``FileNotFoundError``
    traceback several cells later. This function checks every file at once and stops with
    :class:`SystemExit`, which a notebook shows as the message alone.

    Args:
        drive_dir (str | Path | None): Where Drive is, normally ``setup(drive=True).drive_dir``.
        assignment (str | None, optional): The folder under ``CS479``, such as
            ``"assignment8"``. Defaults to ``CS479`` itself.
        needs (Iterable[str | Path], optional): Paths inside the folder that must exist, such
            as ``"train.tsv"`` or ``"best-a9/model_best.pt"``.
        hint (str, optional): Appended to the message when something is missing: what writes
            the files, and what to do. Defaults to no hint.
        create (bool, optional): Make the folder if it does not exist, for a notebook that
            writes there. Defaults to False.

    Returns:
        Path: The folder.

    Raises:
        ValueError: If ``drive_dir`` is None, because :func:`setup` was not asked for Drive.
        SystemExit: If any of ``needs`` is missing. The message names each one, the folder,
            and ``hint``.

    Examples:
        >>> import tempfile
        >>> from pathlib import Path
        >>> with tempfile.TemporaryDirectory() as drive:
        ...     out = course_dir(drive, "assignment8", create=True)
        ...     _ = (out / "train.tsv").write_text("src\\ttgt\\n")
        ...     found = course_dir(drive, "assignment8", needs=["train.tsv"])
        ...     print(found.relative_to(drive).as_posix())
        ...     try:
        ...         course_dir(drive, "assignment8", needs=["train.tsv", "test.tsv"],
        ...                    hint="The A8 notebook writes them.")
        ...     except SystemExit as stopped:
        ...         print(stopped)
        CS479/assignment8
        Not found in .../CS479/assignment8: test.tsv. The A8 notebook writes them.
    """
    if drive_dir is None:
        raise ValueError(
            "No Drive folder: call setup(drive=True) and pass its drive_dir."
        )
    folder = Path(drive_dir) / COURSE_FOLDER
    if assignment is not None:
        folder = folder / assignment
    if create:
        folder.mkdir(parents=True, exist_ok=True)

    missing = [Path(n).as_posix() for n in needs if not (folder / n).exists()]
    if missing:
        where = "" if folder.is_dir() else " (the folder itself does not exist)"
        message = f"Not found in {folder}{where}: {', '.join(missing)}."
        raise SystemExit(f"{message} {hint}".rstrip())
    return folder


def _write_json(path: Path, data: dict[str, Any]) -> None:
    """Write data as JSON, whole or not at all: a dropped session leaves the old file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(path.name + ".partial")
    partial.write_text(
        json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    partial.replace(path)


class Notes:
    """A notebook's notes on Drive: what each step saves for the write-up.

    Colab drops idle sessions, and with them every variable. A notebook whose write-up
    needs a prediction from Step 2 and scores from Step 5 saves each as it goes, into one
    small JSON file beside its outputs, and the write-up reads them back from there, so it
    runs after the setup cell alone.

    Args:
        path (str | Path): The JSON file, such as ``A8_DIR / "a14-notes.json"``. It need
            not exist yet.

    Examples:
        >>> import tempfile
        >>> from pathlib import Path
        >>> with tempfile.TemporaryDirectory() as drive:
        ...     notes = Notes(Path(drive) / "a14-notes.json")
        ...     print(notes.load())
        ...     _ = notes.save(prediction="beam beats greedy by a point")
        ...     _ = notes.save(steps_taken=1000)
        ...     print(Notes(Path(drive) / "a14-notes.json").load())
        {}
        {'prediction': 'beam beats greedy by a point', 'steps_taken': 1000}
    """

    def __init__(self, path: str | Path):
        self.path = Path(path)

    def load(self) -> dict[str, Any]:
        """Everything saved so far, or ``{}`` if nothing has been."""
        if not self.path.exists():
            return {}
        return json.loads(self.path.read_text(encoding="utf-8"))

    def save(self, **items: Any) -> dict[str, Any]:
        """Add ``items`` to the notes, replacing any of the same name, and return them all.

        Each value must be something JSON can hold: text, numbers, lists, dicts.
        """
        notes = {**self.load(), **items}
        _write_json(self.path, notes)
        return notes


# What a later assignment needs from the finished A9, as Lecture 9's Part B leaves them.
BASELINE_FILES = (
    "train.tsv",
    "val.tsv",
    "test.tsv",
    "spm.model",
    "a9-settings.json",
    "best-a9/model_best.pt",
)
# A9's model, for a file saved before TorchLingo recorded settings (October 14, 2026).
BASELINE_SHAPE = {
    "d_model": 512,
    "n_heads": 8,
    "num_encoder_layers": 6,
    "num_decoder_layers": 6,
    "d_ff": 2048,
}


@dataclass
class Baseline:
    """A finished A9, as a later assignment compares against it. See :func:`load_baseline`.

    Attributes:
        dir: Its folder on Drive.
        settings: ``a9-settings.json``, as A9 wrote it.
        no_spaces: Whether the language is written without spaces.
        tokenize: The BLEU tokenizer A9 was scored with.
        bleu_name: That BLEU's name, for a report.
        vocab: The SentencePiece vocabulary, one for both languages.
        spm_path: Where that vocabulary's model is, as a string, for ``create_dataloaders``.
        config: A ``Config`` that builds and trains a model the way A9 did.
        train, val, test: The split, each a DataFrame of text.
        device: Where :meth:`load_best` puts a model.
    """

    dir: Path
    settings: dict[str, Any]
    no_spaces: bool
    tokenize: str
    bleu_name: str
    vocab: Any
    spm_path: str
    config: Any
    train: Any
    val: Any
    test: Any
    device: Any

    def load_best(self, best_dir: str = "best-a9") -> Any:
        """The best model a training kept in a folder here, ready to translate.

        Args:
            best_dir: The folder under :attr:`dir`, such as ``"best-a13"``.

        Returns:
            The model, in evaluation mode.
        """
        from .checkpoint import load_model

        return load_model(
            self.dir / best_dir / "model_best.pt",
            self.device,
            config=self.config,
            src_vocab=self.vocab,
        ).model


def load_baseline(
    env: NotebookEnvironment,
    *,
    needs: Iterable[str | Path] = (),
    folder: str = "assignment8",
) -> Baseline:
    """Open a student's finished A9, the baseline A13, A14 and a Final Project change one thing against.

    Those notebooks are about back-translation or multilingual tags, and teach nothing by
    reloading A9, yet each began with the same thirty lines. The comparison is only fair if
    everything but the one change is A9's, so this reads what A9 left rather than repeating
    it: the model's shape comes from A9's own checkpoint, so "the same configuration" cannot
    drift. A model file too old to record its shape gets A9's, :data:`BASELINE_SHAPE`.

    Args:
        env: What ``setup(drive=True)`` returned.
        needs: Files in the folder this notebook needs besides :data:`BASELINE_FILES`.
        folder: The folder under ``MyDrive/CS479``.

    Returns:
        A :class:`Baseline`.

    Raises:
        SystemExit: If a file is missing. The message names each one.

    Examples:
        >>> baseline = load_baseline(env, needs=["test-a9.hyp"])    # doctest: +SKIP
        >>> model = baseline.load_best("best-a9")                   # doctest: +SKIP
    """
    from .checkpoint import load_checkpoint
    from .config import Config
    from .data_processing.vocab import SentencePieceVocab
    from .evaluation import bleu_name, bleu_tokenizer
    from .preprocessing import load_data

    directory = course_dir(
        env.drive_dir,
        folder,
        needs=[*BASELINE_FILES, *needs],
        hint="Lecture 9's Part B writes them; finish A9 first.",
    )
    settings = json.loads((directory / "a9-settings.json").read_text(encoding="utf-8"))
    recorded = load_checkpoint(directory / "best-a9" / "model_best.pt", "cpu")
    recorded = recorded.get("model_config") or {}
    shape = {key: recorded.get(key, value) for key, value in BASELINE_SHAPE.items()}
    spm_path = str(directory / "spm.model")
    return Baseline(
        dir=directory,
        settings=settings,
        no_spaces=settings["no_spaces"],
        tokenize=bleu_tokenizer(settings["no_spaces"]),
        bleu_name=bleu_name(settings["no_spaces"]),
        vocab=SentencePieceVocab(spm_path),
        spm_path=spm_path,
        config=Config(
            **shape,
            batch_size=64,
            num_workers=0,
            max_decode_length=settings["max_decode_length"],
        ),
        train=load_data(directory / "train.tsv"),
        val=load_data(directory / "val.tsv"),
        test=load_data(directory / "test.tsv"),
        device=env.device,
    )


RERUN_PACKAGES = ("torch", "sacrebleu", "transformers")


def save_rerun(
    path: str | Path,
    *,
    packages: Iterable[str] = RERUN_PACKAGES,
    **settings: Any,
) -> dict[str, Any]:
    """Save what someone needs to rerun a notebook and get the same numbers.

    The course notebooks end by writing a small JSON record for the write-up: the
    settings that decided the numbers (model, seed, steps, how it was scored) and the
    versions of the code that produced them. The settings are the notebook's to name;
    this adds the versions, under ``"versions"``: :func:`torchlingo.version_label`, which
    says which ``course`` commit was installed, and each of ``packages``.

    Args:
        path (str | Path): The JSON file, such as ``A8_DIR / "a14-rerun.json"``.
        packages (Iterable[str], optional): Packages whose installed versions to record.
            Defaults to torch, sacrebleu and transformers. One not installed is recorded
            as ``"not installed"``.
        **settings: The settings, each something JSON can hold.

    Returns:
        dict: The record as written.

    Examples:
        >>> import tempfile
        >>> from pathlib import Path
        >>> with tempfile.TemporaryDirectory() as drive:
        ...     record = save_rerun(Path(drive) / "a14-rerun.json", seed=479, steps=1000,
        ...                         packages=["torch", "no-such-package"])
        ...     back = json.loads((Path(drive) / "a14-rerun.json").read_text())
        >>> back == record, back["seed"], back["versions"]["no-such-package"]
        (True, 479, 'not installed')
    """
    from . import version_label

    versions = {"torchlingo": version_label()}
    for package in packages:
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = "not installed"
    record = {**settings, "versions": versions}
    _write_json(Path(path), record)
    return record
