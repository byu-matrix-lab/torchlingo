"""Top-level package for TorchLingo.

This package exposes the core modules (config, models, data_processing,
preprocessing) so codebases/tests can import `torchlingo` rather than top-level
module names. It mirrors the project layout used in the repository.
"""

import json
from importlib.metadata import PackageNotFoundError, distribution, version

try:
    __version__ = version("torchlingo")
except PackageNotFoundError:
    # Package is not installed
    __version__ = "unknown"


def version_label() -> str:
    """Say which TorchLingo code is installed, not only which release it follows.

    `__version__` is the release number from `pyproject.toml`. The course notebooks
    install from the repository's `course` branch, which moves between releases, so
    two students both seeing "0.2.5" may be running different code. pip records where
    an install came from, and this reads it back: a git install adds the branch and
    commit, an editable checkout says so, and a PyPI install is the release alone.

    Returns:
        str: For example ``"0.2.5"``, ``"0.2.5 (course @ 11aa496)"`` or
        ``"0.2.5 (editable checkout)"``.

    Examples:
        >>> import torchlingo
        >>> torchlingo.version_label().startswith(torchlingo.__version__)
        True
    """
    try:
        record = distribution("torchlingo").read_text("direct_url.json")
    except PackageNotFoundError:
        return __version__
    if not record:
        return __version__
    try:
        origin = json.loads(record)
    except ValueError:
        return __version__
    vcs = origin.get("vcs_info") or {}
    if vcs.get("commit_id"):
        ref = vcs.get("requested_revision") or "git"
        return f"{__version__} ({ref} @ {vcs['commit_id'][:7]})"
    if (origin.get("dir_info") or {}).get("editable"):
        return f"{__version__} (editable checkout)"
    return __version__


from . import (
    checkpoint,
    colab,
    config,
    data_processing,
    diagnostics,
    evaluation,
    inference,
    inference_fast,
    models,
    preprocessing,
    training,
    training_checkpoint,
)

__all__ = [
    "__version__",
    "checkpoint",
    "colab",
    "config",
    "data_processing",
    "diagnostics",
    "evaluation",
    "inference",
    "inference_fast",
    "models",
    "preprocessing",
    "training",
    "training_checkpoint",
    "version_label",
]
