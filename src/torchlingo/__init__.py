"""Top-level package for TorchLingo.

This package exposes the core modules (config, models, data_processing,
preprocessing) so codebases/tests can import `torchlingo` rather than top-level
module names. It mirrors the project layout used in the repository.
"""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("torchlingo")
except PackageNotFoundError:
    # Package is not installed
    __version__ = "unknown"

from . import (
    checkpoint,
    config,
    data_processing,
    evaluation,
    inference,
    models,
    preprocessing,
    training,
)

__all__ = [
    "__version__",
    "checkpoint",
    "config",
    "data_processing",
    "evaluation",
    "inference",
    "models",
    "preprocessing",
    "training",
]
