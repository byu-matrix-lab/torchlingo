"""Top-level package for TorchLingo.

This package exposes the core modules (config, models, data_processing,
preprocessing) so codebases/tests can import `torchlingo` rather than top-level
module names. It mirrors the project layout used in the repository.
"""

from importlib.metadata import version, PackageNotFoundError

try:
    __version__ = version("torchlingo")
except PackageNotFoundError:
    # Package is not installed
    __version__ = "unknown"

from . import config
from . import models
from . import data_processing
from . import preprocessing
from . import training
from . import inference
from . import evaluation

__all__ = [
    "__version__",
    "config",
    "models",
    "data_processing",
    "preprocessing",
    "training",
    "inference",
    "evaluation",
]
