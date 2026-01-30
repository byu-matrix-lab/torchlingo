"""Top-level package for TorchLingo.

This package exposes the core modules (config, models, data_processing,
preprocessing) so codebases/tests can import `torchlingo` rather than top-level
module names. It mirrors the project layout used in the repository.
"""

__version__ = "0.0.5"

from . import checkpoint
from . import config
from . import data_processing
from . import inference
from . import models
from . import preprocessing
from . import training

__all__ = [
    "__version__",
    "checkpoint",
    "config",
    "data_processing",
    "inference",
    "models",
    "preprocessing",
    "training",
]
