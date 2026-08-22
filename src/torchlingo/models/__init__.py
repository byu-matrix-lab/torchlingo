"""Model package aggregating available architectures for the torchlingo package.

Exposes a factory `get_model(name, **kwargs)` to instantiate models.
"""

from .lstm_simple import SimpleSeq2SeqLSTM
from .transformer_simple import SimpleTransformer

_MODEL_MAP = {
    "transformer_simple": SimpleTransformer,
    "lstm_simple": SimpleSeq2SeqLSTM,
}


def get_model(name: str, **kwargs):
    if name not in _MODEL_MAP:
        raise ValueError(
            f"Unknown model '{name}'. Available: {list(_MODEL_MAP.keys())}"
        )
    return _MODEL_MAP[name](**kwargs)


__all__ = ["SimpleSeq2SeqLSTM", "SimpleTransformer", "get_model"]
