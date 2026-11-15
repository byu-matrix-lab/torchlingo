"""Vocabularies, datasets and batches: text becomes tensors here.

The line between this package and :mod:`torchlingo.preprocessing` is the tensor. Preprocessing
works with files and DataFrames of text; this package turns that text into token ids
(``SimpleVocab``, ``SentencePieceVocab`` and the rest), wraps the pairs in an ``NMTDataset``,
and pads them into batches (``collate_fn``, ``create_dataloaders``) for training.
"""

from .batching import BucketBatchSampler, collate_fn, create_dataloaders
from .dataset import NMTDataset
from .vocab import (
    BaseVocab,
    CharVocab,
    JiebaVocab,
    MeCabVocab,
    SentencePieceVocab,
    SimpleVocab,
)

__all__ = [
    "BaseVocab",
    "BucketBatchSampler",
    "CharVocab",
    "JiebaVocab",
    "MeCabVocab",
    "NMTDataset",
    "SentencePieceVocab",
    "SimpleVocab",
    "collate_fn",
    "create_dataloaders",
]
