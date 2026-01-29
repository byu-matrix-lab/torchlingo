# Dataset

PyTorch Dataset implementation for parallel neural machine translation data.

## Overview

`NMTDataset` loads parallel text corpora and provides encoded tensor pairs suitable for training. It handles:

- Loading multiple file formats (TSV, CSV, JSON, Parquet)
- Data cleaning (removing blanks, normalizing whitespace)
- Automatic vocabulary building
- Sequence encoding with special tokens
- Length truncation

## Quick Start

```python
from torchlingo.data_processing import NMTDataset

# Basic usage - vocabularies built automatically
dataset = NMTDataset("data/train.tsv")

# Access a sample
src_tensor, tgt_tensor = dataset[0]

# Use vocabularies
print(f"Source vocab size: {len(dataset.src_vocab)}")
print(f"Target vocab size: {len(dataset.tgt_vocab)}")
```

## API Reference

::: torchlingo.data_processing.dataset.NMTDataset
    options:
      show_source: true
      members:
        - __init__
        - __len__
        - __getitem__

## Examples

### Custom Column Names

```python
# If your data has different column names
dataset = NMTDataset(
    "data/train.csv",
    src_col="english",
    tgt_col="spanish",
)
```

### Reusing Vocabularies

```python
# Build vocab on training data
train_dataset = NMTDataset("data/train.tsv")

# Reuse for validation (important!)
val_dataset = NMTDataset(
    "data/val.tsv",
    src_vocab=train_dataset.src_vocab,
    tgt_vocab=train_dataset.tgt_vocab,
)
```

### With SentencePiece

```python
from torchlingo.data_processing import SentencePieceVocab

# Load pre-trained SentencePiece models
src_sp = SentencePieceVocab("models/src.model")
tgt_sp = SentencePieceVocab("models/tgt.model")

dataset = NMTDataset(
    "data/train.tsv",
    src_vocab=src_sp,
    tgt_vocab=tgt_sp,
)
```

### Custom Max Length

```python
from torchlingo.config import Config

config = Config(max_seq_length=256)
dataset = NMTDataset("data/train.tsv", config=config)
```

## Attributes

| Attribute | Type | Description |
| --------- | ---- | ----------- |
| `df` | `pd.DataFrame` | Loaded and cleaned dataframe |
| `src_sentences` | `list[str]` | Raw source sentences |
| `tgt_sentences` | `list[str]` | Raw target sentences |
| `src_vocab` | `SimpleVocab` or `SentencePieceVocab` | Source vocabulary |
| `tgt_vocab` | `SimpleVocab` or `SentencePieceVocab` | Target vocabulary |
| `max_length` | `int` | Maximum sequence length |
| `eos_idx` | `int` | End-of-sequence token index |

## Data Cleaning

`NMTDataset` automatically cleans your data:

1. **NaN handling**: Replaced with empty strings
2. **Whitespace**: Stripped from beginning and end
3. **Empty rows**: Removed if either source or target is blank
4. **Type conversion**: All text converted to strings

## Truncation Behavior

Sequences longer than `max_length` are truncated while preserving the EOS token:

```python
# If max_length=10 and sequence has 15 tokens:
# Original: [SOS, t1, t2, ..., t13, EOS]
# Truncated: [SOS, t1, t2, ..., t8, EOS]  # 10 tokens total
```

## Pre-tokenized Data

If your data has pre-tokenized columns:

```python
# Data with src_tokenized and tgt_tokenized columns
dataset = NMTDataset(
    "data/train.tsv",
    src_tok_col="src_tokenized",
    tgt_tok_col="tgt_tokenized",
)

# Dataset will use pre-tokenized columns for vocabulary building
```
