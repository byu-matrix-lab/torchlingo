# Data Processing

The `data_processing` module provides tools for loading, encoding, and batching parallel text data for neural machine translation.

## Overview

```mermaid
flowchart LR
    A[Raw Data] --> B[NMTDataset]
    B --> C[Vocab implementations]
    C --> D[DataLoader]
    D --> E[collate_fn]
    E --> F[Batched Tensors]
```

## Submodules

<div class="grid cards" markdown>

-   :material-database:{ .lg .middle } **Dataset**

    ---

    PyTorch Dataset for parallel text corpora.

    [:octicons-arrow-right-24: Dataset Reference](dataset.md)

-   :material-book-alphabet:{ .lg .middle } **Vocabulary**

    ---

    Vocabulary classes for encoding and decoding text.

    [:octicons-arrow-right-24: Vocabulary Reference](vocab.md)

-   :material-package-variant:{ .lg .middle } **Batching**

    ---

    Collation functions and bucketed batch samplers.

    [:octicons-arrow-right-24: Batching Reference](batching.md)

</div>

## Quick Start

```python
from torchlingo.data_processing import (
    NMTDataset,
    BaseVocab,
    SimpleVocab,
    SentencePieceVocab,
    MeCabVocab,
    JiebaVocab,
    collate_fn,
    create_dataloaders,
)

# Load data and build vocabularies
dataset = NMTDataset("data/train.tsv")

# Create data loaders
train_loader, val_loader = create_dataloaders(
    train_file="data/train.tsv",
    val_file="data/val.tsv",
    batch_size=32,
)

# Iterate
for src_batch, tgt_batch in train_loader:
    # src_batch: [batch_size, src_len]
    # tgt_batch: [batch_size, tgt_len]
    pass
```

## Key Classes

| Class | Purpose |
| ----- | ------- |
| [`NMTDataset`](dataset.md#torchlingo.data_processing.dataset.NMTDataset) | PyTorch Dataset for parallel text |
| [`BaseVocab`](vocab.md#torchlingo.data_processing.vocab.BaseVocab) | Shared vocabulary interface |
| [`SimpleVocab`](vocab.md#torchlingo.data_processing.vocab.SimpleVocab) | Simple whitespace vocabulary |
| [`SentencePieceVocab`](vocab.md#torchlingo.data_processing.vocab.SentencePieceVocab) | Subword vocabulary wrapper |
| [`MeCabVocab`](vocab.md#torchlingo.data_processing.vocab.MeCabVocab) | Japanese morphological analysis |
| [`JiebaVocab`](vocab.md#torchlingo.data_processing.vocab.JiebaVocab) | Chinese word segmentation |
| [`BucketBatchSampler`](batching.md#torchlingo.data_processing.batching.BucketBatchSampler) | Length-based batching |

## Key Functions

| Function | Purpose |
| -------- | ------- |
| [`collate_fn()`](batching.md#torchlingo.data_processing.batching.collate_fn) | Pad sequences in a batch |
| [`create_dataloaders()`](batching.md#torchlingo.data_processing.batching.create_dataloaders) | Create train/val loaders |

## Typical Workflow

### 1. Load Data

```python
dataset = NMTDataset("data/train.tsv")
print(f"Samples: {len(dataset)}")
print(f"Source vocab: {len(dataset.src_vocab)}")
```

### 2. Access Samples

```python
src_tensor, tgt_tensor = dataset[0]
print(f"Source shape: {src_tensor.shape}")  # [seq_len]
print(f"Target shape: {tgt_tensor.shape}")  # [seq_len]
```

### 3. Create Batches

```python
from torch.utils.data import DataLoader

loader = DataLoader(
    dataset,
    batch_size=32,
    shuffle=True,
    collate_fn=collate_fn,
)
```

### 4. Use in Training

```python
for src_batch, tgt_batch in loader:
    logits = model(src_batch, tgt_batch[:, :-1])
    # ... compute loss
```
