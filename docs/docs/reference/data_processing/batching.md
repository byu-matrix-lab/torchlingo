# Batching

Data loading and batching utilities for neural machine translation.

## Overview

This module provides:

- **`collate_fn`**: Pads variable-length sequences for batching
- **`BucketBatchSampler`**: Groups similar-length sequences to minimize padding
- **`create_dataloaders`**: Convenience function for creating train/val loaders

## Quick Start

```python
from torch.utils.data import DataLoader
from torchlingo.data_processing import NMTDataset, collate_fn, create_dataloaders

# Manual dataloader
dataset = NMTDataset("data/train.tsv")
loader = DataLoader(dataset, batch_size=32, collate_fn=collate_fn)

# Or use the convenience function
train_loader, val_loader = create_dataloaders(
    train_file="data/train.tsv",
    val_file="data/val.tsv",
    batch_size=32,
)
```

## API Reference

### collate_fn

::: torchlingo.data_processing.batching.collate_fn
    options:
      show_source: true

### BucketBatchSampler

::: torchlingo.data_processing.batching.BucketBatchSampler
    options:
      show_source: true
      members:
        - __init__
        - __iter__
        - __len__

### create_dataloaders

::: torchlingo.data_processing.batching.create_dataloaders
    options:
      show_source: true

## Examples

### Basic Batching

```python
from torch.utils.data import DataLoader
from torchlingo.data_processing import NMTDataset, collate_fn

dataset = NMTDataset("data/train.tsv")

loader = DataLoader(
    dataset,
    batch_size=32,
    shuffle=True,
    collate_fn=collate_fn,
    num_workers=4,  # Parallel data loading
)

for src_batch, tgt_batch in loader:
    print(f"Source: {src_batch.shape}")  # [32, max_src_len]
    print(f"Target: {tgt_batch.shape}")  # [32, max_tgt_len]
    break
```

### Bucketed Batching

Bucketing groups sequences of similar length together, reducing wasted computation on padding:

```python
from torch.utils.data import DataLoader
from torchlingo.data_processing import NMTDataset, collate_fn, BucketBatchSampler

dataset = NMTDataset("data/train.tsv")

sampler = BucketBatchSampler(dataset, batch_size=32)

loader = DataLoader(
    dataset,
    batch_sampler=sampler,  # Note: batch_sampler, not batch_size
    collate_fn=collate_fn,
)
```

### Using create_dataloaders

The easiest way to create properly configured loaders:

```python
from torchlingo.data_processing import create_dataloaders

train_loader, val_loader = create_dataloaders(
    train_file="data/train.tsv",
    val_file="data/val.tsv",
    batch_size=32,
    shuffle_train=True,
    use_bucketing=True,  # Optional: enable bucketing
)
```

### With Shared Vocabularies

```python
from torchlingo.data_processing import NMTDataset, create_dataloaders

# Build vocab on training data
train_dataset = NMTDataset("data/train.tsv")

# Pass to create_dataloaders
train_loader, val_loader = create_dataloaders(
    train_file="data/train.tsv",
    val_file="data/val.tsv",
    src_vocab=train_dataset.src_vocab,
    tgt_vocab=train_dataset.tgt_vocab,
    batch_size=32,
)
```

## How Padding Works

The `collate_fn` pads sequences to the maximum length in each batch:

```
Before padding:
  Sample 1: [2, 5, 6, 3]         # length 4
  Sample 2: [2, 7, 8, 9, 10, 3]  # length 6
  Sample 3: [2, 11, 3]           # length 3

After padding (to length 6):
  Sample 1: [2, 5, 6, 3, 0, 0]   # padded with 0s
  Sample 2: [2, 7, 8, 9, 10, 3]  # no padding needed
  Sample 3: [2, 11, 3, 0, 0, 0]  # padded with 0s
```

## How Bucketing Works

`BucketBatchSampler` assigns sequences to buckets based on length:

```
Bucket 0 (len ≤ 10):  [sample_3, sample_7, sample_12, ...]
Bucket 1 (len ≤ 20):  [sample_1, sample_5, sample_9, ...]
Bucket 2 (len ≤ 40):  [sample_2, sample_8, ...]
Bucket 3 (len > 40):  [sample_4, sample_6, ...]
```

Batches are created from samples within the same bucket, so sequences in a batch have similar lengths.

### Benefits of Bucketing

| Without Bucketing | With Bucketing |
| ----------------- | -------------- |
| Random mixing of lengths | Similar lengths grouped |
| More padding tokens | Minimal padding |
| Wasted computation | Efficient computation |

### Custom Bucket Boundaries

```python
sampler = BucketBatchSampler(
    dataset,
    batch_size=32,
    bucket_boundaries=[10, 25, 50, 100],  # Custom boundaries
)
```

## Configuration

All batching functions respect the `Config` object:

```python
from torchlingo.config import Config

config = Config(
    batch_size=64,
    pad_idx=0,
)

train_loader, val_loader = create_dataloaders(
    train_file="data/train.tsv",
    val_file="data/val.tsv",
    config=config,
)
```

## Performance Tips

1. **Use `num_workers`** for parallel data loading:
   ```python
   DataLoader(..., num_workers=4)
   ```

2. **Enable bucketing** for datasets with variable lengths:
   ```python
   create_dataloaders(..., use_bucketing=True)
   ```

3. **Pin memory** for GPU training:
   ```python
   DataLoader(..., pin_memory=True)
   ```

4. **Adjust batch size** based on GPU memory:
   - Larger batches = faster training but more memory
   - Start with 32, increase until you run out of memory
