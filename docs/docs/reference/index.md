# API Reference

Welcome to the TorchLingo API reference. This section documents all public classes, functions, and modules.

## Module Overview

TorchLingo is organized into four main modules:

<div class="grid cards" markdown>

-   :material-cog:{ .lg .middle } **Config**

    ---

    Configuration management for hyperparameters, paths, and feature toggles.

    [:octicons-arrow-right-24: Config Reference](config.md)

-   :material-database:{ .lg .middle } **Data Processing**

    ---

    Dataset classes, vocabulary builders, and data loaders.

    [:octicons-arrow-right-24: Data Processing Reference](data_processing/index.md)

-   :material-brain:{ .lg .middle } **Models**

    ---

    Neural network architectures for sequence-to-sequence translation.

    [:octicons-arrow-right-24: Models Reference](models/index.md)

-   :material-file-document:{ .lg .middle } **Preprocessing**

    ---

    Data loading, tokenization, and preprocessing utilities.

    [:octicons-arrow-right-24: Preprocessing Reference](preprocessing/index.md)

</div>

## Quick Links

### Most Used Classes

| Class | Description |
| ----- | ----------- |
| [`Config`](config.md) | Centralized configuration |
| [`NMTDataset`](data_processing/dataset.md) | PyTorch dataset for parallel text |
| [`SimpleVocab`](data_processing/vocab.md) | Simple vocabulary builder |
| [`MeCabVocab`](data_processing/vocab.md#mecabvocab) | Japanese morphological analysis |
| [`JiebaVocab`](data_processing/vocab.md#jiebavocab) | Chinese word segmentation |
| [`SimpleTransformer`](models/transformer.md) | Transformer encoder-decoder |
| [`SimpleSeq2SeqLSTM`](models/lstm.md) | LSTM encoder-decoder |

### Most Used Functions

| Function | Description |
| -------- | ----------- |
| [`get_default_config()`](config.md#torchlingo.config.get_default_config) | Get default configuration |
| [`load_data()`](preprocessing/base.md#torchlingo.preprocessing.base.load_data) | Load data from file |
| [`collate_fn()`](data_processing/batching.md#torchlingo.data_processing.batching.collate_fn) | Batch collation with padding |
| [`create_dataloaders()`](data_processing/batching.md#torchlingo.data_processing.batching.create_dataloaders) | Create train/val data loaders |

## Import Patterns

### Recommended Imports

```python
# Configuration
from torchlingo.config import Config, get_default_config

# Data processing
from torchlingo.data_processing import (
    NMTDataset,
    SimpleVocab,
    SentencePieceVocab,
    MeCabVocab,
    JiebaVocab,
    collate_fn,
    create_dataloaders,
)

# Models
from torchlingo.models import SimpleTransformer, SimpleSeq2SeqLSTM

# Preprocessing
from torchlingo.preprocessing import load_data, save_data, parallel_txt_to_dataframe
```

### Full Module Import

```python
import torchlingo

# Access submodules
cfg = torchlingo.config.get_default_config()
model = torchlingo.models.SimpleTransformer(...)
```

## Conventions

### Parameter Fallback

Most functions accept both explicit parameters and a `config` object. The priority is:

1. Explicit parameter (if provided)
2. Config object value (if provided)
3. Default config value

```python
# These are equivalent:
dataset = NMTDataset("data.tsv", max_length=100)
dataset = NMTDataset("data.tsv", config=Config(max_seq_length=100))
```

### Type Annotations

All public APIs are fully type-annotated:

```python
def encode(self, sentence: str, add_special_tokens: bool = True) -> List[int]:
    ...
```

### Docstring Style

We use Google-style docstrings:

```python
def function(param1: int, param2: str) -> bool:
    """Short description.

    Longer description if needed.

    Args:
        param1: Description of param1.
        param2: Description of param2.

    Returns:
        Description of return value.

    Raises:
        ValueError: When something is wrong.
    """
```

## Version Compatibility

- **Python**: 3.10+
- **PyTorch**: 2.0+
- **Pandas**: 1.5+

## Getting Help

- 📖 Check the [Concepts](../concepts/what-is-nmt.md) section for explanations
- 🎓 Follow the [Tutorials](../tutorials/index.md) for step-by-step guides
- 🐛 Report issues on GitHub
