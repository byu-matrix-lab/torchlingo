# Multilingual

Utilities for handling multilingual data in neural machine translation.

## Overview

Multilingual NMT trains a single model to translate between multiple language pairs. This module provides utilities for:

- Adding language tags to source sentences
- Creating bidirectional (EN↔X) training data
- Preparing multilingual datasets from base preprocessed data

## Why Multilingual?

| Benefit | Description |
| ------- | ----------- |
| **Transfer learning** | Low-resource languages benefit from related high-resource languages |
| **Single model** | One model instead of N² models for N languages |
| **Zero-shot** | May translate between language pairs not seen during training |
| **Bidirectional** | Train both EN→X and X→EN directions simultaneously |

## Quick Start

```python
from torchlingo.preprocessing.multilingual import (
    add_language_tags,
    preprocess_multilingual,
)

# Add language tags to a DataFrame
df_tagged = add_language_tags(df, tag="<es>")

# Or run the full multilingual pipeline
preprocess_multilingual()  # Uses config defaults
```

## API Reference

::: torchlingo.preprocessing.multilingual.add_language_tags
    options:
      show_source: true

::: torchlingo.preprocessing.multilingual.preprocess_multilingual
    options:
      show_source: true

## Language Tag Format

Language tags are prepended to the source sentence to indicate the target language:

```
Source: "Hello world" → "<es> Hello world"
Target: "Hola mundo"
```

The model learns to associate `<es>` with Spanish translations.

### How It Works

```python
from torchlingo.preprocessing.multilingual import add_language_tags
import pandas as pd

# Original data
df = pd.DataFrame({
    'src': ['Hello', 'Goodbye'],
    'tgt': ['Hola', 'Adiós']
})

# Add Spanish target tag
df_tagged = add_language_tags(df, tag='<es>')
print(df_tagged['src'][0])
# '<es> Hello'
```

## Examples

### Basic Usage - Adding Tags

```python
from torchlingo.preprocessing.multilingual import add_language_tags
from torchlingo.preprocessing.base import load_data

# Load English-Spanish data
df = load_data("data/train.tsv")

# Add target language tag
df_tagged = add_language_tags(df, tag="<es>")

print(df_tagged.head())
#                   src            tgt
# 0  <es> Hello world   Hola mundo
# 1  <es> Good morning  Buenos días
```

### Full Multilingual Pipeline

The `preprocess_multilingual` function handles the complete bidirectional data preparation:

```python
from torchlingo.preprocessing.multilingual import preprocess_multilingual
from torchlingo.config import Config

# Configure your paths and tags
config = Config(
    train_file="data/train.tsv",
    val_file="data/val.tsv",
    test_file="data/test.tsv",
    lang_tag_en_to_x="<es>",   # Tag for English→Spanish
    lang_tag_x_to_en="<en>",   # Tag for Spanish→English
    multi_train_file="data/multi_train.tsv",
    multi_val_file="data/multi_val.tsv",
    test_en_x_file="data/test_en_es.tsv",
    test_x_en_file="data/test_es_en.tsv",
)

# Run the pipeline
preprocess_multilingual(config=config)
```

This will:

1. Load your base train/val/test splits
2. Create EN→X copies (tagged with `<es>`)
3. Create X→EN copies (columns swapped, tagged with `<en>`)
4. Combine and shuffle training/validation data
5. Save separate test sets for each direction

### Understanding Bidirectional Data

```python
# Input data (EN→ES):
# src: "Hello"  → tgt: "Hola"

# After preprocess_multilingual:

# Direction 1 (EN→ES):
# src: "<es> Hello"  → tgt: "Hola"

# Direction 2 (ES→EN):  
# src: "<en> Hola"   → tgt: "Hello"
```

Both directions are combined into the training data, shuffled together.

## Configuration Options

The multilingual pipeline uses these configuration options:

| Option | Description | Default |
| ------ | ----------- | ------- |
| `train_file` | Base training data path | `data/train.tsv` |
| `val_file` | Base validation data path | `data/val.tsv` |
| `test_file` | Base test data path | `data/test.tsv` |
| `src_col` | Source column name | `src` |
| `tgt_col` | Target column name | `tgt` |
| `lang_tag_en_to_x` | Tag for EN→X direction | `<es>` |
| `lang_tag_x_to_en` | Tag for X→EN direction | `<en>` |
| `multi_train_file` | Output multilingual train path | `data/multi_train.tsv` |
| `multi_val_file` | Output multilingual val path | `data/multi_val.tsv` |
| `test_en_x_file` | Output EN→X test path | `data/test_en_x.tsv` |
| `test_x_en_file` | Output X→EN test path | `data/test_x_en.tsv` |

## Best Practices

!!! tip "Run Base Preprocessing First"
    The multilingual pipeline expects that base preprocessing has already created `train_file`, `val_file`, and `test_file`. Run `preprocess_base()` first!

!!! tip "Consistent Tags"
    Use consistent language tags throughout your project. Common conventions:
    
    - `<lang>` - Simple: `<en>`, `<es>`, `<fr>`
    - `>>lang<<` - OPUS style: `>>en<<`, `>>es<<`
    - `[lang]` - Bracketed: `[en]`, `[es]`

!!! tip "Separate Test Sets"
    The pipeline creates separate test sets for each direction. This lets you evaluate EN→X and X→EN performance independently.

## Common Workflow

```python
from torchlingo.preprocessing.base import preprocess_base
from torchlingo.preprocessing.multilingual import preprocess_multilingual
from torchlingo.config import Config, set_default_config

# 1. Set up configuration
config = Config(
    src_file="data/english.txt",
    tgt_file="data/spanish.txt",
    lang_tag_en_to_x="<es>",
    lang_tag_x_to_en="<en>",
)
set_default_config(config)

# 2. Base preprocessing (creates train/val/test)
preprocess_base()

# 3. Multilingual preprocessing (creates bidirectional data)
preprocess_multilingual()

# Now you have:
# - data/multi_train.tsv (bidirectional training)
# - data/multi_val.tsv (bidirectional validation)
# - data/test_en_x.tsv (EN→ES test)
# - data/test_x_en.tsv (ES→EN test)
```
