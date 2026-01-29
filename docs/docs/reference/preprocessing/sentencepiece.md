# SentencePiece

SentencePiece tokenization utilities for subword segmentation.

## Overview

[SentencePiece](https://github.com/google/sentencepiece) is a language-independent subword tokenizer that can handle any language without pre-tokenization. This module provides utilities for training SentencePiece models and using them with TorchLingo.

## Why SentencePiece?

| Problem | Solution |
| ------- | -------- |
| Unknown words | Split into known subwords |
| Large vocabulary | Control vocabulary size |
| Language-agnostic | Works for any language |
| Rare words | Shared subword representations |

### Example

```
"unfortunately" → ["▁un", "fortun", "ately"]
"preprocessing" → ["▁pre", "process", "ing"]
```

The `▁` character marks the start of a word.

## Quick Start

### Training a Model

```python
from torchlingo.preprocessing import train_sentencepiece

train_sentencepiece(
    input_files=["data/train.txt"],
    model_prefix="models/sp",
    vocab_size=8000,
    model_type="bpe",
)
# Creates: models/sp.model, models/sp.vocab
```

### Using a Model

```python
from torchlingo.data_processing import SentencePieceVocab

sp = SentencePieceVocab("models/sp.model")

# Tokenize
tokens = sp.tokenize("Hello world")
print(tokens)  # ['▁Hello', '▁world']

# Encode to indices
indices = sp.encode("Hello world", add_special_tokens=True)
print(indices)  # [2, 145, 892, 3]

# Decode back
text = sp.decode(indices, skip_special_tokens=True)
print(text)  # "Hello world"
```

## API Reference

::: torchlingo.preprocessing.sentencepiece.train_sentencepiece
    options:
      show_source: true

## Training Options

### Model Types

| Type | Description | Use Case |
| ---- | ----------- | -------- |
| `bpe` | Byte Pair Encoding | General purpose, most common |
| `unigram` | Unigram LM | Better for some Asian languages |
| `char` | Character-level | When subwords aren't needed |
| `word` | Word-level | Pre-tokenized data |

```python
# BPE (default, recommended)
train_sentencepiece(input_files, model_prefix, model_type="bpe")

# Unigram
train_sentencepiece(input_files, model_prefix, model_type="unigram")
```

### Vocabulary Size

```python
# Small vocab (faster, less coverage)
train_sentencepiece(input_files, model_prefix, vocab_size=4000)

# Medium vocab (balanced)
train_sentencepiece(input_files, model_prefix, vocab_size=16000)

# Large vocab (better coverage, more parameters)
train_sentencepiece(input_files, model_prefix, vocab_size=32000)
```

**Rule of thumb**:

- Small datasets: 4K-8K
- Medium datasets: 8K-16K
- Large datasets: 16K-32K

### Character Coverage

```python
# Full coverage (all characters in vocabulary)
train_sentencepiece(input_files, model_prefix, character_coverage=1.0)

# Allow rare characters to be UNK (useful for noisy data)
train_sentencepiece(input_files, model_prefix, character_coverage=0.9995)
```

For CJK languages (Chinese, Japanese, Korean), use `0.9995` since they have many characters.

### Normalization

```python
# NMT-NFKC (default, good for translation)
train_sentencepiece(
    input_files, model_prefix,
    normalization_rule_name="nmt_nfkc"
)

# Other options
train_sentencepiece(input_files, model_prefix, normalization_rule_name="nfc")
train_sentencepiece(input_files, model_prefix, normalization_rule_name="identity")
```

## Examples

### Complete Training Pipeline

```python
from pathlib import Path
from torchlingo.preprocessing import (
    load_data,
    save_data,
    train_sentencepiece,
)

# Prepare training text (combine src and tgt for shared vocab)
df = load_data("data/train.tsv")
train_text = df['src'].tolist() + df['tgt'].tolist()

# Save as text file for SentencePiece
text_path = Path("data/train_combined.txt")
with open(text_path, "w") as f:
    f.write("\n".join(train_text))

# Train SentencePiece
train_sentencepiece(
    input_files=[str(text_path)],
    model_prefix="models/shared_sp",
    vocab_size=16000,
    model_type="bpe",
)
```

### Separate Source/Target Models

```python
# Source language model
with open("data/train_src.txt", "w") as f:
    f.write("\n".join(df['src']))
    
train_sentencepiece(
    ["data/train_src.txt"],
    "models/sp_en",
    vocab_size=8000,
)

# Target language model
with open("data/train_tgt.txt", "w") as f:
    f.write("\n".join(df['tgt']))
    
train_sentencepiece(
    ["data/train_tgt.txt"],
    "models/sp_es",
    vocab_size=8000,
)
```

### Using with NMTDataset

```python
from torchlingo.data_processing import NMTDataset, SentencePieceVocab

# Load trained models
src_sp = SentencePieceVocab("models/sp_en.model")
tgt_sp = SentencePieceVocab("models/sp_es.model")

# Use with dataset
dataset = NMTDataset(
    "data/train.tsv",
    src_vocab=src_sp,
    tgt_vocab=tgt_sp,
)
```

## Configuration via Config

```python
from torchlingo.config import Config

config = Config(
    use_sentencepiece=True,
    sentencepiece_model_prefix="models/sp",
    vocab_size=16000,
    sp_model_type="bpe",
    sp_character_coverage=1.0,
    sp_normalization_rule_name="nmt_nfkc",
)
```

## Best Practices

### 1. Train on Training Data Only

```python
# ✅ Good: Train on training data
train_sentencepiece(["data/train.txt"], model_prefix)

# ❌ Bad: Including test data
train_sentencepiece(["data/train.txt", "data/test.txt"], model_prefix)
```

### 2. Combine Source and Target for Shared Vocab

For similar languages (e.g., Spanish-Portuguese), a shared vocabulary works well:

```python
combined = df['src'].tolist() + df['tgt'].tolist()
```

### 3. Use Appropriate Vocab Size

| Dataset Size | Recommended Vocab |
| ------------ | ----------------- |
| < 100K sentences | 4K-8K |
| 100K-1M sentences | 8K-16K |
| > 1M sentences | 16K-32K |

### 4. Inspect the Model

```python
import sentencepiece as spm

sp = spm.SentencePieceProcessor()
sp.load("models/sp.model")

# Check vocabulary
print(f"Vocab size: {sp.get_piece_size()}")

# See how text is tokenized
text = "The quick brown fox"
pieces = sp.encode_as_pieces(text)
print(f"Pieces: {pieces}")

ids = sp.encode_as_ids(text)
print(f"IDs: {ids}")
```

## Troubleshooting

??? question "Model produces too many UNK tokens"
    Increase `character_coverage`:
    ```python
    train_sentencepiece(..., character_coverage=1.0)
    ```

??? question "Vocabulary is too small/large"
    Adjust `vocab_size`:
    ```python
    train_sentencepiece(..., vocab_size=16000)
    ```

??? question "Training is slow"
    For large datasets, use sampling:
    ```python
    # SentencePiece automatically samples if data is large
    train_sentencepiece(..., input_sentence_size=10000000)
    ```
