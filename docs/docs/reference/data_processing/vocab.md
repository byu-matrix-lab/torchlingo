# Vocabulary

Vocabulary classes for encoding and decoding text in neural machine translation.
All implementations share the `BaseVocab` interface so they can be swapped
without changing calling code.

## Overview

TorchLingo provides a common interface plus multiple implementations:

| Class | Use Case |
| ----- | -------- |
| `BaseVocab` | Abstract interface for encoding/decoding |
| `SimpleVocab` | Simple whitespace tokenization, good for small datasets |
| `SentencePieceVocab` | Subword tokenization, handles rare words better |
| `MeCabVocab` | Japanese morphological analysis (requires `fugashi`) |
| `JiebaVocab` | Chinese word segmentation (requires `jieba`) |

All vocab classes implement the `BaseVocab` contract, so any function annotated with
`BaseVocab` can accept any implementation.

## Quick Start

```python
from torchlingo.data_processing import (
    BaseVocab,
    SimpleVocab,
    SentencePieceVocab,
    MeCabVocab,
    JiebaVocab,
)

# Simple vocabulary
vocab: BaseVocab = SimpleVocab()
vocab.build_vocab(["Hello world", "How are you"])
indices = vocab.encode("Hello world", add_special_tokens=True)
# [2, 4, 5, 3]  (SOS, Hello, world, EOS)

# SentencePiece vocabulary
sp_vocab = SentencePieceVocab("models/sp.model")
indices = sp_vocab.encode("Hello world", add_special_tokens=True)

# Japanese vocabulary (requires: pip install torchlingo[japanese])
jp_vocab = MeCabVocab(min_freq=1)
jp_vocab.build_vocab(["私は学生です", "彼は先生です"])
indices = jp_vocab.encode("私は学生です")

# Chinese vocabulary (requires: pip install torchlingo[chinese])
zh_vocab = JiebaVocab(min_freq=1)
zh_vocab.build_vocab(["我是学生", "他是老师"])
indices = zh_vocab.encode("我是学生")
```

## API Reference

### BaseVocab

::: torchlingo.data_processing.vocab.BaseVocab
        options:
            show_source: true
            members:
                - __len__
                - build_vocab
                - token_to_idx
                - idx_to_token
                - tokens_to_indices
                - indices_to_tokens
                - encode
                - decode

### SimpleVocab

::: torchlingo.data_processing.vocab.SimpleVocab
    options:
      show_source: true
      members:
        - __init__
        - build_vocab
        - __len__
        - token_to_idx
        - idx_to_token
        - tokens_to_indices
        - indices_to_tokens
        - encode
        - decode

### SentencePieceVocab

::: torchlingo.data_processing.vocab.SentencePieceVocab
    options:
      show_source: true
      members:
        - __init__
        - __len__
        - tokenize
        - detokenize
        - encode
        - decode

### MeCabVocab

::: torchlingo.data_processing.vocab.MeCabVocab
    options:
      show_source: true
      members:
        - __init__
        - build_vocab
        - __len__
        - token_to_idx
        - idx_to_token
        - encode
        - decode

### JiebaVocab

::: torchlingo.data_processing.vocab.JiebaVocab
    options:
      show_source: true
      members:
        - __init__
        - build_vocab
        - __len__
        - token_to_idx
        - idx_to_token
        - encode
        - decode

## Examples

### Building a Vocabulary

```python
from torchlingo.data_processing import SimpleVocab

vocab = SimpleVocab(min_freq=2)  # Ignore words appearing < 2 times

sentences = [
    "The cat sat on the mat",
    "The dog ran in the park",
    "A bird flew over the house",
]
vocab.build_vocab(sentences)

print(f"Vocabulary size: {len(vocab)}")
print(f"Token mapping: {vocab.token2idx}")
```

### Encoding and Decoding

```python
# Encode with special tokens
indices = vocab.encode("The cat", add_special_tokens=True)
print(indices)  # [2, 4, 5, 3]  (SOS, The, cat, EOS)

# Decode back to text
text = vocab.decode(indices, skip_special_tokens=True)
print(text)  # "The cat"
```

### Custom Special Tokens

```python
vocab = SimpleVocab(
    pad_token="[PAD]",
    unk_token="[UNK]",
    sos_token="[BOS]",
    eos_token="[EOS]",
    pad_idx=0,
    unk_idx=1,
    sos_idx=2,
    eos_idx=3,
)
```

### Using SentencePiece

```python
from torchlingo.data_processing import SentencePieceVocab

# Load a pre-trained model
sp = SentencePieceVocab("data/sp_model.model")

# Tokenize (see subwords)
tokens = sp.tokenize("unfortunately")
print(tokens)  # ['▁un', 'fortun', 'ately']

# Encode to indices
indices = sp.encode("unfortunately", add_special_tokens=True)

# Decode back
text = sp.decode(indices, skip_special_tokens=True)
print(text)  # "unfortunately"
```

## Special Tokens

Both vocabulary types reserve these special tokens:

| Token | Default Index | Purpose |
| ----- | ------------- | ------- |
| `<pad>` | 0 | Padding for batching |
| `<unk>` | 1 | Unknown/out-of-vocabulary words |
| `<sos>` | 2 | Start of sequence marker |
| `<eos>` | 3 | End of sequence marker |

## Vocabulary Comparison

| Feature | SimpleVocab | SentencePieceVocab | MeCabVocab | JiebaVocab |
| ------- | ----- | ------------------ | ---------- | ---------- |
| Tokenization | Whitespace | BPE/Unigram subwords | Japanese morphemes | Chinese words |
| Unknown words | Maps to `<unk>` | Splits into subwords | Maps to `<unk>` | Maps to `<unk>` |
| Vocab size | Grows with data | Fixed at training | Grows with data | Grows with data |
| Training required | No | Yes (separate step) | No | No |
| Dependencies | None | `sentencepiece` | `fugashi`, `unidic-lite` | `jieba` |
| Best for | Small datasets, prototyping | Production, large datasets | Japanese text | Chinese text |
| Languages | Space-separated | Any | Japanese | Chinese, other CJK |

## Frequency Filtering

`SimpleVocab` supports minimum frequency filtering:

```python
vocab = SimpleVocab(min_freq=5)  # Words must appear 5+ times
vocab.build_vocab(sentences)

# Rare words are mapped to UNK
rare_word_idx = vocab.token_to_idx("supercalifragilistic")
print(rare_word_idx == vocab.unk_idx)  # True
```

## Saving and Loading

```python
import pickle

# Save
with open("vocab.pkl", "wb") as f:
    pickle.dump(vocab, f)

# Load
with open("vocab.pkl", "rb") as f:
    vocab = pickle.load(f)
```

For `SentencePieceVocab`, the model file itself is the saved state:

```python
# Just keep the .model file
sp = SentencePieceVocab("models/sp.model")
```

## Asian Language Examples

### Japanese with MeCabVocab

```python
from torchlingo.data_processing import MeCabVocab

# Create and build vocabulary
vocab = MeCabVocab(min_freq=1)
sentences = [
    "私は学生です",  # "I am a student"
    "彼は先生です",  # "He is a teacher"
]
vocab.build_vocab(sentences)

# Encode (morphemes are automatically detected)
indices = vocab.encode("私は学生です", add_special_tokens=True)

# Decode (reconstructs without spaces)
text = vocab.decode(indices, skip_special_tokens=True)
print(text)  # "私は学生です"
```

### Chinese with JiebaVocab

```python
from torchlingo.data_processing import JiebaVocab

# Create and build vocabulary
vocab = JiebaVocab(min_freq=1)
sentences = [
    "我是学生",  # "I am a student"
    "他是老师",  # "He is a teacher"
]
vocab.build_vocab(sentences)

# Encode (words are automatically segmented)
indices = vocab.encode("我是学生", add_special_tokens=True)

# Decode (reconstructs without spaces)
text = vocab.decode(indices, skip_special_tokens=True)
print(text)  # "我是学生"

# Use accurate mode (default) or full mode
vocab_full = JiebaVocab(cut_all=True)  # Full segmentation mode
```

### Installing Asian Language Support

```bash
# Japanese only
pip install torchlingo[japanese]

# Chinese only
pip install torchlingo[chinese]

# Both Japanese and Chinese
pip install torchlingo[asian]
```
