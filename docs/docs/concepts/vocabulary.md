# Vocabulary & Tokenization

How do we convert human language into numbers that neural networks can process? This page covers vocabulary building and tokenization strategies.

## The Challenge

Neural networks compute with numbers, but we work with text:

```
"Hello world" → [???] → Neural Network → [???] → "Hola mundo"
```

We need a systematic way to convert between text and numbers.

## Simple Tokenization

The simplest approach: split on whitespace and assign numbers.

### Word-Level Tokenization

```python
sentence = "Hello world, how are you?"
tokens = sentence.split()
# ['Hello', 'world,', 'how', 'are', 'you?']
```

!!! warning "Punctuation Problem"
    Notice "world," and "you?" include punctuation. This means "world" and "world," are different tokens!

### Building a Vocabulary

TorchLingo's `SimpleVocab` class handles this:

```python
from torchlingo.data_processing import SimpleVocab

vocab = SimpleVocab()

# Build from training sentences
sentences = [
    "Hello world",
    "How are you",
    "Hello friend",
    "World peace",
]
vocab.build_vocab(sentences)

print(f"Vocabulary: {vocab.token2idx}")
```

```
{
    '<pad>': 0, '<unk>': 1, '<sos>': 2, '<eos>': 3,
    'Hello': 4, 'world': 5, 'How': 6, 'are': 7,
    'you': 8, 'friend': 9, 'World': 10, 'peace': 11
}
```

### Encoding and Decoding

```python
# Text → Numbers
indices = vocab.encode("Hello world", add_special_tokens=True)
# [2, 4, 5, 3]  (SOS, Hello, world, EOS)

# Numbers → Text
text = vocab.decode([2, 4, 5, 3], skip_special_tokens=True)
# "Hello world"
```

## The Unknown Word Problem

What happens with words not in our vocabulary?

```python
# "goodbye" wasn't in training data
indices = vocab.encode("Goodbye world")
# [1, 5]  (UNK, world)
```

The `<unk>` token (index 1) replaces unknown words. This loses information!

### Frequency Filtering

We can ignore rare words (they're often typos or noise):

```python
from torchlingo.config import Config

config = Config(min_freq=2)  # Require 2+ occurrences
vocab = SimpleVocab(config=config)
vocab.build_vocab(sentences)
```

Only words appearing ≥2 times make it into the vocabulary.

## Subword Tokenization

Modern NMT uses **subword tokenization** to handle the unknown word problem.

### The Idea

Instead of whole words, break text into smaller pieces:

```
"unfortunately" → ["un", "fortunate", "ly"]
"preprocessing" → ["pre", "process", "ing"]
```

Benefits:

- ✅ No unknown words (any word can be broken down)
- ✅ Smaller vocabularies
- ✅ Shared representations ("walk", "walking", "walked" share "walk")

### SentencePiece

TorchLingo supports [SentencePiece](https://github.com/google/sentencepiece), a popular subword tokenizer:

```python
from torchlingo.data_processing import SentencePieceVocab

# Load a pre-trained model
sp_vocab = SentencePieceVocab(model_path="data/sp_model.model")

# Tokenize
tokens = sp_vocab.tokenize("unfortunately")
# ['▁un', 'fortun', 'ately']  (▁ marks word boundaries)

# Encode
indices = sp_vocab.encode("unfortunately", add_special_tokens=True)
```

### Training SentencePiece

To train your own SentencePiece model:

```python
from torchlingo.preprocessing import train_sentencepiece

train_sentencepiece(
    input_files=["data/train.txt"],
    model_prefix="data/sp_model",
    vocab_size=8000,        # Target vocabulary size
    model_type="bpe",       # BPE or unigram
)
```

Algorithms:

| Type      | Description                                  |
| --------- | -------------------------------------------- |
| **BPE**   | Byte Pair Encoding - merges frequent pairs   |
| **Unigram** | Probabilistic - maximizes likelihood       |
| **char**  | Character-level                              |
| **word**  | Word-level (no subword splitting)            |

## Comparing Approaches

| Approach    | Vocab Size | Unknown Words | Example                    |
| ----------- | ---------- | ------------- | -------------------------- |
| Word        | Large      | Common        | ["Hello", "world"]         |
| BPE         | Medium     | Rare          | ["Hel", "lo", "▁world"]    |
| Character   | Small (~100) | None        | ["H","e","l","l","o"," ","w","o","r","l","d"] |

!!! tip "Rule of Thumb"
    - Small datasets (<100K sentences): Word-level with min_freq
    - Medium datasets: BPE with 8K-16K vocabulary
    - Large datasets: BPE with 32K-64K vocabulary

## Special Tokens Deep Dive

### The Four Core Tokens

```python
# TorchLingo defaults
PAD_TOKEN = "<pad>"  # Index 0
UNK_TOKEN = "<unk>"  # Index 1  
SOS_TOKEN = "<sos>"  # Index 2
EOS_TOKEN = "<eos>"  # Index 3
```

### PAD (Padding)

Batches need uniform tensor shapes. PAD fills shorter sequences:

```
Batch before padding:
  "Hello world"      → [SOS, 4, 5, EOS]
  "How are you today" → [SOS, 6, 7, 8, 9, EOS]

Batch after padding:
  [SOS, 4,   5, PAD, PAD, EOS]
  [SOS, 6,   7,   8,   9, EOS]
```

The model learns to ignore PAD tokens.

### UNK (Unknown)

Replaces out-of-vocabulary words:

```python
# "pizza" not in vocabulary
vocab.encode("I love pizza")
# [SOS, I, love, UNK, EOS]
```

### SOS (Start of Sequence)

Tells the decoder to start generating:

```
Decoder input:  [SOS]
Decoder output: "El"

Decoder input:  [SOS, El]
Decoder output: "gato"
```

### EOS (End of Sequence)

Signals the end of a sequence:

- During training: marks where the target ends
- During inference: tells the model to stop generating

```python
# Stop when EOS is generated
while predicted_token != vocab.eos_idx:
    predicted_token = decode_next()
```

## Vocabulary Configuration

### Via Config Object

```python
from torchlingo.config import Config

config = Config(
    # Vocabulary settings
    min_freq=2,
    vocab_size=32000,  # For SentencePiece
    
    # Special tokens
    pad_token="<pad>",
    unk_token="<unk>",
    sos_token="<sos>",
    eos_token="<eos>",
    
    # Token indices
    pad_idx=0,
    unk_idx=1,
    sos_idx=2,
    eos_idx=3,
)
```

### Per-Vocab Override

```python
vocab = SimpleVocab(
    min_freq=5,
    pad_token="[PAD]",  # Custom token strings
    unk_token="[UNK]",
)
```

## Shared vs. Separate Vocabularies

### Separate (Default)

Source and target languages have their own vocabularies:

```python
dataset = NMTDataset("train.tsv")
# dataset.src_vocab - English vocabulary
# dataset.tgt_vocab - Spanish vocabulary
```

Best for very different languages (e.g., English↔Chinese).

### Shared

One vocabulary for both languages:

```python
# Build combined vocabulary
all_sentences = src_sentences + tgt_sentences
shared_vocab = SimpleVocab()
shared_vocab.build_vocab(all_sentences)

# Use for both
dataset = NMTDataset(
    "train.tsv",
    src_vocab=shared_vocab,
    tgt_vocab=shared_vocab,
)
```

Best for:

- Similar languages (Spanish↔Portuguese)
- Multilingual models
- When source and target share many words

## Practical Tips

### Inspecting Your Vocabulary

```python
# Most common words
from collections import Counter

counter = Counter(vocab.token_freqs)
print(counter.most_common(20))

# Check vocabulary coverage
test_sentence = "This is a test sentence"
tokens = test_sentence.split()
known = sum(1 for t in tokens if t in vocab.token2idx)
print(f"Coverage: {known}/{len(tokens)} = {known/len(tokens):.1%}")
```

### Saving and Loading

```python
import pickle

# Save
with open("vocab.pkl", "wb") as f:
    pickle.dump(vocab, f)

# Load
with open("vocab.pkl", "rb") as f:
    vocab = pickle.load(f)
```

### Handling Case

The default vocabulary is **case-sensitive** ("Hello" ≠ "hello").

For case-insensitive vocabulary:

```python
# Preprocess your data
sentences = [s.lower() for s in sentences]
vocab.build_vocab(sentences)
```

## SentencePiece Configuration

For more control over SentencePiece:

```python
from torchlingo.config import Config

config = Config(
    use_sentencepiece=True,
    sentencepiece_model_prefix="data/sp_model",
    vocab_size=16000,
    sp_model_type="bpe",  # "bpe", "unigram", "char", "word"
    sp_character_coverage=1.0,  # 0.9995 for CJK languages
)
```

## Asian Language Support

Languages like Japanese, Chinese, and Korean don't use spaces between words. TorchLingo provides specialized vocabularies for these languages.

### The Problem with Whitespace Tokenization

```python
# English: spaces separate words
"Hello world".split()  # → ["Hello", "world"] ✓

# Japanese: no spaces!
"私は学生です".split()  # → ["私は学生です"] ✗ (whole sentence as one token)

# Chinese: also no spaces!
"我是学生".split()  # → ["我是学生"] ✗
```

### Japanese with MeCabVocab

TorchLingo uses [MeCab](https://taku910.github.io/mecab/) (via [fugashi](https://github.com/polm/fugashi)) for Japanese morphological analysis:

```python
from torchlingo.data_processing import MeCabVocab

# Create vocabulary
vocab = MeCabVocab(min_freq=1)

# Build from Japanese sentences
sentences = [
    "私は学生です",    # "I am a student"
    "彼は先生です",    # "He is a teacher"
]
vocab.build_vocab(sentences)

# Encode
indices = vocab.encode("私は学生です")

# Decode (reconstructs without spaces - Japanese convention)
text = vocab.decode(indices)  # "私は学生です"
```

!!! info "Installation"
    MeCabVocab requires additional dependencies:
    ```bash
    pip install torchlingo[japanese]
    # or
    pip install fugashi[unidic-lite]
    ```

### Chinese with JiebaVocab

TorchLingo uses [jieba](https://github.com/fxsjy/jieba) for Chinese word segmentation:

```python
from torchlingo.data_processing import JiebaVocab

# Create vocabulary
vocab = JiebaVocab(min_freq=1)

# Build from Chinese sentences
sentences = [
    "我是学生",  # "I am a student"
    "他是老师",  # "He is a teacher"
]
vocab.build_vocab(sentences)

# Encode
indices = vocab.encode("我是学生")

# Decode (reconstructs without spaces - Chinese convention)
text = vocab.decode(indices)  # "我是学生"
```

!!! info "Installation"
    JiebaVocab requires jieba:
    ```bash
    pip install torchlingo[chinese]
    # or
    pip install jieba
    ```

### JiebaVocab Modes

Jieba offers different segmentation modes:

```python
# Accurate mode (default) - recommended for NMT
vocab = JiebaVocab(cut_all=False)

# Full mode - finds all possible words
vocab = JiebaVocab(cut_all=True)

# Paddle mode - uses deep learning for better accuracy
# Requires: pip install paddlepaddle-tiny
vocab = JiebaVocab(use_paddle=True)
```

### Comparing Asian Language Approaches

| Language | Vocabulary Class | Tokenizer | Example |
|----------|-----------------|-----------|---------|
| Japanese | `MeCabVocab` | MeCab/fugashi | "私は学生" → ["私", "は", "学生"] |
| Chinese | `JiebaVocab` | jieba | "我是学生" → ["我", "是", "学生"] |
| Korean | `SentencePieceVocab` | SentencePiece | Use BPE/Unigram model |

!!! tip "Korean"
    For Korean, we recommend using `SentencePieceVocab` with a BPE or Unigram model trained on Korean text. Korean has spaces between some words but not consistently, making subword tokenization a good choice.

### Installing All Asian Language Support

To install support for all Asian languages at once:

```bash
pip install torchlingo[asian]
```

This installs both fugashi (for Japanese) and jieba (for Chinese).

## Next Steps

Now that you understand vocabularies, learn about the models that use them:

[Model Architectures :material-arrow-right:](models.md){ .md-button .md-button--primary }
