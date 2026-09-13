# LSTM

Simple LSTM-based sequence-to-sequence model for neural machine translation.

## Overview

`SimpleSeq2SeqLSTM` implements a classic encoder-decoder architecture using LSTM (Long Short-Term Memory) cells. The encoder reads the source sequence and compresses it into a context vector, which initializes the decoder to generate the target sequence.

## Architecture

```
┌─────────────────────────────────────────────────┐
│                    ENCODER                      │
│  ┌───────────────────────────────────────────┐  │
│  │         Token Embedding                   │  │
│  │   src_vocab_size → emb_dim                │  │
│  └───────────────────────────────────────────┘  │
│                      ↓                          │
│  ┌───────────────────────────────────────────┐  │
│  │         LSTM Layers × N                   │  │
│  │   Process sequence step by step           │  │
│  │   Output: (hidden_state, cell_state)      │  │
│  └───────────────────────────────────────────┘  │
└─────────────────────────────────────────────────┘
                       ↓
           (final_hidden, final_cell)
                       ↓
┌─────────────────────────────────────────────────┐
│                    DECODER                      │
│  ┌───────────────────────────────────────────┐  │
│  │         Token Embedding                   │  │
│  │   tgt_vocab_size → emb_dim                │  │
│  └───────────────────────────────────────────┘  │
│                      ↓                          │
│  ┌───────────────────────────────────────────┐  │
│  │         LSTM Layers × N                   │  │
│  │   Initialized with encoder states         │  │
│  │   Process target sequence                 │  │
│  └───────────────────────────────────────────┘  │
│                      ↓                          │
│  ┌───────────────────────────────────────────┐  │
│  │         Linear Output                     │  │
│  │   hidden_dim → tgt_vocab_size             │  │
│  └───────────────────────────────────────────┘  │
└─────────────────────────────────────────────────┘
```

## Quick Start

```python
from torchlingo.models import SimpleSeq2SeqLSTM

model = SimpleSeq2SeqLSTM(
    src_vocab_size=10000,
    tgt_vocab_size=10000,
    emb_dim=256,
    hidden_dim=512,
    num_layers=2,
)

# Training
logits = model(src_batch, tgt_batch)  # [batch, tgt_len, vocab]
```

## API Reference

::: torchlingo.models.lstm_simple.SimpleSeq2SeqLSTM
    options:
      show_source: true
      members:
        - __init__
        - forward

## Constructor Parameters

| Parameter | Type | Default | Description |
| --------- | ---- | ------- | ----------- |
| `src_vocab_size` | int | *required* | Source vocabulary size |
| `tgt_vocab_size` | int | *required* | Target vocabulary size |
| `emb_dim` | int | 256 | Embedding dimension |
| `hidden_dim` | int | 512 | LSTM hidden dimension |
| `num_layers` | int | 2 | Number of stacked LSTM layers |
| `dropout` | float | 0.1 | Dropout between LSTM layers |
| `pad_idx` | int | 0 | Padding token index |
| `attention` | bool | False | Let the decoder attend over encoder outputs |
| `attn_type` | str | `"dot"` | Scorer: `"dot"` (Luong) or `"additive"` (Bahdanau) |
| `config` | Config | None | Configuration object |

## Examples

### Basic Training

```python
import torch
from torchlingo.models import SimpleSeq2SeqLSTM

model = SimpleSeq2SeqLSTM(
    src_vocab_size=10000,
    tgt_vocab_size=10000,
    emb_dim=256,
    hidden_dim=512,
)

# Dummy data
src = torch.randint(0, 10000, (32, 20))  # [batch, src_len]
tgt = torch.randint(0, 10000, (32, 25))  # [batch, tgt_len]

# Forward pass
logits = model(src, tgt[:, :-1])  # [32, 24, 10000]

# Compute loss
criterion = torch.nn.CrossEntropyLoss(ignore_index=0)
loss = criterion(
    logits.reshape(-1, logits.size(-1)),
    tgt[:, 1:].reshape(-1)
)
```

### With Config

```python
from torchlingo.config import Config

config = Config(
    lstm_emb_dim=256,
    lstm_hidden_dim=512,
    lstm_num_layers=3,
    lstm_dropout=0.2,
)

model = SimpleSeq2SeqLSTM(
    src_vocab_size=10000,
    tgt_vocab_size=10000,
    config=config,
)
```

### Inference (Greedy and Beam Search)

Use the library's decoders rather than writing the loop yourself. Both accept an LSTM
model:

```python
from torchlingo.inference import beam_search_decode, greedy_decode

decoded = greedy_decode(model, src_batch, max_len=50)      # list[list[int]]
tokens = beam_search_decode(model, src, beam_size=5)       # one sentence
```

The search is the same code that decodes a Transformer — see
[Decoding](../../concepts/decoding.md). Three methods make that possible:

| Method | Purpose |
| ------ | ------- |
| `encode_source(src)` | Returns `(enc_out, hidden, src_pad_mask)` — everything the decoder may need |
| `decode_prefix(tgt, hidden, enc_out, mask)` | Scores a whole target prefix; the counterpart to a Transformer's `decode(tgt, memory)` |
| `decode_step(token, hidden, enc_out, mask)` | Advances one token, carrying state forward |

If you do want to step manually — to inspect attention at each step, say — use
`encode_source` and `decode_step` so the encoder outputs actually reach the
decoder:

```python
enc_out, hidden, src_pad_mask = model.encode_source(src)
ys = [tgt_vocab.sos_idx]
alignments = []

for _ in range(max_len):
    last = torch.tensor([ys[-1:]], device=src.device)
    logits, hidden, weights = model.decode_step(last, hidden, enc_out, src_pad_mask)
    alignments.append(weights)          # None when attention is disabled
    next_token = logits[:, -1, :].argmax().item()
    ys.append(next_token)
    if next_token == tgt_vocab.eos_idx:
        break
```

!!! warning "Don't re-implement the decoder loop"
    Driving `model.decoder` directly from the encoder's final `(h, c)` — as
    earlier versions of this page showed — throws away the per-token encoder
    outputs. On an attention model that silently disables attention, and the
    model will appear to work while producing worse translations.

## How LSTMs Work

### The Information Bottleneck

The encoder compresses the entire source sequence into a fixed-size vector (the final hidden state). This becomes the "context" for the decoder.

```
"I love cats" → [Encode] → hidden_vector → [Decode] → "Me gustan los gatos"
```

**Limitation**: Long sequences can be hard to compress into a single vector.

With `attention=True`, the decoder additionally reads *every* encoder output,
weighted per step, so the sentence no longer has to survive that squeeze. Run
`python examples/attention_alignment.py` to see the difference measured on a
task with a known correct alignment.

### Hidden and Cell States

LSTMs maintain two types of state:

- **Hidden state (h)**: Short-term memory, used for output
- **Cell state (c)**: Long-term memory, carries information across time steps

```python
# After encoding
# h: [num_layers, batch, hidden_dim]
# c: [num_layers, batch, hidden_dim]
```

### Stacked Layers

Multiple LSTM layers create a deeper network:

```
Layer 3: Higher-level patterns
    ↑
Layer 2: Intermediate features  
    ↑
Layer 1: Low-level features
    ↑
Input embeddings
```

## LSTM vs Transformer

| Aspect | LSTM | Transformer |
| ------ | ---- | ----------- |
| Processing | Sequential | Parallel |
| Long dependencies | Difficult; easier with `attention=True` | Easy (attention) |
| Training speed | Slower | Faster |
| Memory efficiency | O(n) | O(n²) |
| Simplicity | Simpler | More complex |
| Parameters | Fewer | More |

### When to Use LSTM

- ✅ Small datasets (< 50K examples)
- ✅ Limited GPU memory
- ✅ Learning/educational purposes
- ✅ Real-time inference on CPU

### When to Use Transformer

- ✅ Large datasets
- ✅ Best translation quality
- ✅ GPU available for training
- ✅ Long sequences

## Model Variants

### Small

```python
model = SimpleSeq2SeqLSTM(
    src_vocab_size, tgt_vocab_size,
    emb_dim=128,
    hidden_dim=256,
    num_layers=1,
)
# ~3M params
```

### Medium

```python
model = SimpleSeq2SeqLSTM(
    src_vocab_size, tgt_vocab_size,
    emb_dim=256,
    hidden_dim=512,
    num_layers=2,
)
# ~15M params
```

### Large

```python
model = SimpleSeq2SeqLSTM(
    src_vocab_size, tgt_vocab_size,
    emb_dim=512,
    hidden_dim=1024,
    num_layers=4,
)
# ~50M params
```

## Limitations

1. **Information bottleneck** (default): with `attention=False`, all source information must fit in the final hidden state
2. **Sequential processing**: can't parallelize across time steps

!!! tip "Adding Attention"
    Pass `attention=True` to let the decoder look back at every encoder output
    instead of relying on the final hidden state alone. Both classic scorers are
    implemented — see [Attention](attention.md). Attention is **off by default**,
    so the bottlenecked model remains the baseline you compare against.
