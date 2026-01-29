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

### Inference (Greedy Decoding)

```python
def greedy_decode_lstm(model, src, tgt_vocab, max_len=50, device="cpu"):
    model.eval()
    src = src.to(device).unsqueeze(0)  # Add batch dim
    
    # Encode
    src_emb = model.src_embed(src)
    _, (h, c) = model.encoder(src_emb)
    
    # Decode
    ys = [tgt_vocab.sos_idx]
    
    for _ in range(max_len):
        tgt_tensor = torch.tensor([ys[-1:]], device=device)
        tgt_emb = model.tgt_embed(tgt_tensor)
        output, (h, c) = model.decoder(tgt_emb, (h, c))
        logits = model.output(output)
        next_token = logits[:, -1, :].argmax().item()
        ys.append(next_token)
        
        if next_token == tgt_vocab.eos_idx:
            break
    
    return ys
```

## How LSTMs Work

### The Information Bottleneck

The encoder compresses the entire source sequence into a fixed-size vector (the final hidden state). This becomes the "context" for the decoder.

```
"I love cats" → [Encode] → hidden_vector → [Decode] → "Me gustan los gatos"
```

**Limitation**: Long sequences can be hard to compress into a single vector.

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
| Long dependencies | Difficult | Easy (attention) |
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

1. **No attention**: This simple model doesn't use attention, so it struggles with long sequences
2. **Information bottleneck**: All source information must fit in the hidden state
3. **Sequential processing**: Can't parallelize across time steps

!!! tip "Adding Attention"
    For production LSTM models, consider adding Bahdanau or Luong attention to allow the decoder to look back at the encoder outputs. This is not implemented in TorchLingo's simple model.
