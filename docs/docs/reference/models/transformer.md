# Transformer

Simple transformer-based sequence-to-sequence model with RoPE embeddings.

## Overview

`SimpleTransformer` implements the encoder-decoder Transformer architecture from "Attention Is All You Need" (Vaswani et al., 2017), enhanced with Rotary Position Embeddings (RoPE) for better position encoding.

## Architecture

```
┌─────────────────────────────────────────────────┐
│                    ENCODER                       │
│  ┌───────────────────────────────────────────┐  │
│  │   Token Embedding + RoPE Position         │  │
│  │   src_vocab_size → d_model                │  │
│  └───────────────────────────────────────────┘  │
│                      ↓                           │
│  ┌───────────────────────────────────────────┐  │
│  │         Transformer Encoder × N           │  │
│  │   • Multi-Head Self-Attention             │  │
│  │   • Feed-Forward Network                  │  │
│  │   • LayerNorm + Residual                  │  │
│  └───────────────────────────────────────────┘  │
└─────────────────────────────────────────────────┘
                       ↓
              Encoder Output (Memory)
                       ↓
┌─────────────────────────────────────────────────┐
│                    DECODER                       │
│  ┌───────────────────────────────────────────┐  │
│  │   Token Embedding + RoPE Position         │  │
│  │   tgt_vocab_size → d_model                │  │
│  └───────────────────────────────────────────┘  │
│                      ↓                           │
│  ┌───────────────────────────────────────────┐  │
│  │         Transformer Decoder × N           │  │
│  │   • Masked Multi-Head Self-Attention      │  │
│  │   • Cross-Attention (to encoder)          │  │
│  │   • Feed-Forward Network                  │  │
│  │   • LayerNorm + Residual                  │  │
│  └───────────────────────────────────────────┘  │
│                      ↓                           │
│  ┌───────────────────────────────────────────┐  │
│  │         Linear Generator                  │  │
│  │   d_model → tgt_vocab_size                │  │
│  └───────────────────────────────────────────┘  │
└─────────────────────────────────────────────────┘
```

## Quick Start

```python
from torchlingo.models import SimpleTransformer

model = SimpleTransformer(
    src_vocab_size=10000,
    tgt_vocab_size=10000,
    d_model=512,
    n_heads=8,
    num_encoder_layers=6,
    num_decoder_layers=6,
)

# Training
logits = model(src_batch, tgt_batch)  # [batch, tgt_len, vocab]

# Inference
memory = model.encode(src_batch)
logits = model.decode(tgt_batch, memory)
```

## API Reference

::: torchlingo.models.transformer_simple.SimpleTransformer
    options:
      show_source: true
      members:
        - __init__
        - forward
        - encode
        - decode

## Constructor Parameters

| Parameter | Type | Default | Description |
| --------- | ---- | ------- | ----------- |
| `src_vocab_size` | int | *required* | Source vocabulary size |
| `tgt_vocab_size` | int | *required* | Target vocabulary size |
| `d_model` | int | 512 | Model hidden dimension |
| `n_heads` | int | 8 | Number of attention heads |
| `num_encoder_layers` | int | 6 | Encoder transformer blocks |
| `num_decoder_layers` | int | 6 | Decoder transformer blocks |
| `d_ff` | int | 2048 | Feed-forward inner dimension |
| `max_seq_length` | int | 512 | Maximum sequence length |
| `dropout` | float | 0.1 | Dropout rate |
| `pad_idx` | int | 0 | Padding token index |
| `config` | Config | None | Configuration object |

## Examples

### Basic Training

```python
import torch
from torchlingo.models import SimpleTransformer
from torchlingo.config import Config

config = Config(d_model=256, n_heads=8)
model = SimpleTransformer(
    src_vocab_size=10000,
    tgt_vocab_size=10000,
    config=config,
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

### Inference (Greedy Decoding)

```python
def greedy_decode(model, src, tgt_vocab, max_len=50, device="cpu"):
    model.eval()
    src = src.to(device)
    
    # Encode source
    memory = model.encode(src)
    
    # Start with SOS
    ys = torch.tensor([[tgt_vocab.sos_idx]], device=device)
    
    for _ in range(max_len):
        logits = model.decode(ys, memory)
        next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)
        ys = torch.cat([ys, next_token], dim=1)
        
        if next_token.item() == tgt_vocab.eos_idx:
            break
    
    return ys[0].tolist()
```

### Custom Masking

```python
# Manual padding mask
src_pad_mask = (src == pad_idx)  # True where padding

# Manual causal mask
tgt_len = tgt.size(1)
causal_mask = torch.triu(
    torch.ones(tgt_len, tgt_len, dtype=torch.bool),
    diagonal=1
)

logits = model(
    src, tgt,
    src_key_padding_mask=src_pad_mask,
    tgt_mask=causal_mask,
)
```

## Key Features

### Rotary Position Embeddings (RoPE)

Unlike standard sinusoidal or learned position embeddings, RoPE encodes position information directly into the attention computation:

- Better extrapolation to longer sequences
- Captures relative positions naturally
- No separate position embedding layer

### Automatic Masking

The model automatically generates:

1. **Padding masks**: Prevent attention to PAD tokens
2. **Causal masks**: Prevent decoder from seeing future tokens

### Embedding Scaling

Embeddings are scaled by √d_model to maintain variance:

```python
src_emb = self.src_tok_emb(src) * math.sqrt(self.d_model)
```

### Xavier Initialization

Weights are initialized using Xavier uniform for stable training:

```python
for m in self.modules():
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
```

## Model Variants

### Tiny (Demo/Testing)

```python
model = SimpleTransformer(
    src_vocab_size, tgt_vocab_size,
    d_model=64, n_heads=2,
    num_encoder_layers=1, num_decoder_layers=1,
)
# ~500K params, trains in seconds
```

### Base

```python
model = SimpleTransformer(
    src_vocab_size, tgt_vocab_size,
    d_model=512, n_heads=8,
    num_encoder_layers=6, num_decoder_layers=6,
)
# ~65M params, good quality
```

### Large

```python
model = SimpleTransformer(
    src_vocab_size, tgt_vocab_size,
    d_model=1024, n_heads=16,
    num_encoder_layers=6, num_decoder_layers=6,
    d_ff=4096,
)
# ~200M params, high quality
```

## Computational Considerations

| Aspect | Complexity |
| ------ | ---------- |
| Self-attention | O(n² × d) |
| Memory | O(n² + n × d) |
| Parameters | O(L × d²) |

Where n = sequence length, d = d_model, L = num layers.

!!! warning "Long Sequences"
    Attention has O(n²) complexity. For very long sequences (>1000 tokens), consider using flash attention or other efficient attention implementations.
