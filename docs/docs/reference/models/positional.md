# Positional Encoding

Position encoding implementation for Transformer models.

## Overview

Transformers process all positions in parallel and have no inherent sense of order. Positional encodings add position information to token embeddings so the model can distinguish between "The cat sat" and "sat cat The".

TorchLingo implements the **sinusoidal positional encoding** from the original Transformer paper ("Attention Is All You Need", Vaswani et al., 2017).

## Quick Start

```python
from torchlingo.models.positional import SinusoidalPositionalEncoding
import torch

pos_enc = SinusoidalPositionalEncoding(d_model=512, max_seq_len=2048)

# Apply to embeddings
embeddings = torch.randn(2, 100, 512)  # [batch, seq_len, d_model]
positioned = pos_enc(embeddings)
```

## API Reference

::: torchlingo.models.positional.SinusoidalPositionalEncoding
    options:
      show_source: true
      members:
        - __init__
        - forward

## How Positional Encoding Works

### The Problem

Attention computes:

$$
\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V
$$

This is **permutation invariant**—shuffling the input gives shuffled output with the same attention weights. We need to inject position information.

### The Sinusoidal Solution

The original Transformer uses fixed sinusoidal patterns. For position $pos$ and dimension pair $i$:

$$
PE_{(pos, 2i)} = \sin\left(\frac{pos}{10000^{2i/d}}\right)
$$

$$
PE_{(pos, 2i+1)} = \cos\left(\frac{pos}{10000^{2i/d}}\right)
$$

Each dimension pair oscillates at a different wavelength, from $2\pi$ up to $10000 \cdot 2\pi$, so every position gets a unique fingerprint. The encoding is simply added to the token embeddings:

```python
embeddings = token_embeddings + position_encodings
```

**Key insight**: for any fixed offset $k$, $PE_{pos+k}$ is a linear function of $PE_{pos}$ (a rotation in each sine/cosine pair). This makes it easy for the model to learn to attend by *relative* position — which is what matters for translation ("the adjective comes right before the noun").

### Alternatives You'll See in the Wild

| Method | Idea | Parameters |
| ------ | ---- | ---------- |
| **Sinusoidal** (this library) | Fixed sine/cosine waves added to embeddings | None |
| **Learned** | `nn.Embedding(max_len, d_model)` added to embeddings | max_len × d_model |
| **Rotary (RoPE)** | Rotate the query/key vectors *inside attention* so dot products depend only on relative offsets | None |

Learned encodings can't represent positions beyond the trained maximum. RoPE is popular in modern LLMs but requires a custom attention implementation (it must be applied to Q and K inside every attention layer, not to the input embeddings) — a great extension exercise once you understand the sinusoidal version.

## Usage in SimpleTransformer

`SimpleTransformer` applies the encoding automatically:

```python
model = SimpleTransformer(
    src_vocab_size=10000,
    tgt_vocab_size=10000,
    d_model=512,
    max_seq_length=2048,  # positions precomputed up to this length
)
```

Token embeddings are scaled by $\sqrt{d_{model}}$ (so they don't drown out the position signal) and the encoding is added in the `_embed` method before the encoder/decoder layers. Dropout is applied to the sum, as in the original paper.

## Handling Long Sequences

The table is precomputed for `max_seq_len` positions but extends itself automatically if a longer input arrives:

```python
pos_enc = SinusoidalPositionalEncoding(d_model=512, max_seq_len=512)
long_seq = torch.randn(1, 1500, 512)  # Beyond 512
positioned = pos_enc(long_seq)  # Table is rebuilt to 1500 positions
```

Because the encoding is a fixed function of position, it produces valid values for any position. Model *quality* at positions far beyond those seen in training is a separate question — for best results, train with sequences close to your expected inference length.
