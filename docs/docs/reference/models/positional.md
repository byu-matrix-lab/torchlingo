# Positional Encoding

Position embedding implementations for Transformer models.

## Overview

Transformers process all positions in parallel and have no inherent sense of order. Positional encodings add position information to token embeddings so the model can distinguish between "The cat sat" and "sat cat The".

TorchLingo implements **Rotary Position Embeddings (RoPE)**, a modern approach that encodes position directly in the attention computation.

## Quick Start

```python
from torchlingo.models.positional import RoPEEmbedding
import torch

rope = RoPEEmbedding(d_model=512, max_seq_length=2048)

# Apply to embeddings
embeddings = torch.randn(2, 100, 512)  # [batch, seq_len, d_model]
positioned = rope(embeddings)
```

## API Reference

::: torchlingo.models.positional.RoPEEmbedding
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

### Solution 1: Sinusoidal Encoding (Original)

The original Transformer used fixed sinusoidal patterns:

$$
PE_{(pos, 2i)} = \sin\left(\frac{pos}{10000^{2i/d}}\right)
$$

$$
PE_{(pos, 2i+1)} = \cos\left(\frac{pos}{10000^{2i/d}}\right)
$$

This is added to token embeddings:

```python
embeddings = token_embeddings + position_embeddings
```

### Solution 2: Learned Encoding

Learn a position embedding for each position:

```python
pos_emb = nn.Embedding(max_seq_length, d_model)
embeddings = token_embeddings + pos_emb(positions)
```

### Solution 3: Rotary Position Embeddings (RoPE)

RoPE encodes position by **rotating** embeddings in 2D subspaces:

```
For position p, rotate embedding by angle θ × p
```

**Key insight**: The dot product between rotated vectors depends on their **relative** position, not absolute positions.

## RoPE Benefits

| Feature | Traditional | RoPE |
| ------- | ----------- | ---- |
| Position encoding | Added | Multiplied (rotation) |
| Captures | Absolute position | Relative position |
| Extrapolation | Poor | Better |
| Learned | Often | No |

### Why Relative Position Matters

For translation, relative position often matters more than absolute:

- "The big cat" → Adjective is before noun (relative position -1)
- "cat big The" → Wrong order

RoPE naturally captures these relative relationships.

## Implementation Details

### Rotation Matrix

For each 2D subspace (dimensions 2i and 2i+1):

$$
\begin{pmatrix}
\cos(m\theta_i) & -\sin(m\theta_i) \\
\sin(m\theta_i) & \cos(m\theta_i)
\end{pmatrix}
\begin{pmatrix}
x_{2i} \\
x_{2i+1}
\end{pmatrix}
$$

Where $m$ is the position and $\theta_i = 10000^{-2i/d}$.

### In Code

```python
def apply_rope(x, cos, sin):
    # Split into even/odd dimensions
    x1, x2 = x[..., ::2], x[..., 1::2]
    
    # Apply rotation
    rotated = torch.cat([
        x1 * cos - x2 * sin,
        x1 * sin + x2 * cos,
    ], dim=-1)
    
    return rotated
```

## Usage in SimpleTransformer

`SimpleTransformer` uses RoPE automatically:

```python
model = SimpleTransformer(
    src_vocab_size=10000,
    tgt_vocab_size=10000,
    d_model=512,
    max_seq_length=2048,  # RoPE will work up to this length
)
```

The RoPE embedding is applied in the `_embed` method before passing to encoder/decoder layers.

## Configuration

```python
from torchlingo.models.positional import RoPEEmbedding

rope = RoPEEmbedding(
    d_model=512,           # Must match model dimension
    max_seq_length=2048,   # Maximum supported sequence length
)
```

## Comparison with Other Methods

### Sinusoidal vs RoPE

```python
# Sinusoidal: Add position info
embedded = token_emb + pos_emb  # Sum

# RoPE: Rotate by position
embedded = rotate(token_emb, position)  # Rotation
```

### Learned vs RoPE

| Aspect | Learned | RoPE |
| ------ | ------- | ---- |
| Parameters | max_len × d_model | None (fixed) |
| Generalization | Only trained positions | Any position |
| Memory | Stores embeddings | Computes on-the-fly |

## Handling Long Sequences

RoPE extrapolates better than traditional encodings to positions beyond training:

```python
# Trained on sequences up to 512
# RoPE can handle 1024+ with reasonable quality

rope = RoPEEmbedding(d_model=512, max_seq_length=2048)
long_seq = torch.randn(1, 1500, 512)  # Beyond 512
positioned = rope(long_seq)  # Still works!
```

However, performance may degrade for very long extrapolation. For best results, train with sequences close to your expected inference length.
