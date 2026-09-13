# Attention

Readable, from-scratch attention mechanisms for the LSTM decoder.

## Overview

The Transformer gets its attention from `torch.nn.Transformer`, where the
mechanism is real but invisible. This module is the version you can actually
read — roughly fifteen lines per scorer.

Every attention mechanism here does the same three things:

1. **Score** each source position against the current decoder state.
2. **Softmax** those scores into weights that sum to 1.
3. **Average** the encoder outputs using those weights, producing a *context* vector.

Only step 1 differs between the two implementations.

| Scorer | Score | Paper | Parameters |
| ------ | ----- | ----- | ---------- |
| `"dot"` | `h_t · h_s` | Luong et al., 2015 | none |
| `"additive"` | `v · tanh(W_dec h_t + W_enc h_s)` | Bahdanau et al., 2014 | three matrices |

!!! tip "Read them in publication order"
    Bahdanau's additive score *learns* how to compare decoder and encoder
    states. Luong's dot score notices that if the two already live in the same
    space, an inner product will do — no parameters at all. The Transformer then
    keeps the dot product, adds a `1/sqrt(d_k)` scale, and runs it in parallel
    heads. Reading the first two makes the third one familiar rather than alien.

## Quick Start

```python
from torchlingo.models import SimpleSeq2SeqLSTM

model = SimpleSeq2SeqLSTM(
    src_vocab_size=10000,
    tgt_vocab_size=10000,
    attention=True,
    attn_type="dot",      # or "additive"
)

logits, weights = model(src, tgt, return_attention=True)
# weights: [batch, tgt_len, src_len], each row summing to 1
```

Attention is **off by default**. That keeps the classic bottlenecked seq2seq as
the baseline and makes the with-versus-without comparison a single visible flag.

!!! warning "Checkpoints are not interchangeable"
    Enabling attention adds parameters, so a checkpoint trained with
    `attention=False` will not load into a model built with `attention=True`,
    or the reverse. Build the model the same way you trained it.

## Using the mechanisms directly

```python
from torchlingo.models.attention import build_attention

attn = build_attention("dot", hidden_dim=512)
context, weights = attn(dec_out, enc_out, src_pad_mask)
```

| Argument | Shape | Meaning |
| -------- | ----- | ------- |
| `dec_out` | `[batch, tgt_len, hidden]` | Decoder states — the queries |
| `enc_out` | `[batch, src_len, hidden]` | Encoder outputs — keys and values |
| `src_pad_mask` | `[batch, src_len]` | `True` marks padding to ignore |
| → `context` | `[batch, tgt_len, hidden]` | Weighted average of `enc_out` |
| → `weights` | `[batch, tgt_len, src_len]` | The alignment matrix |

## Padding

Masked positions are pushed to the most negative finite value before the
softmax, not to `-inf`. Both give a weight of essentially zero, but `-inf`
produces `NaN` if an entire row is padded — a silent corruption that is
unpleasant to track down later.

!!! note "Masking the weights is not the whole story"
    The encoder LSTM itself must also stop at the real end of each sentence, or
    the state handed to the decoder describes padding rather than the sentence.
    `SimpleSeq2SeqLSTM.encode_source` packs the batch to guarantee this. Masking
    attention alone does not fix it — the leak is in the recurrence, not the
    alignment.

## Seeing what it learned

Attention weights are the most directly inspectable quantity in an NMT model.
See [Visualization](../visualization.md) for the renderers, and run:

```bash
python examples/attention_alignment.py
```

That example trains on a task whose correct alignment is known in advance (the
target is the reversed, word-substituted source), so "did attention work?"
becomes a measurable number instead of a heatmap you squint at:

```
configuration            val loss   alignment acc
----------------------------------------------------------------
no attention               1.1031             n/a
dot (Luong)                0.6801           98.4%
additive (Bahdanau)        0.6533           93.6%
```

Chance is about 12.5% on that task.

## API Reference

::: torchlingo.models.attention
    options:
      show_source: true
      members:
        - DotProductAttention
        - AdditiveAttention
        - build_attention
