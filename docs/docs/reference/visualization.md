# Visualization

Rendering attention alignments.

## Overview

Each row of an attention matrix is a probability distribution saying "while
producing this target token, here is how much I looked at each source token."
Rendering that turns an abstraction into something you can check against your
own intuition about the sentence pair.

Two renderers are provided:

| Function | Output | Needs |
| -------- | ------ | ----- |
| [`plot_attention`](#torchlingo.visualization.plot_attention) | matplotlib heatmap | matplotlib |
| [`format_attention`](#torchlingo.visualization.format_attention) | shaded text grid | nothing extra |

`format_attention` exists because a figure window is not always available — over
SSH, in a log file, in CI, in a doctest.

## Quick Start

```python
from torchlingo.visualization import format_attention, plot_attention

logits, weights = model(src, tgt, return_attention=True)
src_tokens = src_vocab.indices_to_tokens(src_ids)
tgt_tokens = tgt_vocab.indices_to_tokens(tgt_ids)

print(format_attention(weights, src_tokens, tgt_tokens))
plot_attention(weights, src_tokens, tgt_tokens, title="Luong dot attention")
```

Both take a single sentence: either `[tgt_len, src_len]`, or
`[1, tgt_len, src_len]`, which is unwrapped for you. A real batch raises, rather
than silently showing you sentence zero — index it yourself with `weights[i]`.

## Reading the text grid

Rows are target tokens, columns are source tokens, and shading runs
`·` `░` `▒` `▓` `█` from no attention to full attention. Here is a model trained
on the reversal task from `examples/attention_alignment.py`, where the target is
the source translated word-for-word and reversed:

```
       <sos>  bird wants     a   cat   old   dog <eos>
<sos>    ···   ···   ···   ···   ···   ···   ███   ···
perro    ···   ···   ···   ···   ···   ▓▓▓   ░░░   ···
viejo    ···   ···   ···   ···   ███   ···   ···   ···
gato     ···   ···   ···   ███   ···   ···   ···   ···
un       ···   ···   ▓▓▓   ···   ···   ···   ···   ···
quiere   ···   ███   ···   ···   ···   ···   ···   ···
pajaro   ▓▓▓   ░░░   ···   ···   ···   ···   ···   ···
```

The anti-diagonal is exactly right: to emit the first target word the decoder
looks at the *last* source word. Each row is offset by one from the token it
names, because row `n` is the state that *predicts* token `n + 1`.

Pass `show_values=True` for rounded percentages instead of shading when you need
the actual numbers.

## API Reference

::: torchlingo.visualization
    options:
      show_source: true
      members:
        - format_attention
        - plot_attention
