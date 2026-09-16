# Visualization

Two views of the same decode: what the model **looked at**, and what the search
**considered and discarded**.

| Question | Functions |
| --- | --- |
| What did the decoder attend to? | `plot_attention`, `format_attention` |
| What did the search consider and prune? | `plot_beam_search`, `format_beam_search` |

The second is the one students find least intuitive, because pruning is invisible in the
output. A translation tells you what won and never what lost — yet the entire argument for
beam search is about paths greedy never explores. See [Beam search](#beam-search) below.

## Attention

### Overview

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

## Beam search

Beam search keeps several hypotheses alive and discards the rest at every step. The
output tells you which one won; it never tells you what was thrown away, and that is
exactly where the interesting behaviour is.

Pass a list as `trace` and the search records every candidate it scored:

```python
from torchlingo.inference import beam_search_decode
from torchlingo.visualization import format_beam_search

trace = []
tokens = beam_search_decode(model, src, beam_size=3, trace=trace)
print(format_beam_search(trace, itos=tgt_vocab.idx2token, winner=tokens))
```

Tracing is observation only. It never changes the result, and costs nothing when omitted.

### Reading the output

```
step 0
  + -1.953  <s> w23
  > -2.007  <s> w19
  + -2.214  <s> w4
step 1
  + -3.245  <s> w19 w16
  > -3.383  <s> w19 w21
  + -3.501  <s> w4 w16
  . -3.543  <s> w23 w19
    ... 5 more considered
```

| Marker | Meaning |
| --- | --- |
| `>` | kept, and on the path that eventually won |
| `+` | kept into the next step |
| `.` | pruned here |

**Look for a `>` sitting below a `+`.** That is the whole lesson. At step 0 above, the
eventual winner ranked *second*: greedy decoding would have committed to `w23` and never
recovered. Beam search found the better translation only because the beam was wide enough
to carry a hypothesis that did not look best at the time.

If the `>` is always at the top, beam search did nothing greedy would not have done — and
on an easy sentence that is the common case, which is worth seeing too.

`plot_beam_search` shows the same thing as scores over steps: kept candidates filled,
pruned ones hollow, and the winning path drawn as a line. The visual question is whether
that line ever dips below other filled points.

## API Reference

::: torchlingo.visualization
    options:
      show_source: true
      members:
        - format_attention
        - plot_attention
        - format_beam_search
        - plot_beam_search
