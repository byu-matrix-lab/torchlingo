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

### Works on both architectures

`return_attention=True` means the same thing on `SimpleSeq2SeqLSTM` and
`SimpleTransformer`, and returns the same `(batch, tgt_len, src_len)` shape, so
the code above does not care which model you hand it.

On the Transformer the weights come from the **last decoder layer** with heads
averaged. That is the conventional choice for alignment plots. For every layer,
or per-head maps, use
[`capture_cross_attention`](#torchlingo.models.transformer_simple.capture_cross_attention)
directly.

!!! warning "Call `eval()` first"
    In training mode, attention dropout randomly zeroes weights and rescales
    the survivors, so rows will not sum to 1 and the picture is not the one
    inference uses. This is easy to miss because the plot still looks
    plausible. Check it:

    ```python
    assert torch.allclose(weights.sum(-1), torch.ones(weights.shape[:-1]), atol=1e-5)
    ```

??? note "Why the Transformer needs extra machinery and the LSTM does not"
    The LSTM computes attention in TorchLingo's own code, so returning the
    weights is a matter of not throwing them away.

    The Transformer uses PyTorch's `nn.Transformer`, and
    `TransformerDecoderLayer._mha_block` calls attention with
    `need_weights=False` hardcoded. That is a deliberate performance choice: it
    allows a fused kernel that never materializes the attention matrix at all.
    The weights are not hidden behind an option, they are genuinely never
    computed, which is why a forward hook on `multihead_attn` comes back with
    `None`.

    `capture_cross_attention` temporarily replaces each layer's
    `multihead_attn.forward` to force `need_weights=True`, records what comes
    back, and restores the original afterwards — including if the forward pass
    raises. The speed cost is why this is opt-in rather than always on.

    This is worth knowing beyond TorchLingo. Fast paths that discard
    intermediate values are common in deep learning libraries, and "the
    framework will not give me X" often means "X is never computed on the path
    you are taking."

### What it looks like on a real model

From the pretrained checkpoint in Tutorial 5, translating a sentence it has
never seen:

```
EN  The cat sleeps on the mat.
ES  El código de la catura.

      <sos>  ▁The    ▁c    at    ▁s    le    ep     s   ▁on  ▁the  ▁mat     . <eos>
▁El     ···   ▓▓▓   ···   ···   ···   ···   ···   ···   ···   ···   ···   ···   ···
▁c      ···   ···   ░░░   ···   ░░░   ···   ···   ···   ···   ···   ···   ···   ···
ó       ···   ···   ░░░   ░░░   ░░░   ···   ···   ···   ···   ···   ░░░   ···   ···
d       ···   ···   ···   ···   ···   ···   ░░░   ░░░   ···   ···   ░░░   ···   ···
igo     ···   ···   ···   ···   ···   ░░░   ░░░   ░░░   ···   ···   ░░░   ···   ···
▁de     ···   ···   ···   ···   ···   ···   ···   ···   ░░░   ···   ···   ░░░   ···
▁la     ···   ···   ···   ···   ···   ···   ···   ···   ···   ░░░   ···   ░░░   ···
▁c      ···   ···   ░░░   ░░░   ░░░   ···   ···   ···   ···   ···   ···   ···   ···
at      ···   ···   ░░░   ░░░   ░░░   ···   ···   ···   ···   ···   ░░░   ···   ···
ura     ···   ···   ···   ···   ···   ░░░   ░░░   ░░░   ···   ···   ░░░   ···   ···
```

The translation is wrong, and the map shows *how* it is wrong, which a correct
translation would not.

`▁El` attends sharply to `▁The` — the one word it got right, and the one place
the grid is dark. `▁c` and `at` attend to the `▁c`/`at` pieces of "cat", so the
model half-recognized the word and produced *catura*. Everywhere else the row is
a flat wash of `░░░`: attention spread thinly across the whole source, which is
what "has not learned what to look at" looks like.

Compare that to Tutorial 4's model, trained on a synthetic task with a known
correct alignment, where the map is a clean diagonal. The contrast is the
lesson: a sharp attention map is evidence the model learned *something*, and a
diffuse one is evidence it did not.

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
