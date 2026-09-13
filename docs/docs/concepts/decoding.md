# Decoding

## Greedy is the default — beam search is opt-in

!!! warning "Read this before reporting a BLEU score"
    Everywhere TorchLingo takes a `decode_strategy` argument, it defaults to
    `"greedy"`. That includes `translate_batch()` and, most importantly,
    `evaluate_model()`.

    Greedy decoding is fast and simple, but it usually scores **lower** than beam
    search. A BLEU number produced with the defaults is not your model's best score.

```python
# Default: greedy
translate_batch(model, sentences, src_vocab, tgt_vocab)

# Beam search: opt in explicitly
translate_batch(model, sentences, src_vocab, tgt_vocab,
                decode_strategy="beam", beam_size=5)

evaluate_model(model, dataloader, src_vocab, tgt_vocab,
               decode_strategy="beam", beam_size=5)
```

Always say which strategy produced a score you report. Comparing a greedy BLEU against
a published beam BLEU is not a like-for-like comparison, and the gap can be a point or
more.

### Choosing a strategy

| Strategy | What it does | When |
|---|---|---|
| `"greedy"` | Takes the single highest-probability token at every step | Quick checks, debugging, during training |
| `"beam"` | Keeps `beam_size` hypotheses and picks the best complete one | Reporting results, final translations |

Beam search costs roughly `beam_size` times more computation. That cost is the reason
the rest of this page exists.

### Both strategies work on both architectures

`greedy_decode` and `beam_search_decode` each accept a `SimpleTransformer` or a
`SimpleSeq2SeqLSTM`, and the search is identical either way — the same beams, the same
pruning, the same length normalization, the same tie-breaking rule.

That is worth pausing on, because it is easy to absorb the opposite idea from the
Transformer-centric literature: **beam search is a property of decoding, not of the
model.** Anything that can score the next token given a prefix can be beam-searched. In
`inference.py` that is literally all the two paths differ by — one calls the
Transformer's `decode(tgt, memory)`, the other the LSTM's
`decode_prefix(tgt, hidden, enc_out)`, and the forty lines of search below them are
shared.

```python
# Identical call, either architecture
tokens = beam_search_decode(model, src, beam_size=5)
```

## Reference and fast implementations

TorchLingo ships **two implementations of beam search**. They produce exactly the
same translations. They differ only in how much of the work happens in one call to
the model.

| | Module | Use it when |
|---|---|---|
| Reference | `torchlingo.inference` | Reading, teaching, debugging, small inputs |
| Fast | `torchlingo.inference_fast` | Decoding a real test set |

The rest of this page explains why both exist, which to use, and what the difference
costs.

## The short version

The two modules expose **the same function names**. Switching between them is a one-line
import change, and nothing else in your code moves:

```python
# Reading the algorithm, or translating a handful of sentences:
from torchlingo.inference import beam_search_decode, translate_batch

# Translating an evaluation set — note only the module name differs:
from torchlingo.inference_fast import beam_search_decode, translate_batch
```

That is deliberate. The two implementations are interchangeable: same signature, same
arguments, byte-identical output. If they were not interchangeable, they would not share
a name.

When you need both in one file, import the modules instead of the functions, so it stays
obvious which is which:

```python
from torchlingo import inference, inference_fast

slow = inference.beam_search_decode(model, src, beam_size=5)
fast = inference_fast.beam_search_decode(model, src, beam_size=5)
assert slow == fast          # always true
```

The output is identical either way. If you are unsure, use the fast one — nothing about
your results changes.

## Why two?

Beam search is one of the ideas this library exists to teach. The reference
implementation is written so you can read it:

```python
for _ in range(max_len):
    candidates = []
    for tokens, score in beams:              # one model call per beam
        out = model.decode(tgt, memory, ...)
        log_probs = F.log_softmax(out[0, -1, :], dim=-1)
        top_log_probs, top_idx = _canonical_topk(log_probs, beam_size)
        for lp, idx in zip(top_log_probs.tolist(), top_idx.tolist()):
            candidates.append((tokens + [idx], score + lp))
    candidates.sort(key=...)
    beams = candidates[:beam_size]
```

The whole search is about 40 lines, and the loop structure mirrors how beam search is
described in a textbook: expand every live hypothesis, score the expansions, keep the
best `beam_size`.

That readability has a cost. Look at the inner loop: it calls `model.decode()` once
**per beam**. With `beam_size=5`, that is five separate calls to the model at every
step, each processing a single sequence.

## What that costs

Measured on 8 sentences, `max_len=25`, `beam_size=5`:

| | `decode()` calls | sequences per call | positions forwarded |
|---|---|---|---|
| greedy | 25 | 8.0 | 2,600 |
| beam (reference) | 968 | **1.0** | 12,968 |
| ratio | **38.7×** | 4.8× | 5.0× |

Read the last column first. Beam search does about **5× the arithmetic** of greedy —
which is roughly what `beam_size=5` should cost, and is not a problem.

Now read the first column. It issues **38.7× more calls** to the model, every one of
them with a batch of a single sequence.

That gap is the whole story. The work is not the bottleneck; the number of trips to the
GPU is. Each call carries fixed overhead — Python dispatch, kernel launch, moving data —
and at batch size 1 that overhead dominates the arithmetic. The GPU spends most of its
time waiting for the next instruction instead of computing.

This is one of the most transferable lessons in practical deep learning: **many small
operations are slower than one large operation, even when the total arithmetic is
identical.**

### That 38.7× is a budget, and it splits in two

Think of 38.7× as the speedup *available* from batching. It is not one lever — it is two
independent ones, and their effects multiply:

| Lever | Worth | Why |
|---|---|---|
| Batch the **beams** | ~`beam_size` | Every step evaluates `beam_size` hypotheses, currently one call each |
| Batch the **sentences** | ~`num_sentences` | Every sentence is decoded on its own, currently one at a time |

With `beam_size=5` and 8 sentences: 5 × 8 = 40 ≈ 38.7. Greedy in the table above already
pulls the sentence lever (all 8 sentences go through together), which is why it looks so
much better.

`inference_fast.beam_search_decode` pulls the **beam** lever. Measured: 4.8×, a little under
`beam_size` because fewer beams remain live late in the search. It still decodes one
sentence at a time, so the sentence lever is untouched and available.

So from the batched decoder, expect roughly `beam_size` — and note that the remaining
lever is worth more the larger your test set is, since it scales with the number of
sentences.

## What the fast version changes

Not the algorithm. The same beam search, the same scores, the same tie-breaking, the
same output.

The change rests on one observation:

!!! note "All live beams always have the same length"
    Every step appends exactly one token to every hypothesis, and finished hypotheses
    are retired before expansion. So the surviving beams can be stacked into a single
    `(n_live, t)` tensor with no padding at all.

Once they are stacked, the five separate calls become one:

```python
tgt = torch.tensor([tokens for tokens, _ in live])   # (n_live, t)
memory_batch = memory.expand(n_live, -1, -1)         # a view, not a copy
out = model.decode(tgt, memory_batch, ...)           # ONE call for all beams
log_probs = F.log_softmax(out[:, -1, :], dim=-1)     # (n_live, vocab)
```

Calls per decode drop from `O(max_len × beam_size)` to `O(max_len)`. Measured on a small
CPU model: **140.9 ms/sentence → 39.1 ms/sentence**, a 3.6× speedup, with byte-identical
output.

## How we know the output is identical

This is the part that makes two implementations safe to maintain.

The reference implementation is the **specification**. Every fast implementation must
reproduce its output exactly. That is not a convention anyone has to remember — it is
enforced by the test suite's structure:

```python
class BeamSuperiorityContract:          # fixtures and invariants, no TestCase
    BEAM_DECODE = None                  # supplied by each subclass
    ...

class ReferenceBeamSuperiorityTests(BeamSuperiorityContract, unittest.TestCase):
    BEAM_DECODE = staticmethod(beam_search_decode)

class BatchedBeamSuperiorityTests(BeamSuperiorityContract, unittest.TestCase):
    BEAM_DECODE = staticmethod(inference_fast.beam_search_decode)
```

Every fixture and invariant runs against **both** implementations. Adding a third is one
subclass, and it inherits the entire suite. Because failures name the implementation, a
divergence is unambiguous rather than a mystery.

The suite covers padding invariance, `beam_size=1` agreeing with greedy, determinism,
exact-tie behavior, and golden token sequences — including a fixture whose best path is
only reachable with `beam_size >= 2`, so an implementation that quietly degraded into
greedy would fail loudly.

## A warning you may see

```
UserWarning: Beam decoding 3000 sentences with the reference implementation,
which evaluates one beam per model call and is written for readability rather
than speed. For inputs this size use torchlingo.inference_fast.translate_batch,
which produces identical output.
```

This fires once per process when beam decoding more than 100 sentences through the
reference path. It is telling you that you are on the readable path, not the fast one,
and that switching costs nothing in accuracy.

To silence it:

```python
import warnings
warnings.filterwarnings("ignore", module="torchlingo.inference")
```

## Exercises

1. **Measure it yourself.** Wrap `model.decode` in a counter and decode a few sentences
   with each implementation. Confirm the call counts and check the outputs match token
   for token.
2. **Break it on purpose.** Change the fast implementation to ignore `beam_size` and
   always keep one beam. Which contract tests fail, and do the failure messages tell you
   what went wrong?
3. **Where does the remaining time go?** The fast version still calls `_canonical_topk`
   once per beam per step. How much would batching that too actually save — and how would
   you keep the tie-breaking provably identical if you did?

## See also

- [Model Architectures](models.md) — the encoder-decoder these decoders drive
- [Training Loop](training.md) — where the model comes from
- API reference for `torchlingo.inference` and `torchlingo.inference_fast`
