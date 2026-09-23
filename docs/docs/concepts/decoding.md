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

### You have already met this algorithm

Beam search is usually introduced in NMT as though it were a translation technique. It
is not. It is **best-first search with a fixed-width frontier**, and you have almost
certainly seen it before under that description, in a course on search or AI.

The mapping is exact:

| General search | Beam search when decoding |
|---|---|
| State | The prefix generated so far |
| Successors | Every possible next token |
| Path cost | Cumulative log probability of the prefix |
| Heuristic | The model itself — it scores how promising a prefix is |
| Frontier | The `beam_size` live hypotheses |
| Goal test | The `<eos>` token |

The one thing that makes it *beam* search rather than plain best-first search is the
fixed-width frontier. A complete best-first search over this space is hopeless: the
branching factor is the vocabulary size, often tens of thousands, and the depth is the
length of the output. Keeping only the best `beam_size` states at each level is what
makes it tractable, and throwing the rest away is what makes it **incomplete** — it can
miss the optimal sequence, and it regularly does.

So the honest framing is not "beam search finds the best translation." It is: *greedy
search is beam search with a frontier of one, and widening the frontier trades
computation for a better chance of finding a high-scoring sequence, with no guarantee.*
[What the knobs actually do](#what-the-knobs-actually-do) shows what that trade buys in
practice. It is less than you might expect, and past a point widening the frontier makes
the translations *worse* — a searchable objective and a good translation are not the
same thing.

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

### Watching the search prune

The argument for beam search is about the paths greedy never explores — and those are
invisible in the output, which only ever shows the winner. Pass a list as `trace` to see
them:

```python
trace = []
tokens = beam_search_decode(model, src, beam_size=3, trace=trace)
print(format_beam_search(trace, itos=tgt_vocab.idx2token, winner=tokens))
```

The case worth finding is a step where the eventual winner was **not** ranked first. That
is the moment greedy would have gone elsewhere and been unable to come back. See
[Visualization](../reference/visualization.md#beam-search) for how to read it.

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

--8<-- "docs/_generated/decode_bench.md"

Every number on this page comes from `scripts/bench_decode.py`, and the table above is
generated from its output rather than typed in. Run it yourself:

```bash
python scripts/bench_decode.py
```

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

`inference_fast.beam_search_decode` pulls the **beam** lever. The measured reduction is in
the second table above, a little under `beam_size` because fewer beams remain live late in
the search. It still decodes one sentence at a time, so the sentence lever is untouched and
available.

So from the batched decoder, expect roughly `beam_size` — and note that the remaining
lever is worth more the larger your test set is, since it scales with the number of
sentences.

## What the knobs actually do

Everything above is about cost. This section is about what you get back, which is a
different question and has a less comfortable answer.

Measured on the pretrained model from
[Tutorial 5](../tutorials/05-real-translations.ipynb), because the answer depends on
having a model that is wrong often enough to be interesting. Tutorial 3's toy model is
so decisive that every beam size returns the same translation, which is why the sweep
there teaches nothing.

--8<-- "docs/_generated/decoding_sweep.md"

### Why every table here ends with a signature

That last block — `nrefs:1|case:mixed|eff:no|tok:13a|smooth:exp|version:2.6.0` — is the
**sacreBLEU signature**, and it is not decoration.

BLEU counts matching n-grams, so it depends entirely on where you decide the token
boundaries are. Score the same translations with a different tokenizer and the number
moves, sometimes by several points, with nothing about the number itself to warn you.
That is why two papers both reporting "BLEU 26" may not be making the same claim, and it
is the reason sacreBLEU exists at all: it fixes the settings and makes you state them.

So a bare BLEU number is not a measurement. It is a measurement with the units left off.

You can watch the effect directly:

```python
from torchlingo.evaluation import compute_bleu

prediction = ["The Cat Sat On The Mat Today And Slept"]
reference = ["the cat sat on the mat today and slept"]

cased = compute_bleu(prediction, reference, lowercase=False)
lowered = compute_bleu(prediction, reference, lowercase=True)

print(cased.score, cased.signature)      # 0.0    ... case:mixed ...
print(lowered.score, lowered.signature)  # 100.0  ... case:lc    ...
```

Identical text, identical metric, and the score goes from 0 to 100. The *only* thing
that distinguishes the two numbers is a field in the signature.

`compute_bleu` attaches `.signature` to every result for this reason, so reporting it
costs nothing. It also records one thing sacreBLEU cannot see: for CJK text TorchLingo
tokenizes to characters *before* sacreBLEU runs and then passes `tokenize="none"`, so
sacreBLEU's own signature would say `tok:none` and omit the step that actually happened.
The signature carries `tokenization:char` instead. A signature that hides a tokenization
decision is worse than no signature, because it invites a comparison that is not valid.

!!! tip "The habit, not the string"
    This generalizes past BLEU. Any number you will later compare against another number
    needs to travel with the settings that produced it. Tutorial 6 makes the same point
    from the other direction — a measurement you cannot reproduce is not a measurement —
    and the corpus-comparison mistake recorded there is exactly what happens when two
    things change and only one is written down.

### Most of the gain is the first beam, and past the peak it reverses

Read the paired table, not the columns.

**Greedy to any beam width is the big move**, worth between +1.18 and +1.65 BLEU, in the
same direction on every subset.

**After that it rises, plateaus, then declines.** Beam 2 to beam 3 is a real gain
(+0.39 ± 0.14). Beam 3 to beam 5 is nothing (+0.04 ± 0.14). Beam 5 to beam 10 is a real
**loss** (−0.47 ± 0.08), and the round trip from beam 2 to beam 10 nets out at zero.
Beam 10 costs about **7x** beam 2 in seconds to arrive back where it started.

So quality peaks around beam 3 to 5, which is roughly where the literature's usual
default sits. The surprise is not the peak, it is that going past it actively hurts. The
next section explains why.

!!! warning "This conclusion depends on the model, and we watched it change"
    An earlier version of this page, measured on a weaker checkpoint, reported that
    **no** beam width was distinguishable from any other, and drew the lesson that only
    the greedy-versus-beam decision matters.

    That was an honest reading of the data at the time and it was wrong. Retraining on
    ~19% more data lifted the model from BLEU 4.96 to 7.32, and at that quality the
    beam-to-beam differences separate from the noise: what had been a flat line became a
    peak with a measurable decline after it.

    Nothing about the earlier table looked unreliable. It had five seeds, paired
    comparisons and error bars, and it still supported a conclusion the next model
    overturned. "No difference detectable" had meant *this model was too weak to show
    one* — the same trap as Tutorial 3's five identical beam sizes, one level up.

    The transferable habit is to state what a result was measured on, and to re-run it
    when that changes. Every number here comes from one checkpoint, one language pair,
    and one test set.

!!! note "Why the error bars are the point"
    Look at the BLEU column of the first table on its own and beam 5 appears best, at
    9.20 against 9.16 for beam 3. That gap is 0.04 with a standard error of 0.14: it is
    not a result, and on a different draw of sentences the winner moves.

    The paired comparison is what rescues the analysis. Because every configuration
    decodes the *same* sentences, the per-run difference cancels the "which sentences
    did we happen to sample" variance that dominates the raw error bars. Unpaired, even
    beam 5 versus beam 10 looks like a wash; paired, it is a clear loss.

    Any claim of the form "beam size *n* is best for my model" needs this treatment. It
    is very easy to publish the noise instead.

### Wider beams produce shorter translations

The `mean length` column falls in a straight line: **12.26** tokens at greedy down to
**9.69** at beam 10, while the reference translations average **11.62**.

This is not a quirk of this model. Beam search maximizes total log probability, and
every additional token multiplies in another probability below 1, so a longer sequence
is a lower-scoring sequence. Widen the search and it finds shorter, higher-scoring
candidates that greedy walked straight past. Beam 10 finds sequences greedy never
considered, and those sequences are systematically too short.

**This is why going past the peak hurts.** Follow the two columns together. Up to beam
3, the search is still finding better translations and the shortening is mild. By beam
10, mean length has fallen to 9.69 against a reference average of 11.62, and the
sequences it is now finding are higher-probability but too short to contain the
reference's n-grams. Quality and probability have come apart: the search is succeeding
at its stated objective and failing at the task.

That gap between "what the search maximizes" and "what you wanted" is the single most
useful idea on this page, and it is what `alpha` exists to paper over.

### `alpha` is doing less than you would think

Given the length bias just described, you would expect the correction for it to matter.
It does not, over most of its range.

At the default `alpha=0.6`, length normalization is **indistinguishable from turning it
off entirely** (`alpha=0.0` changes BLEU by −0.12 ± 0.07). So are `alpha=0.3`
(−0.05 ± 0.06) and `alpha=1.0` (+0.15 ± 0.11). Only `alpha=1.5` separates from the
rest, and it is clearly *worse*: −0.85 ± 0.12, with output ballooning to 13.33 tokens
against a reference average of 11.62.

Read that carefully, because it is a stronger claim than it looks. Across 0.0 to 1.0 —
from no normalization at all to full per-token averaging — **this knob does nothing you
can measure**, while the bias it exists to correct is plainly visible in the length
column above. The shipped default is doing no work.

Two things that does not mean.

**It is not evidence that length normalization is useless in general.** It is evidence
about this model at this quality on this test set. A stronger model, a longer-sentence
corpus, or a language pair with a different length ratio could all change it.

**It is not settled why.** TorchLingo applies normalization during *pruning* as well as
at final selection, which is defensible but non-standard, and that could blunt it.
Distinguishing "the default is too weak" from "normalizing during pruning cancels it out"
needs one more experiment: sweep `alpha` with the correction applied only at final
selection and compare. That is the open question, now with numbers attached to it.

**And the method is the transferable part.** Sweep the knob, pair the comparisons, put
error bars on them, and read output length next to BLEU. That is what turned this from
an assumption into a finding.

### Exercise

Reproduce the table, then break it:

```bash
python scripts/sweep_decoding.py --sentences 50 --seeds 1
```

One seed and 50 sentences gives you no error bars and a different "best" beam size than
the table above. That is the experiment most people actually run. Add seeds until the
answer stops moving, and notice how many it takes.

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

Calls per decode drop from `O(max_len × beam_size)` to `O(max_len)`, with byte-identical
output. The measured call reduction and speedup are in the second table above; both come
from `scripts/bench_decode.py`.

Note that the two do not match. Calls fall by more than wall clock does, because removing
a call does not remove the per-step Python bookkeeping, the per-row tie-breaking, or the
`log_softmax` — and each surviving call now does `beam_size` times more work, which is not
free. Reproducing that gap yourself is the point of the harness: it is the difference
between "fewer calls" and "faster", and they are not the same claim.

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
