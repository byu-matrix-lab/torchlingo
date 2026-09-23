# Evaluation

Every score in this project so far has been BLEU. That is one metric's opinion,
and a metric is a model of translation quality, not a measurement of it.

This page is about what that opinion leaves out.

## BLEU can score a perfect translation zero

Start with the failure, because it is easy to reproduce and hard to forget:

```python
from torchlingo.evaluation import compute_bleu

predictions = ["Hello world", "How are you"]
references  = ["Hello world", "How are you doing"]

print(compute_bleu(predictions, references).score)   # 0.0
```

One translation is *exactly* right. The other is close. BLEU says zero.

Standard BLEU is the geometric mean of 1- to 4-gram precisions, and a geometric
mean is zero if any term is zero. These sentences are two and three words long,
so they contain no 4-grams at all, the 4-gram precision is zero, and the whole
score collapses no matter how good the translations are.

**BLEU is a corpus-level metric.** On a handful of short sentences it is not
merely noisy, it is meaningless. That is not a bug in sacreBLEU; it is what the
formula says. It is also why Tutorial 3's BLEU of 100 was the broken
measurement and Tutorial 5's 7 was the working one.

## Three metrics, one set of translations

TorchLingo ships three, all from sacreBLEU:

| Function | Metric | Counts | Direction |
|---|---|---|---|
| [`compute_bleu`](../reference/evaluation.md) | BLEU | word n-grams, 1 to 4 | higher is better |
| [`compute_chrf`](../reference/evaluation.md) | chrF | character n-grams, plus word n-grams | higher is better |
| [`compute_ter`](../reference/evaluation.md) | TER | edits per reference word | **lower** is better |

Scored on the same held-out sentences from the
[Tutorial 5](../tutorials/05-real-translations.ipynb) checkpoint:

--8<-- "docs/_generated/metric_comparison.md"

### What to take from that

**The magnitudes are not comparable at all.** BLEU says 7, chrF says 30, on
*identical* translations. Neither is wrong. They are counting different things:
chrF works at the character level, so it gets partial credit for a word with the
right stem and the wrong ending, where BLEU gets nothing. For morphologically
rich languages that difference is the whole argument for chrF.

So "BLEU 30 is decent" is a sentence about BLEU, not about translation. A number
means nothing without the metric, and — as
[Decoding](decoding.md#why-every-table-here-ends-with-a-signature) covers — not
much without the signature either.

**TER runs the other way.** 0 is perfect, and the model scores about 90, meaning
roughly nine edits per ten reference words. Put TER in a column beside BLEU
without saying which direction it runs and every reader misreads it once.

**But they agreed on the ranking.** All three prefer beam search over greedy.
That is the more important observation, and it is why practitioners compare
systems rather than quote absolute scores: *differences* between systems
measured the same way survive the choice of metric far better than the numbers
themselves do.

Agreement is not guaranteed. It is a property of this model, this test set and
these three metrics, and the honest way to know is to check rather than assume —
which is the same discipline as
[controlling one variable](when-it-fails.md) and
[checking the test set is clean](../tutorials/06-diagnosing-failures.ipynb).

## Where this stops being a settled question

The three metrics above are *surface* metrics: they compare strings. The field
has largely moved to **learned** metrics — COMET, CometKiwi, xCOMET, MetricX —
which score a translation with a neural model trained on human judgements. They
correlate better with human ratings, and they are what WMT evaluates with.

They also have failure modes that a string-matching metric cannot have, because
they inherit whatever their encoder learned.

!!! note "An open question, from this lab"
    The Matrix Lab's own survey work studies exactly this. Treating a set of
    automatic metrics as partially independent observers of the same
    translations, it finds that learned metrics **drift from the rest of the
    field at their encoder's pretraining coverage boundary** — and that the
    dividing line is pretraining coverage rather than whether the metric sees a
    reference. Every learned metric built on XLM-R steps at XLM-R's boundary;
    metrics built on other encoders step at no boundary tested.

    The practical consequence: a learned metric can be systematically
    optimistic or pessimistic for a language depending on what its encoder was
    pretrained on, in a way no single score reveals. If you are evaluating a
    low-resource pair, the metric is part of the experiment.

That is the grown-up version of this page. The reason to learn BLEU's collapse
on short text is not BLEU; it is that every metric has a shape, and the shape
decides what it can and cannot see.

## What to do

- **Report more than one metric**, and say which is which.
- **Report the signature**, so the number is reproducible.
- **Compare systems, not absolutes** — the difference is the robust part.
- **Never score a handful of sentences with BLEU.** Use chrF, or get more data.
- On a real evaluation, **look at the output too**. Every metric on this page
  can be satisfied by a translation you would not accept.

## See also

- [`torchlingo.evaluation`](../reference/evaluation.md) — the API, and the
  reference-shape trap that made two of these three functions wrong until it was
  caught
- [Decoding](decoding.md) — what the decoding knobs buy, and the sacreBLEU
  signature
- [Tutorial 6](../tutorials/06-diagnosing-failures.ipynb) — including what a
  contaminated test set does to any of these numbers
