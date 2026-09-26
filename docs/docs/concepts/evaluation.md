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

## What counts as a good score

"BLEU 30 is decent" has to come from somewhere. The most widely used anchor is
[Google Cloud Translation's interpretation table](https://docs.cloud.google.com/translate/docs/bleu-scores):

| BLEU | Interpretation |
| --- | --- |
| < 10 | Almost useless |
| 10–19 | Hard to get the gist |
| 20–29 | The gist is clear, but with significant grammatical errors |
| 30–40 | Understandable to good |
| 40–50 | High quality |
| 50–60 | Very high quality, adequate and fluent |
| > 60 | Often better than human |

**The Tutorial 5 model scores 7 to 9.** That is the bottom row. The tutorial
says the translations are not good; this is the calibrated version of the same
statement, and it is why a BLEU in the single digits is the expected result of
this much data and this much training rather than a sign something broke.

!!! warning "A scale is not a law"
    These cut points are a vendor's rule of thumb for their own systems, not a
    property of BLEU. They shift with language pair, domain and how many
    references you score against. Quote the source when you quote the scale.

    Other metrics need their own scales, and this is the hard part. The Matrix
    Lab's `mtsurvey` dashboard carries bands for every metric it reports and is
    explicit about how each was obtained: BLEU's are anchored to the table
    above, BLASER's to the human XSTS scale it predicts, and the rest are
    *derived* from BLEU's boundaries by equipercentile mapping over its own
    scored corpus. A derived band is only as transferable as the corpus it came
    from — which is the same caution as everything else on this page.

## Three families, and the one that needs no reference

Everything above compares a translation to a reference. That is not the only
option, and the distinction matters as soon as you leave a benchmark:

| Family | How it scores | Examples |
| --- | --- | --- |
| **Surface** | String overlap with a reference | BLEU, chrF, TER |
| **Reference-based neural** | A model trained on human judgements, given the reference | COMET, xCOMET, BLEURT, MetricX |
| **Quality estimation (QE)** | A model given **only the source and the output** | CometKiwi, MetricX-QE, BLASER-QE |

QE is the one worth knowing about, because references are the scarce resource.
You have them for a test set and never for the translations you actually care
about — the ones your system produces in the wild. A QE metric will score those.

TorchLingo ships only the surface family; the rest need large pretrained models
and are out of scope for a teaching library. The lab's `mtsurvey` dashboard runs
all three across many models, languages and datasets, and is the place to see
them disagree at scale.

## The question nobody asks: is it even the right language?

`mtsurvey` reports one thing that is not a quality metric at all — an
**on-target rate**, the fraction of outputs actually written in the target
language, detected with a language-ID model.

It exists because multilingual systems fail this way and nothing else catches
it. A model that answers in English when asked for Spanish, or drifts into a
related language, produces fluent, confident output. BLEU will be low, but BLEU
is low for many reasons, and "low BLEU" does not tell you which. An on-target
rate does, in one number.

TorchLingo's model is English→Spanish only and trained on nothing else, so it
cannot make this mistake in an interesting way. Keep the question anyway: it
belongs beside the checks in
[Tutorial 6](../tutorials/06-diagnosing-failures.ipynb), and it is the first
thing to test on any multilingual system.

## Where this stops being a settled question

The three metrics above are *surface* metrics: they compare strings. The field
has largely moved to **learned** metrics — COMET, CometKiwi, xCOMET, MetricX —
which score a translation with a neural model trained on human judgements. They
correlate better with human ratings, and they are what WMT evaluates with.

They also have failure modes that a string-matching metric cannot have, because
they inherit whatever their encoder learned.

!!! note "An open question, from this lab"
    The Matrix Lab's `mtsurvey` project benchmarks these across many models,
    languages and datasets, and the survey work built on it studies exactly
    this. Treating a set of automatic metrics as partially independent
    observers of the same translations, it finds that learned metrics **drift
    from the rest of the field at their encoder's pretraining coverage
    boundary** — and that the dividing line is pretraining coverage rather than
    whether the metric sees a reference.

    The practical consequence: a learned metric can be systematically
    optimistic or pessimistic for a language depending on what its encoder was
    pretrained on, in a way no single score reveals. If you are evaluating a
    low-resource pair, the metric is part of the experiment.

    This is the natural next step after this page. Everything here is one
    small model on one language pair; `mtsurvey` is where the same questions
    are asked at the scale that makes the answers interesting.

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
