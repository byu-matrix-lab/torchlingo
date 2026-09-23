# When It Fails

Every other page here shows the machinery working. This one is about what to do when it
does not, which is where you will spend most of your time.

The hard part of a failing model is rarely the fix. It is the **diagnosis**: a model that
will not learn looks exactly like a model whose data is broken, which looks exactly like
a model that is learning fine and being measured wrong. Four very different problems, one
symptom.

Worse, as a student you cannot tell "this is broken" from "I did it wrong" — and the
second is far more likely to be your first guess. It is often not the answer.

## Every failure below actually happened here

None of these are invented. Each is a real incident in TorchLingo's own history, with the
symptom that showed, the explanation that looked obvious, and what it turned out to be.

| Symptom | Looked like | Actually was | How it was found |
|---|---|---|---|
| Loss will not fall | Bad hyperparameters, model too small | The corpus was not parallel: the two columns were unrelated documents zipped together | Length correlation 0.001 where parallel text scores 0.97 |
| Translations all empty | Broken decoder | Undertrained. Loss 3.02 against `ln(24) = 3.18`, so the model had barely moved off uniform | Comparing loss to the uniform-guess baseline |
| Every beam size gives the same answer | Beam search does not help | The model was too certain to search. The experiment could not have detected an effect | Repeating it on an uncertain model |
| Beam 3 is best | A finding worth reporting | Sampling noise. The winner moved on a different subset | Five seeds and paired comparisons |
| Crash on Apple Silicon | A bug in the model | An unimplemented PyTorch op on the MPS backend | The same code running on CPU |
| BLEU near 100 | An excellent model | Testing on the training sentences | Asking where the test set came from |

Read down the "Looked like" column. Every one of those is a reasonable first guess, and
every one is wrong. That is the point.

## A diagnostic order

Work outward from the data, because a fault there invalidates everything downstream. Each
step is cheap and rules out a whole class of causes.

### 1. Is the data what you think it is?

The most expensive failure to debug, because everything downstream behaves plausibly. A
misaligned corpus trains a model that produces confident, fluent, unrelated output.

```python
from torchlingo.preprocessing import diagnose_alignment

report = diagnose_alignment(frame)
print(report)
if not report.looks_aligned():
    raise SystemExit("stop here; nothing below this will make sense")
```

See [Data Pipeline](data-pipeline.md#cleaning-is-not-the-same-as-checking) for what these
checks measure and where they run out. Also look at ten rows by hand. It takes a minute
and catches things no check will.

### 2. Is the model learning *anything*?

Do not ask whether the loss is "good". Ask whether it has left the starting line.

A model that has learned nothing predicts uniformly over the vocabulary, giving a
cross-entropy loss of `ln(vocab_size)`. That is your floor for *no learning at all*:

```python
import math
print(f"uniform-guess loss: {math.log(len(tgt_vocab)):.2f}")
```

If your loss sits near that number, the model is not training. Suspect the learning rate,
a detached graph, a frozen parameter, or a loss computed on the wrong tensor — not the
architecture.

This is exactly how the empty-translation bug was caught: loss 3.02, baseline 3.18. It
had learned almost nothing, and five epochs was simply not enough.

### 3. Is it learning the *wrong* thing?

Training loss falling while validation loss rises is ordinary overfitting. The
interesting failures are subtler:

- **Copying the source.** Check whether output equals input more often than chance. This
  library's corpus generator drops tags and view counts for exactly this reason: they are
  byte-identical across languages, and training on them teaches copying.
- **Collapsing to a frequent output.** If most inputs produce the same translation, the
  model has found a shortcut that scores tolerably and stopped.
- **Memorizing.** Perfect on training sentences, useless on new ones. Tutorial 3 scores
  BLEU 100 for this reason and says so.

### 4. Is the measurement lying?

A model can be fine and the number wrong.

- **Leakage.** If your test sentences appear in training, you are measuring memory. Split
  by *document*, not by sentence: consecutive sentences in one talk share vocabulary and
  topic. This corpus carries a `talk` column so that split is possible.
- **Decoding strategy.** Greedy and beam BLEU are not comparable. Always say which
  produced a number.
- **Noise.** A one-point BLEU difference on a few hundred sentences may be nothing. See
  [Decoding](decoding.md) for a worked case where the apparent winner changed with the
  sample.

### 5. Is it the environment?

Last, because it is rarest, and because assuming it first is how people waste a day.

The tell is that the *same code* behaves differently somewhere else. TorchLingo's
decoders once crashed on Apple Silicon and worked on CPU: an unimplemented op in the MPS
backend, nothing to do with the model. If something works on one device and not another,
suspect the device.

## The habit worth taking away

Before believing a result, ask **what would this look like if it were wrong?**

The beam-size sweep in Tutorial 3 returns five identical rows. Read one way, beam size
does not matter. Read properly, the experiment could not have detected an effect if there
were one, because the model was certain. Same output, opposite conclusions.

A measurement you cannot imagine failing is a measurement you are not yet reading.

## See also

- [Data Pipeline](data-pipeline.md#cleaning-is-not-the-same-as-checking) — the alignment checks, and repairing drift
- [Decoding](decoding.md) — what the decoding options actually buy, with error bars
- [Training](training.md) — the loop, optimizers and schedulers when things go right
