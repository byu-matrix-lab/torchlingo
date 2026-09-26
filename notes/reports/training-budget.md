# Training budget beat data by about 7x, and a published comparison had it backwards

**Legacy finding, recorded 2026-09-23. Not generated from JSON** — it predates the
`notes/reports/` pipeline, and the runs behind it are not reproducible from a committed
source. Kept as prose for that reason, and flagged so nobody mistakes it for a generated
report.

Moved here from `notes/TASKS.md` on 2026-09-26. It is a finding, not a task, and it was
sitting where nobody would look for it — which mattered, because it is **the prior for the
learning curve** (Task #119).

## How it surfaced

Coulson, on PR #34: "the BLEU scores are extremely low ... worth looking into if it wasn't
flagged before." The scores were expected and documented. Looking into them anyway found a
measurement error.

## The root cause

`train_model` appended to `val_losses` in two places: the periodic step-triggered
validation from `config.val_interval`, and the epoch-end validation. One list, two
different measurements, and a docstring promising "per epoch".

```
              train_losses   val_losses   true epochs
  baseline         20            36            20
  new              36            72            36
```

Reading `len(val_losses) == 36` off the baseline gave "36 epochs", so the new run was set
to `--epochs 36` to match it. **The baseline had run 20.**

## What that did to the published comparison

It gave one model 19% more data *and* 80% more training, while the write-up claimed data
was the only difference. Re-run with epochs actually matched:

| data | epochs | BLEU | |
|---|---|---|---|
| 53,520 pairs | 20 | 4.96 | baseline as shipped |
| 53,520 pairs | 36 | **7.01** | control: same data, more epochs |
| 64,311 pairs | 36 | 7.32 | more data *and* more epochs |

- epochs 20 → 36, data fixed: **+2.05 BLEU**
- +20% data, epochs fixed: **+0.29 ± 0.22, 95% CI [−0.16, +0.71]**

**The data effect's interval crosses zero.** Training budget mattered roughly **7x** more
than the recovered data, and the recovered data bought nothing measurable.

So the published claim was wrong twice: about 88% of the +2.33 was training length, and the
residual is not significant. The diagnosis in tutorial 5 — "data-starved" — is also wrong.
The model was **undertrained**, which has a different fix.

## Why this is the prior for the learning curve

Task #119 asks whether going past 100K pairs helps. This finding is the reason that
question has to be designed rather than just run:

- **The confound is not hypothetical here.** It has already produced a wrong published
  conclusion in this repository, in exactly the direction a learning curve is vulnerable
  to: more data brings more gradient steps per epoch unless something holds them equal.
- **The effect sizes set the scale.** If 20% more data bought +0.29 ± 0.22 BLEU, then
  adjacent points on a curve will differ by less than seed noise unless the data steps are
  large. That is the argument for wide spacing — doubling — and for measuring seed noise
  before reading any gap as real.
- **It says which lever to expect to dominate.** At these corpus sizes the budget did. A
  curve that shows data mattering more would be surprising and should be checked before
  being believed.

## What has changed since

The `val_losses` defect is fixed. Two related measurements have also landed and both point
the same way as this one:

- The **length ladder** (`length-ladder.md`) separated memory from time: batch count drives
  epoch time, the length cap drives whether a run fits. Neither is about data volume.
- The 9.5x step-unit mismatch with OpenNMT, recorded under Task #95, is another instance of
  the same class of error — two numbers that look comparable and are not.

The pattern across all of them is worth naming, since it has now described several separate
defects: **two things that must agree, with nothing checking that they do.**
