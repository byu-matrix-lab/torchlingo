# notes/reports/

Experimental outcomes. **`TASKS.md` tracks work; this directory records what was
measured.** The two were conflated once and it went badly: a memory curve ended up
inside a task entry, where nobody would look for it and where its numbers sat beside
the arithmetic that had been done on them, including the arithmetic that turned out
to be wrong.

## The rule: numbers come from JSON, never from typing

Every report here is **generated** from a JSON file in `data/`. The JSON is the single
source of truth; the Markdown is a build artifact.

```
notes/reports/
  README.md                 this file
  data/
    length-ladder.json      measured numbers, written by the experiment
  length-ladder.md          GENERATED -- do not edit
```

To regenerate, from the repository root:

```bash
python scripts/render_report.py                 # rewrite the Markdown
python scripts/render_report.py --check         # fail if it is out of date
```

`--check` is the part that matters. Without it, "generated" is an honour system, and a
number gets hand-corrected in the Markdown once and the two copies diverge silently
from then on. See Task #81.

### Why this is worth the ceremony

A figure that is typed twice is a figure that will disagree with itself. This
repository has already published a chrF of 100.00 that should have been 78.40, and a
decode length that had two different values in two places. Both were single numbers
that nothing checked.

It also means a rerun is cheap. Re-running the experiment rewrites the JSON, the
render rewrites every table and every derived ratio, and nothing has to be chased
through prose.

## What belongs in a report

- What was measured, with the configuration it was measured under. A memory figure
  without its batch size and model dimensions is not reusable.
- The numbers, generated.
- What the numbers mean, and **what they do not support.** The length ladder's first
  two rungs looked like a clean quadratic and were in fact the same run twice, because
  the cap was silently not being applied; saying so is more useful than the curve.
- Corrections, marked as corrections, rather than quiet edits. A reader who saw the
  earlier version needs to know it changed.

## Provenance, and a note on what is safe to publish here

The length-ladder measurements were taken on the German corpus, which is private and
deliberately outside this repository. What is recorded here is machine behaviour —
memory, seconds per epoch, loss — plus **aggregate** corpus statistics such as length
percentiles and what fraction of pairs a cap truncates. No corpus text, and nothing
from which the corpus could be reconstructed.

That line is drawn deliberately rather than by habit: the truncation percentages are
what make the memory figures interpretable, so omitting them would leave a report that
cannot be acted on. Existing practice in `TASKS.md` already records the same class of
aggregate.
