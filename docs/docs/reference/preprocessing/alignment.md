# Alignment Checks

Two cheap checks for whether a parallel corpus is actually parallel.

## Overview

A parallel corpus can be structurally perfect and still be worthless: correct columns,
no blank rows, real sentences on both sides, and yet row `n` of the source is not a
translation of row `n` of the target.

That is not hypothetical. The corpus shipped with this library was once misaligned in
exactly that way, and nothing in the repository caught it. A student training on it
would have watched the loss refuse to fall with no way to tell bad data from a mistake
of their own.

This module is the check that was missing.

| Check | Idea | Parallel text | Scrambled |
|---|---|---|---|
| `length_correlation` | A translation is about as long as its source | ~0.97 | ~0.09 |
| `anchor_agreement` | Names and numbers survive translation | ~0.41 | ~0.06 |

Neither is proof, and each has a blind spot the other covers, so run both. See
[Data Pipeline](../../concepts/data-pipeline.md#cleaning-is-not-the-same-as-checking)
for the reasoning and the measured numbers.

## API Reference

::: torchlingo.preprocessing.alignment.diagnose_alignment
    options:
      show_source: true

::: torchlingo.preprocessing.alignment.AlignmentReport
    options:
      show_source: true

::: torchlingo.preprocessing.alignment.length_correlation
    options:
      show_source: true

::: torchlingo.preprocessing.alignment.anchor_agreement
    options:
      show_source: true

::: torchlingo.preprocessing.alignment.anchors
    options:
      show_source: true

::: torchlingo.preprocessing.alignment.shuffle_target_side
    options:
      show_source: true

## Examples

### Checking a corpus before you train on it

```python
import pandas as pd
from torchlingo.preprocessing import diagnose_alignment

frame = pd.read_csv("data/example.tsv", sep="\t")
report = diagnose_alignment(frame)

print(report)
# AlignmentReport(length_correlation=0.9694, anchor_agreement=0.4102,
#                 scorable_rows=36821, rows=73082)

if not report.looks_aligned():
    raise SystemExit(f"corpus looks misaligned: {report}")
```

### Custom column names

```python
report = diagnose_alignment(frame, src_col="english", tgt_col="spanish")
```

### Seeing the checks fail

A check you have never watched fail is a check you cannot read. Break a corpus you know
is good and run the checks again:

```python
from torchlingo.preprocessing import shuffle_target_side

broken = shuffle_target_side(frame)
print(diagnose_alignment(broken).looks_aligned())   # False
```

`shuffle_target_side` rotates the target column by one row. Every column stays
individually intact and every *pairing* breaks, which is what misalignment is.

### Adjusting the thresholds

The defaults are deliberately conservative: well below what a good corpus achieves, far
above what a broken one manages. A noisier corpus or a more distant language pair may
legitimately score lower.

```python
report.looks_aligned(min_correlation=0.6, min_agreement=0.15)
```

## Limitations

**The anchor check needs distinctive anchors.** It compares *which* names and numbers
the two sides share, so a token appearing in nearly every row carries no signal. A
single speaker's name, repeated throughout their own talks, is shared by every pairing
whether that pairing is right or wrong. Scramble such a corpus and agreement stays at
100% while the length correlation collapses.

**The length check needs varied lengths.** A corpus of uniformly short sentences has
little length signal to correlate.

**Neither is proof.** A corpus can pass both and still be subtly misaligned, for example
if it is offset by a whole document rather than scrambled. Look at the numbers, then
look at some rows.
