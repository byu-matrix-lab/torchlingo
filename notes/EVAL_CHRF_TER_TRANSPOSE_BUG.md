# Bug: `compute_chrf` and `compute_ter` do not transpose references

Found 2026-09-23 while building the CS 479 Lecture 6 in-class activity around
`torchlingo.evaluation`.

## Status: diagnosis confirmed, fix already written — **PR #58**, open

Independently found and fixed the same day. Do not write a second fix; it would
conflict with PR #58. Merging that PR closes this note, and the note should be
deleted at the same time.

Reproduced on `main` exactly as described, and verified against the PR #58
branch:

| | `main` | PR #58 | sacreBLEU direct |
|---|---|---|---|
| chrF (`word_order=2`) | 100.0 | **78.40** | 78.40 |
| TER | 0.0 | **16.67** | 16.67 |
| BLEU | 0.0 | 0.0 | 0.0 |

PR #58 factors out `_as_reference_streams` exactly as proposed below, calls it
from all three functions, and corrects both docstring values (chrF `63.39` →
`57.12`; TER `80.00` → `16.67`, on a new example, since the old one did not
discriminate). Its `tests/test_metric_reference_shape.py` passes 13 tests, and
`--doctest-modules` on the module passes 6.

**Two corrections to this note**, so the next reader does not inherit them:

1. **The "correct value" snippet contradicts the value.** The note says the
   right answer is 78.4 and shows `sacrebleu.corpus_chrf(preds, [refs]).score`
   to get it. That call returns **76.97**. Our `compute_chrf` defaults to
   `word_order=2` (chrF++) while sacreBLEU's `corpus_chrf` defaults to `0`
   (plain chrF); 78.4 is the `word_order=2` figure and is the right target.
   Written literally, the proposed regression test would fail against a
   *correct* implementation, by 1.43 points. PR #58's test passes `word_order`
   explicitly and says why in its module docstring.
2. **"Where it bit" overstates the blast radius.** Nothing needs regenerating.
   There are no recorded chrF or TER numbers anywhere in `docs/` or in a
   checkpoint scores file — searched, none exist. `examples/evaluate.py` and
   `examples/train.py` are the only callers of `evaluate_model`, and neither
   commits its output. The wrong numbers were computed at runtime and never
   written down, which is also why nothing caught the bug.

The rest of the note is accurate and is left as written.

## Symptom

`compute_chrf` returns a silently wrong score. No exception is raised.

```python
from torchlingo.evaluation import compute_chrf
preds = ["Hello world", "How are you"]
refs  = ["Hello world", "How are you doing"]
compute_chrf(preds, refs).score   # -> 100.0
```

The correct corpus chrF for those two pairs is **78.4**:

```python
import sacrebleu
sacrebleu.corpus_chrf(preds, [refs]).score   # -> 78.4
```

`compute_ter` has the same defect. `compute_bleu` is correct.

## Cause

All three functions start by normalizing a single-reference list into
per-sentence form:

```python
if references and isinstance(references[0], str):
    references = [[ref] for ref in references]
# references is now [[r1], [r2], [r3], ...]
```

`compute_bleu` then transposes that into the shape sacrebleu actually wants,
which is a list of reference *streams*:

```python
# src/torchlingo/evaluation.py, in compute_bleu
num_refs = len(references[0]) if references else 0
transposed_refs = []
for ref_idx in range(num_refs):
    transposed_refs.append([sent_refs[ref_idx] for sent_refs in references])
bleu_metric.corpus_score(predictions, transposed_refs)
```

`compute_chrf` and `compute_ter` skip that step and pass the per-sentence form
straight through:

```python
return sacrebleu.corpus_chrf(predictions, references, word_order=word_order)
return sacrebleu.corpus_ter(predictions, references, normalized=normalized)
```

sacrebleu reads the outer dimension as streams, so with N sentences it sees N
streams of length 1 instead of 1 stream of length N. It does not raise; it
scores the wrong thing.

## Fix

Factor the transpose out of `compute_bleu` into a helper and call it from all
three functions. Something like:

```python
def _as_reference_streams(references):
    """Normalize refs to sacrebleu's list-of-streams shape.

    Accepts either ["ref", ...] (one reference per sentence) or
    [["ref_a", "ref_b"], ...] (several references per sentence) and returns
    [[sent1_ref_a, sent2_ref_a, ...], [sent1_ref_b, sent2_ref_b, ...]].
    """
    if not references:
        return [[]]
    if isinstance(references[0], str):
        references = [[ref] for ref in references]
    num_refs = len(references[0])
    return [[sent_refs[i] for sent_refs in references] for i in range(num_refs)]
```

`compute_bleu` keeps its character-tokenization branch, which operates on the
per-sentence form, so call the helper after that branch rather than before.

## Docstrings to re-check

The examples in the module are wrong wherever they exercise the broken path.
Regenerate the expected values after fixing:

- `compute_chrf` docstring claims `chrF: 63.39`
- `compute_ter` docstring claims `TER: 80.00`

The `compute_bleu` docstring values and the module-level `BLEU: 0.00` example
are correct and should not change.

## Test worth adding

A regression test that would have caught this: score N > 1 sentences where at
least one hypothesis differs from its reference, and assert the result equals
`sacrebleu.corpus_chrf(preds, [refs]).score`. The two-sentence example above is
enough: 100.0 vs 78.4 is not a rounding difference.

Also worth asserting that `compute_chrf` and `compute_bleu` agree on a perfect
translation set (both 100.0) and on a completely wrong one, which pins the
shape at both ends.

## Where it bit

`evaluate_model` has `compute_chrf_score: bool = True` by default, so every
call to it has been reporting a wrong chrF (line 393). TER is opt-in
(`compute_ter_score: bool = False`) and so is affected less often. There is a
second pair of call sites further down the module, around lines 443-446.

`docs/docs/tutorials/` and `examples/evaluate.py` both call `evaluate_model`,
which routes through these functions. Any chrF or TER number recorded in the
generated docs or in a checkpoint's scores file is suspect until this is fixed
and those are regenerated.
