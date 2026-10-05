# Your three decisions: Lecture 9, Lecture 7 and the renumbering are all merged

**Baton back to you, 2026-10-05; last updated the evening of 2026-10-05.** Answers
`from-cowork/2026-10-05-d-baton.md` and the three files it indexes, and takes in
`2026-09-30-8a-slide-reordered.md`, which had not been picked up. Roadmap v12 and the audit's
errata are committed as you left them. **All three decisions are merged on `main`**; the final
tutorial filenames, which your flip waits on, are in section 3.

## 1. Lecture 9: merged, PR #197

Applied cell by cell as written. The asserts stay hard, `WHY` and `BLEU_PREDICTION` included:
the notebook is Colab only, so CI never runs it and the blanks cost nothing, and Wednesday's
clock does not argue for softer stops. Checked outside Colab against a synthetic A8 split: with
the blanks as shipped it stops at `unknown_rate`'s `NotImplementedError`; filled in, it runs end to
end and prints the guess, the reason, the unchanged settings and the prediction.

**Merged before Wednesday's class**, so Lecture 9 runs the new notebook.

## 2. Lecture 7: merged, PR #198, option 2

Option 2, Eric's preference: no assert on `PREDICTION`. An empty one prints "No prediction
written..." and the report says `Prediction: (none written)`. So the notebook runs unedited from
its badge and in CI. **Two wording departures**, both following from that:

- Part B's opener ends "The report at the end prints it, and says so if it is missing", not "The
  cell refuses to run until you have".
- "both training calls above" (the resume paragraph) is now "every training call above": there
  are three.

**Your roadmap's Lecture 7 note says "writes a prediction the cell requires"**; with option 2 the
cell asks and the report shows the gap. Please reword when you flip the note.

Executed unedited in 28 seconds: *The dog sleeps* held out came out *El perro corre* (chrF 44.7),
which is the case the "What to look at" text anticipates, and the *Good night* contrast came out
*Buenos días* (chrF 24.8). Both seen lines 11/11, chrF 100.

## 3. The renumbering: merged, PR #201 — flip when ready

**The final filenames**, all in `docs/docs/tutorials/`:

| tutorial | file | lecture |
|---|---|---|
| 1 | `01-data-and-vocab.ipynb` (unchanged) | 4, 9 |
| 2 | `02-evaluating-translations.ipynb` (was 07) | 6 |
| 3 | `03-train-tiny-model.ipynb` (was 02) | 7 |
| 4 | `04-attention-and-alignment.ipynb` (unchanged) | 8a, 8b |
| 5 | `05-transformer-attention.ipynb` (was 08) | 8b |
| 6 | `06-diagnosing-failures.ipynb` (unchanged) | 9 |
| 7 | `07-real-translations.ipynb` (was 05) | 10 |
| 8 | `08-inference-and-beamsearch.ipynb` (was 03) | 10 |

**Redirect stubs** stay at the five old paths: one Markdown cell, "This tutorial has moved", with
the new file's Colab badge and link. So an old badge on a slide or a Content page you have not
flipped yet lands one click from the right notebook, and nothing breaks if the flip takes a
day. The stubs go after the term (Task #187).

**Rewritten:** every reference by number or filename in the tutorials, the docs site and its
nav (reordered to the new sequence), the course notebooks, scripts, tests, `CLAUDE.md`,
`TASKS.md`, `NOTEBOOK_AUDIT.md`, `briefing.md`, and the repository half of the roadmap, whose
notebook map is regenerated. Tutorial 7 also had sentences that spoke of tutorial 8 in the past
tense; those are now tense-neutral, and its What's Next points on to tutorial 8.

**Left alone on purpose, as you asked to be told:**

- **Your half of the roadmap**, above "Which notebook serves which lecture": yours to flip.
- **Frozen hand-offs and archives** in `notes/handoff/`: "tutorial 8" in them keeps meaning what
  it meant when written. This file uses the new numbers.
- `CHANGELOG.md` (a record of releases as they were), `data/`, and `notes/legacy-f2025/`.

## 4. Your two extra items

- **Encoder self-attention for tutorial 5: Eric said go.** Task #188, in two halves. The library
  half is PR #202: `capture_self_attention` beside `capture_cross_attention`. It is your
  pre-hook, with one finding for the record: in eval mode under `no_grad`, PyTorch's encoder
  layers take a fused path that never calls `self_attn`, which is why a pre-hook is the only
  approach that works. The tutorial cell (the first layer's four heads, their average, and a
  sharpness line per layer) is written and executed, but **waits for the next PyPI release**,
  because Colab installs from PyPI and the function is not there yet. On this model and
  tutorial 5's sentence, each head puts 0.90–1.00 of every row on one piece, their average
  0.40, and later layers sit near 0.3: your slides' point, in numbers.
- **The pretrained model's facts:** recorded in Task #49, the retrain task, so a retrain announces
  itself to you and keeps writing `trained_minutes` and `train_pairs`. Nothing is scheduled to
  move those numbers.

## 5. From your Sep 30 file

- **Our 8a step-order question was answered there** (the slide was reordered before class).
  Closed; thank you.
- **Tutorial 8 on the 8b deck and in the roadmap:** done on your side, so Task #183 is closed.
  (That was tutorial 8 by the old numbering, now tutorial 5.)
- **The A9 date in the transcript index** in `torchlingo-private` now reads Oct 12 (v9).
- **The regex refresher: Eric's answer is to keep it a fill-in worksheet.** No change to it.
- **The A12 files** (Task #176) are still with Eric.

## 6. One change to the hand-off protocol

**Eric, 2026-10-05: one main hand-off file per baton pass; sub-files are allowed.** Your Oct 5
pass came as four siblings (`2026-10-05-…`, `-b-`, `-c-`, `-d-baton`), with the last one acting
as the index. From now on:

- **The main file is the message.** It is the one the reader opens, and everything needing an
  answer or a decision is in it.
- **A long spec may go in a sub-file**, in a folder named after the main file
  (`2026-10-06-slug.md` beside `2026-10-06-slug/lecture-10-spec.md`), linked from the main file
  with a line on what it holds.
- **The `-b` suffix still means a new pass** on the same day, never a second file for one pass.

`notes/handoff/README.md` has it under "Rules".

## 7. Which tasks moved

`notes/TASKS.md` is reconciled in the pull requests carrying this file.

| | |
|---|---|
| **Task #183** | closed: tutorial 5 (then 8) is on the 8b deck and in the roadmap |
| **Tasks #184, #185, #186** | opened and closed today: Lecture 9, Lecture 7 and the renumbering, merged |
| **Task #187** | new: remove the redirect stubs, after the term |
| **Task #188** | new, and a go: self-attention in tutorial 5; library in PR #202, the cell after a release |
| **Task #49** | the decks' quoted checkpoint facts added, so a retrain is announced |
| **Task #152** | unchanged: still no report from Coulson's Colab run of 8a; chased here now, not through Eric |
| **Task #176** | unchanged: blocked on Eric's A12 files |

No questions for you this time.
