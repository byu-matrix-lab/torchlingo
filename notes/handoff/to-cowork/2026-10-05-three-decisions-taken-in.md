# Your three decisions: Lecture 9 and Lecture 7 are in review, the renumbering is next

**Baton back to you, 2026-10-05.** Answers `from-cowork/2026-10-05-d-baton.md` and the three files
it indexes, and takes in `2026-09-30-8a-slide-reordered.md`, which had not been picked up. Roadmap
v12 and the audit's errata are committed as you left them. **Live: this file is updated in place
when the renumbering merges**, with the final filenames, since that is what your flip waits on.

## 1. Lecture 9 (the one with a clock): PR #197

Applied cell by cell as written. The asserts stay hard, `WHY` and `BLEU_PREDICTION` included:
the notebook is Colab only, so CI never runs it and the blanks cost nothing, and Wednesday's
clock does not argue for softer stops. Checked outside Colab against a synthetic A8 split: with
the blanks as shipped it stops at `unknown_rate`'s `NotImplementedError`; filled in, it runs end to
end and prints the guess, the reason, the unchanged settings and the prediction.

**In review; I will say here when it merges.** If it is not merged by Wednesday morning, the class
runs the current notebook, as you planned.

## 2. Lecture 7: PR #198, option 2

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

## 3. The renumbering: next, as Task #186

Not started in code yet: it regenerates the roadmap's notebook map, so it waits for this hand-off
(which commits your v12) to land first, rather than stacking on it. Then: the five renames, every
cross-reference by number, stubs at the five old paths, and the list you asked for of what the
grep found and was left alone. Hand-offs and archives in `notes/handoff/` will be among those:
they are frozen records, so "tutorial 8" in them keeps meaning what it meant when written.
Removing the stubs after the term is Task #187.

## 4. Your two extra items

- **Encoder self-attention for tutorial 8 (to be 5):** filed as Task #188, a candidate rather than a
  commitment, with your hook described. It reads like a good cell; it comes after the three
  decisions.
- **The pretrained model's facts:** recorded in Task #49, the retrain task, so a retrain announces
  itself to you and keeps writing `trained_minutes` and `train_pairs`. Nothing is scheduled to
  move those numbers.

## 5. From your Sep 30 file

- **Our 8a step-order question was answered there** (the slide was reordered before class).
  Closed; thank you.
- **Tutorial 8 on the 8b deck and in the roadmap:** done on your side, so Task #183 is closed.
- **The A9 date in the transcript index** in `torchlingo-private` now reads Oct 12 (v9).
- **The regex refresher** (worksheet or runnable) and **the A12 files** (Task #176) are Eric's;
  both raised with him today.

## 6. Which tasks moved

`notes/TASKS.md` is reconciled in the pull request carrying this file.

| | |
|---|---|
| **Task #183** | closed: tutorial 8 is on the 8b deck and in the roadmap |
| **Task #184** | new: Lecture 9, PR #197, in review, runs Wed Oct 7 |
| **Task #185** | new: Lecture 7, PR #198, in review |
| **Task #186** | new: the renumbering, next |
| **Task #187** | new: remove the redirect stubs, after the term |
| **Task #188** | new: encoder self-attention in tutorial 8 (to be 5), a candidate |
| **Task #49** | the decks' quoted checkpoint facts added, so a retrain is announced |
| **Task #152** | unchanged: still no report from Coulson's Colab run of 8a |
| **Task #176** | unchanged: blocked on Eric's A12 files |

No questions for you this time.
