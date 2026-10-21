# A11 also scores the student's own A8 and A9 systems (a requirement, today)

*From the Cowork session, Wed Oct 21 2026, about 13:20. Eric decided it in chat: "For A11: add the
requirement to score on A8 and A9." Context: `Roadmap/Assignment Arc Plan_v1.md` (course folder),
the A11 section. A11 is live, due Mon Oct 26 at 10:00, so the notebook should be on `main` today.*

## Why

A11's outcome is COMET and COMETKiwi on the commercial systems' A6 output. The student never
turns COMET on their own system until A13. The addition keeps every existing step and adds one:
COMET and COMETKiwi on the student's own A8 and A9 translations of their own test set, and
whether COMET agrees with BLEU that A9's pieces helped.

## What to add to `lecture-11-comet-on-a6`

A new step after the report (Step 5) and before the summary, so the summary becomes Step 7:
**"Step 6: your own systems."**

1. **Reads from Drive**, `MyDrive/CS479/assignment8` (the A8 notebook's `OUT_DIR`, the A9
   notebook's `A8_DIR`): `test.tsv` (src, tgt), `test.hyp` (A8) and `test-a9.hyp` (A9). Clear
   errors when a file is missing (name which assignment wrote it). Check that the line counts
   match the test set. If A9 Part B saved its own re-translation of A8 (it re-scores A8 with the
   same code), prefer that file over A8's `test.hyp`, so both systems are scored as A9 reported
   them; otherwise use `test.hyp` and say so.
2. **Scores both systems** with COMET (`wmt22-comet-da`, with the references) and COMETKiwi
   (`wmt22-cometkiwi-da`, without), using the student's `comet_inputs`, in the same GPU session
   after the login cell. About 2,000 sentences × 2 systems × 2 models: a few minutes on a T4.
   Also BLEU (spBLEU with `NO_SPACES`, as A9) and chrF for the same files, so the table is
   complete.
3. **Self-contained and rerun-safe.** A student who has already finished Steps 1 to 5 runs the
   login cell and then this step alone: it loads the two models itself and saves to its own file
   (for example `a11-own-notes.json` in `assignment11`), so a dropped session does not cost the
   A6 scores, and the A6 notes file is untouched.
4. **Prints for the summary:** a table, metric by system (A8, A9), with which system each metric
   prefers; and, sentence by sentence over the test set, how often COMET and chrF prefer the same
   system (the student's `pairwise_agreement`), plus the first three sentences where they
   disagree (SRC, REF, A8, A9, both scores).
5. **A coverage note**, in the markdown above the cell: COMET and COMETKiwi read through
   XLM-R/InfoXLM, which covers about 100 languages. For a language outside them, the scores mean
   less, and the student says so. (If the notebook can tell from the language name, then it can
   print a reminder.)

## The summary (Step 7): a fifth question

**5. Your own systems.** "Do COMET and COMETKiwi agree with BLEU (as A9 reported it) about
whether the pieces helped? Use the sentence-by-sentence count: on how many sentences do COMET and
chrF prefer the same system? Pick one sentence where they disagree and say which you believe, and
why."

## Tests

`tests/test_lecture_11.py`: the step with COMET stubbed; a missing `test-a9.hyp` (message names
A9); mismatched line counts; the summary now has five questions.

## The roadmap (yours, so we do not both edit it)

Lecture 11's `can` line gains the own-system half, for example: "Run COMET and COMETKiwi on their
own A6 data **and on their own A8 and A9 test output**; reconcile them with BLEU, chrF and their
human ranking; ...". The A11 row's points go from 26 to 30 (below).

## Already done or being done on the course side

- **Learning Suite, the A11 page (`jLj_`):** a directions bullet for Step 6, "five summary
  questions", question 5, What to submit, and two rubric rows: "COMET and COMETKiwi scores for
  your A8 and A9 systems on your test set" (2) and "Question 5: your own systems" (2). Points
  26 to 30. The gradebook entry follows.
- **`torchlingo-private/notes/assignments/A11-directions.md`:** the same, with a History line.
- **The Lecture 11 deck:** the Assignment 11 slide (five questions, 30 points; the diagram gains
  the A8 and A9 outputs) in a `_v2` candidate.

Please reply in `to-cowork/` when it is on `main`, with the step's exact name and file names, so
the page's wording can be matched to them.
