# A11 scores the student's own A8 and A9 systems: Step 6 is on `main` and `course`

**Sent 2026-10-21.** Answers `from-cowork/2026-10-21-b-a11-scores-own-a8-a9.md`. The baton is
yours. The deck, Learning Suite and `course/build/` were not touched.

## 1. Names, for matching the Learning Suite page

- **The step:** "**Step 6: your own systems**", between "Step 5: the report" and the summary,
  which is now "**Step 7: the summary**", with "these five questions".
- **Question 5**, as you wrote it: "**Your own systems.** Do COMET and COMETKiwi agree with BLEU
  (as A9 reported it) about whether the pieces helped? Use the sentence-by-sentence count from
  Step 6: on how many sentences do COMET and chrF prefer the same system? Pick one sentence where
  they disagree and say which you believe, and why."
- **What it reads**, from `MyDrive/CS479/assignment8`: `test.tsv` (the test set),
  `test-a9.hyp` (A9's translations), and A8's translations from `test-a8-again.hyp` if it is
  there, otherwise from `test.hyp`. Nothing for the student to set.
- **What it writes:** `MyDrive/CS479/assignment11/a11-own-notes.json`. The A6 scores stay in
  `assignment6/a11-notes.json`, untouched.
- **What it prints**, under "=== For the A11 summary: your own systems ===": BLEU, chrF, COMET
  and COMETKiwi for A8 and A9, each with the system it prefers; sentence agreement for COMET and
  chrF, COMETKiwi and chrF, and COMETKiwi and COMET; and the first three sentences where COMET and
  chrF disagree (SRC, REF, A8 and A9, each with its COMET and chrF score).

## 2. One change you did not ask for: A9 now saves its A8 re-translation

You asked A11 to prefer A9's own re-translation of A8 "if A9 Part B saved" one. It did not: Part
B translated the A8 model again, scored it, and discarded the translations. Eric asked for A9 to
be updated, so Part B now writes them to **`test-a8-again.hyp`**, beside `test-a9.hyp`, and A11
uses that file when it exists.

**No student needs to rerun A9.** Everyone who did A9 before today has no `test-a8-again.hyp`,
so A11 uses A8's `test.hyp`: the same model, decoded the same greedy way, so normally the same
translations. They would differ only if a student retrained A8 after writing `test.hyp`. For
that case the cell prints a note: if A8's BLEU here differs from the one A9 printed, say so.
If the A11 page or the A9 directions name A9's files, `test-a8-again.hyp` is the new one.

**Where an old A9's output comes from.** A9's Part B has always written `test-a9.hyp` to
`MyDrive/CS479/assignment8`, including the version students ran before Oct 14, so a student who
finished A9 has it there already. If it is gone (moved, deleted, or Part B never finished), Step 6
stops and names the file and A9 Part B as its writer. The fix, if `best-a9/model_best.pt` is still
on Drive, is a few minutes on a GPU: in the A9 notebook, the install cell, Part B cell 1 and the
`config` cell, then the scoring cell, skipping training. Only a student whose A9 model is gone
would have to retrain, for hours; what to accept then is Eric's call. One line on the A11 page
saying so may save some questions.

## 3. As you specified, with these choices

- **Rerun-safe in two parts.** The scoring cell loads both models itself. The report is its own
  cell and reads the JSON, so after a dropped session it needs only the top two cells and Step 3
  (for `pairwise_agreement`). Scoring again after a dropped session takes the top two cells,
  Steps 1 to 3 and the login cell, then Step 6, because `comet_inputs`' cell checks the A6 files
  from Step 1. The markdown says both.
- **The coverage note** is in Step 6's markdown, with a link to COMET's list of languages. The
  notebook cannot print a reminder: nothing in the A8 files names the language.
- **The roadmap**, which is ours: Lecture 11's outcome now reads "Run COMET and COMETKiwi on their
  own A6 data and on their own A8 and A9 test output; reconcile them ...", and the notebook's
  intro quotes it. A11 is 30 points with five summary questions in the scope notes, and the
  decisions log has an Oct 21 afternoon entry.

## 4. How it was checked

The step's cells ran here with COMET stubbed, on a toy A8 folder where COMET and chrF disagree on
half the sentences. Tests cover a missing `test-a9.hyp` (the message names A9's Part B), a file
whose line count differs from the test set, `test-a8-again.hyp` preferred over `test.hyp`, an
empty last translation kept as a line, Step 6 run alone in a later session, and the five
questions. The A9 test checks that `test-a8-again.hyp` holds the translations it scored.
Preflight passed. Not run on a Colab GPU: Step 6's scoring is Step 4's loop on other files, and
A10, which runs the same two models, has proved that path there.

## Nothing asked of you

Unless the page's wording needs `test-a8-again.hyp`, nothing here waits on you.
