# New due dates, a simpler A13, A14's size, A15 dropped

*From the Cowork session, written Wed Oct 21 2026 about 23:50 and revised Thu Oct 22 about 00:40 and 06:50 (Eric
changed the schedule and the scope before this note was read; this version replaces the earlier one).
Answers `to-cowork/2026-10-21-c-a12-a13-on-course.md` ("Asked of you", items 3 and 4). The baton is yours
once you read this.*

## 1. The schedule (Eric, Oct 22): no two assignment windows overlap

Each assignment's window now ends when the next one opens, all due at 11:59 pm:

| | Window | Days |
|---|---|---|
| A12 | Mon Oct 26 to **Fri Oct 30** | 4 |
| A13 | Fri Oct 30 to **Wed Nov 4** | 5 |
| A14 | Wed Nov 4 to **Tue Nov 10** | 6 |
| A15 | **dropped this term**, folded into the Final Project | |

The gradebook already shows these. A14 may overlap the Nov 9 outline review (Eric) and ends before the
Nov 11 proposal presentations. Please update the roadmap sources (`roadmap.yaml`'s assignment entries
A12 to A15 and their windows, the Lecture 12 to 15 outcome lines' `shown_by`, and
`sections/16-decisions-log.md`), render, and commit.

**One code change first:** A12's Fri Oct 30 and A14's Tue Nov 10 fall on days with no schedule row, and
`render_roadmap.py` rejects that (`test_an_assignment_due_off_the_schedule_is_rejected`). Either let an
assignment be due between lecture rows (shown on the row before, with its date and time stated), or add
a due time and day to the label; your call. A15 needs a way to be listed as dropped, or removed from the
schedule with a decisions-log line. The final-project proposal entry says Nov 16 ("may move"); Learning
Suite has the Final Project Proposal due Mon Nov 9 at 10:00, the outline-review day, and the
presentations on the Nov 11 and Nov 16 attendance quizzes. Please make the roadmap match.

**A15:** the gradebook entry is hidden and out of the final grade; the page is unpublished, not deleted,
and kept for next year; its 20 points are dropped. The Final Project page's Path B now names a
speech-to-speech pipeline (speech recognition, the student's own MT, speech synthesis, with word error
rate and one sentence traced through the three stages) as an example. The deferred A15 notebook stays
deferred, for next year.

## 1b. Fold the assignment arc plan into the roadmap (Eric, Oct 22)

Cowork's `Assignment Arc Plan_v6.md` is retired (archived) now that every decision in it is taken; the
roadmap becomes the one record. Three moves, all in the roadmap sources:

**(a) Lecture outcome lines** in `roadmap.yaml` (`lecture_outcomes`), matching the live pages:

- Lecture 12, `can:` "Prompt a decoder-only model to translate a low-resource language from a few
  example pairs, with no training; measure how much random and nearest examples improve translation,
  and tell a real difference from the spread between random draws; explain how examples in the prompt
  help or hurt when the model's weights never change, and what sentence-by-sentence translation cannot
  fix." `shown_by:` A12 (unchanged otherwise).
- Lecture 13, `can:` "Say what makes a language low-resource beyond its count of sentence pairs; use
  monolingual text to make up for scarce parallel data (back-translate it with a reverse model, train on
  the synthetic pairs, and measure whether it helped); tell how much of a change came from the new
  training data and how much from the decoding method, and test whether a difference is real."
  `shown_by:` A13.
- Lecture 14: unchanged, `shown_by:` A14.
- Lecture 15: keep the `can:` line; `shown_by:` "the Final Project's Path B example (A15 dropped this
  term)".

**(b) A new subsection** in `sections/21-curriculum-outcomes.md`, after "Where learning outcomes live":

> ### Practice before the Final Project
>
> Each Final Project requirement is met at least once in an assignment first. Fall 2026:
>
> | Final Project requirement | Practiced in |
> |---|---|
> | A held-out test with an overlap check | A8, A9, A14 |
> | BLEU or chrF plus COMET | A6, A10, A11, A13 |
> | One change against a named baseline (Path A) | A9, A12, A13, A14 |
> | A hypothesis and a prediction before any score | A9 to A14 |
> | Is the difference real (paired bootstrap) | A13, A14 |
> | Where it helped and where it hurt | A13 (A9 next offering) |
> | Rerunnable (versions, seeds, decoding) | A12 to A14 |
> | The completed notebook submitted | A7, A11 to A14 (A8 to A10 next offering) |
> | At least 3 raters, and their agreement | A6 (one rater); first met in the project itself |
> | An application for an audience (Path B) | the speech-pipeline example on the Final Project page |
>
> The gaps (raters, Path B) are where the project asks for something new; `Next Offering Notes`
> in the course folder carries the fixes.

**(c) Decisions log** (`sections/16-decisions-log.md`), newest first, above "Oct 21, evening":

> - **Oct 22.** No two assignment windows overlap (Eric). A12 is due Fri Oct 30 (moved for breathing
>   room, Eric), A13 Wed Nov 4 and A14 Tue Nov 10, all at 11:59 pm, each opening when the one before
>   closes; A14 may overlap the Nov 9 outline review (Eric) and ends before the Nov 11 proposal
>   presentations. A15 is folded into the Final Project (Eric): its page unpublished and kept for next
>   year, its 20 points dropped, the speech pipeline named as a Path B example. A13 is simplified after
>   Eric asked whether any assignment had grown too complicated for its outcomes: one retrain on sampled
>   back-translations instead of greedy and sampled (the Oct 21 entry's three trainings become two,
>   one with NLLB), the MTEval partner exchange dropped, seven questions, 40 points to 30 (Eric). A12
>   keeps its work; its outcomes go from five to three, and both pages' outcomes now state the
>   low-resource lesson (Eric). The assignment arc plan is retired into the roadmap (Eric).

## 2. A13: simpler (Eric agreed, Oct 22). Notebook changes, please

The A13 page is live with these changes; the notebook should match it. **Do this first**, before A14:
A13 opens Fri Oct 30, so the simplified notebook should be on `main` by **Wed Oct 28** (Lecture 13).

- **One retrain, on sampled back-translations only.** Drop the greedy back-translation and its retrain.
  The student now trains the reverse model (or uses NLLB) and retrains once: two trainings, or one with
  NLLB. Outputs: A9 and A13, each decoded greedy and beam 5, so **four** test outputs, BLEU, chrF, the
  paired bootstrap (A13 against A9), and COMET on the four.
- **Drop the MTEval partner exchange** (old Step 6, `a13-mteval`, `read_ranks`, the agreement count) and
  its write-up question. Reason: a step that depends on a classmate stalls two people in a 5-day window,
  and it has nothing to do with back-translation. (My earlier MTEval notes, `reference.txt` as the
  source, are moot.)
- **Write-up questions, now seven** (the page's wording; please mirror it in the notebook's last cell):
  1. Hypothesis against the result: BLEU and chrF with intervals for A9 and A13, COMET system scores,
     the bootstrap, what the 95% interval means.
  2. **Why sample the back-translations?** Find two sentences in your back-translated English and say
     what in them would help or hurt a model learning from them. Edunov et al. (2018) found sampling
     helps in all but resource-poor settings: is your language one, and what does your result suggest?
  3. **How much came from the data, and how much from decoding?** (greedy and beam rows; the question's
     body as before, its title reworded Oct 22 morning).
  4. Which side of each pair is real, as before.
  5. What makes your language low-resource, as before.
  6. Where it helped and where it hurt, as before.
  7. How to rerun: `a13-rerun.json`.
- **Points:** 30 (was 40; Eric, Oct 22), so the notebook should not state 40 anywhere.
- **Uploads:** `train-a13.tsv` (one training set), `val.tsv`, `test.tsv`, every `test-*.hyp`,
  `a13-rerun.json`, the write-up, the notebook.
- Learning Outcomes on the page, now three: what makes a language low-resource beyond pair counts;
  using monolingual text to make up for scarce parallel data (back-translate with a reverse model, train
  on the synthetic pairs, measure whether it helped); how much of a change came from the new training
  data and how much from the decoding method (greedy or beam search), and whether a difference is real. If the notebook's opening cell lists outcomes, mirror these.
- The page says A13 "trains twice for a student with their own back-translator (once with NLLB) ... plus
  about 12 minutes to back-translate 100,000 sentences." Correct it if your timings say otherwise.

## 3. A12: outcomes only; the notebook is unchanged

The page's outcomes went from five to three (keeping the pool apart is a step, question 4; NLLB covers
only Telugu, so it left the outcomes but question 5 stays), reworded Oct 22 morning so the low-resource
lesson is stated: (1) prompt a decoder-only model to translate a low-resource language from a few example
pairs, with no training; (2) measure how much random and nearest examples improve translation, and tell a real difference
from the spread between random draws; (3) explain how examples in the prompt help or hurt when the
model's weights never change, and what sentence-by-sentence translation cannot fix. No notebook change,
unless its opening cell lists outcomes; then mirror these.

## 4. A14: please build the notebook now (Eric, Oct 22)

**Go on #123 and #99.** They were waiting on A14's due date; it is settled (below). Build
`docs/docs/course/lecture-14-multilingual.ipynb` (or your naming), the A14 course notebook, to the
design below, and merge it to `main` by **Mon Nov 2** (Lecture 14), so Cowork can write the page and
directions from it and the page can open Wed Nov 4. If #123 (the two-directions case, never run) turns
up a blocker, then say so in your `to-cowork` note at once rather than at the end.

- **Due Tue Nov 10** (above), opening Wed Nov 4 when A13 closes.
- **Size:** 6 days removes the need to halve the run. Train the bidirectional model on A9's training set
  in both directions (twice A9's pairs, each tagged with its target) for **A9's epochs**, or half if
  your timings put a full run past about 3 hours on an L4. Same architecture and pieces as A9; the
  SentencePiece model trained on both sides as A9's already is, if that holds; otherwise say what changes.
- **The question:** does one model that translates both ways match two one-way models? English to X
  against A9; X to English against the student's A13 reverse model. **Fallback:** a student who used NLLB
  in A13 has no reverse model of their own, so they compare X to English against NLLB and say so, or
  compare English to X only. One stated decoder for all (greedy, and beam 5 as A13 does), the paired
  bootstrap, and the write-up question on tagging and why zero-shot works at all (the A14 outcome).
- **Notebook submitted**, with an "If ..., then ..." hypothesis and a prediction before any score
  prints, outcomes and rerun file, as in A12 and A13 (the new roadmap table in 1b counts on it).
- **Keep it small:** the two original outcomes plus the comparison. Eric asked tonight whether any
  assignment had grown too complicated for its outcomes, and A13 had.
- **Please report, in your `to-cowork` note:** the measured wall-clock time of the A14 training run on
  an L4 (and on an A100 if you have it), at the epochs you chose, beside A9's. Cowork uses it to confirm
  A14's difficulty (estimated 7 of 10) for the course's points-against-difficulty table, and to tell
  students how long to budget.

## 5. Where to commit what

- **torchlingo (this repository):** your usual branch and PR flow. Please also commit this note
  (it is untracked in `notes/handoff/from-cowork/`) and the archive of earlier notes if you move them.
- **torchlingo-private** (`github.com/byu-matrix-lab/torchlingo-private`, branch `main`, on Eric's
  machine at `~/Projects/torchlingo-private`): Cowork has pushed everything; `origin/main` is
  **`193eea7`** and the working tree is clean. Pull (fast-forward) before you start. Commit directly
  to `main` and push to `origin main`; no branches or PRs there.
  - **Yours:** `course/data/` (sample data, such as the A6 MTEval file), anything under `runs/`,
    `checkpoints/`, `hpc/` and `scripts/` that you already own, and any new sample outputs the A13
    or A14 notebooks need (put them in `course/data/a13-...` or `a14-...`).
  - **Cowork's; please do not edit:** `notes/assignments/*-directions.md` and its README (transcripts
    of the live Learning Suite pages), `course/build/` and `course/figures/` (deck scripts and
    figures). If one of those needs a change, then say so in your `to-cowork` note.
  - **Before you hand the baton back,** push, and name your last torchlingo-private commit in the
    `to-cowork` note, so Cowork starts from it. Cowork does not commit to torchlingo-private while
    you hold the baton.

## 6. Already settled, for the record

A12's additions are required. A13's embedding filter is cut. Partners and MTEval: moot (above). A12
students who finished with the old notebook rerun (no one has started).

## Course side, next (Cowork)

When the A14 notebook lands: the A14 page (Learning Outcomes, What to do, What to submit, Rubric at
30 points), `A14-directions.md`, and the A14 slide in the Lecture 14 deck. The Lecture 12 deck `_v3`
(A12 "Due Friday") awaits Eric's promotion. Done this morning: the A12 and A13 pages and directions,
gradebook dates and points, A15 hidden, the Final Project page's Path B example, the arc plan retired.
