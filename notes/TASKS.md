# TorchLingo — Session Task List

Opened 2026-08-22, last updated 2026-09-26. Numbered for reference in conversation.
Completed work is removed rather than marked done — git history is the record.

**Numbers here are task numbers, and they collide with pull request numbers.**
Tasks run to #125 and PRs to #100, so every number below 101 names one of each. Say
"Task #37" or "PR #37" in conversation and in GitHub comments; a bare `#37` is
ambiguous, and on GitHub it auto-links to the pull request whether or not that
was meant.

## This file is the whole list; the session mirror is only what is urgent

**Convention, Eric's, 2026-09-25.** A task stays on the in-session task list only if it
contributes to **Lecture 7 (Mon Sep 28)** or **Lecture 8 (Wed Sep 30)**. Everything else
lives here and here alone.

That keeps the working list short enough to be read, without losing anything: this file
is authoritative and always has every task. The mirror is a filter over it, not a second
copy of it.

What passes the filter as of 2026-09-27: **#152**, the A8 kickoff notebook, because Lecture 8a
is Wed Sep 30; **#121**, because Lecture 9 has nothing else on its own subject; and **#135**,
because the kickoff notebook has a student start a real training run and the examples they copy
from still do not checkpoint.

**Read the "Critical for" column rather than the dates.** Labelling every task by lecture showed
that only fifteen of sixty-two are critical to a lecture in the next ten days — and that two of
the nearest-looking dates, both on Lecture 6's notebook, belonged to a lecture that had already
run.

Lecture 7 runs Monday with tutorial 2 as its activity. Lecture 8 is split into 8a and 8b, both
decks built, and Task #42 closed on Cowork's word that it needs nothing.

The test to apply is *"must this be done for Lecture 7 or Lecture 8 to happen
correctly?"* — not "is this related to them". #44 failed that test on inspection: it
gates the sdist against shipping a Git LFS pointer, which sounds release-critical until
you check, and the installed wheel turns out to carry **no data files at all**.

Everything here is work that can be finished and then deleted. Standing rules live in
`CLAUDE.md`; decisions and findings live where they apply. `notes/README.md` maps the
rest of this directory, including the `handoff/` protocol used to exchange messages
with the Cowork session rebuilding the CS 479 decks.

## Status

"In review" means the work is written and sitting in an open pull request; the task
is not finished until that PR merges, and only then does the row disappear.

**Tasks #93 to #100 have external deadlines.** They come from
`notes/CS479_COURSE_ROADMAP.md`: CS 479 is pivoting to TorchLingo *this* semester, and
eighteen students hit the library in class on **Mon Sep 28**. Those tasks are dated, they
are sequenced against a calendar nobody here controls, and they outrank everything else in
this file until Oct 28. See "The CS 479 pivot" below for the schedule and the reasoning.

### Reading the "Critical for" column

Eric, 2026-09-27: annotate every task with the lecture it is critical for. Assigned by what
each task actually touches, **not** by how near its date is — several tasks with imminent
dates are not lecture-critical, and several with no date are.

| label | means | date |
|---|---|---|
| **L6** | Human and Automatic MT Evaluation — **has already run** | Wed Sep 23 · A6 due Mon Sep 28 |
| **L7** | Research Paper Reviews; Intro to Neural Networks | Mon Sep 28 · A6 due |
| **L8a** | Neural MT: Encoder-Decoder, and Why Attention Was Invented | Wed Sep 30 · **A8 introduced** |
| **L8b** | Neural MT: The Transformer | Mon Oct 5 |
| **L8a/8b** | both halves of the split | |
| **L9** | Morphology and Terminology in NMT | Wed Oct 7 · **A8 due, A9 set** |
| **10+** | Lecture 10 onward, or no lecture depends on it | |
| **hyg** | roadmap and process integrity — **critical, but not to one lecture** | |

**A lecture that has already run still owns its work.** `L6` means "this belongs to Lecture
6", not "this is overdue" — the deadline pressure is gone, the subject matter is not, and a
student revisiting the lecture or an instructor preparing Fall 2027 meets it there. Filing
those as `10+` lost the one piece of information the label exists to carry.

`hyg` is not a synonym for "later". It is the category that keeps the other labels true: a
generated map that has drifted, a gate that reports without blocking, or a task list that
describes last week will mislead every lecture at once rather than one of them.

| | Task | Critical for | State |
|---|---|---|---|
| #152 | **A8 kickoff notebook for Lecture 8a** | **L8a** | **Wed Sep 30** — say by Tue Sep 29 if it will not be ready |
| #153 | Pre-norm answered for Cowork: `SimpleTransformer` is post-norm | **L8b** | **Answered** — deliver in the next hand-off; both 8b slides stand |
| #154 | Execute only the zero-dependency notebooks | **hyg** | **Decided: keep CI lightweight** — structural half merged |
| #155 | A8's "What To Do" overlaps the kickoff notebook's first four steps | **L8a** | **Your call** — Cowork flagged it and did not act |
| #144 | Tutorial 2 is a tutorial used as Lecture 7's in-class activity | **L7** | **Your call** — the one notebook where family and role disagree |
| #145 | Say which assignment each notebook jump-starts | **hyg** | **Your call** — four proposed; blocks #146 |
| #146 | Stamp the purpose cell into all ten notebooks, and gate it | **hyg** | Open — mechanics done in PR #117; waits on #144 and #145 |
| #147 | A9 has no notebook, and one notebook would serve it and Lecture 9 | **L9** | **Due Mon Oct 12** — raises #121's value; do them as one |
| #148 | Five assignments are missing from the schedule | **hyg** | Open — asked in the ninth entry, **unanswered**; see #156 |
| #156 | The restructure orphaned #148's unanswered question | **hyg** | **Before the next baton** — re-raise it where they will read |
| #132 | Quick Start has no notebook, and its badge opens a different one | **10+** | Open — **was only ever in the session mirror** |
| #149 | `collect_benchmark.py` silently drops a run file it cannot find | **10+** | Open — it wrote a 2-run report over a 21-run source |
| #150 | `torchlingo-private` has no git remote, so nothing in it is backed up | **10+** | **Your call** — it holds the corpus prep and all the HPC tooling |
| #151 | The LSTM asks for dropout it cannot apply | **10+** | Open — nine tests warn; a student setting it gets nothing |
| #139 | Lecture 9 is claimed by tutorial 1 and not actually served | **L9** | Open — **folded into #121**; tutorial 1 keeps 9 only as `reference` |
| #140 | Split tutorial 4 at Part 6 — Parts 6 to 8 are Lecture 8 material | **L8a/8b** | Open — **a Lecture 8 split makes this land somewhere** |
| #142 | Decide which notebook owns BLEU before splitting either | **10+** | Open — tutorial 3 Part 5 versus the planned tutorial 7 |
| #49 | The shipped checkpoint predates the enlarged corpus | **L8b** | Open — `train_pairs` 64,311 against a corpus of 86,430 |
| #120 | The grader now has a source repository | **10+** | Open — point the course at it; decide on diagnostics |
| #121 | A Lecture 9 subword notebook, and it is ours | **L9** | **Lecture 9 is Wed Oct 7; A9 Wed Oct 14** — the only notebook L9 has on its own subject |
| #122 | Make Assignment 9's control hard to get wrong in code | **L9** | Open — worth more than the wording fix |
| #123 | A14's two-directions case has never been run | **10+** | Open — highest uncertainty, due Oct 28 |
| #135 | `examples/*.py` still do not checkpoint | **L8a** | Open — the five a student is most likely to copy |
| #129 | Extract a shared `~/Projects/hpc` | **10+** | Open — after #128 gives a second implementation to diff |
| #97 | SentencePiece on versus off, controlled | **L9** | **Due Mon Oct 12** |
| #102 | Inference cannot resume a long decode | **10+** | **Needed by Mon Oct 19** — largest undone piece |
| #98 | Back-translation as a documented workflow | **10+** | **Due Mon Oct 26** |
| #99 | Multilingual tagging tutorial, replacing the OpenNMT handout | **10+** | **Due Wed Oct 28** |
| #101 | Give the tutorials stable unique names | **hyg** | Open — **a semester boundary**, not mid-course |
| #103 | Extend the notebook gate to `docs/docs/course/` | **hyg** | Open — four course notebooks on `main`, still ungated |
| #106 | A token cap breaks Assignment 9's control | **L9** | Open — one sentence in the assignment |
| #107 | The optimizations exist and nothing uses them | **10+** | **Half done** — experiments bucket now; library default unchanged |
| #108 | Nothing releases the device allocator's cache | **10+** | Open — **demoted**: length, not cache, is the driver |
| #113 | Land the PRs still open | **hyg** | **PR #127** and **PR #128** — both need your merge |
| #118 | What does a paid Colab session actually provide? | **L8a** | **Coulson** — now blocks a live decision, not a claim |
| #114 | The wheel ships no data, so tutorials 4 and 5 cannot find it | **10+** | Open |
| #8 | Verify Eole claims before syllabus use | **10+** | Open |
| #9 | `pre-commit install` (still not installed) | **hyg** | Open |
| #15 | Migrate history-blind `DummyTransformer` tests | **10+** | Open |
| #22 | `examples/` and `scripts/` are outside the lint gate | **hyg** | Open |
| #28 | Attention params skip `_init_weights` | **L8a/8b** | Open |
| #36 | CI actions pinned to a deprecated Node runtime | **hyg** | Open |
| #44 | Gate the sdist on "no Git LFS pointer shipped" | **hyg** | Open |
| #48 | Audit pedagogical value; write down sequencing and outcomes | **hyg** | In progress — in the roadmap |
| #51 | The docs gate reports but does not block | **hyg** | Open — repo settings |
| #52 | Try Moore (2002) if more of the corpus is wanted | **10+** | Open |
| #53 | Notebook gate runs 2 of 6 tutorials in CI, and looks green | **hyg** | Open |
| #60 | Nobody is told when main goes red | **hyg** | Open |
| #66 | Adopt `nltk.translate.gale_church`; split #29 into two jobs | **10+** | Open |
| #68 | Cite `torcheck` as prior art in the diagnostics docs | **10+** | Open |
| #71 | Decide whether to report the Joey NMT breakage upstream | **10+** | Open — Eric's call |
| #74 | Diagnose the 93 docs warnings | **hyg** | Open — **the anchor half shipped** in PR #51 |
| #79 | An order-dependent test | **hyg** | **Fails on unmodified `main`** — so it blocks nothing; CI cannot see it |
| #81 | Fail the build on hand-typed generated numbers | **hyg** | Open |
| #82 | Add an on-target language check to `torchlingo.diagnostics` | **10+** | Open |
| #85 | Only BLEU carries a signature; chrF and TER do not | **L6** | Confirmed 2026-09-27 — Lecture 6's notebook now reports both |
| #86 | `evaluate_model` has no test, and it is what callers use | **10+** | Open — after PR #58 |
| #88 | Open the tutorial 7 PR | **10+** | Open — nothing blocks it; **see #142 first** |
| #89 | Fail the docs build when a page is off-nav | **hyg** | Open |
| #91 | `metric_comparison.json` records no BLEU signature | **10+** | Open — nothing blocks it |
| #92 | Tutorials 3 and 5 bypass the library's own evaluation API | **10+** | Open |

## The CS 479 pivot

From `notes/CS479_COURSE_ROADMAP.md`, handed over 2026-09-24. OpenNMT-py is in maintenance
mode and CS 479 is moving to TorchLingo **this** semester. Five assignments train a model
(Lectures 7, 8, 9, 13, 14), all five are written against OpenNMT today, and they are
consecutive and cumulative, so it is one crossing rather than five.

This converts the repository from a teaching library with a hypothetical audience into the
thing eighteen students have to get working, on their own data, on a deadline.

### Context lives in the roadmap, not here

The calendar, the assignment sequence, the ordering argument and the paid-Colab
requirement are all in **`notes/CS479_COURSE_ROADMAP.md`** (v3, maintained by the Cowork
session) and in **`notes/handoff/briefing.md`**. Both are kept current; restating them here
produced two copies that disagreed within days.

What stays below are the dated tasks themselves.

### #97 SentencePiece on versus off, controlled

Assignment 9 retrains Assignment 8's system with SentencePiece so the two can be compared.
The comparison is the assignment, so the only thing that may differ between the two runs is
the tokenizer. Same concern as #59: nothing currently checks that a comparison controlled
its variables.

### #98 Back-translation as a documented workflow

Train X-to-English on reversed data, back-translate held-out target sentences, add the
synthetic pairs, retrain English-to-X, compare. Reversing the direction should be a
configuration change rather than a second project.

### #99 Multilingual tagging tutorial

Bidirectional English and X from the Assignment 8/9 system, directions intermingled,
separate test sets per direction, target-language tagging. Replaces the
"MNMT Guide Using OpenNMT.docx" handout.

### #102 Inference cannot resume, so a long decode cannot survive an interruption

**The largest undone piece for the second half.** Training has checkpoint-and-resume, verified
in Colab. Inference has nothing.

Assignment 13 back-translates at least as many sentences as the training set, so 100,000 or
more. **Throughput is not the problem** — that was the expected answer and the wrong one: the
decode benchmark measures 27.5 ms per sentence batched, so 100K extrapolates to well under an
hour. Read it as an order of magnitude, since it was measured on 8 sentences at `max_len=25`.

The problem is that a multi-hour decode dying at hour two starts from zero, which is the
failure that made resume a priority for training one level up.

Either **write output incrementally and skip inputs already done** — better for students, since
it asks no discipline of them — or **decode in explicit shards**, cheaper to build but relying
on the student following the workflow.

**Done when** an interrupted bulk decode can be restarted without redoing finished work.
Needed for Assignment 13's material, due in class **Mon Oct 19**.

### #103 Extend the notebook gate to `docs/docs/course/`

`scripts/execute_notebooks.py` globs **only** `docs/docs/tutorials/*.ipynb`, and
mkdocs-jupyter runs with `execute: false` and `allow_errors: true`. So a notebook anywhere
else can rot completely and neither the docs build nor a reader surfaces it.

The Cowork session has been asked to write lecture notebooks into
`docs/docs/course/lecture-NN-<slug>.ipynb`. The moment the first one lands it is ungated.

- Extend the glob, and add `REQUIREMENTS` entries so anything needing LFS data or a GPU
  **skips** rather than fails.
- Decide whether a GPU-dependent course notebook can be gated at all, or should be declared
  exempt explicitly rather than silently.
- Makes #53 worse until #53 is fixed: a second directory widens the gap between what the
  green check proves and what it appears to prove.

Blocked until the first course notebook exists; there is nothing to gate before that.

### #107 The optimizations already exist and nothing uses them

Measured on the 100K split, real tokenizer, batch 64:

```
real tokens                2.90 M per epoch
random batching, padded   10.97 M    3.79x waste
length-bucketed, padded    2.90 M    1.00x waste   -> 74% saving
```

`BucketBatchSampler`, `create_dataloaders(use_bucketing=...)`, `train_model(use_amp=...)`,
`num_workers` and `pin_memory` all already exist and **all default off**, so the saving is
available and unclaimed by default.

**Half done 2026-09-26:** the experiments bucket now (`ladder.py`, `benchmark_a8.py`). The
library default is unchanged.

**Done when** a decision is recorded on whether `use_bucketing` should default True. It changes
batch composition and therefore results, which argues against flipping it silently — but 74%
matters enormously on a Colab budget, so the course guidance should say to enable it even if the
default stays.

Not a gap: decoding is already optimized, 27.5 ms against 109.5 ms, because `inference_fast`
batches beams within a sentence.

### #108 Nothing releases the device allocator's cache

The one genuine absence rather than an unused flag: there is no `empty_cache()` call
anywhere in `src/torchlingo/`. On CUDA that is usually harmless. On Metal it is not,
because the memory is the machine's, and a spike is held against everything else running.

- Smallest useful version: release at each epoch boundary, on whichever backend is
  active, with a comment saying why. Epoch granularity is far too coarse to cost
  throughput.
- Bigger and not obviously right: cap a batch by total tokens rather than sentence count,
  which bounds the worst case instead of releasing after it. Real MT toolkits do this, but
  it changes what `batch_size` means and a teaching library should not do that lightly.

### #113 Land the five PRs still open

**Not blocking anything.** 0.2.0 is published and verified from PyPI, so what remains is
improvement rather than repair.

Open: **PR #51** nav entries, **PR #52** tutorial 6 imports, **PR #54** length normalization,
**PR #57** the causal-mask convention, **PR #59** Transformer attention.

**PR #55 merged 2026-09-26** and the signature is verified live:
`nrefs:1|case:mixed|eff:no|tok:13a|smooth:exp|version:2.6.0`. That closed #70 and unblocked
#85 and #91.

**All are refreshed and green against today's `main`.** Refresh again before merging any that
sit: an old green tick was taken against an old base, and these have already been found
missing a gate that `main` gained after their checks ran.

**Done when** all six are merged. They need a reviewer, which is Coulson.

Knock-on: **PR #55** is what **Task #85** and **Task #91** wait on. (Task, not PR: PRs #84 and #85 exist and
are unrelated — the collision `CLAUDE.md`'s numbering rule describes.)

### #135 `examples/*.py` still do not checkpoint

The #103 audit found that no `train_model` caller checkpointed. Three are fixed —
`benchmark_a8.py`, `ladder.py`, `train_example_model.py` — and five are not:
`examples/train.py`, `attention_alignment.py`, `multilingual_training_example.py`,
`train_ceb_cmn_simple.py`, and the `TRAINING_GUIDE_CEB_CMN.md` walkthrough.

**These matter most and were done last**, which is the wrong way round: `examples/` is the code
a student is most likely to copy into their own Colab notebook, and a Colab disconnect is
exactly the failure `training_checkpoint` exists for.

Left separate deliberately: those five files sit outside the lint gate (#22), so a change there
is unguarded, and a five-file mechanical diff would bury its own review.

### #144 Tutorial 2 is a tutorial used as Lecture 7's in-class activity

**Eric, 2026-09-26:** the tutorials are out-of-class and the course notebooks are in-class
active learning. True of every notebook but one.

`02-train-tiny-model` lives in `docs/docs/tutorials/` and its declared role is `activity` —
Lecture 7's in-class activity, which is what the roadmap's table said before the metadata
existed. So for this notebook the collection and the use disagree.

That was worth surfacing rather than smoothing over, and it corrected the tooling: `family`
says where a notebook *lives*, `role` says how it is *used*, and a reader is told the role. The
two are independent fields for exactly this case.

**The remaining question is yours.** Either tutorial 2 really is Lecture 7's activity — in which
case nothing changes and the exception is deliberate — or Lecture 7 should get a course notebook
of its own and tutorial 2 reverts to out-of-class reading. The metadata will say whichever you
decide; today it says the first.

Worth weighing alongside #145: tutorial 2 is also the clearest A8 jump start in the repository,
and "the thing students work through before the big assignment" may be a better description of
it than either in-class or out-of-class.

### #145 Say which assignment each notebook jump-starts

**Eric, 2026-09-26:** in-class notebooks are often a jump start on the next assignment, and some
tutorials should be pivoted to serve that purpose too.

`leads_to` exists for this as of PR #117, validated against the eleven assignments parsed out of
the schedule's own column, so a notebook states its deadline without repeating it.

**Four proposals. Which assignment a notebook prepares is a pedagogical call, so these are
arguments rather than decisions:**

| notebook | assignment | why |
|---|---|---|
| `02-train-tiny-model` | **A8**, due Lecture 10 | A8 is "create and run an NMT model". Tutorial 2 *is* that at small scale — prepare, build, train, test, save. The clearest one in the repository. |
| `05-real-translations` | **A8** | The "now run it on unseen data" half, and it already carries the pretrained checkpoint. |
| `01-data-and-vocab` | **A5**, due Lecture 6 | Its Part 1 is loading and cleaning a parallel corpus. Retrospective this year, real next. |
| a Lecture 9 notebook | **A9**, due Lecture 11 | Does not exist yet. See #147. |

**Deliberately not proposed:** `03-inference-and-beamsearch`. Its BLEU half points at A6 or A11,
but #142 has not settled which notebook owns evaluation, and pointing it at an assignment before
that is decided would make the question harder to answer.

### #146 Stamp the purpose cell into all ten notebooks, and gate it

The mechanics landed in PR #117 and are idempotent — verified stable over four runs and an
in-place update, on one notebook of each format version. What remains is applying them and
adding the banner to `--check` so it cannot drift.

**Held deliberately until #144 and #145 are answered**, because the cell is student-facing text
and the wording encodes decisions that are not mine: whether tutorial 2 is an in-class activity,
and which assignment each notebook starts.

Two smaller choices to make at the same time: the cell currently sits **before** the title rather
than after it, and the exact phrasing of the opening sentence.

### #147 A9 has no notebook, and one notebook would serve it and Lecture 9

A9 is SentencePiece, **due Mon Oct 12** at Lecture 11. Lecture 9 is morphology and terminology
and has no notebook either (#121, answer owed to Cowork by Oct 3).

These are one piece of work, not two. A subword notebook serves the lecture *and* starts the
assignment, and #139 found the opening for free: tutorial 1's Part 2 ends on `<unk>` for "Hello
universe", which is the motivating example for everything A9 asks a student to do.

So #121's value is higher than it looked. Build it once, declare `serves_lectures: [9]` and
`leads_to: ["A9"]`, and two gaps close together.

### #148 Five assignments are missing from the schedule

The schedule names A4, A5, A6, A8, A9, A10, A11, A12, A13, A14 and A16. **A1, A2, A3, A7 and
A15 appear nowhere in the roadmap at all** — checked 2026-09-26, not inferred from the gaps.

Either the numbering genuinely skips them, or the single source of truth is missing five
assignments. Worth establishing which, because it is no longer only a documentation question:
`leads_to` validates against this table, so a notebook cannot declare that it jump-starts an
assignment the schedule does not name. If A7 exists and is unlisted, the validator will reject a
correct declaration.

### #151 The LSTM asks for dropout it cannot apply

Found 2026-09-26 while checking whether #7's deprecation warnings were really gone. They are —
but the suite's ten remaining warnings are one numpy division and **nine instances of this**:

> `UserWarning: dropout option adds dropout after all but last recurrent layer, so non-zero
> dropout expects num_layers greater than 1, but got dropout=0.2 and num_layers=1`

PyTorch applies recurrent dropout *between* layers, so with one layer there is nowhere to put it
and the value is discarded. The model is built with `dropout=0.2` and one layer, so **the dropout
does nothing and nothing says so** except a warning nobody reads.

**Why this matters more here than in a normal library.** A student who sets `dropout` on an LSTM
to reduce overfitting will see no change, conclude that dropout does not help, and be wrong. The
library's job is to make the concept legible, and this quietly teaches the opposite of the truth.

Worth deciding rather than patching: either pass `dropout=0.0` when `num_layers == 1` and say why
in the docstring, or default the LSTM to two layers. The first is honest about the limitation; the
second makes the knob work. Not the same choice, and the second changes a default.

### #132 Quick Start has no notebook, and its badge opens a different one

`docs/docs/getting-started/quickstart.md` carries a Colab badge, and the badge opens
`tutorials/02-train-tiny-model.ipynb` — a different document from the one the reader is on.

So the page a newcomer is pointed at first has no runnable form of its own, and the button
that looks like "run this" runs something else. Whichever way it is fixed — give Quick Start
its own notebook, or change the badge to say what it opens — the present state is the one that
misleads.

**Recovered from the session task list on 2026-09-27**, where it had been living *only*. It was
found by checking, before deleting the session's copy, that each task existed here too — and
this one did not. Four of the five checked out; this is why the check was worth running.

The general lesson is worth more than the task: the session list is a mirror, and a mirror that
holds something the original does not is not a mirror. Anything created there has to be written
here before the session ends.

### #149 `collect_benchmark.py` silently drops a run file it cannot find

`rows = [run_record(p) for p in args.runs if p.exists()]`. A path that does not exist is
filtered out without a word, and the script then reports how many runs it *did* write — which
reads like success.

**It happened on 2026-09-26 and wrote a two-run report over a twenty-one-run source.** The cause
was `zsh` not word-splitting an unquoted `$RUNS`, so all 21 paths arrived as one long
nonexistent path. The report regenerated cleanly, with a correct-looking table, from two runs.

Caught only because the summary it prints listed two rows where twenty-three were expected. A
reader who trusted the file would have published a curve with the entire sweep missing.

**Fix:** fail on a named path that does not exist, and say which. `--runs` names files
explicitly, so a missing one is always a mistake rather than an absence to tolerate. This is the
same shape as #137, which was fixed an hour earlier in the other script: a filter that should
have been an error.

Worth checking `scripts/collect_ladder.py` for the same pattern while in there.

### #150 `torchlingo-private` has no git remote, so nothing in it is backed up

`git remote -v` prints nothing. The repository is local to this machine only.

It holds the corpus preparation scripts, `benchmark_a8.py`, `split_curve.py`, `runtime_guard.py`,
the whole `hpc/` directory, and the findings prose that every published report is generated from.
The German corpus itself is deliberately unversioned and can be rebuilt from the TMX; the
*scripts* cannot be rebuilt from anything.

Committing has been giving a false sense of safety — including to me, which is how this was
found: I reported "pushed private" after a push that had failed for want of a remote.

**Your call**, because it is an account and privacy decision rather than a technical one: a
private GitHub repository under the lab, or a different backup entirely. The `.gitignore` already
keeps the corpus and checkpoints out, so a remote would carry scripts and notes only.

### #152 A8 kickoff notebook for Lecture 8a

**Cowork's request, and it has the nearest deadline on this list: Lecture 8a is Wed Sep 30.**
They asked to be told by **Tue Sep 29** if it will not be ready, because a deck slide pointing at
a notebook that does not exist is worse than no slide.

**Why it exists, which is the strongest argument in the request.** Lectures 4, 5 and 6 each had
an in-class activity that started the assignment. **The largest assignment in the course has
none.** A student leaves Lecture 8a with a training run already going, on their own A5 corpus, or
they leave with a handout.

Ten steps, twenty minutes, on the student's own data:

1. Install and verify — tutorial 2's two-cell pattern.
2. Mount Drive, load their A5 corpus, report the pair count immediately.
3. Deduplicate on the source side; report how many went, count and percentage.
4. Split by source group into 100K / 2K / 2K.
5. `check_contamination`, and make it **loud** — an empty intersection is the thing to see.
6. Length histogram, then the 100-token cap, reporting the percentage dropped on *their* data.
7. Write the six files back to Drive.
8. Build config and model, print the parameter count — d_model 256, 8 heads, 3 + 3.
9. Print ln(V) beside the first loss, so Lecture 7's reference point is on screen.
10. Start training with `val_loader`, `save_dir` on Drive, and checkpointing.

**Cowork's design decision, which they invited pushback on:** the notebook supplies the dedupe
and split code, and the student runs the verification and reads the numbers. That is the right
split for twenty minutes — the judgement is in reading a contamination check and a length
histogram, not in retyping a groupby.

Step 10 is where #135 stops being abstract: this notebook is the one place a student's first real
training run begins, and it must not be the place that teaches them to run without checkpoints.

### #153 Pre-norm answered for Cowork: `SimpleTransformer` is post-norm

**Answered 2026-09-27; deliver it in the next hand-off.** Two slides in the 8b deck depend on it
and both are correct as written.

`SimpleTransformer` constructs `nn.Transformer` without `norm_first`, so PyTorch's default of
`False` applies — confirmed on the built module rather than inferred from the constructor:
`encoder.layers[0].norm_first is False`. That is post-norm, the ordering of the 2017 paper, so
the encoder-block anatomy slide and the residual-stream slide both stand.

Worth keeping the method as much as the answer: the question was settled by building the model
and reading the attribute, because the constructor does not mention the argument and a reader
cannot tell from the call site which default is in force.

### #154 The notebook gate still runs tutorials only, and `course/` has seven

`scripts/execute_notebooks.py` defaults to `docs/docs/tutorials`. Seven notebooks now live in
`docs/docs/course/` — three arrived today — and **none of them is executed by anything.**

This is the older #103 with a sharper edge: the course notebooks are what students open *in the
room*, on a clock, and a broken one costs twenty minutes of class rather than a confusing evening.

The obstacle is real rather than neglect: several need Google Drive, a network download, or a
HuggingFace token, so they cannot all run in CI. The tractable version is to run the ones that
are self-contained and skip the rest by their declared `needs`, which is the mechanism the
tutorials already use.

**Half done 2026-09-27, and the investigation split the task in two.**

**What shipped: every notebook is now checked structurally**, in the lint job, at no CI cost.
`notebook_meta.py --check` verifies valid JSON, a non-empty cell list, no token-shaped strings,
no `INSTRUCTOR` marker, and — for course notebooks — no committed outputs, no execution counts
and a Colab badge. These are exactly the checks that were run *by hand* when three notebooks
arrived from Cowork, so the hand pass is now automated; it recurs on every baton and is the
kind of thing skipped on a busy day.

**One rule inverts between the families, and it nearly shipped wrong.** The first version
flagged all six tutorials for committed outputs. `docs/mkdocs.yml` sets `execute: false`, so a
tutorial's committed outputs *are* what the docs site renders — strip them and the published
page shows code with no results. A course notebook is run from the top in Colab, so its outputs
are dead weight. Scoped accordingly, with a test asserting the inversion in both directions.

**What remains: actually executing them, which needs a dependency decision rather than code.**
The blockers were measured rather than assumed:

| notebook | blocker |
|---|---|
| `lecture-04-regex-refresher` | **none** — pure Python, could run today |
| `lecture-04-tmx-cleaning` | `translate-toolkit`, small |
| `lecture-05-sentence-alignment` | `nltk`, small, may want a corpus download |
| `lecture-03-word-embeddings` | `sentence-transformers` + a model download |
| `lecture-12-llm-context` | `transformers`, `datasets` + a model download |
| `lecture-06-mt-evaluation` | Part 4 uploads through Drive |
| `lecture-10-comet-install` | `unbabel-comet` **and a HuggingFace token** — can never run in CI |

So the question is how much CI time and dependency surface to spend, and it is not one answer:
two are free, two are cheap, two are heavy, and one is impossible. Worth noting that `needs`
cannot express any of this — it holds repo-relative *paths*, and these are *capabilities*, so
gating execution needs a new field rather than a longer list.

### #155 A8's "What To Do" overlaps the kickoff notebook's first four steps

Cowork flagged this and deliberately did not act on it: if the kickoff notebook walks a student
through dedupe, split, contamination check and length cap, then A8's handout describing those
same steps is either redundant or contradictory.

**Eric's to settle**, because it changes an assignment students are about to start. The options
are to trim the handout to what the notebook does not cover, or to leave it as the written record
and accept the duplication.

### #139 Lecture 9 is claimed by tutorial 1 and not actually served

The map says tutorial 1 serves Lectures 4 and 9. **For Lecture 9 that is an over-claim**,
inherited from the hand-written table's "1, the vocabulary half" and now checkable for the first
time.

Read on 2026-09-26: tutorial 1's Part 2 is word-level `SimpleVocab` only. No SentencePiece, no
subwords, no morphology. Its last cell encodes "Hello universe", prints `<unk>`, and stops.

That makes it the **motivating example** for Lecture 9 rather than coverage of it — it ends
precisely at the cliff edge where the lecture's subject begins.

Two consequences:

1. **Do not split tutorial 1 to serve Lecture 9.** Extracting Part 2 would register as coverage
   in the map while teaching the prerequisite, which is worse than the visible gap it replaces.
2. **#121's notebook should start where Part 2 stops** — same corpus, same `<unk>`, then
   subwords. That is a stronger opening than starting cold, and it is free.

Either correct tutorial 1's `serves_lectures` to `[4]`, or leave the 9 and add a `note` saying
it motivates rather than covers. The first is honest; the second keeps the pointer a student
revisiting Lecture 9 would benefit from. Pick one deliberately.

### #140 Split tutorial 4 at Part 6 — Parts 6 to 8 are Lecture 8 material

The only notebook whose content genuinely spans two lectures without saying so:

| parts | subject | lecture |
|---|---|---|
| 1 to 5 — bottleneck, known-alignment task, ablation, alignment accuracy, the picture | measuring alignment | 19, as declared |
| 6 to 8 — Bahdanau versus Luong, the Transformer's mechanism, cross-attention on the real model | architectures | 8, undeclared |

It was previously held back because **Lecture 8 is over-subscribed** — tutorial 6 as reading and
A8 both land there, and a fourth artifact would not have helped.

**A Lecture 8 split changes that verdict.** Two sessions can absorb the architecture half where
one could not, and this is the one split whose seam is in the content rather than in the
paperwork. Sequence it after the split, not before, so it lands in a lecture that exists.

Cost: a new nav entry and a new Colab badge. No rename of the existing file, so no link a
student is holding gets broken — which is why this is separable from #101.

### #142 Decide which notebook owns BLEU before splitting either

Tutorial 3's Parts 1 to 4 are decoding; **Part 5 "BLEU Score Evaluation" and "Understanding
BLEU" are Lecture 6's ground**, not Lecture 22's.

Do not split it yet. A "tutorial 7, evaluation" is planned for the same ground and is unmerged,
so splitting first would produce two evaluation tutorials and a choice nobody made.

**Also a numbering correction:** the hand-off entry that mentions this writes a bare `#88` for
the task. PR #88 is the unrelated rung-5 ladder change, already merged. Task #88 is the
evaluation tutorial. Exactly the collision the naming rule in `CLAUDE.md` exists to prevent.

### #101 Give the tutorials stable unique names

**Still a semester boundary, and the metadata namespace (PR #113) is why it can wait rather than
why it can be skipped.** `serves_lectures` already carries what a filename could not — tutorial 1
naming two lectures, tutorial 4 naming one while containing another's material — so the renaming
question is now purely about *links*, not about expressing the mapping.

Decks cite filenames and Colab badges embed paths, so a rename mid-semester breaks a link a
student is holding. Cowork asked to be told before any rename.

### #129 Extract a shared `~/Projects/hpc`

Two projects now want the same scaffolding, but torchlingo's cluster path is one day old and
mtsurvey's `hpc/bin` is entangled with its own concepts. Extracting from one mature
implementation and one newborn is designing for a shape nobody can see yet.

**Trigger, so this is evidence-based rather than aesthetic:** if writing torchlingo's cluster
scripts means copying more than a few lines out of mtsurvey, extract instead of copy. So far
only conventions have been copied, not code.

Candidates already identifiable: job waiting, array-manifest indexing, the login/venv
environment guard, the no-internet-on-compute-nodes convention, and the babysitter pattern.

### #49 The shipped checkpoint predates the enlarged corpus

**Confirmed still live on 2026-09-26** by reading the checkpoint rather than the note: it
reports `train_pairs: 64311` against a corpus of 86,430 in `data/example.tsv`. So the shipped
model has never seen about a quarter of the data it is meant to represent.

Nothing is broken — the held-out talks are still whole talks and still held out, since the
corpus grew by a strict superset. It is stale rather than wrong.

Worth retraining because everything downstream reads off this one checkpoint: tutorial 5
shows its translations, the decoding sweep measures on it, and its BLEU is the first number a
student meets.

- Rerun `scripts/train_example_model.py`. Its default is now 40 epochs as a ceiling with
  early stopping, so this no longer risks reproducing the undertrained baseline that caused
  the training-budget error — see [`reports/training-budget.md`](reports/training-budget.md).
- Regenerate `docs/docs/_generated/decoding_sweep.json` afterwards. `--rerender` is not
  enough; the sweep itself must re-run, which takes about an hour.
- Not urgent, and **not** on the CS 479 critical path: no assignment depends on it.

### #120 The grader now has a source repository

**Resolved 2026-09-26, and it went the good way.** Eric supplied the source:
**<https://github.com/byu-matrix-lab/data-cleaning-pipeline-grader>**, in the lab's own
organisation.

That retires the finding Cowork raised the same day. Their trace had found only four
PyInstaller binaries and an `Instructions.md` in a personal OneDrive — 27 MB Windows, 25 MB
Intel Mac, 105 MB M-series, **301 MB Linux**, the Windows and M-series builds dating from
September 2023, with no license, no version, and nothing matching in `byu-matrix-lab`. The
conclusion drawn from that, that the course would lose the tool whenever the account was
reclaimed, no longer holds.

**Inspected 2026-09-26, and it is in better shape than the binaries suggested.** One
`grader.py` of 16 KB plus a README — no build, no PyInstaller, nothing to sign. The README
documents usage: `python3 grader.py source target output`, with a fourth argument `False`
for non-UTF-8 input. Running from source **deletes the download-and-override-Gatekeeper step
entirely**, on all three platforms, which was the part that actually bit students.

**Two blockers, both Eric's, before the decks can point at it:**

1. **It is `PRIVATE`.** Eighteen students cannot clone a private repository in the lab org.
2. **It has no license** — `licenseInfo` is null, so strictly nobody has been granted the
   right to use or redistribute it. That is also what made the OneDrive arrangement fragile.
   A permissive license additionally settles whether TorchLingo could ever absorb the checks.

**Then, in order:** verify it runs on a student-shaped input (ours, small), then Cowork
updates the Lecture 4 and 5 decks to clone it rather than download binaries.

**One caveat:** last pushed **2024-09-17**, so it predates the current assignments. Worth
checking its checks still match what Lectures 4 and 5 ask for before a deck points at it.

**Dropped:** the diagnostics-rewrite contingency. It was only ever justified by the source
being gone. `torchlingo.diagnostics` already does the alignment and contamination halves, so
convergence is now a design question rather than a rescue, and should not be decided under
deadline.

### #121 A Lecture 9 subword notebook, and it is ours to write

**Cowork needs an answer by Oct 3; Lecture 9 is Mon Oct 5, Assignment 9 due Oct 12.**

The existing SentencePiece handout is OpenNMT-specific and has to be replaced.
`docs/docs/course/lecture-09-subword-tokenization.ipynb` is the natural form, and that
directory is now ours.

**The demonstration it should carry**, which is already agreed and is better than a diagram:
before subwording a student's tokens are words; after it the same 100-token cap excludes a
*different* set of sentences; they can count the difference on their own data.
`audit_bitext.py`'s cap-pricing table is exactly this measurement, so the notebook and the
audit tool tell one story.

The one-line version Cowork asked for: **their tokens stop being words, the same sentence
gets roughly 1.8x longer, and every length-based rule they have written now selects a
different set of sentences.**

**Load-bearing as of 2026-09-27, and the dates moved.** Lecture 9 is **Wed Oct 7** and A9 is
**Wed Oct 14**, both two lectures later than before the split. After tutorial 6 moved to Lecture
9 and tutorial 3 to Lecture 10, **this is the only notebook Lecture 9 will have that is about
Lecture 9's own subject.**

**The scope is no longer guesswork.** Last year's predecessor is in `notes/legacy-f2025/` with a
README. Nineteen cells, of which the reusable core is three: install SentencePiece, train it,
look at the pieces. Everything else was OpenNMT plumbing this library replaces — including a
`pip install "numpy<2.0"` carrying a comment that it works around OpenNMT being unmaintained,
which Cowork fairly called the pivot's own epitaph.

**Two things it did that the new one must not:**

- **It fit the tokenizer on a toy corpus rather than the student's training split.** A9 is a
  controlled comparison on their own data, so it must fit on **train only**. Fitting on the full
  corpus leaks test material into the vocabulary, and nothing crashes when it does.
- **It said nothing about the length cap.** Expressing the cap in tokens breaks A9's comparison,
  because changing the tokenizer changes which pairs the cap excludes. The notebook is where
  "choose the sentence set once, with the subword tokenizer, and use it for both runs" has to
  become concrete — a student will not derive it from the handout. That is #106 and #122.

**And the teaching now arrives after the constraint.** Lecture 10 is Mon Oct 12 and A9 is due Wed
Oct 14, so decoding is taught two days before the assignment it protects and some students will
already have run it. Cowork's fix is to state the rule on Lecture 9's own A9 slide and teach the
reason at Lecture 10. Nothing for this repository to do, but worth knowing why the A9 wording
carries a rule with no explanation attached.

### #122 Make Assignment 9's control hard to get wrong in code, not just in prose

**Cowork fixed the wording; they also said that if the library can make it hard to get wrong,
that is worth more. It is.**

A9 asks students to hold everything fixed except the tokenizer. If the length cap is
expressed in tokens, changing the tokenizer changes the training set, so the comparison has
two variables and the write-up credits all of it to the tokenizer.

The wording fix is "choose the sentence set once, with the subword tokenizer, and use that
same set for both runs". A student can still not do that, and nothing will tell them.

What would: a way to select a sentence set once and carry it between runs — an explicit id
list, or a split written to disk and reused, rather than a cap re-applied per run. Worth
scoping against what `NMTDataset` already offers before adding anything.

### #123 A14's two-directions case has never been run

**Due Wed Oct 28. Lowest risk by date, highest uncertainty by evidence.**

`preprocessing.multilingual` has never been exercised at "two directions, intermingled,
separate test sets per direction", which is precisely what Assignment 14 asks for. Lecture
14's handout is a Word document written for OpenNMT and needs replacing outright.

Distinct from **#99**, which is the tutorial. This is the question of whether the code path
works at all, and it should be answered before a notebook is written on top of it.

### #118 What does a paid Colab session actually provide?

**Coulson measures it, Eric's call.** The other half of the A8 memory question: the ladder in
[`reports/length-ladder.md`](reports/length-ladder.md) gives demand at each length cap, this
gives the ceiling. Neither half is useful alone.

**What to ask for**, because "how much RAM does Colab have" is not the useful question:

- Which accelerator a paid session actually assigns — T4, L4 and A100 are roughly 16, 24 and
  40 GB, and the answer changes the recommendation.
- Device memory available to the *process*, not the instance total.
- Whether it varies between students or within a session. An assignment cannot ride on a
  lucky draw.

**What to compare against:** the measured curve, where the agreed 100-token cap holds 9.60 GiB
against 35.80 uncapped. **Carry the caveat** that those are MPS unified-memory figures, so
they transfer as a scaling *shape* rather than absolute numbers — which is exactly why this
measurement is not redundant with ours.

**Done when** the accelerator and its per-process memory are known, so the handout can say
whether 9.60 GiB fits. Until then **no memory figure goes in front of students** — only the
ordering, which is solid: batch count drives epoch time, the length cap drives whether the run
fits at all.

### #114 The wheel ships no data, so tutorials 4 and 5 cannot find what they load

The installed package contains **zero** data files — verified against the real wheel. So
`data/example.tsv` and `data/pretrained/` exist in the repository and not in a pip install, and
a student who opens tutorial 4 or 5 from its Colab badge, installs with pip and runs it fails at
the load with nothing explaining why.

Tutorial 2 is unaffected: it builds its corpus inline. Its prose does say the file "ships with
the repo", true of the repo and false of the wheel, so that wants rewording either way.

Options: fetch over HTTP in the notebooks that need it, ship it as package data, or say plainly
that those tutorials need a clone. **The first keeps the Colab badge honest**, which is the point
of having one.

**Done when** a pip-installed tutorial 4 or 5 either runs or fails with a message that says what
to do. Same shape as the wheel-missing-modules defect: what the repository has is not what the
wheel carries, and nothing checks.

## Code — decoding performance

### Decision (2026-09-22): #2, #3 and #6 are descoped

**Eric's call, taking the recommendation from the competitive assessment.** Batched beam
search across sentences, KV caching, and multi-GPU training via DDP are not being built.

*Why.* Each is high-complexity, low-teaching-value, and duplicates what CTranslate2,
Marian and Joey NMT already do better than this repository ever would. The 2026-08-22
design decision below concedes the premise without drawing the conclusion: the fast path
"is necessarily harder to read than the 85-line version" and is kept separate *because*
it cannot be followed line by line. Something a student cannot read is not teaching them
anything, so the case for carrying it in a teaching library had to be made rather than
assumed — and on inspection it could not be.

The competitive assessment is what forced the question. Joey NMT covers this ground,
is actively maintained, and reaches 93.62 BLEU on its toy task in under four minutes on
CPU, so the decoder-performance work is the *least* differentiated thing we could spend
effort on. What is differentiated — diagnosis, empirical discipline, a corpus with a
documented repair history, documentation that executes — is where the time goes instead.
The full assessment, including the hands-on numbers, is in `docs/docs/related-work.md`.

*What this does not touch.* #4 (length-normalization semantics) stays open and is
unaffected: it is a correctness-and-teaching question with real evidence behind it, not a
performance one. `inference_fast.py` stays as it is — already merged, already tested by
`tests/test_decoding_equivalence.py`, and still the faster path for anyone who wants it.
Nothing is being removed; we are declining to extend it.

*If this is ever reversed*, the technical notes below are kept deliberately intact — the
`(batch x k, t)` flattening, the ragged-completion bookkeeping, and the warning that a KV
cache leaves the call count unchanged and must be read on the positions-forwarded column
instead. That last one would cost a day to rediscover.

*Also worth telling students.* "We could make this faster and chose not to, because the
readable version is the point, and here is the toolkit to use when speed actually
matters" is a better lesson than a fast path nobody reads. Candidate for
`concepts/decoding.md`.

### Design decision (2026-08-22): reference and fast implementations live side by side

The optimized decoders are **added alongside** the simple ones, not layered into them.
The existing `greedy_decode` / `beam_search_decode` stay as the readable reference a
student can follow line by line; batching and caching go in separate, clearly named
implementations.

*Why:* readability is the reason this library exists. A batched, KV-cached beam search is
necessarily harder to read than the 85-line version — index bookkeeping across
`(batch x beam)`, cache invalidation, ragged completion. Folding that into the one
implementation trades away the thing the repo is for, to buy speed that only matters at
scales students often are not working at anyway.

*What this unlocks:* #3 (KV cache) was previously marked "decide whether to do it at all,
since it may compromise readability." That constraint is gone. The fast path can be as
dense as it needs to be, because the readable path is preserved. #3 moves from
questionable to straightforwardly worth doing.

> **Superseded 2026-09-22.** This paragraph answered "may we build it?" and read the
> answer as "so we should." The descope decision above answers the question that was
> never asked: *should* we, given that it duplicates CTranslate2 and Marian and teaches
> nothing a student can read. The side-by-side design remains correct for the code that
> already exists — it is only the conclusion about #3 that is withdrawn.

*What this demands:* two implementations silently diverging is the obvious failure mode.
`tests/test_decoding_equivalence.py` already covers this — it was written as a
characterization oracle for a refactor, but the natural reading is now stronger:

> The simple implementation is the **specification**. The fast implementation must
> produce token-identical output on every fixture in that module.

Every fast variant should be run against the same fixtures as the reference, ideally
parameterized so adding an implementation automatically inherits the whole suite.

*Resolved 2026-08-26:*

**Module layout — a flat sibling module, `src/torchlingo/inference_fast.py`.**
`inference.py` keeps the reference decoders and the shared helpers (`_canonical_topk`,
`_rank_key`) and is not touched by the optimization work. Matches the repo's existing
flat-module convention (`config.py`, `training.py`, `evaluation.py`); subpackages are
reserved for places with several peers (`models/`, `preprocessing/`). Rejected: putting
both in `inference.py`, which would push it past 700 lines and defeat the split;
and an `inference/` subpackage, which makes a reader navigate a directory to find an
85-line function.

**`translate_batch` — mirrored, not switched.** The reference wrapper stays as-is;
`inference_fast.py` gets its own `translate_batch`. This keeps the dependency arrow
one-way: **fast imports from reference, never the reverse.** A selector parameter or a
fast-by-default wrapper would force `inference.py` to import `inference_fast.py`,
coupling the module a student is meant to read to the one they are not.

**Guidance — three layers, because docs alone will not catch the failure case.**
1. `docs/docs/concepts/decoding.md`: reference vs fast, carrying the measurement
   (38.7x the `decode()` calls for 5.0x the math, every call at batch size 1).
2. Bidirectional docstring cross-references between each implementation and its
   counterpart.
3. A threshold-based, once-per-process `warnings.warn` when the reference path is used
   on a large input, naming `inference_fast.translate_batch` and noting the output is identical.
   This is the layer that actually works: it fires at the moment of pain, whereas the
   student who most needs it is mid-experiment and not reading docs.

**Test structure — shared contract base class, one subclass per implementation.**
Lift the fixtures and invariants in `test_decoding_equivalence.py` into a
`DecoderContractTests` mixin with the decode callable supplied by each subclass:

```python
class DecoderContractTests:          # not a TestCase itself
    DECODE = None
    # ...every fixture and invariant...

class ReferenceBeamTests(DecoderContractTests, unittest.TestCase):
    DECODE = staticmethod(beam_search_decode)

class BatchedBeamTests(DecoderContractTests, unittest.TestCase):
    DECODE = staticmethod(inference_fast.beam_search_decode)
```

Adding an implementation is one subclass and it inherits the whole suite; failures name
the implementation, so a divergence is unambiguous. The reference is the specification
and runs on every invocation, which is what keeps it from rotting into a museum piece.

### The 38.7x is a budget split across two levers

Batching offers ~38.7x fewer model calls, but as two independent levers whose effects
multiply — worth recording, because the figure was originally quoted here as if any one
task could deliver it:

| Lever | Worth | Task |
|---|---|---|
| Batch across beams | ~`beam_size` | #1 |
| Batch across sentences | ~`num_sentences` | #2 |

`beam_size=5` x 8 sentences = 40 ~= 38.7. #1 recovers roughly `beam_size`; the rest needs
#2, which is worth more the larger the test set.

**As of 2026-09-22 that remaining ~8x is being left on the table deliberately** — #2 is
descoped. #1's share is already merged. Kept here because the split is the thing worth
knowing: the figure was once quoted as if any single task could deliver all of it.

Full explanation for students lives in `docs/docs/concepts/decoding.md` — keep it there
rather than duplicating it into code and notes.

**#4 Resolve length-normalization semantics** — in review as PR #54

`inference.py:203` applies length normalization during *pruning*, not only at final selection,
comparing normalized scores across lengths mid-search. Defensible but non-standard.

**The evidence, from #40**, measured on the tutorial 5 model across five held-out subsets:
`alpha=0.6`, the shipped default, is **indistinguishable from `alpha=0.0`** (−0.07 ± 0.03
BLEU), while `alpha=1.0` gives +0.25 ± 0.06. The bias it targets is plainly there — mean output
length falls from 12.61 tokens at greedy to 9.73 at beam 10, against references averaging 11.62.

So the question is not whether the semantics are defensible, but **why a correction that
measurably does nothing is on by default.** Two candidates the evidence cannot separate: the
default is too weak, or normalizing during pruning blunts it.

**Done when** `alpha` has been swept with normalization applied only at final selection, which
separates them — cheap now that `scripts/sweep_decoding.py` exists.

## Code — other gaps

**#7 One PyTorch deprecation warning left** — in review as PR #57
On torch 2.13.0, "Support for mismatched key_padding_mask and attn_mask is deprecated",
raised from the decode path. It will eventually break. The decode path passes a boolean
`tgt_key_padding_mask` alongside a float `tgt_mask`; making both the same dtype should
settle it.

The other warning this entry used to list, the nested-tensor prototype notice from
`nn.Transformer`, is gone. It was a side effect of disabling the encoder's nested-tensor
fast path, which had to go because the op behind it is unimplemented on Apple's MPS
backend and made every library decoder raise `NotImplementedError` on Apple Silicon.
Worth knowing for the next device-specific bug: CI runners are x86 Linux, so nothing in
the matrix can reproduce that class of failure — the lab's Macs are the only place it
shows up, which is also where the students are.

## Evaluation / tooling

**#8 Verify Eole claims hands-on before syllabus use**
Specifically: COMET/MetricX integration in the training loop, and 7B-13B finetuning on a
single 24GB GPU. Both are from Eole's README, not from running it.
- Needs a separate venv: Eole requires Python >= 3.11 and torch >= 2.10, **< 2.13**.
  This repo's `.venv` has torch 2.13.0.

**#9 Run `pre-commit install`**
`.pre-commit-config.yaml` exists in the repo but hooks are not installed in this clone.

**#91 `metric_comparison.json` records no BLEU signature**

`docs/docs/_generated/metric_comparison.json` on the PR #58 branch ends with
`"bleu_signature": "not recorded"`, because `scripts/compare_metrics.py` was written
before PR #55 gave `compute_bleu` a `.signature`. The generated page therefore publishes
scores with no record of how they were produced, which is the exact thing #70 exists to
prevent, in the one place that is *generated* and so should have been easiest to get
right.

- Once #55 and #58 are both in, teach `compare_metrics.py` to read `.signature` and
  regenerate both the JSON and the markdown.
- Related to #85: chrF and TER have no signature to record yet, so this lands properly
  only after that one.

**#90 `CLAUDE.md`'s numbering example is stale**

The "Say PR #X and Task #Y" rule says *"tasks run to #62, pull requests to #42, so every
number below 43 names one of each."* Tasks now run to #92 and PRs to #64. The rule is
right and its reasoning is intact; only the arithmetic has rotted, in the file that
teaches the convention.

- One line, in `CLAUDE.md` rather than `notes/`, so it cannot ride along on a notes-only
  PR. Fold it into the next change that touches `CLAUDE.md` for another reason.
- Consider dropping the specific numbers instead. They were never the point, and they
  will be stale again within a week.

**#85 Only BLEU carries a signature; chrF and TER do not**

Task #70's subject says "with every score", but PR #55 attaches `.signature` to `compute_bleu`
alone:

```
BLEU  nrefs:1|case:mixed|eff:no|tok:13a|smooth:exp|version:2.6.0
chrF  MISSING
TER   MISSING
```

**chrF is the case that proves this is not cosmetic.** `compute_chrf` defaults to
`word_order=2` (chrF++) while sacreBLEU's `corpus_chrf` defaults to `0`. That one undeclared
parameter caused two separate confusions here: a correct implementation judged broken by 0.45
points against the wrong baseline, and a bug note stating the right value beside a snippet
returning the wrong one. A chrF signature declares `nw:2` and prevents both — so chrF needs one
*more* than BLEU does.

**Done when** `get_signature()` is called on CHRF and TER as it is on BLEU, and
`tests/test_bleu_signature.py` covers all three. Cheap once PR #55 is in.

**#86 `evaluate_model` has no test, and it is the function callers use**

PR #58 tests the three metric wrappers well and leaves `evaluate_model` and
`save_translations` untested. But `evaluate_model` is what `examples/evaluate.py`,
`examples/train.py` and any student actually call, and it is **where the transpose bug did its
damage**: `compute_chrf_score` defaults True, so every call reported a wrong chrF.

Fixing the wrappers without testing the aggregator leaves the same hole one level up, where the
thing under test is not the thing being used.

**Done when** `evaluate_model` runs end to end on a tiny fixture and its returned bleu/chrf/ter
match the three wrappers called directly — under default flags and with
`compute_ter_score=True` — and `save_translations` is covered too.

**#82 Add an on-target language check to `torchlingo.diagnostics`**
Borrowed from mtsurvey, which reports an on-target rate using GlotLID: the share of
hypotheses actually written in the target language. It catches a failure BLEU hides
badly, because a model that copies the source scores non-zero against a related-language
reference and looks merely weak rather than broken. Fits the existing `CheckResult`
shape. The dependency is the open question: GlotLID is a FastText model download, so the
check should degrade to a clear "not installed" rather than fail.

## Possible tooling to productize

**#81 Fail the build on hand-typed generated numbers**
mtsurvey gates its write-ups on a claim checker: every quoted number must resolve to an
entry in the generating JSON, and a hand-typed constant fails CI. This repo has the
inputs for it already — `docs/docs/_generated/*.json` from `scripts/sweep_decoding.py`
and `scripts/compare_metrics.py` — but nothing enforces the link, which is how the
chrF `63.39` in #80 survived. Scope it to the generated pages first rather than all of
`docs/`.


---

## Follow-ups from the training-budget finding

**The finding itself now lives in [`reports/training-budget.md`](reports/training-budget.md).**
It is a measurement, not a task, and it is the prior for #119's learning curve: training
budget beat data by about 7x, and the data effect's confidence interval crossed zero.

What remains here are the corrections it generated.

**#60 Nobody is told when main goes red**

main broke on 2026-09-19 and stayed broken until someone happened to look.

**No PR could have caught it**, which is the point. #25 added a docstring example that cannot
pass; #32 added the `--doctest-modules` gate that runs it, having branched from a main where
that file did not exist. Both green alone; the *combination* fails. A merge-order interaction
can only surface in a post-merge run on main — so that run is load-bearing and unwatched.

Nothing notifies on a failed push-to-main run; it sits in the Actions tab, and the next person
to open a PR inherits a red main and may assume they caused it.

**Done when** a failed push-to-main run notifies someone: an `if: failure()` step posting to the
lab's Discord, or just enabling GitHub's own Actions-failure email. Neither needs new
infrastructure.

Same family as #51 and #53: a check that does not block, a check that runs a fifth of what it
claims, and a check nobody reads.

## Tests

**#15 Existing `DummyTransformer` is history- and memory-blind**
`tests/test_training_inference.py:16` computes logits from a zero tensor, so its output
depends only on decoding *position*. Verified: logits are byte-identical for different
decoder histories AND for different encoder memories. The three existing beam tests
therefore cannot detect scrambled beam state or bad memory expansion.
- Mitigated by the new `tests/test_decoding_equivalence.py`, but the old tests should
  eventually migrate to the history-sensitive fixture rather than sitting alongside it.

**#79 `test_different_optimizers_produce_different_results` is order-dependent**

**Reproduced 2026-09-26, with a mechanism.** It failed on PR #57's branch under
`python -m unittest discover tests` while `main` passed the same run, and it passes under
`pytest` on both. So the two documented ways to run the suite disagree, which is the fourth
instance this month of two things that must agree with nothing checking that they do.

**The trigger is test count, not PR #57's content.** #57 changes only the decode path — a float
causal mask becomes the boolean one training already used — and this test trains Adam against
SGD and never decodes. What #57 adds is a 150-line test file, so more RNG is consumed before
this test runs. That is exactly the failure mode predicted when this was filed: *a test that
depends on whatever seeded the RNG before it will come back.*

**Stronger than that, as of 2026-09-26: it fails on unmodified `main` with nothing added.**
Stashing a branch's changes and running `python -m unittest discover tests` on `main` alone gave
726 tests and the same single failure. So it does not need a PR to add test lines — the current
`main` already sits on the wrong side of it under `unittest discover`, while `pytest` passes 737.

That matters for two reasons. It is not a property of whichever branch happens to surface it, so
no PR should be held for it. And **CI only runs `pytest`**, so the suite CLAUDE.md documents
first is the one nobody is watching.

**The assertion is the defect.** It checks `adam_improved or sgd_improved` — "at least one
should show improvement" — on a tiny fixture over 5 epochs. With an unlucky starting state
neither improves and the test fails, while the claim it is named for, that the two optimizers
take *different trajectories*, is never checked at all.

- Seed inside the test, and assert the **difference** between the two curves rather than
  improvement in either.

**Fixed 2026-09-27, and it was three defects rather than one.** Worth recording because each
was hidden behind the previous, and only the first was the one filed:

1. **The optimizer test was unseeded**, so its models' initial weights came from whatever RNG
   state earlier tests left. Seeded identically for both optimizers, which also makes it the
   controlled comparison its name claims. Its assertion now checks that the two curves
   *differ*, which is the actual claim — and that they differ by epoch two, so an optimizer
   ignored until epoch three would still fail.
2. **A sibling had the same disease**, and fixing the first revealed it:
   `test_learning_rate_affects_convergence_speed`, also unseeded, also asserting a stochastic
   inequality.
3. **That sibling was measuring the wrong quantity, and that is why it looked flaky.** It
   compared `losses[0] - losses[1]` between two rates and called the larger drop faster. But
   `train_losses[0]` is the *average over epoch one*, so a rate that learns quickly within
   epoch one reports a lower first value and therefore a smaller later drop — the formula
   inverts the thing it is measuring. It now asserts the loss *reached*, which is what
   "converges faster" means.
   **And its rates were wrong:** at lr 0.01 the "fast" model reaches a worse final loss than
   the slow one at two of three seeds, because 0.01 overshoots on a 32-dimensional model over
   twelve identical rows. Measured, not assumed. Changed to 0.001 — the library default —
   against 1e-05, which holds at every seed with a wide margin.

**CI now runs both documented runners.** `pytest` and `python -m unittest discover tests`. It
costs a second pass over the suite, and it is the price of "the tests pass" meaning the same
thing on a laptop and in CI.

**Twelve more tests have the same shape**, found by walking the AST for tests that build a
model, train it, and assert an inequality without seeding:

```
test_checkpoint.py            test_legacy_training_state_dict_extracts_weights
test_integration_robust.py    test_full_pipeline_with_sentencepiece
                              test_training_with_validation_and_early_stopping
                              test_training_handles_gradient_explosion
                              test_training_with_amp
test_model_convergence.py     test_transformer_overfits_small_dataset
                              test_transformer_loss_decreases_monotonically_on_simple_task
                              test_lstm_overfits_small_dataset
                              test_gradient_clipping_prevents_nan_loss
                              test_early_stopping_triggers_on_no_improvement
                              test_best_checkpoint_has_lowest_val_loss
                              test_larger_model_achieves_lower_loss
```

**Deliberately not rewritten.** Most assert effects large enough to survive any starting
point — a model *will* overfit ten identical rows — and rewriting twelve working tests to
pre-empt a failure none of them is showing would be churn. The two that broke were the two
whose claims were weak or inverted, which is the pattern worth watching rather than the
absence of a seed.

**`test_larger_model_achieves_lower_loss` is the one to look at first if another goes.** It is
the same shape as the learning-rate test: a comparative claim between two configurations on a
toy corpus, where the effect may be smaller than the noise. The both-runner gate will surface
it rather than leaving it to a laptop.
- **This now blocks clean verification of every PR that adds tests**, since adding tests is what
  perturbs the ordering. That makes it worth fixing before #54, #59, #103 and #105 land rather
  than after.
- Worth sweeping for others while in there: run the suite under both runners and diff. CI uses
  `pytest`; `CLAUDE.md` documents `unittest discover`. Nothing checks they agree.

**#36 CI actions are pinned to a deprecated Node runtime**
Every run now warns: `actions/checkout@v4`, `actions/setup-python@v5` and
`actions/download-artifact@v4` target Node 20, which GitHub deprecated, and are being
forced onto Node 24. It is a warning today and a hard failure whenever GitHub drops the
shim. Bump the action versions. Unrelated to anything in flight, and cheap.

**#44 Gate the sdist on "no Git LFS pointer shipped"**

CI checks out without LFS to keep runs light. A build that packages an LFS-tracked file in a
no-LFS checkout ships **130 bytes of pointer text under the name of a 17 MB corpus**, with
nothing complaining, and it would reach PyPI looking fine and open as garbage.

`MANIFEST.in` now excludes both files explicitly, so this is closed *by construction* rather
than *by check* — a future `recursive-include` reopens it silently. Two things that must agree
with nothing checking they do.

**Done when** a release-job step scans the built sdist and wheel for any member beginning
`version https://git-lfs` and fails on a hit. Cheap, no LFS dependency, and it catches the class
rather than today's two files.

## Inference gaps

## Lint and tooling gaps

**Standing gotcha from #45, which is now fixed in #21**

A PR opened *before* #21 merged still shows no checks, because for `pull_request` events
GitHub reads the workflow from the PR branch rather than from main. Rebase such a PR onto
current main, or dispatch a run with
`gh workflow run tests_and_build.yml --ref <branch>`. PRs opened since #21 get checks
automatically, whatever branch they target.


**#22 `examples/` and `scripts/` are outside the lint gate**

CLAUDE.md and CI lint `src` and `tests` only. `ruff check examples` turns up 32 pre-existing
errors across five files — unsorted and unused imports, `f`-strings without placeholders,
deprecated `typing.List`, a blind `except Exception`. `scripts/` has the same gap.

**These are the code students are most likely to copy**, which makes them arguably the worst
place in the repo to let style rot.

The gap keeps widening: `scripts/` has since gained `bench_decode.py`,
`execute_notebooks.py`, `train_example_model.py`, `sweep_decoding.py`, `diagnose_corpus.py` and
`render_report.py`, all linted by hand on the way in and none of them gated. Hand-linting is
exactly what stops happening once whoever does it moves on.

**Done when** the scope is `src tests examples scripts`. Fix under its own PR first, since a
five-file mechanical diff would bury any review it rode along with.

## Visualization

All three raised by Coulson on Discord, 2026-09-14, after reviewing the open PRs:

> "if we can add visualization to any of the options that we present it could be useful
> for the students. I saw that it was added for attention, but did we add it for beam
> search as well? ... students should have learned about this in 312 ... but I think a
> reminder in this tool may be useful."

**The answer to his direct question was no.** That is now fixed: `visualization.py` gained
`format_beam_search` and `plot_beam_search`, and `beam_search_decode` takes an optional
trace recording candidates *before* pruning. The two below are what remains.

## Docs and tutorials

**#78 Tutorial 5's committed outputs predate the retrained checkpoint** — in review as PR #57
The notebook shipped translations produced by the old checkpoint, so a student reading
the page and a student running the cell saw different results. Re-executed against the
current checkpoint in PR #57. Re-executing is what surfaced #7: the fresh run baked three
PyTorch warnings and a `/Users/ringger/...` path into the committed outputs, which main
did not have, so both are fixed in the same PR.

**#83 Show attention on the Transformer, not only the LSTM** — in review as PR #59
Tutorial 4 taught alignment on the LSTM's additive attention, which is the architecture
students do *not* use for the rest of the course. PR #59 adds Part 8: load the pretrained
Transformer, decode one held-out sentence, and read its cross-attention through
`attention_for_sequence`. Two things that made this non-obvious and are worth keeping in
the prose: PyTorch hardcodes `need_weights=False` inside
`TransformerDecoderLayer._mha_block`, so a plain forward hook returns `None` and you need
`capture_cross_attention`; and the teacher-forced second pass is *exact* rather than an
approximation, because the decoder is causally masked.
- Costs CI nothing, but it does make tutorial 4 depend on `data/pretrained/model.pt`, so
  tutorial 4 joins the LFS skip list. See #53.

**#88 Open the tutorial 7 PR** — unblocked 2026-09-26, PR #58 merged

Written, executed and verified on the **local** branch `docs/evaluation-tutorial` (commit
`1516aa0`), held locally rather than stacked. It needs PR #55 for the `.signature` it prints.

At PR time: rebase onto a `main` carrying #58 and #55; expect a small nav conflict in
`docs/mkdocs.yml`, since PR #51 also edits the tutorials block — **tutorial 7 goes after
tutorial 6**; re-execute and re-run every check; say in the body that it closes Task #84.

Already verified on the branch: 7/7 notebooks execute cleanly, no local paths in outputs, every
number quoted in prose appears in an output, `mkdocs --strict` exits 0, ruff clean, tests pass.
It needs no LFS artifact, so it takes the CI gate to 3 of 7.

**Done when** the PR is open and green.

**#89 Fail the docs build when a page is off-nav**

`mkdocs` reports off-nav pages at **INFO**, so `--strict` exits 0 while they sit unreachable.

**Measured 2026-09-26, after #84 was fixed: 53 pages, 8 off-nav**, and the split is the whole
design of the check — 5 are `_generated/*.md` snippets pulled in with `--8<--` includes and are
*supposed* to be off-nav, while 3 are genuine orphans: `MULTILINGUAL_ANALYSIS.md`,
`MULTILINGUAL_QUICKSTART.md`, `TESTING_GUIDE.md`.

The trap has caught five pages across two PRs, and the second was noticed only because someone
was auditing the first. That does not catch the third.

**The exception list is the whole design problem.** `_generated/*.md` are *supposed* to be
off-nav — they are snippet files pulled in with `--8<--` includes, not standalone pages — so a
naive check cries wolf on five files immediately. They must be declared deliberately.

`MULTILINGUAL_ANALYSIS.md`, `MULTILINGUAL_QUICKSTART.md` and `TESTING_GUIDE.md` are genuinely
orphaned and predate all of this; decide whether they are nav pages or should leave the docs
tree.

**Done when** every page under `docs/docs/` is either in the nav or on a declared exception
list, and the build fails when one is neither.

**#92 Tutorials 3 and 5 bypass the library's own evaluation API**

Both call `sacrebleu` directly rather than `torchlingo.evaluation`. Tutorial 6 and the
new tutorial 7 use the library. So a student meets two different ways to score, and the
library's own evaluation API is the one the earlier tutorials never touch.

This is also part of why the chrF/TER transpose bug survived: nothing in `docs/` or
`tests/` exercised `compute_chrf` or `compute_ter`, so there was no path along which the
wrong number could be noticed.

- Route tutorials 3 and 5 through `compute_bleu`, which is the function they are already
  imitating.
- Worth doing after #58 lands, so the tutorials pick up the signature and the fixed
  reshaping at the same time.

**#53 The notebook gate is weaker than its green check implies**

CI checks out without Git LFS on purpose, so `data/example.tsv` is a pointer and
`execute_notebooks.py` skips tutorials 2 through 5. **The job passes having run two notebooks
of six**, and the check mark looks identical either way.

Found concretely in #50, which adds an assertion inside tutorial 3 whose whole purpose is to
fire when the model stops being decisive. It cannot fire in CI, because tutorial 3 does not run
there.

The skip list only grows as tutorials touch real data. It can go the other way: tutorial 7 uses
fixed strings, needs no artifact, and runs — a design lever worth knowing, since a tutorial
whose subject needs no trained model should not acquire one.

Options, cheapest first: **say it in the check** (the job already knows what it skipped, so put
that in the job summary); **fetch LFS for the notebook job only**, ~28 MB per run, which the
keep-CI-light decision ruled out on four-version-matrix grounds that do not apply to one job;
or a **scheduled full run** with LFS.

**Done when** the green check states what it did not run. Same family as #51 and #60.

**#63 Three pages have no mkdocs nav entry** — in review as PR #51

`docs/mkdocs.yml` belonged to a PR we were not stacking on, so tutorial 6,
`reference/diagnostics.md` and `related-work.md` shipped without nav entries. Off-nav is INFO
rather than a warning, so `--strict` stays clean and nothing blocks — but each page is reachable
only by direct link until PR #51 lands.

| Page | Place it |
|---|---|
| `tutorials/06-diagnosing-failures.ipynb` | Tutorials, after `05-real-translations.ipynb` |
| `reference/diagnostics.md` | API Reference, after `config.md` |
| `related-work.md` | Top level, near Home |

Also undo at the same time: `related-work.md` names `torchlingo.diagnostics` as plain code
rather than linking to `reference/diagnostics.md`, because linking a page absent from `main`
fails `--strict`. Make it a link once that page is there.

The same trap has already caught the next pair of pages — see Task #84 — which is what #89 is
for.

**#65 Tutorial 6 and `torchlingo.diagnostics` are two copies of the same checks** — in review as PR #52

PR #44 defines the five checks inline in the notebook; PR #45 ships them as a module.
Until one sources from the other they can drift, and the notebook is the copy a student
reads.

- Only after **both** have merged — doing it in either PR would stack it on the other.
- Keep the student seeing the logic; the pedagogy depends on it. Import the functions and
  show the source (`inspect.getsource`), or keep a short annotated call, rather than
  silently calling a black box.

**#66 Adopt `nltk.translate.gale_church`; split #29 into two different jobs**

#29 conflated two goals that want different tools, and proposed hand-writing an algorithm that
is a dependency away: `nltk.translate.gale_church.align_blocks()` ships with exactly the priors
#29 specifies, including `VARIANCE_CHARACTERS=6.8`.

- **To recover the 98 talks (~13k pairs):** use Vecalign or Bertalign. Embedding-based aligners
  measurably beat length-based ones — an English–Slovak evaluation (*Scientific Reports*, 2023)
  puts both significantly ahead. Gale-Church is the wrong tool for the production job.
- **To teach alignment:** implement it, because the implementation *is* the lesson, but pin the
  output against NLTK's as a test oracle rather than shipping ours as the only word on it.

**Done when** those two are separate tasks with the right tool on each. Same split applies to
#52.

**#68 Cite `torcheck` as prior art in the diagnostics docs**

`pengyan510/torcheck` already does PyTorch sanity checking, including frozen-parameter
verification. PR #45 does not mention it, which implies more novelty than is warranted.

The framings genuinely differ and both are defensible: torcheck registers with the
optimizer and asserts *during* training that parameters do or do not change;
`torchlingo.diagnostics` inspects *after the fact* and sorts parameters into
frozen/dead/live, naming the culprit.

- Add a short prior-art note to `docs/docs/reference/diagnostics.md`: what it does, how it
  differs, when to reach for it instead.
- Check its current maintenance status first. It was found, not evaluated.

**#71 Decide whether to report the Joey NMT breakage upstream**

Their shipped quickstart does not run on current PyTorch: `joeynmt/builders.py` passes
`verbose=False` to `torch.optim.lr_scheduler.ReduceLROnPlateau`, which PyTorch removed,
so `scheduling: "plateau"` raises `TypeError` before the first step. The toy config also
sets `use_cuda: True` and `fp16: True`, which fail on any CPU-only machine.

Found while running their tutorial as a baseline (2026-09-20). Not reported — filing an
upstream issue is outward-facing and is Eric's call.

- One-line fix upstream; a courteous thing to send given we cite them favourably.
- If yes: report from a clean clone, not the patched scratch copy.

**#74 A broken anchor and 93 unexplained warnings in the docs build** — anchor in PR #51; the 93 warnings still open

Two pre-existing docs-hygiene items, both visible in every `mkdocs build --strict` run
and neither currently failing it:

- `reference/visualization.md` links to
  `#torchlingo.models.transformer_simple.capture_cross_attention`, and no such anchor
  exists on that page. Reported as INFO, so `--strict` does not catch it — the same class
  of rot as #26, which *was* caught only because those links were WARNINGs.
- 93 `Div ... unclosed ... closing implicitly` warnings, from the Material card grids.
  Believed benign; never actually diagnosed. Confirm, then either fix the markup or
  record why it is acceptable so the next person does not re-investigate.
  **New detail, 2026-09-23:** these print as `[WARNING]` yet `--strict` still exits 0,
  so they are not reaching the logger `--strict` gates on; they come from a markdown
  extension writing to its own. Worth establishing which, because it means `--strict`
  is a weaker gate than its name implies and other extension warnings are escaping it
  too. Same shape as #89, one layer down.

**#51 The docs gate reports but does not block**

#26 added a `docs` job running `mkdocs build --strict`. It runs on every PR and takes 30
seconds, but it only *reports*: the repo ruleset decides what blocks a merge, and the job
is not in it.

A check nobody is required to pass is a check that gets ignored the first time it is
inconvenient. Add "Build docs strictly" to the required status checks. Same for the
notebook gate if it is not already there. Repository settings, not a code change.

**#52 Try Moore (2002) if more of the corpus is wanted**

Raised by Eric. [Moore (2002)](https://aclanthology.org/2002.amta-papers.14/) aligns in two
passes: a length-based pass like the one #29 ships, whose confident pairs train IBM Model 1,
then a second pass scoring length *and* word correspondence.

**Honestly, it would buy little here**, which is why #29 shipped length alone: Gale-Church
recovered 13,152 of a possible 13,305 pairs at a quality indistinguishable from the talks that
never needed repair. The 153 it gave up are the ceiling, and the 89 single-stream talks are
unreachable by either method.

Where it *would* matter: the case pinned by `test_a_long_dropped_sentence_is_handled_worse`.
Length treats a long deletion as so improbable that a poor one-to-one scores better, whereas a
dropped sentence shares no *words* with anything — exactly what lexical evidence sees. The
method is also the transferable part for any noisier corpus.

**Done when** either the code lands or this is closed as not worth it. Also a good teaching
progression for #48: length alone, why it fails, then lexical evidence.

**#48 Audit what we have built for pedagogical value, and write down the sequencing**

Enough accumulated that nobody could say what a student is meant to learn, in what order, or
where the gaps are. It was built task by task, each justified alone, never against a
curriculum.

**First pass is written**, and now lives in `CS479_COURSE_ROADMAP.md` under "Repository
side": the sequencing, the outcomes, the gaps and the redundancy findings. The largest gap it found: everything teaches the
machinery working, nothing teaches **why a model fails**, which is what a student actually
hits.

**What remains is instructor-owned and cannot be done here.** Those outcomes are
reverse-engineered from the material, so they describe what exists rather than what the course
needs. Four questions for Eric are listed at the bottom of that file.

One more, from `docs/docs/related-work.md`: **does Joey NMT belong in the syllabus** as a
comparison point — the same system as a configured toolkit rather than a library you call —
rather than only in related work? Its toy config trains in 3m52s on CPU to 93.62 BLEU, cheap
enough to run beside ours.

**Done when** those questions are answered and the roadmap states outcomes the course wants
rather than outcomes the code implies.

**#28 Attention parameters skip `_init_weights`**
`SimpleSeq2SeqLSTM._init_weights` matches on `weight_ih` / `weight_hh` / `bias`, so
`AdditiveAttention`'s `W_dec`/`W_enc`/`v` and `attn_combine` keep PyTorch's default Linear
init. Defensible — they train well, additive reaches 93.6% alignment accuracy — but it is
currently implicit rather than chosen. Either extend `_init_weights` deliberately or leave
a comment saying the default is intended. Small, and worth settling while it is fresh.

