# TorchLingo — Session Task List

Opened 2026-08-22, last updated 2026-10-17. Numbered for reference in conversation.
Completed work is removed rather than marked done — git history is the record.

**Numbers here are task numbers, and they collide with pull request numbers.**
The two ranges have grown up together, so almost every number names one of each. Say
"Task #37" or "PR #37" in conversation and in GitHub comments; a bare `#37` is
ambiguous, and on GitHub it auto-links to the pull request whether or not that
was meant.

## This file is the whole list; the session mirror is only what is urgent

**Convention, Eric's, 2026-09-25.** A task stays on the in-session task list only if it
contributes to one of the next two lectures. Everything else lives here and here alone.

That keeps the working list short enough to be read, without losing anything: this file
is authoritative and always has every task. The mirror is a filter over it, not a second
copy of it.

**As of 2026-10-14, at the baton pass to Cowork:** everything open is here. The cell-by-cell
audit of every course notebook is applied, and every assignment notebook has the student write
the method, with checks, a prediction that stops the run, and write-up questions for its
lecture's conceptual outcome. What is left on the course side waits on Cowork's answers, each
row saying so: Tasks #176, #229, #231, #233, #235 and #242, and part of Task #169. On this side,
Tasks #99 and #123 (both needed by Wed Oct 28, when A14 is set) and Task #243, the architectural
review, are the open work.

**Since then, 2026-10-15 to 17, on this side:** the COMET install in A10, A11 and A13 was fixed
twice (PRs #295 and #297); every install cell now runs in a copy of Colab's runtime on each push to
`course` (Task #244, PR #302); a red run on `main` or `course` opens a `ci-red` issue (Task #60,
PR #301); and `scripts/land_pr.sh` lands a PR (Task #245, PR #304).

**2026-10-19: the baton is back with this side.** Cowork's hand-off answered Tasks #176, #231,
#233, #235 and #242, so all but #235 are open work here; #235 needed nothing, since the roadmap
already matched the notebook. Task #229 now waits on one student's MTEval results file, and
Task #169's LLM helper on Eric. Task #252 closed with the roadmap edits.
Tasks #246 to #253 were filed on 2026-10-17 from a pass over the roadmap for what it still needs
from this side, and Tasks #254 to #256 from the session's repeated manual steps.

**Read the "Critical for" column rather than the dates.** The test to apply is *"must this be
done for that lecture to happen correctly?"* — not "is this related to it". A task can sound
lecture-critical and fail that test on inspection.

Everything here is work that can be finished and then deleted. Standing rules live in
`CLAUDE.md`; decisions and findings live where they apply. `notes/README.md` maps the
rest of this directory, including the `handoff/` protocol used to exchange messages
with the Cowork session rebuilding the CS 479 decks.

## Status

"In review" means the work is written and sitting in an open pull request; the task
is not finished until that PR merges, and only then does the row disappear.

**Task #99 has an external deadline.** It comes from
`notes/CS479_COURSE_ROADMAP.md`, is sequenced against a calendar nobody here controls, and
outranks everything else in this file until A14 is set at Lecture 14, Wed Oct 28. See "The CS 479 pivot" below for the schedule and the reasoning.

### Reading the "Critical for" column

Eric, 2026-09-27: annotate every task with the lecture it is critical for. Assigned by what
each task actually touches, **not** by how near its date is — several tasks with imminent
dates are not lecture-critical, and several with no date are.

| label | means | date |
|---|---|---|
| **L1** | Course Overview and History of MT — **has already run** | Wed Sep 2 · A1 due |
| **L2** | Translation Challenges for MT — **has already run** | Wed Sep 9 · A2 due |
| **L3** | Introduction to Word Embeddings — **has already run** | Mon Sep 14 · A3 due |
| **L4** | Data Preparation for MT Training — **has already run** | Wed Sep 16 |
| **L5** | Data Preparation for MT Training, Part 2 — **has already run** | Mon Sep 21 · A4 due |
| **L4/L5** | both data-preparation lectures | |
| **L6** | Human and Automatic MT Evaluation — **has already run** | Wed Sep 23 · A5 due; A6 due Mon Sep 28 |
| **L7** | Research Paper Reviews; Intro to Neural Networks — **has already run** | Mon Sep 28 · A6 due, **A7 set** |
| **L8a** | Neural MT: Encoder-Decoder, and Why Attention Was Invented — **has already run** | Wed Sep 30 · A7 due, **A8 introduced** |
| **L8b** | Neural MT: The Transformer — **has already run** | Mon Oct 5 |
| **L8c** | Finishing the Transformer, the A8 Debrief, and the Subword Activity — **has already run** | Wed Oct 7 · **A8 due, A9 set** |
| **L8a/8b** | the halves of the split | |
| **L9** | Morphology and Terminology in NMT — **has already run** | Mon Oct 12 |
| **L10** | Overview of MT Quality Estimation — **has already run** | Wed Oct 14 · A9 due · A10 set (install COMET) |
| **L11** | Neural Quality Estimation and Evaluation | Mon Oct 19 · A10 due · A11 set |
| **L12** | Using LLMs for MT; Expanding Context Awareness | Wed Oct 21 · A12 set |
| **L13** | Strategies for NMT of Low-Resource Languages | Mon Oct 26 · A11 due · A13 set (back-translation) |
| **14+** | Lecture 14 onward: multilingual and zero-shot, speech, and later | from Wed Oct 28 · A14 set |
| **lib** | library, tooling or docs work **no lecture depends on** | |
| **hyg** | roadmap and process integrity — **critical, but not to one lecture** | |

Since roadmap v16 (Oct 7) everything from Lecture 9 on sits one session later than it did, and
the dates in the task rows below were shifted with it.

**A lecture that has already run still owns its work.** `L6` means "this belongs to Lecture
6", not "this is overdue" — the deadline pressure is gone, the subject matter is not, and a
student revisiting the lecture or an instructor preparing Fall 2027 meets it there. Filing
those as `10+` lost the one piece of information the label exists to carry. L1 to L5 were added
on 2026-09-28 for the same reason: the vocabulary started at L6, so work belonging to the first
five lectures could only be filed as `10+`, and Task #120's grader, which is Lectures 4 and 5's
tool, was. The same day `10+` itself was retired: it meant both "Lecture 10 onward" and "no
lecture at all", and those are opposite answers to the question the column asks. L10 to L13 and
`14+` now carry the first meaning, and `lib` the second.

`hyg` is not a synonym for "later". It is the category that keeps the other labels true: a
generated map that has drifted, a gate that reports without blocking, or a task list that
describes last week will mislead every lecture at once rather than one of them.

**P1 to P3 rank the work; the table is sorted by it** (Eric, 2026-10-05). **P1**: students are
blocked now, or a lecture or due date within about three weeks needs it. **P2**: matters this
term, or guards `main` and the students' path, but nothing near is waiting on it. **P3**:
library-only, a lecture that has already run, or after the term. Re-rank when the calendar
moves; "Critical for" says *which* lecture, P says *how soon*.

| | P | Task | Critical for | State |
|---|---|---|---|---|
| #176 | **P1** | `lecture-12-llm-context`'s loader matches the real A12 files, and Step 1 says where the A12 shared folder is | **L12** | **Open, before Wed Oct 21** (A12 is set at Lecture 12) — Cowork answered on Oct 19: one folder per language under `LLM_Assignment_Data/`, `src.txt` English and `tgt.txt` the language, line-aligned, no header or split; Step 1 names "the A12 page's data link" and copies both into `MyDrive/CS479/a12-data`; the language list becomes Efik, Kosraean, Palauan, Pohnpeian, Telugu, Yapese |
| #123 | **P1** | A14's two-directions case has never been run | **14+** | Open — highest uncertainty; needed by Wed Oct 28, when A14 is set; comes before #99 |
| #99 | **P1** | The A14 notebook: a bidirectional system from A9, tagged and intermingled, scored both ways | **14+** | **Needed by Wed Oct 28**, when A14 is set; after #123. Rescoped Oct 17 from a tutorial to the course notebook (see below) |
| #229 | **P1** | A11: fit Step 1 to the A6 files as students actually have them, and to MTEval's definition of average rank | **L11** | **Waits on one student's MTEval results file** (asked of Eric, Oct 19) — Cowork confirmed the average-rank definition and that A6 prescribed no layout, so one `.txt` per column stays. Every student has the MTEval results `.tsv` from A6; its columns are unknown, and reading it directly would give a sentence-level human comparison on the ranked 50 |
| #231 | **P1** | A13: point Step 1 at a course-recommended source of monolingual text, if the course has one | **L13** | **Open** — Cowork answered on Oct 19 (Eric agreed Oct 15): first the target side of the student's Church TM segments unused in A8, then public text (Wikipedia, MADLAD-400 or CC-100, OPUS) if short; both deduplicated against the A8/A9 test set and training targets, with a check like A8's; no minimum; the student reports how many came from where. Never carve it from the A8 training targets. The write-up questions, the resume-safe write-up, `EPOCHS` from A9's checkpoint and the prediction stop are done |
| #169 | **P2** | Wrap the plumbing, keep the lesson inline, in every notebook | **hyg** | **Open** — 0.2.8's `course_dir`, `load_model` and the `translate_to_file` decodes moved A9, A11, A12, A13, 8a, 8b, 8c, Lecture 7 and tutorial 5; left: A12's loader (the layout is known, with #176) and an LLM helper (waits on Eric: Cowork could not say whether any lecture after 12 prompts an LLM) |
| #246 | **P2** | Ask Cowork what Lectures 15 to 23 and the final project need from this side | **14+** | **For the next hand-off** — the roadmap has no notebook for them, by design, but Lecture 18's edit-distance activity, Lecture 19's IBM Model 1 and the projects may want code |
| #180 | **P2** | Nothing catches a notebook whose Colab badge cannot install TorchLingo | **hyg** | Open — a per-PR check in `notebook_meta.py` (Eric, Sep 29: keep it) |
| #160 | **P2** | An NLLB spotlight lecture — data, architecture **and** curriculum | **L13** | **Cowork designs it** (Eric, Sep 27), home at Lecture 13 or 14; our part is verifying the paper's numbers |
| #171 | **P2** | Document GPU nondeterminism next to the seeding conventions | **L9** | Open — draft wording below; two identical Apple-GPU runs scored BLEU 8.36 and 8.53 |
| #53 | **P2** | Notebook gate runs 2 of 6 tutorials in CI, and looks green | **hyg** | Open |
| #82 | **P2** | Add an on-target language check to `torchlingo.diagnostics` | **14+** | Open |
| #243 | **P2** | An architectural review of TorchLingo, pedagogy first | **lib** | Open — a review, reported before any change (Eric, Oct 14) |
| #167 | **P3** | Prune merged branches and stale worktrees | **hyg** | **Left:** `myles_testing` (ask Myles), `coulson`, and four local branches whose commits are not in `main` |
| #132 | **P3** | Quick Start has no notebook, and its badge opens a different one | **lib** | Open — **was only ever in the session mirror** |
| #149 | **P3** | `collect_benchmark.py` silently drops a run file it cannot find | **lib** | Open — it wrote a 2-run report over a 21-run source |
| #49 | **P3** | The shipped checkpoint predates the enlarged corpus | **L8b** | Open — `train_pairs` 64,311 against a corpus of 86,430 |
| #120 | **P3** | The grader now has a source repository | **L4/L5** | **Blocked on Eric** — the repo is private and unlicensed; then point the Lecture 4 and 5 decks at it |
| #129 | **P3** | Extract a shared `~/Projects/hpc` | **lib** | Open — after #128 gives a second implementation to diff |
| #101 | **P3** | Give the tutorials stable unique names | **hyg** | Open — **a semester boundary**, not mid-course |
| #108 | **P3** | Nothing releases the device allocator's cache | **lib** | Open — **demoted**: length, not cache, is the driver |
| #170 | **P3** | Mixed precision for A9's runs, if a Colab GPU shows it pays | **L9** | Open — **measure first**; not a release blocker after all (see below) |
| #8 | **P3** | Verify Eole claims before syllabus use | **lib** | Open |
| #9 | **P3** | `pre-commit install` (still not installed) | **hyg** | Open |
| #15 | **P3** | Migrate history-blind `DummyTransformer` tests | **lib** | Open |
| #22 | **P3** | `examples/` and `scripts/` are outside the lint gate | **hyg** | Open — **`scripts/notebook_meta.py` is now library code living there** |
| #36 | **P3** | CI actions pinned to a deprecated Node runtime | **hyg** | Open |
| #44 | **P3** | Gate the sdist on "no Git LFS pointer shipped" | **hyg** | Open |
| #48 | **P3** | Audit pedagogical value; write down sequencing and outcomes | **hyg** | In progress — in the roadmap |
| #51 | **P3** | The docs gate reports but does not block | **hyg** | Open — repo settings |
| #52 | **P3** | Try Moore (2002) if more of the corpus is wanted | **lib** | Open |
| #66 | **P3** | Adopt `nltk.translate.gale_church`; split #29 into two jobs | **L5** | Open |
| #68 | **P3** | Cite `torcheck` as prior art in the diagnostics docs | **lib** | Open |
| #71 | **P3** | Decide whether to report the Joey NMT breakage upstream | **lib** | Open — Eric's call |
| #74 | **P3** | Diagnose the 93 docs warnings | **hyg** | Open — **the anchor half shipped** in PR #51 |
| #81 | **P3** | Fail the build on hand-typed generated numbers | **hyg** | Open |
| #254 | **P3** | One way to wait for a GitHub run, shared by the scripts | **hyg** | Open — `promote_course.sh` and `land_pr.sh` each carry a loop, and four more were typed on Oct 16–17 |
| #255 | **P3** | `student_path.sh --at COMMIT`: run a notebook as it was | **hyg** | Open — showing a fix fails before and passes after took three hand extractions on Oct 16 |
| #256 | **P3** | Check `TASKS.md` itself: duplicate entries, rows without bodies, empty sections | **hyg** | Open — the Oct 17 clean-up was mostly mechanical |
| #89 | **P3** | Fail the docs build when a page is off-nav | **hyg** | Open |
| #91 | **P3** | `metric_comparison.json` records no BLEU signature | **lib** | Open — nothing blocks it |
| #247 | **P3** | Next offering: the A8 kickoff's cap and dedupe become the student's | **L8a** | Open — the roadmap's one notebook rated **high**; after the term |
| #248 | **P3** | Does "about 35 epochs" hold for a student with 30K pairs? | **L8a** | Open — a measurement, for the next offering's A8 handout |
| #249 | **P3** | Roadmap gap: what to do when training goes wrong | **lib** | Open — `concepts/training.md` has the loop, not batch size against learning rate or overfitting on a small corpus |
| #250 | **P3** | Roadmap gap: model size against latency | **lib** | Open — what a student meets when a Colab session is slow |
| #251 | **P3** | Roadmap gap: teach the training-budget episode | **lib** | Open — "which lever are you pulling?", from `reports/training-budget.md` |
| #253 | **P3** | Lecture 19: IBM Model 1 by hand, set against 8a's attention alignment | **L19** | **After Cowork rescopes Lecture 19** (Mon Nov 30); asked in #246 |

## The CS 479 pivot

From `notes/CS479_COURSE_ROADMAP.md`, handed over 2026-09-24. OpenNMT-py is in maintenance
mode and CS 479 is moving to TorchLingo **this** semester. Five assignments train a model
(Lectures 7, 8, 9, 13, 14), all five are written against OpenNMT today, and they are
consecutive and cumulative, so it is one crossing rather than five.

This converts the repository from a teaching library with a hypothetical audience into the
thing twenty-four students have to get working, on their own data, on a deadline.

### Context lives in the roadmap, not here

The calendar, the assignment sequence, the ordering argument and the paid-Colab
requirement are all in **`notes/CS479_COURSE_ROADMAP.md`** (v3, maintained by the Cowork
session) and in **`notes/handoff/briefing.md`**. Both are kept current; restating them here
produced two copies that disagreed within days.

What stays below are the dated tasks themselves.

### #99 The A14 notebook: a bidirectional system from A9

Bidirectional English and X from the A9 system, directions intermingled, separate test sets per
direction, target-language tagging, BLEU and COMET both ways. Replaces the
"MNMT Guide Using OpenNMT.docx" handout. **Needed by Wed Oct 28**, when Lecture 14 sets A14
(due Wed Nov 4); it was filed as due Nov 2, A14's old place in the calendar.

**Rescoped 2026-10-17 from a tutorial to the course notebook.** It was filed as a tutorial, but
A14 is where a student builds this system, and `CLAUDE.md` puts a lesson an assignment needs in
the course notebook, which never assumes a tutorial was read. A separate tutorial on the same
system would read as a repeat. If a library-documentation page on `preprocessing.multilingual`
is still wanted afterwards, it links to this notebook rather than retraining the same thing.

**Built to the rule in the roadmap's "Where the line is crossed"**, as A12 and A13 are: the
student writes the tagging and the intermingling as short functions that raise
`NotImplementedError`, each checked on a known case and then by `check_contamination` on both
directions' held-out sets; a prediction stops the run before training (does the reverse
direction score higher or lower than A9's?); the notebook supplies the training, decoding and
scoring. It starts from what Lecture 9's Part B leaves on Drive, as `lecture-13-back-translation`
does, and the write-up asks why tagging works at all, Lecture 14's conceptual outcome.

**Done when** `lecture-14-...` is in `docs/docs/course/` with its metadata, roadmap row, a
`tests/notebook_cells.py` test at toy size, and `student_path.sh --colab --install-only` passing,
and is promoted to `course` by Oct 28. The A14 Learning Suite text is Cowork's, rewritten against
the notebook; the hand-off that announces the notebook asks for it.


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

### #167 Prune merged branches and stale worktrees

Merged PRs keep their branches, deliberately: `CLAUDE.md` prefers merging without
`--delete-branch` while the PR self-closing mechanism is unexplained. So they accumulate — 2026-09-28
alone left fifteen on `origin` — along with scratch `git worktree`s pointing at merged branches
(`git worktree list` shows them; three were made that day for parallel PRs).

Both are deletions, so **ask Eric first**. Worktrees: `git worktree remove`. Branches: only those
whose PR is MERGED, checked per branch, never by pattern. **Done when** `git worktree list` shows
only the main checkout and no merged PR's branch remains on `origin`.

**The check, per branch:** PR MERGED *and* branch tip equal to the commit that PR merged, so
nothing pushed after a merge is lost. Record each deleted name and commit first
(`.git/deleted-branches-*.txt`), so any can be restored with `git branch <name> <commit>`.

**Left, deliberately:**

- `myles_testing` passes both checks but is Myles's, so ask him.
- `coulson` (on `origin`) has no PR.
- Four local branches have commits whose content is not in `main` and no PR:
  `docs/evaluation-tutorial`, `feat/signatures-for-chrf-and-ter`, `feat/split-lectures-8a-8b`,
  `notes/crash-audit-tasks`. Each was probably superseded, but that has to be read, not assumed.
- The `backup/...` branch is a backup; keep it.

The `pr/*` refs in `git branch -r` are GitHub's read-only pull-request refs fetched by this
clone's config, not branches; leave them.

### #170 Mixed precision for A9's runs, if a Colab GPU shows it pays

`train_model(..., use_amp=True)` runs most of the arithmetic in 16-bit, which on an A100 or L4
usually trains faster and in less memory. The 8a kickoff trains its 56.4M model in 32-bit, and
**the speed-up has never been measured here** for that model.

**Corrected 2026-09-29: it does not need 0.2.4.** It was parked on the next release because
0.2.3 does not save the mixed-precision loss scale in a checkpoint, so a resumed run would lose
it. But `use_amp` chooses **bfloat16** on any GPU that supports it and then uses no loss scaler
at all (checked in the released 0.2.3, not only on `main`). A100, L4 and G4 all support
bfloat16, so on every GPU the kickoff allows there is no scale to lose. Only float16, on a GPU
without bfloat16 such as the T4 the kickoff already rules out, needs the fix.

**Relabelled from L8a to L9.** Changing the kickoff the night before Lecture 8a would reach every
student through the badge untested on a GPU. And bfloat16 shifts the numbers slightly, so A8 and
A9 should use the same setting, and A9 is where students retrain anyway.

**Done when** someone has timed `use_amp=True` against 32-bit for the 56.4M model on a Colab A100
(a few batches at worst-case length, as `scripts/colab_memory_probe.py` does for memory), and,
if the gain is material, A9's settings say to use it for both of A9's runs, with the time estimates
updated to match.

### #171 Document GPU nondeterminism next to the seeding conventions

On 2026-09-28 two runs with the same seed, code and data on an Apple GPU scored BLEU 8.36 and
8.53. Float addition is not associative and GPU kernels may sum in a different order each run, so
a seed buys reproducibility on a CPU only. Nothing in `CLAUDE.md` says so, and a difference
between two GPU runs is easy to read as the effect of a change. Draft for `CLAUDE.md`'s
Conventions, after the bucketing paragraph:

> **A seed makes a run reproducible on a CPU, not on a GPU.** GPU kernels may sum in a different
> order from run to run, and float addition is not associative, so two runs with the same seed,
> code and data drift apart: two identical runs on an Apple GPU scored BLEU 8.36 and 8.53. So a
> difference between two GPU runs is not evidence until it is larger than that drift; compare
> several seeds, or reproduce on a CPU, before attributing it to a change. Tests that assert
> exact equality run on the CPU for this reason. Checkpoints save every random generator, so what
> a seed *does* control survives a resume.

Open question before landing it: whether Lecture 9 or the A9 handout should say the same to
students, since A9 compares two GPU runs' BLEU. **Done when** `CLAUDE.md` carries it and that
question is answered.

### #169 Wrap the plumbing, keep the lesson inline, in every notebook

Eric, 2026-09-28: large code cells lose a new student. The rule is in `CLAUDE.md`: code a student
learns nothing from reading belongs in the library; code that *is* the lesson stays written out.

On 2026-10-14 every code cell of 25 lines or more was reviewed (38 cells, 16 notebooks), and
three helpers came out of it in 0.2.8: `course_dir` (find the course folder on Drive and name
every missing file at once), `load_model` (rebuild a model and its vocabularies from one file)
and `translate_to_file` in place of hand-written decode loops. A9, A11, A12, A13, 8a, 8b, 8c,
Lecture 7 and tutorial 5 use them. The long cells that remain are lessons (tutorial 8's beam
search, 8b's attention), data fixtures, or marked `# Setup:` cells serving one notebook, which
the rule allows. A13's `load_best` stays hand-written on purpose: it also loads the A9 model,
which students trained before checkpoints described themselves, so `load_model` refuses it.

**Left:**

- **A12's loader** (`lecture-12-llm-context`, its `# Setup:` cell): a folder loader for one TSV
  or aligned `.txt` files, extending `parallel_txt_to_dataframe`. The layout is now known
  (see #176).
- **An LLM translate helper**, only if a second notebook comes to prompt an LLM. Until then,
  A12's prompting code serves one notebook and stays where it is. Cowork could not say whether
  any of Lectures 14 onward will (Oct 19); the one case so far is A11's extra credit, which runs
  outside the notebook. The question is Eric's.

**Done when** no notebook carries plumbing a library helper covers.

### #132 Quick Start has no notebook, and its badge opens a different one

`docs/docs/getting-started/quickstart.md` carries a Colab badge, and the badge opens
`tutorials/03-train-tiny-model.ipynb` — a different document from the one the reader is on.

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

### #176 The notebook audit

Cowork audited all eighteen notebooks from a student's seat at Eric's request (2026-09-29):
[`NOTEBOOK_AUDIT.md`](NOTEBOOK_AUDIT.md). Its second half, "Implementation directions", is
written to be carried out without asking: six rules for every notebook (standard opener, one
closing section, prose ceilings, marked setup cells, **no project history**, the text rules in
force), per-notebook directions with replacement text, and an order of work. The directions are
the spec; these rows only track which notebooks are done. Eric's test where they leave room:
**the job of each notebook is to teach.**

One task per group that shares a deadline, so the "Critical for" column stays honest.

- **#176, `lecture-12-llm-context`** (L12): rebuilt on Oct 13 as a runnable notebook, run end to
  end with a real model, and tested in CI with the model stubbed (`tests/test_lecture_12.py`).
  Its loader accepts one TSV or two line-aligned text files, a guess at the A12 folder's layout,
  and looks for a file whose name contains `en`. The real layout (Cowork, Oct 19): one folder per
  language, `src.txt` (English) and `tgt.txt`, line-aligned, no header, no split; Church general
  conference text, the same lines in every language; Efik 22,611 lines, Kosraean (folder
  `Kosrean`) 70,208, Palauan 42,662, Pohnpeian 50,818, Telugu 49,611, Yapese 54,080. Step 1 names
  "the A12 page's data link" rather than pasting the URL, so a changed link is fixed in one place.
  **Done when** the loader is checked against one language's real files.

Each PR checks the notebook's prose against its ceiling and runs `scripts/student_path.sh` on it.
**Done when** every notebook the audit names has had its directions applied; then this section
and `NOTEBOOK_AUDIT.md` go.

### #180 Nothing catches a notebook whose Colab badge cannot install TorchLingo

Found 2026-09-29 doing Task #173: tutorial 6 had a Colab badge and no install cell, so Run all
from the badge stopped at `import torchlingo`. **Tutorial 2 had the same defect**, found the same
day by the first student-path run (fixed in PR #192). Every CI gate passed both times, because CI
runs against the repository's own editable install. `CLAUDE.md` now makes Run-all-from-the-badge
part of "self-contained", and nothing checks it.

**The weekly run is not enough, and the check is not redundant with it.** `student_path.yml`
(`execute_notebooks.py --as-student`) would catch this, since tutorial 6 declares no requirements,
but it runs on Mondays and on release tags, so it reports only after the defect has merged, and it
had not yet run at all when this was found. A static check blocks the PR.

**Extend `hygiene()` in `notebook_meta.py`**, no new tool: a notebook with a Colab badge that
imports `torchlingo` must have a code cell installing it (the standard install cell `CLAUDE.md`
prescribes) before its first `torchlingo` import. Notebooks that never import `torchlingo`, such
as Lectures 3 to 6 with their own packages, are exempt. It runs in the lint job beside the other
hygiene checks. **Done when** that check exists, with tests that fail on tutorial 6 as it was
before PR #184 and tutorial 2 as it was before PR #192, and it passes on every notebook as they
are now.

### #160 An NLLB spotlight lecture — data, architecture **and** curriculum

Eric, 2026-09-27: "NLLB is generally state of the art and deserves this level of attention." The
course currently treats NLLB as a data story. His point is that data is one leg of three, and the
architectural and training-curriculum decisions are the ones students never hear.

The three-act shape, handed to Cowork in the 2026-09-27 outgoing entry: **data** (LASER3 bitext
mining, back-translation, FLORES-200 as the benchmark that made 200-language comparison possible,
and the quality and toxicity filtering), **architecture** (sparsely gated mixture-of-experts, the
regularization that stops low-resource pairs being swamped, and the fact that the runnable dense
models are *distillations* of the big sparse one), and **curriculum** (pairs introduced on a
schedule rather than all at once, with high- and low-resource balance managed deliberately).

Slots naturally at Lecture 13 or 14 — it is the worked example both currently lack.

**Our half, and the only part that is ours:** the structure above is from memory of the 2022
paper, not a fresh read. Expert counts, parameter counts, the exact regularization name and the
curriculum schedule all need verifying against the paper before they reach a slide. Offered to
Cowork; do it when they ask, and do not let the remembered figures propagate first.

### #22 `examples/` and `scripts/` are outside the lint gate

CI runs `ruff check src tests` and `ruff format --check src tests`. Both other source trees are
ungated, and `scripts/generate_sentencepiece_models.py` fails three rules on unmodified `main`:
`EXE001` (shebang, no exec bit), `I001` (unsorted imports), `BLE001` (blind `except Exception`).

**What changed the priority**, 2026-09-27: `scripts/notebook_meta.py` is now ~900 lines, carries
77 tests, and gates the roadmap's generated map in CI. It is library code that happens to live in
`scripts/`, and it is the one file here whose breakage would be least visible — a formatting drift
in it is exactly what silently defeated a mutation test earlier today.

`examples/` is worse: `ruff check examples` found 32 errors across five files (unsorted and
unused imports, `f`-strings without placeholders, `typing.List`, a blind `except Exception`), and
**these are the code students are most likely to copy**. The 2026-10-16 scripts
(`student_path.sh`'s Python, `land_pr.sh`) were checked by hand on the way in, which is what stops
happening once whoever does it moves on. (This entry was filed twice; merged on 2026-10-17.)

**Done when** the scope in CI and `preflight.sh` is `src tests examples scripts`. Fix is widening
those lines and clearing the violations. Keep it as its own PR, so pre-existing fixes are not
buried inside a feature diff.

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

Worth retraining because everything downstream reads off this one checkpoint: tutorial 7
shows its translations, the decoding sweep measures on it, and its BLEU is the first number a
student meets.

- Rerun `scripts/train_example_model.py`. Its default is now 40 epochs as a ceiling with
  early stopping, so this no longer risks reproducing the undertrained baseline that caused
  the training-budget error — see [`reports/training-budget.md`](reports/training-budget.md).
- Regenerate `docs/docs/_generated/decoding_sweep.json` afterwards. `--rerender` is not
  enough; the sweep itself must re-run, which takes about an hour.
- Not urgent, and **not** on the CS 479 critical path: no assignment depends on it.
- **The decks now quote this checkpoint** (Cowork, 2026-10-05): `d_model` 128, 4 heads, 3 + 3
  layers, `d_ff` 512, 2,544,056 parameters, 3,000 shared pieces, 64,311 training pairs, 36
  epochs, 24.3 minutes, train loss 3.73, val 3.70, test 1,889 pairs, on the 8b model slide, 8a's
  alignment slide and Lecture 5's corpus slide. **A retrain must be announced in a hand-off** so
  the slides follow, and must keep writing `trained_minutes` and `train_pairs`.

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

1. **It is `PRIVATE`.** Twenty-four students cannot clone a private repository in the lab org.
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

### #123 A14's two-directions case has never been run

**Needed by Wed Oct 28, when Lecture 14 sets A14. Highest uncertainty by evidence.** It was
filed as due Mon Nov 2, A14's old place in the calendar.

`preprocessing.multilingual` has never been exercised at "two directions, intermingled,
separate test sets per direction", which is precisely what Assignment 14 asks for. Lecture
14's handout is a Word document written for OpenNMT and needs replacing outright.

Distinct from **#99**, which is the tutorial. This is the question of whether the code path
works at all, and it should be answered before a notebook is written on top of it.

The notebook starts from A9 (Eric, Oct 13: every assignment after A9 does): its split,
SentencePiece model and configuration, as `lecture-13-back-translation` does. A9's shared
SentencePiece model already covers both languages, so two directions need no new vocabulary.

## What the rest of the roadmap needs from this side

Filed 2026-10-17 from a pass over `notes/CS479_COURSE_ROADMAP.md` for what it asks of the
repository and nothing here tracked: its "What the table cannot say", "Where the line is
crossed", "Open questions" and "Gaps" sections, and Lectures 14 to 23's scope notes. Lecture 14's
notebook is Task #99. Lecture 15's A15 runs on Azure, outside the repository, and Lectures 16 to
18 set no assignment, so none has a row of its own until #246 says otherwise.

### #246 Ask Cowork what Lectures 15 to 23 and the final project need from this side

The roadmap pairs ten sessions with no notebook, by design ("inventing one to fill a row would
be the redundancy `CLAUDE.md` rules out"). Three may still want code, and only the course side
knows: Lecture 18's in-class post-editing activity measures normalized edit distance (does it
want a small notebook, or `torchlingo.evaluation`'s TER?); Lecture 19 works IBM Model 1 by hand
(#253); and the final projects, proposed Nov 16, may build on a student's A9 or A14 system (does
the course want a "starting a project from your model" page?). **Done when** the next hand-off
to Cowork has asked all three and the answers are rows here, or this row is closed as nothing
needed. Asked together, as `CLAUDE.md`'s "ask everything at once" wants.

### #247 Next offering: the A8 kickoff's cap and dedupe become the student's

The roadmap's "Where the line is crossed": `lecture-08a-a8-kickoff` is the one notebook rated
**high**, since it performs the cap, the dedupe, the split and the verification that the A8
slide on AI use calls the student's, and the student edits two file names. It was deliberate,
since the run has to be going when class ends, and A8 has run for this term. **Done when**, for
the next offering, the cap and the source-side dedupe are the student's functions, checked on
known cases and then by `check_contamination`; the split, model and training call stay given;
and a prediction of ln V and the first logged loss stops the run before training. After the
term: changing it now would reach no student and risk the notebook A9 and A13 read from.

### #248 Does "about 35 epochs" hold for a student with 30K pairs?

An open question in the roadmap. The 30-epoch convergence was measured at 100K pairs with a
subword vocabulary; A8 uses words, and low-resource students have far fewer pairs. The A8
instruction's escape hatch ("keep going while validation loss falls") covers a student who
reads it. **Done when** a run at 30K pairs, word vocabulary, A8's 56.4M configuration, has a
measured convergence epoch in `notes/reports/`, and the next A8 handout's guidance follows it.
Needs the A5 audit's count of students well below 100K to say whether it matters.

### #249 Roadmap gap: what to do when training goes wrong

The roadmap's first gap. `concepts/training.md` covers the loop, optimizers, schedulers,
clipping and early stopping, but not what a student does when training goes wrong: batch size
against learning rate, or what overfitting looks like on a small corpus. **Done when** the page
covers both, with a measured example from this repository's corpus rather than a described one
(documentation that executes), linked from tutorial 6's lookup table.

### #250 Roadmap gap: model size against latency

The roadmap's third gap. `concepts/decoding.md` explains beam search's cost and why a GPU at
batch size 1 is mostly overhead; nothing covers model size against latency, which is what a
student meets when a Colab session is slow. **Done when** `concepts/decoding.md` or a new
section of it carries a generated table (decode time per sentence against parameter count, on
CPU and on one GPU), produced by a script into `docs/docs/_generated/`, not typed.
`scripts/bench_decode.py` is the place to start, not a new script.

### #251 Roadmap gap: teach the training-budget episode

The roadmap's second gap. "How much data is enough" is open in general, but the material has a
measured answer to a better question, *which lever are you actually pulling?*: at this scale
training longer helped and more data did not, and the first attempt to answer it was wrong in a
way only a control run caught (`notes/reports/training-budget.md`). The roadmap says the episode
is worth teaching directly. **Done when** a docs page (or a section of `concepts/training.md`,
beside #249) tells it to students, with the report's numbers included from its JSON, not typed.

### #253 Lecture 19: IBM Model 1 by hand, set against 8a's attention alignment

Lecture 19's outcome: "Work IBM Model 1 by hand on a toy pair; explain EM; relate statistical
alignment to the attention alignment of 8a." The roadmap says the lecture needs rescoping around
material now in Lecture 5, and has no notebook. A notebook could run EM on a toy corpus and set
its alignment beside the attention alignment tutorial 4 measures, which is the link the outcome
names and nothing else in the course makes. **After Cowork rescopes Lecture 19** (asked in
#246); Mon Nov 30. **Done when** Cowork has said whether the lecture wants it, and if so it is
built to the course-notebook rules.

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

## Code — other gaps

### #243 An architectural review of TorchLingo, pedagogy first

Eric, 2026-10-14: an architectural review of the library, **keeping pedagogy as the foremost
requirement**. The library grew feature by feature, each change justified alone. This term added
`colab`, `checkpoint.load_model`, `translate_to_file`, `first_logged_loss` and the diagnostics,
and nobody has since asked whether the whole still reads as one design a student can follow.

The test for every finding is the one `CLAUDE.md` puts first: does it make the library easier for
a CS 479 student to read, use and learn from? Speed, generality and third-party use come after,
and a change that serves them at the cost of readability is a finding against it, not for it.

What to look at:

- **Module boundaries and the data flow.** Does the path from text to `TrainResult` to translation
  read in the order a student meets it? Is each module's job one sentence?
- **Two ways to do one thing.** For example `load_checkpoint` beside `load_model`,
  `translate_batch` beside `translate_to_file`, the reference decoders beside `inference_fast`,
  and `Config` defaults beside explicit parameters. For each pair, is the reason it exists
  written where a student would find it?
- **The `Config` pattern.** Is the rule that explicit parameters always win applied everywhere,
  and is it something a student can predict?
- **Interfaces detected with `hasattr`** (Transformer against LSTM; vocabularies) against an
  explicit protocol. Which one teaches the contract better?
- **What the notebooks actually use**, from the long-cell survey of 2026-10-14 and the course
  notebooks' imports. Public API nobody calls is a candidate to cut, and plumbing every notebook
  repeats is a candidate to add.
- **Docstrings and docs as the student's first reading**: the Examples that run, and the names
  that say what they do.

**Done when** a report in `notes/reports/` lists the findings, each with the student-facing
reason, a recommendation and a cost, and Eric has chosen which to act on. Each one chosen
becomes its own task. No code changes before then, as with the long-cell survey: review
precedes implementation.

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

**#254 One way to wait for a GitHub run, shared by the scripts**

On 2026-10-16 and 17, waiting on GitHub was written by hand four times: a dispatched workflow on
a branch (twice), a PR's checks, and the student-path run on `course`. `gh run watch` is not
the answer, since one network blip killed it mid-run. `promote_course.sh` and `land_pr.sh` each
carry their own loop, with slightly different rules for a failed call. The global rule on
authenticated connections asks every one of them for the same three things: an attempt cap, a
deadline, and a stop at the first authentication failure.

**`scripts/await_run.sh`**, taking a run id, a workflow and commit, or `--pr N`, that does those
three things, treats any other failed call as "not yet", prints a timestamped line per poll, and
exits with the run's conclusion. **Done when** `promote_course.sh` and `land_pr.sh` call it
instead of their own loops, and `CLAUDE.md`'s Commands list names it.

**#255 `student_path.sh --at COMMIT`: run a notebook as it was**

Proving the Colab check (Task #244) caught the October COMET break meant running A11 as it was at
three commits: `git show COMMIT:path > file` by hand for each, then `student_path.sh` on each
file. That recurs whenever a notebook fix needs "fails before, passes after" shown, which is how
a fix to an install cell should be verified. **Done when** `student_path.sh --at COMMIT NOTEBOOK`
extracts the notebook at that commit into the run directory and runs it, so
`--colab --install-only --at b98709c docs/docs/course/lecture-11-comet-on-a6.ipynb` reproduces
the numpy failure in one command. The install cell still installs from `course`; the header says
so, since only the notebook travels back.

**#256 Check `TASKS.md` itself: duplicate entries, rows without bodies, empty sections**

The 2026-10-17 clean-up found Task #22 filed twice, Task #79 done since Sep 27 and still listed,
two empty sections, and a paragraph splitting the label table. The first three are mechanical,
and a file whose convention is "a finished task is deleted" has nothing checking it. **A
`--check-tasks` in `notebook_meta.py`'s family, or a small `scripts/check_tasks.py`, run by
preflight**: every task number has one table row and at most one body; a body exists only with a
row; no section heading is empty; the table has no non-row line inside it. It cannot tell a done
task from an open one; that stays a reading at each baton pass. **Done when** it fails on the
file as it was at `061031f` and passes now.


---

## The training-budget finding

**The finding itself now lives in [`reports/training-budget.md`](reports/training-budget.md).**
It is a measurement, not a task, and it is the prior for #119's learning curve: training
budget beat data by about 7x, and the data effect's confidence interval crossed zero.

## Tests

**#15 Existing `DummyTransformer` is history- and memory-blind**
`tests/test_training_inference.py:16` computes logits from a zero tensor, so its output
depends only on decoding *position*. Verified: logits are byte-identical for different
decoder histories AND for different encoder memories. The three existing beam tests
therefore cannot detect scrambled beam state or bad memory expansion.
- Mitigated by the new `tests/test_decoding_equivalence.py`, but the old tests should
  eventually migrate to the history-sensitive fixture rather than sitting alongside it.

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

## Docs and tutorials

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

**#53 The notebook gate is weaker than its green check implies**

CI checks out without Git LFS on purpose, so `data/example.tsv` is a pointer and
`execute_notebooks.py` skips what are now tutorials 3, 4, 7 and 8. **The job passes having run two notebooks
of six**, and the check mark looks identical either way.

Found concretely in #50, which adds an assertion inside tutorial 8 whose whole purpose is to
fire when the model stops being decisive. It cannot fire in CI, because tutorial 8 does not run
there.

The skip list only grows as tutorials touch real data. It can go the other way: tutorial 2 uses
fixed strings, needs no artifact, and runs — a design lever worth knowing, since a tutorial
whose subject needs no trained model should not acquire one.

Options, cheapest first: **say it in the check** (the job already knows what it skipped, so put
that in the job summary); **fetch LFS for the notebook job only**, ~28 MB per run, which the
keep-CI-light decision ruled out on four-version-matrix grounds that do not apply to one job;
or a **scheduled full run** with LFS.

**Done when** the green check states what it did not run. Same family as #51.


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

