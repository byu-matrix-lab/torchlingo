# TorchLingo — Session Task List

Opened 2026-08-22, last updated 2026-10-25. Numbered for reference in conversation.
Completed work is removed rather than marked done — git history is the record.

**This list is library and notebook engineering.** Course work (decks, Learning Suite, the
assignment pages, and anything waiting on Cowork or Eric for a course decision) is in the one
list both sessions share, `torchlingo-private/notes/course/TASKS.md` (Eric, 2026-10-25). Task
numbers are kept across the two, so a number names the same task in either file.

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

**2026-10-19: the baton came back and went out again the same day.** Cowork's hand-off answered
Tasks #176, #231, #233, #235 and #242. Tasks #231, #233, #235, #242 and #252 are closed, and
Task #176 is on `course` pending one real A12 folder. The return hand-off,
`to-cowork/2026-10-19-judges-scored-a12-a13-on-course.md`, asks for that folder's first lines,
an MTEval results file (Task #229) and what Lectures 14 onward need (Task #246). Task #169's
LLM helper waits on Eric.

**2026-10-19, afternoon: the baton came back with Lecture 11 turned around.** Lecture 11 is
recorded, and its activity judges Spanish into English; the scores are in
`torchlingo-private/course/figures/l11-judges/` (`*_es-en.*`). Eric's scope: both sessions stay on
Lectures 11 to 13 and A11 to A13, and later lectures become roadmap notes, so Task #246 is closed
into the roadmap's open questions, and Cowork holds the MTEval file (Task #229) until Lectures 11
to 13 are done. Task #169's LLM helper: "almost certainly" needed, to be confirmed.

**2026-10-20: Cowork sent the MTEval sample and the A12 folders.** A11's Step 1 now reads the
student's MTEval results file and compares each metric with the student's own ranks, sentence by
sentence (Task #229, closed).

**2026-10-21: the arc plan's A12 and A13 are on `course`, and the baton went to Cowork.**
Sampled and noised-beam decoding and the NLLB accessory (Task #267, PR #327), A12's required
additions (PR #328) and the A13 rewrite (Task #268, PR #329) landed. Task #270, A14's rewrite,
is Tasks #123 and #99 here, waiting on the A14 due date, which Eric sent to Cowork. Filed:
Task #271 (A15, later), Task #272 (batched beam) and Task #273 (the install-only check before
merge).

**2026-10-22: the baton came back with a new schedule, a simpler A13 and A14's go.** No two
assignment windows overlap: A12 is due Fri Oct 30, A13 Wed Nov 4, A14 Tue Nov 10, and A15 is
dropped this term, folded into the Final Project (Eric). On this side: A13's notebook simplified
to one retrain on sampled back-translations, no partner step, seven questions, with A12's
opening outcomes (Task #274, PR #334); the roadmap's schedule, outcomes, practice table and
decisions log, with `render_roadmap.py` taking due dates between lecture rows and a dropped
assignment (Task #275, PR #335); and the A14 notebook (Task #99, PR #336), after Task #123's toy
run found the two-directions path works with A9's pieces. All three landed the same day by
admin merge (Eric's OK) and were promoted together. Task #271 moves to the next offering;
Task #278 filed.

**2026-10-24 and 25: the baton came back with A12's sentence before, and a fine-tuning
tutorial.** A12 gains a fourth outcome, context for disambiguation: Cowork's Step 6 in the
Lecture 12 notebook, reviewed here, landed and on `course` before A12 opens (PR #339). The
roadmap took the Final Project's seven parts and the late-semester shift (PR #340). Landing got
faster: a push to `main` cancels the run it supersedes, and `land_pr.sh` lands several PRs with
one promotion (PR #338). Tasks #160, #120 and #253 moved to the course list. A12's placebo is
drawn per sentence, its chart survives tied draws, and it ran end to end on Efik (PR #341).
Tutorial 9, fine-tuning NLLB, landed (PR #342), with Adafactor after AdamW's checkpoints
measured 6.9 GB each. Eric chose **go** on rebuilding A14 on it, with one Colab run first:
A14 is rebuilt both ways with COMET (PR #343), on `torchlingo_harness.nllb.fine_tune`. Filed:
Task #279 (that Colab run), and Tasks #281 to #294 from a two-lens review of every notebook
(Eric's request): code that belongs in the library, and where students decide. The P1 to P3
criteria were rewritten with Eric and every row re-ranked. Then, the same day: A13 and A14 ask
which differences are real, and A13 names its training ingredients (PR #345); tutorials 4, 6, 7
and 8 checkpoint every run (PR #346); `CLAUDE.md` records Eric's two principles (PR #347); and
Tutorial 9 became "how much fine-tuning is enough, and what it costs", so that it and A14 do not
look alike (PR #348). Filed: Task #295, a batch landing stuck on a last PR that triggers no CI.

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
| **L11** | LLM Judges, Sentences and Spans — **has already run** | Wed Oct 21 · A11 set |
| **L12** | Using LLMs for MT; Expanding Context Awareness | Mon Oct 26 · A11 due · A12 set (due Fri Oct 30) |
| **L13** | Strategies for NMT of Low-Resource Languages | Wed Oct 28 · A13 set (opens Fri Oct 30, due Wed Nov 4) |
| **14+** | Lecture 14 onward: multilingual and zero-shot, speech, and later | from Mon Nov 2 · A14 set (opens Wed Nov 4, due Tue Nov 10) |
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

**P1 to P3 rank the work; the table is sorted by it** (criteria Eric approved Oct 25, replacing
those of Oct 5):

- **P1: a student meets it within about three weeks.** Something students run or are graded on
  in that window is broken, missing, or unverified on the path they will actually take; or a
  dated promise to Eric or Cowork falls inside the window.
- **P2: it makes this term's assignment notebooks load-bearing and correct, or guards what
  students are served.** A lesson an assignment leaves to a tutorial; a decision, prediction or
  interpretation missing from a coming assignment; a standing rule that shipped code breaks;
  protection for `main` and `course`; or something Eric asked for, with no date.
- **P3: everything else.** Library refactors that change nothing students see this term,
  tutorials (they carry no points, so students pressed for time skip them: Eric, Oct 25), lectures that
  have already run, the next offering, and housekeeping.

Within a level, the earlier date comes first. A tutorial-only task is P3 unless it breaks a
standing rule. Re-rank when the calendar moves; "Critical for" says *which* lecture, P says *how
soon*. Re-ranked Oct 25: Tasks #272, #169, #171 and #82 went from P2 to P3, and Task #280, the
two-lens review of every notebook, became Tasks #281 to #294.

| | P | Task | Critical for | State |
|---|---|---|---|---|
| #279 | **P1** | One Colab L4 run of the rebuilt A14 before it opens | **14+** | Open — Eric's condition for the go (Oct 25): Eric or Coulson runs `lecture-14-multilingual` once on an L4 on a real A8 language, before Wed Nov 4. Postponed to Mon Oct 26 (Eric). The Colab install-only check passed on the promoted notebook (Oct 25), so the install cells are known good; the bfloat16 path, L4 memory, the download and the real step time are not (this Mac's GPU, float32: about 3 to 7 s a step). Open: who runs it, and whose A9 files: a student-like `MyDrive/CS479/assignment8`, or a test folder built from the course's German split with a correctly sized A9 model. Fix whatever it finds before Nov 4 |
| #295 | **P2** | A batch whose last PR triggers no CI promotes nothing | **hyg** | Open — found Oct 25: `land_pr.sh 345 346 347 --promote` merged all three, then waited to promote PR #347's commit, which changed only `CLAUDE.md`, so CI's path filter ran nothing on it and `promote_course.sh --await` waited for a run that never comes. Promoted PR #346's commit by hand instead (the trees differ only in `CLAUDE.md`). Fix in the existing tools: promote the newest commit of the batch whose checks ran, or have `promote_course.sh` accept a commit whose only changes since a checked commit are files CI does not watch; and the batch's stop report should name that commit, not `main`'s head |
| #273 | **P2** | Run the install-only student check on the notebooks a PR changes, before it merges | **hyg** | Open — PR #329's COMET cell read a setup-cell helper, harmless to a student running from the top, and `student_path.sh --colab --install-only` caught it only after promotion, on `course` 6405a02 (fixed in PR #331). Extend `merge_check.sh` or `land_pr.sh` to run it on each changed notebook with an install cell (about 35 s each; needs uv), not a new tool |
| #180 | **P2** | Nothing catches a notebook whose Colab badge cannot install TorchLingo | **hyg** | Open — a per-PR check in `notebook_meta.py` (Eric, Sep 29: keep it) |
| #53 | **P2** | Notebook gate runs 2 of 6 tutorials in CI, and looks green | **hyg** | Open |
| #243 | **P2** | An architectural review of TorchLingo, pedagogy first | **lib** | Open — a review, reported before any change (Eric, Oct 14) |
| #281 | **P3** | No assignment notebook leaves a lesson to a tutorial: A7 to A11 | **L7** | Open — Eric, Oct 25: tutorials carry no points, so students pressed for time skip them, and the assignment notebooks are load-bearing. Done Oct 25 for A12 to A14, the assignments still to come: A12 relies on no tutorial; A13 and A14 gained a sentence each on beam search and chrF (PR #345). A7 to A11 have run; audit them for the next offering |
| #287 | **P3** | `load_model` takes a configuration for checkpoints saved before Oct 14 | **lib** | Open — it raises `ValueError` on files that do not record the model's settings (before PR #280), which covers most students' A8 and A9 models, so A8 to A14 rebuild models by hand. With a fallback, the notebooks can call it, next offering |
| #288 | **P3** | One COMET helper in `torchlingo.evaluation` | **lib** | Open — the install block is in A10, A11, A13 and A14, the scoring loop in A10, A11 and A14; A13 uses COMET's command line. Keep the student's own `comet_inputs` where that is the lesson. COMET's install pins `transformers` below 5 |
| #289 | **P3** | `paired_bootstrap` and a score table in `torchlingo.evaluation` | **lib** | Open — near-identical in A13 and A14, the identical-output *p* = 1 fix included; Final Projects will want it |
| #290 | **P3** | Notes and rerun-record helpers in `torchlingo.colab` | **lib** | Open — `load_notes` and `save_note` in A13 and A14; the rerun record in A12, A13, A14 and Tutorial 9, with library versions added by hand each time |
| #291 | **P3** | NLLB plumbing through the accessory | **lib** | Open — A12's cell 20 reloads NLLB and decodes greedily by hand where `load_nllb` does it; Tutorial 9's cell 14 resumes decodes by hand where `translate_lines_to_file` does it |
| #292 | **P3** | Decisions and named ingredients in the lectures already run | **L5** | Open — stopping cells: Lecture 5 (predict *c*), A8 (predict the duplicate share, then print the most repeated), A9's Part B (the choice its markdown promises), A11 (will COMET confirm A9's BLEU verdict?); name gradient clipping, dropout and the learning rate in Lectures 7, 8a and 9; and Lecture 8c's student code is tensor plumbing, where a `which_head` exercise would be about translation. Next offering |
| #293 | **P3** | Should tutorials 1 to 8 have stopping cells? | **hyg** | Open — none has a prediction, decision or TODO. Eric decides; candidates from the review: tutorial 1's ids, tutorial 2's ship decision, tutorial 3's rebuild prediction, tutorial 4's alignment accuracy, tutorial 6's which check catches the broken corpus |
| #294 | **P3** | Small duplications across notebooks | **lib** | Open — the tiny phrase model in Lecture 7 and tutorials 3 and 8; attention capture written twice (8b by wrapping, 8c with hooks); a `read_lines` in five notebooks; tutorial 7 rebuilds a model by hand where `load_model` works; tutorial 6 repeats one `train_model` call four times |
| #272 | **P3** | Batch beam search across sentences, not only across beams | **lib** | Open — `inference_fast.beam_search_decode` decodes one sentence at a time: measured Oct 21 at A9's size (d512, 6+6 layers, vocabulary 8,000, Apple GPU, an untrained model running the full 40 tokens), 372 ms a sentence against greedy's 7.3, so beam back-translation of 100,000 sentences is about 10 hours. A13 back-translates by sampling only (Eric, Oct 22), so no student waits on it. Output must stay token-identical (`tests/test_decoding_equivalence.py`) |
| #169 | **P3** | Wrap the plumbing, keep the lesson inline, in every notebook | **hyg** | **Open** — 0.2.8's `course_dir`, `load_model` and the `translate_to_file` decodes moved A9, A11, A12, A13, 8a, 8b, 8c, Lecture 7 and tutorial 5; left: an LLM helper, once a second notebook prompts an LLM (Eric, Oct 19: later lectures "almost certainly" will, to be confirmed against the roadmap; an open question there). A12's loader stays in its notebook's `# Setup:` cell: it reads exactly two named files, and serves one notebook (Eric, Oct 19). Tasks #287 to #291 are its next pieces |
| #171 | **P3** | Document GPU nondeterminism next to the seeding conventions | **L9** | Open — draft wording below; two identical Apple-GPU runs scored BLEU 8.36 and 8.53 |
| #82 | **P3** | Add an on-target language check to `torchlingo.diagnostics` | **lib** | Open |
| #271 | **P3** | An A15 notebook: speech to speech around the student's own MT | **L15** | **Next offering.** A15 is dropped this term, folded into the Final Project as a Path B example; its page is kept for next year (Eric, Oct 22). The arc plan's A15: the student's best of A8, A9 or A13 in the middle, Whisper for speech recognition and MMS for synthesis where Azure falls short, word error rate on the 10 sentences, and one sentence traced through all three stages |
| #278 | **P3** | `train_model` accepts `device` as a string | **lib** | Open — `train_model(device="cpu")` raises `AttributeError` at `device.type` (`training.py`); every notebook passes a `torch.device`, so no student meets it. Found Oct 22 in Task #123's toy run |
| #167 | **P3** | Prune merged branches and stale worktrees | **hyg** | **Left:** `myles_testing` (ask Myles), `coulson`, and four local branches whose commits are not in `main` |
| #132 | **P3** | Quick Start has no notebook, and its badge opens a different one | **lib** | Open — **was only ever in the session mirror** |
| #149 | **P3** | `collect_benchmark.py` silently drops a run file it cannot find | **lib** | Open — it wrote a 2-run report over a 21-run source |
| #49 | **P3** | The shipped checkpoint predates the enlarged corpus | **L8b** | Open — `train_pairs` 64,311 against a corpus of 86,430 |
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

- **An LLM translate helper**, only if a second notebook comes to prompt an LLM. Until then,
  A12's prompting code serves one notebook and stays where it is. Eric, Oct 19: lectures after
  12 "almost certainly" will, to be confirmed against the roadmap's scope notes (an open question
  there); the one case so far is A11's extra credit, which runs outside the notebook.

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

## What the rest of the roadmap needs from this side

Filed 2026-10-17 from a pass over `notes/CS479_COURSE_ROADMAP.md` for what it asks of the
repository and nothing here tracked: its "What the table cannot say", "Where the line is
crossed", "Open questions" and "Gaps" sections, and Lectures 14 to 23's scope notes. Lecture 14's
notebook exists (`lecture-14-multilingual`, Oct 22). Lecture 15's A15 is dropped this term (Eric, Oct 22), and Lectures 16 to
18 set no assignment, so none has a row of its own. What Lectures 15 to 23 and the final project
may want from the repository (Lecture 18's edit distance, Lecture 19's IBM Model 1, a page on
starting a project from a student's own model) is an open question in the roadmap, held until
Lectures 11 to 13 are done (Eric, Oct 19).

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

