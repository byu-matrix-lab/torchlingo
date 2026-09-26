# TorchLingo — Session Task List

Opened 2026-08-22, last updated 2026-09-23. Numbered for reference in conversation.
Completed work is removed rather than marked done — git history is the record.

**Numbers here are task numbers, and they collide with pull request numbers.**
Tasks run to #114 and PRs to #81, so every number below 82 names one of each. Say
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

What passes the filter today: #16 and #113, because the release is what Monday depends
on; #42 and #94, which are Lecture 7 itself; #95, #96, #100, #109 and #110, which
Assignment 8's text needs before it is handed out on Sep 30; and #112, which builds the
measurement the others wait on.

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

| | Task | State |
|---|---|---|
| #94 | One Colab link for the Lecture 7 in-class activity | **Due Mon Sep 28** — in review, PR #73 |
| #96 | Restate the step count in epochs | **Due Wed Sep 30** — unit decided |
| #100 | Put the alignment check in front of students | **Due Wed Sep 30** |
| #95 | One end-to-end 100K-pair run: wall clock and BLEU | **Due Wed Oct 7** — corpus ready, start now |
| #97 | SentencePiece on versus off, controlled | **Due Mon Oct 12** |
| #102 | Inference cannot resume a long decode | **Needed by Mon Oct 19** — largest undone piece |
| #98 | Back-translation as a documented workflow | **Due Mon Oct 26** |
| #99 | Multilingual tagging tutorial, replacing the OpenNMT handout | **Due Wed Oct 28** |
| #101 | Give the tutorials stable unique names | Open — after the tutorial PRs land |
| #103 | Extend the notebook gate to `docs/docs/course/` | **Unblocked** — four have arrived |
| #106 | A token cap breaks Assignment 9's control | Open — one sentence in the assignment |
| #107 | The optimizations exist and nothing uses them | Open — 74% of an epoch is wasted padding |
| #108 | Nothing releases the device allocator's cache | Open — measured: 99.8% of held memory is reclaimable |
| #109 | A8's 100K floor has no low-resource variant | **Open — Eric, before Oct 7** |
| #110 | OpenNMT evidence implies 65 to 165 epochs, not 30 to 36 | Open — settle before A8's text |
| #111 | Does the Lecture 6 activity notebook go public? | **Open — Eric's decision** |
| #112 | Expired Lecture 6 refs in the briefing; build the ladder | Briefing done — ladder blocked on PR #84 |
| #113 | Land the six PRs still open | All six refreshed and green against today's `main` |
| #115 | The four course notebooks, and the deck's link switch | **Done** — PR #85 merged; badges resolve |
| #116 | `create_dataloaders` discards the `Config` it is handed | **Done** — PR #84 merged; ladder unblocked |
| #117 | A PR based on `main` self-closed during an unrelated merge | Open — recovered; contradicts `CLAUDE.md` |
| #118 | What does a paid Colab session actually provide? | **Coulson** — blocks any A8 memory claim |
| #114 | The wheel ships no data, so tutorials 4 and 5 cannot find it | Open |
| #4 | Resolve length-normalization semantics | In review — PR #54 |
| #7 | PyTorch deprecation warnings | In review — PR #57 |
| #8 | Verify Eole claims before syllabus use | Open |
| #9 | `pre-commit install` (still not installed) | Open |
| #15 | Migrate history-blind `DummyTransformer` tests | Open |
| #22 | `examples/` and `scripts/` are outside the lint gate | Open |
| #28 | Attention params skip `_init_weights` | Open |
| #36 | CI actions pinned to a deprecated Node runtime | Open |
| #42 | Lecture 7 assignment | **Routed to Cowork** — likely closes with nothing built |
| #44 | Gate the sdist on "no Git LFS pointer shipped" | Open |
| #48 | Audit pedagogical value; write down sequencing and outcomes | In progress — `notes/CURRICULUM.md` |
| #51 | The docs gate reports but does not block | Open — repo settings |
| #52 | Try Moore (2002) if more of the corpus is wanted | Open |
| #53 | Notebook gate runs 2 of 6 tutorials in CI, and looks green | Open |
| #60 | Nobody is told when main goes red | Open |
| #63 | Add mkdocs nav entries for three pages | In review — PR #51 |
| #65 | Make tutorial 6 import the checks instead of redefining them | In review — PR #52 |
| #66 | Adopt `nltk.translate.gale_church`; split #29 into two jobs | Open |
| #68 | Cite `torcheck` as prior art in the diagnostics docs | Open |
| #70 | Print the sacreBLEU signature with every score | In review — PR #55 |
| #71 | Decide whether to report the Joey NMT breakage upstream | Open — Eric's call |
| #72 | Prune the prose entries for finished tasks | Open |
| #74 | Fix the broken anchor and diagnose the 93 docs warnings | Anchor in PR #51; the 93 warnings still open |
| #78 | Tutorial 5's committed outputs predate the retrained checkpoint | In review — PR #57 |
| #79 | An order-dependent test; does not reproduce on main today | Open |
| #81 | Fail the build on hand-typed generated numbers | Open |
| #82 | Add an on-target language check to `torchlingo.diagnostics` | Open |
| #83 | Show attention on the Transformer, not only the LSTM | In review — PR #59 |
| #84 | The two evaluation pages ARE off-nav on main now | **Open — the prediction came true** |
| #85 | Only BLEU carries a signature; chrF and TER do not | Open — after PRs #55 and #58 |
| #86 | `evaluate_model` has no test, and it is what callers use | Open — after PR #58 |
| #88 | Open the tutorial 7 PR | **Unblocked — PR #58 has merged** |
| #89 | Fail the docs build when a page is off-nav | Open |
| #90 | `CLAUDE.md`'s numbering example is stale | Open |
| #91 | `metric_comparison.json` records no BLEU signature | Open — after PRs #55 and #58 |
| #92 | Tutorials 3 and 5 bypass the library's own evaluation API | Open |

## The CS 479 pivot

From `notes/CS479_COURSE_ROADMAP.md`, handed over 2026-09-24. OpenNMT-py is in maintenance
mode and CS 479 is moving to TorchLingo **this** semester. Five assignments train a model
(Lectures 7, 8, 9, 13, 14), all five are written against OpenNMT today, and they are
consecutive and cumulative, so it is one crossing rather than five.

This converts the repository from a teaching library with a hypothetical audience into the
thing eighteen students have to get working, on their own data, on a deadline.

### The calendar

Day-of-week verified against 2026 for every date below.

| Task | Needed in class | Assignment due | Slack |
|---|---|---|---|
| #94 Lecture 7 Colab link | **Mon Sep 28** | — | 4 days |
| #96 step count to epochs | Wed Sep 30 | Wed Oct 7 | 6 days |
| #100 alignment check | Wed Sep 30 | Wed Oct 7 | 6 days |
| #95 100K run, wall clock and BLEU | Wed Sep 30 | **Wed Oct 7** | 13 days |
| #97 SentencePiece on/off | Mon Oct 5 | Mon Oct 12 | 18 days |
| #98 back-translation | Mon Oct 19 | Mon Oct 26 | 32 days |
| #99 multilingual tagging | Wed Oct 21 | Wed Oct 28 | 34 days |

**Two dates decide the pivot.** Mon Sep 28 is the first contact and the one that happens in
a room with eighteen laptops. Wed Oct 7 is Assignment 8, the heaviest in the course, and the
one that answers whether students can actually train on their own data.

### Order, and the one place to push back

The roadmap's suggested order is straight down the calendar. Two changes:

**Resume is already verified, so nothing gates #95.** The roadmap puts "verify Colab resume"
first, on the grounds that it had been written twice and run zero times. That is no longer
true. Coulson tested it in Colab and reported on PR #17, 2026-09-23: *"Was able to mount and
continue runs using Google Drive!"* That covers #38's item 1 (mounts, lands under `MyDrive`)
and item 3 (continues rather than restarting), which were the two that mattered. His review
shows as `DISMISSED` only because a later push dismissed it; the finding stands.

What is left is thin and not worth its own experiment: no transcript, so there is no record
that it trained *only* the remaining epochs rather than restarting and looking like it
continued. #95 runs on Colab anyway and will exercise resume as a side effect, so that
confirmation is folded into #95 rather than run separately.

**#95 is the long pole in wall-clock, not in effort.** A 100K-pair run is hours that cannot
be compressed, so it starts now and the short work happens while it trains.

So: start #95, then #94 before Monday, then #96 and #100 off the numbers #95 produces.

### What the course does not need

Worth stating, because it bounds the work. The course uses SacreBLEU and COMET directly, so
TorchLingo's evaluation module must be **correct** but need not become a metrics suite. LLM
prompting (Lecture 12), quality estimation (Lectures 10, 11) and speech (Lectures 15, 16)
all run on other tooling. Course decks and assignment text are Eric's, not this
repository's.

### Two discrepancies in the roadmap itself

- Its "Open questions" section says *"Assignment 8 is due Sep 30."* The schedule table and
  the hard dates both say **Wed Oct 7**; Sep 30 is when the material is first needed in
  class. The later date is the one used above.
- The header says "corrected 2026-09-25" and the repository assessment "Sep 25", but it was
  handed over on Sep 24. **Resolved: a UTC timestamp**, so the document is not from the
  future and nothing else in it needs re-dating.

### Students are on paid Colab

Eric's call, 2026-09-24: **students should be running on a paid Colab subscription**, and
anyone who has not started one needs to now. Two consequences for the work below.

- Benchmark #95 on a paid-tier GPU, not a free T4. The roadmap's "free Colab T4" framing is
  superseded, and a wall-clock number measured on the wrong tier would be worse than none,
  because it would be quoted at students.
- The subscription is a **prerequisite with a deadline**, like the MTEval accounts. It is
  course communication rather than library work, so it sits with Eric, but it belongs in
  whatever the Lecture 7 activity tells students to have ready.

Resume still matters on paid Colab. Sessions are longer, not unlimited, and a student who
loses a 100K run at hour three loses it just as completely.

### #94 One Colab link for the Lecture 7 in-class activity

Twenty minutes, eighteen laptops, no local install, on a paid Colab GPU. Tutorial 2, "Train
a Tiny Model", already maps onto the OpenNMT Quickstart activity it replaces, and six
tutorials already carry Open-in-Colab badges, so this is framing rather than capability.

- It must **fail loudly rather than quietly**. The Fall 2025 Lecture 8 debrief shows the
  install activity went badly for several students on a stack that had been in use for
  years; a new framework raises that risk.
- The failure mode to design against is a student who gets no output and cannot tell
  whether the library is broken or they are.

### #95 One end-to-end 100K-pair run

The largest controlled run on record is **64,311 pairs, 36 epochs, BLEU 7.32**. The
assignment asks for at least 100,000 pairs and expects "reasonably intelligible output".

Two numbers are wanted, and the assignment text cannot be honest without them: **wall-clock
time on a paid Colab GPU**, and **achievable BLEU**. A third falls out for free: whether it
fits a student's compute budget at all.

- Start it now. It is hours of wall clock that cannot be compressed.
- Run at the epoch count #96 settles on, not at some other number, or it measures the wrong
  thing.
- Confirm resume along the way, which closes the one residual from #38 item 3.
- Route the numbers through a JSON single source of truth, as with every other quantitative
  claim here.

### #96 Restate the step count in epochs

**Decided 2026-09-24: epochs are the unit.** So this is no longer a choice between units,
only the question of which epoch count.

**The 20,000 came from a config nobody here has.** Eric has never used OpenNMT; the figure
is inherited from Steve Richardson's Fall 2025 offering. Converting it needs that config's
`batch_type` and `batch_size`, and without them the translation swings wildly: at sentence
batches of 64, 20,000 steps is 12.8 epochs on 100K pairs, while at token batches of 4096 it
is closer to 33. Those two readings differ by almost 3x, and one of them undertrains badly.

So stop reverse-engineering it. **Set the epoch count from this repository's own
measurements instead**, which is the only evidence available that is actually about
TorchLingo:

| data | epochs | BLEU |
|---|---|---|
| 53,520 | 20 | 4.96 |
| 53,520 | 36 | **7.01** |
| 64,311 | 36 | **7.32** |

Holding data fixed, 20 to 36 epochs bought **+2.05 BLEU**. Adding 20% more data at fixed
epochs bought **+0.29, CI [−0.16, +0.71]**, crossing zero. Training budget mattered roughly
**7x** more than data volume.

**Recommendation: 30 to 36 epochs.** It is the only budget measured to produce this
library's best output, and it happens to coincide with the token-batching reading of 20,000
steps, which is weak corroboration rather than the basis.

- Note the risk this exposes in the assignment as written: it puts a hard floor on the thing
  that did not matter (100K pairs) and a soft, unit-ambiguous floor on the thing that did.
- `step_limit` exists and is honoured in `training.py:450`, so a step figure *can* still be
  expressed if a student is handed one. It is a cap rather than a target, so it is a fallback
  and not the recommendation.
- Whether 36 epochs on 100K fits a session at all is #95's question. 36 epochs is 56,268
  steps at `batch_size=64`.

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

### #100 Put the alignment check in front of students

**Not in the roadmap's numbered sequence; added here, so it is the easiest one to cut.**

Tutorial 6 already checks corpus alignment with `diagnose_alignment` and makes the point
that a misaligned corpus still loads, still batches and still shows a falling loss.

That is the most expensive mistake available in this course. Lectures 4 and 5 are entirely
about producing an aligned corpus, and Lecture 8 is where a bad one finally surfaces, three
weeks later. The roadmap calls getting this in front of students before Assignment 8 high
value for low effort, and it agrees with the Fall 2025 debrief, which lists dirty data as
one of the three reasons student models produced bad output.

### #102 Inference cannot resume, so a long decode cannot survive an interruption

Found 2026-09-24 while mapping the curriculum placements onto repository work. Not on any
list before that, and it is the **largest undone piece of work for the second half**.

Training has checkpoint-and-resume, verified in Colab. **Inference has nothing.**

Assignment 13 back-translates at least as many sentences as the training set, so 100,000 or
more. Throughput is *not* the problem, which was the expected answer and the wrong one: the
decode benchmark measures **27.5 ms per sentence** with batched beams, so 100K extrapolates
to well under an hour and less on a GPU. Read that as an order of magnitude, since it was
measured on 8 sentences at `max_len=25` and a real model at `max_len=60` will be several
times slower.

The problem is that a multi-hour decode dying at hour two starts over from zero. That is
exactly the failure that made resume a priority for training, one level up.

- **Write output incrementally and skip inputs already done.** Better for students, because
  it asks no discipline of them.
- **Or decode in explicit shards**, so a failure costs one shard. Cheaper to build, easier
  to explain, relies on the student following the workflow.

Needed before Assignment 13's material is due in class on Mon Oct 19. Nothing before then
blocks on it, so it is not this week's work, but it is not small either.

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

### The machine crash of 2026-09-25, and what it taught

A 36-epoch benchmark run left unattended consumed all application memory and took the
machine down. It died in epoch 1 around step 800 of 1563, so within minutes rather than
over hours. Written down because three separate findings came out of it and two of them
are about this repository rather than about the accident.

**What actually caused it.** Not the library. The benchmark script used a plain
`DataLoader(shuffle=True)`, so batches were random rather than length-sorted, and padded
toward the long tail. Attention memory is quadratic in the longest member of a batch:

| longest in batch | per batch, forward only |
|---|---|
| 100 tokens | 0.18 GB |
| 512 tokens | 4.83 GB |
| 568 tokens | 5.95 GB |

Roughly double that with activations stored for backward. On Metal that is unified
memory, so it is application RAM, and nothing ever releases it. See #107 and #108.

**The library's defaults are defensible and were not changed to cover this.**
`NMTDataset` truncates at `max_seq_length`, which is 512, and 512 is genuinely the
positional encoding's capacity. Lowering a library default to compensate for a script
that failed to pass `max_length` would have hidden the lesson.

**Nothing was lost.** Both repositories came back clean: no stale locks, no partial
commits, no corrupt index. The session task list and this file both survived.

### #107 The optimizations already exist and nothing uses them

Measured on the 100K training split, real tokenizer, batch 64:

```
real tokens                2.90 M per epoch
random batching, padded   10.97 M    3.79x waste
length-bucketed, padded    2.90 M    1.00x waste
saving                     8.06 M tokens per epoch, 74%
```

`BucketBatchSampler` and `create_dataloaders(use_bucketing=...)` already exist, as does
`train_model(use_amp=...)` with bfloat16 where supported, and `num_workers` and
`pin_memory`. **All of them default off, and the benchmark used none.** That is a failure
against the standing rule to check the inventory before hand-rolling, and it is most of
why memory ran away.

- Rewrite the benchmark to use bucketing, and AMP on a GPU.
- Decide whether `use_bucketing` should default True. It changes batch composition and
  therefore results, which argues against flipping it silently; but 74% matters
  enormously to a student on a Colab budget, so the course guidance should say to enable
  it even if the default stays.
- Not a gap: decoding is already optimized, 27.5 ms against 109.5 ms for the reference
  path, because `inference_fast` batches the beams within a sentence.

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

### #109 Assignment 8's 100K floor has no low-resource variant

**From the Cowork session, and it moves a conclusion. Eric's call, needed before Oct 7.**

"Students have up to 200K bitext" is wrong as a general statement. The Lecture 4 and 5
assignment says medium and high-resource languages prepare **at least** 200K, while
**low-resource students prepare everything that exists for their language** — and the
course defines low-resource as under 200K available. For some of them that is far below
200K before any cleaning.

So the 104,000-pair threshold will flag those students, and the right reading is *not*
that they cannot do Assignment 8. It is that **A8's floor was written for the medium and
high-resource case and has no low-resource variant.** That is an assignment design
question rather than a data problem.

It also bounds Part C's arithmetic. "200K raw minus 29.9% equals 140K clean, clears the
floor with 36K spare" holds only for the students who started at 200K. For the rest the
subtraction starts lower and the floor may be unreachable however clean their pipeline is.

This makes the A5 audit more valuable, not less: it becomes the thing that says how many
students need a different assignment and how much smaller it has to be. Languages were
chosen in Lecture 2 and bitexts delivered Sep 23, so it is a lookup, not a forecast.

### #110 The OpenNMT evidence implies 65 to 165 epochs, not 30 to 36

The Cowork session found a config, with a caveat. The Fall 2025 *instructor* notebook
sets only `train_steps: 1000`, but a Fall 2025 **student** submission that ran the real
20,000-step assignment used `batch_type: tokens`, `batch_size: 8192`, `accum_count: 2`.

At those settings, 20,000 steps on 100K pairs is somewhere around **65 to 165 epochs**,
taking 20 to 50 subword tokens per sentence as the bracket. That is two to five times the
30-to-36 recommendation and outside the 12.8-to-33 range the briefing considered.

**Do not simply raise the number.** The 30-to-36 comes from measured convergence here:
validation loss had flattened at 36 epochs, −0.0010 per epoch over the last five. Both
can be true — OpenNMT students may have been training well past convergence, and 20,000
steps may never have been tuned. And it is one student's file, not the reference config.

But the gap is too large to split, and training budget dominated data volume by roughly
7x in this repository's own runs, so it is the parameter least safe to guess at. What
settles it: run #95 as a ladder that reports validation loss per epoch, so the flattening
point is visible rather than assumed. If loss is still falling at 36, the OpenNMT figure
is evidence rather than noise.

### #111 Does the Lecture 6 activity notebook go in the public repository?

**Eric's decision.** The Cowork session held it back rather than committing it, which was
the right call.

"CS 479 MT Evaluation Activity - Lecture 6" is already shared with students on Colab. Its
Part 4 hands them a working scoring cell — `compute_bleu` and `compute_chrf` already
written — which is Assignment 6 step 3. Eric saw that, judged it plumbing rather than the
assignment, and shipped it. The ranking and the analysis are the graded thinking and both
are untouched.

Their distinction is the one worth keeping: a Colab link shared with one cohort and a
public repository are different questions, and an answer to the second is not an answer to
the first. Three options: public as-is, public with the scoring cell replaced by a prompt,
or route to `torchlingo-private`.

If it lands, two fixes first: its install cell is the old `!pip install -q` pattern, and it
carries local `compute_chrf` and `compute_ter` wrappers written to route around the
transpose bug. Those come out when PR #58 merges.

Worth noting they also audited `CS479_COURSE_ROADMAP.md` for anything that should not be
public before it was committed — no URLs, no SharePoint links, no student names, no
credentials.

### #113 Land the six PRs still open

**No longer blocking anything.** Four of the nine landed on 2026-09-26 and **0.2.0 is
published**, verified by installing from PyPI: 11 of 11 modules import, chrF reads 78.40
rather than 100.00, and the decode budget is one number. So Monday is covered and what
remains is improvement rather than repair.

Landed: **#53** the version guard and the bump, **#58** the chrF and TER fix, **#73**
tutorial 2's install, **#81** the decode-length unification.

Also landed 2026-09-26, both admin-merged with Eric's per-PR approval: **PR #85** the four
course notebooks, and **PR #84** the dropped `Config`. PR #85 was what gated the Lecture 6
deck's link switch, so that is unblocked — the badges resolve off `main` now.

Still open, six: **PR #51** nav entries, **PR #52** tutorial 6 imports, **PR #54** the
length-normalization resolution, **PR #55** the sacreBLEU signature, **PR #57** the
causal-mask convention, **PR #59** Transformer attention.

**All six were refreshed on 2026-09-26 and are green against today's `main`.** They had
each been 27 commits behind. The refresh was free at that moment because none carried an
approval, so pushing dismissed nothing — the right time to do it, since a refresh *after*
approval costs the approval.

The staleness was not hypothetical. Before the refresh all six were green on ten checks and
**none of them had ever run `Tag matches pyproject version`**, the gate PR #53 added. They
were green against a `main` that did not have it, which is precisely the PR #17 pattern.
All six now carry the gate and pass it.

The two with real semantic overlap came out clean, which is worth recording because the
expectation was the opposite: **PR #54** touches `inference.py`, which PR #81 rewrote for
decode length, and **PR #55** touches evaluation, which PR #58 rewrote for reference shape.
Tests pass on both.

**One genuine conflict, and it was self-inflicted.** PR #51 collided with `main` on
`docs/mkdocs.yml`, because PR #85 added the Course Notebooks nav section in the same place
PR #51 adds tutorial 6. A `merge-tree` dry run *before* PR #85 merged had reported all
eight clean, so the conflict was created in between. Resolved by keeping both — tutorial 6
inside Tutorials, the Course section after it — and `mkdocs build --strict` exits 0.

Two of those now have knock-on effects worth knowing. PR #51 is what fixes **Task #84**,
which stopped being a prediction and became a live defect the moment PR #58 landed.
PR #55 is what **Task #85** and **Task #91** wait on.

*Written as "Task #84", "Task #85" deliberately: PRs #84 and #85 now exist and are
unrelated to those tasks. This paragraph used bare numbers until 2026-09-26 and was
exactly the collision the numbering rule in `CLAUDE.md` describes.*

The original cautions still hold for every one of them, because the release proved both
worth respecting: a green tick is green only against the base the checks last saw, and
each of these predates today's `main`. Refresh and let CI re-run rather than trusting an
old green. That is how all four of the merged ones were handled.

### #118 What does a paid Colab session actually provide?

**Eric's call 2026-09-26: Coulson measures it.** This is the other half of the A8 memory
question. The ladder gives demand at each length cap; he gives the ceiling. Neither half is
useful alone.

**What to ask for**, because "how much RAM does Colab have" is not the useful question:

- Which accelerator a paid session actually assigns. T4, L4 and A100 differ enormously —
  roughly 16, 24 and 40 GB — and the answer changes the recommendation.
- Device memory available to the *process*, not the instance total.
- Whether it varies between students or within a session. An assignment cannot depend on a
  lucky draw.

**What to compare it against.** Measured on the 64 GiB M-series machine, batch 64,
`d_model` 256, 3 layers, bucketing on:

| cap | device held | seconds/epoch |
|---|---|---|
| 5 tokens | 0.31 GiB | 136.2 |
| 10 tokens | 1.28 GiB | 138.1 |
| uncapped (512) | 35.80 GiB | 192.0 |

4.1x per doubling early, flattening somewhere below the uncapped figure. Rungs 20, 40 and 80
pin where, and that is what decides whether Eric's 100-token cap is comfortable or marginal.

**Caveat to carry into the comparison:** MPS unified memory is not CUDA device memory, so
these transfer as a *scaling shape* rather than absolute numbers. Confirming the shape on the
accelerator students actually get is part of why this is worth doing rather than assuming.

**This blocks any memory claim in the A8 handout.** A line was already sent to Cowork saying
the 100-token cap "keeps A8 inside a paid Colab session", and retracted in
`notes/handoff/to-cowork.md` as unsupported. What *is* safe to tell students is the ordering
of the levers: if a session dies, look at the length cap first, because batch count drives
epoch time while sequence length drives whether the run fits at all.

### #117 A PR based on `main` closed itself during an unrelated merge

**Second instance, and it breaks the stacking explanation.** On 2026-09-26 **PR #84** went
OPEN to CLOSED — never merged, no merge commit — at 17:42:16Z, one second after **PR #86**
was squash-merged at 17:42:15Z with `--admin --delete-branch`. The two are unrelated: PR #86
was notes, PR #84 was `src/torchlingo/data_processing/batching.py`.

Caught by the post-merge open-set comparison. Before: `#86 #85 #84 #59 #57 #55 #54 #52 #51`.
After: the same minus **both** #86 and #84.

Recovered fully. The remote branch and commit `2841e14` survived with all three call sites
intact, and `gh pr reopen 84` restored it with base `main`.

**Why this matters past the recovery.** `CLAUDE.md` says of the three PR-loss failures that
"every one of them is specific to a PR whose base is another PR", and lists PR #26's silent
close with cause never established. **PR #84's base was `main`.** So the silent close is not
stacking-specific, and that sentence is now contradicted by evidence. PR #26 and PR #84 are
two instances of one unexplained mechanism.

The shared signature, which is the lead worth following: both closed within one second of an
unrelated merge, and PR #86's merge passed `--delete-branch`.

- Correct the `CLAUDE.md` passage and add PR #84 to its table.
- Test whether `--delete-branch` on an admin squash-merge is implicated. If it is, stop
  passing it and delete branches as a separate step.
- Keep the open-set comparison. It cost one command and saved a pull request — which is the
  case that rule was written on, now with a second data point.

### #115 The four course notebooks, and Monday's link switch

**In review, PR #85. Deadline Mon Sep 28.** The Cowork session wrote
`lecture-03-word-embeddings`, `lecture-04-tmx-cleaning`, `lecture-05-sentence-alignment`
and `lecture-06-mt-evaluation` into `docs/docs/course/`, and ownership of that directory
moved here: changes to the notebooks are made from this side, and Cowork sends
instructions through `notes/handoff/from-cowork.md` rather than editing the tree.

Eric's decision: the repository copy becomes canonical on **Mon Sep 28** after Assignment 6
closes at 10:00, the Lecture 6 deck's link changes from the Drive URL to the Colab badge
off `main`, and the Drive copy is retired. **The badge cannot resolve until PR #85
merges**, which is what makes this dated rather than tidy.

Verified before publishing: no stored outputs, no credentials, no local paths, no student
names; badges match the real remote and tutorial 2's known-good pattern;
`mkdocs build --strict` exits 0 with all four rendering.

- **Correction on the record.** Cowork cleared Lecture 4's inline TMX as "44 lines of
  synthetic TMX, not Church material" and read it against the private repository's concern
  on that basis. It is real Church curriculum and scripture text in English–Spanish
  ("2 Nephi 31:20", "Charity never faileth", `<ph>` tags). Its conclusion about the
  assignment stubs still holds; the data characterisation did not, and Eric's go-public
  ruling had rested on it. Re-put to him with the correction and confirmed: publishes as
  is, being short publicly available scripture rather than corpus material.
- **Queued for after Monday, not before:** swap the notebook's local `compute_chrf` /
  `compute_ter` wrappers back to `torchlingo.evaluation` imports, and verify **chrF 78.40**
  on its Part 1 example. 100.00 means the reference shape is wrong again. Students are in a
  Drive copy until Monday and nothing moves underneath a live assignment.
- **Leave alone:** the Part 4 `# TODO:` cell. It pre-writes Assignment 6's step 3
  deliberately and Eric ruled it fine.
- **Never publish:** the INSTRUCTOR notebooks for Lectures 3 and 4, which carry worked
  solutions.
- Gating these is **#103**, now unblocked.

### #116 `create_dataloaders` discards the `Config` it is handed

**In review, PR #84.** It accepts a `Config` and uses it for batch size, workers, device
and padding, then builds its `NMTDataset` objects without passing it on, so the datasets
fall back to `get_default_config()` and every dataset-level field is dropped in silence.

```
config asked for max_seq_length : 5
dataset.max_length actually is  : 512
```

Cost already paid: the length ladder in **#112** produced two rungs that agreed to three
decimals because both ran at 512. Nothing raised.

Wider than `max_seq_length`: `NMTDataset` resolves `src_col`, `tgt_col`, `src_tok_col`,
`tgt_tok_col`, `max_length` and `eos_idx` from the config. A custom column name is the
likeliest to bite a student — it surfaces as a missing-column `ValueError` naming the
*default* columns, pointing them at the corpus instead of the config that was discarded.

Fixed by forwarding `config` to all three constructions, `None` included so the dataset
resolves the default itself and callers passing nothing are unaffected. Five tests,
including one that asserts on tensor shapes rather than the stored attribute, since the
attribute would pass if `max_length` were stored and never read. Suite: 711 OK, 21 skipped.

### #114 The wheel ships no data, so tutorials 4 and 5 cannot find what they load

Found 2026-09-25 while checking whether #44 was release-critical. It is not, and the
reason is the finding: the installed package contains **zero** data files. Verified
against the real 0.0.8 wheel.

So `data/example.tsv` and `data/pretrained/` exist in the repository and not in a pip
install. A student who opens tutorial 4 or 5 from its Colab badge, installs with pip, and
runs it will fail at the load, with nothing explaining why.

- **Monday is unaffected.** Tutorial 2 builds its own corpus inline; its only mention of
  `data/example.tsv` is prose. Checked.
- Tutorial 2's prose says the file "ships with the repo", which is true of the repo and
  false of the wheel a student installs. Worth rewording either way.
- Options: fetch the file over HTTP in the notebooks that need it, ship it as package
  data, or say plainly that those tutorials need a clone. The first keeps the Colab badge
  honest, which is the point of having one.
- Related to #53's finding that the published wheel was missing five modules. Same
  shape: what the repository has is not what the wheel carries, and nothing checks.

### #112 Expired Lecture 6 references, and the ladder that should have been built first

Two pieces of cleanup on my own work.

**The briefing contradicts itself.** Part A2's prose retired the Lecture 6 placement once
Lecture 6 turned out to have already run, but its table still lands two rows there, and
suggestion 3 still says to run the evaluation tutorial as Lecture 6's Colab activity.
Their better idea for that tutorial: **reading before Assignment 8** rather than a class
activity. Worth taking.

**Build the ladder rather than another single long run.** 5 tokens, then 10, 20, 40, 80,
measuring peak resident memory and seconds per epoch at each rung and stopping before the
wall instead of finding it by crashing. What the crash says it needs:

- bucketed batching and AMP on, per #107, so it measures the configuration anyone would
  actually use rather than the worst one
- peak memory per rung, not only time
- a ceiling that aborts the rung rather than the machine
- per-epoch validation loss, so #110's flattening question falls out of the same data
- every number to JSON

This replaces the single 36-epoch shot as the way to answer #95, and it is what should
have been built first.

**Progress 2026-09-26, and it found a library defect instead of a memory curve.**

`torchlingo-private/scripts/ladder.py` is written and two rungs have run. The rungs are
worthless as length measurements and valuable for a different reason.

| | rung "5" | rung "10" |
|---|---|---|
| seconds/epoch | 192.0 | 189.3 |
| batches/epoch | 1559 | 1559 |
| peak resident | 3.54 GiB | 3.40 GiB |
| device held | 35.80 GiB | 35.47 GiB |
| reclaimed by `empty_cache` | 35.73 | 35.40 |
| val loss | 7.0392, 6.3099 | 7.0405, 6.3025 |

Agreement to three decimals across every axis is not a finding about sequence length; it
is two runs of the same thing. **`create_dataloaders` was discarding the `Config` it was
handed** — see **#116** — so both rungs trained at the default 512 cap. The ladder's knob
was never connected.

**Retraction.** This file and the Cowork handoff were told that rung 5 held 35.80 GiB "at
5 tokens, where attention is negligible, so the crash was #108 rather than sequence
length". The premise was false: it was never 5 tokens. The conclusion has to be rebuilt
from what the rungs actually show, which is narrower —

- At **full length**, a 2-epoch run holds ~35.5 GiB of which **99.8% is reclaimable
  cache**. That is direct support for **#108**, and it is measured rather than argued.
- Peak resident memory is only ~3.5 GiB, so **RSS is the wrong number to guard**. The
  watchdog's first version measured it alone and would have sat quietly through the growth
  that took the machine down. It now checks resident *and* device-held and kills on the
  worse; verified firing at `device-held 5.67 GiB` while resident was 0.88.
- **Sequence length is not exonerated, because it was never varied.** Whether attention
  drives the wall is still open, and the real ladder has not been walked yet.
- Wall clock at 1559 batches is ~123 ms/batch, so **batch count dominates**, not sequence
  length. If throughput is the goal, larger batches beat shorter sequences.

**Rung 5 re-run for real, 2026-09-26, and it settles the question the other way.** PR #84
merged, so the cap now reaches the data. Same script, same seed, same corpus, the only
difference being that `max_seq_length=5` is now obeyed:

| | rung 5, real cap | the same command at the 512 cap |
|---|---|---|
| device held | **0.31 GiB** | 35.80 GiB |
| reclaimable by `empty_cache` | 0.25 GiB | 35.73 GiB |
| peak resident | 0.91 GiB | 3.54 GiB |
| seconds/epoch | 136.2 | 192.0 |
| batches/epoch | 1562 | 1559 |
| val loss | 5.5253, 4.8594 | 7.0392, 6.3099 |

**Sequence length is the driver.** A genuine 5-token cap holds **115x less** device memory
than the same run uncapped. The original pre-ladder diagnosis — attention memory quadratic
in the batch's longest sequence — was right, and the retraction recorded above was an
artifact of the broken knob rather than a correction to the physics. Restated plainly so the
record is not left pointing the wrong way.

**What this demotes.** #108 remains a real defect, but it is secondary rather than the
cause. At 5 tokens there is only 0.25 GiB to reclaim, so unreleased cache is not what
creates 35 GiB — length is. Calling `empty_cache` would have released the spike *after* it
was taken, which on unified memory is often too late. The allocation has to be bounded, not
returned.

**What it confirms.** Wall clock barely moved: 136.2 s/epoch against 192.0, a 1.4x gain for
a 100x memory reduction, at essentially the same batch count. **Batch count dominates
throughput, sequence length dominates memory.** They are separate levers and the crash was
a memory failure, so the lever that matters for #95 is the cap, while the lever for epoch
time is batch size.

Next: walk rungs 10, 20, 40 and 80, one at a time. The prior two rungs are preserved in
`data/ladder-backup/` rather than overwritten, being the invalid 512-cap pair.

Superseded note, kept because the reasoning above depends on it: the rungs completed so
far have to be discarded rather than reused, since they are all the same configuration.

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

**#2 Batch beam search across sentences** — **DESCOPED 2026-09-22**, see the decision
above. Notes kept because they are the expensive part to rediscover.

Remove the batch-size-1 restriction in `inference_fast.py`; flatten to `(batch x k, t)`.
Validate it with `python scripts/bench_decode.py`: the sentence lever should show up as a
further drop in `decode()` calls with positions forwarded unchanged, and
`tests/test_bench_decode.py` will fail until the committed numbers are regenerated, which
is the intended prompt to update the docs.
**This is where the remaining ~8x lives** (the sentence axis above) — and it scales with
the number of sentences decoded, so it matters more on a real test set than #1 does.
- Files: `src/torchlingo/inference_fast.py` (the raise, and the per-row loop in
  `inference_fast.translate_batch`)
- Hard part is bookkeeping for ragged completion — sentences finishing at different steps.
- Needs a contract adapter: it takes a batch rather than one sentence, so it does not slot
  into the current `BEAM_DECODE` shape unchanged.
- `tests/test_training_inference.py:502`
  (`test_beam_search_decode_raises_on_batch_size_gt_one`) stays valid: under the
  side-by-side design the *reference* implementation keeps that restriction. The batched
  variant gets its own tests rather than inverting this one.

**#3 Incremental decoding / KV cache** — **DESCOPED 2026-09-22**, see the decision above.

Removes the O(L^2) prefix recomputation. Independent of the two axes above: it reduces the
work *inside* each call rather than the number of calls.
- ~~DECISION NEEDED: may compromise the readability that makes this repo worth using for
  teaching. Consider stopping at #2 for an educational library.~~ **Resolved by the
  side-by-side design above:** the reference implementation stays readable regardless, so
  the fast path is free to be dense. Worth doing.
- `scripts/bench_decode.py` is the right instrument, but note it measures the wrong axis
  for this one: a KV cache leaves the **call count unchanged** and cuts *positions
  forwarded* instead. Read that column, not the call column, or the harness will make a
  real improvement look like no change at all.

**#4 Resolve length-normalization semantics** — in review as PR #54
`inference.py:203` applies length normalization during *pruning*, not only at final
selection — comparing normalized scores across different lengths mid-search. Defensible
but non-standard. Preserve exactly during #1/#2 so perf work stays reviewable; raise as
a separate question.

**Now has evidence, from #40.** Measured on the tutorial 5 model across five held-out
subsets, paired:

- `alpha=0.6`, the shipped default, is **indistinguishable from `alpha=0.0`**
  (−0.07 ± 0.03 BLEU). It is not doing the job it exists for.
- `alpha=1.0` is a real if small gain, +0.25 ± 0.06.
- The bias it targets is plainly present: mean output length falls monotonically with
  beam width, 12.61 tokens at greedy to 9.73 at beam 10, against references averaging
  11.62.

So the question is no longer whether the semantics are defensible in the abstract. It is
why a correction that measurably does nothing is on by default. Two candidate answers,
and the evidence does not distinguish them: the default is too weak, or normalizing
during pruning blunts it. Sweeping `alpha` with normalization applied only at final
selection would separate the two, and that is now a cheap experiment because
`scripts/sweep_decoding.py` exists.

## Code — other gaps

**#6 Multi-GPU training via DDP** — **DESCOPED 2026-09-22**, see the decision under
*Code — decoding performance*. The weakest of the three for a course and the highest
ongoing maintenance: the lab's students train on laptops and Colab, where there is one
GPU or none.

Not implemented. `config.py:663` states multi-GPU "requires custom DataParallel setup."
That sentence is now the honest final answer rather than a placeholder, and should be
left in place.

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
Task #70's subject says "with every score", but PR #55 attaches `.signature` to
`compute_bleu`'s result alone. Measured on the combined main + PR #55 + PR #58 tree:

```
BLEU  nrefs:1|case:mixed|eff:no|tok:13a|smooth:exp|version:2.6.0
chrF  MISSING
TER   MISSING
```

chrF is the case that proves this is not cosmetic. `compute_chrf` defaults to
`word_order=2` (chrF++) while sacreBLEU's `corpus_chrf` defaults to `0`. That one
undeclared parameter caused two separate confusions here: a correct implementation
judged broken by 0.45 points against the wrong baseline, and
`notes/EVAL_CHRF_TER_TRANSPOSE_BUG.md` stating the right value (78.4) beside a snippet
returning 76.97. A chrF signature declares `nw:2` and prevents both, so chrF needs one
more than BLEU does.
- Call `get_signature()` on CHRF and TER as `compute_bleu` already does, and widen
  `tests/test_bleu_signature.py` to all three. Cheap once #55 and #58 are both in.

**#86 `evaluate_model` has no test, and it is the function callers use**
Coverage on the combined tree, after PR #58 adds the first evaluation tests this repo
has ever had:

| Function | Tested by |
|---|---|
| `compute_bleu` | `test_metric_reference_shape`, `test_bleu_signature` |
| `compute_chrf`, `compute_ter`, `_as_reference_streams` | `test_metric_reference_shape` |
| `evaluate_model` | **nothing** |
| `save_translations` | **nothing** |

PR #58 tests the three wrappers well, but `evaluate_model` is what
`examples/evaluate.py`, `examples/train.py` and any student actually call, and it is
where the transpose bug did its damage: `compute_chrf_score: bool = True` is the
default, so every call reported a wrong chrF. Fixing the wrappers without testing the
aggregator leaves the same hole one level up, where the thing under test is not the
thing being used.
- Run `evaluate_model` end to end on a tiny fixture and assert its returned bleu/chrf/ter
  match the three wrappers called directly, under default flags and with
  `compute_ter_score=True`. Cover `save_translations` too.

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

## The training-budget finding, and what it invalidated

Raised by Coulson on PR #34: "the BLEU scores are extremely low ... worth looking into
if it wasn't flagged before." The scores were expected and documented. Looking into them
anyway found a measurement error of mine.

**The root cause (#54).** `train_model` appended to `val_losses` in two places: the
periodic step-triggered validation from `config.val_interval`, and the epoch-end
validation. One list, two different measurements, and a docstring promising "per epoch".

```
              train_losses   val_losses   true epochs
  baseline         20            36            20
  new              36            72            36
```

I read `len(val_losses) == 36` off the baseline, concluded 36 epochs, and passed
`--epochs 36` to "match" it. The baseline had run **20**.

**What that did to #49.** The comparison gave one model 19% more data *and* 80% more
training, while its writeup claimed data was the only difference. Re-running with epochs
actually matched:

| data | epochs | BLEU | |
|---|---|---|---|
| 53,520 pairs | 20 | 4.96 | baseline as shipped |
| 53,520 pairs | 36 | **7.01** | control: same data, more epochs |
| 64,311 pairs | 36 | 7.32 | more data *and* more epochs |

- epochs 20 → 36, data fixed: **+2.05 BLEU**
- +20% data, epochs fixed: **+0.29 ± 0.22, 95% CI [−0.16, +0.71]**

The data effect's interval crosses zero. Training budget mattered roughly **7x** more
than the recovered data, and the recovered data bought nothing measurable.

So the published claim was wrong twice: ~88% of the +2.33 was training length, and the
residual is not significant. The diagnosis in tutorial 5 — "data-starved" — is also
wrong; the model was **undertrained**, which has a different fix.

**#55 Correct tutorial 5.** It currently teaches "more data" as the top lever, measured.
The honest version is the better lesson: the interesting hypothesis was wrong, the boring
one (you stopped training too early) was right, and only controlling the variable told
them apart. Numbers to use are in the table above; artifacts in
`docs/docs/_generated/checkpoint_comparison.json`, which also needs regenerating from the
controlled run.

**#56 Correct #34.** Its description and commit message both claim "same architecture,
same 36 epochs, same seed — the only thing that changed is the data." False. The
checkpoint itself is fine and worth shipping; only the explanation of why it is better
needs replacing.

**#57 Add a note to #27.** It claims "They are 18% more data", which is true, and makes
no BLEU claim, so nothing there is wrong. But the implicit case for the work is quality,
and the measured quality effect is indistinguishable from zero at this scale. Worth
saying so plainly, and restating the real justification: it is a correctness fix for data
being discarded for a fixable reason, it teaches Gale-Church, and it will matter at a
scale where the model is not the binding constraint.

**#58 The script's default undertrains.** `train_example_model.py` defaults to
`--epochs 20`, which is what produced the BLEU 4.96 checkpoint. 36 epochs gives 7.01 on
the same data, and by then validation loss has flattened (mean change over the last five
epochs: −0.0010/epoch). A default that stops well short of convergence teaches the wrong
thing about training, and it is what made "more data" look like the answer. Change the
default to 36, or add early stopping on the validation curve so the run ends when it
should rather than when a hardcoded count runs out.

**#59 Nothing checks that a comparison controlled its variables.** This is the general
version, and the reason the error survived review. `compare_checkpoints.py` pins the test
set and bootstraps the difference, which is why the *measurement* was sound; it never
looks at how the two checkpoints were trained. It has both checkpoint dicts in hand and
could refuse, or at least warn loudly, when `len(train_losses)`, `train_pairs`,
`model_config` or the seed differ — printing what differs alongside the BLEU delta so a
reader sees the confound next to the number. Same shape as every other finding on this
list: two things that must agree, with nothing checking they do.

**#60 Nobody is told when main goes red**

main broke on 2026-09-19 and stayed broken until someone happened to look.

The break itself is instructive: **no PR could have caught it.** #25 introduced
`preprocessing/alignment.py` carrying a docstring example that asserts
`looks_aligned()` on a single-row frame, which cannot pass. #32 added the
`--doctest-modules` gate that runs it. #32 could not have fixed the example,
because it branched from a main where the file did not exist yet. Both were
green on their own branches; the *combination* fails.

That is a merge-order interaction, and the only place it can surface is a
post-merge run on main. Which means the post-merge run is load-bearing and
currently nobody watches it:

- A PR's checks run against the *merge result*, so #39 was green while main was
  red. Green on your PR says nothing about the branch you are merging into.
- Nothing notifies on a failed push-to-main run. It sits in the Actions tab.
- The next person to open a PR inherits a red main and may reasonably assume
  their branch caused it.

Cheapest fix that would have caught this: notify on failure of the `push` to
main run — a GitHub Actions step on `if: failure()` posting to the Discord
channel the lab already uses, or simply enabling GitHub's own "Actions failure"
email for the repo. Neither needs new infrastructure.

Worth pairing with #51 and #53, which are the same family: a check that reports
but does not block, a check that runs a fifth of what it claims, and a check
nobody reads. Each is individually defensible and together they mean a green
tick carries less than it appears to.

## Tests

**#15 Existing `DummyTransformer` is history- and memory-blind**
`tests/test_training_inference.py:16` computes logits from a zero tensor, so its output
depends only on decoding *position*. Verified: logits are byte-identical for different
decoder histories AND for different encoder memories. The three existing beam tests
therefore cannot detect scrambled beam state or bad memory expansion.
- Mitigated by the new `tests/test_decoding_equivalence.py`, but the old tests should
  eventually migrate to the history-sensitive fixture rather than sitting alongside it.

**#79 `test_different_optimizers_produce_different_results` is order-dependent**
Filed after it failed under `python -m unittest discover tests`, the runner `CLAUDE.md`
documents, while CI's `pytest tests/` was green. **Re-checked on main 2026-09-23: it now
passes under `unittest discover`, 4 runs out of 4** (693 tests, 21 skipped), so the
failure is intermittent rather than a standing red suite. That is worse to leave alone,
not better — a test that depends on whatever seeded the RNG before it will come back.
- The assertion is the weak part: it checks that Adam or SGD *improved* over 2 epochs on
  a tiny fixture. When the two optimizers produced byte-identical curves
  (`Epoch 1/2 | Train: 2.1852` / `Epoch 2/2 | Train: 2.1852`) the real claim — that the
  two optimizers take *different* trajectories — was the thing that had failed.
- Seed inside the test, and assert the difference rather than the improvement.
- Worth sweeping for other order-dependent tests while in there: run the suite under
  both runners and diff. The two documented ways to run the suite disagreeing, with
  nothing checking that they agree, is the same shape as the tag-vs-`pyproject` defect.

---

## Release

**#38 Colab checkpointing has never been run in Colab**
PR #17 adds `training_checkpoint.py` with `is_colab()`, `mount_drive()` and a Drive-backed
default directory. None of it has ever executed in Colab. CI cannot cover it: GitHub
runners have no Drive to mount. Josh said the same of his original in PR #1, so this code
path has now been **written twice and run zero times**.

Asked Coulson on PR #17 to try it. What needs checking:
1. `mount_drive()` actually mounts, and `default_checkpoint_dir` lands under `MyDrive`
   rather than the runtime's own disk — a checkpoint on runtime disk dies with the
   runtime, defeating the purpose.
2. `latest.pt` and `best.pt` appear in Drive; the startup free-space line is sane.
3. Interrupt the runtime partway, re-run the same cell: it should print a resume line and
   train only the remaining epochs.

Item 3 is the one that matters. If it restarts from epoch 0 the feature does not work,
whatever the unit tests say. **Item 3 is now #93**, promoted out of here because CS 479
gave it a deadline; items 1 and 2 stay in this task.
- Open: whether to gate the #17 merge on this, or merge with the limitation documented,
  which it currently is in both the module docstring and the reference page.
- **Coulson accepted on 2026-09-16:** "I will return to this to review and test in Colab
  when I finish the other PRs." He has since reviewed everything else, so #17 is next in
  his queue and this is the one open item with a named owner.

**#36 CI actions are pinned to a deprecated Node runtime**
Every run now warns: `actions/checkout@v4`, `actions/setup-python@v5` and
`actions/download-artifact@v4` target Node 20, which GitHub deprecated, and are being
forced onto Node 24. It is a warning today and a hard failure whenever GitHub drops the
shim. Bump the action versions. Unrelated to anything in flight, and cheap.

**#44 Gate the sdist on "no Git LFS pointer shipped"**

`data/example.tsv` and `data/pretrained/model.pt` moved to Git LFS, and CI checks out
without LFS on purpose to keep runs light. That combination has a sharp edge: a build
that packages an LFS-tracked file in a no-LFS checkout ships 130 bytes of pointer text
under the name of a 17 MB corpus, with nothing in the build complaining. It would reach
PyPI looking fine and open as garbage.

`MANIFEST.in` now excludes both explicitly, so this is closed *by construction* rather
than *by check*. Two things that must agree with nothing checking they do, again:
a future `recursive-include` would reopen it silently.

- Add a release-job step that scans the built sdist and wheel for any member beginning
  `version https://git-lfs` and fails on a hit.
- Cheap, no LFS dependency, catches the whole class rather than today's two files.
- **Match on the first line, not anywhere in the file.** A `grep -rl` for the string
  flagged `tests/test_data_integrity.py` and `tests/test_sentencepiece.py`, which contain
  it as the literal the skip logic compares against. The detector would fail the build on
  the detector. Compare `head -c 23` instead.

Verified once by hand against the CI-built artifacts from run 34992013848, which is the
real case: a no-LFS checkout. The sdist carries only the small multilingual examples and
the wheel carries no data at all.

Related, worth watching rather than acting on: LFS storage and bandwidth come out of the
org's quota. Two files at ~28 MB is nothing, but every clone by every student fetches
them. If a course section of 60 blows through the free tier, the fallback is to host the
corpus outside git and download it on first use.

## Inference gaps

**#34 Surface attention weights from greedy and beam decoding**
Raised by Coulson on PR #10: can we visualize alignments for beam search too?

Not today. Weights come only from a teacher-forced `model(src, tgt, return_attention=True)`,
which aligns a translation you already have. Both decoders compute weights and throw them
away — `inference.py:237` in greedy, and the LSTM beam path added in #12. So you can plot
the alignment of a *reference* translation but not of one the model generated, which is
the more interesting picture.
- Greedy is straightforward: accumulate the per-step weights.
- Beam is not. Weights belong to a hypothesis and hypotheses get pruned, so either carry
  per-beam weight history and filter to the winner, or re-run `decode_prefix` on the
  winning sequence once the search finishes. The second is cheaper and matches how the
  reference already re-scores prefixes.
- Shape it as an opt-in `return_attention=False` on both decoders so the default return
  type does not move — #9 has just standardized those, along with the contract tests.
**Correction (2026-09-18): this is bigger than recorded, and the "cheap alternative"
above does not exist.** I claimed decode-then-teacher-force already yields the alignment,
making this ergonomics rather than a missing feature. That is true of the LSTM only:

```
SimpleSeq2SeqLSTM.forward(src, tgt, return_attention=False)   <- exists
SimpleTransformer.forward(src, tgt, ...)                      <- no such parameter
```

`SimpleTransformer` has **no attention-returning path at all**. There is nothing to
teacher-force into and no recipe to document. And the Transformer is the architecture the
tutorials train, the pretrained checkpoint uses, and a student is most likely to reach
for, so the gap is in the worse place.

Getting cross-attention out of `nn.Transformer` is also not a one-liner. PyTorch hardcodes
`need_weights=False` inside `TransformerDecoderLayer._mha_block`, so a forward hook on
`multihead_attn` captures `(output, None)`. The options are to wrap each layer's
`multihead_attn.forward` to force `need_weights=True` while capturing, or to subclass the
decoder layer and override `_mha_block`. The wrapper is less invasive and can be a context
manager, which also keeps the cost off the default path.

Revised shape:

1. A way to capture Transformer cross-attention at all. This is the real work and
   everything else depends on it.
2. Then the original item: surface it from greedy and beam decoding rather than only from
   a teacher-forced pass.

Worth doing because it is also a good lesson. Explaining *why* the weights are not simply
available — a fused fast path that discards them unless asked — teaches something true
about how these libraries are built.

Knock-on for #48: the audit lists "attention appears three times" under redundancy. The
sharper problem is that it appears three times for the **LSTM** and zero times for the
Transformer. Tutorial 4 teaches attention on an LSTM trained on a synthetic reversal task;
a student who moves to the Transformer cannot inspect attention on the model they are
using.

## Lint and tooling gaps

**Standing gotcha from #45, which is now fixed in #21**

A PR opened *before* #21 merged still shows no checks, because for `pull_request` events
GitHub reads the workflow from the PR branch rather than from main. Rebase such a PR onto
current main, or dispatch a run with
`gh workflow run tests_and_build.yml --ref <branch>`. PRs opened since #21 get checks
automatically, whatever branch they target.


**#22 `examples/` is outside the lint gate**
CLAUDE.md and CI lint `src` and `tests` only. Running `ruff check examples` turns up 32
pre-existing errors across `train.py`, `evaluate.py`, `inference_ceb_cmn.py`,
`train_ceb_cmn_simple.py` and `multilingual_training_example.py` — unsorted imports,
unused imports and variables, `f`-strings with no placeholders, deprecated `typing.List`,
a blind `except Exception`.
- These are *examples*, i.e. the code students are most likely to copy, so they are
  arguably the worst place in the repo to let style rot.
- Not fixed here: it is unrelated to attention and a 5-file mechanical diff would bury
  the review, exactly the reasoning applied to the ruff split in #14.
- Suggested: fix under its own PR, then add `examples` to the lint scope so it stays
  fixed.
- **`scripts/` has the same gap**, found the same way: `generate_sentencepiece_models.py`
  trips EXE001 (shebang, not executable) and BLE001 (blind `except Exception`). Widen the
  scope to `src tests examples scripts` in one go.
- The gap keeps widening. `scripts/` has gained `bench_decode.py`, `execute_notebooks.py`,
  `train_example_model.py`, `sweep_decoding.py` and `diagnose_corpus.py`, all linted by
  hand on the way in and none of them gated. Hand-linting is exactly the thing that stops
  happening once whoever is doing it moves on.

---

## Course material

**#42 Lecture 7 assignment** — *routed to the Cowork session 2026-09-26*
Captured so it is not lost. Never startable from this side, for the reason below.

**Eric's call 2026-09-26: this is Cowork's to scope**, and if it needs a notebook they write
one into `docs/docs/course/` as they did the other four. Requested in
`notes/handoff/to-cowork.md`.

Much of the original unknown has since been answered by the pivot: Lecture 7 is **Mon Sep
28**, in CS 479, and is two lectures in one — the paper-review assignment plus
neural-network foundations — with **tutorial 2 as the in-class activity**, already merged.
What remains unknown is the only thing that matters here: what students submit.

**The likely answer is that this task closes with nothing built.** A paper review needs no
supporting material, and the activity already exists. That was said plainly in the request
rather than left open, because the alternative is inventing a deliverable to fill a
placeholder. Closes on Cowork's word either way.

**The assignment itself lives in the LMS, not in this repo.** So the work here is whatever
*supporting material* the assignment needs — a starter notebook, a script with gaps to
fill, a dataset slice — not the assignment text. That also means the deliverable may be
small or may be nothing at all, depending on what the assignment asks students to do.

`contributing.md` previously documented an `assignments/` directory that never existed.
Corrected when this was filed, and the page now says where assignments actually live.

Material an assignment could build on, all now on `main`:
- Tutorial 4 ends with an ablation and a measurable alignment accuracy, which is already
  close to an assignment shape.
- `scripts/bench_decode.py` measures decode call counts against wall clock; the original
  #12 entry flagged this as "useful as a student exercise in its own right", and the gap
  between the two numbers is a real lesson.
- `examples/attention_alignment.py` runs the same comparison at larger scale.
- The decoding contract tests demonstrate specification-by-test, if the assignment is
  about correctness rather than modelling.

## Visualization

All three raised by Coulson on Discord, 2026-09-14, after reviewing the open PRs:

> "if we can add visualization to any of the options that we present it could be useful
> for the students. I saw that it was added for attention, but did we add it for beam
> search as well? ... students should have learned about this in 312 ... but I think a
> reminder in this tool may be useful."

**The answer to his direct question was no.** That is now fixed: `visualization.py` gained
`format_beam_search` and `plot_beam_search`, and `beam_search_decode` takes an optional
trace recording candidates *before* pruning. The two below are what remains.

**#40 Visualize the effect of decoding options**
Distinct from #39: that shows how the search works on one run, this shows what the knobs
do across runs — `beam_size`, `alpha`, greedy versus beam.

Tutorial 3 already sweeps beam sizes 1, 2, 3, 5, 10 and prints a table where every row is
identical, because the toy model is decisive. That teaches nothing. On a model where the
answers differ, the sweep would show the diminishing returns past beam 3-5 that the docs
currently **assert in prose without evidence** — the same gap #12 closed for the
performance numbers.

Also worth showing `alpha`: its effect on output length is easy to demonstrate and hard to
intuit from the formula, and it connects to the open question in #4.

Smaller than #39; can reuse `scripts/bench_decode.py`'s structure for sweeping and
emitting JSON.

**#41 Connect beam search back to prior coursework**
The point stands regardless of which course number is right: students have likely met beam
search as a general search algorithm before meeting it as a decoder. The docs teach it
from scratch in NMT terms and never connect it to what they already know. A short framing
— best-first search with a fixed-width frontier, where the heuristic is the model's log
probability and pruning is what makes it tractable — lets them transfer understanding
instead of rebuilding it.

Cheap: a note box in `concepts/decoding.md` and a line in the tutorial. No code.

- **Open question for Coulson, not for us to settle:** which course. He says 312 and flags
  his own uncertainty; the answer recorded when he asked a related question on PR #8 was
  that the walkthrough sits in CS 479, with whether it should be taught earlier left open.
  Reference the concept rather than a number until that is confirmed.

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

**#84 The two evaluation pages ARE off-nav, as of 2026-09-26** — no longer a prediction
PR #58 merged, and it did not touch `docs/mkdocs.yml`. So `concepts/evaluation.md` and
`reference/evaluation.md` are now on `main` and unreachable from the site. The thing this
task predicted has happened.

`mkdocs build --strict` still exits 0, because a page missing from the nav is an INFO
rather than a warning — which is exactly how the three pages in #63 went unreachable, and
is the second time the same trap has closed.

- **PR #51 is the fix**, since it is the one that edits the nav. Landing it closes this.
- The nav entries also exist on the local `docs/evaluation-tutorial` branch, which had to
  edit `mkdocs.yml` anyway to place tutorial 7. Either route works; #51 is nearer.
- Two occurrences is the argument for #89, which makes the check explicit instead of
  relying on someone noticing a third time.

**#88 Open the tutorial 7 PR** — unblocked 2026-09-26, PR #58 has merged

The evaluation tutorial is written, executed and verified on the **local** branch
`docs/evaluation-tutorial` (commit `1516aa0`). Not pushed: it depends on unmerged work,
and the standing rule is to hold such work locally rather than stack.

Depends on PR #58 (the chrF/TER transpose fix, and `concepts/evaluation.md`) and PR #55
(the `.signature` the notebook prints). The branch is `main` with both merged in.

At PR time:

1. Rebase onto a `main` that carries #58 and #55.
2. Re-check `docs/mkdocs.yml`. The branch adds three nav entries: tutorial 7,
   `concepts/evaluation.md` and `reference/evaluation.md`. PR #51 also edits the
   tutorials nav block to add tutorial 6, so expect a small textual conflict there.
   **Tutorial 7 belongs after tutorial 6.**
3. Re-execute the notebook and re-run every check.
4. Say in the body that it closes #84.

Already verified on the branch: 7/7 notebooks execute cleanly, zero errors, zero
warnings and zero local paths in the committed outputs, every number quoted in prose
appears in an output, `mkdocs --strict` EXIT=0, ruff clean, 693 tests pass. In CI the
gate would run **3 of 7** rather than 2 of 6, because the notebook needs no LFS artifact.

**#89 Fail the docs build when a page is off-nav**

`mkdocs` reports pages missing from the nav at **INFO**, so `--strict` exits 0 while they
sit unreachable. Measured on the tutorial 7 branch: `--strict` EXIT=0 with **11** pages
off-nav.

The trap has now caught five pages across two PRs (#63's three, #84's two), and the
second was noticed only because someone happened to be auditing the first. That is not a
process that catches the third.

- Check that every page under `docs/docs/` is either in the nav or on a declared
  exception list, and fail when one is neither.
- **The exception list is the whole design problem.** `_generated/*.md` are *supposed*
  to be off-nav: they are snippet files pulled into other pages with `--8<--` includes,
  not standalone pages. So the orphan list is not a to-do list, and a naive check would
  cry wolf on five files immediately. Declare those deliberately.
- `MULTILINGUAL_ANALYSIS.md`, `MULTILINGUAL_QUICKSTART.md` and `TESTING_GUIDE.md` are
  genuinely orphaned and predate all of this; decide whether they are nav pages or
  should move out of the docs tree.

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

**#49 The pretrained checkpoint predates the enlarged corpus**

`data/pretrained/model.pt` was trained on 73,082 pairs. After #29 the corpus holds
86,430, so the shipped model has never seen 18% of the data it is meant to represent.

Nothing is broken: the model's held-out talks are still whole talks, still held out, and
still present in the corpus, because #29 is a strict superset. It is stale rather than
wrong.

Worth retraining because everything downstream reads off this one checkpoint. Tutorial 5
shows its translations, #40 measures decoding options on it, and its BLEU of roughly 5 is
the number a student meets first.

- Rerun `scripts/train_example_model.py`; about 20 epochs.
- Regenerate `docs/docs/_generated/decoding_sweep.json` afterwards, since #40's numbers
  are measured on this checkpoint. `--rerender` is not enough; the sweep itself must
  re-run, which takes about an hour on CPU.
- Check whether BLEU actually moves. 18% more data on a small model may buy very little,
  and that is worth knowing either way. If it does not move, say so in tutorial 5 rather
  than quietly retraining.

**#50 Tutorial 3 still teaches the wrong lesson about beam size**

#40 put the real measurement in `concepts/decoding.md`, but tutorial 3 is untouched: it
still sweeps `beam_size` over 1, 2, 3, 5, 10 and prints a table where every row is
identical, because its toy model is decisive. A student runs it, sees no difference, and
draws the obvious and wrong conclusion.

The cell is not wrong to exist — a sweep is the right thing to show. It just has nothing
to show on that model.

- Minimum: say so in the notebook. "Every row is identical because this model is too
  small to be uncertain; see the measured version on a real model" costs two sentences
  and removes the misconception.
- Better: have the cell assert the rows are identical and explain why, so it becomes a
  deliberate demonstration of when a sweep tells you nothing.
- Tutorial 5 is where a real sweep belongs, since it has the model for it.

**#53 The notebook gate is weaker than its green check implies**

CI checks out without Git LFS on purpose, so `data/example.tsv` is a pointer there and
`scripts/execute_notebooks.py` skips tutorials 2 through 5. **The job passes having run
exactly two notebooks out of six** — tutorial 1 and tutorial 6, the only two that need no
LFS artifact — and the check mark on the PR looks the same either way.
- The skip list only grows. PR #59 gives tutorial 4 a second requirement
  (`data/pretrained/model.pt`) so it can show Transformer cross-attention; tutorial 4 was
  already skipped for the corpus, so the CI count is unchanged, but the gap between "what
  CI proves" and "what a reader runs" widens with every tutorial that touches real data.
- It can go the other way. Tutorial 7 (#88) uses fixed strings rather than a model, so it
  needs no LFS artifact and **runs in CI**, taking the gate to 3 of 7. Worth noting as a
  design lever: a tutorial whose subject does not require a trained model should not
  acquire one, because that is the difference between a lesson CI protects and one it
  cannot see.

That was the accepted trade when LFS went in, and it is still the right one. The problem
is that nothing says so at the point where someone reads the green check.

Found concretely in #50. That change adds an assertion inside tutorial 3, whose whole
purpose is to fire when the model stops being decisive and the surrounding explanation
stops being true. It cannot fire in CI, because tutorial 3 does not run there. It was
verified by running the gate locally, twice, which is not a thing that keeps happening on
its own.

Options, roughly in order of cost:

- **Say it in the check.** The job already prints "2/2 notebooks executed cleanly" and
  lists what it skipped. Make the job summary carry that so it is visible on the PR
  without opening the log. Cheapest, and removes the false impression.
- **Fetch LFS for the notebook job only.** One `lfs: true` on one job, roughly 28 MB per
  run. This is the option the keep-CI-light decision ruled out, but the reasoning there
  was about every job on a four-version matrix, not about one job that is the only place
  tutorials execute at all. Worth revisiting on those narrower terms.
- **A scheduled full run.** Nightly or weekly with LFS, so drift is caught within a day
  without touching per-PR cost.

Related: #51, which is the same shape from the other direction. That gate runs and does
not block; this one blocks and does not run.

PR #44 was partial relief: tutorial 6 needs no LFS artifact, so it does run in CI, taking
the gate from 1 of 5 to the 2 of 6 above. The false impression is unchanged — the green
check still does not say what it skipped.

**#63 Three pages have no mkdocs nav entry** — in review as PR #51

`docs/mkdocs.yml` belonged to a PR we were not stacking on, so three pages shipped
without a nav entry: tutorial 6, `reference/diagnostics.md`, and `related-work.md`. A
page absent from nav is INFO rather than a warning under `--strict` — verified on each
branch, the build stays clean — so none of this blocks. But each page is reachable only
by direct link until PR #51 lands. PR #51 also restores the `capture_cross_attention`
link that could not resolve without the nav entry; see #74.
- The same trap has already caught the next pair of pages. See #84.

| Page | From | Place it |
|---|---|---|
| `tutorials/06-diagnosing-failures.ipynb` | PR #44 | Tutorials, after `05-real-translations.ipynb` |
| `reference/diagnostics.md` | PR #45 | API Reference, after `config.md` |
| `related-work.md` | PR #46 | Top level, near Home |

One more thing to undo at the same time: `related-work.md` refers to
`torchlingo.diagnostics` as plain code text rather than linking to
`reference/diagnostics.md`, because linking a page that does not exist on `main` fails
`--strict`. Once PR #45 has merged, make it a link.

**#65 Tutorial 6 and `torchlingo.diagnostics` are two copies of the same checks** — in review as PR #52

PR #44 defines the five checks inline in the notebook; PR #45 ships them as a module.
Until one sources from the other they can drift, and the notebook is the copy a student
reads.

- Only after **both** have merged — doing it in either PR would stack it on the other.
- Keep the student seeing the logic; the pedagogy depends on it. Import the functions and
  show the source (`inspect.getsource`), or keep a short annotated call, rather than
  silently calling a black box.

**#66 Adopt `nltk.translate.gale_church`; split #29 into two different jobs**

#29 conflates two goals that want different tools, and proposes hand-writing an algorithm
that is already a dependency away.

`nltk.translate.gale_church.align_blocks()` ships with exactly the priors #29 specifies:
(1,1)=0.89, (1,2)=(2,1)=0.089, (2,2)=0.011, (0,1)=(1,0)=0.0099, and
`VARIANCE_CHARACTERS=6.8`.

- **To recover the 98 talks (~13k pairs)** — use Vecalign or Bertalign. Embedding-based
  aligners measurably outperform length-based ones; an English–Slovak evaluation
  (*Scientific Reports*, 2023) puts Vecalign and Bertalign significantly ahead, with
  hunalign and Bleualign behind. Gale-Church is the wrong tool for the production job.
- **To teach alignment** — implement it, because the implementation *is* the lesson, but
  pin the output against NLTK's as a test oracle rather than shipping ours as the only
  word on it.

Same split applies to #52 (Moore 2002): still a good lesson, superseded in practice.

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

**#70 Print the sacreBLEU signature with every score** — in review as PR #55

Adopted from the Joey NMT baseline run, which logs
`nrefs:1|case:mixed|eff:no|tok:13a|smooth:exp|version:2.6.0` next to every BLEU.

We already use sacreBLEU, so the signature is available and we are throwing it away. A
BLEU number without it is not reproducible by someone who was not there — which is
precisely the discipline tutorial 6 and #59 are trying to teach. Currently we teach it
and do not practise it.

- `compute_bleu` returns a `sacrebleu.metrics.BLEU`; surface `.get_signature()`.
- Print it wherever a score is reported: `scripts/compare_checkpoints.py`, tutorial 5,
  `concepts/decoding.md`, and the generated `decoding_sweep.json`.

**#71 Decide whether to report the Joey NMT breakage upstream**

Their shipped quickstart does not run on current PyTorch: `joeynmt/builders.py` passes
`verbose=False` to `torch.optim.lr_scheduler.ReduceLROnPlateau`, which PyTorch removed,
so `scheduling: "plateau"` raises `TypeError` before the first step. The toy config also
sets `use_cuda: True` and `fp16: True`, which fail on any CPU-only machine.

Found while running their tutorial as a baseline (2026-09-20). Not reported — filing an
upstream issue is outward-facing and is Eric's call.

- One-line fix upstream; a courteous thing to send given we cite them favourably.
- If yes: report from a clean clone, not the patched scratch copy.

**#72 Prune the prose entries for retired tasks**

The Status table was reconciled against the merge history on 2026-09-23. The prose
entries further down mostly remain, so the file still describes shipped work as though it
were pending. Anything whose row is gone from the table should be gone from the body too.

- Read each before deleting. Some carry findings worth keeping even though the task is
  finished — the recurring "two things that must agree, with nothing checking they do"
  observation is one, and it has now described four separate defects. Move those into the
  place they apply rather than losing them with the task.
- Not a `sed`. The table is the index people read and it is correct now; the bodies need
  a careful pass.


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

Raised by Eric: Moore improved on Gale and Church for bitext alignment.
[Moore (2002)](https://aclanthology.org/2002.amta-papers.14/) aligns in two passes — a
length-based pass like the one #29 ships, whose confident pairs train an IBM Model 1
word-translation model, then a second pass scoring length *and* word correspondence, with
the search confined to segments the first pass found plausible.

What it would buy here, honestly: not much, and that is why #29 shipped length alone.
Gale-Church recovered 13,152 of a possible 13,305 pairs at a quality indistinguishable
from the talks that never needed repair. The 153 it gave up are the ceiling.

Where it would matter:

- The limitation pinned by `test_a_long_dropped_sentence_is_handled_worse`. Length treats
  a long deletion as so improbable that a poor one-to-one scores better; a dropped
  sentence shares no *words* with anything, which is precisely what lexical evidence
  sees.
- The 89 talks present in only one language stream. Length cannot help there and neither
  can Moore, so those are gone regardless.
- Any future corpus noisier than this one. The method is the transferable part.

Also a genuinely good teaching progression if #48 wants one: length alone, then why it
fails, then lexical evidence. Cited in `concepts/data-pipeline.md` and in the module
already, so the pointer exists whether or not the code follows.

**#47 Docstring examples are not executed, and 30 of them fail**

`pytest --doctest-modules src/torchlingo` reports **30 failed, 24 passed**. Nothing runs
doctests, so CLAUDE.md's "keep examples runnable and concise" is unenforced and the
examples students are most likely to copy have rotted.

Found the usual way: an example written in #46 asserted `looks_aligned()` on a
single-row frame, which cannot pass, because one row has no length variation to
correlate. It was wrong the day it was written and nothing noticed.

Failures span `preprocessing/base.py`, `multilingual.py`, `multilingual_helpers.py`,
`sentencepiece.py` and `training.py`, among others. Only `preprocessing/alignment.py` is
fixed so far, in #29.

- Fix in batches by module, since the failures are unrelated to each other.
- Then add `--doctest-modules` to the test job so it stays fixed. Fixing without gating
  just resets the clock, the same lesson as #26.
- Same shape as #14 and #16: two things that must agree, with nothing checking.

**#48 Audit what we have built for pedagogical value, and write down the sequencing**

Enough has accumulated that nobody can now say what a student is meant to learn, in what
order, or where the gaps are. The material was built task by task, each one justified on
its own, and never against a curriculum.

**Does something like this already exist?** Partly, and not enough.
`docs/docs/tutorials/index.md` has a "Learning Path" section, but it is student-facing
navigation over the five notebooks: a card per tutorial with a one-line description. It
states no outcomes, covers none of the concept pages or library modules, and predates
most of what exists now. It is a table of contents, not an audit.

What the artifact should carry:

- **Sequencing.** What depends on what. Some of this is already load-bearing and
  undocumented: tutorial 3 loads the checkpoint tutorial 2 trains, and #40's lesson only
  works on a model that is wrong often enough to be interesting, which is why it uses
  tutorial 5's checkpoint rather than tutorial 3's toy.
- **Learning outcomes per unit**, stated as what a student can *do* afterwards, not what
  was covered.
- **Coverage gaps**, which is the real output. Likely candidates on a first glance:
  training dynamics beyond "loss goes down", evaluation beyond BLEU, and anything about
  why a model fails rather than how it works.
- **Redundancy**, the other half. Beam search is now explained in `concepts/decoding.md`,
  reimplemented in tutorial 3, and visualized in two places.

Worth auditing against, since each was justified pedagogically when it was built:

| Where | What it teaches |
|---|---|
| Tutorials 1-5 | The end-to-end path, toy model through real translations |
| `concepts/decoding.md` | Greedy vs beam, cost, what the knobs buy (#40), search framing (#41) |
| `concepts/data-pipeline.md` | Loading, cleaning, alignment detection (#46) and repair (#29) |
| `concepts/vocabulary.md` | Words vs subwords |
| `concepts/models.md`, `training.md`, `what-is-nmt.md` | Architecture and training |
| `reference/visualization.md` | Attention maps, beam search traces |
| Generated measurements | `decode_bench`, `decoding_sweep`, `alignment_diagnosis`, `realign_report` |

**First pass written: `notes/CURRICULUM.md`.** What it found:

- Two load-bearing dependencies nobody had written down. Tutorial 3 cannot run without
  tutorial 2's checkpoint, and tutorial 3's model is too small to demonstrate the thing
  tutorial 3 teaches, which is why #40 had to measure on tutorial 5's model and why #50
  exists.
- Five coverage gaps, the largest being **why a model fails**. Everything teaches the
  machinery working; nothing teaches diagnosis, which is what a student actually hits.
  Others: evaluation beyond BLEU, training dynamics when training goes wrong, how much
  data is enough, and inference cost in practice.
- Beam search now appears four times and attention three. Defensible, but currently by
  accumulation rather than decision.

Still open, and genuinely instructor-owned: the outcomes in that file are reverse-
engineered from the material, so they describe what exists rather than what the course
needs. Four questions are listed at the bottom of it for you. #42 is the same shape.

One more question for the audit, from the competitive assessment written up in
`docs/docs/related-work.md`: **does Joey NMT belong *in* the syllabus** as a comparison
point — "here is the same system as a configured toolkit rather than a library you
call" — instead of only in related work? Their toy config trains in 3m52s on CPU to
93.62 BLEU, so it is cheap enough for a student to run beside ours.


**#46 Teach the corpus repair instead of doing it silently**

Coulson's suggestion on his approval of #19: the data cleaning "could be recorded and used
as an example for cleaning data. Instead of doing it quietly in the background we could
explain it to the students to reinforce the idea of clean data."

He is right, and right about its weak part too. The repair currently lives entirely in
`scripts/realign_corpus.py`, so the single most pedagogically loaded thing in the repo is
the one thing no student sees. What makes it teachable is that the diagnosis is already
quantified and the numbers are dramatic:

```
                      broken    repaired
length correlation     0.001       0.969
anchor agreement        1.4%       38.9%
```

That is a complete lesson in how to tell misaligned parallel data from your own modeling
mistake, which is exactly the confusion a student cannot resolve on their own.

- Use the alignment diagnosis as the spine, not the cleaning filters. Coulson notes the
  stage-direction removal "does almost nothing so it may be a poor example," and he is
  right: it removes exactly one line. It belongs as a footnote at most.
- The two checks are cheap enough to run live in a notebook on the shipped corpus, and
  the broken state can be reconstructed by re-zipping the columns, so students can see
  both numbers move.
- `tests/test_data_integrity.py` already encodes the thresholds and the reasoning. The
  tutorial and the test should quote the same source rather than restate the numbers.
- Open question for the author: standalone tutorial, or a section inside tutorial 1 where
  the corpus is first loaded.


**#26 Four broken doc links block `mkdocs build --strict`**
All four link to source files as though they were doc pages, so docs cannot be gated in
CI as-is:
`MULTILINGUAL_ANALYSIS.md` → `preprocessing/multilingual.py` and → `config.py`;
`MULTILINGUAL_QUICKSTART.md` → `examples/multilingual_training_example.py`;
`TESTING_GUIDE.md` → `preprocessing/sentencepiece.py#L102`.
Point them at the mkdocstrings reference pages or at GitHub URLs, then add a docs build to
CI. A fifth warning (missing return annotation in `visualization.py`) was introduced by #5
and fixed there.

**#28 Attention parameters skip `_init_weights`**
`SimpleSeq2SeqLSTM._init_weights` matches on `weight_ih` / `weight_hh` / `bias`, so
`AdditiveAttention`'s `W_dec`/`W_enc`/`v` and `attn_combine` keep PyTorch's default Linear
init. Defensible — they train well, additive reaches 93.6% alignment accuracy — but it is
currently implicit rather than chosen. Either extend `_init_weights` deliberately or leave
a comment saying the default is intended. Small, and worth settling while it is fresh.

**#29 Recover the last 98 talks with a sentence aligner**
#20 keeps only talks whose two transcripts have identical line counts (564 of 662). The
remaining 98 have differing counts — median delta 0, 90th percentile 1, max 19 — so their
segmentation diverged slightly rather than catastrophically. A Gale-Church length-based
aligner handling 1-1, 1-2, 2-1, 1-0 and 0-1 would recover roughly **13k additional pairs**
on top of the 73k already in hand.
- Explicitly *not* done in #20: guessing at alignment is how this corpus got into trouble
  in the first place, and 73k correct pairs beat 86k uncertain ones for a teaching library.
- Worth doing only if the extra data is actually wanted; it is a real aligner, not a
  one-liner, and `scripts/realign_corpus.py` is the natural place for it.

