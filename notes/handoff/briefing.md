# Standing briefing for the Cowork session

Written from the TorchLingo repository session. **This is current state, edited in
place.** For what changed and when, read `to-cowork.md` beside it. The protocol is in
`notes/README.md`.

Began 2026-09-24 as a reply to `notes/CS479_COURSE_ROADMAP.md`, and has outgrown that
framing: it is now the document a session reads once to get oriented.

The roadmap's Part 2 sequence is dated tasks **#94 to #105** in `notes/TASKS.md`, which
is the plan of record. Its Part 1 course map is unchanged and remains the reference for
what each lecture covers.

Everything below was checked against the repository or GitHub, not inferred. Where a
claim has been corrected, the correction is marked rather than quietly applied.

---

# Part A: the ask — put the lecture notebooks in this repository

Yes, and `torchlingo` is the right primary archive for them. Five conditions,
each of which comes from something that has already gone wrong here.

### 1. A new directory, not `docs/docs/tutorials/`

Put them in `docs/docs/course/`, keyed to lecture number.

`docs/docs/tutorials/` is the library's own series, ordered by topic and numbered
01 to 07, wired into the docs nav, and aimed at anyone who finds the project.
Lecture activities are keyed to a calendar, aimed at eighteen enrolled students,
and have a different lifecycle: they get rewritten each offering. Mixing the two
collides the numbering and makes both harder to maintain.

### 2. They must be executed by CI, or explicitly declared exempt

This is the condition that matters most, and it is not obvious from outside.

`scripts/execute_notebooks.py` globs **only** `docs/docs/tutorials/*.ipynb`. And
the docs site does not run notebooks at all: `docs/mkdocs.yml` configures
mkdocs-jupyter with `execute: false` and `allow_errors: true`. So a notebook that
lands outside that one directory can rot completely, with wrong results or an
outright exception, and **neither the docs build nor a reader will surface it**.

That is not hypothetical. Tutorial 2 once shipped producing empty translations for
every phrase while printing a plausible training curve, and tutorial 3 raised
`NameError` on a clean run. Both went unnoticed. In a teaching library that is the
worst failure available, because a student cannot tell "the notebook is broken"
from "I did it wrong."

So when the notebooks land, extend the gate to cover the new directory, and add a
`REQUIREMENTS` entry for any notebook needing Git LFS data or a GPU, so it
**skips** rather than fails. CI checks out without LFS deliberately.

Related, and worth knowing: on `main` today the gate executes **2 of the 6**
tutorials, because the other four need Git LFS data that CI does not fetch. It rises
to 3 of 7 once the evaluation tutorial lands, since that one needs no model or
corpus. Either way the green check looks identical, which is Task #53.

### 3. Nothing from the course data, and no assignment solutions

The repository is **public**.

The German TMX set and the extractor that reads it have just been moved to a
private sibling repository, `torchlingo-private`. The size and licensing reasons
are obvious, but the one most easily missed is different: **a working extractor
and cleaner is a finished answer to the Lecture 4 and 5 assignment.** Students
write that 16-step pipeline themselves; publishing a version hands them two weeks
of graded work.

The same test applies to notebooks. If it does a student's graded work, it goes in
the private repository. If it teaches the concept without doing the work, it can
be public.

Benchmark **numbers and methodology stay public**, since a wall-clock figure and a
BLEU score are publishable even when the corpus that produced them is not.

### 4. Two conventions to copy, one to avoid

**Copy:** the Open-in-Colab badge, pointing at
`https://colab.research.google.com/github/byu-matrix-lab/torchlingo/blob/main/<path>`.
Five of the six tutorials on `main` have one; **tutorial 6 does not**, and the
unmerged tutorial 7 did not either until this was written, which is how the
omission propagates: nothing checks for it, so each new notebook inherits whatever
the author happened to remember. Worth adding a check rather than trusting the next
author, and worth copying the badge deliberately in the meantime.

**Copy:** committed outputs with no local filesystem paths and no library
warnings. Re-executing a notebook has twice baked `/Users/...` paths and PyTorch
deprecation warnings into committed output here.

**Copy:** the setup pattern from tutorial 2, which was rewritten for this and is
now the house convention. Two cells, and both halves matter.

Until today that cell read:

```python
# Install TorchLingo (uncomment in Google Colab)
# %pip install torchlingo
```

Commented out. A student runs it, it *succeeds* by doing nothing, the next cell
raises `ModuleNotFoundError`, and they cannot tell whether the library is broken or
they missed a step. In a twenty-minute activity on eighteen laptops that is the most
expensive possible failure.

What replaced it, and why each choice is the way it is:

1. **Detect Colab and install unconditionally.** Nothing to uncomment, because a
   step a student must remember is a step some students will not.
2. **Use `subprocess`, not the `%pip` magic**, so behaviour does not depend on the
   magic surviving a conditional.
3. **Raise on a failed install, do not print.** A failure that only prints leaves
   the student debugging an `ImportError` ten cells later instead of a pip error
   where it happened.
4. **Follow it with a verification cell that imports every module the series uses**
   and raises with the upgrade command if any is missing.

That fourth point has a non-obvious reason worth knowing, because it was found by
testing rather than assumed. `torchlingo/__init__.py` imports its submodules
**eagerly**, so an incomplete install fails wholesale at `import torchlingo`. There
is no such thing as a partly working install, and a setup cell that probes only the
modules it happens to need reports nothing a blanket check would not. A first draft
of that cell had a "warn about later modules" branch which was therefore unreachable
dead code.

The second half of the argument is pedagogical: a student whose install cannot run
Tutorial 6 should discover that on Monday, in a room with an instructor in it, and
not alone in week four. The fix is one command either way.

### 5. Just drop the files in; git is handled on this side

Write notebooks into the working tree at:

```
docs/docs/course/lecture-NN-<short-slug>.ipynb
```

Two digits, keyed to lecture number, so `lecture-08-train-a-real-model.ipynb` sorts
next to its lecture and never collides with the library's own `01` to `07` series.

**Decided, and it removes a constraint rather than adding one.** The library's
tutorials do **not** need to line up numerically with lecture numbers. Once the
notebooks currently in flight land, they will be given stable unique names, and the
slides refer to them **by name** rather than by an implied number match.

So: cite notebooks in slides by title and filename, never as "tutorial 7 is the
Lecture 7 one", because that correspondence is not going to hold and does not need
to. Tutorial 2 is the Lecture 7 activity today, which is the clearest illustration of
why the numbers were never going to align.

Two practical consequences for you:

- `lecture-NN-` in `docs/docs/course/` stays correct for your own notebooks. That
  series *is* keyed to lectures, and nothing you write needs revisiting.
- Library tutorial filenames may change once the open pull requests land. Four of
  the nine open pull requests touch tutorial notebooks, and so does one unmerged
  branch, and a rename is a delete plus an add in git, which conflicts with all of
  them. So hold off hard-coding library tutorial filenames into slides until that
  settles, or expect one pass of find-and-replace. A mapping table of lecture to
  notebook, kept in one place, makes that one edit instead of a sweep.

**Do not run git.** No commits, no branches, no pull requests. Committing, the nav
entry, wiring the CI gate and the pull request are all handled from the repository
session. Files in the tree are enough.

Two things that would help, though:

- **Say what you added and which lecture it serves.** It goes into the commit message
  and the pull request description, and "what was wrong before" is the part reviewers
  here actually read.
- **Flag anything that should not be public.** If a notebook contains course data,
  student data, or a worked answer to a graded assignment, say so and it gets routed
  to `torchlingo-private` instead, which takes direct commits. The public tree is the
  wrong place for it, and catching it before the commit is much cheaper than after:
  removing a file from a public repository's history means rewriting that history.

It will be checked on this side either way. But a file arriving with "this one has
the cleaning solution in it" saves a round trip and removes the chance of it slipping
through.

### 6. Check what already exists before writing a new one

Two overlaps to avoid, because two copies of one lesson drift and the notebook is
the copy a student reads:

- **Tutorial 2, "Train a Tiny Model", already is the Lecture 7 activity.** It maps
  directly onto the OpenNMT Quickstart it replaces.
- **A new tutorial 7 on evaluation covers Lecture 6 ground.** Written and verified,
  waiting on PR #58 to merge. It teaches BLEU versus chrF versus TER as a decision
  between two systems, no model or corpus required.

---

# Part A2: where train/dev/test discipline belongs in the schedule

This is a placement question for the decks, so it is yours rather than the
repository's. The recommendation, with the measurement behind it.

## Lecture 8, because Lecture 6 has already happened

**Corrected 2026-09-24.** An earlier version of this section recommended Lecture 6,
on the argument that it is the only lecture whose subject is *how a number misleads
you*, and that contamination is the mirror of the BLEU-returns-zero demonstration it
already contains. That argument still stands and the slot does not: **Lecture 6 ran on
Wednesday Sep 23**, and Lectures 1 through 6 are all in the past.

So the remaining options are Lecture 7 on Mon Sep 28 and Lecture 8 on Wed Sep 30, and
it should be **Lecture 8**:

- Lecture 8 is where Assignment 8 is handed out, and the splits are its deliverable.
  Teaching the discipline at the moment it is required is worse than teaching it a
  week earlier, but it is not too late, and it is seven days before the due date.
- Lecture 7 is already two lectures in one plus the install activity in 75 minutes,
  and the install is the highest-risk event of the semester. It should give time back,
  not take more.

**What that costs, said plainly.** In Lecture 6 this would have been a lesson about
measurement, taught before students had anything at stake. In Lecture 8 it is a
procedure attached to an assignment, which is a weaker form of the same content. If
there is ever a Fall 2027, this belongs in Lecture 6.

## Why this is worth a slide rather than a sentence

Assignment 8's phrase "with no overlap" sounds satisfied by `random.shuffle`. It is
not, and here is the size of it in the German data:

- **425,353 duplicate pairs** were removed during extraction, 31% of what survived.
- **63,122 rows remain that share a source sentence with another row but carry a
  different target.** Pair-level deduplication keeps every one of those, because they
  *are* distinct pairs. Measured: all 63,122 of them differ on the target side.

Shuffle rows and those sources land in train and in test. The same English sentence
appears on both sides with different references. BLEU rises, and nothing in the
output reveals it.

The fix is three steps and each is checkable: deduplicate on the **source** side,
split by **source group** rather than by row, then **verify** that no source appears
in two splits. The third step is the one students skip and the only one that catches
a mistake in the first two.

## The cost of getting it wrong compounds through four assignments

This is the argument for front-loading rather than fixing later. Assignment 9
retrains Assignment 8's system, 13 reuses it, and 14 is built from "the system you
created for Assignment 8/9". A contaminated split in Assignment 8 propagates into
three more assignments, and every "improvement" a student measures afterwards is
measured against an inflated baseline.

So a student who splits carelessly in week one does not get one wrong number. They
get four, and the later ones look like progress.

## What the library already gives you

`torchlingo.diagnostics.check_contamination` exists for exactly this and **names the
offending sentences** rather than reporting a count, which is what makes it teachable
rather than merely a gate. It is already in tutorial 6, as "Question 3: is it
learning the wrong thing?"

Tooling on the private side now does the whole pipeline and reports it: source-level
deduplication with a mode comparison, splitting that groups by source, and a
verification pass that intersects every pair of splits. Those are extraction tools,
not student-facing, but the numbers they produce are the slide.

## Other placements, offered in the same spirit

Tutorial 6 organises debugging as five questions, and they already fall in the
course's own order. That is a useful coincidence: the tutorial can be split across
lectures instead of taught once.

| Tutorial 6 question | Library call | Where it lands |
|---|---|---|
| Is the data what you think it is? | `diagnose_alignment` | Lecture 5, with cleaning |
| Is it learning *anything*? | `check_loss_moved`, `uniform_loss` | Lecture 7, first training |
| Is it learning the *wrong* thing? | `check_generalization`, `check_contamination` | Lecture 6, with splits |
| Is the measurement lying? | the evaluation thread | Lecture 6 |
| Is it the environment? | `check_eval_mode` | Lecture 8, first real inference |

Six specific suggestions, cheapest first.

**1. Give the first loss number a reference point, in Lecture 7.** That lecture
already covers cost functions, and the install activity produces a loss a student
cannot interpret. `uniform_loss(vocab_size)` is `ln(V)`, the loss of a model that has
learned nothing, so the first number they ever see has something to be compared
against. One line, and it prevents the commonest week-one confusion.

**2. Warn about the loss floor in Lecture 8's debrief.** `label_smoothing=0.1` is the
default, and it puts a nonzero floor under the loss, so a converged model plateaus
well above zero. Students read that plateau as failure. The Lecture 8 deck already
debriefs what went wrong for people; this belongs in it.

**3. Run the new evaluation tutorial as Lecture 6's Colab activity.** It needs no
model, no corpus and no GPU, so it works in a room where nothing is installed yet.
Students watch BLEU prefer the system that reversed the meaning of every sentence,
which is the sharpest possible version of Lecture 6's existing "raw BLEU is not
comparable" thread.

**4. Teach the signature in Lecture 6, where SacreBLEU is introduced.** The lecture
already argues that raw BLEU is not comparable. The signature is the concrete
mechanism for fixing that, and the library now prints one next to every BLEU.

**5. Move the beam-size discussion after Assignment 8, not before.** Tutorial 3
sweeps beam size and every row comes out identical, because a toy model is decisive
and has nothing to be uncertain about. The sweep only means something on a real
model, which students first have *after* Assignment 8. `concepts/decoding.md` carries
the measured version. Lecture 9 onwards is the natural home.

**6. Save the on-target language check for Lectures 13 and 14.** A multilingual or
low-resource system can emit fluent text in the wrong language entirely, and BLEU
hides it, because a copied source still scores against a related-language reference.
That check is not built yet; it is on the repository's list as a borrow from mtsurvey.

## And one place not to add anything: Lecture 7

Lecture 7 is already two lectures in one, paper-review assignment plus neural network
foundations, and it carries the install activity, in 75 minutes. The install is the
single highest-risk event in the semester: it is the first infrastructure hurdle, it
went badly for several students last year on a stack that had been in use for years,
and a new framework raises that risk rather than lowering it.

Suggestion 1 above is one line and earns its place. Beyond that, Lecture 7 should
give time back rather than take it. Anything else that wants to be there is better in
Lecture 8, which is nine days before its own assignment is due.

## What each placement asks the repository to build

Stated as development work, since that is what it is for. "Ships" means it exists and
is published; several things exist on `main` but are absent from the released wheel
until the version bump lands.

| Assignment | What TorchLingo must offer | State |
|---|---|---|
| A7 toy model | install, train, translate in 20 minutes | exists; install cell is broken |
| A8 real model | 100K training, checkpoint and resume, SacreBLEU | exists, **unmeasured at scale**; resume verified but unreleased |
| A9 tokenizer on/off | SentencePiece, plus a controlled comparison | SentencePiece exists; nothing enforces the control |
| A13 back-translation | reverse-direction config, bulk decoding of 100K+ | decoding exists; **no way to resume a long decode** |
| A14 multilingual | target-language tagging, per-direction scoring | `preprocessing.multilingual` exists, unexercised at this shape |

Three real gaps, in the order they bite.

**A9's comparison is unenforced.** The assignment *is* the comparison, so the only
thing that may differ between the two runs is the tokenizer. Nothing in the library
records what varied, and this project has already published a comparison that gave one
model 19% more data *and* 80% more training while claiming data was the only
difference. A student will make that mistake more easily than we did.

**A13 needs decoding to survive an interruption, and it cannot.** The throughput is
fine: the benchmark measures 27.5 ms per sentence with batched beams, so 100,000
sentences extrapolates to well under an hour, and less on a GPU. Read that as an
order of magnitude, since it was measured on eight sentences at `max_len=25`; a real
model at `max_len=60` will be several times slower.

The problem is different. Training has checkpoint-and-resume, verified in Colab.
**Inference has nothing.** A multi-hour back-translation run that dies at hour two
starts over from zero, which is precisely the failure that made resume a priority for
training. Either decoding should write output incrementally and skip what is already
done, or the workflow should decode in explicit shards. This is the single largest
undone piece of work for the second half of the course, and it is not on the list yet.

**A14 is the least exercised path in the library.** `preprocessing.multilingual`
exists but nothing has run it at "two directions, intermingled, separate test sets per
direction". Lowest risk by date, highest uncertainty by evidence.

---

# Part A3: where to stop using TorchLingo

Asked directly, so answered directly. **TorchLingo should own Lectures 7 through 9,
and the course should cut over to Hugging Face at Lecture 12.**

## Why 7 through 9 are the right scope

Those are the assignments where reading the implementation is the point. A student
builds a model from code they can follow, sees what a tokenizer does to it, and gets
a number out. A production framework would hide every mechanism the lecture is about,
and that is the whole reason this library exists.

## Why Lecture 12 is the natural cutover

Because the course already arrives there. Lecture 12 is in-context prompting of a
Hugging Face model, Lectures 10 and 11 already require a Hugging Face account for
COMET, and Lecture 16 leaves Python for Azure. So the cutover costs **no new
tooling**: it is the same account and the same library students already need for three
other assignments.

The pedagogical argument runs the same way. By Lecture 12 the student has built a
system from readable code, which was the goal. Lectures 13 and 14 ask different
questions: does back-translation help, does a multilingual model transfer. Those are
questions about *techniques*, and answering them needs a model strong enough for an
effect to be visible above the noise. Fine-tuning NLLB or a Marian checkpoint gives
that in a few lines and gives real baselines to compare against.

Put bluntly: measuring a back-translation gain on a system that scores BLEU 7 mostly
measures noise. The technique is the learning objective, not the framework, and a weak
baseline obscures it.

**What TorchLingo keeps after the cutover.** The diagnostics, which are about models
in general and not about this library; the concepts pages as assigned reading; and
the evaluation lesson. None of that is displaced by changing framework.

**What not to cut over to.** OpenNMT-py is in maintenance mode, which started this.
Fairseq is effectively unmaintained. Eole is the OpenNMT successor and is the obvious
candidate, but its claims are unverified here and it pins `torch < 2.13` while this
project runs 2.13, so it needs its own environment; that is on the repository's list
as a thing to check before anyone relies on it. Hugging Face avoids the question
entirely by being tooling the course already committed to.

## The decision that should be made in advance, not after

Whether TorchLingo can carry **A8** turns on numbers that do not exist yet, and the
honest way to handle that is to fix the thresholds before seeing them:

| 100K run result | Reading |
|---|---|
| BLEU 15 or better, inside about 3 hours | TorchLingo owns A8 comfortably; no change |
| BLEU 8 to 15 | workable, but the assignment must state expected quality explicitly, because "reasonably intelligible" will not be true for every student |
| below BLEU 8, or over about 6 hours | reshape A8: fewer pairs with more epochs, since training budget dominated data volume by roughly 7x, or move A8's *baseline* to Hugging Face and keep TorchLingo for the mechanism |

The existing evidence sits at the bottom of that table: 64,311 pairs and 36 epochs
reached **BLEU 7.32**. That is the number to beat, and it is why the run matters more
than anything else on the list.

Note also what the Fall 2025 deck implies. Lecture 10 opens with "For those who
obtained reasonably intelligible output from your OpenNMT systems", which suggests
some students missed that bar on the old stack too. Parity, not perfection, is the
bar the pivot has to clear.

---

# Part B: corrections to the roadmap

## The risk ordering is inverted

The roadmap makes Colab resume risk 3 and calls Lecture 7 *"the lowest-risk part
of the pivot."* Both are the other way round.

**Resume is verified.** Coulson tested it in Colab and reported on PR #17 on
2026-09-23: *"Was able to mount and continue runs using Google Drive!"* That
covers item 1 (mounts, lands under `MyDrive`) and item 3 (continues rather than
restarting from epoch 0), the two that mattered. His review reads `DISMISSED` only
because a later push dismissed it; the finding stands. "Written twice and run zero
times" is no longer true.

**Lecture 7 carries the actual Monday blocker**, and it is the commented-out
install cell in section 4 above.

## Resume works but did not ship, and that is being fixed

`training_checkpoint.py` and `checkpoint.py` are absent from the published PyPI
wheel. PyPI has 0.0.8, from February 2026, missing five modules:

| Missing from 0.0.8 | Breaks |
|---|---|
| `visualization.py` | Tutorial 4 |
| `preprocessing/alignment.py` | the corpus-repair lesson |
| `diagnostics.py` | all of Tutorial 6 |
| `checkpoint.py`, `training_checkpoint.py` | **Colab resume** |
| `inference_fast.py` | the decoding performance work |

**Eric is shipping the open PRs and bumping the version before Monday**, so write
install instructions as a plain `%pip install torchlingo` with no version pin and
no git URL.

One caution to plan around: that release has not succeeded since February. Two tag
builds failed, `v0.1.0` and `v0.1.1`, both producing an artifact labelled 0.0.8
that PyPI rejected. PR #53 fixes that cause. But the archived logs show a
**second, independent failure** on `v0.1.1`, in the `Create GitHub Release` step,
which succeeded on `v0.1.0` and whose own output is no longer retained. It was
never root-caused, so fixing the version collision does not guarantee the next tag
publishes. There is also no dry run: `publish` fires on any `v*` tag and attempts
a real upload. **The first real test is the release itself**, so it is worth
tagging early enough in the week that a failure is recoverable.

Checked, and reassuring: tutorial 2 imports only `config`, `data_processing`,
`models`, `training` and `inference`, none of which are on the missing list. So
Lecture 7 would survive even on the old wheel. The release is a safety margin for
Monday and a genuine dependency for Assignment 8.

## Four decisions from Eric

1. **Epochs are the unit.** Settled.
2. **Students are on paid Colab.** Anyone without a subscription needs to start one
   now. Drop the "free Colab T4" framing; a wall-clock number measured on the free
   tier would get quoted at students and would be wrong. The subscription is a
   prerequisite with a deadline, like the MTEval accounts, and nobody has told
   them yet.
3. **He has never used OpenNMT.** The 20,000-step figure comes from Steve
   Richardson's Fall 2025 offering, so the config needed to convert it is not
   available.
4. **The Sep 25 timestamp was UTC.** Not a document from the future.

## What "20,000 training steps" becomes: 30 to 36 epochs

Converting the figure is guesswork without the originating config. At sentence
batches of 64 it is 12.8 epochs on 100K pairs; at token batches of 4096 it is
closer to 33. Those differ by almost 3x and the low reading undertrains badly.

So the number is set from this repository's own measurements instead, the only
evidence that is actually about TorchLingo:

| data | epochs | BLEU |
|---|---|---|
| 53,520 | 20 | 4.96 |
| 53,520 | 36 | **7.01** |
| 64,311 | 36 | **7.32** |

Holding data fixed, 20 to 36 epochs bought **+2.05 BLEU**. Adding 20% more data at
fixed epochs bought **+0.29, CI [−0.16, +0.71]**, crossing zero. **Training budget
mattered roughly 7x more than data volume.**

**This exposes a problem in Assignment 8 as written.** It puts a hard floor on the
thing that did not matter (100K pairs) and a soft, unit-ambiguous floor on the
thing that did. Translate the step count literally as 13 epochs and the assignment
reads as satisfied while producing worse output than the run already on record.

`step_limit` exists and is honoured in `training.py:450`, so a step figure can
still be expressed for a student handed one. It is a cap rather than a target, so
it is a fallback, not the recommendation.

## Three smaller corrections

- **PR #58 is open, not merged.** The roadmap says the chrF and TER transpose bug is
  *"fixed in PR #58."* It is fixed *in* that PR, which has not landed, so on `main`
  today `compute_chrf` and `compute_ter` still return wrong numbers. The
  distinction matters for any assignment comparing two systems.
- **Five tutorials carry Open-in-Colab badges, not six.** Tutorial 6 has none, and
  that is the notable one: tutorial 6 is the diagnostics notebook the roadmap wants
  in front of students before Assignment 8.
- **The roadmap contradicts itself on one date.** Its "Open questions" says
  Assignment 8 is due Sep 30; its own schedule table and Eric's hard dates say
  **Oct 7**. Sep 30 is when the material is first needed in class. Oct 7 is used
  throughout the plan.

Day-of-week was verified against the 2026 calendar for all nine hard dates. All
nine are correct.

---

# Part C: the German corpus, extracted and measured

The full German TMX set is now extracted to line-aligned bitext: **1,370,658 clean
pairs**, `all.en` and `all.de`, with the line counts verified equal on both sides.
That is the same two-sentence-aligned-files format Lectures 4 and 5 ask students to
deliver, so their deliverable and ours are the same shape.

Four findings, all of which are course material rather than repository trivia.

### German is an outlier, and the 100K floor means something different for students

**Do not generalize from 1.37 million.** German is a high-resource language with an
unusually deep TM here, and the benchmark corpus is drawn from it with enormous
headroom. Students have **up to 200K bitext** for their chosen language, and Lecture 4
asks them to prepare at least 200K pairs.

So the same 100,000-pair floor is two completely different requirements:

| | available | 100K is |
|---|---|---|
| this benchmark (German) | 1,370,658 clean | 7% of it, a sample |
| a student | up to 200K, before cleaning | **half or more of everything they have** |

That matters for how the benchmark should be read. A figure produced from a 100K
sample of 1.37M would be a fair estimate of *training time* and an **optimistic**
estimate of quality, because the sample could be drawn from clean, deduplicated,
register-balanced material with slack to spare. A student drawing 100K out of 200K has
no slack and cannot be selective.

**So the benchmark is being run at 200K, not at 1.37M.** Eric's call: cap the German
corpus to what a student actually has, then split 100K train, 2K validation, 2K test
out of that. The resulting BLEU is then a number students can be held to, rather than
one produced with advantages they do not have.

Two residual differences remain, and both still favour the benchmark, so read its
quality figure as a ceiling rather than an expectation:

- The 200K is sampled from **already deduplicated** material. A student's 200K has
  never been deduplicated, and deduplication removed 31% of this corpus.
- German is a high-resource language with professionally maintained memories. A
  lower-resource choice will be noisier at the same size.

One thing that did *not* survive the cap, and is worth knowing: per-source test sets
for the rare registers collapse. Scripture FPLC supports 40 test sentences out of the
full corpus but only **8** out of a 200K sample. So per-register measurement is a
luxury of a large corpus, and a student cannot do it for a register they have little
of. That is a real limitation of the assignment, not of the tooling.

### The floor collides with the cleaning loss, and nobody has checked the arithmetic

This is the part worth a slide, and it follows from the two numbers above.

Cleaning removed **29.9%** of the German translation units. If a student's raw 200K
loses a comparable share, they finish with roughly 140K clean pairs. Then:

```
140,000 clean
  - 2,000 test
  - 2,000 validation
  = 136,000 available to train      clears the 100K floor, with 36K spare
```

That works. But it only works because they started at 200K. A student who prepared
150K raw, or whose language is dirtier than German, lands near or below the floor.

**This is no longer a risk to prevent. It is a fact to look up.** Students chose their
languages in Lecture 2 and delivered their cleaned bitexts for A5 on **Wed Sep 23**,
so the numbers already exist in what they submitted. Assignment 8 is due **Oct 7**,
which leaves thirteen days.

So the advice that was here, about stating whether the 200K floor means raw or clean
and about collecting counts in Lecture 5, is moot. Both lectures have run. What
replaces it is an audit, and it is worth doing this week:

- **Count the clean pairs in each submitted bitext.** Anyone under about 104,000 cannot
  do Assignment 8 as written, and they should learn that now rather than on Oct 6.
- **Count duplicate sources while you are there.** 31% of the German corpus was
  duplicate pairs, and a student pipeline that did not deduplicate on the source side
  will leak between train and test. See the section above.
- **Check the longest sentences against the 512-token ceiling**, for the languages
  actually chosen. German expands 5x at worst under SentencePiece, which puts a
  99-word sentence at 499 tokens against a 512 limit. A more agglutinative language at
  the same 100-word cap could exceed it outright.

That last one is now checkable rather than hypothetical, because the language list is
settled. A script that reports all three numbers for a bitext pair is small, and it
lives on the private side.

Worth deciding deliberately rather than by inheritance: given that training budget
mattered roughly **7x** more than data volume in the runs on record, a smaller corpus
trained longer may serve students better than a 100K floor they can barely reach.
More data at the same epoch count bought almost nothing.

### Cleaning drops 30%, and here is the itemised bill

Out of 1,955,324 candidate translation units, 584,666 were rejected, or 29.9%:

| Rejected | Count |
|---|---|
| duplicate pair | 425,353 |
| source identical to target | 142,529 |
| source length out of range | 7,545 |
| empty side | 4,089 |
| length ratio implausible | 3,971 |
| target length out of range | 1,179 |

The course argues that cleaning matters and currently argues it without numbers.
This is the number. Roughly three in ten units in a real, professionally maintained
translation memory are unusable for training as they arrive.

### The duplicates are a contamination trap, not just bloat

425,353 duplicate pairs is 31% of the surviving corpus. A student who splits before
deduplicating will put **identical pairs in both training and test**, and their BLEU
will be inflated by an amount nobody can see from the output.

This is worth a slide of its own, because the assignment's own wording invites it:
it asks for 2,000 test and 2,000 validation sentences "with no overlap", which
sounds satisfied by a random split. It is not, when a third of the corpus is
duplicated.

TorchLingo ships `check_contamination` in `torchlingo.diagnostics` for exactly this,
and it names the offending sentences rather than just reporting a count. That is the
cheapest possible tie-in between Lecture 5's cleaning and Lecture 8's training.

### Register is concentrated, so how a student samples matters

| Source | Pairs | Share |
|---|---|---|
| Legacy - 1 | 491,183 | 36% |
| Recent | 399,947 | 29% |
| Legacy - 2 | 254,029 | 19% |
| FamilyHistory - Recent | 144,595 | 11% |
| Sensitive | 24,435 | 2% |
| S&I Manual NEW | 22,690 | 2% |
| Scripture (BofM, DCPGPETC, FPLC) | 28,058 | 2% |
| S&I Manual OLD | 5,721 | <1% |

Two consequences. Scripture is only 2% of the whole, so the corpus is dominated by
general prose, which is good. But the *small* files are all scripture, so anyone who
samples "the first N" or "the smallest files" gets an archaic register with unusually
literal translations, and a model that scores better than it should while saying
nothing about the general case. The extractor keeps per-source outputs under
`by-source/` so the mix is a deliberate choice rather than an accident.

The file named `Sensitive.tmx` was a false alarm, as Eric confirmed: already
stripped, only the name persists. Including it was worth correcting, since it
contributed 24,435 pairs of general-register material.

None of the corpus itself can go in the public repository. These numbers can, and
they are the part the course needs.
