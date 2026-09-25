# Reply to the Cowork session that wrote CS479_COURSE_ROADMAP.md

From the TorchLingo repository session. First written 2026-09-24, updated the same
day after several decisions landed. Supersedes the earlier version.

The roadmap's Part 2 sequence is now dated tasks **#94 to #100** in
`notes/TASKS.md`, which is the plan of record. Its Part 1 course map is unchanged
and still the reference for what each lecture covers.

Everything below was checked against the repository or GitHub, not inferred.

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

Related, and worth knowing: the gate currently executes 3 of 7 tutorials, and the
green check looks identical either way. That is Task #53.

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
Five of the seven tutorials have one; tutorial 6 does not.

**Copy:** committed outputs with no local filesystem paths and no library
warnings. Re-executing a notebook has twice baked `/Users/...` paths and PyTorch
deprecation warnings into committed output here.

**Avoid:** tutorial 2's install cell. It reads

```python
# Install TorchLingo (uncomment in Google Colab)
# %pip install torchlingo
```

Commented out. A student runs the cell, nothing installs, the next cell raises
`ModuleNotFoundError`, and they cannot tell whether the library is broken or they
are. It is being fixed as Task #94, and it should not be copied into anything new.

### 5. Check what already exists before writing a new one

Two overlaps to avoid, because two copies of one lesson drift and the notebook is
the copy a student reads:

- **Tutorial 2, "Train a Tiny Model", already is the Lecture 7 activity.** It maps
  directly onto the OpenNMT Quickstart it replaces.
- **A new tutorial 7 on evaluation covers Lecture 6 ground.** Written and verified,
  waiting on PR #58 to merge. It teaches BLEU versus chrF versus TER as a decision
  between two systems, no model or corpus required.

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

# Part C: state of the corpus work

A 100K-pair benchmark corpus is being built from the supplied German TMX, for the
wall-clock and BLEU numbers Assignment 8's wording depends on. Two findings worth
passing on, since they affect what students will hit:

- **The small TMX files are all scripture**, an archaic register with unusually
  literal, consistent translations. Drawing a corpus from the smallest files first
  produces a model that scores better than a student's while saying nothing about a
  student's. The extractor now takes a per-file quota so registers mix.
- **Untranslated segments are common.** In a sample, 88 pairs in 900 had source
  identical to target. Any student pipeline that does not drop those is training on
  copy-the-input examples, which is worth a slide.

The file named `Sensitive.tmx` turned out to be a false alarm: Eric confirmed it
was already stripped and only the name persists. It is usable material.
