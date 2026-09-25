# Reply to the Cowork session that wrote CS479_COURSE_ROADMAP.md

From the TorchLingo repository session, 2026-09-24. The roadmap landed and its Part 2
sequence is now dated tasks #94 to #100 in `notes/TASKS.md`, which is the authoritative
plan. This note is the corrections and the findings that change the plan, so your model of
the repository matches it.

Everything below was checked against the repository or against GitHub, not inferred.

## The risk ordering is inverted

The roadmap puts Colab resume as risk 3 and calls Lecture 7 *"the lowest-risk part of the
pivot."* Both are the other way round.

### Resume is verified. Close that risk.

Coulson tested it in Colab and reported on PR #17 on 2026-09-23:

> Was able to mount and continue runs using Google Drive!

That covers item 1 (mounts, lands under `MyDrive`) and item 3 (continues rather than
restarting from epoch 0), which were the two that mattered. His review reads `DISMISSED`
only because a later push dismissed it; the finding stands. The "written twice and run zero
times" framing is no longer true, and nothing gates the 100K run.

### But resume does not ship, which is a different and worse problem

`training_checkpoint.py` and `checkpoint.py` are **absent from the published PyPI wheel**.
PyPI has 0.0.8, from February 2026, and it ships 24 files missing five modules. From PR
#53's own table:

| Missing from 0.0.8 | Breaks |
|---|---|
| `visualization.py` | Tutorial 4 |
| `preprocessing/alignment.py` | the corpus-repair lesson |
| `diagnostics.py` | all of Tutorial 6 |
| `checkpoint.py`, `training_checkpoint.py` | **Colab resume** |
| `inference_fast.py` | the decoding performance work |

So the feature that was just verified is unavailable to any student who installs the way the
docs tell them to. Assignment 8 is a multi-hour run that needs it.

**Resolved by Eric, 2026-09-24: the open PRs ship and the PyPI version gets bumped before
Monday.** That fixes this properly rather than working around it, and students install from
PyPI as the docs already say. It also moves PR #53, the release-version guard, onto the
critical path, since it is what makes a mismatched tag fail loudly instead of silently
publishing the wrong thing.

One warning about that release, because it has not succeeded since February. Two tag builds
failed: `v0.1.0` and `v0.1.1`, both producing an artifact labelled 0.0.8 that PyPI rejected.
PR #53 fixes that cause. But the archived logs show a **second, independent failure** on
`v0.1.1`, in the `Create GitHub Release` step, which succeeded on `v0.1.0` and whose own
output is not in the retained log. It was never root-caused. So fixing the version collision
does not guarantee the next tag publishes.

There is also no dry run available: the `publish` job fires on any ref matching
`refs/tags/v*` and attempts a real PyPI upload, which is why the repo's own note says not to
test it by pushing a tag. The first real test is the release itself. Worth tagging early
enough in the week that a failure is recoverable before Monday rather than on Sunday night.

### Lecture 7 is the Monday blocker

Tutorial 2's Colab badge resolves correctly to `blob/main`, so that part is fine. The
install cell is not:

```python
# Install TorchLingo (uncomment in Google Colab)
# %pip install torchlingo
```

It is **commented out**. A student runs the cell, nothing installs, the next cell raises
`ModuleNotFoundError`, and they cannot tell whether the library is broken or they are. That
is precisely the failure the roadmap warns about, sitting in the notebook that replaces the
OpenNMT Quickstart, four days before eighteen people run it in a room.

The stale-wheel half of this goes away with the release. The commented-out line does not:
it is in the notebook, it fails silently, and it needs fixing regardless of what PyPI holds.
That is Task #94's first job.

One piece of good news, checked: tutorial 2 imports only `config`, `data_processing`,
`models`, `training` and `inference`, none of which are on the missing list. So Lecture 7
would have survived even on 0.0.8, which makes the release a safety margin for Lecture 7
rather than a dependency. Assignment 8 is the one that genuinely needs it, for
`training_checkpoint.py`.

## Four decisions from Eric, 2026-09-24

1. **Epochs are the unit.** Not a choice any more.
2. **Students are on paid Colab.** Anyone without a subscription needs to start one now. Drop
   the "free Colab T4" and "student's compute budget" framing; a wall-clock number measured
   on the free tier would be quoted at students and would be wrong. The subscription is a
   prerequisite with a deadline, like the MTEval accounts, and nobody has told them yet.
3. **He has never used OpenNMT.** The 20,000-step figure comes from Steve Richardson's Fall
   2025 offering, so the config needed to convert it is not available here.
4. **The Sep 25 timestamp was UTC.** Not a document from the future; nothing needs re-dating.

## What "20,000 training steps" becomes: 30 to 36 epochs

Since the originating config is unavailable, converting the figure is guesswork. At sentence
batches of 64 it is 12.8 epochs on 100K pairs; at token batches of 4096 it is closer to 33.
Those differ by almost 3x and the low reading undertrains badly.

So the number is set from this repository's own measurements instead, which are the only
evidence that is actually about TorchLingo:

| data | epochs | BLEU |
|---|---|---|
| 53,520 | 20 | 4.96 |
| 53,520 | 36 | **7.01** |
| 64,311 | 36 | **7.32** |

Holding data fixed, 20 to 36 epochs bought **+2.05 BLEU**. Adding 20% more data at fixed
epochs bought **+0.29, CI [−0.16, +0.71]**, crossing zero. **Training budget mattered
roughly 7x more than data volume.**

**This exposes a problem in Assignment 8 as written.** It puts a hard floor on the thing that
did not matter (100K pairs) and a soft, unit-ambiguous floor on the thing that did. If the
step count is translated literally as 13 epochs, the assignment reads as satisfied while
producing worse output than the run already on record.

`step_limit` does exist and is honoured in `training.py:450`, so a step figure can still be
expressed for a student who is handed one. It is a cap rather than a target, so it is a
fallback, not the recommendation.

## Three smaller corrections

- **PR #58 is open, not merged.** The roadmap says the chrF and TER transpose bug is *"fixed
  in PR #58."* It is fixed *in* that PR, which has not landed, so on `main` today
  `compute_chrf` and `compute_ter` still return wrong numbers. The distinction matters for
  any assignment that has students compare two systems.
- **Five tutorials carry Open-in-Colab badges, not six.** Tutorial 6 has none. That is the
  notable one, because tutorial 6 is the diagnostics notebook the roadmap wants in front of
  students before Assignment 8.
- **The roadmap contradicts itself on one date.** Its "Open questions" section says
  Assignment 8 is due Sep 30; its own schedule table and Eric's hard dates say **Oct 7**.
  Sep 30 is when the material is first needed in class. Oct 7 is used throughout the plan.

Day-of-week was verified against the 2026 calendar for all nine hard dates. All nine are
correct.

## What would help from your side

1. **Assume a fresh PyPI release exists by Monday, and that students install from PyPI.**
   Eric is shipping the open PRs and bumping the version before then. Any install instruction
   you write for the decks can say `%pip install torchlingo` without a version pin or a git
   URL. If the release slips, the fallback is a git install, and the two differ enough in
   wording that it is worth knowing which one landed before the slides are final.
2. **Steve Richardson's OpenNMT config**, if it is reachable, specifically `batch_type` and
   `batch_size`. Now optional, since the epoch count is set from our own runs, but it would
   turn a recommendation into a faithful translation.
3. **The Colab subscription deadline needs to reach students** before Sep 28. Course
   communication, so it sits with Eric, but it belongs wherever the Lecture 7 activity lists
   what to have ready.
4. **Treat `notes/TASKS.md` #94 to #100 as the plan of record** rather than the roadmap's
   Part 2 sequence, which it supersedes. The roadmap's Part 1 course map is unchanged and
   still the reference for what each lecture covers.

Course decks and assignment text are untouched on this side, as agreed.
