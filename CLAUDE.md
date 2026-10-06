# CLAUDE.md

Guidance for Claude Code working in this repository.

## Project overview

**TorchLingo exists first and foremost to support CS 479.** Eric's framing, 2026-09-26.
Third-party use of the library is a later concern.

That is a tie-breaker, not a slogan: when a choice could serve either a CS 479 student or a
general newcomer, it serves the student, and a gap only matters if a lecture or an assignment
walks into it. **`notes/CS479_COURSE_ROADMAP.md` is the single source of truth** for what the
course needs, and where that judgement gets made.

It is still a clean, documented PyTorch NMT library with Transformer and LSTM implementations —
that is *how* it serves the course, and it keeps the general-audience option open for later
without paying for it now.

**Goals, in priority order:**

1. **CS 479 works.** Twenty-four students, on their own data, against dated assignments. A
   defect on that path outranks anything else in this file.
2. **Educational clarity** — clean, readable code designed for learning. The reason the library
   exists rather than a configured toolkit, and what we decline to trade for speed.
3. **Documentation that executes** — runnable examples, generated numbers rather than typed ones.
4. **Simplicity** — avoid over-engineering; focus on core NMT concepts.
5. **Accessibility** — beginner-friendly, Google-style docstrings.

## Environment

Use the repository venv at `.venv`; create it at the repo root if absent. Activate it for every
`python`/`pip` invocation.

```bash
python3 -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
```

## Commands

```bash
scripts/preflight.sh                 # everything CI checks — see the workflow section
scripts/student_path.sh NOTEBOOK     # a notebook as a Colab student runs it — see the course section
python scripts/execute_notebooks.py --as-student   # every notebook that way, one environment each
ruff check --fix src tests && ruff format src tests
python -m unittest discover tests
python -m unittest tests.test_config -v                                    # one module
python -m unittest tests.test_preprocessing.TestLoadDataParallelFiles -v   # one case
./scripts/serve_docs.sh              # or: mkdocs build
./scripts/build_wheel.sh             # or: python -m build
```

## Architecture

Modules under `src/torchlingo/`:

| | |
|---|---|
| `config.py` | `Config` dataclass and module-level constants |
| `models/` | `transformer_simple.py`, `lstm_simple.py`, `positional.py` (sinusoidal encoding) |
| `data_processing/` | `dataset.py` (`NMTDataset`), `vocab.py` (`BaseVocab`, `SimpleVocab`, `SentencePieceVocab`), `batching.py` (collate, `create_dataloaders`) |
| `preprocessing/` | `base.py` (`load_data`, `save_data`, `parallel_txt_to_dataframe`, `split_data`), `sentencepiece.py`, `multilingual.py` |
| `training.py` | `train_model()`, `TrainResult`, optional TensorBoard |
| `inference.py` | `greedy_decode`, `beam_search_decode`, `translate_batch` |
| `diagnostics.py` | `uniform_loss`, `check_contamination`, `check_gradients`, and friends |

**Config pattern.** Every function takes an optional `Config`, and **explicit parameters always
win**:

```python
def f(param=None, config: Optional[Config] = None):
    cfg = config if config is not None else get_default_config()
    param = param if param is not None else cfg.param
```

**Interfaces.** Transformers expose `encode()`/`decode()`; LSTMs expose `src_embed`, `encoder`,
`decoder`, `output`. Both share `forward(src, tgt)`, and inference detects the type with
`hasattr()`. Vocabs inherit `BaseVocab` and must implement `encode(text, add_special_tokens=True)
-> List[int]` and `decode(ids, skip_special_tokens=True) -> str`; `pad_idx`, `sos_idx` and
`eos_idx` may be attributes.

**Data flow.** Raw parallel text → `load_data()` → DataFrame → `NMTDataset` → `DataLoader` with a
collate function → `train_model()` → `TrainResult`; then `translate_batch()`.

**Also available.** SentencePiece (`train_sentencepiece`, then `SentencePieceVocab`; special
tokens are embedded in the model). TensorBoard via `config.use_tensorboard`, logging to
`config.tensorboard_dir / config.experiment_name`. `preprocessing.multilingual` for multi-parallel
corpora, language-pair filtering and back-translation.

## Conventions

`UPPER_SNAKE_CASE` constants, `PascalCase` classes, `snake_case` functions, `_leading_underscore`
for private. Annotate public functions and methods. Google-style docstrings with Args, Returns,
Raises and Examples, kept runnable. Prefer relative imports inside the package
(`from ..config import Config`).

**Pass `num_workers` explicitly wherever it matters; do not rely on the default.** The library
default stays 4 (Eric, 2026-09-28: fine as-is, as long as each invocation passes what it needs).
It matters because workers cost ~23 s of fixed start-up under macOS's `spawn` at every corpus size
measured (8 to 20,000 rows) and never won there, and `create_dataloaders` hands the value to all
three loaders. Course notebooks and anything a student runs pass `num_workers=0`.

**Length bucketing is the same kind of choice: off by default, on where training time is
spent.** It changes batch composition and therefore results, so the default does not flip
silently; but on a 100,000-pair corpus it removed 74% of padded tokens, and on 2026-09-28 it
trained 1.86x faster per batch (on an Apple GPU, synthetic text). The A8 kickoff notebook opts in
with `create_dataloaders(..., use_bucketing=True)`, seeding Python's `random`, which the bucketing
sampler shuffles with.

## Workflow after code changes

1. **Format and lint**: `ruff check --fix src tests && ruff format src tests`
2. **Test**: relevant tests, or the full suite if core behaviour changed.
3. **Update documentation** if needed — search `docs/` for usages of anything you edited. Docs
   are for beginners: clear explanations, worked examples.
4. **Run `scripts/preflight.sh` before pushing.**

```bash
scripts/preflight.sh            # ~50s: static gates, both test runners, doctests, docs
scripts/preflight.sh --quick    # ~1s:  the four static gates only
scripts/preflight.sh --all      # also executes the runnable notebooks (minutes)
```

It runs exactly what CI runs, cheapest gate first, stopping at the first failure: `ruff check`,
`ruff format --check`, `render_report.py --check`, `notebook_meta.py --check`, the suite under
**both** pytest and `unittest discover`, `--doctest-modules`, `mkdocs build --strict`, and
optionally `execute_notebooks.py`.

**CI gates seven things across four jobs, and running them from memory means running most of
them.** The forgotten one is the one that fails: on 2026-09-28 a PR burned a full cycle on
`render_report.py --check` — 0.4s locally — because a generated report had been hand-edited. A
GitHub round trip is four to six minutes.

## Pull requests

### Say "PR #X" and "Task #Y", never a bare `#N`

The two numbering schemes overlap almost completely, so **most numbers name one of each**. Write
"PR #37" or "Task #37" in prose, commit messages and GitHub comments alike. Task *subjects* keep
their bare `#N` prefix — this is about how they are referred to, not how they are titled.

Not pedantry: Task #37 was retired in the same breath as PR #37 was called open, both as "#37".
And on GitHub a bare `#N` **auto-links to the pull request** of that number, so an unqualified
task reference silently becomes a wrong link.

### Do not stack pull requests

Every PR targets `main`. If work depends on something unmerged, keep it on a local branch and
open the PR once the dependency lands. Split large changes by *concern* into independent PRs, not
into a chain. Merge promptly — most stacks here existed because something sat waiting for review.

Evidence, not taste. Stacking has failed two ways that are specific to it:

| Auto-close | PRs #11 and #12 closed when the base branch was deleted on merge |
|---|---|
| Lost approvals | a rebase changing no content dismissed the approvals on PRs #27 and #34 |

GitHub has no rebase exemption for `dismiss_stale_reviews_on_push`, so no configuration makes the
pattern safe.

To unwind an existing stack, **order matters**: merge the base PR *without* `--delete-branch`,
retarget the dependents to `main`, *then* delete the branch. Deleting first auto-closes the
dependents, which is how #11 and #12 were lost.

### A PR can close itself, and targeting `main` is not safety

PR #26 and PR #84 both went OPEN to CLOSED, never merged, within one second of an unrelated
squash-merge that passed `--delete-branch`. Different files, mechanism never established. **PR #84
was based on `main`**, so this is not stacking-specific.

- **Compare the open-PR set after every merge** — it should be exactly what it was, minus the one
  merged. This is how #84 was caught and reopened in one command; #26 took a day to notice.
- **Prefer merging without `--delete-branch`**, deleting the branch as a separate step, until the
  mechanism is understood. The cost is one command.

### Retargeting a PR dismisses its approvals

`gh pr edit <N> --base main` flips `APPROVED` to `REVIEW_REQUIRED` immediately — no commit, no
push, no content change, and no warning. Separate from push-dismissal, and learned expensively:
unwinding one stack cost three approvals an hour after a reviewer worked through nine PRs.

**Retarget before asking for review, never after.** If unavoidable on an approved PR, say so in
the re-review request and confirm the content is unchanged.

### A green PR is green against the base it last saw

Checks record "passed against `main` as it was when they ran"; they are not re-run because `main`
moved, and GitHub still shows `MERGEABLE / CLEAN`. PR #17 carried ten green ticks while `main` had
grown a `--doctest-modules` gate its new module failed — merging would have turned `main` red.

**Before merging any PR more than a few days old**, merge `main` into it and let CI re-run, or
build the merge result locally.

### Check which branch you are on before a destructive git command

`git checkout <b> || git checkout -b <b>` can fail *both* ways — a branch held by another
worktree cannot be checked out, and `-b` then fails because it exists — and a following
`git reset --hard` runs anyway, against whatever branch you were standing on. That silently moved
local `main` onto another branch's commits here. Put `git branch --show-current` between the
checkout and anything destructive, and read it.

## `docs/docs/course/` belongs to this repository

The CS 479 in-class notebooks live here. The Cowork session writes their content but does **not**
run git: changes arrive as requests in `notes/handoff/from-cowork/`, and committing, the nav entry
and the pull request happen here.

- **Instructor notebooks never go public.** Lectures 3 and 4 have separate INSTRUCTOR notebooks
  carrying worked solutions. They stay out of this tree.
- **Editing a notebook here does not disturb a live assignment.** Eric, 2026-09-27: students work
  from a *published* copy. **Do not defer a notebook fix because an assignment is in flight** — a
  rule that used to say the opposite blocked two Lecture 6 tasks for nothing. What remains true:
  **Drive copies are never deleted while students are in them**; only the copies on Eric's desktop
  are Cowork's to remove. Deleting what a student is working in is the hazard; editing the source
  is not.
- **New notebooks open with the standard install cell, then `torchlingo.colab.setup`.** The
  install runs `%pip` only in Colab and ends with `import torchlingo`, so a failed install stops
  in that cell rather than one later. Its second `%pip` line, `--force-reinstall --no-deps`, is
  not optional: pip treats an installed TorchLingo with the same version number as satisfying
  the first, so a session that already had one kept older code (found 2026-10-05); `setup(gpu=, drive=, data=)` does the device, the Drive mount and
  the downloads. Copy tutorial 3's first two code cells. Not the old commented-out install,
  which was the bug, nor the 20-to-60-line cells that replaced it.
- **Notebooks install from the `course` branch, not PyPI.** Eric, 2026-10-05: `%pip install
  "torchlingo @ git+https://github.com/byu-matrix-lab/torchlingo@course"`. A notebook and the
  library code it calls then reach students together, with no release in between; tutorial 5's
  self-attention cell waited on a release for exactly that reason. **`course` moves only through
  `scripts/promote_course.sh`**, which refuses a commit whose checks on `main` have not all
  passed, and only fast-forwards. Never point a notebook at `@main`: a red `main` would break
  every badge at once, and `main` was red the morning this was decided. Promote after merging
  anything students should see, with `scripts/promote_course.sh --await` to wait for the merge's
  checks rather than refuse while they run; each push to `course` runs the student-path workflow. PyPI
  releases continue for everyone else. Since `__version__` no longer identifies the code a
  student runs, `setup()` prints `torchlingo.version_label()`, e.g. `0.2.5 (course @ 11aa496)`:
  ask for that line with any bug report.
- **Notebooks are self-contained and carry no due dates.** Eric, 2026-09-29: due dates live in
  Learning Suite, and only there. Not "due before Lecture 8a", not "(due at Lecture 9)" in the
  purpose cell, not a "when" column keyed to a lecture; and no weekdays or calendar dates at all
  (Eric, 2026-09-28: the notebooks are meant to outlive one term's calendar). Naming a lecture
  or assignment for *what it is* is fine: "the A8 notebook", "Lecture 8a explains why",
  "turn in on Learning Suite". "Today", meaning the class session, is fine. Self-contained also
  means Run all from the Colab badge works: nothing to uncomment, no other notebook's output.

**Wrap the plumbing; keep the lesson inline.** Eric, 2026-09-28: large code cells lose a new
student. So code a student learns nothing from reading (installs, downloads, Drive, file
loading, exact splits, loader construction, padding arithmetic) belongs in the library, and a
notebook calls it. Code that *is* the lesson stays written out: tutorial 8's beam search, 8a's
length cap, dedupe and contamination check. Data fixtures stay visible too. Plumbing that stays
in a notebook because it serves only that notebook, such as tutorial 6's toy-corpus `Vocab`, opens
with `# Setup: run this cell, no need to read it.` (Eric, 2026-09-29).
Before writing a helper, check the library has not got one already: on the day this rule was
written, `create_dataloaders`, `parallel_txt_to_dataframe` and `evaluate_model` all existed and
no course notebook used them.

**Every notebook declares what it needs from its environment, in `requires`:** one of `pip`,
`download`, `colab`, `hf-token`, `blanks`. A closed set, because a typo would otherwise read as
"runnable in CI" — the opposite of what whoever wrote it meant. `needs` is repo-relative *paths*;
`requires` is *capabilities*; they are not interchangeable.

**A notebook declaring nothing gets executed in CI**, which is the only way a course notebook is
ever checked by running it. `lecture-10-comet-install` shipped `else:` followed by an unindented
`drive` — a bare SyntaxError in Lecture 10's own assignment notebook, unnoticed because nothing
executed this directory. A student would have hit it in the room.

So code that does not parse must declare `blanks`, and a `blanks` declaration must correspond to
real blanks — checked both ways, or the marker rots into a licence to ship broken cells. Note that
`!pip install` can appear *inside* a `try` block, so a plain `compile()` flags notebooks that run
perfectly well.

**Green CI is not a student's run, in two ways.** CI executes notebooks on x86 Linux against the
repository's own install, with `data/` linked in. Students have neither:

| | CI | student | what it missed |
|---|---|---|---|
| **environment** | editable install, the checkout's `data/` | the `course` branch, only what a cell downloads | tutorials 4 and 7 read `data/` a wheel lacks; tutorial 8 needs a checkpoint from another runtime |
| **hardware** | x86 Linux | Colab GPUs, and Apple Silicon on the lab's Macs | an op unimplemented on MPS made every decoder raise `NotImplementedError` |

For the first, run **`scripts/student_path.sh NOTEBOOK`** before shipping a notebook students open
from a badge: a fresh environment, the notebook's own install cell, Colab faked. It installs
what `course` holds, so to test a change before promoting it, run it after the change is on
`course` — or edit the install cell's `@course` locally to your branch name. The workflow
`student_path.yml` runs every notebook that way on each push to `course` and weekly, through
`execute_notebooks.py --as-student`; it tests what students are served, so it is not a PR check.
For the second there is no CI answer; a device-specific bug shows up on a Mac or in Colab or not
at all.

## Every training run checkpoints, and it is not a flag

Any script calling `train_model` passes a `TrainingCheckpointer`. Unconditionally — not behind
`--resumable`, not "when the run is long enough to be worth it." **A flag makes it optional, and
the run that skips it is always the one you could least afford to lose**; "short enough not to
bother" is judged before the run, which is exactly when you do not know.

Two needs, two mechanisms, and you want both:

| `save_dir` | keeps the best model, so the run leaves an artifact rather than only a number |
|---|---|
| `checkpointer` | makes the run resumable, so dying at hour three does not cost hours one and two |

A 36-epoch benchmark here ran 65 minutes and produced a BLEU figure and **no model**, so the
longer run that should have continued from it started from scratch. This is also the practice the
library exists to demonstrate: a script that teaches checkpointing while not doing it teaches the
opposite.

## Where things belong

Four files accumulate knowledge, and the wrong one is how things get lost.

| | holds | test |
|---|---|---|
| `notes/TASKS.md` | **only tasks that can be finished and removed** | could this row ever disappear? |
| `CLAUDE.md` | standing habits and practices | will this still be true next month? |
| `notes/reports/` | experimental outcomes | is this a measurement? |
| `notes/handoff/` | the conversation with the Cowork session | is this a message to someone? |

**A finished task is deleted, not marked done.** Marking it "Done" in place leaves the file
describing shipped work as pending. If it carries something durable, move that to the right file
*first* — git history keeps the rest.

Three things are **not** tasks and must not be filed as them: a standing habit, a watch item, and
a finding. A watch item in particular looks like a task and never completes.

The cost is not tidiness. A measurement of training budget beating data by roughly 7x sat inside a
task entry for days, where nobody would look — and it was the prior for a learning-curve
experiment later designed without it. It now lives in `notes/reports/training-budget.md`.

## Handing the baton

Two Claude sessions work on CS 479: this one, in the repository, and a Cowork session that owns
the course decks. They cannot message each other, so `notes/handoff/` is the channel. The protocol
— which file is the mailbox, how entries are written — is in `notes/README.md`.

Two rules belong here rather than there.

**First: a baton pass in either direction means reconciling `notes/TASKS.md` in the same sitting**,
and saying in the reply which tasks moved. A hand-off is when the lists go stale and the only
moment both sides know what changed. One baton return closed a task outright, made another moot,
and created five that were invisible from this side — including `grader.exe` having no source or
license, which the course had carried blind for a year. None of that is discoverable by reading
code.

**Second: an unanswered question gets re-raised in the next hand-off, not left in the old one** —
flagged as a repeat, because a question asked twice with no acknowledgement is a different signal
from one asked once. Each hand-off is read once, on arrival, so a question inherited by an archived
file is invisible thereafter.

**But first verify it really is unanswered, in the archive and not just in the newest file.** This
half is the more important one, because the original example for this rule was wrong: #148 was
described here as unanswered when Cowork had answered it with slide numbers and written "Close
#148". Their reply had moved to `archive/` when the files were split. The rule as first written had
a third ask drafted for a closed question. Three of their eight answers had gone unacted on for a
week by the same route, including a Lecture 6 split that was ours to do and had no task row.

So: **read the last replies against your own open questions, close what was answered, and only
then carry forward what genuinely was not.**
