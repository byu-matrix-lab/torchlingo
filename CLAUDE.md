# CLAUDE.md

Guidance for Claude Code working in this repository.

## Project overview

**TorchLingo exists first and foremost to support CS 479.** Eric's framing, 2026-09-26.
Third-party use of the library is a later concern.

That is a tie-breaker, not a slogan: when a choice could serve either a CS 479 student or a
general newcomer, it serves the student, and a gap only matters if a lecture or an assignment
walks into it. **`notes/CS479_COURSE_ROADMAP.md` is the single source of truth** for what the
course needs, and where that judgement gets made. It is generated: edit its sources in
`notes/roadmap/` (facts in `roadmap.yaml`, prose in `sections/`) and run
`scripts/render_roadmap.py --write`, never the Markdown itself.

**Nothing else restates the roadmap's facts.** Eric, 2026-10-26: *"the readme should not
duplicate the roadmap."* A lecture's title or date, or an assignment's due date, time, points
or window, is written in `roadmap.yaml` and pointed to from everywhere else: not copied into a
README, a table in `notes/TASKS.md`, a hand-off or a notebook. Two hand-kept copies had already
drifted that day: the transcripts README in torchlingo-private and the lecture legend in
`notes/TASKS.md` still had A12 and A13 at their Oct 22 dates. Where a copy must exist, a check
compares it. The course transcripts' `Due:` lines and Total rows are one such copy, checked by
`torchlingo-private/scripts/check_transcripts.py` at each hand-off.

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
scripts/merge_check.sh PR            # preflight on a PR merged with main, before merging it
scripts/land_pr.sh PR [--promote]    # merge_check, merge, compare open PRs, clean up, promote
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

**Two decoders.** `inference.py` holds the reference decoders, the readable ones a student
follows line by line, and they are the **specification**: `inference_fast.py`, the batched
path, must give token-identical output on every fixture in `tests/test_decoding_equivalence.py`
(Aug 22). Speed goes in the fast module, never folded into the reference. The reference warns,
once per process, on a large input, naming `inference_fast.translate_batch`. Batching across
sentences, KV caching and multi-GPU training were descoped (Eric, Sep 22): each is complex,
teaches little, and is done better by CTranslate2, Marian or Joey NMT
(`docs/docs/related-work.md`). Task #243 is to look again at two points against this: the
reference module imports the fast one in `translate_to_file`, and Task #272 proposes batching
across sentences.

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
scripts/preflight.sh --quick    # ~1s:  the five static gates only
scripts/preflight.sh --all      # also executes the runnable notebooks (minutes)
```

It runs exactly what CI runs, cheapest gate first, stopping at the first failure: `ruff check`,
`ruff format --check`, `render_report.py --check`, `render_roadmap.py --check`,
`notebook_meta.py --check`, the suite under **both** pytest and `unittest discover`,
`--doctest-modules`, `mkdocs build --strict`, and optionally `execute_notebooks.py`.

**In a git worktree, run preflight, not the gates by hand.** The venv's editable install points
at the main checkout, so a bare `python -m pytest` in a worktree tests the main checkout's
`torchlingo`, not the branch's: on 2026-10-14 a branch's new module failed to import in its own
tests, and a library change can as easily pass against the old code. Preflight puts the tree's
own `src` first on `PYTHONPATH`, uses the main checkout's `.venv` when the worktree has none,
and stops if `torchlingo` still imports from anywhere else. Running a gate by hand there, set
`PYTHONPATH=$PWD/src` first.

**CI gates eight things across four jobs, and running them from memory means running most of
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
- **Land a PR with `scripts/land_pr.sh PR`**, which does all of this in order: `merge_check.sh`,
  the merge without `--delete-branch`, the open-PR comparison (stopping before any deletion if
  another PR closed), then the worktree, only if it is clean, and the branch. With `--promote`
  it then runs `promote_course.sh --await` and waits for the student-path run on that commit.
  `--admin` goes only with Eric's explicit OK for that PR, or on a notes-only PR.
- **Landing several PRs together, promote once, with the last** (Eric, 2026-10-19):
  `scripts/land_pr.sh 334 335 336 --promote` lands them in that order, each checked against
  the `main` the one before left, and promotes only the last. That is one student-path run,
  about four minutes since it runs a job per notebook (Task #262), instead of one per PR. If it fails, the cause is one of the batch, not exactly one PR; with
  small PRs that is quickly found. On 2026-10-19 three PRs promoted one at a time spent three
  runs back to back. Add `--no-wait` to skip waiting for that run; promotion has happened by
  then, and the script prints the command that checks it. Each push to `main` cancels the CI
  run on the commit before it, so a batch leaves one run per workflow, not one per merge.

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
build the merge result locally: **`scripts/merge_check.sh PR`** does the latter in a scratch
worktree and runs preflight on it, ending `CLEAN`, `CONFLICT` (with the files) or
`PREFLIGHT-FAIL`. When several PRs merge in a row, run it on each just before merging, against
the `main` the previous merge left: on 2026-10-14 that caught a generated-roadmap conflict and a
`CHANGELOG.md` conflict, each one merge after its PR's own CI had gone green.

### Check which branch you are on before a destructive git command

`git checkout <b> || git checkout -b <b>` can fail *both* ways — a branch held by another
worktree cannot be checked out, and `-b` then fails because it exists — and a following
`git reset --hard` runs anyway, against whatever branch you were standing on. That silently moved
local `main` onto another branch's commits here. Put `git branch --show-current` between the
checkout and anything destructive, and read it.

## `docs/docs/course/` belongs to this repository

The CS 479 course notebooks live here, and **both sessions write them, together**. Eric,
2026-10-13: *"both Claude Code agent and cowork agent can collaborate on these, as we're doing."*
Neither side owns a notebook: whoever holds the baton edits, and the hand-off says what changed
and what is asked of the other side. What each brings differs. Cowork knows the decks, Learning
Suite, the assignment texts and the course folder's data; this side knows the library and can
run and test the notebooks. So a notebook built here asks Cowork for what only the course side
knows, and one Cowork edits is reviewed here against the library. Cowork does **not** run git:
committing, the checks, the nav entry, the pull request and promotion to `course` happen here.
Each side goes as far as it can before handing off; see "Handing the baton".

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
- **Every tutorial opens the same way** (Task #217, 2026-10-13): `# Tutorial N: Title` in title
  case; the badge; `**Reading. About N minutes.**` (or `Reference`); two to four sentences on
  the question and why now; **"By the end you will be able to:"** followed by numbered
  *abilities*, not "you will have seen"; one line, "These serve Lecture X's outcome *...*",
  quoting the roadmap's lecture outcome; then the Colab line. Copy tutorial 8's first cell.
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

**Every notebook has its own specific purpose and learning outcome, which no other notebook
shares, whatever its type.** Eric, 2026-10-12: *"students ignore the appearance of
redundancy."* A notebook that looks like one they have already seen gets skipped, including the
part of it that was new. So the test is the appearance, not only the content: the same corpus,
the same model or the same opening as another notebook reads as a repeat even when the lesson
differs.

The two kinds still differ in job. A course notebook (`docs/docs/course/`) is where a student
does the assignment: dated, tied to a lecture, often on their own data; only five of eleven use
TorchLingo. A tutorial (`docs/docs/tutorials/`) is where they understand a mechanism, on fixed
data with committed outputs, and it doubles as the library's documentation. When two notebooks
teach the same thing, one keeps it and the other links: the course notebook if an assignment
needs it in class, the tutorial otherwise. The case that prompted this: tutorial 3 and
`lecture-07-toy-model` both trained the same twelve-phrase model. Lecture 7 keeps the loss
curve, the held-out test and resuming a run; the A8 notebook keeps `save_dir` and reloading; and
tutorial 3 became what a checkpoint does not hold, the vocabulary, and the two ways a rebuild
goes wrong. Its first rewrite, making a run survive, repeated Lecture 7's Part B: check the
course notebooks themselves, not only their look, before giving a tutorial its purpose.

**A course notebook stands on its own; it never assumes a tutorial was read.** Eric,
2026-10-12: *"some students don't 'read the textbook' or do the tutorials, so the Lecture
notebooks can't assume all of the lessons from the tutorial notebooks."* So dependency runs one
way. A course notebook may assume earlier course notebooks and the lectures, never a tutorial:
anything it needs a student to understand, it explains at first use, in a sentence or two, and a
link to a tutorial is for depth, not a prerequisite. A tutorial may assume the course notebooks.
This is also what decides the tie in the rule above: a lesson an assignment needs lives in the
course notebook. Eric, 2026-10-25, stronger still: **tutorials carry no points**, so students
pressed for time are likely to skip them. So **the assignment notebooks are load-bearing**, and
an assignment notebook and a tutorial must not look alike. When they share material, the
assignment keeps it and the tutorial is given a purpose of its own.

**An application of deep learning, not a deep-learning class.** Eric, 2026-10-25: *"the students
should certainly be aware of and informed of the DL ingredients while staying focused on the
translation problem."* So the machinery (the training loop, the optimizer, mixed precision,
accumulation) runs in the library, and the notebook's markdown names each ingredient where it
matters and says in a sentence what it does. What the student writes and decides is about
translation: the data, the tags or codes, the budget, the hypothesis, the comparison and what it
means. Eric, the same day: *"we want the students to continue making decisions and being mindful
throughout the whole notebook's process. This isn't a coding class, but writing some amount of
code, with clear directions is valuable for the learning process."* A long stretch of cells a
student only runs is a place to ask for a prediction, a choice or a reading of the result.

**Wrap the plumbing; keep the lesson inline.** Eric, 2026-09-28: large code cells lose a new
student. So code a student learns nothing from reading (installs, downloads, Drive, file
loading, exact splits, loader construction, padding arithmetic) belongs in the library, and a
notebook calls it. Code that *is* the lesson stays written out: tutorial 8's beam search, 8a's
length cap, dedupe and contamination check. Data fixtures stay visible too. Any long cell a
student only runs — plumbing that serves only that notebook, such as tutorial 6's toy-corpus
`Vocab`, or a training or scoring loop — opens with `# Setup: run this cell, no need to read it.`
(Eric, 2026-09-29).
Before writing a helper, check the library has not got one already: on the day this rule was
written, `create_dataloaders`, `parallel_txt_to_dataframe` and `evaluate_model` all existed and
no course notebook used them.

**House patterns for a course notebook's cells**, from the 2026-10-13 audit of A7 to A13;
`lecture-12-llm-context` and `lecture-13-back-translation` show them all.

- **Instructions to the student live in markdown**, not only in code comments, which a student
  running cells does not read.
- **A prediction gets its own code cell, before the reveal or the long run it predicts, and stops
  while empty**, or it is skipped and written after the fact:

  ```python
  PREDICTION = ""   # <- your prediction, in a sentence or two
  assert PREDICTION.strip(), "Write your prediction in PREDICTION, then run this cell again."
  ```

  The write-up then asks the student to reconcile it with the result.
- **A TODO's specification lives in markdown; its docstring stays short** and does not give the
  recipe, or the docstring is the answer.
- **An exercise's check runs on a case whose answer the student can work out**, and its failure
  message shows what their code returned (`got ...`), so a failure says what to fix.
- **The write-up asks about the lecture's conceptual outcome, not only its numbers** (Eric,
  2026-10-13). A write-up of numbers alone can be filled in without the idea.
- **Whatever a later step or the write-up needs survives a dropped Colab session.** Save it to
  Drive, e.g. a small JSON beside the outputs (A13's `a13-notes.json`), and have the write-up cell
  read it from there, so it runs after the setup cell alone.
- **Every GPU notebook tells students to disconnect when done** (Runtime → Disconnect and delete
  runtime) **and when not to** (while a training or decode cell runs) (Eric, 2026-10-13): an idle
  GPU runtime spends the student's Colab quota. `notebook_meta.py --check` enforces the first
  half for any course notebook calling `setup(gpu=True)`.

**Every notebook declares what it needs from its environment, in `requires`:** one of `pip`,
`download`, `colab`, `hf-token`, `blanks`, `exercises`. A closed set, because a typo would otherwise read as
"runnable in CI" — the opposite of what whoever wrote it meant. `needs` is repo-relative *paths*;
`requires` is *capabilities*; they are not interchangeable.

**A notebook declaring nothing gets executed in CI**, and one declaring `colab` does not. For a
`colab` notebook, write a test with `tests/notebook_cells.py`. It runs the notebook's own cells
in-process with Drive faked and the GPU requirement lifted, and shrinks sizes only by
substitutions that must match. `tests/test_lecture_09_part_b.py` is the example: Part B's first
run found a `NameError` that students would have hit the next day.
`lecture-10-comet-install` shipped `else:` followed by an unindented
`drive` — a bare SyntaxError in Lecture 10's own assignment notebook, unnoticed because nothing
executed this directory. A student would have hit it in the room.

So code that does not parse must declare `blanks`, and a `blanks` declaration must correspond to
real blanks — checked both ways, or the marker rots into a licence to ship broken cells. Note that
`!pip install` can appear *inside* a `try` block, so a plain `compile()` flags notebooks that run
perfectly well. Code that parses but stops until a student acts — an exercise that raises
`NotImplementedError`, a prediction that stops while empty — declares `exercises`, so CI does not
execute it unanswered, and gets a `tests/notebook_cells.py` test that fills the answers in and
checks that each stop fires (`tests/test_lecture_13.py`).

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

**Before changing an install cell, run `scripts/student_path.sh --colab --install-only
NOTEBOOK`.** `--colab` builds the environment on Colab's Python, with the versions of the
packages Colab ships that an install can collide with, read fresh from googlecolab/backend-info;
`--install-only` runs only the install cells, the notebook's imports and a COMET score, so it
covers notebooks a full run skips, A10, A11 and A13 among them. It fails if an install replaces
Colab's numpy or torch, and lists everything else it replaced. In October 2026 a COMET install
downgraded numpy and broke Colab for students, while a laptop's plain student path passed it.
The workflow runs this on every notebook too, in a second job.

**After editing a notebook, by any tool, run `python scripts/notebook_meta.py --normalize
NOTEBOOK`.** Every notebook is stored in one form: literal UTF-8, one-space indent, a final
newline, sources as line lists, no cell ids in an nbformat 4.0 notebook, and keys in the order
the file has them. `--normalize` rewrites the named notebooks into it, refreshes the purpose
cells, regenerates the map and checks; `--check` fails on a notebook that is not in it. Not an
`nbformat` re-save, which sorts every key and turned each one-cell edit into a diff across the
file (until 2026-10-14 that re-save, then `--purpose`, `--write` and `--check`, was the
sequence after every edit). One trap is left that no command fixes: NotebookEdit's cell ids
are synthetic, by index, so an insert shifts every later id. Edit bottom-up.

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

**So `notes/TASKS.md` holds the table and the tasks' details, and nothing else** (Eric,
2026-10-26: *"TASKS.md should only contain completable tasks. CLAUDE.md is for standing
rules"*). Until that day it opened with a page of dated log, one paragraph per baton pass, and
carried the decoders' design decisions and the rules below. The log is git history and
`notes/handoff/`; the rest is here.

## The task list

**The session's task list mirrors only what is urgent** (Eric, 2026-09-25). A task is on it
only if it contributes to one of the next two lectures, or is being worked now. `notes/TASKS.md`
always has every task; the mirror is a filter over it, not a second copy. Course work (decks,
Learning Suite, the assignment pages, anything waiting on Cowork or Eric for a course decision)
is in `torchlingo-private/notes/course/TASKS.md`, under the same numbers.

**"Critical for" says which lecture a task is for** (Eric, 2026-09-27). It is assigned by what
the task touches, not by how near its date is, and the test is *"must this be done for that
lecture to happen correctly?"*, not "is this related to it".

| label | means |
|---|---|
| **L1** to **L13**, with **L8a** to **L8c** | that lecture |
| **L4/L5**, **L8a/8b** | both of those lectures |
| **14+** | Lecture 14 onward |
| **lib** | library, tooling or docs work **no lecture depends on** |
| **hyg** | roadmap and process integrity: **critical, but not to one lecture** |

A lecture that has already run still owns its work: `L6` means "this belongs to Lecture 6",
not "overdue", and an instructor preparing the next offering meets it there. `hyg` is not a
synonym for "later": it keeps the other labels true, since a drifted map or a gate that only
reports misleads every lecture at once.

**P1 to P3 say how soon; the table is sorted by them** (Eric, Oct 25):

- **P1: a student meets it within about three weeks.** Something students run or are graded on
  in that window is broken, missing, or unverified on the path they will take; or a dated
  promise to Eric or Cowork falls inside the window.
- **P2: it makes this term's assignment notebooks load-bearing and correct, or guards what
  students are served.** A lesson an assignment leaves to a tutorial; a decision, prediction or
  interpretation missing from a coming assignment; a standing rule that shipped code breaks;
  protection for `main` and `course`; or something Eric asked for, with no date.
- **P3: everything else.** Library refactors that change nothing students see this term,
  tutorials (required reading, but they carry no points), lectures that have already run, the
  next offering, and housekeeping.

Within a level, the earlier date comes first. Re-rank when the calendar moves. A task whose
work sits in an open pull request says "In review" and keeps its row until the PR merges.

## Handing the baton

Two Claude sessions work on CS 479: this one, in the repository, and a Cowork session that owns
the course decks. They cannot message each other, so `notes/handoff/` is the channel. The protocol
— which file is the mailbox, how entries are written — is in `notes/README.md`.

Three rules belong here rather than there.

**First: go as far as you can before handing off.** Eric, 2026-10-13: *"Each agent should go as
far as it can before handing off so that the number of hand-offs are minimized for any given
task or notebook."* A hand-off costs a round trip of hours, so one pass should carry a piece of
work as far as this side can take it, and stop only where it genuinely needs the other side.
Concretely:

- **Build around the gap, not up to it.** If a notebook waits on Cowork's data, build everything
  that does not depend on it (a built-in sample, the analysis, the checks) and leave one marked
  seam, so the answer slots in without another pass. The Lecture 12 notebook is the example: its
  loader waits on the A12 files; nothing else does.
- **Ask everything at once.** Collect every question a task will need answered before sending,
  including the ones for its next step, rather than one question per pass.
- **Finish what you received.** When the baton arrives with an edit to review, fix what you can
  fix yourself rather than sending it back with a note, and say what you changed.
- **Hand off for what only the other side can do**: Cowork for the decks, Learning Suite, the
  assignment texts and the course folder; this side for git, running notebooks and the library.
  Not for review of something either side could have checked itself.

**Second: a baton pass in either direction means reconciling `notes/TASKS.md` in the same sitting**,
and saying in the reply which tasks moved. A hand-off is when the lists go stale and the only
moment both sides know what changed. One baton return closed a task outright, made another moot,
and created five that were invisible from this side — including `grader.exe` having no source or
license, which the course had carried blind for a year. None of that is discoverable by reading
code.

**Third: an unanswered question gets re-raised in the next hand-off, not left in the old one** —
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
