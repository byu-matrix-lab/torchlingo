# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

TorchLingo is an educational PyTorch library for Neural Machine Translation (NMT), designed for students and instructors. It provides a clean, well-documented implementation of Transformer and LSTM architectures for learning and experimentation.

## Environment Setup

**Virtual Environment**
- Use the repository virtual environment at `.venv` if present; otherwise create it at the repo root
- macOS / zsh / Linux setup:
  ```bash
  python3 -m venv .venv
  source .venv/bin/activate
  pip install -e ".[dev]"
  ```
- Windows setup:
  ```bash
  python -m venv .venv
  .venv\Scripts\activate
  pip install -e ".[dev]"
  ```
- Always use the activated venv for every `python`/`pip` invocation

## Common Development Commands

**Linting & Formatting** (ALWAYS run after code changes):
```bash
ruff check --fix src tests
ruff format src tests
```

**Testing**:
```bash
# Run all tests
python -m unittest discover tests

# Run specific test module
python -m unittest tests.test_config -v

# Run single test case
python -m unittest tests.test_preprocessing.TestLoadDataParallelFiles.test_load_parallel_txt_files -v
```

**Documentation**:
```bash
# Serve docs locally (script in scripts/)
./scripts/serve_docs.sh

# Build documentation
mkdocs build
```

**Building Package**:
```bash
# Build wheel (script in scripts/)
./scripts/build_wheel.sh

# Or manually
python -m build
```

## Code Architecture

### Module Organization

The codebase is organized into focused modules under `src/torchlingo/`:

- **`config.py`**: Central configuration system with `Config` dataclass and module-level constants
- **`models/`**: Neural network architectures
  - `transformer_simple.py`: Transformer encoder-decoder with sinusoidal positional encoding
  - `lstm_simple.py`: LSTM seq2seq baseline
  - `positional.py`: Sinusoidal positional encoding (Vaswani et al., 2017)
- **`data_processing/`**: Dataset and vocabulary management
  - `dataset.py`: `NMTDataset` for loading parallel corpora
  - `vocab.py`: `BaseVocab`, `SimpleVocab`, `SentencePieceVocab` implementations
  - `batching.py`: Collate functions for DataLoader
- **`preprocessing/`**: Data preprocessing and tokenization
  - `base.py`: Core data loading utilities (`load_data`, `save_data`, `parallel_txt_to_dataframe`)
  - `sentencepiece.py`: SentencePiece model training and tokenization
  - `multilingual.py`: Multilingual data handling and back-translation
- **`training.py`**: High-level training loop with `train_model()`, `TrainResult`, optional TensorBoard logging
- **`inference.py`**: Decoding utilities (`greedy_decode`, `beam_search_decode`, `translate_batch`)

### Key Design Patterns

**Config Pattern**: All functions follow this pattern for configuration:
```python
def some_function(param1, param2=None, config: Optional[Config] = None):
    cfg = config if config is not None else get_default_config()
    param2 = param2 if param2 is not None else cfg.param2
    # ... use cfg and param2
```
Explicit function parameters ALWAYS take precedence over config values.

**Model Interface**:
- Transformer models expose `encode()` and `decode()` methods
- LSTM models expose `src_embed`, `encoder`, `decoder`, `output` modules
- Both support the same `forward(src, tgt)` signature for training
- Inference functions detect model type via `hasattr()` checks

**Vocabulary Interface**:
- All vocabs inherit from `BaseVocab` protocol
- Must implement: `encode(text, add_special_tokens=True)` → `List[int]`
- Must implement: `decode(ids, skip_special_tokens=True)` → `str`
- Special token indices (`pad_idx`, `sos_idx`, `eos_idx`) can be stored as vocab attributes

**Data Flow**:
1. Raw parallel text → `load_data()` or `parallel_txt_to_dataframe()` → pandas DataFrame
2. DataFrame → `NMTDataset(data_file, src_vocab, tgt_vocab)` → PyTorch Dataset
3. Dataset → `DataLoader` with custom collate function → batched tensors
4. Training: `train_model(model, train_loader, val_loader)` → `TrainResult`
5. Inference: `translate_batch(model, sentences, src_vocab, tgt_vocab)` → translated strings

### Special Features

**SentencePiece Integration**:
- Train models with `train_sentencepiece(input_files, model_prefix, vocab_size)`
- Load with `SentencePieceVocab(model_path)`
- Special tokens are embedded in the SentencePiece model during training

**TensorBoard Support**:
- Enabled via `config.use_tensorboard = True`
- Logs to `config.tensorboard_dir / config.experiment_name`
- Tracks train/val loss, learning rate, per-step and per-epoch metrics

**Multilingual Training**:
- `preprocessing.multilingual` provides utilities for multi-parallel corpora
- Language pair identification and filtering
- Back-translation data augmentation support

## Coding Conventions

**Naming**:
- Constants: `UPPER_SNAKE_CASE` (e.g., `DATA_DIR`, `VOCAB_SIZE`)
- Classes: `PascalCase` (e.g., `Config`, `SimpleTransformer`)
- Functions/variables: `snake_case` (e.g., `load_data`)
- Private names: leading underscore (`_resolve_device`)

**Type Hints**: Annotate public functions and methods (use `Optional[...]`, `Union[...]` where needed)

**Docstrings**: Google style with Args, Returns, Raises, and Examples for public APIs. Keep examples runnable and concise.

**Imports**: Inside the package prefer relative imports (e.g., `from ..config import Config`)

## Workflow After Code Changes

ALWAYS complete these steps after making code changes:

1. **Format and Lint**:
   ```bash
   ruff check --fix src tests
   ruff format src tests
   ```

2. **Run Tests**:
   ```bash
   # Run relevant tests or full suite if core behavior changed
   python -m unittest discover tests
   ```

3. **Update Documentation** (if needed):
   - Search for usages of edited functions/classes in `docs/` folder
   - Update or add documentation as needed
   - Docs should be tailored to beginners with clear explanations and code examples

## Pull Requests

### Say "PR #X" and "Task #Y", never a bare `#N`

The two numbering schemes overlap almost completely — tasks run to #62, pull
requests to #42, so every number below 43 names one of each. Write **"PR #37"**
for a pull request and **"Task #37"** for a task-list item, in prose, commit
messages and GitHub comments alike.

This is not pedantry. Task #37 ("stacked PRs fight the stale-review rule") was
retired in the same breath as PR #37 (the `val_losses` fix) was listed as open
and awaiting review, and both were called "#37". On GitHub there is a second
reason: a bare `#N` in a comment auto-links to the pull request of that number,
so an unqualified task reference silently becomes a wrong link.

Task *subjects* keep the bare `#N` prefix. This is about how they are referred
to, not how they are titled.

### Do not stack pull requests

Every PR targets `main`. If a change depends on
work that is not merged yet, wait for it to merge rather than opening a PR whose
base is another PR.

This is a decision made on evidence, not taste. Stacked PRs have failed in three
distinct ways in this repository:

| | |
|---|---|
| Auto-close | PRs #11 and #12 closed when the base branch was deleted on merge |
| Lost approvals | a rebase that changed no content dismissed the approval on PRs #27 and #34 |
| Silent close | PR #26 closed unmerged during an unrelated merge, cause never established |

Each cost real time to recover from. GitHub has no rebase exemption for
`dismiss_stale_reviews_on_push`, so there is no configuration that makes the
pattern safe here.

**Corrected 2026-09-26.** This passage used to claim that every one of those
failures "is specific to a PR whose base is another PR". That is no longer
true, and the correction matters more than the original claim did.

**PR #84 closed itself while based on `main`.** It went OPEN to CLOSED, never
merged and with no merge commit, at 17:42:16Z — one second after the unrelated
PR #86 was squash-merged at 17:42:15Z with `--admin --delete-branch`. The two
touched different files. It was caught only by the open-set comparison below,
and recovered in full with `gh pr reopen`.

So the silent close is **not** stacking-specific, and PR #26 and PR #84 are two
instances of one unexplained mechanism. Both closed within a second of an
unrelated merge that passed `--delete-branch`. A later notes merge without that
flag closed nothing, which is one data point each way rather than a finding.

Two consequences:

- **Do not read "my PR targets `main`" as "my PR is safe."** Compare the open set
  after every merge regardless of what anything is based on.
- Prefer merging **without** `--delete-branch`, deleting the branch as a separate
  step afterwards, until the mechanism is understood. The cost is one command;
  the failure it may avoid cost a day to notice the first time.

The stacking argument still stands on its own two remaining legs — auto-close and
lost approvals are both genuinely specific to stacked PRs.

What to do instead:

- Merge promptly. Most stacks in this repo existed because something sat waiting
  for review, not because the work genuinely had to be sequenced.
- If work truly depends on unmerged work, keep it on a local branch and open the
  PR once the dependency lands.
- If a change is large, split it by *concern* into independent PRs against
  `main`, not into a chain.

If you inherit a stack that already exists, **unwind it in this order**:

1. Merge the base PR **without** `--delete-branch`.
2. Retarget the dependents to `main`.
3. *Then* delete the branch.

Deleting first auto-closes the dependents, which is how #11 and #12 were lost.

### Check the open-PR set after every merge

After a merge, the set of open PRs should be exactly what it was, minus the one
merged. Compare it.

This costs one command and catches a failure that took a day to notice: PR #26 was
closed unmerged one second after an unrelated merge, its base branch was never
deleted, and the cause was never established. A check is worth more than a
diagnosis when the mechanism is unknown.

**It has now paid for itself.** On 2026-09-26 the same thing happened to PR #84,
one second after PR #86's unrelated merge, and the comparison is the only reason
anyone noticed: before was `#86 #85 #84 #59 #57 #55 #54 #52 #51`, after was the
same set minus **both** #86 and #84. Reopening took one command because the loss
was caught immediately. PR #84 was based on `main`, so do not skip this check on
the grounds that nothing is stacked.

### Retargeting a PR dismisses its approvals

Changing a PR's base with `gh pr edit <N> --base main` flips it from `APPROVED`
to `REVIEW_REQUIRED` immediately. No commit, no push, no content change — just
the base pointer moving. Nothing in the UI warns you first.

This is separate from the push-dismissal above, and it was learned the expensive
way: unwinding one stack cost three approvals an hour after a reviewer had
worked through nine PRs in a sitting.

So **retarget before asking for review, never after**. If a retarget is
unavoidable on an approved PR, say so when you ask for the re-review, and
confirm the content is unchanged so the reviewer can trust their earlier
judgement rather than repeat it.

### A green PR is green against the base it last saw

Status checks record "passed against `main` as it was when they ran". They are
not re-run because `main` moved, and GitHub will still show `MERGEABLE / CLEAN`
alongside a wall of green ticks.

That gap is not theoretical. PR #17 carried ten green checks and merged cleanly
on paper, but `main` had since grown a `--doctest-modules` gate that its new
module failed — the checks predated the gate's existence. Merging on that green
would have turned `main` red, which is the same merge-order interaction that
broke `main` once before: a file and the gate that runs it arriving from
different branches, each green alone.

**Before merging any PR more than a few days old**, merge `main` into it and let
CI re-run, or build the merge result locally and test it. Do not treat an old
green tick as evidence about today's `main`.

### Check which branch you are on before a destructive git command

`git checkout <branch> || git checkout -b <branch>` can fail *both* ways — a
branch held by another worktree cannot be checked out, and `-b` then fails
because it already exists. A following `git reset --hard origin/<something>`
will run anyway, against whatever branch you were standing on.

That sequence silently moved local `main` onto another branch's commits here.
Nothing reached the remote, but it went unnoticed for several commands. Put
`git branch --show-current` between the checkout and anything destructive, and
read it.

## Project Goals

This library prioritizes:
- **Educational clarity**: Clean, readable code designed for learning
- **Documentation**: Comprehensive docs with runnable examples
- **Simplicity**: Avoid over-engineering; focus on core NMT concepts
- **Accessibility**: Beginner-friendly with Google-style docstrings
