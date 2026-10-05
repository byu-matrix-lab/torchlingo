# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- **`CharVocab`, and `create_dataloaders(..., target_units="characters")`**: one target token
  per character, for languages written without spaces between words. The source side stays
  words. `use_sentencepiece=True` overrides `target_units`, so adding the SentencePiece
  settings to a character-level call switches it to subword pieces: Assignment 9's settings
  work unchanged on a character-level A8 notebook.
- **`SimpleVocab.build_vocab` warns when most of a corpus will become `<unk>`**
  (`MostlyUnknownWarning`): at least half of all word occurrences below `min_freq`, on 200
  sentences or more. The usual cause is a language written without spaces (Chinese, Japanese,
  Thai), where each sentence counts as one word and nearly none repeat; the trained model then
  writes little but `<unk>`. Two A8 students hit exactly that. The warning names the cause when
  sentences average under three words, and points at SentencePiece, `JiebaVocab` and
  `MeCabVocab`. English and Spanish corpora measure 24-28% at 200 sentences, well under the
  threshold.

### Changed
- **Decoding shows `<unk>` instead of dropping it.** `decode(..., skip_special_tokens=True)`, and
  so `translate_batch`, now removes only `<pad>`, `<sos>` and `<eos>`, in every vocabulary.
  `<unk>` is something the model wrote; dropping it made a model that writes nothing else print
  empty lines, which is how the two A8 students' failure looked. `SentencePieceVocab` shows it as
  `<unk>` rather than SentencePiece's `⁇`. **BLEU can move slightly** for a model that outputs
  `<unk>`: the token never matches a reference, and the hypothesis is longer. To compare with a
  score from an earlier version, re-score the same model; no retraining is needed.

## [0.2.5] - 2026-10-05

### Added
- **`capture_self_attention(encoder, average_heads=True)`**, the encoder's counterpart to
  `capture_cross_attention`: one self-attention map per encoder layer, heads averaged or kept
  separate. It needs a different mechanism, because in eval mode under `torch.no_grad()`
  PyTorch's encoder layers take a fused path that never calls `self_attn`, so a pre-hook repeats
  each layer's attention call on the same input. Tested exact against the slow path, pre- and
  post-norm, with padding.

### Fixed
- The visualization reference page now renders `capture_cross_attention`, which it linked to
  but never showed.

## [0.2.4] - 2026-09-30

### Fixed
- **`create_dataloaders(use_sentencepiece=True, sp_model_path=...)` uses that one model for both
  sides.** Without `sp_tgt_model_path` it used to load the target vocabulary from the configured
  default path, `data/sp_model.model`, and fail with a file-not-found error wherever that file
  did not exist, which is every Colab runtime. This is exactly Assignment 9's call.
  `sp_tgt_model_path` still selects a separate target model.
- **A resumed run now draws what an unbroken one would have.** Training checkpoints save the
  random number generators (torch, CUDA, Python's `random`, NumPy), so a run resumed at an epoch
  boundary gets the same data order and dropout masks, and on a CPU ends on identical weights.
  `load(..., restore_rng=False)` leaves the caller's generators alone.
- Training checkpoints save the mixed-precision `GradScaler`, so a resumed `use_amp` run keeps
  its loss scale instead of restarting it. Checkpoints from earlier versions still load.
- `SimpleSeq2SeqLSTM` with `num_layers=1` uses no recurrent dropout instead of accepting a rate
  PyTorch would ignore with a warning: one layer has no "between layers" to apply it.

## [0.2.3] - 2026-09-28

### Changed
- `train_sentencepiece` is quiet by default: SentencePiece's per-merge training log, thousands of
  lines at an 8,000-piece vocabulary, no longer fills a notebook cell. Warnings and errors still
  appear; `verbose=True` restores the full log.

## [0.2.2] - 2026-09-28

### Added
- `torchlingo.colab.setup(gpu=, drive=, data=)` prepares a notebook in one call: the device,
  the Google Drive mount, and downloads of repository files a pip install does not ship. It
  behaves the same in Colab and in a clone.
- `torchlingo.colab.fetch_data(paths)` downloads missing repository files, fetching Git LFS
  files from the LFS host by itself; it also replaces LFS pointer files a clone left behind.
- `preprocessing.split_exact(df, n_val, n_test, seed)` splits by exact counts, which
  `split_data`'s ratios cannot express.
- `diagnostics.padding_report(loader)` measures a loader's padded tokens against a plain
  shuffle, and the examples its batches drop.

### Fixed
- `evaluate_model` and `save_translations` are tested against the metric wrappers they
  aggregate; no behaviour changed.

## [0.2.1] - 2026-09-28

### Fixed
- **Resuming mid-epoch no longer skips the rest of that epoch.** A periodic checkpoint recorded
  the epoch in progress as complete, so a resumed run trained fewer steps and ended on the wrong
  learning rate. Checkpoints now carry `batches_into_epoch`; older checkpoints still load.
- A run with no `val_loader` now writes its epoch-boundary checkpoint, so its loss history
  survives a resume.
- `create_dataloaders` forwards its `Config` to the datasets it builds.
- One causal-mask convention throughout, which removes a PyTorch deprecation warning from decoding.

### Added
- `check_contamination(..., name="val")` names the held-out set in its report; the default is
  unchanged.
- `norm_first` on `SimpleTransformer` and `Config` for pre-norm Transformers. The default stays
  post-norm, the 2017 arrangement, and is now pinned by a test.
- chrF and TER results carry a sacreBLEU `.signature`, as BLEU already did.

### Changed
- Beam search's length penalty is documented as applying at final selection only, since
  pruning cannot see `alpha`.

## Earlier changes, never assigned to a release

This block was the `[Unreleased]` section until 0.2.1. Some of it shipped in 0.1.x and 0.2.0,
but this file was not updated when those versions were tagged, so which release carried which
change is not recorded here.

### Added
- py.typed marker file for PEP 561 compliance (enables type checking for downstream users)
- pytest configuration and migration from unittest
- mypy configuration in pyproject.toml
- pre-commit hooks configuration
- Python 3.13 support and classifier
- Comprehensive CHANGELOG.md
- **ML**: Standard sinusoidal positional encoding (Vaswani et al., 2017) for Transformer models
- **ML**: LSTM weight initialization with Xavier uniform (input-hidden) and orthogonal (hidden-hidden) initialization
- **ML**: Embedding dropout layer for Transformer models (standard 0.1 dropout rate)
- **ML**: Default learning rate scheduler with warmup and inverse square root decay (Transformer schedule)
- **ML**: Weight decay (L2 regularization) support with default value of 1e-4
- **ML**: Full training state checkpointing (optimizer, scheduler, epoch, losses)
- **ML**: Evaluation module with BLEU, chrF, and TER metrics using sacrebleu

### Changed
- **BREAKING**: Moved training scripts (train_ceb_cmn.py, train_ceb_cmn_simple.py, inference_ceb_cmn.py) to examples/
- **BREAKING**: Moved documentation guides (MULTILINGUAL_*.md, TESTING_GUIDE.md) to docs/docs/
- **BREAKING ML**: Replaced broken RoPE (Rotary Position Embeddings) with standard sinusoidal positional encoding
- **BREAKING ML**: Optimizer changed from Adam to AdamW with weight_decay parameter
- Fixed CI/CD runner from non-standard ubuntu-slim to ubuntu-latest
- Version management now uses single-source pattern via importlib.metadata
- MANIFEST.in now includes data files and py.typed marker
- Fixed .gitignore conflicting patterns for data directory
- Unified ruff version constraint to >=0.12 across all extras
- Removed ruff and black from docs extra (moved to dev only)
- Removed redundant pyarrow from CI install command
- Improved CI permissions with job-level scoping
- Updated docs workflow to use setup-python@v5
- **ML**: Checkpoints now save full training state (model, optimizer, scheduler) for proper resumption
- **ML**: Learning rate scheduler is now always active (defaults to Transformer schedule if not provided)

### Fixed
- Critical: CI/CD pipeline was broken due to invalid ubuntu-slim runner
- Critical: sdist packages were missing data files causing FileNotFoundError
- Version duplication between pyproject.toml and __init__.py
- .gitignore patterns that could prevent data files from being tracked
- **Critical ML**: RoPE was applied to embeddings instead of Q/K matrices, causing positional encoding to be lost
- **Critical ML**: No weight decay caused overfitting on small datasets
- **Critical ML**: Missing LSTM weight initialization led to suboptimal convergence
- **ML**: No default learning rate scheduler caused training instability without manual configuration
- **ML**: Checkpoints missing optimizer/scheduler state prevented proper training resumption

## [0.0.7] - 2024-02-13

### Added
- Multilingual training support
- Enhanced documentation with tutorials
- Example training scripts for Cebuano-Mandarin translation

### Changed
- Improved package structure and organization
- Updated dependencies

## [0.0.6] - 2024-02-XX

### Added
- Initial public release
- Core transformer and LSTM models
- SentencePiece tokenization support
- Data processing utilities
- TensorBoard integration
- Comprehensive documentation
- CI/CD pipeline for automated testing and publishing

### Features
- Educational-focused API design
- Support for multiple language pairs
- Back-translation for data augmentation
- Multilingual training capabilities

[Unreleased]: https://github.com/byu-matrix-lab/torchlingo/compare/v0.0.7...HEAD
[0.0.7]: https://github.com/byu-matrix-lab/torchlingo/compare/v0.0.6...v0.0.7
[0.0.6]: https://github.com/byu-matrix-lab/torchlingo/releases/tag/v0.0.6
