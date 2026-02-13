# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

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
