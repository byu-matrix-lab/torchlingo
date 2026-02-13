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
- License classifier for better PyPI metadata
- Comprehensive CHANGELOG.md

### Changed
- **BREAKING**: Moved training scripts (train_ceb_cmn.py, train_ceb_cmn_simple.py, inference_ceb_cmn.py) to examples/
- **BREAKING**: Moved documentation guides (MULTILINGUAL_*.md, TESTING_GUIDE.md) to docs/docs/
- Fixed CI/CD runner from non-standard ubuntu-slim to ubuntu-latest
- Version management now uses single-source pattern via importlib.metadata
- MANIFEST.in now includes data files and py.typed marker
- Fixed .gitignore conflicting patterns for data directory
- Unified ruff version constraint to >=0.12 across all extras
- Removed ruff and black from docs extra (moved to dev only)
- Removed redundant pyarrow from CI install command
- Improved CI permissions with job-level scoping
- Updated docs workflow to use setup-python@v5

### Fixed
- Critical: CI/CD pipeline was broken due to invalid ubuntu-slim runner
- Critical: sdist packages were missing data files causing FileNotFoundError
- Version duplication between pyproject.toml and __init__.py
- .gitignore patterns that could prevent data files from being tracked

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
