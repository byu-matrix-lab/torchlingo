# Contributing to TorchLingo

Thank you for your interest in contributing to TorchLingo! This guide will help you get started.

## Ways to Contribute

- 🐛 **Report bugs** via GitHub issues
- 💡 **Suggest features** via GitHub discussions
- 📖 **Improve documentation** — typos, clarifications, examples
- 🧪 **Add tests** — increase coverage
- 🚀 **Submit code** — bug fixes, new features

## Development Setup

### 1. Fork and Clone

```bash
git clone https://github.com/commanderjcc/torchlingo.git
cd torchlingo
```

### 2. Create a Virtual Environment

```bash
python -m venv .venv
source .venv/bin/activate  # macOS/Linux
# or: .venv\Scripts\activate  # Windows
```

### 3. Install in Development Mode

```bash
pip install -e ".[dev,test,docs]"
```

### 4. Run Tests

```bash
pytest tests/ -v
```

## Code Style

We use **Ruff** for linting and formatting:

```bash
# Check style
ruff check src/

# Auto-fix issues
ruff check --fix src/

# Format code
ruff format src/
```

### Style Guidelines

- **Line length**: 88 characters max
- **Docstrings**: Google style
- **Type hints**: Required for public APIs
- **Imports**: Sorted with isort

```python
# Good example
def encode(self, sentence: str, add_special_tokens: bool = True) -> List[int]:
    """Encode a sentence to vocabulary indices.

    Args:
        sentence: Raw text to encode.
        add_special_tokens: Whether to add SOS/EOS tokens.

    Returns:
        List of integer indices.

    Raises:
        ValueError: If sentence is empty.
    """
    ...
```

## Running Tests

```bash
# All tests
pytest

# Specific file
pytest tests/test_vocab.py

# With coverage
pytest --cov=torchlingo --cov-report=html
```

### Writing Tests

Tests live in `tests/` and use pytest:

```python
# tests/test_vocab.py
import pytest
from torchlingo.data_processing import SimpleVocab


class TestSimpleVocab:
    def test_build_vocab(self):
        vocab = SimpleVocab()
        vocab.build_vocab(["hello world", "hello friend"])
        
        assert "hello" in vocab.token2idx
        assert len(vocab) >= 4  # At least special tokens

    def test_encode_decode(self):
        vocab = SimpleVocab()
        vocab.build_vocab(["hello world"])
        
        indices = vocab.encode("hello world", add_special_tokens=True)
        text = vocab.decode(indices, skip_special_tokens=True)
        
        assert text == "hello world"
```

## Documentation

### Building Docs Locally

```bash
cd docs
pip install -r requirements.txt
mkdocs serve
```

Visit `http://localhost:8000` to preview.

### Documentation Guidelines

- Use **clear, simple language** — imagine explaining to a beginner
- Include **code examples** for every concept
- Add **type hints** to all public functions
- Use **admonitions** for tips, warnings, notes

```markdown
!!! tip "Helpful Tip"
    Use this format for tips.

!!! warning "Watch Out"
    Use this for common mistakes.
```

## Submitting Changes

### 1. Create a Branch

```bash
git checkout -b feature/your-feature-name
```

### 2. Make Your Changes

- Keep commits focused and atomic
- Write clear commit messages
- Add tests for new functionality
- Update documentation if needed

### 3. Push and Create PR

```bash
git push origin feature/your-feature-name
```

Then create a Pull Request on GitHub.

### PR Guidelines

- **Title**: Clear, descriptive (e.g., "Add beam search decoding")
- **Description**: Explain what and why
- **Tests**: Include tests for new code
- **Docs**: Update docs if behavior changes

## Issue Guidelines

### Bug Reports

Include:

1. TorchLingo version
2. Python version
3. Minimal code to reproduce
4. Expected vs actual behavior
5. Full error traceback

### Feature Requests

Include:

1. What problem does this solve?
2. Proposed solution
3. Alternatives considered

## Project Structure

```
torchlingo/
├── src/torchlingo/          # Main package
│   ├── config.py            # Configuration
│   ├── data_processing/     # Datasets, vocab, batching
│   ├── models/              # Neural network architectures
│   └── preprocessing/       # Data loading, tokenization
├── tests/                   # Test suite
├── docs/                    # Documentation (MkDocs)
└── assignments/             # Course assignments
```

## Questions?

- Open a GitHub Discussion for questions
- Check existing issues before creating new ones
- Join our community (if applicable)

Thank you for helping make TorchLingo better! 🎉
