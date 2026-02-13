# Contributing to TorchLingo

Thank you for your interest in contributing to TorchLingo! We welcome contributions from the community.

## Quick Start

1. **Fork** the repository on GitHub
2. **Clone** your fork: `git clone https://github.com/YOUR-USERNAME/torchlingo.git`
3. **Install** for development: `pip install -e ".[dev,docs]"`
4. **Create a branch**: `git checkout -b my-feature`
5. **Make your changes** and add tests
6. **Run tests**: `pytest tests/`
7. **Format code**: `ruff format src/ tests/`
8. **Submit** a pull request

## Development Setup

```bash
# Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install with all development dependencies
pip install -e ".[dev,docs,asian]"

# Install pre-commit hooks (optional but recommended)
pre-commit install

# Run tests
pytest tests/

# Format and lint code
ruff format src/ tests/
ruff check src/ tests/ --fix
```

## Ways to Contribute

- 🐛 **Report bugs** via [GitHub Issues](https://github.com/byu-matrix-lab/torchlingo/issues)
- 💡 **Suggest features** via [GitHub Discussions](https://github.com/byu-matrix-lab/torchlingo/discussions)
- 📖 **Improve documentation** — typos, clarifications, examples
- 🧪 **Add tests** — increase coverage
- 🚀 **Submit code** — bug fixes, new features

## Code Style

- We use [Ruff](https://docs.astral.sh/ruff/) for linting and formatting
- Follow PEP 8 conventions
- Add type hints to public APIs
- Write docstrings in Google style
- Keep line length to 88 characters

## Testing

- All new features must include tests
- Maintain or improve test coverage
- Tests use pytest framework
- Run the test suite before submitting: `pytest tests/`

## Documentation

- Update docstrings for any modified functions/classes
- Add examples to docstrings when helpful
- Update relevant documentation in `docs/`
- Check that docs build successfully: `mkdocs serve -f docs/mkdocs.yml`

## Full Contributing Guide

For detailed information about our development workflow, code conventions, and more, please see our **[Full Contributing Guide](https://byu-matrix-lab.github.io/torchlingo/contributing/)**.

## Questions?

- Check our [documentation](https://byu-matrix-lab.github.io/torchlingo/)
- Ask in [GitHub Discussions](https://github.com/byu-matrix-lab/torchlingo/discussions)
- Open an [issue](https://github.com/byu-matrix-lab/torchlingo/issues)

## License

By contributing to TorchLingo, you agree that your contributions will be licensed under the AGPL-3.0-or-later license.
