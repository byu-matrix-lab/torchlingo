Follow these rules when editing, testing, and interacting with this repository.

## Environment & setup
- Use the repository virtual environment at `.venv` if present; otherwise create it at the repo root.
- macOS / zsh quick setup:
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]" || pip install -r requirements.txt || true
```
- Use the activated venv for every `python`/`pip` invocation.

## Coding conventions
- Naming:
  - Constants: UPPER_SNAKE_CASE (e.g., `DATA_DIR`, `VOCAB_SIZE`).
  - Classes: PascalCase (e.g., `Config`, `SimpleTransformer`).
  - Functions / variables: snake_case (e.g., `load_data`).
  - Private names: leading underscore.
- Type hints: annotate public functions and methods (use `Optional[...]`, `Union[...]` where needed).
- Docstrings: include Args, Returns, Raises, and Examples for public APIs in Google style. Keep examples runnable and concise.
- Imports: inside the package prefer relative imports (e.g., `from ..config import Config`).

## Config & call patterns
- `config.py` exposes module-level constants and a `Config` class. All functions should accept `config: Optional[Config] = None` as well as explicit parameters.
- Explicit function parameters take precedence over `cfg` values.

```python
cfg = config if config is not None else get_default_config()
param = explicit_param if explicit_param is not None else cfg.param
```

## Error handling & validation
- Validate inputs early and raise specific exceptions (`ValueError`, `FileNotFoundError`) with clear messages.

# Completing a Task
- ALWAYS complete the following steps after completing any code changes.

## Linting & formatting
- After edits, ALWAYS run `ruff` to check for style issues and format code:
```bash
source .venv/bin/activate
ruff check --fix src tests
ruff format src tests
```
## Testing & verification
- Tests use `unittest` and live under the `tests/` directory.
- Run tests with the venv active:
```bash
source .venv/bin/activate
python -m unittest discover tests
```
- Run a single module or test case:
```bash
python -m unittest tests.test_config -v
python -m unittest tests.test_preprocessing.TestLoadDataParallelFiles.test_load_parallel_txt_files -v
```
- After edits run the relevant tests (or full suite if core behavior changed) before finishing.

## Documentation
- After all code changes, ALWAYS update or add documentation as needed.
- Documentation is built with `mkdocs` and lives in the `docs/` directory.
- Search for usages of the edited functions/classes in the `docs/` folder and update as needed.
- Docs should be tailored to beginners with clear explanations and code examples.
- Review existing docs to understand style and structure before adding new content.
