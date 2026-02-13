# TorchLingo Testing Guide

This guide describes the comprehensive test suite for TorchLingo, including recent improvements to ensure robustness of model training and tokenization.

## Test Organization

Tests are located in the `tests/` directory and organized by functionality:

- **`test_config.py`**: Configuration system tests
- **`test_models.py`**: Model architecture tests (Transformer & LSTM)
- **`test_vocab.py`**: Vocabulary implementations (SimpleVocab, SentencePieceVocab, etc.)
- **`test_dataset.py`**: Dataset loading and preprocessing
- **`test_batching.py`**: Batch collation and padding
- **`test_preprocessing.py`**: Data preprocessing utilities
- **`test_sentencepiece.py`**: SentencePiece tokenization tests
- **`test_training_inference.py`**: Training loop and inference tests
- **`test_positional.py`**: Positional encoding tests
- **`test_integration_robust.py`**: ⭐ **NEW** - End-to-end integration tests
- **`test_model_convergence.py`**: ⭐ **NEW** - Model convergence and optimization tests

## Running Tests

### Run All Tests
```bash
cd torchlingo
python -m unittest discover tests -v
```

### Run Specific Test File
```bash
python -m unittest tests.test_integration_robust -v
```

### Run Specific Test Class
```bash
python -m unittest tests.test_integration_robust.TestEndToEndTransformerPipeline -v
```

### Run Single Test Case
```bash
python -m unittest tests.test_integration_robust.TestEndToEndTransformerPipeline.test_full_pipeline_with_simple_vocab -v
```

## Recent Improvements

### 1. Fixed Unicode Encoding Issue ✅

**Problem**: SentencePiece training failed on Windows when processing text with Unicode characters (accents, emojis, non-Latin scripts).

**Solution**: Fixed `train_sentencepiece()` in [src/torchlingo/preprocessing/sentencepiece.py](src/torchlingo/preprocessing/sentencepiece.py#L102) to explicitly use UTF-8 encoding when creating temporary files.

**Change**:
```python
# Before
with tempfile.NamedTemporaryFile(mode="w", delete=False) as tmp:

# After
with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", delete=False) as tmp:
```

**Impact**: SentencePiece now handles multilingual text correctly on all platforms.

## New Test Suites

### Integration Tests (`test_integration_robust.py`)

Comprehensive end-to-end tests covering the full pipeline:

#### `TestEndToEndTransformerPipeline`
- **`test_full_pipeline_with_simple_vocab`**: Complete workflow from raw data → vocabulary → dataset → training → inference with SimpleVocab
- **`test_full_pipeline_with_sentencepiece`**: Full pipeline with SentencePiece tokenization

#### `TestLSTMEndToEndPipeline`
- **`test_lstm_full_pipeline`**: LSTM model end-to-end workflow

#### `TestModelTrainingRobustness`
- **`test_training_with_validation_and_early_stopping`**: Validates early stopping based on validation loss
- **`test_training_handles_gradient_explosion`**: Ensures gradient clipping prevents NaN losses
- **`test_training_with_amp`**: Tests automatic mixed precision training

#### `TestSentencePieceEdgeCases`
- **`test_sentencepiece_handles_unicode`**: Unicode characters (café, 日本語, emoji 😀, greek α)
- **`test_sentencepiece_handles_empty_and_whitespace`**: Various whitespace scenarios
- **`test_sentencepiece_handles_very_long_sequences`**: Very long text (500+ words)

#### `TestInferenceRobustness`
- **`test_greedy_decode_batch_processing`**: Various batch sizes (1, 2, 5, 10)
- **`test_translate_batch_handles_varying_lengths`**: Sentences of different lengths

### Convergence Tests (`test_model_convergence.py`)

Tests that verify models actually learn:

#### `TestTransformerConvergence`
- **`test_transformer_overfits_small_dataset`**: Verifies model can overfit (proof of learning capacity)
- **`test_transformer_loss_decreases_monotonically_on_simple_task`**: Loss trends downward on copy task

#### `TestLSTMConvergence`
- **`test_lstm_overfits_small_dataset`**: LSTM learning capacity verification

#### `TestOptimizationComponents`
- **`test_different_optimizers_produce_different_results`**: Adam vs SGD behavior
- **`test_learning_rate_affects_convergence_speed`**: LR impact on training
- **`test_gradient_clipping_prevents_nan_loss`**: Gradient clipping effectiveness

#### `TestValidationAndEarlyStopping`
- **`test_early_stopping_triggers_on_no_improvement`**: Patience mechanism
- **`test_best_checkpoint_has_lowest_val_loss`**: Checkpoint saving correctness

#### `TestModelCapacity`
- **`test_larger_model_achieves_lower_loss`**: Model size impact on performance

## Test Coverage Summary

### Current Coverage (383 tests, 20 skipped)

- ✅ **Configuration System**: All parameters, config precedence pattern
- ✅ **Model Architectures**: Transformer & LSTM forward/backward passes
- ✅ **Vocabularies**: SimpleVocab, SentencePieceVocab, MeCab, Jieba
- ✅ **Data Loading**: Parallel text, TSV/CSV/JSON/Parquet formats
- ✅ **Preprocessing**: Tokenization, normalization, data splitting
- ✅ **Training**: Loss computation, gradient clipping, validation, early stopping
- ✅ **Inference**: Greedy decode, beam search, batch translation
- ✅ **Edge Cases**: Unicode, empty strings, very long sequences, varying batch sizes
- ✅ **Convergence**: Overfitting tests, loss decrease verification
- ✅ **Optimization**: Different optimizers, learning rates, AMP

### Skipped Tests (20)

Tests are skipped when optional dependencies are not installed:
- MeCab tests (require `fugashi` + `unidic-lite`)
- Jieba tests (require `jieba`)
- Some SentencePiece tests that require pre-trained models in `data/`

## Key Testing Patterns

### 1. Config Precedence Pattern
All tests verify the config override pattern works correctly:
```python
# Explicit param > passed config > default config
param = explicit_param if explicit_param is not None else cfg.param
```

### 2. Temporary Directories
Tests use `tempfile.TemporaryDirectory()` for isolation:
```python
with tempfile.TemporaryDirectory() as tmpdir:
    tmp = Path(tmpdir)
    # Test code using tmp
```

### 3. Learning Verification
Convergence tests verify actual learning:
```python
# Loss should decrease
self.assertLess(final_loss, initial_loss * 0.5)
```

### 4. Edge Case Coverage
Tests systematically cover edge cases:
- Empty inputs
- Very long sequences
- Unicode and special characters
- Various batch sizes
- Different data distributions

## Best Practices for Adding Tests

1. **Test Naming**: Use descriptive names that explain what's being tested
   ```python
   def test_transformer_overfits_small_dataset(self):
   ```

2. **Docstrings**: Include clear docstrings explaining the test purpose
   ```python
   """Verify Transformer can overfit a tiny dataset (proof of learning)."""
   ```

3. **Isolation**: Use temporary directories, don't rely on external state

4. **Assertions**: Use specific assertions with helpful messages
   ```python
   self.assertLess(loss, threshold, f"Loss {loss} exceeded threshold {threshold}")
   ```

5. **Edge Cases**: Think about boundary conditions and failure modes

6. **Integration**: Test components together, not just in isolation

## Common Issues and Solutions

### Issue: Tests fail with "ModuleNotFoundError: No module named 'torchlingo'"
**Solution**: Install package in development mode:
```bash
pip install -e .
```

### Issue: SentencePiece tests fail with Unicode errors on Windows
**Solution**: Already fixed! UTF-8 encoding is now explicitly specified.

### Issue: Tests are slow
**Solution**: Run specific test files or classes instead of the full suite during development.

### Issue: MeCab or Jieba tests are skipped
**Solution**: These are optional. To run them, install dependencies:
```bash
pip install torchlingo[japanese]  # For MeCab
pip install torchlingo[chinese]   # For Jieba
pip install torchlingo[asian]     # For both
```

## Continuous Testing Workflow

### Before Committing
```bash
# 1. Format code
ruff format src tests

# 2. Fix linting issues
ruff check --fix src tests

# 3. Run relevant tests
python -m unittest tests.test_models -v
python -m unittest tests.test_integration_robust -v

# 4. Run full suite (if time permits)
python -m unittest discover tests -v
```

### After Making Changes
- **Models**: Run `test_models.py` + `test_model_convergence.py`
- **Training**: Run `test_training_inference.py` + `test_model_convergence.py`
- **Preprocessing**: Run `test_preprocessing.py` + `test_sentencepiece.py`
- **Config**: Run `test_config.py` + relevant module tests
- **Integration**: Run `test_integration_robust.py`

## Test Metrics

After running the full suite, you should see:
```
Ran 383 tests in ~55s

OK (skipped=20)
```

This indicates:
- ✅ 363 tests passed
- ⏭️ 20 tests skipped (optional dependencies)
- ❌ 0 failures or errors

## Additional Resources

- **Test Coverage**: Consider using `coverage.py` to measure test coverage
  ```bash
  pip install coverage
  coverage run -m unittest discover tests
  coverage report
  coverage html  # Creates htmlcov/index.html
  ```

- **Parallel Testing**: Use `pytest-xdist` for faster test execution
  ```bash
  pip install pytest pytest-xdist
  pytest tests/ -n auto
  ```

## Contributing New Tests

When adding new features:

1. **Write tests first** (TDD approach)
2. **Cover happy path** - normal usage
3. **Cover edge cases** - empty, null, extreme values
4. **Cover error cases** - invalid inputs, missing data
5. **Add integration test** if feature spans multiple components
6. **Document** - add docstrings and update this guide if needed

---

**Test Status**: ✅ All 383 tests passing (as of latest run)

**Maintained by**: TorchLingo contributors
