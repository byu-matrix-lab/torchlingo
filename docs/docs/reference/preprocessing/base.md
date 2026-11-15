# Base Utilities

Core data loading, saving, and splitting utilities for neural machine translation.

## Overview

This module provides foundational functions for working with parallel corpora:

- Load data from TSV, CSV, JSON, and Parquet formats
- Convert parallel text files to DataFrames
- Save processed data
- Split datasets into train/val/test sets
- Read and write plain text one sentence per line, and tell whether a language is written
  without spaces

## API Reference

::: torchlingo.preprocessing.base.read_lines
    options:
      show_source: true

::: torchlingo.preprocessing.base.write_lines
    options:
      show_source: true

::: torchlingo.preprocessing.base.written_without_spaces
    options:
      show_source: true

::: torchlingo.preprocessing.base.load_data
    options:
      show_source: true

::: torchlingo.preprocessing.base.save_data
    options:
      show_source: true

::: torchlingo.preprocessing.base.parallel_txt_to_dataframe
    options:
      show_source: true

::: torchlingo.preprocessing.base.split_data
    options:
      show_source: true

::: torchlingo.preprocessing.base.split_exact
    options:
      show_source: true



## Examples

### Loading Data

```python
from torchlingo.preprocessing import load_data

# Auto-detect format from extension
df = load_data("data/train.tsv")

# Explicit format
df = load_data("data/train.csv", format="csv")

# Supported formats
df_tsv = load_data("data.tsv")        # Tab-separated
df_csv = load_data("data.csv")        # Comma-separated
df_json = load_data("data.json")      # JSON Lines
df_parquet = load_data("data.parquet") # Parquet
```

### Converting Parallel Files

Many NMT datasets come as separate source and target files:

```
english.txt:
  Hello world
  Good morning

spanish.txt:
  Hola mundo
  Buenos días
```

Convert them to a DataFrame:

```python
from torchlingo.preprocessing import parallel_txt_to_dataframe

df = parallel_txt_to_dataframe(
    src_path="english.txt",
    tgt_path="spanish.txt",
)

print(df)
#            src            tgt
# 0  Hello world     Hola mundo
# 1  Good morning  Buenos días
```

### Custom Column Names

```python
df = parallel_txt_to_dataframe(
    src_path="english.txt",
    tgt_path="spanish.txt",
    src_col="english",
    tgt_col="spanish",
)
```

### Saving Data

```python
from torchlingo.preprocessing import save_data

# Save as TSV (recommended)
save_data(df, "output/train.tsv")

# Save as other formats
save_data(df, "output/train.csv")
save_data(df, "output/train.json")
save_data(df, "output/train.parquet")
```

### Splitting Data

```python
from torchlingo.preprocessing import split_exact

train_df, val_df, test_df = split_exact(
    df,
    n_val=2000,   # validation pairs
    n_test=2000,  # test pairs; everything else is training data
    seed=479,     # the same seed gives the same test set every time
)

print(f"Train: {len(train_df)}")  # len(df) - 4000
print(f"Val: {len(val_df)}")      # 2000
print(f"Test: {len(test_df)}")    # 2000
```

`split_exact` is what the A8 notebook uses: assignments name held-out sets by count. For
proportions instead, `split_data(df, train_ratio=0.8, val_ratio=0.1, seed=42)` splits by
ratio, with the rest going to test.

## Format Details

### TSV (Tab-Separated Values)

**Recommended for text data** because tabs rarely appear in natural text.

```tsv
src	tgt
Hello world	Hola mundo
How are you?	¿Cómo estás?
```

### CSV (Comma-Separated Values)

Requires quoting if text contains commas:

```csv
src,tgt
"Hello, world","Hola, mundo"
How are you?,¿Cómo estás?
```

### JSON Lines

One JSON object per line, good for complex metadata:

```json
{"src": "Hello world", "tgt": "Hola mundo"}
{"src": "Good morning", "tgt": "Buenos días"}
```

### Parquet

Binary columnar format:

- Efficient compression
- Fast loading for large files
- Not human-readable

```python
# Parquet is great for large datasets
save_data(df, "data/large_corpus.parquet")
df = load_data("data/large_corpus.parquet")  # Fast!
```

## Error Handling

### Mismatched Line Counts

```python
# english.txt has 1000 lines, spanish.txt has 999 lines
df = parallel_txt_to_dataframe("english.txt", "spanish.txt")
# Raises: ValueError: Parallel files have different number of lines
```

### Missing Columns

```python
df = load_data("data_without_src_column.tsv")
# Later: ValueError when used with NMTDataset
```

### Unsupported Format

```python
df = load_data("data.xml")
# Raises: ValueError: Unsupported format: xml
```

## Best Practices

1. **Use TSV for text**: Tabs are rare in natural language
2. **Always shuffle before splitting**: Ensures representative splits
3. **Check data after loading**: Verify row counts and sample content
4. **Use Parquet for large data**: Faster loading, smaller files

```python
# Good workflow
from torchlingo.preprocessing import load_data

df = load_data("data/corpus.tsv")
print(f"Loaded {len(df)} rows")
print(df.head())  # Inspect samples
print(df.isnull().sum())  # Check for missing values
```
