# Config

Configuration management for TorchLingo.

The `Config` class provides a centralized way to manage hyperparameters, file paths, and feature toggles. All TorchLingo functions accept an optional `config` parameter for customization.

## Quick Start

```python
from torchlingo.config import Config, get_default_config

# Use defaults
config = get_default_config()

# Customize
config = Config(
    batch_size=64,
    learning_rate=1e-4,
    d_model=512,
)

# Pass to functions
dataset = NMTDataset("data.tsv", config=config)
model = SimpleTransformer(..., config=config)
```

## API Reference

::: torchlingo.config.Config
    options:
      show_source: true
      members:
        - __init__
        - clone
        - to_dict
        - from_dict

::: torchlingo.config.get_default_config
    options:
      show_source: true

## Configuration Categories

### Directory Paths

| Parameter | Default | Description |
| --------- | ------- | ----------- |
| `data_dir` | `src/data` | Data files directory |
| `checkpoint_dir` | `src/checkpoints` | Model checkpoints |
| `output_dir` | `src/outputs` | Generated outputs |

### Data Settings

| Parameter | Default | Description |
| --------- | ------- | ----------- |
| `data_format` | `"tsv"` | File format: `tsv`, `csv`, `parquet`, `json`, `txt` |
| `src_col` | `"src"` | Source column name |
| `tgt_col` | `"tgt"` | Target column name |

### Vocabulary Settings

| Parameter | Default | Description |
| --------- | ------- | ----------- |
| `min_freq` | `2` | Minimum token frequency |
| `vocab_size` | `32000` | Target vocab size (SentencePiece) |
| `use_sentencepiece` | `False` | Use subword tokenization |

### Special Tokens

| Parameter | Default | Description |
| --------- | ------- | ----------- |
| `pad_token` | `"<pad>"` | Padding token |
| `unk_token` | `"<unk>"` | Unknown token |
| `sos_token` | `"<sos>"` | Start of sequence |
| `eos_token` | `"<eos>"` | End of sequence |
| `pad_idx` | `0` | Padding index |
| `unk_idx` | `1` | Unknown index |
| `sos_idx` | `2` | SOS index |
| `eos_idx` | `3` | EOS index |

### Model Architecture

| Parameter | Default | Description |
| --------- | ------- | ----------- |
| `d_model` | `512` | Transformer hidden dimension |
| `n_heads` | `8` | Attention heads |
| `num_encoder_layers` | `6` | Encoder depth |
| `num_decoder_layers` | `6` | Decoder depth |
| `d_ff` | `2048` | Feed-forward dimension |
| `dropout` | `0.1` | Dropout rate |

### LSTM Settings

| Parameter | Default | Description |
| --------- | ------- | ----------- |
| `lstm_emb_dim` | `256` | Embedding dimension |
| `lstm_hidden_dim` | `512` | Hidden dimension |
| `lstm_num_layers` | `2` | Number of layers |
| `lstm_dropout` | `0.1` | Dropout rate |

### Training Settings

| Parameter | Default | Description |
| --------- | ------- | ----------- |
| `batch_size` | `64` | Batch size |
| `learning_rate` | `1e-4` | Learning rate |
| `max_seq_length` | `128` | Maximum sequence length |
| `num_epochs` | `10` | Training epochs |
| `scheduler_type` | `"cosine"` | LR scheduler: `"cosine"`, `"plateau"`, `"transformer"`, `"noam"`, or `"none"` |
| `scheduler_patience` | `3` | Validations without improvement before `plateau` scheduler halves the LR |
| `warmup_steps` | `4000` | Warmup steps for `cosine`, `transformer`, and `noam` schedulers |

### Experiment Tracking (TensorBoard)

| Parameter | Default | Description |
| --------- | ------- | ----------- |
| `use_tensorboard` | `False` | Enable TensorBoard logging |
| `tensorboard_dir` | `./runs` | Directory for TensorBoard event files |
| `experiment_name` | `"baseline"` | Experiment name (becomes subdirectory in tensorboard_dir) |
| `log_interval` | `100` | Steps between logging metrics |
| `val_interval` | `1000` | Steps between validation checks |
| `save_interval` | `5000` | Steps between checkpoint saves |

## Examples

### Creating Custom Configs

```python
# Small model for testing
test_config = Config(
    d_model=64,
    n_heads=2,
    num_encoder_layers=1,
    num_decoder_layers=1,
    batch_size=8,
)

# Production model
prod_config = Config(
    d_model=512,
    n_heads=8,
    num_encoder_layers=6,
    num_decoder_layers=6,
    batch_size=64,
    learning_rate=5e-5,
)
```

### Cloning and Modifying

```python
base = get_default_config()
experiment = base.clone()
experiment.learning_rate = 1e-5
experiment.batch_size = 128
```

### Saving and Loading

```python
import json

# Save to JSON
config = Config(batch_size=32)
with open("config.json", "w") as f:
    json.dump(config.to_dict(), f)

# Load from JSON
with open("config.json", "r") as f:
    config = Config.from_dict(json.load(f))
```

## Module Constants

For maximum compatibility, module-level constants are also available:

```python
from torchlingo import config

print(config.BATCH_SIZE)      # Default batch size
print(config.DATA_DIR)        # Data directory path
print(config.PAD_IDX)         # Padding token index
```

!!! warning "Prefer Config Objects"
    Module constants are read-only. For customization, always use `Config` objects.
