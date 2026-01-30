# Checkpointing

TorchLingo provides built-in support for automatic checkpointing during training. This ensures students don't lose their training progress if their computer goes to sleep or the Colab runtime disconnects.

!!! tip "Perfect for Coursework"
    This feature is designed specifically for students running long training jobs. Your model, optimizer, and training state are automatically saved at regular intervals.

## Quick Start

### The Easiest Way: Config-Based Checkpointing

The simplest way to enable checkpointing is through the Config class. Just set `use_colab_checkpointing=True` and `train_model` handles everything automatically:

```python
from torchlingo.config import Config
from torchlingo.training import train_model

# Enable checkpointing in your config
cfg = Config(
    use_colab_checkpointing=True,
    experiment_name="hw3_translation",  # Used for checkpoint folder
    colab_checkpoint_interval_minutes=10,  # Save every 10 minutes
)

# train_model automatically creates the checkpointer
result = train_model(
    model=model,
    train_loader=train_loader,
    config=cfg,
    num_epochs=20
)
```

That's it! In Colab, your model will be saved to Google Drive. Locally, it saves to `./checkpoints/<experiment_name>/`.

### Manual Checkpointer

For more control, create a checkpointer explicitly:

```python
from torchlingo.checkpoint import LocalCheckpointer
from torchlingo.training import train_model

# Create a local checkpointer
checkpointer = LocalCheckpointer(
    experiment_name="hw3_translation",
    save_interval_minutes=10,
)

# Pass it to train_model
result = train_model(
    model=model,
    train_loader=train_loader,
    checkpointer=checkpointer,
    num_epochs=20
)
```

## Two Checkpointer Classes

TorchLingo provides two checkpointer classes:

### LocalCheckpointer

Saves checkpoints to a local directory. Works everywhere.

```python
from torchlingo.checkpoint import LocalCheckpointer

checkpointer = LocalCheckpointer(
    experiment_name="my_experiment",
    checkpoint_dir="./my_checkpoints",  # Optional, defaults to ./checkpoints/<name>
    save_interval_minutes=10,
)
```

### ColabCheckpointer

Extends `LocalCheckpointer` to save to Google Drive in Colab environments. Automatically mounts Drive.

```python
from torchlingo.checkpoint import ColabCheckpointer

# Only works in Google Colab!
checkpointer = ColabCheckpointer(
    experiment_name="hw3_translation",
    drive_path="My Drive/torchlingo_checkpoints",
)
```

!!! warning "Colab Only"
    `ColabCheckpointer` raises a `RuntimeError` if used outside Google Colab. Use `LocalCheckpointer` for local development.

### Automatic Selection

When you use `use_colab_checkpointing=True` in your config, TorchLingo automatically selects the right checkpointer:

- **In Colab** → Uses `ColabCheckpointer` (saves to Google Drive)
- **Locally** → Uses `LocalCheckpointer` (saves to local disk)

## Resuming Training

If your training is interrupted, you can easily resume:

```python
from torchlingo.checkpoint import LocalCheckpointer

checkpointer = LocalCheckpointer(experiment_name="hw3_translation")

# Check if a checkpoint exists
if checkpointer.has_checkpoint():
    state = checkpointer.load(model, optimizer, scheduler)
    print(f"Resumed from epoch {state.epoch}, step {state.global_step}")
```

With `colab_auto_resume=True` in your config, this happens automatically!

## Key Features

### Auto-Save Intervals

- **Time-based**: Save every N minutes (default: 10 minutes)
- **Step-based**: Save every N training steps (disabled by default)

```python
checkpointer = LocalCheckpointer(
    experiment_name="my_exp",
    save_interval_minutes=5,   # Save every 5 minutes
    save_interval_steps=1000,  # Also save every 1000 steps
)
```

### Multiple Checkpoint Versions

By default, keeps 3 recent step checkpoints plus:
- `checkpoint_latest.pt` - Most recent save
- `checkpoint_best.pt` - Best validation loss

### Drive Space Warnings

The checkpointer warns you if disk space is low:

```
⚠ LOW DISK SPACE WARNING!
  Available: 123.4 MB
  Recommended minimum: 500.0 MB
  Checkpoints may fail to save. Free up space on your Drive!
```

TorchLingo provides two methods for checking drive space:

**Fast Method (default)** - Uses `shutil.disk_usage()` to check the mounted filesystem. This is fast and suitable for repeated checks during training loops.

```python
from torchlingo.checkpoint import get_drive_free_space

# Returns bytes, works on any path
free_bytes = get_drive_free_space(Path("./checkpoints"))
print(f"Free: {free_bytes / (2**30):.2f} GB")
```

**API Method (most accurate)** - Queries Google's servers directly for your exact account quota. This accounts for storage shared across Drive, Gmail, and Photos.

```python
from torchlingo.checkpoint import get_true_drive_free_space

# Returns GB, only works in Colab
free_gb = get_true_drive_free_space()
if free_gb == float('inf'):
    print("Unlimited storage!")
elif free_gb is not None:
    print(f"Exactly {free_gb:.2f} GB free")
```

!!! tip "Which method to use?"
    - Use the **fast method** (default) for repeated checks during training
    - Use the **API method** at notebook startup to verify account state before heavy work

### Full Training State

Checkpoints include everything needed to resume:
- Model weights
- Optimizer state (momentum, etc.)
- LR scheduler state
- Training metrics (losses, epoch, step)

## Configuration Options

### LocalCheckpointer Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `experiment_name` | Required | Unique name for your experiment |
| `checkpoint_dir` | `./checkpoints/<name>` | Directory for checkpoint files |
| `save_interval_minutes` | `10.0` | Auto-save interval in minutes |
| `save_interval_steps` | `0` | Auto-save every N steps (0=disabled) |
| `keep_n_checkpoints` | `3` | Number of recent checkpoints to keep |
| `save_optimizer` | `True` | Save optimizer state |
| `save_scheduler` | `True` | Save scheduler state |
| `verbose` | `True` | Print status messages |

### ColabCheckpointer Additional Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `drive_path` | `"My Drive/torchlingo_checkpoints"` | Path within Google Drive |
| `mount_point` | `"/content/drive"` | Where to mount Drive |
| `auto_mount` | `True` | Auto-mount Google Drive |

### Config Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `use_colab_checkpointing` | `False` | Enable auto-checkpointing |
| `colab_checkpoint_interval_minutes` | `10.0` | Save interval in minutes |
| `colab_checkpoint_interval_steps` | `0` | Save interval in steps |
| `colab_drive_path` | `"My Drive/torchlingo_checkpoints"` | Drive path |
| `colab_keep_n_checkpoints` | `3` | Checkpoints to keep |
| `colab_auto_resume` | `True` | Auto-resume from checkpoint |

## Detailed Examples

### Manual Checkpoint Management

For custom training loops:

```python
from torchlingo.checkpoint import LocalCheckpointer

checkpointer = LocalCheckpointer(
    experiment_name="en_to_spanish",
    save_interval_steps=1000,
)

for epoch in range(num_epochs):
    for step, batch in enumerate(train_loader):
        loss = train_step(model, batch, optimizer)
        
        # step_callback checks if it's time to save
        checkpointer.step_callback(
            model=model,
            optimizer=optimizer,
            step=global_step,
            epoch=epoch,
            metrics={"train_loss": loss.item()}
        )
        
        global_step += 1
```

### Loading Specific Checkpoints

```python
# Load the most recent checkpoint
state = checkpointer.load(model, optimizer)

# Load the best checkpoint (lowest validation loss)
state = checkpointer.load(model, optimizer, checkpoint_type="best")

# Load a specific step
state = checkpointer.load(model, optimizer, checkpoint_type="5000")

# Load from explicit path
state = checkpointer.load(model, checkpoint_path="./my_checkpoint.pt")
```

### Quick One-Liner Saves

For simple cases:

```python
from torchlingo.checkpoint import save_checkpoint_locally

# Quick save to local disk
path = save_checkpoint_locally(model, "my_experiment", epoch=5)
```

## Checkpoint Directory Structure

```
checkpoints/
└── hw3_translation/
    ├── checkpoint_latest.pt    # Most recent
    ├── checkpoint_best.pt      # Best validation loss
    ├── checkpoint_step_00001000.pt
    ├── checkpoint_step_00002000.pt
    └── checkpoint_step_00003000.pt
```

## Tips for Students

!!! info "Recommended Settings for Coursework"
    ```python
    cfg = Config(
        use_colab_checkpointing=True,
        experiment_name="hw3_<your_name>",  # Make it unique!
        colab_checkpoint_interval_minutes=5,  # Save often
        colab_auto_resume=True,
    )
    ```

### Best Practices

1. **Use descriptive experiment names**: Include assignment number and your name
2. **Save frequently**: 5-10 minutes is good for Colab
3. **Enable auto_resume**: Set `colab_auto_resume=True` in config
4. **Test locally first**: The checkpointer works locally too

### Troubleshooting

**"RuntimeError: ColabCheckpointer requires Google Colab"**: Use `LocalCheckpointer` instead, or use config-based setup which auto-selects.

**"Google Drive not mounted"**: The checkpointer will prompt you to authorize. Follow the link.

**"No checkpoint found"**: Ensure you're using the same `experiment_name`.

**"LOW DISK SPACE WARNING"**: Free up space on your Google Drive.

## API Reference

::: torchlingo.checkpoint.LocalCheckpointer
    options:
      members:
        - save
        - load
        - has_checkpoint
        - step_callback
        - should_save
        - get_state
        - update_state

::: torchlingo.checkpoint.ColabCheckpointer

::: torchlingo.checkpoint.CheckpointState

::: torchlingo.checkpoint.create_checkpointer_from_config

::: torchlingo.checkpoint.is_colab

::: torchlingo.checkpoint.get_drive_free_space

::: torchlingo.checkpoint.get_true_drive_free_space

::: torchlingo.checkpoint.save_checkpoint_locally

::: torchlingo.checkpoint.save_checkpoint_to_drive
