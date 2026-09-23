# Training Checkpoints

Resumable training state, so a Colab disconnect costs minutes rather than the whole run.

## Two different kinds of checkpoint

TorchLingo has two modules with "checkpoint" in the name. They answer different
questions, and picking the wrong one is the easy mistake:

| | `torchlingo.checkpoint` | `torchlingo.training_checkpoint` |
| --- | --- | --- |
| Saves | a **finished model** for inference | **training state** for resuming |
| Contents | weights, architecture config, tokenizers | weights, optimizer, scheduler, epoch, step, loss history |
| Use when | you are done training and want a portable file | you are mid-training and might get interrupted |

A training checkpoint is not a substitute for the other: it does not carry the
tokenizers, so it is not portable to someone else's machine. Save both.

## Quick Start

```python
from torchlingo.training import train_model
from torchlingo.training_checkpoint import (
    TrainingCheckpointer, default_checkpoint_dir, mount_drive
)

mount_drive()   # mounts Google Drive in Colab; a no-op anywhere else

checkpointer = TrainingCheckpointer(
    "my-experiment",
    checkpoint_dir=default_checkpoint_dir("my-experiment"),
    save_every_seconds=300,
)

result = train_model(model, train_loader, val_loader,
                     num_epochs=20, checkpointer=checkpointer)
```

Run that cell again after a disconnect and training resumes from the last save.
There is no separate "resume" call: `train_model` checks for a checkpoint and
picks up from it when one exists.

!!! tip "In Colab, point it at Drive"
    `default_checkpoint_dir` does this for you: a folder under `MyDrive` when
    running in Colab, a local `checkpoints/` directory otherwise. A checkpoint
    written to the Colab runtime's own disk dies with the runtime, which defeats
    the purpose.

## What gets saved

Two files, and only two:

| File | Written |
| --- | --- |
| `latest.pt` | every automatic save, overwritten in place |
| `best.pt` | only when validation loss improves |

Keeping exactly two is deliberate. On Drive, a growing pile of numbered
checkpoints is how a student silently fills their quota and starts getting write
failures mid-run.

Saving the **optimizer** matters more than it looks. Adam carries per-parameter
moment estimates; resuming from weights alone throws those away and the loss
visibly jumps as the optimizer rebuilds them. The scheduler is saved for the same
reason.

## When it saves

- Every `save_every_seconds` (default 600), or every `save_every_steps` if you
  set that instead. Set either to `0` to disable that trigger.
- Always at the end of an epoch, regardless of the interval, because an epoch is
  the natural resume point and the cost is small relative to one.

## Failure behavior

Writes go to a temporary file and are then moved into place. A runtime that dies
mid-write cannot leave a half-written `latest.pt` — which would fail at load
time, exactly when the work it was protecting is already gone.

If a checkpoint cannot be loaded (truncated, or written by a different model
architecture), `train_model` reports it and starts fresh rather than raising.
Failing to resume should cost you the history, not the ability to train.

!!! warning "The Colab path is not covered by CI"
    GitHub's runners have no Google Drive to mount, so `is_colab`, `mount_drive`
    and the Drive-backed paths cannot be exercised automatically. Everything
    else here is tested, including save, resume, interval logic and corruption
    handling. The Drive integration has been written and reviewed carefully but
    has not been run in a live Colab session — if you use it there, please report
    what happens.

## API Reference

::: torchlingo.training_checkpoint
    options:
      show_source: true
      members:
        - TrainingCheckpointer
        - CheckpointState
        - is_colab
        - mount_drive
        - default_checkpoint_dir
