# Training

`train_model` runs the training loop, and `TrainResult` is what it hands back.

For what the loop does at each step (forward pass, loss, backward pass, optimizer step) and why,
read [Training Loop](../concepts/training.md) first. This page is the API.

## A run, as the A8 notebook does it

```python
from torchlingo.training import train_model
from torchlingo.training_checkpoint import TrainingCheckpointer

checkpointer = TrainingCheckpointer(experiment_name="assignment-8")   # Drive, in Colab

result = train_model(
    model,
    train_loader,
    val_loader=val_loader,
    num_epochs=EPOCHS,
    device=device,
    config=config,
    gradient_clip=1.0,
    save_dir=OUT_DIR / "best",
    log_every=50,
    checkpointer=checkpointer,
)

print(result.train_losses[-1], result.val_losses[-1])
print(result.best_checkpoint)          # a file under OUT_DIR / "best"
```

## Two things to pass, for two different needs

| argument | what it gives you |
| --- | --- |
| `save_dir` | the best model so far, as one file [`load_model`](checkpoint.md) reloads in a new session |
| `checkpointer` | a run that resumes where it stopped after a dropped session, rather than starting again |

Pass both. A run with only `save_dir` leaves a model but starts over after a disconnect; a run
with only `checkpointer` resumes but leaves no single file to translate with. See
[Training Checkpoints](training_checkpoint.md) for how resuming works.

## Settings that are arguments, not `Config` fields

Most settings come from the `config` you pass, but a few belong to the call:

- **`gradient_clip`** clips the gradient norm at that value. It defaults to `None`, meaning no
  clipping, whatever `Config.grad_clip` says: `train_model` does not read that field. The course
  notebooks pass `gradient_clip=1.0`.
- **`device`** defaults to the GPU when there is one, else the CPU. Pass it explicitly to use
  anything else, an Apple GPU (`"mps"`) for instance.
- **`log_every`** prints a `Train Loss` line, the mean loss since the last one, every that many
  batches. Left at `0`, it uses `config.log_interval`, 100 by default.

## What comes back

`TrainResult.train_losses` and `TrainResult.val_losses` hold one mean loss per epoch, so either
can be plotted against the epoch number. `best_checkpoint` is the path of the model file written
under `save_dir`, or `None` when no `save_dir` was given.

## API

::: torchlingo.training
    options:
      show_source: true
      members:
        - train_model
        - TrainResult
