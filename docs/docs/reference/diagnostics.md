# Diagnostics

Checks that say **what** is wrong with a model, not merely that something is.

A model that does not work fails silently. Training runs, the loss curve looks like a
loss curve, translations come out, and every number is wrong. Nothing raises, because
none of it is an error in the sense Python understands — a frozen parameter and a
trained one have the same type.

So the checks have to be deliberate. Each answers one question against a stated
threshold and returns a [`CheckResult`](#torchlingo.diagnostics.CheckResult) rather than
printing, so it reads the same in a notebook, a script, or an assertion.

## The order matters

Each question is cheaper to answer than the one below it, and a failure at any level
makes everything below it meaningless. There is no point tuning a learning rate on a
corpus whose two sides do not match.

| # | Question | Check | Cost |
| - | -------- | ----- | ---- |
| 1 | Is the data what you think it is? | [`diagnose_alignment`](preprocessing/alignment.md) | seconds, no model |
| 2 | Is the model learning *anything*? | `check_loss_moved`, then `check_gradients` | one backward pass |
| 3 | Is it learning the *wrong* thing? | `check_generalization` | one training run |
| 4 | Is the measurement lying? | `check_contamination` | seconds |
| 5 | Is it the environment? | `check_eval_mode` | instant |

Question 1 lives in `torchlingo.preprocessing.alignment` rather than here, because it
needs no model at all.

Tutorial 6 demonstrates all five by breaking a working model on purpose and watching
each check fire.

## Quick start

```python
from torchlingo.diagnostics import (
    check_contamination,
    check_eval_mode,
    check_generalization,
    check_gradients,
    check_loss_moved,
)

result = train_model(model, train_loader, val_loader, num_epochs=20)

print(check_loss_moved(result.train_losses, vocab_size=len(vocab)))
print(check_generalization(result.train_losses, result.val_losses))
print(check_contamination(test_df.src, train_df.src))
print(check_eval_mode(model))
```

A `CheckResult` is truthy when it passed, so the same call works as a guard:

```python
assert check_eval_mode(model), "refusing to score a model in training mode"
```

## Reading a gradient report

`check_loss_moved` tells you the loss is flat. It cannot tell you *why*, and that is the
question you actually need answered.

[`gradient_report`](#torchlingo.diagnostics.gradient_report) runs one backward pass and
sorts every parameter into three buckets, each a different bug with a different fix:

| Bucket | Meaning | Usual cause |
| ------ | ------- | ----------- |
| **frozen** | `requires_grad=False` | a submodule was frozen, or the optimizer was built over a filtered parameter list |
| **dead** | gradient is `None` or all zeros | the graph was detached, or that output never reaches the loss |
| **live** | a nonzero gradient arrived | healthy |

```python
logits = model(src, tgt[:, :-1])
loss = criterion(logits.reshape(-1, logits.size(-1)), tgt[:, 1:].reshape(-1))

report = gradient_report(model, loss)
print(report.summary())
# live=42 frozen=26 dead=0; first frozen: transformer.encoder.layers.0.self_attn.in_proj_weight
```

This needs no training at all — it runs on a freshly built model in well under a second,
which makes it the cheapest way to tell "needs more epochs" apart from "these parameters
were never going to move."

!!! tip "All live, and still not learning?"
    Then gradients are being computed and thrown away. The bug is in the optimizer, not
    the model: check that `optimizer.step()` is actually called, and that the learning
    rate is not zero.

!!! warning "`gradient_report` consumes the graph"
    It calls `backward()` on the loss you hand it, so pass one you are not going to call
    `backward()` on yourself. It clears existing gradients first, so a stale `.grad` left
    over from an earlier step cannot make a dead parameter look live.

## Why these thresholds

The defaults are smoke-detector settings, chosen to sit well clear of both the healthy
and the broken case rather than to mark a precise boundary. On tutorial 6's task a
working model reaches a train/validation gap of −0.167 and a memorizing one +1.084, so
`DEFAULT_MAX_GAP` of 0.30 separates them with room on both sides.

Anything landing between the thresholds is worth looking at by hand. None of these
checks is proof: a model can pass all five and still be wrong.

!!! note "A healthy model usually scores *better* on validation"
    Dropout is active during training and disabled during validation, so the expected
    gap is slightly negative. `check_generalization` fires on the *sign* flipping and the
    gap widening, not on any gap at all.

## API Reference

::: torchlingo.diagnostics
    options:
      show_source: true
      members:
        - CheckResult
        - GradientReport
        - uniform_loss
        - check_loss_moved
        - gradient_report
        - check_gradients
        - check_generalization
        - check_contamination
        - check_eval_mode
