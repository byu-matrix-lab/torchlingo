"""Checks that say *what* is wrong with a model, not merely that something is.

A model that does not work fails silently. Training runs, the loss curve looks
like a loss curve, translations come out, and every number is wrong. Nothing
raises, because nothing here is an error in the sense Python understands —
a frozen parameter and a trained one have the same type.

So the checks have to be deliberate. Each one in this module answers a single
question with a stated threshold, and returns a :class:`CheckResult` rather
than printing, so it can be asserted in a script as easily as read in a
notebook.

**Use them in order.** Each question is cheaper to answer than the one below
it, and a failure at any level makes everything below it meaningless — there is
no point tuning a learning rate on a corpus whose two sides do not match:

1. **Is the data what you think it is?** — not here; see
   :func:`torchlingo.preprocessing.alignment.diagnose_alignment`, which needs no
   model at all.
2. **Is the model learning anything?** — :func:`check_loss_moved`, then
   :func:`check_gradients` to find out which parameters could not.
3. **Is it learning the wrong thing?** — :func:`check_generalization`.
4. **Is the measurement lying?** — :func:`check_contamination`.
5. **Is it the environment?** — :func:`check_eval_mode`.

Of these, :func:`gradient_report` is the one with no cheaper substitute. The
others tell you that something is wrong; it names the parameters that were
never going to move, needs no training, and runs in well under a second.

Worked examples of all five, each demonstrated by breaking a working model on
purpose, are in tutorial 6.

Example:
    The usual shape. ``CheckResult`` is truthy when the check passed, so it
    reads naturally in an ``assert`` or an ``if``:

    >>> import torch
    >>> model = torch.nn.Linear(4, 4)
    >>> bool(check_eval_mode(model))
    False
    >>> _ = model.eval()
    >>> bool(check_eval_mode(model))
    True
"""

import math
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field

import torch
from torch import nn

# Defaults chosen to sit well clear of both the healthy and the broken case on
# the tutorial 6 task, where a working model reaches a train/validation gap of
# -0.167 and a memorizing one +1.084. They are smoke-detector thresholds, not
# precise boundaries: anything landing between is worth looking at by hand.
DEFAULT_MIN_DROP = 0.05
DEFAULT_MAX_GAP = 0.30


@dataclass(frozen=True)
class CheckResult:
    """The outcome of one diagnostic.

    Attributes:
        name (str): What was checked, as a short phrase.
        passed (bool): Whether it cleared its threshold.
        detail (str): The measurements behind the verdict, including the
            threshold, so the number is never reported without its standard.
    """

    name: str
    passed: bool
    detail: str

    def __bool__(self) -> bool:
        """Report whether the check passed, so the result reads as a condition.

        Returns:
            bool: ``self.passed``.
        """
        return self.passed

    def __str__(self) -> str:
        """Render as one line, tagged PASS or FAIL.

        Returns:
            str: A line suitable for printing in a notebook or a log.
        """
        return f"[{'PASS' if self.passed else 'FAIL'}] {self.name} — {self.detail}"


@dataclass(frozen=True)
class GradientReport:
    """Every parameter of a model, sorted by whether it can learn.

    The three buckets are distinct failures with distinct fixes, which is the
    reason for separating them rather than reporting a single count:

    - **frozen** — ``requires_grad=False``. Someone froze a submodule, or built
      the optimizer over a filtered parameter list and the filter was wrong.
    - **dead** — gradient is ``None`` or all zeros. The graph was detached, or
      that output never reaches the loss.
    - **live** — a nonzero gradient arrived. Healthy.

    A model can be entirely live and still not learn, which is the point of
    keeping this separate from :func:`check_loss_moved`: if every parameter is
    live and the loss is flat, gradients are being computed and then discarded,
    so the bug is in the optimizer rather than the model.

    Attributes:
        frozen (list[str]): Parameter names with ``requires_grad=False``.
        dead (list[str]): Parameter names whose gradient is absent or all zero.
        live (list[str]): Parameter names that received a nonzero gradient.
    """

    frozen: list[str] = field(default_factory=list)
    dead: list[str] = field(default_factory=list)
    live: list[str] = field(default_factory=list)

    def all_live(self) -> bool:
        """Report whether every parameter received a nonzero gradient.

        Returns:
            bool: True when nothing is frozen and nothing is dead.
        """
        return not self.frozen and not self.dead

    def summary(self) -> str:
        """Describe the three buckets, naming the first casualty in each.

        Returns:
            str: Counts, plus one example name per non-empty failure bucket —
                enough to start looking without printing hundreds of names.
        """
        parts = [
            f"live={len(self.live)} frozen={len(self.frozen)} dead={len(self.dead)}"
        ]
        if self.frozen:
            parts.append(f"first frozen: {self.frozen[0]}")
        if self.dead:
            parts.append(f"first dead: {self.dead[0]}")
        return "; ".join(parts)


def uniform_loss(vocab_size: int) -> float:
    """Return the cross-entropy of guessing uniformly over the vocabulary.

    This is the number that means "learned nothing": ``ln(V)``. A training loss
    sitting there, flat, is not a slow model but a disconnected one.

    It is a floor for ignorance rather than an exact prediction. A freshly
    initialized model usually scores slightly *above* it, because its random
    logits are confidently wrong about some tokens, which costs more than
    spreading the probability evenly.

    Args:
        vocab_size (int): Number of types the model chooses between.

    Returns:
        float: ``ln(vocab_size)``.

    Raises:
        ValueError: If ``vocab_size`` is not at least 1.

    Example:
        >>> round(uniform_loss(125), 3)
        4.828
    """
    if vocab_size < 1:
        raise ValueError(f"vocab_size must be >= 1, got {vocab_size}")
    return math.log(vocab_size)


def check_loss_moved(
    train_losses: Sequence[float],
    vocab_size: int | None = None,
    min_drop: float = DEFAULT_MIN_DROP,
) -> CheckResult:
    """Check that the training loss fell at all.

    The weakest possible question about training, and worth asking first
    precisely because it is weak: it separates "this model is learning slowly"
    from "this model is not connected to its optimizer."

    Args:
        train_losses (Sequence[float]): Training loss per epoch, in order.
        vocab_size (int | None): If given, the report also quotes
            :func:`uniform_loss` so the loss can be read against what guessing
            would score.
        min_drop (float): Required decrease from first epoch to last.

    Returns:
        CheckResult: Passing when ``first - last >= min_drop``.

    Raises:
        ValueError: If ``train_losses`` is empty.

    Example:
        A learning rate of zero, the most common version of this bug:

        >>> bool(check_loss_moved([5.341, 5.339, 5.337], vocab_size=125))
        False

        And a run that worked:

        >>> bool(check_loss_moved([5.223, 2.053, 0.265]))
        True
    """
    if not train_losses:
        raise ValueError("train_losses is empty; nothing to check")

    first, last = train_losses[0], train_losses[-1]
    drop = first - last
    detail = f"first={first:.3f} last={last:.3f} drop={drop:.4f} (need >={min_drop})"
    if vocab_size is not None:
        detail += f"; guessing scores {uniform_loss(vocab_size):.3f}"
    return CheckResult("loss moved", drop >= min_drop, detail)


def gradient_report(model: nn.Module, loss: torch.Tensor) -> GradientReport:
    """Sort every parameter into frozen, dead, or live by one backward pass.

    Call this when :func:`check_loss_moved` fails. It answers the question that
    a flat loss curve cannot: *which* parameters were never going to move.

    Existing gradients are cleared before the backward pass, so a stale
    ``.grad`` left over from earlier training cannot make a dead parameter look
    live. This consumes the graph, so pass a loss you are not going to call
    ``backward()`` on yourself.

    Args:
        model (nn.Module): The model whose parameters to inspect.
        loss (torch.Tensor): A scalar loss computed from ``model``, still
            attached to its graph.

    Returns:
        GradientReport: The three buckets, by parameter name.

    Raises:
        ValueError: If ``loss`` is not a scalar, or has no grad history — both
            of which are themselves diagnoses. A loss with no ``grad_fn`` means
            the graph was detached before it ever reached this call.

    Example:
        Freeze half a model and watch it get named:

        >>> import torch
        >>> from torch import nn
        >>> model = nn.Sequential(nn.Linear(4, 4), nn.Linear(4, 2))
        >>> _ = model[0].requires_grad_(False)
        >>> report = gradient_report(model, model(torch.ones(1, 4)).sum())
        >>> report.frozen
        ['0.weight', '0.bias']
        >>> report.all_live()
        False
    """
    if loss.ndim != 0:
        raise ValueError(f"loss must be a scalar, got shape {tuple(loss.shape)}")
    if loss.grad_fn is None:
        raise ValueError(
            "loss has no grad_fn, so no gradient can flow from it. The graph was "
            "detached (a .detach(), .item(), or torch.no_grad() between the model "
            "and the loss), which is itself the bug."
        )

    model.zero_grad(set_to_none=True)
    loss.backward()

    report = GradientReport()
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            report.frozen.append(name)
        elif parameter.grad is None or float(parameter.grad.abs().max()) == 0.0:
            report.dead.append(name)
        else:
            report.live.append(name)
    return report


def check_gradients(model: nn.Module, loss: torch.Tensor) -> CheckResult:
    """Check that a gradient reaches every parameter.

    A pass/fail wrapper over :func:`gradient_report`, for when you want the
    verdict rather than the names.

    Args:
        model (nn.Module): The model whose parameters to inspect.
        loss (torch.Tensor): A scalar loss computed from ``model``.

    Returns:
        CheckResult: Passing when nothing is frozen and nothing is dead.

    Example:
        >>> import torch
        >>> from torch import nn
        >>> model = nn.Linear(4, 2)
        >>> bool(check_gradients(model, model(torch.ones(1, 4)).sum()))
        True
    """
    report = gradient_report(model, loss)
    return CheckResult(
        "gradients reach every parameter", report.all_live(), report.summary()
    )


def check_generalization(
    train_losses: Sequence[float],
    val_losses: Sequence[float],
    max_gap: float = DEFAULT_MAX_GAP,
) -> CheckResult:
    """Check that validation loss has not pulled away from training loss.

    The tell is the *sign* of the gap. A healthy model usually scores slightly
    **better** on validation than on training, because dropout is active during
    training and disabled during validation. When that ordering flips and the
    gap keeps widening, the model is memorizing.

    Watch this across epochs rather than testing it once. Early in training a
    memorizing run and a healthy one are indistinguishable by this check —
    overfitting is not a state the model starts in, it is something it does to
    itself over time.

    Args:
        train_losses (Sequence[float]): Training loss per epoch.
        val_losses (Sequence[float]): Validation loss per epoch.
        max_gap (float): Largest tolerated ``val - train`` at the last epoch.

    Returns:
        CheckResult: Passing when the final gap is at most ``max_gap``.

    Raises:
        ValueError: If either sequence is empty.

    Example:
        A healthy run, validation slightly below training:

        >>> bool(check_generalization([0.265], [0.099]))
        True

        And one that memorized 60 sentence pairs:

        >>> bool(check_generalization([0.540], [1.624]))
        False
    """
    if not train_losses or not val_losses:
        raise ValueError("train_losses and val_losses must both be non-empty")

    train_last, val_last = train_losses[-1], val_losses[-1]
    gap = val_last - train_last
    return CheckResult(
        "generalizes",
        gap <= max_gap,
        f"train={train_last:.3f} val={val_last:.3f} gap={gap:+.3f} (need <=+{max_gap})",
    )


def check_contamination(
    test_sources: Iterable[str],
    train_sources: Iterable[str],
    max_examples: int = 3,
) -> CheckResult:
    """Check that no test sentence also appears in training.

    The most dangerous failure is the one that makes your numbers look *good*.
    A contaminated test set does not report an error; it reports a score partway
    between the truth and a lie, and nothing about the number says which.

    This is exact string matching, so it is a floor rather than a guarantee.
    Near-duplicates differing only in whitespace, casing, or punctuation slip
    past it — normalize both sides first if that is a risk for your corpus.

    Args:
        test_sources (Iterable[str]): Source-side sentences of the test set.
        train_sources (Iterable[str]): Source-side sentences used in training.
        max_examples (int): How many offending sentences to quote in the detail.

    Returns:
        CheckResult: Passing when the two sets are disjoint.

    Example:
        >>> bool(check_contamination(["a fresh sentence"], ["a seen sentence"]))
        True
        >>> result = check_contamination(["a seen sentence"], ["a seen sentence"])
        >>> bool(result)
        False
    """
    test_list = list(test_sources)
    shared = sorted(set(test_list) & set(train_sources))

    detail = f"{len(shared)}/{len(test_list)} test sources also appear in training"
    if shared:
        quoted = ", ".join(repr(s) for s in shared[:max_examples])
        detail += f"; e.g. {quoted}"
    return CheckResult("test set is clean", not shared, detail)


def check_eval_mode(model: nn.Module) -> CheckResult:
    """Check that a model is in evaluation mode before you measure it.

    In training mode dropout stays active, randomly zeroing activations and
    rescaling the survivors. The model still runs and still produces plausible
    output, but every number it gives you is wrong — and a different kind of
    wrong each time you ask.

    The signature is that two identical runs disagree. That is worth more than
    any single number: a measurement you cannot repeat is not a measurement.
    The bias can be large; on the tutorial 6 model, train-mode validation loss
    comes out roughly 2.5x the true value.

    Args:
        model (nn.Module): The model about to be evaluated or decoded from.

    Returns:
        CheckResult: Passing when ``model.training`` is False.

    Example:
        >>> import torch
        >>> model = torch.nn.Linear(2, 2)
        >>> bool(check_eval_mode(model))
        False
        >>> _ = model.eval()
        >>> bool(check_eval_mode(model))
        True
    """
    in_eval = not model.training
    detail = (
        "model.training=False, so results are deterministic"
        if in_eval
        else "model.training=True, so dropout is active and results will not reproduce"
    )
    return CheckResult("model is in eval mode", in_eval, detail)
