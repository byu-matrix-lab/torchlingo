"""Visualizing what a model attends to, and what a search considered.

Two things here, answering two different questions about the same decode:

- **Attention** shows what the decoder *looked at* while producing each token.
- **Beam search** shows what it *considered and discarded* on the way there.

The second is the one students find least intuitive, because pruning is
invisible in the output: a translation tells you what won, never what lost, and
the whole argument for beam search is about the paths greedy never explores.

Attention weights are the most directly *inspectable* quantity in an NMT model:
each row is a probability distribution saying "while producing this target
token, here is how much I looked at each source token." Plotting that matrix
turns an abstraction into something a student can check against their own
intuition about the sentence pair.

Two renderers are provided:

- :func:`plot_attention` draws the familiar alignment heatmap with matplotlib.
- :func:`format_attention` returns a shaded text grid instead. It needs nothing
  beyond the standard library, so it works over SSH, in a plain terminal, in
  log files, and in doctests -- anywhere a figure window is not available.

Typical usage:
    >>> import torch
    >>> weights = torch.tensor([[0.8, 0.1, 0.1], [0.1, 0.2, 0.7]])
    >>> print(format_attention(weights, ["el", "gato", "duerme"], ["the", "sleeps"]))
               el   gato duerme
    the       ▓▓▓    ···    ···
    sleeps    ···    ░░░    ▓▓▓
"""

import torch

from .inference import BeamStep

_SHADES = "·░▒▓█"
"""Ramp from no attention to full attention, five levels.

The lightest level is a dot rather than a space so that a near-zero cell still
reads as a cell. A masked or padded position looks the same as an ignored one,
which is correct -- the model gave both no weight.
"""


def _shade(weight: float) -> str:
    """Map a weight in [0, 1] to a shading character.

    Args:
        weight (float): Attention weight.

    Returns:
        str: A single character from the shading ramp.
    """
    index = round(max(0.0, min(1.0, weight)) * (len(_SHADES) - 1))
    return _SHADES[index]


def _as_matrix(weights: torch.Tensor) -> torch.Tensor:
    """Coerce attention weights to a 2-D (tgt_len, src_len) matrix.

    Args:
        weights (torch.Tensor): Weights of shape (tgt_len, src_len) or
            (batch, tgt_len, src_len) with a batch size of 1.

    Returns:
        torch.Tensor: A detached 2-D tensor on the CPU.

    Raises:
        ValueError: If the tensor is not 2-D, or is 3-D with batch size != 1.
    """
    weights = weights.detach().cpu()
    if weights.dim() == 3:
        if weights.size(0) != 1:
            raise ValueError(
                f"Expected a single sentence, got a batch of {weights.size(0)}. "
                "Index the batch first, e.g. weights[0]."
            )
        weights = weights[0]
    if weights.dim() != 2:
        raise ValueError(
            f"Expected attention weights with 2 dimensions, got {weights.dim()}."
        )
    return weights


def _check_labels(
    weights: torch.Tensor,
    src_tokens: list[str],
    tgt_tokens: list[str],
) -> None:
    """Verify that label counts match the weight matrix.

    Args:
        weights (torch.Tensor): Matrix of shape (tgt_len, src_len).
        src_tokens (list[str]): Source token labels.
        tgt_tokens (list[str]): Target token labels.

    Raises:
        ValueError: If either label list has the wrong length.
    """
    tgt_len, src_len = weights.shape
    if len(src_tokens) != src_len:
        raise ValueError(
            f"Got {len(src_tokens)} source labels for {src_len} source positions."
        )
    if len(tgt_tokens) != tgt_len:
        raise ValueError(
            f"Got {len(tgt_tokens)} target labels for {tgt_len} target positions."
        )


def format_attention(
    weights: torch.Tensor,
    src_tokens: list[str],
    tgt_tokens: list[str],
    show_values: bool = False,
) -> str:
    """Render an attention matrix as a shaded text grid.

    Rows are target tokens, columns are source tokens, and each row sums to 1.
    Darker cells mean the decoder looked harder at that source token while
    producing that target token.

    Args:
        weights (torch.Tensor): Attention weights of shape (tgt_len, src_len),
            or (1, tgt_len, src_len).
        src_tokens (list[str]): Source token strings, one per source position.
        tgt_tokens (list[str]): Target token strings, one per target position.
        show_values (bool, optional): Print two-digit percentages instead of
            shading. Useful when you need the exact numbers. Defaults to False.

    Returns:
        str: A multi-line string suitable for printing.

    Raises:
        ValueError: If the weights are not a single 2-D matrix, or the label
            counts do not match its shape.

    Examples:
        >>> import torch
        >>> w = torch.tensor([[0.9, 0.1]])
        >>> print(format_attention(w, ["gato", "duerme"], ["cat"], show_values=True))
              gato duerme
        cat     90     10
    """
    matrix = _as_matrix(weights)
    _check_labels(matrix, src_tokens, tgt_tokens)

    label_w = max((len(t) for t in tgt_tokens), default=0)
    cell_w = max(6, max((len(t) for t in src_tokens), default=0) + 1)

    # Header and cells are both right-justified in the same column width, so
    # each column of shading sits under its source token.
    header = " " * label_w + "".join(t.rjust(cell_w) for t in src_tokens)
    lines = [header]
    for row_label, row in zip(tgt_tokens, matrix.tolist()):
        if show_values:
            cells = "".join(f"{round(w * 100):d}".rjust(cell_w) for w in row)
        else:
            cells = "".join((_shade(w) * 3).rjust(cell_w) for w in row)
        lines.append(row_label.ljust(label_w) + cells)
    return "\n".join(lines)


def plot_attention(
    weights: torch.Tensor,
    src_tokens: list[str],
    tgt_tokens: list[str],
    title: str | None = None,
    cmap: str = "viridis",
    ax: "matplotlib.axes.Axes | None" = None,  # noqa: F821
) -> "matplotlib.axes.Axes":  # noqa: F821
    """Draw an attention alignment heatmap with matplotlib.

    Args:
        weights (torch.Tensor): Attention weights of shape (tgt_len, src_len),
            or (1, tgt_len, src_len).
        src_tokens (list[str]): Source token strings, one per source position.
        tgt_tokens (list[str]): Target token strings, one per target position.
        title (str, optional): Title for the plot.
        cmap (str, optional): Matplotlib colormap name. Defaults to "viridis".
        ax (matplotlib.axes.Axes, optional): Existing axes to draw into. A new
            figure and axes are created when omitted.

    Returns:
        matplotlib.axes.Axes: The axes the heatmap was drawn into.

    Raises:
        ImportError: If matplotlib is unavailable. It ships as a TorchLingo
            dependency, so this normally cannot happen; the guard exists for
            stripped-down installs, and points at :func:`format_attention`.
        ValueError: If the weights are not a single 2-D matrix, or the label
            counts do not match its shape.

    Examples:
        >>> import torch
        >>> ax = plot_attention(torch.rand(3, 4).softmax(-1),
        ...                     ["a", "b", "c", "d"], ["x", "y", "z"])
        ... # doctest: +SKIP
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise ImportError(
            "plot_attention requires matplotlib. It normally ships with "
            "TorchLingo; reinstall with 'pip install matplotlib', or use "
            "format_attention for a text rendering that needs no extra packages."
        ) from exc

    matrix = _as_matrix(weights)
    _check_labels(matrix, src_tokens, tgt_tokens)

    if ax is None:
        _fig, ax = plt.subplots(
            figsize=(max(4, len(src_tokens) * 0.6), max(3, len(tgt_tokens) * 0.5))
        )

    image = ax.imshow(matrix.numpy(), aspect="auto", cmap=cmap, vmin=0.0, vmax=1.0)
    ax.set_xticks(range(len(src_tokens)), src_tokens, rotation=45, ha="right")
    ax.set_yticks(range(len(tgt_tokens)), tgt_tokens)
    ax.set_xlabel("source")
    ax.set_ylabel("target")
    if title:
        ax.set_title(title)
    ax.figure.colorbar(image, ax=ax, label="attention weight")
    ax.figure.tight_layout()
    return ax


def _decode_tokens(tokens: list[int], itos: list[str] | None) -> str:
    """Render a hypothesis as text, falling back to raw ids without a vocabulary."""
    if itos is None:
        return " ".join(str(t) for t in tokens)
    return " ".join(itos[t] if t < len(itos) else f"<{t}>" for t in tokens)


def format_beam_search(
    trace: list[BeamStep],
    itos: list[str] | None = None,
    winner: list[int] | None = None,
    top: int = 6,
    max_steps: int | None = None,
) -> str:
    """Render a beam search as text: what was considered, kept, and discarded.

    Produced from the ``trace`` argument of
    :func:`torchlingo.inference.beam_search_decode`.

    Each step lists candidates in the order the search ranked them. The marker
    in the first column is the point of the whole display:

    ==========  ==========================================================
    ``>``       kept, and a prefix of the hypothesis that eventually won
    ``+``       kept into the next step
    ``.``       pruned here
    ==========  ==========================================================

    The steps worth looking at are the ones where a ``>`` sits below a ``+``:
    the eventual winner ranked *below* another hypothesis at that moment and
    survived only because the beam was wide enough to carry it. That is exactly
    the situation greedy decoding cannot recover from, and it is otherwise
    invisible in the output.

    Args:
        trace (list[BeamStep]): Steps recorded during decoding.
        itos (list[str], optional): Index-to-token mapping. Raw ids are shown
            when omitted.
        winner (list[int], optional): The returned token sequence, used to mark
            which candidates were on the winning path.
        top (int, optional): Candidates to show per step. Defaults to 6.
        max_steps (int, optional): Steps to show. All of them when omitted.

    Returns:
        str: A multi-line string suitable for printing.

    Raises:
        ValueError: If the trace is empty.

    Examples:
        >>> from torchlingo.inference import BeamCandidate, BeamStep
        >>> step = BeamStep(0, [BeamCandidate([2, 7], -0.2, -0.2, True)])
        >>> print(format_beam_search([step], itos=["<pad>", "<unk>", "<s>", "</s>", "", "", "", "hola"]))
        step 0
          + -0.200  <s> hola
    """
    if not trace:
        raise ValueError("Empty trace: pass trace=[] to beam_search_decode first.")

    steps = trace if max_steps is None else trace[:max_steps]
    winning_prefixes = set()
    if winner is not None:
        winning_prefixes = {tuple(winner[: i + 1]) for i in range(len(winner))}

    lines: list[str] = []
    for record in steps:
        lines.append(f"step {record.step}")
        for candidate in record.candidates[:top]:
            if candidate.kept and tuple(candidate.tokens) in winning_prefixes:
                marker = ">"
            elif candidate.kept:
                marker = "+"
            else:
                marker = "."
            lines.append(
                f"  {marker} {candidate.normalized:6.3f}  "
                f"{_decode_tokens(candidate.tokens, itos)}"
            )
        pruned = len(record.candidates) - top
        if pruned > 0:
            lines.append(f"    ... {pruned} more considered")
    return "\n".join(lines)


def plot_beam_search(
    trace: list[BeamStep],
    winner: list[int] | None = None,
    top: int = 6,
    title: str | None = None,
    ax: "matplotlib.axes.Axes | None" = None,  # noqa: F821
) -> "matplotlib.axes.Axes":  # noqa: F821
    """Plot candidate scores per step, separating survivors from pruned paths.

    Kept candidates are drawn filled, pruned ones hollow, and the winning path
    is connected by a line. The visual question the plot answers is whether the
    winning line ever dips below other kept points — which is precisely when
    beam search earns its cost over greedy.

    Args:
        trace (list[BeamStep]): Steps recorded during decoding.
        winner (list[int], optional): The returned token sequence.
        top (int, optional): Candidates to plot per step.
        title (str, optional): Title for the plot.
        ax (matplotlib.axes.Axes, optional): Existing axes to draw into.

    Returns:
        matplotlib.axes.Axes: The axes drawn into.

    Raises:
        ImportError: If matplotlib is unavailable.
        ValueError: If the trace is empty.
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise ImportError(
            "plot_beam_search requires matplotlib. Use format_beam_search for a "
            "text rendering that needs no extra packages."
        ) from exc

    if not trace:
        raise ValueError("Empty trace: pass trace=[] to beam_search_decode first.")

    if ax is None:
        _fig, ax = plt.subplots(figsize=(max(5, len(trace) * 0.7), 4))

    winning_prefixes = set()
    if winner is not None:
        winning_prefixes = {tuple(winner[: i + 1]) for i in range(len(winner))}

    winning_x: list[int] = []
    winning_y: list[float] = []
    for record in trace:
        for candidate in record.candidates[:top]:
            on_winning_path = tuple(candidate.tokens) in winning_prefixes
            ax.scatter(
                record.step,
                candidate.normalized,
                facecolors="tab:blue" if candidate.kept else "none",
                edgecolors="tab:blue" if candidate.kept else "tab:grey",
                zorder=3 if candidate.kept else 2,
            )
            if on_winning_path:
                winning_x.append(record.step)
                winning_y.append(candidate.normalized)

    if winning_x:
        ax.plot(
            winning_x, winning_y, color="tab:red", linewidth=2, zorder=4, label="winner"
        )
        ax.legend(loc="best")

    ax.set_xlabel("step")
    ax.set_ylabel("length-normalized score")
    if title:
        ax.set_title(title)
    ax.figure.tight_layout()
    return ax


__all__ = [
    "format_attention",
    "format_beam_search",
    "plot_attention",
    "plot_beam_search",
]
