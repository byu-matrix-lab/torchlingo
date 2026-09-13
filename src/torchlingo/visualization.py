"""Visualizing what a model attends to.

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


__all__ = ["format_attention", "plot_attention"]
