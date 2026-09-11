"""Attention mechanisms for sequence-to-sequence models.

This module contains the readable, from-scratch attention implementations that
the rest of the library builds on. It exists for a specific reason: the
Transformer in :mod:`torchlingo.models.transformer_simple` gets its attention
from :class:`torch.nn.Transformer`, where the mechanism is real but invisible.
Here it is roughly fifteen lines you can read.

**The problem attention solves.** A plain LSTM encoder-decoder squeezes the
entire source sentence through one fixed-size hidden state. Every source token
is encoded and then thrown away; only the final state survives to the decoder.
That is an *information bottleneck*, and it is why translation quality falls off
sharply on long sentences. Attention removes it by letting the decoder look back
at every encoder output, weighted by relevance, at every step.

**Two scoring functions.** Both compute the same three things -- a score per
source position, a softmax over those scores, and a weighted average of encoder
outputs -- and differ only in how the score is computed:

============  ====================================  ============================
Scorer        Score                                 Notes
============  ====================================  ============================
``additive``  ``v @ tanh(W_dec h_t + W_enc h_s)``   Bahdanau et al., 2014
``dot``       ``h_t @ h_s``                         Luong et al., 2015
============  ====================================  ============================

The progression is worth following. Bahdanau's additive score learns three
matrices to compare decoder and encoder states. Luong's dot score observes that
if the two live in the same space you can just take an inner product -- no
parameters at all. The Transformer then keeps the dot product, adds a
``1 / sqrt(d_k)`` scale, and applies it in parallel heads. Reading these two in
order makes the third one familiar rather than alien.

Note:
    The list of supported scorers is :data:`torchlingo.config.ATTENTION_TYPES`,
    imported here rather than defined here. ``Config`` validates
    ``lstm_attn_type`` while it is still being imported, and every model module
    imports ``Config``, so defining the list in this module would make that
    import circular.

Typical usage:
    >>> import torch
    >>> attn = DotProductAttention()
    >>> dec_out = torch.randn(2, 3, 16)   # (batch, tgt_len, hidden)
    >>> enc_out = torch.randn(2, 5, 16)   # (batch, src_len, hidden)
    >>> context, weights = attn(dec_out, enc_out)
    >>> context.shape, weights.shape
    (torch.Size([2, 3, 16]), torch.Size([2, 3, 5]))
    >>> torch.allclose(weights.sum(-1), torch.ones(2, 3))
    True
"""

import torch
from torch import nn

from ..config import ATTENTION_TYPES


def _masked_softmax(
    scores: torch.Tensor,
    src_pad_mask: torch.Tensor | None,
) -> torch.Tensor:
    """Softmax over source positions, ignoring padding.

    Padding positions are set to the most negative finite value for the dtype
    rather than ``-inf``. Both give a weight of essentially zero, but ``-inf``
    produces ``NaN`` if an entire row is padded, which is exactly the kind of
    silent corruption that is painful to track down later.

    Args:
        scores (torch.Tensor): Raw scores of shape (batch, tgt_len, src_len).
        src_pad_mask (torch.Tensor, optional): Boolean mask of shape
            (batch, src_len) where ``True`` marks a padding position.

    Returns:
        torch.Tensor: Attention weights of shape (batch, tgt_len, src_len),
            each row summing to 1 over the source axis.
    """
    if src_pad_mask is not None:
        # (batch, src_len) -> (batch, 1, src_len) so it broadcasts over tgt_len.
        fill = torch.finfo(scores.dtype).min
        scores = scores.masked_fill(src_pad_mask.unsqueeze(1), fill)
    return scores.softmax(dim=-1)


class DotProductAttention(nn.Module):
    """Luong dot-product attention (Luong et al., 2015).

    Scores each source position by taking the inner product of the decoder
    state with the encoder state. This has **no learned parameters at all** --
    the comparison is pure geometry, which is what makes it the direct ancestor
    of Transformer self-attention.

    Requires the decoder and encoder hidden sizes to match, since an inner
    product is only defined between vectors in the same space.

    Note:
        The Transformer divides these scores by ``sqrt(d_k)`` before the
        softmax; this implementation does not, matching Luong's paper. The
        scale matters at large ``d_k``, where unscaled dot products grow large
        enough to push the softmax into saturation and flatten the gradient.

    Examples:
        >>> import torch
        >>> attn = DotProductAttention()
        >>> context, weights = attn(torch.randn(1, 2, 8), torch.randn(1, 4, 8))
        >>> weights.shape
        torch.Size([1, 2, 4])
    """

    def forward(
        self,
        dec_out: torch.Tensor,
        enc_out: torch.Tensor,
        src_pad_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute context vectors and attention weights.

        Args:
            dec_out (torch.Tensor): Decoder states, shape (batch, tgt_len, hidden).
            enc_out (torch.Tensor): Encoder outputs, shape (batch, src_len, hidden).
            src_pad_mask (torch.Tensor, optional): Boolean mask of shape
                (batch, src_len); ``True`` marks padding to be ignored.

        Returns:
            tuple[torch.Tensor, torch.Tensor]: The context vectors of shape
                (batch, tgt_len, hidden), and the attention weights of shape
                (batch, tgt_len, src_len).

        Raises:
            ValueError: If the decoder and encoder hidden sizes differ.
        """
        if dec_out.size(-1) != enc_out.size(-1):
            raise ValueError(
                "Dot-product attention requires matching hidden sizes, got "
                f"decoder {dec_out.size(-1)} and encoder {enc_out.size(-1)}. "
                "Use attn_type='additive', which learns a projection instead."
            )
        # "How well does each decoder state match each source state?"
        # (batch, tgt_len, hidden) @ (batch, hidden, src_len) -> (batch, tgt_len, src_len)
        scores = torch.bmm(dec_out, enc_out.transpose(1, 2))
        weights = _masked_softmax(scores, src_pad_mask)
        # Weighted average of the source representations.
        context = torch.bmm(weights, enc_out)
        return context, weights


class AdditiveAttention(nn.Module):
    """Bahdanau additive attention (Bahdanau et al., 2014).

    Scores each source position with a small one-hidden-layer network over the
    concatenated decoder and encoder states. Because the comparison is learned
    rather than geometric, the two sides may have different hidden sizes.

    This is the mechanism from the paper that introduced attention to NMT. It
    works well, but note that the ``tanh`` scoring network does *not* reappear
    in the Transformer -- the dot product does.

    Args:
        hidden_dim (int): Decoder hidden size.
        enc_dim (int, optional): Encoder hidden size. Defaults to ``hidden_dim``.
        attn_dim (int, optional): Width of the scoring network's hidden layer.
            Defaults to ``hidden_dim``.

    Attributes:
        W_dec (nn.Linear): Projects the decoder state into the scoring space.
        W_enc (nn.Linear): Projects each encoder state into the scoring space.
        v (nn.Linear): Collapses the scoring space to one scalar per position.

    Examples:
        >>> import torch
        >>> attn = AdditiveAttention(hidden_dim=8, enc_dim=16)
        >>> context, weights = attn(torch.randn(1, 2, 8), torch.randn(1, 4, 16))
        >>> context.shape, weights.shape
        (torch.Size([1, 2, 16]), torch.Size([1, 2, 4]))
    """

    def __init__(
        self,
        hidden_dim: int,
        enc_dim: int | None = None,
        attn_dim: int | None = None,
    ):
        super().__init__()
        enc_dim = enc_dim if enc_dim is not None else hidden_dim
        attn_dim = attn_dim if attn_dim is not None else hidden_dim
        self.W_dec = nn.Linear(hidden_dim, attn_dim, bias=False)
        self.W_enc = nn.Linear(enc_dim, attn_dim, bias=False)
        self.v = nn.Linear(attn_dim, 1, bias=False)

    def forward(
        self,
        dec_out: torch.Tensor,
        enc_out: torch.Tensor,
        src_pad_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute context vectors and attention weights.

        Args:
            dec_out (torch.Tensor): Decoder states, shape (batch, tgt_len, hidden).
            enc_out (torch.Tensor): Encoder outputs, shape (batch, src_len, enc_dim).
            src_pad_mask (torch.Tensor, optional): Boolean mask of shape
                (batch, src_len); ``True`` marks padding to be ignored.

        Returns:
            tuple[torch.Tensor, torch.Tensor]: The context vectors of shape
                (batch, tgt_len, enc_dim), and the attention weights of shape
                (batch, tgt_len, src_len).
        """
        # Broadcast decoder over source and source over decoder, so that every
        # (target position, source position) pair gets its own score.
        dec_proj = self.W_dec(dec_out).unsqueeze(2)  # (batch, tgt_len, 1, attn_dim)
        enc_proj = self.W_enc(enc_out).unsqueeze(1)  # (batch, 1, src_len, attn_dim)
        scores = self.v(torch.tanh(dec_proj + enc_proj)).squeeze(-1)
        weights = _masked_softmax(scores, src_pad_mask)
        context = torch.bmm(weights, enc_out)
        return context, weights


def build_attention(
    attn_type: str,
    hidden_dim: int,
    enc_dim: int | None = None,
) -> nn.Module:
    """Construct an attention module by name.

    Args:
        attn_type (str): One of ``"dot"`` (Luong) or ``"additive"`` (Bahdanau).
        hidden_dim (int): Decoder hidden size.
        enc_dim (int, optional): Encoder hidden size. Defaults to ``hidden_dim``.

    Returns:
        nn.Module: An attention module whose ``forward`` takes
            ``(dec_out, enc_out, src_pad_mask)`` and returns
            ``(context, weights)``.

    Raises:
        ValueError: If ``attn_type`` is not a supported scorer.

    Examples:
        >>> build_attention("dot", hidden_dim=16)
        DotProductAttention()
    """
    if attn_type == "dot":
        return DotProductAttention()
    if attn_type == "additive":
        return AdditiveAttention(hidden_dim, enc_dim=enc_dim)
    raise ValueError(
        f"Unknown attn_type '{attn_type}'. Expected one of {list(ATTENTION_TYPES)}."
    )


__all__ = [
    "ATTENTION_TYPES",
    "AdditiveAttention",
    "DotProductAttention",
    "build_attention",
]
