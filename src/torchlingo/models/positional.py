"""Sinusoidal positional encoding for transformer models.

Transformers have no built-in notion of token order, so position information
must be added to the token embeddings. This module implements the classic
fixed sinusoidal positional encoding from "Attention Is All You Need":
each position is represented by sine and cosine waves of geometrically
increasing wavelengths, allowing the model to attend by relative offsets
and to generalize to positions beyond those seen in training.

Reference:
    Vaswani et al. (2017): "Attention Is All You Need"
    https://arxiv.org/abs/1706.03762

Typical usage:
    >>> pos_enc = SinusoidalPositionalEncoding(d_model=64, max_seq_len=512)
    >>> x = torch.randn(2, 10, 64)  # (batch, seq_len, d_model)
    >>> x = pos_enc(x)
"""

import math
from typing import Optional

import torch
import torch.nn as nn

from ..config import Config, get_default_config


class SinusoidalPositionalEncoding(nn.Module):
    """Fixed sinusoidal positional encoding added to token embeddings.

    For position ``pos`` and dimension index ``i``, the encoding is:

    - ``PE(pos, 2i)   = sin(pos / 10000^(2i / d_model))``
    - ``PE(pos, 2i+1) = cos(pos / 10000^(2i / d_model))``

    The encodings are precomputed once for ``max_seq_len`` positions and
    stored in a buffer; the forward pass simply slices and adds them to the
    input embeddings, followed by dropout. If a longer sequence is seen at
    runtime, the table is rebuilt on the input's device.

    Args:
        d_model (int, optional): Embedding dimension. Falls back to
            config.d_model.
        max_seq_len (int, optional): Number of positions to precompute.
            Falls back to config.max_seq_length.
        dropout (float, optional): Dropout applied after adding the encoding.
            Falls back to config.dropout.
        base (float, optional): Base for the geometric progression of
            wavelengths. Defaults to 10000.0.
        config (Config, optional): Configuration object. Defaults to default
            config.

    Attributes:
        d_model (int): Embedding dimension.
        max_seq_len (int): Number of precomputed positions.
        base (float): Wavelength base.
        pe (torch.Tensor): Precomputed encodings of shape
            (max_seq_len, d_model), registered as a non-persistent buffer.
    """

    def __init__(
        self,
        d_model: Optional[int] = None,
        max_seq_len: Optional[int] = None,
        dropout: Optional[float] = None,
        base: float = 10000.0,
        config: Optional[Config] = None,
    ) -> None:
        super().__init__()
        cfg = config if config is not None else get_default_config()
        d_model = d_model if d_model is not None else cfg.d_model
        max_seq_len = max_seq_len if max_seq_len is not None else cfg.max_seq_length
        dropout = dropout if dropout is not None else cfg.dropout

        self.d_model = d_model
        self.max_seq_len = max_seq_len
        self.base = base
        self.dropout = nn.Dropout(p=dropout)
        self.register_buffer(
            "pe", self._build_table(max_seq_len, d_model, base), persistent=False
        )

    @staticmethod
    def _build_table(
        seq_len: int,
        d_model: int,
        base: float,
        device: Optional[torch.device] = None,
    ) -> torch.Tensor:
        """Compute the sinusoidal encoding table.

        Args:
            seq_len (int): Number of positions to encode.
            d_model (int): Embedding dimension.
            base (float): Wavelength base.
            device (torch.device, optional): Device to build the table on.

        Returns:
            torch.Tensor: Encoding table of shape (seq_len, d_model).
        """
        position = torch.arange(seq_len, dtype=torch.float32, device=device)
        # One frequency per sin/cos pair: base^(-2i / d_model)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32, device=device)
            * (-math.log(base) / d_model)
        )
        angles = position.unsqueeze(1) * div_term.unsqueeze(0)
        pe = torch.zeros(seq_len, d_model, device=device)
        pe[:, 0::2] = torch.sin(angles)
        # For odd d_model there is one fewer cosine column than sine column.
        pe[:, 1::2] = torch.cos(angles[:, : d_model // 2])
        return pe

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Add positional encodings to a batch of embeddings.

        Args:
            x (torch.Tensor): Token embeddings of shape
                (batch_size, seq_len, d_model).

        Returns:
            torch.Tensor: Embeddings with positional information added,
                same shape as the input, after dropout.
        """
        seq_len = x.size(1)
        if seq_len > self.pe.size(0):
            # Rebuild on the input's device so the cache never lags behind
            # a model.to(device) move.
            self.pe = self._build_table(seq_len, self.d_model, self.base, x.device)
        return self.dropout(x + self.pe[:seq_len].unsqueeze(0))
