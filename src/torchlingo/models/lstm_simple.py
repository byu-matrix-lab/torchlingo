"""Simple LSTM-based sequence-to-sequence model for neural machine translation.

This module implements a basic encoder-decoder architecture using LSTM layers
for sequence-to-sequence learning tasks, such as machine translation. The model
encodes source sequences and decodes target sequences with shared hidden state.

The decoder can optionally attend over the encoder outputs. Attention is **off
by default**, which keeps the classic bottlenecked seq2seq of Sutskever et al.
(2014) as the baseline and makes the with-versus-without comparison an explicit
one-flag experiment. See :mod:`torchlingo.models.attention` for the mechanism
itself and the reason it exists.

Typical usage:
    >>> model = SimpleSeq2SeqLSTM(src_vocab_size=1000, tgt_vocab_size=1000)
    >>> src = torch.randint(0, 1000, (2, 10))
    >>> tgt = torch.randint(0, 1000, (2, 15))
    >>> output = model(src, tgt)
    >>> output.shape
    torch.Size([2, 15, 1000])

    Turning attention on, and recovering the alignments it produces:

    >>> model = SimpleSeq2SeqLSTM(1000, 1000, attention=True, attn_type="dot")
    >>> logits, weights = model(src, tgt, return_attention=True)
    >>> weights.shape  # (batch, tgt_len, src_len)
    torch.Size([2, 15, 10])
"""

import torch
from torch import nn

from ..config import Config, get_default_config
from .attention import build_attention


class SimpleSeq2SeqLSTM(nn.Module):
    """Simple LSTM encoder-decoder model for sequence-to-sequence tasks.

    This model encodes source sequences into a context vector using an LSTM encoder,
    then decodes target sequences using an LSTM decoder initialized with the encoder's
    final hidden and cell states. Token embeddings use padding index from config.

    Without attention, the only channel from encoder to decoder is that final
    hidden state -- a fixed-size summary of the whole source sentence. With
    ``attention=True`` the decoder additionally attends over every encoder
    output, removing that bottleneck.

    Args:
        src_vocab_size (int): Size of the source vocabulary.
        tgt_vocab_size (int): Size of the target vocabulary.
        emb_dim (int, optional): Embedding dimension. Falls back to config.lstm_emb_dim.
        hidden_dim (int, optional): Hidden dimension for LSTM layers. Falls back to config.lstm_hidden_dim.
        num_layers (int, optional): Number of LSTM layers in encoder and decoder. Falls back to config.lstm_num_layers.
        dropout (float, optional): Dropout rate applied between LSTM layers. Falls back to config.lstm_dropout.
        pad_idx (int, optional): Padding token index for embeddings. Falls back to config.pad_idx.
        attention (bool, optional): Whether the decoder attends over encoder outputs.
            Falls back to config.lstm_attention (default ``False``).
        attn_type (str, optional): Scoring function, ``"dot"`` (Luong) or
            ``"additive"`` (Bahdanau). Ignored when ``attention`` is False.
            Falls back to config.lstm_attn_type.
        config (Config, optional): Configuration object. Defaults to default config.

    Attributes:
        pad_idx (int): Resolved padding index.
        src_embed (nn.Embedding): Source token embedding layer.
        tgt_embed (nn.Embedding): Target token embedding layer.
        encoder (nn.LSTM): LSTM encoder for source sequences.
        decoder (nn.LSTM): LSTM decoder for target sequences.
        attention (nn.Module | None): Attention module, or ``None`` when disabled.
        attn_combine (nn.Linear | None): Merges context with the decoder state,
            or ``None`` when attention is disabled.
        output (nn.Linear): Linear output layer projecting decoder hidden state to target vocabulary.
        hidden_dim (int): Hidden dimension size.

    Note:
        Enabling attention adds parameters, so a checkpoint saved with
        ``attention=False`` will not load into a model built with
        ``attention=True`` (and vice versa). Build the model the same way you
        trained it.
    """

    def __init__(
        self,
        src_vocab_size: int,
        tgt_vocab_size: int,
        emb_dim: int | None = None,
        hidden_dim: int | None = None,
        num_layers: int | None = None,
        dropout: float | None = None,
        pad_idx: int | None = None,
        attention: bool | None = None,
        attn_type: str | None = None,
        config: Config | None = None,
    ):
        super().__init__()
        cfg = config if config is not None else get_default_config()
        self.pad_idx = pad_idx if pad_idx is not None else cfg.pad_idx
        emb_dim = emb_dim if emb_dim is not None else cfg.lstm_emb_dim
        hidden_dim = hidden_dim if hidden_dim is not None else cfg.lstm_hidden_dim
        num_layers = num_layers if num_layers is not None else cfg.lstm_num_layers
        dropout = dropout if dropout is not None else cfg.lstm_dropout
        attention = attention if attention is not None else cfg.lstm_attention
        attn_type = attn_type if attn_type is not None else cfg.lstm_attn_type

        self.src_embed = nn.Embedding(src_vocab_size, emb_dim, padding_idx=self.pad_idx)
        self.tgt_embed = nn.Embedding(tgt_vocab_size, emb_dim, padding_idx=self.pad_idx)
        self.encoder = nn.LSTM(
            emb_dim,
            hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout,
        )
        self.decoder = nn.LSTM(
            emb_dim,
            hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout,
        )
        if attention:
            self.attention = build_attention(attn_type, hidden_dim)
            # Luong's "attentional hidden state": fold the context back into the
            # decoder state before predicting, so the context actually reaches
            # the output layer.
            self.attn_combine = nn.Linear(hidden_dim * 2, hidden_dim, bias=False)
        else:
            self.attention = None
            self.attn_combine = None

        self.output = nn.Linear(hidden_dim, tgt_vocab_size)
        self.hidden_dim = hidden_dim

        # Initialize weights for better convergence
        self._init_weights()

    def _init_weights(self):
        """Initialize LSTM weights for better convergence.

        Applies Xavier uniform initialization to input-hidden weights,
        orthogonal initialization to hidden-hidden weights, and zeros
        to biases. This improves training stability compared to default
        PyTorch initialization.
        """
        for name, param in self.named_parameters():
            if "weight_ih" in name:
                # Input-to-hidden weights: Xavier uniform
                nn.init.xavier_uniform_(param)
            elif "weight_hh" in name:
                # Hidden-to-hidden weights: Orthogonal (preserves gradient flow)
                nn.init.orthogonal_(param)
            elif "bias" in name:
                # Biases: Initialize to zero
                nn.init.zeros_(param)

    def encode_source(
        self,
        src: torch.Tensor,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor], torch.Tensor]:
        """Run the encoder and return everything the decoder may need.

        Deliberately returns the per-token encoder outputs alongside the final
        state. Without attention only the final state is used and ``enc_out`` is
        discarded -- that discarded tensor *is* the information bottleneck.

        The source is packed before it enters the LSTM so that each sequence
        stops at its true final token. Without packing the encoder keeps
        consuming ``PAD`` after the sentence ends, and the state handed to the
        decoder describes the padding rather than the sentence -- which makes a
        batched result differ from the same sentence decoded alone. Masking the
        attention weights alone does not fix this; the leak is in the recurrence,
        not the alignment.

        Args:
            src (torch.Tensor): Source token indices, shape (batch_size, src_len).

        Returns:
            tuple: ``(enc_out, hidden, src_pad_mask)`` where ``enc_out`` has
                shape (batch_size, src_len, hidden_dim), ``hidden`` is the
                ``(h, c)`` pair from each sequence's final *non-padding* step,
                and ``src_pad_mask`` is a boolean (batch_size, src_len) tensor
                with ``True`` at padding positions.

        Examples:
            >>> model = SimpleSeq2SeqLSTM(50, 50, emb_dim=8, hidden_dim=8, num_layers=1)
            >>> enc_out, (h, c), mask = model.encode_source(torch.randint(1, 50, (2, 5)))
            >>> enc_out.shape
            torch.Size([2, 5, 8])
        """
        src_emb = self.src_embed(src)
        src_pad_mask = src.eq(self.pad_idx)

        # clamp(min=1): an all-padding row has no real tokens, but packing
        # requires every sequence to have length >= 1. Its outputs are masked
        # out downstream regardless.
        lengths = src.ne(self.pad_idx).sum(dim=1).clamp(min=1).cpu()
        packed = nn.utils.rnn.pack_padded_sequence(
            src_emb, lengths, batch_first=True, enforce_sorted=False
        )
        packed_out, hidden = self.encoder(packed)
        enc_out, _ = nn.utils.rnn.pad_packed_sequence(
            packed_out, batch_first=True, total_length=src.size(1)
        )
        return enc_out, hidden, src_pad_mask

    def _apply_attention(
        self,
        dec_out: torch.Tensor,
        enc_out: torch.Tensor,
        src_pad_mask: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Blend attended source context into the decoder states.

        Args:
            dec_out (torch.Tensor): Decoder states, (batch, tgt_len, hidden_dim).
            enc_out (torch.Tensor): Encoder outputs, (batch, src_len, hidden_dim).
            src_pad_mask (torch.Tensor, optional): Padding mask, (batch, src_len).

        Returns:
            tuple: The (possibly updated) decoder states and the attention
                weights of shape (batch, tgt_len, src_len), or ``None`` for the
                weights when attention is disabled.
        """
        if self.attention is None:
            return dec_out, None
        context, weights = self.attention(dec_out, enc_out, src_pad_mask)
        combined = torch.cat([context, dec_out], dim=-1)
        return torch.tanh(self.attn_combine(combined)), weights

    def forward(
        self,
        src: torch.Tensor,
        tgt: torch.Tensor,
        return_attention: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor | None]:
        """Encode source and decode target sequences.

        Encodes source tokens into context using the LSTM encoder, then passes
        the final hidden and cell states to the LSTM decoder to generate target
        token predictions. When the model was built with ``attention=True``, the
        decoder states are additionally blended with a weighted average of the
        encoder outputs before the output projection.

        Args:
            src (torch.Tensor): Source token indices with shape (batch_size, src_len).
                Values should be in range [0, src_vocab_size).
            tgt (torch.Tensor): Target token indices with shape (batch_size, tgt_len).
                Values should be in range [0, tgt_vocab_size).
            return_attention (bool, optional): Also return the attention weights.
                Defaults to False, which preserves the plain-tensor return type.

        Returns:
            torch.Tensor: Logits of shape (batch_size, tgt_len, tgt_vocab_size) representing
                probability distributions over the target vocabulary for each position.

            If ``return_attention`` is True, returns a ``(logits, weights)``
            tuple instead, where ``weights`` has shape
            (batch_size, tgt_len, src_len) -- or is ``None`` when the model has
            no attention.

        Examples:
            >>> model = SimpleSeq2SeqLSTM(50, 50, emb_dim=8, hidden_dim=8, attention=True)
            >>> src, tgt = torch.randint(1, 50, (2, 5)), torch.randint(1, 50, (2, 3))
            >>> logits, weights = model(src, tgt, return_attention=True)
            >>> weights.shape
            torch.Size([2, 3, 5])
        """
        enc_out, hidden, src_pad_mask = self.encode_source(src)
        tgt_emb = self.tgt_embed(tgt)
        dec_out, _ = self.decoder(tgt_emb, hidden)
        dec_out, weights = self._apply_attention(dec_out, enc_out, src_pad_mask)
        logits = self.output(dec_out)
        return (logits, weights) if return_attention else logits

    def decode_step(
        self,
        last_token: torch.Tensor,
        hidden: tuple[torch.Tensor, torch.Tensor],
        enc_out: torch.Tensor,
        src_pad_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor], torch.Tensor | None]:
        """Advance the decoder by a single token.

        Exists so that incremental decoding in :mod:`torchlingo.inference` does
        not have to reimplement the decoder, which is how the encoder outputs
        came to be silently dropped on the inference path before attention
        existed.

        Args:
            last_token (torch.Tensor): Most recent token ids, shape (batch_size, 1).
            hidden (tuple): The decoder ``(h, c)`` state carried from the
                previous step, or from :meth:`encode_source` at the first step.
            enc_out (torch.Tensor): Encoder outputs from :meth:`encode_source`.
            src_pad_mask (torch.Tensor, optional): Padding mask from :meth:`encode_source`.

        Returns:
            tuple: ``(logits, hidden, weights)`` where ``logits`` has shape
                (batch_size, 1, tgt_vocab_size), ``hidden`` is the updated LSTM
                state, and ``weights`` has shape (batch_size, 1, src_len) or is
                ``None`` when attention is disabled.
        """
        emb = self.tgt_embed(last_token)
        dec_out, hidden = self.decoder(emb, hidden)
        dec_out, weights = self._apply_attention(dec_out, enc_out, src_pad_mask)
        return self.output(dec_out), hidden, weights
