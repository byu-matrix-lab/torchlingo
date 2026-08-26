"""Optimized decoders for TorchLingo models.

The implementations here are **faster counterparts** to those in
:mod:`torchlingo.inference`, not replacements. That module holds the reference
decoders, written to be read: :func:`~torchlingo.inference.beam_search_decode`
is about 40 lines of plain Python that a student can follow line by line. This
module trades some of that readability for speed.

The two are held together by a contract:

    The reference implementation is the specification. Every decoder here must
    produce **token-identical** output for the same model and input.

``tests/test_decoding_equivalence.py`` enforces that: its fixtures and
invariants live in a shared contract base class that every implementation is
run against, so the two cannot silently diverge and the reference cannot rot.

Why a faster beam search is needed
----------------------------------
The reference issues one ``model.decode()`` call per *beam* per step. Measured
on 8 sentences with ``max_len=25`` and ``beam_size=5``:

===================  ================  ===============  ==================
Implementation       ``decode()``      seqs per batch   positions forwarded
===================  ================  ===============  ==================
greedy                            25              8.0               2,600
reference beam                   968              1.0              12,968
ratio                          38.7x             4.8x                5.0x
===================  ================  ===============  ==================

Beam search does roughly 5x the arithmetic of greedy -- about what
``beam_size=5`` should cost -- but issues 38.7x more kernel launches, every one
at batch size 1. That is latency-bound on dispatch, not compute-bound, which is
exactly what batching fixes.

Read that 38.7x carefully: greedy in the table is already batched across
*sentences*, while the reference beam search batches neither sentences nor
beams. The figure is therefore two independent inefficiencies multiplied
together::

    reference beam vs greedy :  38.7x
      of which, beam axis    :   4.8x   <- one call per beam, per step
      of which, sentence axis:   8.0x   <- one sentence at a time
      product                :  38.7x

:func:`beam_search_decode_batched` removes the **beam** factor only, recovering
roughly ``beam_size``. It still decodes one sentence at a time, so the sentence
factor remains. Expect about ``beam_size``, not 38.7x.

Which to use
------------
Use the reference when reading, teaching, or debugging, and for small inputs
where the difference does not matter. Use these when decoding a real test set;
a few thousand sentences through the reference beam search is slow enough that
it discourages using beam search at all.

See Also:
    :mod:`torchlingo.inference`: the reference implementations these mirror.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.utils.rnn import pad_sequence

from .config import Config, get_default_config
from .data_processing.vocab import BaseVocab
from .inference import _canonical_topk, _rank_key


def beam_search_decode_batched(
    model: nn.Module,
    src: torch.Tensor,
    beam_size: int = 5,
    max_len: int = 100,
    alpha: float = 0.6,
    device: torch.device | None = None,
    config: Config | None = None,
) -> list[int]:
    """Beam search that evaluates all live beams in one decoder call per step.

    Drop-in replacement for :func:`~torchlingo.inference.beam_search_decode`
    with an identical signature and identical output. The search itself is
    unchanged; only how the model is invoked differs.

    The key observation is that **every live beam always has the same length**:
    each step appends exactly one token to every hypothesis, and hypotheses that
    have finished are retired before expansion. The live beams therefore stack
    into a single ``(n_live, t)`` tensor with no padding, and one
    ``model.decode()`` call replaces ``n_live`` of them. Over a full decode that
    turns O(max_len * beam_size) calls into O(max_len).

    Args:
        model: Transformer model exposing encode() and decode().
        src: Source tensor with shape (1, src_len). Batch size 1 is assumed;
            batching across *sentences* is a separate change.
        beam_size: Number of beams to maintain.
        max_len: Maximum generated length.
        alpha: Length normalization factor (Wu et al., 2016). Applied during
            pruning as well as final selection, matching the reference.
        device: Torch device. Defaults to model device.
        config: TorchLingo Config for special token indices.

    Returns:
        Best decoded token ID sequence (including SOS/EOS). Token-identical to
        the reference implementation.

    Raises:
        ValueError: If src has batch size other than 1, or the model does not
            expose encode/decode.

    Note:
        Tie-breaking follows the rule in the :mod:`torchlingo.inference` module
        docstring, reusing that module's ``_canonical_topk`` and ``_rank_key``
        so the two implementations cannot drift apart on ties.

    Example:
        >>> tokens = beam_search_decode_batched(model, src, beam_size=5)

    See Also:
        :func:`torchlingo.inference.beam_search_decode`: the readable reference.
    """
    cfg = config if config is not None else get_default_config()
    device = device if device is not None else next(model.parameters()).device

    model.eval()

    if src.size(0) != 1:
        raise ValueError("beam_search_decode_batched currently expects batch size = 1")
    if not (hasattr(model, "encode") and hasattr(model, "decode")):
        raise ValueError(
            "beam_search_decode_batched requires a Transformer-style model "
            "with encode/decode"
        )

    src = src.to(device)
    pad_mask = src.eq(cfg.pad_idx)
    with torch.no_grad():
        memory = model.encode(src, src_key_padding_mask=pad_mask)

    beams: list[tuple[list[int], float]] = [([cfg.sos_idx], 0.0)]
    completed: list[tuple[list[int], float]] = []

    for _ in range(max_len):
        # Retire finished hypotheses before expanding, so the survivors are all
        # the same length and need no padding to stack.
        live = [(tokens, score) for tokens, score in beams if tokens[-1] != cfg.eos_idx]
        completed.extend(
            (tokens, score) for tokens, score in beams if tokens[-1] == cfg.eos_idx
        )
        if not live:
            break

        tgt = torch.tensor(
            [tokens for tokens, _ in live], dtype=torch.long, device=device
        )
        n_live, tgt_len = tgt.shape

        # Expand the encoder output to match, rather than re-encoding per beam.
        # expand() is a view: no copy, no extra memory.
        memory_batch = memory.expand(n_live, -1, -1)
        src_mask_batch = pad_mask.expand(n_live, -1)
        tgt_mask = nn.Transformer.generate_square_subsequent_mask(tgt_len).to(device)

        with torch.no_grad():
            out = model.decode(
                tgt,
                memory_batch,
                src_key_padding_mask=src_mask_batch,
                tgt_key_padding_mask=tgt.eq(cfg.pad_idx),
                tgt_mask=tgt_mask,
            )
        log_probs = F.log_softmax(out[:, -1, :], dim=-1)

        candidates: list[tuple[list[int], float]] = []
        for row, (tokens, score) in enumerate(live):
            # Selection stays per-row against the shared helper. It operates on
            # one vocabulary-sized vector and is cheap next to a decoder pass,
            # and keeping it identical to the reference makes the tie-breaking
            # equivalence obvious rather than something to re-argue.
            top_log_probs, top_idx = _canonical_topk(log_probs[row], beam_size)
            for lp, idx in zip(top_log_probs.tolist(), top_idx.tolist()):
                candidates.append((tokens + [idx], score + lp))

        if not candidates:
            break
        candidates.sort(key=lambda item: _rank_key(item[0], item[1], alpha))
        beams = candidates[:beam_size]

        if all(tokens[-1] == cfg.eos_idx for tokens, _ in beams):
            completed.extend(beams)
            break

    completed.extend(beams)
    best_tokens, _ = min(completed, key=lambda item: _rank_key(item[0], item[1], alpha))
    return best_tokens


def translate_batch_fast(
    model: nn.Module,
    sentences: Sequence[str],
    src_vocab: BaseVocab,
    tgt_vocab: BaseVocab,
    decode_strategy: str = "greedy",
    beam_size: int = 5,
    max_len: int = 100,
    device: torch.device | None = None,
    config: Config | None = None,
) -> list[str]:
    """Translate sentences using the optimized decoders.

    Mirrors :func:`torchlingo.inference.translate_batch` exactly, including its
    output, but routes beam decoding through
    :func:`beam_search_decode_batched`. Greedy decoding is already batched in
    the reference and is reused unchanged.

    Args:
        model: Seq2seq model (Transformer or LSTM).
        sentences: Iterable of raw source sentences.
        src_vocab: Vocabulary implementing BaseVocab.encode(add_special_tokens=True).
        tgt_vocab: Vocabulary implementing BaseVocab.decode(skip_special_tokens=True).
        decode_strategy: "greedy" or "beam".
        beam_size: Beam width when decode_strategy == "beam".
        max_len: Maximum generation length.
        device: Torch device. Defaults to model device.
        config: TorchLingo Config.

    Returns:
        List of decoded text strings aligned with input sentences, identical to
        what :func:`torchlingo.inference.translate_batch` would return.

    See Also:
        :func:`torchlingo.inference.translate_batch`: the readable reference.
    """
    from .inference import greedy_decode

    cfg = config if config is not None else get_default_config()
    device = device if device is not None else next(model.parameters()).device

    encoded = [
        torch.tensor(src_vocab.encode(s, add_special_tokens=True), dtype=torch.long)
        for s in sentences
    ]

    batch_limit = max(1, cfg.batch_size)

    pad_idx = getattr(tgt_vocab, "pad_idx", cfg.pad_idx)
    sos_idx = getattr(tgt_vocab, "sos_idx", cfg.sos_idx)
    eos_idx = getattr(tgt_vocab, "eos_idx", cfg.eos_idx)

    outputs: list[list[int]] = []

    for start in range(0, len(encoded), batch_limit):
        batch_tokens = encoded[start : start + batch_limit]
        padded = pad_sequence(batch_tokens, batch_first=True, padding_value=cfg.pad_idx)

        if decode_strategy == "beam":
            for row in padded:
                outputs.append(
                    beam_search_decode_batched(
                        model,
                        row.unsqueeze(0),
                        beam_size=beam_size,
                        max_len=max_len,
                        device=device,
                        config=cfg,
                    )
                )
        else:
            outputs.extend(
                greedy_decode(model, padded, max_len=max_len, device=device, config=cfg)
            )

    decoded: list[str] = []

    def _strip_after_special(tokens: Sequence[int]) -> list[int]:
        cleaned: list[int] = []
        for t in tokens:
            if t in (eos_idx, pad_idx):
                break
            if t == sos_idx:
                continue
            cleaned.append(int(t))
        return cleaned

    for token_ids in outputs:
        cleaned = _strip_after_special(token_ids)
        decoded_text = tgt_vocab.decode(cleaned, skip_special_tokens=True)
        decoded.append(
            decoded_text.strip() if hasattr(decoded_text, "strip") else decoded_text
        )
    return decoded


__all__ = [
    "beam_search_decode_batched",
    "translate_batch_fast",
]
