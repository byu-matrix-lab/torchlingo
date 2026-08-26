"""Inference utilities for TorchLingo models.

Provides decoding helpers and a batch translation convenience wrapper that
work across Transformer-style and LSTM seq2seq models. These functions are
kept separate from training logic to keep responsibilities focused.

Tie-breaking rule
-----------------
Decoding must be reproducible: for a fixed model and input, the output is
always identical, on any device, with any PyTorch build. Scores tie often in
practice - early in training, with padded or degenerate inputs, and whenever
several continuations are genuinely equally likely - so the rule is explicit:

    Prefer the higher score. Among exactly equal scores, prefer the
    sequence with the lower token IDs, comparing position by position.

``torch.argmax`` already documents that it returns the *first* maximal index,
so greedy decoding satisfies this rule as written. ``torch.topk`` explicitly
does **not** guarantee an order for tied elements, so beam search must not
rely on it; :func:`_canonical_topk` restores a deterministic order without
depending on backend behavior.

Any batched reimplementation of beam search must preserve this rule. Selecting
with ``topk`` over a flattened ``(batch * beam, vocab)`` tensor will otherwise
break ties differently and silently change output.
"""

from __future__ import annotations

import warnings
from collections.abc import Sequence

import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.utils.rnn import pad_sequence

from .config import Config, get_default_config
from .data_processing.vocab import BaseVocab

# Beam decoding more sentences than this through the reference implementation is
# slow enough to be worth flagging. Tuned to stay quiet for the tutorials and
# test suites, and to fire on anything resembling a real evaluation set.
_LARGE_INPUT_WARN_THRESHOLD = 100

_warned_about_large_beam_input = False


def _warn_if_large_beam_input(num_sentences: int) -> None:
    """Point the user at the fast path when beam decoding a large input.

    Fires at most once per process. A student running the reference beam search
    over an evaluation set will otherwise conclude that beam search is
    impractical and fall back to greedy, which is the failure this exists to
    prevent -- and they will reach that conclusion mid-experiment, not while
    reading documentation.

    Args:
        num_sentences: How many sentences are about to be beam decoded.
    """
    global _warned_about_large_beam_input
    if _warned_about_large_beam_input or num_sentences <= _LARGE_INPUT_WARN_THRESHOLD:
        return
    _warned_about_large_beam_input = True
    warnings.warn(
        f"Beam decoding {num_sentences} sentences with the reference "
        "implementation, which evaluates one beam per model call and is "
        "written for readability rather than speed. For inputs this size use "
        "torchlingo.inference_fast.translate_batch_fast, which produces "
        "identical output. Silence with "
        "warnings.filterwarnings('ignore', module='torchlingo.inference').",
        UserWarning,
        stacklevel=3,
    )


def _canonical_topk(
    log_probs: torch.Tensor, k: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Select the ``k`` highest scores, breaking exact ties by lowest index.

    ``torch.topk`` documents that the indices of tied elements are not
    guaranteed to be stable, which makes beam search output device-dependent
    whenever scores tie. This helper takes the same k elements by value but
    fixes the order: strictly better scores first, then the lowest token IDs
    among any tied at the selection boundary.

    Only the boundary tie set is sorted, so the extra cost over ``topk`` is
    proportional to ``k`` rather than to the vocabulary size.

    Args:
        log_probs: 1-D tensor of scores, one per vocabulary entry.
        k: Number of entries to select. Clamped to the tensor length.

    Returns:
        Tuple of (scores, indices), both length ``min(k, len(log_probs))``,
        ordered by descending score then ascending index.

    Example:
        >>> scores = torch.tensor([0.0, 5.0, 5.0, 5.0])
        >>> _canonical_topk(scores, 2)[1].tolist()
        [1, 2]
    """
    k = min(k, log_probs.numel())
    threshold = torch.topk(log_probs, k).values[-1]

    # Everything that could belong in the top-k, including boundary ties.
    candidates = (log_probs >= threshold).nonzero(as_tuple=False).flatten()
    candidates, _ = torch.sort(candidates)  # explicit ascending index order

    # Stable sort keeps that ascending order among equal scores.
    order = torch.argsort(-log_probs[candidates], stable=True)
    chosen = candidates[order][:k]
    return log_probs[chosen], chosen


def _rank_key(tokens: list[int], score: float, alpha: float) -> tuple[float, list[int]]:
    """Build the total-order sort key implementing the tie-breaking rule.

    Sorting ascending by this key puts the preferred hypothesis first: highest
    length-normalized score, then lowest token IDs. Because no two live
    hypotheses share a token sequence, the ordering is total - never ambiguous.

    Args:
        tokens: Hypothesis token IDs.
        score: Cumulative (unnormalized) log probability.
        alpha: Length-normalization strength (Wu et al., 2016).

    Returns:
        Tuple of (negated normalized score, tokens) suitable as a sort key.
    """
    length = len(tokens)
    normalized = score / (((5 + length) / 6) ** alpha)
    return (-normalized, tokens)


def greedy_decode(
    model: nn.Module,
    src: torch.Tensor,
    max_len: int = 100,
    device: torch.device | None = None,
    config: Config | None = None,
) -> list[list[int]]:
    """Greedy autoregressive decoding for Transformer or LSTM models.

    Decodes in mini-batches sized by ``config.batch_size`` to limit device
    memory usage when handling large input batches.

    Args:
        model: Seq2seq model. Transformer models must expose encode/decode; LSTM
            path uses src_embed/encoder/decoder/output modules present in
            SimpleSeq2SeqLSTM.
        src: Source tensor (batch, src_len).
        max_len: Maximum decoded length (including SOS/EOS).
        device: Torch device. Defaults to model's device or CUDA/CPU.
        config: TorchLingo Config providing special token indices.

    Returns:
        List of decoded token ID sequences (one list per batch element).
    """

    cfg = config if config is not None else get_default_config()
    device = device if device is not None else next(model.parameters()).device

    model.eval()

    batch_limit = max(1, cfg.batch_size)

    is_transformer = hasattr(model, "encode") and hasattr(model, "decode")
    is_lstm = all(
        hasattr(model, attr) for attr in ("src_embed", "encoder", "decoder", "output")
    )

    if not (is_transformer or is_lstm):
        raise ValueError(
            "Model must expose encode/decode or LSTM modules for greedy decoding."
        )

    def _decode_transformer(src_chunk: torch.Tensor) -> list[list[int]]:
        src_chunk = src_chunk.to(device)
        pad_mask = src_chunk.eq(cfg.pad_idx)
        with torch.no_grad():
            memory = model.encode(src_chunk, src_key_padding_mask=pad_mask)
            ys = torch.full(
                (src_chunk.size(0), 1), cfg.sos_idx, device=device, dtype=torch.long
            )
            finished = torch.zeros(src_chunk.size(0), dtype=torch.bool, device=device)
            for _ in range(max_len):
                tgt_mask = nn.Transformer.generate_square_subsequent_mask(
                    ys.size(1)
                ).to(device)
                out = model.decode(
                    ys,
                    memory,
                    src_key_padding_mask=pad_mask,
                    tgt_key_padding_mask=ys.eq(cfg.pad_idx),
                    tgt_mask=tgt_mask,
                )
                next_token = out[:, -1, :].argmax(-1)
                ys = torch.cat([ys, next_token.unsqueeze(1)], dim=1)
                finished |= next_token.eq(cfg.eos_idx)
                if finished.all():
                    break
        return ys.cpu().tolist()

    def _decode_lstm(src_chunk: torch.Tensor) -> list[list[int]]:
        src_chunk = src_chunk.to(device)
        with torch.no_grad():
            src_emb = model.src_embed(src_chunk)
            _, (h, c) = model.encoder(src_emb)

            batch_size = src_chunk.size(0)
            ys = torch.full(
                (batch_size, 1), cfg.sos_idx, device=device, dtype=torch.long
            )
            finished = torch.zeros(batch_size, dtype=torch.bool, device=device)

            hidden = (h, c)
            tgt_embed = getattr(model, "tgt_embed", None) or getattr(
                model, "src_embed", None
            )

            for _ in range(max_len):
                last_token = ys[:, -1].unsqueeze(1)
                emb = tgt_embed(last_token)
                dec_out, hidden = model.decoder(emb, hidden)
                logits = model.output(dec_out)
                next_token = logits[:, -1, :].argmax(-1)
                ys = torch.cat([ys, next_token.unsqueeze(1)], dim=1)
                finished |= next_token.eq(cfg.eos_idx)
                if finished.all():
                    break
        return ys.cpu().tolist()

    decoded: list[list[int]] = []
    for src_chunk in src.split(batch_limit):
        if is_transformer:
            decoded.extend(_decode_transformer(src_chunk))
        else:
            decoded.extend(_decode_lstm(src_chunk))

    return decoded


def beam_search_decode(
    model: nn.Module,
    src: torch.Tensor,
    beam_size: int = 5,
    max_len: int = 100,
    alpha: float = 0.6,
    device: torch.device | None = None,
    config: Config | None = None,
) -> list[int]:
    """Beam search decoding for Transformer-style models.

    Args:
        model: Transformer model exposing encode() and decode().
        src: Source tensor with shape (1, src_len). Batch size 1 is assumed.
        beam_size: Number of beams to maintain.
        max_len: Maximum generated length.
        alpha: Length normalization factor (Wu et al., 2016). Applied during
            pruning as well as final selection, so it shapes which hypotheses
            survive rather than only which one is returned.
        device: Torch device. Defaults to model device.
        config: TorchLingo Config for special token indices.

    Returns:
        Best decoded token ID sequence (including SOS/EOS).

    Note:
        Output is deterministic for a fixed model and input on any device.
        Exact score ties are broken toward the lower token IDs; see the
        module docstring for the full rule.

    Note:
        This is the **reference** implementation, written to be read: the whole
        search is visible in about 40 lines. It issues one ``model.decode()``
        call per beam per step, which is roughly ``beam_size`` times more calls
        than necessary. For decoding a real test set, prefer
        :func:`torchlingo.inference_fast.beam_search_decode_batched`, which has
        the same signature and returns token-identical output.

    See Also:
        :func:`torchlingo.inference_fast.beam_search_decode_batched`: the
        batched counterpart, same output and substantially faster.
    """

    cfg = config if config is not None else get_default_config()
    device = device if device is not None else next(model.parameters()).device

    model.eval()

    if src.size(0) != 1:
        raise ValueError("beam_search_decode currently expects batch size = 1")
    if not (hasattr(model, "encode") and hasattr(model, "decode")):
        raise ValueError(
            "beam_search_decode requires a Transformer-style model with encode/decode"
        )

    src = src.to(device)
    pad_mask = src.eq(cfg.pad_idx)
    with torch.no_grad():
        memory = model.encode(src, src_key_padding_mask=pad_mask)

    beams: list[tuple[list[int], float]] = [([cfg.sos_idx], 0.0)]
    completed: list[tuple[list[int], float]] = []

    for _ in range(max_len):
        candidates: list[tuple[list[int], float]] = []
        for tokens, score in beams:
            if tokens[-1] == cfg.eos_idx:
                completed.append((tokens, score))
                continue
            tgt = torch.tensor([tokens], dtype=torch.long, device=device)
            tgt_mask = nn.Transformer.generate_square_subsequent_mask(len(tokens)).to(
                device
            )
            with torch.no_grad():
                out = model.decode(
                    tgt,
                    memory,
                    src_key_padding_mask=pad_mask,
                    tgt_key_padding_mask=tgt.eq(cfg.pad_idx),
                    tgt_mask=tgt_mask,
                )
            log_probs = F.log_softmax(out[0, -1, :], dim=-1)
            top_log_probs, top_idx = _canonical_topk(log_probs, beam_size)
            for lp, idx in zip(top_log_probs.tolist(), top_idx.tolist()):
                candidates.append((tokens + [idx], score + lp))

        if not candidates:
            break
        # Ascending by _rank_key puts the preferred hypothesis first; the
        # token sequence in the key makes the order total, so exact score ties
        # resolve identically on every device.
        candidates.sort(key=lambda item: _rank_key(item[0], item[1], alpha))
        beams = candidates[:beam_size]

        if all(tokens[-1] == cfg.eos_idx for tokens, _ in beams):
            completed.extend(beams)
            break

    completed.extend(beams)
    best_tokens, _ = min(completed, key=lambda item: _rank_key(item[0], item[1], alpha))
    return best_tokens


def translate_batch(
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
    """Translate a batch of raw sentences using provided vocabularies.

    Processes inputs in chunks of ``config.batch_size`` to avoid moving very
    large padded batches to the device at once.

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
        List of decoded text strings aligned with input sentences.

    Warns:
        UserWarning: Once per process, when beam decoding more than
            ``_LARGE_INPUT_WARN_THRESHOLD`` sentences, pointing at
            :func:`torchlingo.inference_fast.translate_batch_fast`.

    See Also:
        :func:`torchlingo.inference_fast.translate_batch_fast`: same output,
        substantially faster for beam decoding on large inputs.
    """

    cfg = config if config is not None else get_default_config()
    device = device if device is not None else next(model.parameters()).device

    if decode_strategy == "beam":
        _warn_if_large_beam_input(len(sentences))

    encoded = [
        torch.tensor(src_vocab.encode(s, add_special_tokens=True), dtype=torch.long)
        for s in sentences
    ]

    batch_limit = max(1, cfg.batch_size)

    # Prefer special-token indices carried by the vocabulary (SentencePiece models
    # often encode these directly) to avoid mismatches with an unrelated config.
    pad_idx = getattr(tgt_vocab, "pad_idx", cfg.pad_idx)
    sos_idx = getattr(tgt_vocab, "sos_idx", cfg.sos_idx)
    eos_idx = getattr(tgt_vocab, "eos_idx", cfg.eos_idx)

    outputs: list[list[int]] = []

    for start in range(0, len(encoded), batch_limit):
        batch_tokens = encoded[start : start + batch_limit]
        padded = pad_sequence(batch_tokens, batch_first=True, padding_value=cfg.pad_idx)

        if decode_strategy == "beam":
            for row in padded:
                tokens = beam_search_decode(
                    model,
                    row.unsqueeze(0),
                    beam_size=beam_size,
                    max_len=max_len,
                    device=device,
                    config=cfg,
                )
                outputs.append(tokens)
        else:
            outputs.extend(
                greedy_decode(model, padded, max_len=max_len, device=device, config=cfg)
            )

    decoded: list[str] = []

    def _strip_after_special(tokens: Sequence[int]) -> list[int]:
        # Drop SOS and everything after the first EOS/PAD using vocab-aware IDs.
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
        # Let the vocab handle decoding (SentencePiece will rejoin subwords correctly).
        decoded_text = tgt_vocab.decode(cleaned, skip_special_tokens=True)
        decoded.append(
            decoded_text.strip() if hasattr(decoded_text, "strip") else decoded_text
        )
    return decoded


__all__ = [
    "beam_search_decode",
    "greedy_decode",
    "translate_batch",
]
