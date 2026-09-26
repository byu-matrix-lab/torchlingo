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
from dataclasses import dataclass

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
        "torchlingo.inference_fast.translate_batch, which produces "
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


@dataclass
class BeamCandidate:
    """One hypothesis considered at one step of beam search.

    Attributes:
        tokens: The hypothesis, including SOS and any EOS.
        score: Cumulative log probability, unnormalized.
        normalized: The length-normalized score the search actually ranks by.
        kept: Whether this candidate survived pruning into the next step.
    """

    tokens: list[int]
    score: float
    normalized: float
    kept: bool


@dataclass
class BeamStep:
    """Every candidate considered at one step, in the order the search ranked them.

    Attributes:
        step: Zero-based step index.
        candidates: All expansions scored at this step, best first. The first
            ``beam_size`` of them have ``kept=True``.
    """

    step: int
    candidates: list[BeamCandidate]


def _rank_key(tokens: list[int], score: float, alpha: float) -> tuple[float, list[int]]:
    """Build the total-order sort key implementing the tie-breaking rule.

    Sorting ascending by this key puts the preferred hypothesis first: highest
    length-normalized score, then lowest token IDs. Because no two live
    hypotheses share a token sequence, the ordering is total - never ambiguous.

    **Where alpha can and cannot act.** The divisor depends only on length, so
    among candidates of equal length it is a shared positive constant and the
    ordering is the same for every alpha. Beam search calls this at two sites,
    and the distinction matters:

    - *Pruning*, where every candidate has just been extended by one token from
      a set of equal-length beams, so they are all the same length and alpha
      changes nothing. ``tests/test_length_normalization.py`` pins that
      invariant, because it depends on finished hypotheses leaving ``beams``
      rather than lingering at a shorter length.
    - *Final selection*, among finished hypotheses that stopped at different
      steps and therefore do differ in length. This is the only place alpha has
      an effect -- and it is a real one: on the pretrained checkpoint it changes
      the returned translation for 15 of 120 sentences at alpha=0 versus the
      0.6 default, and 88 of 120 at alpha=1.5.

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


def attention_for_sequence(
    model: nn.Module,
    src: torch.Tensor,
    tokens: Sequence[int],
    device: torch.device | None = None,
    config: Config | None = None,
) -> torch.Tensor:
    """Cross-attention for a sequence the model generated.

    Both decoders compute attention at every step and throw it away, so this
    recovers it with one teacher-forced pass over the finished sequence rather
    than threading weights through the search.

    **That second pass is exact, not an approximation.** The decoder is causally
    masked, so the state at target position *t* depends only on tokens up to
    *t*; re-running over the whole sequence reproduces each row exactly as the
    incremental decode computed it. Verified to floating-point noise in
    ``tests/test_attention_from_decoding.py``.

    It matters that this is the *generated* sequence. Teacher-forcing the
    **reference** translation, which is what you get by calling the model
    directly, shows you where attention would have gone had the model produced
    the right answer. That is a different and less interesting picture,
    especially on a model whose output is wrong.

    Args:
        model (nn.Module): A Transformer or attention-equipped LSTM.
        src (torch.Tensor): Source token ids, ``(src_len,)`` or ``(1, src_len)``.
        tokens (Sequence[int]): The generated sequence, including ``<sos>`` and
            any trailing ``<eos>``, as returned by the decoders here.
        device (torch.device, optional): Defaults to the model's device.
        config (Config, optional): Supplies ``pad_idx``.

    Returns:
        torch.Tensor: Weights of shape ``(len(tokens) - 1, src_len)``. Row *i*
        is the distribution over source positions while producing
        ``tokens[i + 1]``.

    Raises:
        ValueError: If the model cannot return attention, or ``tokens`` is too
            short to have generated anything.

    Example:
        >>> import torch
        >>> from torchlingo.models import SimpleTransformer
        >>> model = SimpleTransformer(src_vocab_size=30, tgt_vocab_size=30,
        ...                           d_model=16, n_heads=2,
        ...                           num_encoder_layers=1, num_decoder_layers=1)
        >>> _ = model.eval()
        >>> src = torch.tensor([[2, 5, 9, 3]])
        >>> tokens = greedy_decode(model, src, max_len=5)[0]
        >>> weights = attention_for_sequence(model, src, tokens)
        >>> weights.shape[1]
        4
    """
    cfg = config if config is not None else get_default_config()
    if len(tokens) < 2:
        raise ValueError(
            f"tokens has length {len(tokens)}; it must contain <sos> and at "
            "least one generated token for there to be any attention to show."
        )
    if tokens[0] != cfg.sos_idx:
        # Not fatal, but the rows would be labelled wrongly: this function
        # assumes tokens[1:] are the generated ones, so a sequence missing its
        # <sos> shifts every row by one against the tokens a caller plots.
        warnings.warn(
            f"tokens starts with {tokens[0]}, not sos_idx={cfg.sos_idx}. "
            "Row i is the attention while producing tokens[i + 1], so a "
            "sequence without a leading <sos> will be labelled off by one.",
            UserWarning,
            stacklevel=2,
        )

    device = device if device is not None else next(model.parameters()).device
    if src.dim() == 1:
        src = src.unsqueeze(0)
    if src.size(0) != 1:
        raise ValueError(
            f"src has batch size {src.size(0)}; pass one sentence at a time, "
            "since generated sequences differ in length and cannot be stacked."
        )
    src = src.to(device)

    # Drop the final token: the decoder is fed tokens[:-1] and predicts
    # tokens[1:], so the last one is an output with no corresponding input row.
    tgt_in = torch.tensor([list(tokens[:-1])], device=device, dtype=torch.long)

    was_training = model.training
    model.eval()
    try:
        with torch.no_grad():
            result = model(src, tgt_in, return_attention=True)
    except TypeError as exc:
        raise ValueError(
            f"{type(model).__name__} does not accept return_attention. A "
            "SimpleSeq2SeqLSTM needs attention=True, and an LSTM built without "
            "it has no cross-attention to report."
        ) from exc
    finally:
        if was_training:
            model.train()

    if not isinstance(result, tuple):
        # ValueError, not TypeError: the argument types were fine, the model is
        # simply unsuitable for what was asked. Same reasoning as the other
        # raises here, and the tests pin it.
        raise ValueError(  # noqa: TRY004
            f"{type(model).__name__} accepted return_attention but returned no "
            "weights. An LSTM built with attention=False computes none."
        )
    _, weights = result
    if weights is None:
        raise ValueError(
            f"{type(model).__name__} returned None for attention. An LSTM "
            "built with attention=False has no cross-attention to report."
        )
    return weights[0]


def greedy_decode(
    model: nn.Module,
    src: torch.Tensor,
    max_len: int | None = None,
    device: torch.device | None = None,
    config: Config | None = None,
    return_attention: bool = False,
) -> list[list[int]] | tuple[list[list[int]], list[torch.Tensor]]:
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
        return_attention: Also return the cross-attention for each decoded
            sequence, showing where the decoder looked while producing the
            translation it actually produced. Off by default: it costs one
            extra forward pass per sentence.

    Returns:
        List of decoded token ID sequences (one list per batch element). If
        ``return_attention`` is True, returns ``(sequences, weights)`` where
        weights is a **list** of ``(tgt_len, src_len)`` tensors, one per
        sentence. A list rather than a stacked tensor because decoded
        sequences differ in length and padding them would invent attention
        rows that were never computed.

    Example:
        >>> import torch
        >>> from torchlingo.models import SimpleTransformer
        >>> model = SimpleTransformer(src_vocab_size=30, tgt_vocab_size=30,
        ...                           d_model=16, n_heads=2,
        ...                           num_encoder_layers=1, num_decoder_layers=1)
        >>> _ = model.eval()
        >>> src = torch.tensor([[2, 5, 9, 3]])
        >>> tokens, weights = greedy_decode(model, src, max_len=5,
        ...                                 return_attention=True)
        >>> len(weights) == len(tokens)
        True
    """

    cfg = config if config is not None else get_default_config()
    # One source of truth for how long a generation may run. Carrying a literal
    # here is how this drifted: the decoders said 100 and evaluate_model said 200,
    # so two BLEU numbers for one model could differ on any target between them.
    max_len = max_len if max_len is not None else cfg.max_decode_length
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
        # A model exposing encode_source/decode_step drives its own decoder, so
        # the encoder outputs -- and any attention over them -- reach this path.
        # Reimplementing the loop here is what silently dropped them before.
        # Duck-typed LSTM models without those methods keep the inline loop.
        uses_model_step = hasattr(model, "encode_source") and hasattr(
            model, "decode_step"
        )
        with torch.no_grad():
            if uses_model_step:
                enc_out, hidden, src_pad_mask = model.encode_source(src_chunk)
            else:
                src_emb = model.src_embed(src_chunk)
                _, hidden = model.encoder(src_emb)
                enc_out = None
                src_pad_mask = None

            batch_size = src_chunk.size(0)
            ys = torch.full(
                (batch_size, 1), cfg.sos_idx, device=device, dtype=torch.long
            )
            finished = torch.zeros(batch_size, dtype=torch.bool, device=device)

            tgt_embed = getattr(model, "tgt_embed", None) or getattr(
                model, "src_embed", None
            )

            for _ in range(max_len):
                last_token = ys[:, -1].unsqueeze(1)
                if uses_model_step:
                    logits, hidden, _weights = model.decode_step(
                        last_token, hidden, enc_out, src_pad_mask
                    )
                else:
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

    if not return_attention:
        return decoded

    # One sentence at a time: each has its own length, and the padding that
    # makes a batch rectangular would show up as attention over positions the
    # decode never saw.
    weights = [
        attention_for_sequence(model, src[i], tokens, device=device, config=cfg)
        for i, tokens in enumerate(decoded)
    ]
    return decoded, weights


def beam_search_decode(
    model: nn.Module,
    src: torch.Tensor,
    beam_size: int = 5,
    max_len: int | None = None,
    alpha: float = 0.6,
    device: torch.device | None = None,
    config: Config | None = None,
    trace: list[BeamStep] | None = None,
    return_attention: bool = False,
) -> list[int] | tuple[list[int], torch.Tensor]:
    """Beam search decoding for Transformer or LSTM models.

    The search itself is architecture-agnostic — beams, pruning, length
    normalization and tie-breaking are identical either way. Only the step that
    scores the next token given a prefix differs, which is worth noticing: beam
    search is a property of *decoding*, not of the model that does it.

    Args:
        model: Seq2seq model. Transformer models must expose encode/decode;
            the LSTM path uses encode_source/decode_prefix, present on
            SimpleSeq2SeqLSTM.
        src: Source tensor with shape (1, src_len). Batch size 1 is assumed.
        beam_size: Number of beams to maintain.
        max_len: Maximum generated length.
        alpha: Length normalization factor (Wu et al., 2016). Only affects which
            finished hypothesis is returned, never which ones survive pruning:
            every candidate within a step has the same length, so the divisor is
            a shared constant there and cannot reorder them. See
            :func:`_rank_key`. This is therefore equivalent to the conventional
            "normalize at final selection only", despite the key being applied
            at both sites.
        device: Torch device. Defaults to model device.
        config: TorchLingo Config for special token indices.
        trace: If a list is given, one :class:`BeamStep` per step is appended to
            it, recording every candidate considered and whether it survived
            pruning. Costs nothing when omitted and never changes the result;
            see :func:`torchlingo.visualization.format_beam_search`.
        return_attention: Also return the cross-attention for the **winning**
            hypothesis. Costs one extra forward pass.

    Returns:
        Best decoded token ID sequence (including SOS/EOS). If
        ``return_attention`` is True, returns ``(tokens, weights)`` with
        weights of shape ``(len(tokens) - 1, src_len)``.

    Note:
        Attention is recovered by re-running the winner through
        :func:`attention_for_sequence` after the search finishes, rather than
        by carrying weight history on every beam. That is deliberate.
        Hypotheses get pruned, so most of the weights computed during a beam
        search belong to candidates that lost; keeping all of them to discard
        all but one costs memory proportional to ``beam_size`` for no benefit.
        Re-running is exact, because the decoder is causally masked.

        This mirrors what the search already does with scores: it re-scores
        prefixes rather than caching every partial result.

    Note:
        Output is deterministic for a fixed model and input on any device.
        Exact score ties are broken toward the lower token IDs; see the
        module docstring for the full rule.

    Note:
        This is the **reference** implementation, written to be read: the whole
        search is visible in about 40 lines. It issues one ``model.decode()``
        call per beam per step, which is roughly ``beam_size`` times more calls
        than necessary. For decoding a real test set, prefer
        :func:`torchlingo.inference_fast.beam_search_decode`, which has
        the same signature and returns token-identical output.

    See Also:
        :func:`torchlingo.inference_fast.beam_search_decode`: the
        batched counterpart, same output and substantially faster.
    """

    cfg = config if config is not None else get_default_config()
    # One source of truth for how long a generation may run. Carrying a literal
    # here is how this drifted: the decoders said 100 and evaluate_model said 200,
    # so two BLEU numbers for one model could differ on any target between them.
    max_len = max_len if max_len is not None else cfg.max_decode_length
    device = device if device is not None else next(model.parameters()).device

    model.eval()

    if src.size(0) != 1:
        raise ValueError("beam_search_decode currently expects batch size = 1")

    is_transformer = hasattr(model, "encode") and hasattr(model, "decode")
    is_lstm = hasattr(model, "encode_source") and hasattr(model, "decode_prefix")
    if not (is_transformer or is_lstm):
        raise ValueError(
            "beam_search_decode requires a Transformer-style model with "
            "encode/decode, or an LSTM model with encode_source/decode_prefix."
        )

    src = src.to(device)
    pad_mask = src.eq(cfg.pad_idx)

    # The search below is architecture-agnostic. The only thing that differs is
    # how you score the next token given a prefix, so that is all we specialize:
    # encode once, then hand the loop a function from prefix to log-probs.
    if is_transformer:
        with torch.no_grad():
            memory = model.encode(src, src_key_padding_mask=pad_mask)

        def next_log_probs(tokens: list[int]) -> torch.Tensor:
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
            return F.log_softmax(out[0, -1, :], dim=-1)
    else:
        enc_out, init_hidden, src_pad_mask = model.encode_source(src)

        def next_log_probs(tokens: list[int]) -> torch.Tensor:
            # Re-scored from the encoder's initial state every step, exactly as
            # the Transformer path recomputes its prefix. Carrying per-beam
            # (h, c) instead would be faster and would put recurrent-state
            # bookkeeping into the implementation meant to stay readable.
            tgt = torch.tensor([tokens], dtype=torch.long, device=device)
            with torch.no_grad():
                out, _hidden, _weights = model.decode_prefix(
                    tgt, init_hidden, enc_out, src_pad_mask
                )
            return F.log_softmax(out[0, -1, :], dim=-1)

    beams: list[tuple[list[int], float]] = [([cfg.sos_idx], 0.0)]
    completed: list[tuple[list[int], float]] = []

    for _ in range(max_len):
        candidates: list[tuple[list[int], float]] = []
        for tokens, score in beams:
            if tokens[-1] == cfg.eos_idx:
                completed.append((tokens, score))
                continue
            log_probs = next_log_probs(tokens)
            top_log_probs, top_idx = _canonical_topk(log_probs, beam_size)
            for lp, idx in zip(top_log_probs.tolist(), top_idx.tolist()):
                candidates.append((tokens + [idx], score + lp))

        if not candidates:
            break
        # Ascending by _rank_key puts the preferred hypothesis first; the
        # token sequence in the key makes the order total, so exact score ties
        # resolve identically on every device.
        candidates.sort(key=lambda item: _rank_key(item[0], item[1], alpha))

        # Record before pruning, because what was discarded is the interesting
        # half: the sort order above is exactly the ranking, so the first
        # beam_size entries are the survivors.
        if trace is not None:
            trace.append(
                BeamStep(
                    step=len(trace),
                    candidates=[
                        BeamCandidate(
                            tokens=list(tokens),
                            score=score,
                            normalized=-_rank_key(tokens, score, alpha)[0],
                            kept=rank < beam_size,
                        )
                        for rank, (tokens, score) in enumerate(candidates)
                    ],
                )
            )

        beams = candidates[:beam_size]

        if all(tokens[-1] == cfg.eos_idx for tokens, _ in beams):
            completed.extend(beams)
            break

    completed.extend(beams)
    best_tokens, _ = min(completed, key=lambda item: _rank_key(item[0], item[1], alpha))
    if not return_attention:
        return best_tokens
    return best_tokens, attention_for_sequence(
        model, src, best_tokens, device=device, config=cfg
    )


def translate_batch(
    model: nn.Module,
    sentences: Sequence[str],
    src_vocab: BaseVocab,
    tgt_vocab: BaseVocab,
    decode_strategy: str = "greedy",
    beam_size: int = 5,
    max_len: int | None = None,
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
        decode_strategy: "greedy" or "beam". Defaults to "greedy" -- beam
            search is opt-in, and usually produces better translations.
        beam_size: Beam width when decode_strategy == "beam".
        max_len: Maximum generation length.
        device: Torch device. Defaults to model device.
        config: TorchLingo Config.

    Returns:
        List of decoded text strings aligned with input sentences.

    Warns:
        UserWarning: Once per process, when beam decoding more than
            ``_LARGE_INPUT_WARN_THRESHOLD`` sentences, pointing at
            :func:`torchlingo.inference_fast.translate_batch`.

    See Also:
        :func:`torchlingo.inference_fast.translate_batch`: same output,
        substantially faster for beam decoding on large inputs.
    """

    cfg = config if config is not None else get_default_config()
    # One source of truth for how long a generation may run. Carrying a literal
    # here is how this drifted: the decoders said 100 and evaluate_model said 200,
    # so two BLEU numbers for one model could differ on any target between them.
    max_len = max_len if max_len is not None else cfg.max_decode_length
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
