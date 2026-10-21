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

The two random strategies, ``"sample"`` and ``"beam+noise"``, are reproducible
in a different sense: their output is fixed by a ``seed``, and each sentence has
its own random stream, so a sentence comes out the same whatever batch or chunk
it is decoded in.
"""

from __future__ import annotations

import hashlib
import json
import os
import random
import warnings
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.utils.rnn import pad_sequence

from .config import Config, get_default_config
from .data_processing.vocab import BaseVocab
from .models.transformer_simple import create_causal_mask

# Every value decode_strategy accepts. "sample" and "beam+noise" are the two that
# Edunov et al. (2018) found make better back-translations; see sample_decode and
# add_noise.
DECODE_STRATEGIES = ("greedy", "beam", "sample", "beam+noise")
_BEAM_STRATEGIES = ("beam", "beam+noise")

# The word add_noise substitutes for a replaced one. Not a vocabulary's own unknown
# token: noise is applied to text, and a forward model's tokenizer would split a
# literal "<unk>" into pieces anyway. What matters is that it is always the same.
NOISE_FILLER = "<blank>"

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


def _check_decodable(model: nn.Module, what: str) -> None:
    """Raise unless ``model`` exposes a Transformer or an LSTM decoding interface."""
    is_transformer = hasattr(model, "encode") and hasattr(model, "decode")
    is_lstm = all(
        hasattr(model, attr) for attr in ("src_embed", "encoder", "decoder", "output")
    )
    if not (is_transformer or is_lstm):
        raise ValueError(f"Model must expose encode/decode or LSTM modules for {what}.")


def _check_strategy(decode_strategy: str) -> None:
    """Raise on a strategy name this module does not implement.

    Before 2026-10-21 any unrecognized name, ``"Beam"`` included, quietly decoded
    greedily, and the result was labelled with the name that was asked for.
    """
    if decode_strategy not in DECODE_STRATEGIES:
        raise ValueError(
            f"Unknown decode_strategy {decode_strategy!r}; "
            f"choose one of {', '.join(repr(s) for s in DECODE_STRATEGIES)}."
        )


def _sentence_seed(seed: int, index: int) -> int:
    """The seed for sentence ``index`` of a job seeded with ``seed``.

    Hashed rather than ``seed + index``, so jobs seeded 0 and 1 do not share all
    but one of their sentences' random streams.
    """
    digest = hashlib.sha256(f"{seed}:{index}".encode()).digest()
    return int.from_bytes(digest[:8], "little") >> 1


def _sentence_seeds(seed: int | Sequence[int], count: int) -> list[int]:
    """One seed per sentence: derived from an int, or given explicitly as a list."""
    if isinstance(seed, int):
        return [_sentence_seed(seed, i) for i in range(count)]
    seeds = [int(s) for s in seed]
    if len(seeds) != count:
        raise ValueError(f"Got {len(seeds)} seeds for {count} sentences.")
    return seeds


def _decode_steps(
    model: nn.Module,
    src_chunk: torch.Tensor,
    max_len: int,
    device: torch.device,
    cfg: Config,
    choose: Callable[[torch.Tensor, int], torch.Tensor],
) -> list[list[int]]:
    """Run the decoder one token at a time, letting ``choose`` pick each token.

    Greedy decoding and sampling differ in one line: given the logits for the next
    position, ``(batch, vocab)``, and the step number, which token comes next.
    Everything else -- encoding once, growing the prefix, stopping at EOS -- is this
    loop, written once for both.
    """
    is_transformer = hasattr(model, "encode") and hasattr(model, "decode")
    src_chunk = src_chunk.to(device)
    batch_size = src_chunk.size(0)
    ys = torch.full((batch_size, 1), cfg.sos_idx, device=device, dtype=torch.long)
    finished = torch.zeros(batch_size, dtype=torch.bool, device=device)

    with torch.no_grad():
        if is_transformer:
            pad_mask = src_chunk.eq(cfg.pad_idx)
            memory = model.encode(src_chunk, src_key_padding_mask=pad_mask)
        else:
            # A model exposing encode_source/decode_step drives its own decoder,
            # so the encoder outputs -- and any attention over them -- reach this
            # path. Reimplementing the loop here is what silently dropped them
            # before. Duck-typed LSTM models without those methods keep the
            # inline loop.
            uses_model_step = hasattr(model, "encode_source") and hasattr(
                model, "decode_step"
            )
            if uses_model_step:
                enc_out, hidden, src_pad_mask = model.encode_source(src_chunk)
            else:
                src_emb = model.src_embed(src_chunk)
                _, hidden = model.encoder(src_emb)
            tgt_embed = getattr(model, "tgt_embed", None) or getattr(
                model, "src_embed", None
            )

        for step in range(max_len):
            if is_transformer:
                # The model's own helper, not
                # nn.Transformer.generate_square_subsequent_mask, which returns
                # a float mask. The padding masks below are boolean, and PyTorch
                # deprecated mixing the two: passing a float attn_mask beside a
                # boolean key_padding_mask warns on every call and is scheduled
                # for removal. Both are boolean now, which is also what the
                # training path has always used.
                tgt_mask = create_causal_mask(ys.size(1), device)
                logits = model.decode(
                    ys,
                    memory,
                    src_key_padding_mask=pad_mask,
                    tgt_key_padding_mask=ys.eq(cfg.pad_idx),
                    tgt_mask=tgt_mask,
                )
            else:
                last_token = ys[:, -1].unsqueeze(1)
                if uses_model_step:
                    logits, hidden, _weights = model.decode_step(
                        last_token, hidden, enc_out, src_pad_mask
                    )
                else:
                    emb = tgt_embed(last_token)
                    dec_out, hidden = model.decoder(emb, hidden)
                    logits = model.output(dec_out)
            next_token = choose(logits[:, -1, :], step)
            ys = torch.cat([ys, next_token.unsqueeze(1)], dim=1)
            finished |= next_token.eq(cfg.eos_idx)
            if finished.all():
                break
    return ys.cpu().tolist()


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
    _check_decodable(model, "greedy decoding")

    batch_limit = max(1, cfg.batch_size)

    def most_likely(logits: torch.Tensor, step: int) -> torch.Tensor:
        return logits.argmax(-1)

    decoded: list[list[int]] = []
    for src_chunk in src.split(batch_limit):
        decoded.extend(
            _decode_steps(model, src_chunk, max_len, device, cfg, most_likely)
        )

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


def _sample_tokens(
    logits: torch.Tensor,
    draws: torch.Tensor,
    temperature: float,
    top_k: int | None,
) -> torch.Tensor:
    """Pick one token per row of ``logits``, using that row's uniform draw in [0, 1).

    Inverse-CDF sampling: lay the probabilities end to end on [0, 1) and take the
    token whose stretch the draw lands in. A likely token owns a long stretch, so it
    is hit often; an unlikely one, rarely.
    """
    logits = logits.float() / temperature
    if top_k is not None and top_k < logits.size(-1):
        kth_best = logits.topk(top_k, dim=-1).values[:, -1:]
        logits = logits.masked_fill(logits < kth_best, float("-inf"))
    cdf = F.softmax(logits, dim=-1).cumsum(dim=-1)
    # Scaled by the last entry, which rounding can leave a hair away from 1.
    # right=True skips tokens of probability zero, whose stretch is empty.
    targets = (draws.to(cdf.dtype) * cdf[:, -1]).unsqueeze(1)
    picks = torch.searchsorted(cdf, targets, right=True).squeeze(1)
    return picks.clamp(max=logits.size(-1) - 1)


def sample_decode(
    model: nn.Module,
    src: torch.Tensor,
    temperature: float = 1.0,
    top_k: int | None = None,
    seed: int | Sequence[int] = 0,
    max_len: int | None = None,
    device: torch.device | None = None,
    config: Config | None = None,
) -> list[list[int]]:
    """Sample a translation from the model's distribution, one token at a time.

    Greedy decoding takes the most likely next token every step. Sampling instead
    *draws* the next token at random, in proportion to its probability, so a token
    the model gives 30% is chosen about 30% of the time. The output is one
    plausible translation rather than the single most likely path, and repeated
    draws differ.

    That variety is the point in back-translation. Edunov et al. (2018),
    "Understanding Back-Translation at Scale", found that synthetic sources made by
    sampling train a better forward model than greedy or beam ones in all but
    resource-poor settings: beam output is too regular to teach much.

    Batched exactly as :func:`greedy_decode` is, chunk by ``config.batch_size``,
    and about as fast.

    Args:
        model: Seq2seq model (Transformer or LSTM).
        src: Source tensor (batch, src_len).
        temperature: Divides the logits before the softmax. 1.0 samples from the
            model's own distribution; below 1 sharpens it toward greedy (near 0 it
            *is* greedy); above 1 flattens it toward uniform. Must be positive.
        top_k: If set, sample only among the ``top_k`` most likely tokens at each
            step (Edunov et al. also report top-10). None samples from the full
            vocabulary.
        seed: Makes the output reproducible. Sentence ``i`` of ``src`` gets its own
            random stream, derived from ``seed`` and ``i``, so a sentence's sample
            does not depend on the batch size or on which other sentences are
            decoded with it. A list gives each sentence its seed explicitly.
        max_len: Maximum decoded length. Defaults to ``config.max_decode_length``.
        device: Torch device. Defaults to the model's.
        config: TorchLingo Config providing special token indices.

    Returns:
        One decoded token ID sequence per sentence, starting with SOS, as
        :func:`greedy_decode` returns them.

    Raises:
        ValueError: If ``temperature`` is not positive, ``top_k`` is less than 1,
            or a list ``seed`` does not have one entry per sentence.

    Example:
        >>> import torch
        >>> from torchlingo.models import SimpleTransformer
        >>> torch.manual_seed(0)  # doctest: +ELLIPSIS
        <torch._C.Generator object at ...>
        >>> model = SimpleTransformer(src_vocab_size=30, tgt_vocab_size=30,
        ...                           d_model=16, n_heads=2,
        ...                           num_encoder_layers=1, num_decoder_layers=1)
        >>> src = torch.tensor([[2, 5, 9, 3]])
        >>> first = sample_decode(model, src, seed=7, max_len=6)
        >>> first == sample_decode(model, src, seed=7, max_len=6)
        True
    """
    cfg = config if config is not None else get_default_config()
    max_len = max_len if max_len is not None else cfg.max_decode_length
    device = device if device is not None else next(model.parameters()).device
    if temperature <= 0:
        raise ValueError(
            f"temperature must be positive, got {temperature}. "
            "For the most likely token every step, use greedy decoding."
        )
    if top_k is not None and top_k < 1:
        raise ValueError(f"top_k must be at least 1, or None, got {top_k}.")

    model.eval()
    _check_decodable(model, "sampling")

    # Every random number a sentence will use, drawn up front from its own
    # generator: one per step. Drawn on the CPU, which gives the same numbers on
    # every machine, and moved to the device once per chunk.
    seeds = _sentence_seeds(seed, src.size(0))
    draws = torch.stack(
        [torch.rand(max_len, generator=torch.Generator().manual_seed(s)) for s in seeds]
    )

    batch_limit = max(1, cfg.batch_size)
    decoded: list[list[int]] = []
    for start in range(0, src.size(0), batch_limit):
        chunk_draws = draws[start : start + batch_limit].to(device)

        def draw(logits: torch.Tensor, step: int, d=chunk_draws) -> torch.Tensor:
            return _sample_tokens(logits, d[:, step], temperature, top_k)

        chunk = src[start : start + batch_limit]
        decoded.extend(_decode_steps(model, chunk, max_len, device, cfg, draw))
    return decoded


def add_noise(
    text: str,
    seed: int = 0,
    delete_prob: float = 0.1,
    replace_prob: float = 0.1,
    max_distance: int = 3,
    filler: str = NOISE_FILLER,
) -> str:
    """Corrupt a sentence the way Edunov et al. (2018) noise beam back-translations.

    Three operations on the sentence's words, in order: delete each word with
    probability ``delete_prob``; replace each remaining word with ``filler`` with
    probability ``replace_prob``; then shuffle the words, but never move one more
    than ``max_distance`` places from where it started. The defaults are the paper's.

    Why corrupt a translation on purpose: beam search returns the model's most
    regular output, and a forward model trained on it learns little it did not
    know. Noise makes the synthetic source harder to translate, so the forward
    model has to lean on what the source says rather than how it is phrased.

    Words are whatever whitespace separates, so a language written without spaces
    between words (Chinese, Japanese, Thai) gets its noise per run of text between
    spaces, which is not what the paper meant by a word.

    Args:
        text: The sentence to corrupt.
        seed: Makes the noise reproducible: one seed, one result.
        delete_prob: Probability of deleting each word.
        replace_prob: Probability of replacing each word left with ``filler``.
        max_distance: The furthest any word may move in the shuffle. 0 keeps
            the order.
        filler: The word substituted for replaced words.

    Returns:
        The corrupted sentence, its words joined by single spaces. At least one
        word survives deletion, so a non-empty sentence stays non-empty.

    Example:
        >>> add_noise("a b c d e f", seed=1, delete_prob=0.0, replace_prob=0.0,
        ...           max_distance=0)
        'a b c d e f'
        >>> add_noise("the cat sat on the mat", seed=3)
        '<blank> sat cat the on'
    """
    words = text.split()
    if not words:
        return text
    rng = random.Random(seed)
    kept = [word for word in words if rng.random() >= delete_prob]
    if not kept:
        kept = [rng.choice(words)]
    kept = [filler if rng.random() < replace_prob else word for word in kept]
    # Lample et al. (2018): add a random offset in [0, max_distance + 1) to each
    # position and sort. A word more than max_distance places further on starts at
    # least max_distance + 1 ahead and can only gain, so it can never be overtaken.
    keys = [i + rng.random() * (max_distance + 1) for i in range(len(kept))]
    order = sorted(range(len(kept)), key=keys.__getitem__)
    return " ".join(kept[i] for i in order)


def _noised(translations: list[str], seeds: Sequence[int]) -> list[str]:
    """Apply :func:`add_noise` to each translation with that sentence's seed."""
    return [add_noise(text, seed=s) for text, s in zip(translations, seeds)]


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
            # Boolean, to match the key padding masks below. See the note in
            # greedy_decode.
            tgt_mask = create_causal_mask(len(tokens), device)
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
    temperature: float = 1.0,
    top_k: int | None = None,
    seed: int | Sequence[int] = 0,
) -> list[str]:
    """Translate a batch of raw sentences using provided vocabularies.

    Processes inputs in chunks of ``config.batch_size`` to avoid moving very
    large padded batches to the device at once.

    Args:
        model: Seq2seq model (Transformer or LSTM).
        sentences: Iterable of raw source sentences.
        src_vocab: Vocabulary implementing BaseVocab.encode(add_special_tokens=True).
        tgt_vocab: Vocabulary implementing BaseVocab.decode(skip_special_tokens=True).
        decode_strategy: One of :data:`DECODE_STRATEGIES`. Defaults to
            "greedy" -- beam search is opt-in, and usually produces better
            translations. "sample" draws each token at random (see
            :func:`sample_decode`); "beam+noise" runs beam search, then corrupts
            each translation with :func:`add_noise`. Those two are for making
            back-translations, not for scoring a system.
        beam_size: Beam width for "beam" and "beam+noise".
        max_len: Maximum generation length.
        device: Torch device. Defaults to model device.
        config: TorchLingo Config.
        temperature: For "sample": below 1 sharpens the distribution toward
            greedy, above 1 flattens it.
        top_k: For "sample": draw only among the ``top_k`` most likely tokens.
        seed: For "sample" and "beam+noise": fixes the random choices. Sentence
            ``i`` gets its own stream derived from ``seed`` and ``i``, so its
            output does not depend on ``config.batch_size``. A list gives one seed
            per sentence. Ignored by "greedy" and "beam", which are not random.

    Returns:
        List of decoded text strings aligned with input sentences.

    Raises:
        ValueError: If ``decode_strategy`` is not one of :data:`DECODE_STRATEGIES`.

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

    _check_strategy(decode_strategy)
    if decode_strategy in _BEAM_STRATEGIES:
        _warn_if_large_beam_input(len(sentences))
    seeds = _sentence_seeds(seed, len(sentences))

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

        if decode_strategy in _BEAM_STRATEGIES:
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
        elif decode_strategy == "sample":
            outputs.extend(
                sample_decode(
                    model,
                    padded,
                    temperature=temperature,
                    top_k=top_k,
                    seed=seeds[start : start + batch_limit],
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
    if decode_strategy == "beam+noise":
        return _noised(decoded, seeds)
    return decoded


# A bulk decode is resumable because its output file is its own progress record: one
# translation per line, appended a chunk at a time. Rerunning the same call counts the complete
# lines already there and decodes only the rest. Task #102, 2026-10-12: Assignment 13
# back-translates 100,000 sentences or more, and a decode that dies at hour two used to start
# again from zero, which is the failure training's checkpointer already prevents one level up.
#
# The sidecar records what the lines are translations OF. Without it, rerunning with a
# different input file, a different model's settings or a reordered corpus would append to a
# file whose first lines belong to another job, and the result would be misaligned in a way
# no score would reveal.
_PROGRESS_SUFFIX = ".progress.json"


def _fingerprint(sentences: Sequence[str], settings: dict) -> dict:
    """Describe a decoding job precisely enough to tell whether a file belongs to it."""
    digest = hashlib.sha256()
    for sentence in sentences:
        digest.update(sentence.encode("utf-8"))
        digest.update(b"\n")
    return {"inputs_sha256": digest.hexdigest(), "inputs": len(sentences), **settings}


def _complete_lines(path: Path) -> int:
    """Count the finished lines in ``path``, cutting off a partly written last line.

    A crash in the middle of a write can leave a final line with no newline. That line is a
    fragment, not a translation, so it is removed and decoded again.
    """
    data = path.read_bytes()
    keep = data.rfind(b"\n") + 1
    if keep < len(data):
        os.truncate(path, keep)
    return data.count(b"\n", 0, keep)


def translate_to_file(
    model: nn.Module,
    sentences: Sequence[str],
    src_vocab: BaseVocab,
    tgt_vocab: BaseVocab,
    output_path: str | Path,
    decode_strategy: str = "greedy",
    beam_size: int = 5,
    max_len: int | None = None,
    device: torch.device | None = None,
    config: Config | None = None,
    chunk_size: int = 512,
    overwrite: bool = False,
    verbose: bool = True,
    temperature: float = 1.0,
    top_k: int | None = None,
    seed: int = 0,
) -> list[str]:
    """Translate many sentences into a file, resuming where an interrupted run stopped.

    Writes one translation per line to ``output_path``, aligned with ``sentences``, appending
    each chunk as soon as it is decoded. If the run dies, call this again with the same
    arguments: it counts the lines already written and decodes only the sentences after them.
    A file from a *different* job (other sentences, or other decoding settings) is refused
    rather than appended to.

    Decoding goes through :func:`torchlingo.inference_fast.translate_batch`, whose output is
    identical to :func:`translate_batch`'s and much faster for beam search.

    Args:
        model: Seq2seq model (Transformer or LSTM).
        sentences: Every source sentence of the job, in order. Pass the same list on a rerun.
        src_vocab: Source vocabulary.
        tgt_vocab: Target vocabulary.
        output_path: The file to write, one translation per line. Put it on Google Drive in
            Colab, so a disconnect does not take it with the runtime.
        decode_strategy: One of :data:`DECODE_STRATEGIES`; see :func:`translate_batch`.
        beam_size: Beam width for "beam" and "beam+noise".
        max_len: Maximum generation length. Defaults to ``config.max_decode_length``.
        device: Torch device. Defaults to the model's.
        config: TorchLingo Config.
        chunk_size: Sentences decoded between writes. Work lost to an interruption is at most
            one chunk.
        overwrite: Start again from the first sentence, discarding an existing output file.
        verbose: Print progress after each chunk.
        temperature: For "sample"; see :func:`sample_decode`.
        top_k: For "sample"; see :func:`sample_decode`.
        seed: For "sample" and "beam+noise". Recorded with the job, so a rerun with another
            seed is refused rather than appended. Each sentence's random stream depends on
            the seed and its position in ``sentences`` only, so the file is the same
            whatever ``chunk_size`` is and however often the run was interrupted.

    Returns:
        All ``len(sentences)`` translations, read back from the file.

    Raises:
        ValueError: If ``output_path`` holds lines from a different job, or more lines than
            there are sentences, and ``overwrite`` is False; or if ``decode_strategy`` is
            not one of :data:`DECODE_STRATEGIES`.

    Example:
        >>> import tempfile, torch
        >>> from torchlingo.data_processing.vocab import SimpleVocab
        >>> from torchlingo.models import SimpleTransformer
        >>> vocab = SimpleVocab(min_freq=1)
        >>> vocab.build_vocab(["hola mundo", "hello world"])
        >>> model = SimpleTransformer(len(vocab), len(vocab), d_model=16, n_heads=2,
        ...                           num_encoder_layers=1, num_decoder_layers=1, d_ff=32)
        >>> out = tempfile.mkdtemp() + "/test.hyp"
        >>> lines = translate_to_file(model, ["hello world"] * 3, vocab, vocab, out,
        ...                           max_len=4, chunk_size=2, verbose=False)
        >>> len(lines)
        3
    """
    from .inference_fast import translate_batch as fast_translate_batch

    cfg = config if config is not None else get_default_config()
    _check_strategy(decode_strategy)
    max_len = max_len if max_len is not None else cfg.max_decode_length
    output_path = Path(output_path)
    progress_path = output_path.with_name(output_path.name + _PROGRESS_SUFFIX)
    settings = {
        "decode_strategy": decode_strategy,
        "beam_size": beam_size if decode_strategy in _BEAM_STRATEGIES else None,
        "max_len": max_len,
    }
    # Added only for the strategies that use them: a greedy or beam job keeps the record
    # it had before sampling existed, so a decode interrupted then still resumes now.
    if decode_strategy == "sample":
        settings.update(temperature=temperature, top_k=top_k, seed=seed)
    elif decode_strategy == "beam+noise":
        settings["seed"] = seed
    job = _fingerprint(sentences, settings)
    seeds = _sentence_seeds(seed, len(sentences))

    done = 0
    if output_path.exists() and not overwrite:
        recorded = None
        if progress_path.exists():
            recorded = json.loads(progress_path.read_text(encoding="utf-8"))
        if recorded != job:
            raise ValueError(
                f"{output_path} already exists, and its "
                + (
                    "translations are of a different job: other sentences, or other "
                    "decoding settings."
                    if recorded is not None
                    else f"record of what it translated ({progress_path.name}) is missing."
                )
                + " Appending would misalign it with your sentences. Pass overwrite=True "
                "to start again, or choose a different output_path."
            )
        done = _complete_lines(output_path)
        if done > len(sentences):
            raise ValueError(
                f"{output_path} has {done} lines but there are only {len(sentences)} "
                "sentences. Pass overwrite=True to start again."
            )
        if verbose:
            print(f"Resuming: {done:,} of {len(sentences):,} already translated.")
    else:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text("", encoding="utf-8")

    progress_path.write_text(json.dumps(job, indent=2) + "\n", encoding="utf-8")

    for start in range(done, len(sentences), max(1, chunk_size)):
        chunk = sentences[start : start + max(1, chunk_size)]
        translations = fast_translate_batch(
            model,
            chunk,
            src_vocab,
            tgt_vocab,
            decode_strategy=decode_strategy,
            beam_size=beam_size,
            max_len=max_len,
            device=device,
            config=cfg,
            temperature=temperature,
            top_k=top_k,
            seed=seeds[start : start + len(chunk)],
        )
        # One line per sentence, always: a newline inside a translation would shift every
        # later line out of alignment with its source.
        lines = "".join(
            t.replace("\r", " ").replace("\n", " ") + "\n" for t in translations
        )
        with output_path.open("a", encoding="utf-8") as handle:
            handle.write(lines)
            handle.flush()
            os.fsync(handle.fileno())
        if verbose:
            print(
                f"Translated {start + len(chunk):,} of {len(sentences):,}.", flush=True
            )

    # Split on "\n" only, the way the lines were counted: splitlines() also breaks on rarer
    # separators such as U+2028, which can occur inside a translation.
    return output_path.read_text(encoding="utf-8").split("\n")[:-1]


__all__ = [
    "DECODE_STRATEGIES",
    "NOISE_FILLER",
    "add_noise",
    "beam_search_decode",
    "greedy_decode",
    "sample_decode",
    "translate_batch",
    "translate_to_file",
]
