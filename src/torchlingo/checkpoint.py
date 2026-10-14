"""Self-describing translation checkpoints.

A trained translation model is only usable with the *exact* tokenizer it was
trained on. Saving only the model weights makes it easy to later load a
different tokenizer with the same vocabulary size — the checkpoint loads
without error but every token id means something different, producing fluent
nonsense. This module bundles the vocabularies and the model architecture
together with the weights so inference can always reconstruct both.

:func:`~torchlingo.training.train_model` writes such a file whenever it saves a
model, and :func:`load_model` rebuilds the model and its vocabularies from it in
one call, with nothing re-declared. :func:`save_checkpoint` writes one by hand.
:func:`load_checkpoint` returns the file's contents as a dict, and also accepts
a legacy bare ``state_dict`` file so older checkpoints keep working (you must
then rebuild the model and supply the tokenizer yourself).

Examples:
    A new session, a day after training with ``save_dir=OUT_DIR / "best"``,
    needs only the path. `tests/test_checkpoint.py` exercises this for real.

    >>> from torchlingo.checkpoint import load_model
    >>> model, src_vocab, tgt_vocab = load_model(        # doctest: +SKIP
    ...     OUT_DIR / "best" / "model_best.pt")
    >>> translate_batch(model, ["Hello."], src_vocab, tgt_vocab)  # doctest: +SKIP
    ['Hola.']
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, NamedTuple

import torch
from torch import nn

from .data_processing.vocab import BaseVocab, CharVocab, SentencePieceVocab, SimpleVocab
from .models import SimpleSeq2SeqLSTM, SimpleTransformer

CHECKPOINT_FORMAT = "torchlingo-checkpoint-v1"

# The models a checkpoint can name, by class name.
MODEL_CLASSES: dict[str, type[nn.Module]] = {
    "SimpleTransformer": SimpleTransformer,
    "SimpleSeq2SeqLSTM": SimpleSeq2SeqLSTM,
}

_SPECIALS = (
    "pad_token",
    "unk_token",
    "sos_token",
    "eos_token",
    "pad_idx",
    "unk_idx",
    "sos_idx",
    "eos_idx",
)

__all__ = [
    "CHECKPOINT_FORMAT",
    "LoadedModel",
    "describe_model",
    "load_checkpoint",
    "load_model",
    "save_checkpoint",
]


class LoadedModel(NamedTuple):
    """What :func:`load_model` returns: a model ready to translate, and its vocabularies.

    Attributes:
        model: The rebuilt model, its weights loaded, on the requested device, in
            ``eval()`` mode.
        src_vocab: The source vocabulary the model was trained with.
        tgt_vocab: The target vocabulary; the same object as ``src_vocab`` when
            training shared one between the two languages.
    """

    model: nn.Module
    src_vocab: BaseVocab
    tgt_vocab: BaseVocab


def _vocab_state(vocab: BaseVocab | None) -> dict[str, Any] | None:
    """A vocabulary as plain data, or None for a kind a checkpoint cannot carry."""
    if vocab is None:
        return None
    specials = {name: getattr(vocab, name) for name in _SPECIALS}
    if isinstance(vocab, SentencePieceVocab):
        return {
            "type": "SentencePieceVocab",
            "model_proto": vocab.sp.serialized_model_proto(),
            "specials": specials,
        }
    # Exact types only: a subclass may tokenize differently (MeCab, Jieba), and
    # rebuilding it as its parent would encode text wrongly without an error.
    if type(vocab) in (SimpleVocab, CharVocab):
        return {
            "type": type(vocab).__name__,
            "token2idx": dict(vocab.token2idx),
            "min_freq": vocab.min_freq,
            "specials": specials,
        }
    return None


def _vocab_from_state(state: dict[str, Any]) -> BaseVocab:
    """Rebuild a vocabulary from what :func:`_vocab_state` wrote."""
    specials = state["specials"]
    if state["type"] == "SentencePieceVocab":
        return SentencePieceVocab(
            model_proto=state["model_proto"],
            **{k: specials[k] for k in ("pad_idx", "unk_idx", "sos_idx", "eos_idx")},
        )
    cls = {"SimpleVocab": SimpleVocab, "CharVocab": CharVocab}[state["type"]]
    vocab = cls(min_freq=state["min_freq"], **specials)
    vocab.token2idx = dict(state["token2idx"])
    vocab.idx2token = {idx: tok for tok, idx in vocab.token2idx.items()}
    return vocab


def describe_model(
    model: nn.Module,
    src_vocab: BaseVocab | None = None,
    tgt_vocab: BaseVocab | None = None,
) -> dict[str, Any]:
    """What a checkpoint needs, besides the weights, to rebuild ``model`` alone.

    :func:`~torchlingo.training.train_model` merges this into every model file it
    writes. It is empty for a model TorchLingo did not build, whose settings it
    cannot know, so such a model's checkpoints are saved exactly as before.

    Args:
        model: A ``SimpleTransformer`` or ``SimpleSeq2SeqLSTM``.
        src_vocab: The source vocabulary. ``SimpleVocab``, ``CharVocab`` and
            ``SentencePieceVocab`` are recorded; any other kind is left out, and
            must then be passed to :func:`load_model`.
        tgt_vocab: The target vocabulary, recorded the same way. When it is the
            same object as ``src_vocab`` it is stored once.

    Returns:
        A dict with ``format``, ``model_class``, ``model_config``, ``src_vocab``
        and ``tgt_vocab``, or ``{}`` for a model that does not record its settings.

    Examples:
        >>> from torchlingo.models import SimpleTransformer
        >>> model = SimpleTransformer(10, 12, d_model=16, n_heads=2, num_encoder_layers=1,
        ...                           num_decoder_layers=1, d_ff=32)
        >>> described = describe_model(model)
        >>> described["model_class"], described["model_config"]["tgt_vocab_size"]
        ('SimpleTransformer', 12)
        >>> describe_model(torch.nn.Linear(2, 2))
        {}
    """
    name = type(model).__name__
    config = getattr(model, "model_config", None)
    if name not in MODEL_CLASSES or not isinstance(config, dict):
        return {}
    src_state = _vocab_state(src_vocab)
    if tgt_vocab is not None and tgt_vocab is src_vocab and src_state is not None:
        tgt_state: dict[str, Any] | None = {"same_as": "src"}
    else:
        tgt_state = _vocab_state(tgt_vocab)
    return {
        "format": CHECKPOINT_FORMAT,
        "model_class": name,
        "model_config": dict(config),
        "src_vocab": src_state,
        "tgt_vocab": tgt_state,
    }


def save_checkpoint(
    path: str | Path,
    model: nn.Module,
    *,
    model_config: dict[str, Any] | None = None,
    src_sp_model: str | Path | None = None,
    tgt_sp_model: str | Path | None = None,
    src_vocab: BaseVocab | None = None,
    tgt_vocab: BaseVocab | None = None,
) -> None:
    """Save model weights, architecture config, and tokenizer(s) in one file.

    The SentencePiece models are stored as raw bytes so the checkpoint is
    self-contained and portable across machines (no dependence on the original
    file paths).

    Args:
        path: Destination ``.pt`` file. Parent directories are created.
        model: The trained model whose ``state_dict`` will be saved.
        model_config: Keyword arguments needed to reconstruct the model
            (e.g. ``src_vocab_size``, ``tgt_vocab_size``, ``d_model``,
            ``n_heads``, ``num_encoder_layers``, ``num_decoder_layers``,
            ``d_ff``, ``max_seq_length``, ``dropout``). Stored verbatim. Defaults
            to the settings a TorchLingo model records, ``model.model_config``.
        src_sp_model: Path to the source SentencePiece ``.model`` file. If None,
            no source tokenizer is bundled.
        tgt_sp_model: Path to the target SentencePiece ``.model`` file. If None,
            no target tokenizer is bundled. May be the same file as
            ``src_sp_model`` for a shared tokenizer.
        src_vocab: The source vocabulary object, recorded so :func:`load_model`
            can rebuild it (see :func:`describe_model` for which kinds).
        tgt_vocab: The target vocabulary object, recorded the same way.

    Raises:
        ValueError: If no ``model_config`` is given and the model records none.

    Examples:
        >>> import tempfile
        >>> from pathlib import Path
        >>> from torchlingo.models import SimpleTransformer
        >>> model = SimpleTransformer(10, 12, d_model=16, n_heads=2, num_encoder_layers=1,
        ...                           num_decoder_layers=1, d_ff=32)
        >>> path = Path(tempfile.mkdtemp()) / "model.pt"
        >>> save_checkpoint(path, model)
        >>> load_checkpoint(path)["model_config"]["d_model"]
        16
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    bundle: dict[str, Any] = describe_model(model, src_vocab, tgt_vocab)
    if model_config is None:
        model_config = bundle.get("model_config")
    if model_config is None:
        raise ValueError(
            "model_config is required for a model that does not record its settings"
        )
    bundle["format"] = CHECKPOINT_FORMAT
    bundle["model_state_dict"] = model.state_dict()
    bundle["model_config"] = dict(model_config)
    if src_sp_model is not None:
        bundle["src_sp_model"] = Path(src_sp_model).read_bytes()
    if tgt_sp_model is not None:
        bundle["tgt_sp_model"] = Path(tgt_sp_model).read_bytes()

    torch.save(bundle, path)


def load_checkpoint(
    path: str | Path,
    map_location: str | torch.device = "cpu",
) -> dict[str, Any]:
    """Load a checkpoint saved by :func:`save_checkpoint` (or a legacy file).

    Args:
        path: Path to the checkpoint file.
        map_location: Device mapping passed through to ``torch.load``.

    Returns:
        A dict with at least ``model_state_dict``. For bundles it also includes
        ``model_config`` and (when present) ``src_sp_model`` / ``tgt_sp_model``
        as raw bytes; a file written by ``train_model`` also carries its training
        state (``epoch``, ``val_loss``, ``train_losses`` and so on) and the
        vocabularies. ``format`` is ``CHECKPOINT_FORMAT`` for bundles or
        ``"legacy"`` for a bare ``state_dict`` file, so callers can tell whether
        a bundled tokenizer is available.
    """
    obj = torch.load(path, map_location=map_location, weights_only=True)

    if isinstance(obj, dict) and obj.get("format") == CHECKPOINT_FORMAT:
        return obj

    # Legacy: a plain torch.save() from train_model(), either a bare
    # state_dict (an ordered mapping of tensors) or a training-state dict
    # with a 'model_state_dict' key alongside optimizer/scheduler state. The
    # tokenizer is not bundled either way; the caller must supply it.
    if isinstance(obj, dict) and "model_state_dict" in obj:
        return {
            "format": "legacy",
            "model_state_dict": obj["model_state_dict"],
            "model_config": None,
        }
    return {"format": "legacy", "model_state_dict": obj, "model_config": None}


def _saved_vocab(
    raw: dict[str, Any], side: str, given: BaseVocab | None
) -> BaseVocab | None:
    """The vocabulary for ``side`` ("src" or "tgt"): the one given, else the file's."""
    if given is not None:
        return given
    state = raw.get(f"{side}_vocab")
    if state is not None and "same_as" not in state:
        return _vocab_from_state(state)
    # A file from save_checkpoint carries SentencePiece models as bytes.
    proto = raw.get(f"{side}_sp_model")
    if proto is not None:
        return SentencePieceVocab(model_proto=proto)
    return None


def load_model(
    path: str | Path,
    device: torch.device | str | None = None,
    *,
    src_vocab: BaseVocab | None = None,
    tgt_vocab: BaseVocab | None = None,
) -> LoadedModel:
    """Rebuild a trained model and its vocabularies from one checkpoint file.

    Reads a file written by :func:`~torchlingo.training.train_model` (its
    ``save_dir/model_best.pt``) or by :func:`save_checkpoint`, builds the model
    with the settings it was trained with, loads the weights, and rebuilds the
    vocabularies, so a new session needs nothing but the path. It also reads the
    repository's ``data/pretrained/model.pt``, whose SentencePiece model is a
    separate file: pass that vocabulary in.

    Args:
        path: The checkpoint file.
        device: Where to put the model. Defaults to CUDA if available, else CPU.
        src_vocab: A source vocabulary to use instead of the file's. Required
            when the file carries none (a vocabulary kind it cannot record, or a
            file saved without one).
        tgt_vocab: The same for the target side. When only ``src_vocab`` is
            given and the file carries no target vocabulary, it serves both.

    Returns:
        A :class:`LoadedModel`: ``model`` (in ``eval()`` mode), ``src_vocab``
        and ``tgt_vocab``, which unpack in that order.

    Raises:
        ValueError: If the file does not record the model's settings, as files
            saved before this version do not; if a vocabulary is neither in the
            file nor given; or if a vocabulary's size does not match the model's,
            which means it is not the vocabulary the model was trained with.

    Examples:
        >>> import tempfile
        >>> from pathlib import Path
        >>> from torchlingo.data_processing.vocab import SimpleVocab
        >>> from torchlingo.models import SimpleTransformer
        >>> vocab = SimpleVocab(min_freq=1)
        >>> vocab.build_vocab(["hello world"])
        >>> model = SimpleTransformer(len(vocab), len(vocab), d_model=16, n_heads=2,
        ...                           num_encoder_layers=1, num_decoder_layers=1, d_ff=32)
        >>> path = Path(tempfile.mkdtemp()) / "model.pt"
        >>> save_checkpoint(path, model, src_vocab=vocab, tgt_vocab=vocab)
        >>> model, src_vocab, tgt_vocab = load_model(path, device="cpu")
        >>> src_vocab is tgt_vocab, src_vocab.decode(vocab.encode("hello world"))
        (True, 'hello world')
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    raw = torch.load(path, map_location="cpu", weights_only=True)
    config = raw.get("model_config") if isinstance(raw, dict) else None
    if not config or "model_state_dict" not in raw:
        raise ValueError(
            f"{path} holds weights but not the settings the model was built with: it "
            "was saved before TorchLingo recorded them. Build the model and the "
            "vocabularies as the notebook that trained it did, then load the weights "
            "with model.load_state_dict(load_checkpoint(path)['model_state_dict'])."
        )

    config = dict(config)
    # data/pretrained/model.pt records one shared vocabulary size, not two.
    if "vocab_size" in raw:
        config.setdefault("src_vocab_size", raw["vocab_size"])
        config.setdefault("tgt_vocab_size", raw["vocab_size"])
    model_class = MODEL_CLASSES[raw.get("model_class", "SimpleTransformer")]

    src = _saved_vocab(raw, "src", src_vocab)
    tgt_state = raw.get("tgt_vocab")
    if tgt_vocab is None and tgt_state is not None and "same_as" in tgt_state:
        tgt = src
    else:
        tgt = _saved_vocab(raw, "tgt", tgt_vocab)
        if tgt is None and src_vocab is not None and tgt_vocab is None:
            tgt = src_vocab
    for side, vocab in (("src", src), ("tgt", tgt)):
        if vocab is None:
            raise ValueError(
                f"{path} carries no {side} vocabulary; pass {side}_vocab= to load_model."
            )
        expected = config.get(f"{side}_vocab_size")
        if expected is not None and len(vocab) != expected:
            raise ValueError(
                f"the {side} vocabulary has {len(vocab)} entries but the model was "
                f"trained with {expected}: it is not the vocabulary this model was "
                "trained with, and would translate into nonsense."
            )

    model = model_class(**config)
    model.load_state_dict(raw["model_state_dict"])
    model.to(device)
    model.eval()
    return LoadedModel(model, src, tgt)
