"""Self-describing translation checkpoints.

A trained translation model is only usable with the *exact* tokenizer it was
trained on. Saving only the model weights makes it easy to later load a
different tokenizer with the same vocabulary size — the checkpoint loads
without error but every token id means something different, producing fluent
nonsense. This module bundles the SentencePiece tokenizer(s) and the model
architecture config together with the weights so inference can always
reconstruct the matching tokenizer.

Use :func:`save_checkpoint` when training and :func:`load_checkpoint` when
running inference. :func:`load_checkpoint` also accepts a legacy bare
``state_dict`` file so older checkpoints keep working (you must then supply the
tokenizer yourself).

Examples:
    >>> from torchlingo.checkpoint import save_checkpoint, load_checkpoint
    >>> save_checkpoint(
    ...     "model.pt",
    ...     model,
    ...     model_config={"src_vocab_size": 16000, "tgt_vocab_size": 16000,
    ...                   "d_model": 512, "n_heads": 8},
    ...     src_sp_model="data/sp_model.model",
    ...     tgt_sp_model="data/sp_model.model",
    ... )
    >>> ckpt = load_checkpoint("model.pt")
    >>> ckpt["model_config"]["d_model"]
    512
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, Union

import torch
from torch import nn

CHECKPOINT_FORMAT = "torchlingo-checkpoint-v1"

__all__ = ["CHECKPOINT_FORMAT", "save_checkpoint", "load_checkpoint"]


def save_checkpoint(
    path: Union[str, Path],
    model: nn.Module,
    *,
    model_config: Dict[str, Any],
    src_sp_model: Optional[Union[str, Path]] = None,
    tgt_sp_model: Optional[Union[str, Path]] = None,
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
            ``d_ff``, ``max_seq_length``, ``dropout``). Stored verbatim.
        src_sp_model: Path to the source SentencePiece ``.model`` file. If None,
            no source tokenizer is bundled.
        tgt_sp_model: Path to the target SentencePiece ``.model`` file. If None,
            no target tokenizer is bundled. May be the same file as
            ``src_sp_model`` for a shared tokenizer.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    bundle: Dict[str, Any] = {
        "format": CHECKPOINT_FORMAT,
        "model_state_dict": model.state_dict(),
        "model_config": dict(model_config),
    }
    if src_sp_model is not None:
        bundle["src_sp_model"] = Path(src_sp_model).read_bytes()
    if tgt_sp_model is not None:
        bundle["tgt_sp_model"] = Path(tgt_sp_model).read_bytes()

    torch.save(bundle, path)


def load_checkpoint(
    path: Union[str, Path],
    map_location: Union[str, torch.device] = "cpu",
) -> Dict[str, Any]:
    """Load a checkpoint saved by :func:`save_checkpoint` (or a legacy file).

    Args:
        path: Path to the checkpoint file.
        map_location: Device mapping passed through to ``torch.load``.

    Returns:
        A dict with at least ``model_state_dict``. For bundles it also includes
        ``model_config`` and (when present) ``src_sp_model`` / ``tgt_sp_model``
        as raw bytes. ``format`` is ``CHECKPOINT_FORMAT`` for bundles or
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
