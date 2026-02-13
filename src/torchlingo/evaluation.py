"""Evaluation utilities for assessing translation quality.

This module provides functions for computing standard machine translation metrics
including BLEU, chrF, and TER using the sacrebleu library. These metrics allow
objective assessment of translation quality beyond simple loss values.

Typical usage:
    >>> from torchlingo.evaluation import compute_bleu, evaluate_model
    >>> predictions = ["Hello world", "How are you"]
    >>> references = ["Hello world", "How are you doing"]
    >>> bleu = compute_bleu(predictions, references)
    >>> print(f"BLEU: {bleu.score:.2f}")
"""

from typing import List, Dict, Optional, Union
import torch
import sacrebleu
from pathlib import Path

from .inference import translate_batch
from .data_processing.vocab import BaseVocab


def compute_bleu(
    predictions: List[str],
    references: Union[List[str], List[List[str]]],
    lowercase: bool = False,
    tokenize: str = "13a",
) -> sacrebleu.metrics.BLEU:
    """Compute corpus-level BLEU score for translations.

    Uses sacrebleu's implementation of BLEU (Papineni et al., 2002), which is
    the standard metric for machine translation evaluation. BLEU measures n-gram
    overlap between predictions and references, with scores from 0-100 (higher is better).

    Args:
        predictions: List of predicted translations (one string per sample).
        references: List of reference translations. Can be either:
            - List[str]: Single reference per sample
            - List[List[str]]: Multiple references per sample
        lowercase: If True, lowercase both predictions and references before scoring.
        tokenize: Tokenization method. Options: "13a" (default), "intl", "zh", "ja-mecab", "none".

    Returns:
        BLEU object with .score attribute (0-100) and detailed n-gram statistics.

    Example:
        >>> preds = ["The cat sat on the mat", "Hello world"]
        >>> refs = ["A cat sat on a mat", "Hello world"]
        >>> result = compute_bleu(preds, refs)
        >>> print(f"BLEU: {result.score:.2f}")
        BLEU: 54.23
    """
    # Ensure references are in the format sacrebleu expects
    if references and isinstance(references[0], str):
        # Single reference: convert to List[List[str]]
        references = [[ref] for ref in references]

    return sacrebleu.corpus_bleu(
        predictions,
        references,
        lowercase=lowercase,
        tokenize=tokenize,
    )


def compute_chrf(
    predictions: List[str],
    references: Union[List[str], List[List[str]]],
    word_order: int = 2,
) -> sacrebleu.metrics.CHRF:
    """Compute corpus-level chrF score for translations.

    chrF (character n-gram F-score) is more robust to morphological differences
    and works better for morphologically rich languages than BLEU. Scores range
    from 0-100 (higher is better).

    Args:
        predictions: List of predicted translations.
        references: List of reference translations (single or multiple per sample).
        word_order: Include word n-grams up to this order (0 = character-only, 2 = default).

    Returns:
        CHRF object with .score attribute (0-100).

    Example:
        >>> preds = ["Le chat est assis", "Bonjour"]
        >>> refs = ["Le chat s'est assis", "Bonjour monde"]
        >>> result = compute_chrf(preds, refs)
        >>> print(f"chrF: {result.score:.2f}")
    """
    if references and isinstance(references[0], str):
        references = [[ref] for ref in references]

    return sacrebleu.corpus_chrf(
        predictions,
        references,
        word_order=word_order,
    )


def compute_ter(
    predictions: List[str],
    references: Union[List[str], List[List[str]]],
    normalized: bool = False,
) -> sacrebleu.metrics.TER:
    """Compute corpus-level TER (Translation Error Rate) score.

    TER measures the minimum number of edits (insertions, deletions, substitutions,
    shifts) needed to transform predictions into references. Lower scores are better
    (0 = perfect, 100 = completely wrong).

    Args:
        predictions: List of predicted translations.
        references: List of reference translations (single or multiple per sample).
        normalized: If True, apply normalization (lowercase, punctuation removal).

    Returns:
        TER object with .score attribute (0-100, lower is better).

    Example:
        >>> preds = ["The cat sat", "Hello"]
        >>> refs = ["A cat sits", "Hi there"]
        >>> result = compute_ter(preds, refs)
        >>> print(f"TER: {result.score:.2f}")
    """
    if references and isinstance(references[0], str):
        references = [[ref] for ref in references]

    return sacrebleu.corpus_ter(
        predictions,
        references,
        normalized=normalized,
    )


def evaluate_model(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    src_vocab: BaseVocab,
    tgt_vocab: BaseVocab,
    device: Optional[torch.device] = None,
    decode_strategy: str = "greedy",
    beam_size: int = 5,
    max_decode_length: int = 200,
    lowercase: bool = False,
    compute_chrf_score: bool = True,
    compute_ter_score: bool = False,
) -> Dict[str, float]:
    """Evaluate a trained model on a dataset using multiple metrics.

    Generates translations for all samples in the dataloader and computes
    BLEU, chrF, and optionally TER scores against reference translations.

    Args:
        model: Trained seq2seq model (Transformer or LSTM).
        dataloader: DataLoader yielding (src, tgt) batches.
        src_vocab: Source vocabulary for decoding token IDs.
        tgt_vocab: Target vocabulary for decoding token IDs.
        device: Torch device. Defaults to CUDA if available, else CPU.
        decode_strategy: "greedy" or "beam" search decoding.
        beam_size: Beam width for beam search (ignored if greedy).
        max_decode_length: Maximum length for generated translations.
        lowercase: If True, lowercase before computing BLEU.
        compute_chrf_score: Whether to compute chrF score (recommended).
        compute_ter_score: Whether to compute TER score (slower).

    Returns:
        Dictionary with metric scores:
            - "bleu": BLEU score (0-100)
            - "chrf": chrF score (0-100) if compute_chrf_score=True
            - "ter": TER score (0-100, lower is better) if compute_ter_score=True

    Example:
        >>> scores = evaluate_model(
        ...     model, val_loader, src_vocab, tgt_vocab,
        ...     decode_strategy="beam", beam_size=5
        ... )
        >>> print(f"BLEU: {scores['bleu']:.2f}, chrF: {scores['chrf']:.2f}")
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model.eval()
    model = model.to(device)

    predictions = []
    references = []

    with torch.no_grad():
        for src_batch, tgt_batch in dataloader:
            src_batch = src_batch.to(device)

            # Generate predictions
            pred_tokens = translate_batch(
                model=model,
                src_batch=src_batch,
                src_vocab=src_vocab,
                tgt_vocab=tgt_vocab,
                decode_strategy=decode_strategy,
                beam_size=beam_size,
                max_decode_length=max_decode_length,
                device=device,
            )

            # Decode predictions
            for tokens in pred_tokens:
                pred_text = tgt_vocab.decode(tokens)
                predictions.append(pred_text)

            # Decode references (remove SOS/EOS tokens)
            for tgt_seq in tgt_batch:
                # Filter out padding, SOS, EOS tokens
                ref_tokens = [
                    idx.item()
                    for idx in tgt_seq
                    if idx.item() not in [tgt_vocab.pad_idx, tgt_vocab.sos_idx, tgt_vocab.eos_idx]
                ]
                ref_text = tgt_vocab.decode(ref_tokens)
                references.append(ref_text)

    # Compute metrics
    results = {}

    bleu_result = compute_bleu(predictions, references, lowercase=lowercase)
    results["bleu"] = bleu_result.score

    if compute_chrf_score:
        chrf_result = compute_chrf(predictions, references)
        results["chrf"] = chrf_result.score

    if compute_ter_score:
        ter_result = compute_ter(predictions, references)
        results["ter"] = ter_result.score

    return results


def save_translations(
    predictions: List[str],
    references: Optional[List[str]] = None,
    output_path: Union[str, Path] = "translations.txt",
    include_metrics: bool = True,
) -> None:
    """Save predictions and optionally references to a file.

    Args:
        predictions: List of predicted translations.
        references: Optional list of reference translations.
        output_path: Path to save translations.
        include_metrics: If True and references provided, append BLEU score.

    Example:
        >>> save_translations(
        ...     predictions=["Hello world", "Goodbye"],
        ...     references=["Hello there", "Bye"],
        ...     output_path="outputs/translations.txt"
        ... )
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        for i, pred in enumerate(predictions):
            f.write(f"Prediction {i + 1}: {pred}\n")
            if references:
                f.write(f"Reference {i + 1}:  {references[i]}\n")
            f.write("\n")

        if include_metrics and references:
            f.write("=" * 50 + "\n")
            f.write("METRICS\n")
            f.write("=" * 50 + "\n")

            bleu = compute_bleu(predictions, references)
            f.write(f"BLEU: {bleu.score:.2f}\n")

            chrf = compute_chrf(predictions, references)
            f.write(f"chrF: {chrf.score:.2f}\n")

    print(f"Translations saved to {output_path}")
