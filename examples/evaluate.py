"""Standalone evaluation script for trained NMT models.

This script loads a trained model checkpoint and evaluates it on test data.
"""

import argparse
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from torchlingo.config import Config
from torchlingo.data_processing.vocab import SentencePieceVocab
from torchlingo.data_processing.dataset import NMTDataset
from torchlingo.data_processing.batching import collate_fn
from torchlingo.models import SimpleTransformer
from torchlingo.inference import translate_batch
from torchlingo.evaluation import evaluate_model


def load_checkpoint(checkpoint_path: Path, device: torch.device):
    """Load model checkpoint.

    Args:
        checkpoint_path: Path to checkpoint file.
        device: Device to load model on.

    Returns:
        Dictionary containing model state and metadata.
    """
    print(f"\nLoading checkpoint from {checkpoint_path}...")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    print(f"✓ Checkpoint loaded (trained for {checkpoint.get('epoch', 'unknown')} epochs)")
    return checkpoint


def create_model_from_checkpoint(
    checkpoint: dict,
    src_vocab_size: int,
    tgt_vocab_size: int,
    device: torch.device,
    cfg: Config,
) -> SimpleTransformer:
    """Recreate model from checkpoint.

    Args:
        checkpoint: Checkpoint dictionary.
        src_vocab_size: Source vocabulary size.
        tgt_vocab_size: Target vocabulary size.
        device: Device to load model on.
        cfg: Configuration object.

    Returns:
        Loaded SimpleTransformer model.
    """
    print("\nRecreating model architecture...")

    model = SimpleTransformer(
        src_vocab_size=src_vocab_size,
        tgt_vocab_size=tgt_vocab_size,
        d_model=cfg.d_model,
        n_heads=cfg.n_heads,
        num_encoder_layers=cfg.num_encoder_layers,
        num_decoder_layers=cfg.num_decoder_layers,
        d_ff=cfg.d_ff,
        max_seq_length=cfg.max_seq_length,
        dropout=cfg.dropout,
        config=cfg,
    )

    # Load state dict
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"✓ Model loaded with {total_params:,} parameters")

    return model


def run_evaluation(
    model: SimpleTransformer,
    test_dataset: NMTDataset,
    src_vocab: SentencePieceVocab,
    tgt_vocab: SentencePieceVocab,
    cfg: Config,
    device: torch.device,
    output_dir: Path,
    num_samples: int = 10,
    tokenization_mode: str = 'auto',
):
    """Run comprehensive evaluation.

    Args:
        model: Trained model.
        test_dataset: Test dataset.
        src_vocab: Source vocabulary.
        tgt_vocab: Target vocabulary.
        cfg: Configuration object.
        device: Device to run on.
        output_dir: Directory to save results.
        num_samples: Number of sample translations to display.
    """
    print("\n" + "=" * 70)
    print("EVALUATION")
    print("=" * 70)

    model.eval()

    # Sample translations
    print(f"\nSample Translations (showing {num_samples} examples):")
    print("-" * 70)

    import random
    indices = random.sample(range(len(test_dataset)), min(num_samples, len(test_dataset)))

    test_sentences = []
    reference_translations = []

    for idx in indices:
        src_text = test_dataset.src_sentences[idx]
        tgt_text = test_dataset.tgt_sentences[idx]
        test_sentences.append(src_text)
        reference_translations.append(tgt_text)

    # Translate sample
    with torch.no_grad():
        translations = translate_batch(
            model,
            test_sentences,
            src_vocab=src_vocab,
            tgt_vocab=tgt_vocab,
            decode_strategy='greedy',
            max_len=cfg.max_seq_length,
            device=device,
            config=cfg,
        )

    # Display sample results
    for i, (src, ref, hyp) in enumerate(zip(test_sentences, reference_translations, translations)):
        print(f"\nExample {i + 1}:")
        print(f"  Source:     {src[:100]}..." if len(src) > 100 else f"  Source:     {src}")
        print(f"  Reference:  {ref[:100]}..." if len(ref) > 100 else f"  Reference:  {ref}")
        print(f"  Hypothesis: {hyp[:100]}..." if len(hyp) > 100 else f"  Hypothesis: {hyp}")

    print("\n" + "-" * 70)

    # Full evaluation on test set using library's evaluate_model
    print(f"\nEvaluating on full test set ({len(test_dataset):,} examples)...")
    print("This may take a few minutes...")

    start_time = time.time()

    # Use the library's evaluate_model function
    scores = evaluate_model(
        model=model,
        src_sentences=test_dataset.src_sentences,
        tgt_sentences=test_dataset.tgt_sentences,
        src_vocab=src_vocab,
        tgt_vocab=tgt_vocab,
        device=device,
        decode_strategy='greedy',
        max_decode_length=cfg.max_seq_length,
        batch_size=cfg.batch_size,
        tokenization=tokenization_mode,
        compute_chrf_score=True,
        compute_ter_score=False,
        config=cfg,
    )

    elapsed = time.time() - start_time

    # Display results
    print("\n" + "=" * 70)
    print("EVALUATION RESULTS")
    print("=" * 70)
    print(f"\nMetrics:")
    for metric, score in scores.items():
        print(f"  {metric:20s}: {score:.4f}")

    print(f"\nEvaluation time: {elapsed:.1f}s")
    print(f"Examples per second: {len(test_dataset) / elapsed:.1f}")

    # Save results to file
    output_dir.mkdir(parents=True, exist_ok=True)
    results_file = output_dir / 'evaluation_results.txt'

    with open(results_file, 'w') as f:
        f.write("=" * 70 + "\n")
        f.write("EVALUATION RESULTS\n")
        f.write("=" * 70 + "\n\n")
        f.write(f"Test set size: {len(test_dataset):,} examples\n")
        f.write(f"Evaluation time: {elapsed:.1f}s\n")
        f.write(f"Examples per second: {len(test_dataset) / elapsed:.1f}\n\n")
        f.write("Metrics:\n")
        for metric, score in scores.items():
            f.write(f"  {metric:20s}: {score:.4f}\n")

    print(f"\n✓ Results saved to {results_file}")


def main():
    """Main evaluation pipeline."""
    parser = argparse.ArgumentParser(description='Evaluate trained NMT model')

    # Model and data paths
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint (.pt file)')
    parser.add_argument('--test-file', type=str, required=True,
                        help='Path to test data file')
    parser.add_argument('--src-sp-model', type=str, required=True,
                        help='Path to source SentencePiece model')
    parser.add_argument('--tgt-sp-model', type=str, required=True,
                        help='Path to target SentencePiece model')

    # Model architecture (must match training)
    parser.add_argument('--d-model', type=int, default=512,
                        help='Model dimension (default: 512)')
    parser.add_argument('--n-heads', type=int, default=8,
                        help='Number of attention heads (default: 8)')
    parser.add_argument('--num-encoder-layers', type=int, default=6,
                        help='Number of encoder layers (default: 6)')
    parser.add_argument('--num-decoder-layers', type=int, default=6,
                        help='Number of decoder layers (default: 6)')
    parser.add_argument('--d-ff', type=int, default=2048,
                        help='Feed-forward dimension (default: 2048)')
    parser.add_argument('--dropout', type=float, default=0.1,
                        help='Dropout rate (default: 0.1)')
    parser.add_argument('--max-seq-length', type=int, default=256,
                        help='Maximum sequence length (default: 256)')

    # Evaluation settings
    parser.add_argument('--batch-size', type=int, default=64,
                        help='Batch size for evaluation (default: 64)')
    parser.add_argument('--num-samples', type=int, default=10,
                        help='Number of sample translations to display (default: 10)')
    parser.add_argument('--output-dir', type=str, default='evaluation_results',
                        help='Directory to save results (default: evaluation_results)')

    # Data columns
    parser.add_argument('--src-col', type=str, default='src',
                        help='Source column name (default: src)')
    parser.add_argument('--tgt-col', type=str, default='tgt',
                        help='Target column name (default: tgt)')

    # Device
    parser.add_argument('--device', type=str, default='auto',
                        choices=['auto', 'cuda', 'cpu'],
                        help='Device to use (default: auto)')

    # Tokenization mode for BLEU
    parser.add_argument('--tokenization', type=str, default='auto',
                        choices=['auto', 'char', 'word'],
                        help='BLEU tokenization mode: auto (detect from text), '
                             'char (character-level for CJK), word (space-separated). '
                             'Default: auto')

    args = parser.parse_args()

    # Setup
    print("\n" + "=" * 70)
    print("NMT Model Evaluation")
    print("=" * 70)

    # Determine device
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)

    print(f"\nDevice: {device}")

    # Create config
    cfg = Config(
        d_model=args.d_model,
        n_heads=args.n_heads,
        num_encoder_layers=args.num_encoder_layers,
        num_decoder_layers=args.num_decoder_layers,
        d_ff=args.d_ff,
        dropout=args.dropout,
        max_seq_length=args.max_seq_length,
        batch_size=args.batch_size,
        src_col=args.src_col,
        tgt_col=args.tgt_col,
    )

    # Load checkpoint
    checkpoint_path = Path(args.checkpoint)
    checkpoint = load_checkpoint(checkpoint_path, device)

    # Load vocabularies
    print("\nLoading SentencePiece models...")
    src_vocab = SentencePieceVocab(args.src_sp_model, config=cfg)
    tgt_vocab = SentencePieceVocab(args.tgt_sp_model, config=cfg)
    print(f"  Source vocab size: {len(src_vocab):,}")
    print(f"  Target vocab size: {len(tgt_vocab):,}")

    # Create model
    model = create_model_from_checkpoint(
        checkpoint, len(src_vocab), len(tgt_vocab), device, cfg
    )

    # Load test dataset
    print("\nLoading test dataset...")
    test_file = Path(args.test_file)
    test_dataset = NMTDataset(
        test_file,
        src_vocab=src_vocab,
        tgt_vocab=tgt_vocab,
        src_col=args.src_col,
        tgt_col=args.tgt_col,
        config=cfg,
    )
    print(f"  Test examples: {len(test_dataset):,}")

    # Run evaluation
    output_dir = Path(args.output_dir)
    run_evaluation(
        model, test_dataset, src_vocab, tgt_vocab, cfg, device, output_dir,
        args.num_samples, args.tokenization
    )

    print("\n" + "=" * 70)
    print("EVALUATION COMPLETE!")
    print("=" * 70)


if __name__ == '__main__':
    main()
