"""
Inference script for trained Cebuano → Mandarin NMT model.

Load the best checkpoint and translate Cebuano sentences to Mandarin Chinese.
"""

import argparse
from pathlib import Path
from typing import List

import torch
import pandas as pd

from torchlingo.config import Config
from torchlingo.data_processing.vocab import SentencePieceVocab
from torchlingo.models import SimpleTransformer
from torchlingo.inference import translate_batch


def load_model_and_vocabs(
    checkpoint_path: Path,
    src_model_path: Path,
    tgt_model_path: Path,
    cfg: Config,
    device: torch.device,
) -> tuple[SimpleTransformer, SentencePieceVocab, SentencePieceVocab]:
    """Load trained model and vocabularies.

    Args:
        checkpoint_path: Path to model checkpoint (.pt file).
        src_model_path: Path to source SentencePiece model.
        tgt_model_path: Path to target SentencePiece model.
        cfg: Configuration object.
        device: Device to load model on.

    Returns:
        Tuple of (model, src_vocab, tgt_vocab).
    """
    print("Loading vocabularies...")
    src_vocab = SentencePieceVocab(str(src_model_path), config=cfg)
    tgt_vocab = SentencePieceVocab(str(tgt_model_path), config=cfg)
    print(f"  Source vocab size: {len(src_vocab):,}")
    print(f"  Target vocab size: {len(tgt_vocab):,}")

    print("\nInitializing model...")
    model = SimpleTransformer(
        src_vocab_size=len(src_vocab),
        tgt_vocab_size=len(tgt_vocab),
        config=cfg,
    )

    print(f"Loading checkpoint from {checkpoint_path}...")
    state_dict = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()

    print(f"✓ Model loaded successfully on {device}")

    return model, src_vocab, tgt_vocab


def translate_sentences(
    model: SimpleTransformer,
    sentences: List[str],
    src_vocab: SentencePieceVocab,
    tgt_vocab: SentencePieceVocab,
    cfg: Config,
    device: torch.device,
    decode_strategy: str = 'greedy',
    beam_size: int = 5,
) -> List[str]:
    """Translate a list of sentences.

    Args:
        model: Trained model.
        sentences: List of source sentences.
        src_vocab: Source vocabulary.
        tgt_vocab: Target vocabulary.
        cfg: Configuration object.
        device: Device to run on.
        decode_strategy: 'greedy' or 'beam'.
        beam_size: Beam size for beam search.

    Returns:
        List of translated sentences.
    """
    with torch.no_grad():
        translations = translate_batch(
            model,
            sentences,
            src_vocab=src_vocab,
            tgt_vocab=tgt_vocab,
            decode_strategy=decode_strategy,
            beam_size=beam_size,
            max_len=cfg.max_seq_length,
            device=device,
            config=cfg,
        )

    return translations


def interactive_mode(
    model: SimpleTransformer,
    src_vocab: SentencePieceVocab,
    tgt_vocab: SentencePieceVocab,
    cfg: Config,
    device: torch.device,
    decode_strategy: str,
    beam_size: int,
):
    """Interactive translation mode.

    Args:
        model: Trained model.
        src_vocab: Source vocabulary.
        tgt_vocab: Target vocabulary.
        cfg: Configuration object.
        device: Device to run on.
        decode_strategy: Decoding strategy.
        beam_size: Beam size for beam search.
    """
    print("\n" + "=" * 70)
    print("INTERACTIVE TRANSLATION MODE")
    print("=" * 70)
    print("\nEnter Cebuano sentences to translate (or 'quit' to exit)")
    print("Commands:")
    print("  quit - Exit interactive mode")
    print("  batch - Translate multiple sentences at once")
    print("-" * 70)

    while True:
        try:
            user_input = input("\nCebuano > ").strip()

            if not user_input:
                continue

            if user_input.lower() == 'quit':
                print("Exiting...")
                break

            if user_input.lower() == 'batch':
                print("\nEnter sentences (one per line). Empty line to finish:")
                batch_sentences = []
                while True:
                    line = input("  > ").strip()
                    if not line:
                        break
                    batch_sentences.append(line)

                if batch_sentences:
                    print(f"\nTranslating {len(batch_sentences)} sentences...")
                    translations = translate_sentences(
                        model, batch_sentences, src_vocab, tgt_vocab,
                        cfg, device, decode_strategy, beam_size
                    )

                    print("\nTranslations:")
                    for src, tgt in zip(batch_sentences, translations):
                        print(f"  {src}")
                        print(f"  → {tgt}\n")
                continue

            # Single sentence translation
            translations = translate_sentences(
                model, [user_input], src_vocab, tgt_vocab,
                cfg, device, decode_strategy, beam_size
            )

            print(f"Mandarin > {translations[0]}")

        except KeyboardInterrupt:
            print("\n\nExiting...")
            break
        except Exception as e:
            print(f"Error: {e}")


def translate_from_file(
    model: SimpleTransformer,
    input_file: Path,
    output_file: Path,
    src_vocab: SentencePieceVocab,
    tgt_vocab: SentencePieceVocab,
    cfg: Config,
    device: torch.device,
    decode_strategy: str,
    beam_size: int,
    batch_size: int = 32,
):
    """Translate sentences from a file.

    Args:
        model: Trained model.
        input_file: Path to input file (one sentence per line).
        output_file: Path to output file.
        src_vocab: Source vocabulary.
        tgt_vocab: Target vocabulary.
        cfg: Configuration object.
        device: Device to run on.
        decode_strategy: Decoding strategy.
        beam_size: Beam size for beam search.
        batch_size: Batch size for translation.
    """
    print(f"\nReading sentences from {input_file}...")

    # Read input file
    if input_file.suffix == '.tsv':
        df = pd.read_csv(input_file, sep='\t')
        sentences = df['src_text'].tolist() if 'src_text' in df.columns else df.iloc[:, 0].tolist()
    elif input_file.suffix == '.csv':
        df = pd.read_csv(input_file)
        sentences = df.iloc[:, 0].tolist()
    else:  # Plain text file
        with open(input_file, 'r', encoding='utf-8') as f:
            sentences = [line.strip() for line in f if line.strip()]

    print(f"Found {len(sentences):,} sentences")

    # Translate in batches
    print(f"\nTranslating (batch_size={batch_size})...")
    all_translations = []

    for i in range(0, len(sentences), batch_size):
        batch = sentences[i:i + batch_size]
        translations = translate_sentences(
            model, batch, src_vocab, tgt_vocab,
            cfg, device, decode_strategy, beam_size
        )
        all_translations.extend(translations)

        if (i + batch_size) % (batch_size * 10) == 0:
            print(f"  Translated {min(i + batch_size, len(sentences)):,}/{len(sentences):,}")

    print(f"✓ Translated {len(all_translations):,} sentences")

    # Save output
    print(f"\nSaving translations to {output_file}...")
    if output_file.suffix in ['.tsv', '.csv']:
        sep = '\t' if output_file.suffix == '.tsv' else ','
        output_df = pd.DataFrame({
            'source': sentences,
            'translation': all_translations,
        })
        output_df.to_csv(output_file, sep=sep, index=False)
    else:  # Plain text
        with open(output_file, 'w', encoding='utf-8') as f:
            for translation in all_translations:
                f.write(translation + '\n')

    print(f"✓ Saved to {output_file}")


def main():
    """Main inference workflow."""
    parser = argparse.ArgumentParser(
        description='Translate Cebuano to Mandarin using trained model'
    )

    # Model and vocab paths
    parser.add_argument('--checkpoint', type=str,
                        default='checkpoints/ceb_cmn/model_best.pt',
                        help='Path to model checkpoint (default: checkpoints/ceb_cmn/model_best.pt)')
    parser.add_argument('--src-model', type=str,
                        default='data/sp_ceb.model',
                        help='Path to source SentencePiece model (default: data/sp_ceb.model)')
    parser.add_argument('--tgt-model', type=str,
                        default='data/sp_cmn.model',
                        help='Path to target SentencePiece model (default: data/sp_cmn.model)')

    # Translation mode
    parser.add_argument('--mode', type=str, default='interactive',
                        choices=['interactive', 'file', 'examples'],
                        help='Translation mode (default: interactive)')

    # File translation
    parser.add_argument('--input-file', type=str,
                        help='Input file for file mode (one sentence per line, or TSV/CSV)')
    parser.add_argument('--output-file', type=str,
                        help='Output file for file mode')

    # Decoding
    parser.add_argument('--decode-strategy', type=str, default='greedy',
                        choices=['greedy', 'beam'],
                        help='Decoding strategy (default: greedy)')
    parser.add_argument('--beam-size', type=int, default=5,
                        help='Beam size for beam search (default: 5)')

    # Device
    parser.add_argument('--device', type=str, default='auto',
                        choices=['auto', 'cuda', 'cpu'],
                        help='Device to use (default: auto)')

    args = parser.parse_args()

    # Setup
    print("\n" + "=" * 70)
    print("CEBUANO → MANDARIN TRANSLATION")
    print("=" * 70)

    # Determine device
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)

    print(f"\nDevice: {device}")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Decode strategy: {args.decode_strategy}")
    if args.decode_strategy == 'beam':
        print(f"Beam size: {args.beam_size}")

    # Configuration (must match training config)
    cfg = Config(
        d_model=512,
        n_heads=8,
        num_encoder_layers=6,
        num_decoder_layers=6,
        d_ff=2048,
        dropout=0.1,
        max_seq_length=256,
        src_col='src_text',
        tgt_col='tgt_text',
    )

    # Load model and vocabularies
    model, src_vocab, tgt_vocab = load_model_and_vocabs(
        Path(args.checkpoint),
        Path(args.src_model),
        Path(args.tgt_model),
        cfg,
        device,
    )

    # Run appropriate mode
    if args.mode == 'interactive':
        interactive_mode(
            model, src_vocab, tgt_vocab, cfg, device,
            args.decode_strategy, args.beam_size
        )

    elif args.mode == 'file':
        if not args.input_file or not args.output_file:
            print("\nError: --input-file and --output-file required for file mode")
            return

        translate_from_file(
            model,
            Path(args.input_file),
            Path(args.output_file),
            src_vocab,
            tgt_vocab,
            cfg,
            device,
            args.decode_strategy,
            args.beam_size,
        )

    elif args.mode == 'examples':
        # Translate example sentences
        examples = [
            "Maayong buntag",
            "Kumusta ka",
            "Salamat",
            "Maayong hapon",
            "Maayong gabii",
        ]

        print("\n" + "=" * 70)
        print("EXAMPLE TRANSLATIONS")
        print("=" * 70)

        translations = translate_sentences(
            model, examples, src_vocab, tgt_vocab,
            cfg, device, args.decode_strategy, args.beam_size
        )

        for src, tgt in zip(examples, translations):
            print(f"\nCebuano:  {src}")
            print(f"Mandarin: {tgt}")

        print("\n" + "-" * 70)


if __name__ == '__main__':
    main()
