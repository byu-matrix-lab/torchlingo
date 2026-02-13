"""
Comprehensive training script for Cebuano → Mandarin Chinese NMT.

This script performs end-to-end training on the ceb__cmn.csv dataset:
- Data preprocessing and splitting
- SentencePiece tokenization for both languages
- Full-size Transformer training with validation
- TensorBoard logging and checkpointing
- Performance monitoring and evaluation
"""

import time
from pathlib import Path
import argparse

import pandas as pd
import torch
from torch.utils.data import DataLoader

from torchlingo.config import Config
from torchlingo.preprocessing.base import save_data
from torchlingo.preprocessing.sentencepiece import train_sentencepiece
from torchlingo.data_processing.vocab import SentencePieceVocab
from torchlingo.data_processing.dataset import NMTDataset
from torchlingo.data_processing.batching import collate_fn
from torchlingo.models import SimpleTransformer
from torchlingo.training import train_model
from torchlingo.inference import translate_batch
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'


def setup_config(args) -> tuple[Config, int]:
    """Create configuration for training.

    Returns:
        Tuple of (Config, num_epochs)
    """
    cfg = Config(
        # Data paths
        data_dir=Path('data'),
        checkpoint_dir=Path('checkpoints/ceb_cmn'),

        # Model architecture
        d_model=512,
        n_heads=8,
        num_encoder_layers=6,
        num_decoder_layers=6,
        d_ff=2048,
        dropout=0.1,
        max_seq_length=256,

        # Training hyperparameters
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        label_smoothing=0.1,

        # Validation and checkpointing
        val_interval=args.val_interval,
        save_interval=args.save_interval,
        patience=args.patience,

        # SentencePiece
        vocab_size=args.vocab_size,
        sp_character_coverage=0.9995,  # High coverage for Chinese
        sp_model_type='unigram',

        # TensorBoard
        use_tensorboard=True,
        tensorboard_dir=Path('runs'),
        experiment_name=f'ceb_cmn_{args.experiment_name}',

        # Column names
        src_col='src_text',
        tgt_col='tgt_text',

        # Data format
        data_format='tsv',
    )

    # num_epochs is passed to train_model(), not Config
    return cfg, args.num_epochs


def load_and_preprocess_data(cfg: Config) -> tuple[Path, Path, Path]:
    """Load CSV and create train/val/test splits.

    Args:
        cfg: Configuration object.

    Returns:
        Tuple of (train_file, val_file, test_file) paths.
    """
    print("\n" + "=" * 70)
    print("STEP 1: Data Preprocessing")
    print("=" * 70)

    start_time = time.time()

    # Load raw CSV
    data_file = cfg.data_dir / 'ceb__cmn.csv'
    print(f"\nLoading data from {data_file}...")

    df = pd.read_csv(data_file)
    print(f"Loaded {len(df):,} parallel sentences")

    # Verify required columns
    if cfg.src_col not in df.columns or cfg.tgt_col not in df.columns:
        raise ValueError(
            f"CSV must contain '{cfg.src_col}' and '{cfg.tgt_col}' columns. "
            f"Found: {list(df.columns)}"
        )

    # Remove any rows with missing values
    initial_len = len(df)
    df = df.dropna(subset=[cfg.src_col, cfg.tgt_col])
    if len(df) < initial_len:
        print(f"Removed {initial_len - len(df):,} rows with missing values")

    # Filter out extremely long sentences (for memory efficiency)
    def token_count(text):
        return len(str(text).split())

    max_tokens = cfg.max_seq_length
    df['src_len'] = df[cfg.src_col].apply(token_count)
    df['tgt_len'] = df[cfg.tgt_col].apply(token_count)

    before_filter = len(df)
    df = df[(df['src_len'] < max_tokens) & (df['tgt_len'] < max_tokens)]
    df = df.drop(columns=['src_len', 'tgt_len'])

    if len(df) < before_filter:
        print(f"Filtered {before_filter - len(df):,} very long sentences (>{max_tokens} tokens)")

    print(f"Final dataset size: {len(df):,} sentences")

    # Create train/val/test splits (80/10/10)
    print("\nCreating train/val/test splits (80/10/10)...")
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)  # Shuffle

    train_size = int(0.8 * len(df))
    val_size = int(0.1 * len(df))

    train_df = df[:train_size]
    val_df = df[train_size:train_size + val_size]
    test_df = df[train_size + val_size:]

    print(f"  Train: {len(train_df):,} sentences")
    print(f"  Val:   {len(val_df):,} sentences")
    print(f"  Test:  {len(test_df):,} sentences")

    # Save splits
    train_file = cfg.data_dir / f'train.{cfg.data_format}'
    val_file = cfg.data_dir / f'val.{cfg.data_format}'
    test_file = cfg.data_dir / f'test.{cfg.data_format}'

    save_data(train_df, train_file, cfg.data_format)
    save_data(val_df, val_file, cfg.data_format)
    save_data(test_df, test_file, cfg.data_format)

    elapsed = time.time() - start_time
    print(f"\n✓ Preprocessing complete in {elapsed:.1f}s")

    return train_file, val_file, test_file


def train_tokenizers(train_file: Path, cfg: Config) -> tuple[Path, Path]:
    """Train SentencePiece models for source and target languages.

    Args:
        train_file: Path to training data.
        cfg: Configuration object.

    Returns:
        Tuple of (src_model_path, tgt_model_path).
    """
    print("\n" + "=" * 70)
    print("STEP 2: SentencePiece Training")
    print("=" * 70)

    start_time = time.time()

    # Train separate models for each language
    src_model_prefix = str(cfg.data_dir / 'sp_ceb')
    tgt_model_prefix = str(cfg.data_dir / 'sp_cmn')

    print(f"\nTraining Cebuano tokenizer (vocab_size={cfg.vocab_size})...")
    train_sentencepiece(
        [train_file],
        src_model_prefix,
        columns=[cfg.src_col],
        vocab_size=cfg.vocab_size,
        character_coverage=0.9995,  # High coverage
        model_type=cfg.sp_model_type,
        config=cfg,
    )

    print(f"\nTraining Mandarin tokenizer (vocab_size={cfg.vocab_size})...")
    # Chinese needs higher character coverage
    train_sentencepiece(
        [train_file],
        tgt_model_prefix,
        columns=[cfg.tgt_col],
        vocab_size=cfg.vocab_size,
        character_coverage=0.9999,  # Even higher for Chinese
        model_type=cfg.sp_model_type,
        config=cfg,
    )

    src_model_path = Path(f'{src_model_prefix}.model')
    tgt_model_path = Path(f'{tgt_model_prefix}.model')

    elapsed = time.time() - start_time
    print(f"\n✓ Tokenizer training complete in {elapsed:.1f}s")

    return src_model_path, tgt_model_path


def create_datasets(
    train_file: Path,
    val_file: Path,
    test_file: Path,
    src_model_path: Path,
    tgt_model_path: Path,
    cfg: Config,
) -> tuple[NMTDataset, NMTDataset, NMTDataset, SentencePieceVocab, SentencePieceVocab]:
    """Create train/val/test datasets with SentencePiece vocabularies.

    Args:
        train_file, val_file, test_file: Data file paths.
        src_model_path, tgt_model_path: SentencePiece model paths.
        cfg: Configuration object.

    Returns:
        Tuple of (train_dataset, val_dataset, test_dataset, src_vocab, tgt_vocab).
    """
    print("\n" + "=" * 70)
    print("STEP 3: Dataset Creation")
    print("=" * 70)

    start_time = time.time()

    # Load SentencePiece vocabularies
    print("\nLoading SentencePiece models...")
    src_vocab = SentencePieceVocab(str(src_model_path), config=cfg)
    tgt_vocab = SentencePieceVocab(str(tgt_model_path), config=cfg)

    print(f"  Cebuano vocab size: {len(src_vocab):,}")
    print(f"  Mandarin vocab size: {len(tgt_vocab):,}")

    # Create datasets
    print("\nCreating datasets...")
    train_dataset = NMTDataset(
        train_file,
        src_vocab=src_vocab,
        tgt_vocab=tgt_vocab,
        src_col=cfg.src_col,
        tgt_col=cfg.tgt_col,
        config=cfg,
    )

    val_dataset = NMTDataset(
        val_file,
        src_vocab=src_vocab,
        tgt_vocab=tgt_vocab,
        src_col=cfg.src_col,
        tgt_col=cfg.tgt_col,
        config=cfg,
    )

    test_dataset = NMTDataset(
        test_file,
        src_vocab=src_vocab,
        tgt_vocab=tgt_vocab,
        src_col=cfg.src_col,
        tgt_col=cfg.tgt_col,
        config=cfg,
    )

    print(f"  Train dataset: {len(train_dataset):,} examples")
    print(f"  Val dataset:   {len(val_dataset):,} examples")
    print(f"  Test dataset:  {len(test_dataset):,} examples")

    elapsed = time.time() - start_time
    print(f"\n✓ Dataset creation complete in {elapsed:.1f}s")

    return train_dataset, val_dataset, test_dataset, src_vocab, tgt_vocab


def create_model(src_vocab_size: int, tgt_vocab_size: int, cfg: Config) -> SimpleTransformer:
    """Create Transformer model.

    Args:
        src_vocab_size: Source vocabulary size.
        tgt_vocab_size: Target vocabulary size.
        cfg: Configuration object.

    Returns:
        Initialized SimpleTransformer model.
    """
    print("\n" + "=" * 70)
    print("STEP 4: Model Initialization")
    print("=" * 70)

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

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"\nModel Architecture:")
    print(f"  Encoder layers: {cfg.num_encoder_layers}")
    print(f"  Decoder layers: {cfg.num_decoder_layers}")
    print(f"  Model dimension: {cfg.d_model}")
    print(f"  Attention heads: {cfg.n_heads}")
    print(f"  Feed-forward dim: {cfg.d_ff}")
    print(f"\nParameters:")
    print(f"  Total: {total_params:,}")
    print(f"  Trainable: {trainable_params:,}")
    print(f"  Size: ~{total_params * 4 / 1024 / 1024:.1f} MB (fp32)")

    return model


def train(
    model: SimpleTransformer,
    train_dataset: NMTDataset,
    val_dataset: NMTDataset,
    cfg: Config,
    device: torch.device,
    num_epochs: int,
):
    """Train the model.

    Args:
        model: Model to train.
        train_dataset: Training dataset.
        val_dataset: Validation dataset.
        cfg: Configuration object.
        device: Device to train on.
        num_epochs: Number of training epochs.

    Returns:
        TrainResult with training metrics.
    """
    print("\n" + "=" * 70)
    print("STEP 5: Training")
    print("=" * 70)

    # Create data loaders
    print("\nCreating data loaders...")
    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=2,  # Parallel data loading
        pin_memory=True if device.type == 'cuda' else False,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=2,
        pin_memory=True if device.type == 'cuda' else False,
    )

    print(f"  Train batches: {len(train_loader):,}")
    print(f"  Val batches:   {len(val_loader):,}")

    # Training configuration
    print(f"\nTraining Configuration:")
    print(f"  Device: {device}")
    print(f"  Batch size: {cfg.batch_size}")
    print(f"  Learning rate: {cfg.learning_rate}")
    print(f"  Max epochs: {num_epochs}")
    print(f"  Gradient clipping: 1.0")
    print(f"  Label smoothing: {cfg.label_smoothing}")
    print(f"  Validation interval: every {cfg.val_interval} steps")
    print(f"  Save interval: every {cfg.save_interval} steps")
    print(f"  Early stopping patience: {cfg.patience} validations")

    # Create optimizer
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=cfg.learning_rate,
        betas=(0.9, 0.98),
        eps=1e-9,
    )

    # Learning rate scheduler (Transformer paper warmup)
    def lr_lambda(step):
        warmup_steps = 4000
        return min((step + 1) ** -0.5, (step + 1) * warmup_steps ** -1.5) * (cfg.d_model ** 0.5)

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # Create checkpoint directory
    cfg.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # Train
    print("\n" + "-" * 70)
    print("Starting training...")
    print("-" * 70 + "\n")

    start_time = time.time()

    result = train_model(
        model,
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=num_epochs,
        optimizer=optimizer,
        scheduler=scheduler,
        gradient_clip=1.0,
        device=device,
        config=cfg,
        save_dir=cfg.checkpoint_dir,
        use_amp=True if device.type == 'cuda' else False,  # Mixed precision on GPU
        log_every=100,  # Log every 100 steps
    )

    elapsed = time.time() - start_time
    hours = int(elapsed // 3600)
    minutes = int((elapsed % 3600) // 60)
    seconds = int(elapsed % 60)

    print("\n" + "=" * 70)
    print("TRAINING COMPLETE")
    print("=" * 70)
    print(f"\nTime: {hours}h {minutes}m {seconds}s")
    print(f"Final train loss: {result.train_losses[-1]:.4f}")
    if result.val_losses:
        print(f"Final val loss: {result.val_losses[-1]:.4f}")
        print(f"Best val loss: {min(result.val_losses):.4f}")
    if result.best_checkpoint:
        print(f"Best checkpoint: {result.best_checkpoint}")

    return result


def evaluate_inference(
    model: SimpleTransformer,
    test_dataset: NMTDataset,
    src_vocab: SentencePieceVocab,
    tgt_vocab: SentencePieceVocab,
    cfg: Config,
    device: torch.device,
):
    """Evaluate model with sample translations.

    Args:
        model: Trained model.
        test_dataset: Test dataset.
        src_vocab: Source vocabulary.
        tgt_vocab: Target vocabulary.
        cfg: Configuration object.
        device: Device to run on.
    """
    print("\n" + "=" * 70)
    print("STEP 6: Evaluation")
    print("=" * 70)

    model.eval()

    # Sample some test examples
    print("\nSample translations:")
    print("-" * 70)

    test_sentences = []
    reference_translations = []

    # Get 5 random examples from test set
    import random
    indices = random.sample(range(len(test_dataset)), min(5, len(test_dataset)))

    for idx in indices:
        src_text = test_dataset.src_sentences[idx]
        tgt_text = test_dataset.tgt_sentences[idx]
        test_sentences.append(src_text)
        reference_translations.append(tgt_text)

    # Translate
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

    # Display results
    for i, (src, ref, hyp) in enumerate(zip(test_sentences, reference_translations, translations)):
        print(f"\nExample {i + 1}:")
        print(f"  Source:     {src[:100]}..." if len(src) > 100 else f"  Source:     {src}")
        print(f"  Reference:  {ref[:100]}..." if len(ref) > 100 else f"  Reference:  {ref}")
        print(f"  Hypothesis: {hyp[:100]}..." if len(hyp) > 100 else f"  Hypothesis: {hyp}")

    print("\n" + "-" * 70)


def main():
    """Main training pipeline."""
    parser = argparse.ArgumentParser(description='Train Cebuano → Mandarin NMT model')

    # Model architecture
    parser.add_argument('--vocab-size', type=int, default=16000,
                        help='SentencePiece vocabulary size (default: 16000)')

    # Training hyperparameters
    parser.add_argument('--batch-size', type=int, default=32,
                        help='Batch size (default: 32)')
    parser.add_argument('--learning-rate', type=float, default=0.0001,
                        help='Learning rate (default: 0.0001)')
    parser.add_argument('--num-epochs', type=int, default=10,
                        help='Number of epochs (default: 10)')

    # Validation and checkpointing
    parser.add_argument('--val-interval', type=int, default=1000,
                        help='Validation interval in steps (default: 1000)')
    parser.add_argument('--save-interval', type=int, default=2000,
                        help='Checkpoint save interval in steps (default: 2000)')
    parser.add_argument('--patience', type=int, default=5,
                        help='Early stopping patience (default: 5)')

    # Experiment
    parser.add_argument('--experiment-name', type=str, default='baseline',
                        help='Experiment name for logging (default: baseline)')

    # Device
    parser.add_argument('--device', type=str, default='auto',
                        choices=['auto', 'cuda', 'cpu'],
                        help='Device to use (default: auto)')

    args = parser.parse_args()

    # Setup
    print("\n" + "=" * 70)
    print("CEBUANO → MANDARIN NEURAL MACHINE TRANSLATION")
    print("=" * 70)

    cfg, num_epochs = setup_config(args)

    # Determine device
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)

    print(f"\nConfiguration:")
    print(f"  Experiment: {cfg.experiment_name}")
    print(f"  Device: {device}")
    print(f"  Vocabulary size: {cfg.vocab_size:,}")
    print(f"  Batch size: {cfg.batch_size}")
    print(f"  Learning rate: {cfg.learning_rate}")

    # Pipeline
    train_file, val_file, test_file = load_and_preprocess_data(cfg)
    src_model_path, tgt_model_path = train_tokenizers(train_file, cfg)
    train_dataset, val_dataset, test_dataset, src_vocab, tgt_vocab = create_datasets(
        train_file, val_file, test_file, src_model_path, tgt_model_path, cfg
    )
    model = create_model(len(src_vocab), len(tgt_vocab), cfg)
    result = train(model, train_dataset, val_dataset, cfg, device, num_epochs)
    evaluate_inference(model, test_dataset, src_vocab, tgt_vocab, cfg, device)

    print("\n" + "=" * 70)
    print("ALL STEPS COMPLETE!")
    print("=" * 70)
    print(f"\nCheckpoint directory: {cfg.checkpoint_dir}")
    print(f"TensorBoard logs: {cfg.tensorboard_dir / cfg.experiment_name}")
    print("\nTo view training curves:")
    print(f"  tensorboard --logdir {cfg.tensorboard_dir}")


if __name__ == '__main__':
    main()
