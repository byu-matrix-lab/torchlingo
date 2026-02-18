"""
Simple training script for Cebuano → Mandarin Chinese NMT.

This demonstrates how TorchLingo handles all the heavy lifting.
The package takes care of data processing, tokenization, and training.
"""

from pathlib import Path
import pandas as pd
import torch

from torchlingo.config import Config
from torchlingo.preprocessing.base import save_data
from torchlingo.preprocessing.sentencepiece import preprocess_sentencepiece
from torchlingo.data_processing.vocab import SentencePieceVocab
from torchlingo.data_processing.dataset import NMTDataset
from torchlingo.data_processing.batching import collate_fn
from torchlingo.models import SimpleTransformer
from torchlingo.training import train_model


# Configuration - one simple config object
cfg = Config(
    # Paths
    data_dir=Path('data'),
    checkpoint_dir=Path('checkpoints/ceb_cmn'),

    # Model size
    d_model=512,
    n_heads=8,
    num_encoder_layers=6,
    num_decoder_layers=6,

    # Training
    batch_size=32,
    learning_rate=0.0001,
    # Note: num_epochs is passed to train_model(), not Config

    # Validation
    val_interval=1000,
    patience=5,

    # Tokenization
    vocab_size=16000,

    # Logging
    use_tensorboard=True,
    experiment_name='ceb_cmn_simple',

    # Data columns
    src_col='src_text',
    tgt_col='tgt_text',
)

# Number of training epochs (passed to train_model, not Config)
NUM_EPOCHS = 10


print("Loading and splitting data...")
# Load CSV and create splits
df = pd.read_csv(cfg.data_dir / 'ceb__cmn.csv')
df = df.dropna(subset=[cfg.src_col, cfg.tgt_col])  # Remove missing values
df = df.sample(frac=1, random_state=42).reset_index(drop=True)  # Shuffle

# 80/10/10 split
train_size = int(0.8 * len(df))
val_size = int(0.1 * len(df))

train_df = df[:train_size]
val_df = df[train_size:train_size + val_size]
test_df = df[train_size + val_size:]

print(f"Train: {len(train_df):,} | Val: {len(val_df):,} | Test: {len(test_df):,}")

# Save splits
save_data(train_df, cfg.data_dir / 'train.tsv', 'tsv')
save_data(val_df, cfg.data_dir / 'val.tsv', 'tsv')
save_data(test_df, cfg.data_dir / 'test.tsv', 'tsv')


print("\nTraining SentencePiece models...")
# Package handles all tokenization - just call one function
preprocess_sentencepiece(
    train_file=cfg.data_dir / 'train.tsv',
    val_file=cfg.data_dir / 'val.tsv',
    test_file=cfg.data_dir / 'test.tsv',
    config=cfg,
)


print("\nCreating datasets...")
# Package creates datasets from tokenized data
src_vocab = SentencePieceVocab(str(cfg.data_dir / 'sp_model.model'), config=cfg)
tgt_vocab = SentencePieceVocab(str(cfg.data_dir / 'sp_model.model'), config=cfg)

train_dataset = NMTDataset(
    cfg.data_dir / 'tokenized/train.tsv',
    src_vocab=src_vocab,
    tgt_vocab=tgt_vocab,
    config=cfg,
)

val_dataset = NMTDataset(
    cfg.data_dir / 'tokenized/val.tsv',
    src_vocab=src_vocab,
    tgt_vocab=tgt_vocab,
    config=cfg,
)

# Package handles batching
train_loader = torch.utils.data.DataLoader(
    train_dataset,
    batch_size=cfg.batch_size,
    shuffle=True,
    collate_fn=collate_fn,
)

val_loader = torch.utils.data.DataLoader(
    val_dataset,
    batch_size=cfg.batch_size,
    shuffle=False,
    collate_fn=collate_fn,
)


print("\nInitializing model...")
# Package provides ready-to-use models
model = SimpleTransformer(
    src_vocab_size=len(src_vocab),
    tgt_vocab_size=len(tgt_vocab),
    config=cfg,  # All model params come from config
)

print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")


print("\nTraining model...")
# Package handles all training - validation, checkpointing, logging
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
cfg.checkpoint_dir.mkdir(parents=True, exist_ok=True)

result = train_model(
    model,
    train_loader=train_loader,
    val_loader=val_loader,
    num_epochs=NUM_EPOCHS,  # Pass num_epochs here, not in Config
    device=device,
    config=cfg,
    save_dir=cfg.checkpoint_dir,
    use_amp=True if device.type == 'cuda' else False,
)


print("\n" + "=" * 60)
print("TRAINING COMPLETE!")
print("=" * 60)
print(f"Final train loss: {result.train_losses[-1]:.4f}")
print(f"Best checkpoint: {result.best_checkpoint}")
print(f"\nView training curves:")
print(f"  tensorboard --logdir runs")
