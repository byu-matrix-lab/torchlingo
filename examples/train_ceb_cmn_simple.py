"""
Simple training script for Cebuano → Mandarin Chinese NMT.

This demonstrates how TorchLingo handles all the heavy lifting.
The package takes care of data processing, tokenization, and training.
"""

from pathlib import Path
import pandas as pd
import torch
import sys

from torchlingo.config import Config
from torchlingo.preprocessing.base import save_data
from torchlingo.preprocessing.sentencepiece import preprocess_sentencepiece
from torchlingo.data_processing.vocab import SentencePieceVocab
from torchlingo.data_processing.dataset import NMTDataset
from torchlingo.data_processing.batching import collate_fn
from torchlingo.models import SimpleTransformer
from torchlingo.training import train_model
from torchlingo.checkpoint import load_checkpoint, save_checkpoint

# Confirm that gpu is available
if not torch.cuda.is_available():
    print("CUDA is not available. Training will be performed on CPU.")
else:
    print("CUDA is available. Training will be performed on GPU.")

print("Continue?")
input("Type 'yes' to continue or 'no' to stop: ")
if input().lower() == 'no':
    print("Stopping training.")
    # Stop the script
    sys.exit()


cfg = Config(
    data_dir=Path('data'),
    checkpoint_dir=Path('checkpoints/ceb_cmn_very_newest'),
    # LITE model: fits an 8 GB laptop GPU (~4.5 GB at batch 96) and is well
    # matched to ~650k short-sentence pairs. Scale up if you have more VRAM and
    # see underfitting (train loss plateaus high while val loss is still falling).
    d_model=256, n_heads=4, num_encoder_layers=4, num_decoder_layers=4, d_ff=1024,
    dropout=0.1, label_smoothing=0.1,
    max_seq_length=128,         # covers 100% of this dataset
    batch_size=96,              # ~4.5 GB; effective batch = batch_size * ACCUMULATION_STEPS
    num_steps=1_000_000,        # effectively uncapped; epochs + early-stop control training
    val_interval=500, patience=8,
    vocab_size=16000,
    use_tensorboard=True, experiment_name='ceb_cmn_very_newest',
    src_col='src_text', tgt_col='tgt_text',
)
NUM_EPOCHS = 20

# Accumulate gradients over this many micro-batches before each optimizer step.
# Effective batch = batch_size * ACCUMULATION_STEPS = 96 * 3 = 288, at the GPU
# memory cost of just batch_size=96. If you hit out-of-memory, lower batch_size
# and raise this to keep the same effective batch.
ACCUMULATION_STEPS = 3



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


print("\nTraining SentencePiece model...")
# This trains the shared SentencePiece model at data/sp_model.model (used for
# both source and target below). It also writes pre-tokenized copies under
# data/tokenized/, but we intentionally do NOT train on those — see note below.
preprocess_sentencepiece(
    train_file=cfg.data_dir / 'train.tsv',
    val_file=cfg.data_dir / 'val.tsv',
    test_file=cfg.data_dir / 'test.tsv',
    config=cfg,
)


print("\nCreating datasets...")
# SentencePieceVocab tokenizes raw text in its encode() method, so feed it the
# RAW split files — NOT the pre-tokenized files under data/tokenized/. Feeding
# already-tokenized text here would tokenize it a second time (turning the
# literal "▁" markers into their own pieces), so the model would train on a
# token stream that never matches what inference produces from raw text.
src_vocab = SentencePieceVocab(str(cfg.data_dir / 'sp_model.model'), config=cfg)
tgt_vocab = SentencePieceVocab(str(cfg.data_dir / 'sp_model.model'), config=cfg)

train_dataset = NMTDataset(
    cfg.data_dir / 'train.tsv',
    src_vocab=src_vocab,
    tgt_vocab=tgt_vocab,
    config=cfg,
)

val_dataset = NMTDataset(
    cfg.data_dir / 'val.tsv',
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

# Proper optimizer + Noam (inverse-sqrt) schedule — train_model's defaults aren't enough.
# base lr=1.0 lets the schedule set the actual lr; peak auto-scales with d_model
# (≈1e-3 at d_model=256, ≈7e-4 at d_model=512), reached at the end of warmup.
optimizer = torch.optim.Adam(model.parameters(), lr=1.0, betas=(0.9, 0.98), eps=1e-9)
WARMUP, dm = 6000, cfg.d_model
def noam(step):
    step = max(step, 1)
    return dm ** -0.5 * min(step ** -0.5, step * WARMUP ** -1.5)
scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, noam)

result = train_model(
    model, train_loader=train_loader, val_loader=val_loader,
    num_epochs=NUM_EPOCHS, optimizer=optimizer, scheduler=scheduler,
    gradient_clip=1.0, device=device, config=cfg,
    save_dir=cfg.checkpoint_dir, use_amp=(device.type == 'cuda'),
    accumulation_steps=ACCUMULATION_STEPS,
)


# Re-save the best checkpoint as a self-describing bundle that carries the
# SentencePiece tokenizer and model architecture with the weights. This way
# inference always uses the exact tokenizer the model was trained on — loading
# a different same-size vocabulary by mistake produces fluent nonsense, with no
# error to warn you.
if result.best_checkpoint is not None:
    # Reload the BEST weights that train_model saved during training. The model
    # still in memory holds the *final* weights, which may be worse than the best
    # (or even NaN if training diverged) — bundling those would clobber the good
    # checkpoint with bad weights.
    best_state = load_checkpoint(result.best_checkpoint, map_location='cpu')
    model.load_state_dict(best_state['model_state_dict'])

    sp_model_path = cfg.data_dir / 'sp_model.model'
    save_checkpoint(
        result.best_checkpoint,
        model,
        model_config={
            'src_vocab_size': len(src_vocab),
            'tgt_vocab_size': len(tgt_vocab),
            'd_model': cfg.d_model,
            'n_heads': cfg.n_heads,
            'num_encoder_layers': cfg.num_encoder_layers,
            'num_decoder_layers': cfg.num_decoder_layers,
            'd_ff': cfg.d_ff,
            'max_seq_length': cfg.max_seq_length,
            'dropout': cfg.dropout,
        },
        src_sp_model=sp_model_path,
        tgt_sp_model=sp_model_path,
    )
    print(f"Saved self-describing checkpoint (weights + tokenizer) to {result.best_checkpoint}")


print("\n" + "=" * 60)
print("TRAINING COMPLETE!")
print("=" * 60)
print(f"Final train loss: {result.train_losses[-1]:.4f}")
print(f"Best checkpoint: {result.best_checkpoint}")
print(f"\nView training curves:")
print(f"  tensorboard --logdir runs")
