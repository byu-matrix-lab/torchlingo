"""Train the small en-es model the real-data tutorial loads.

Run:
    python scripts/train_example_model.py

Trains on `data/example.tsv`, held out **by talk** so the evaluation set is
genuinely unseen text rather than neighbouring sentences from talks the model
already memorized. Writes a SentencePiece model and a checkpoint under
`data/pretrained/`, both small enough to commit.

This exists because every tutorial before it trained on a dozen toy phrases, so
a student could finish the whole sequence without once seeing a model translate
a sentence it had not been trained on.

Note:
    The resulting model is **not good**. Roughly 58,000 sentence pairs and a few
    minutes of training buys output that is recognisably Spanish, often
    grammatical, and frequently wrong. That is the honest result at this scale
    and the tutorial says so. Tuning until the numbers looked respectable would
    take far longer than a tutorial should and would misrepresent what this much
    data can do.
"""

import argparse
import random
import tempfile
import time
from functools import partial
from pathlib import Path

import pandas as pd
import torch
from torch import optim
from torch.utils.data import DataLoader

from torchlingo.config import Config
from torchlingo.data_processing.batching import collate_fn
from torchlingo.data_processing.dataset import NMTDataset
from torchlingo.data_processing.vocab import SentencePieceVocab
from torchlingo.models import SimpleTransformer
from torchlingo.preprocessing.sentencepiece import train_sentencepiece
from torchlingo.training import train_model

CORPUS = Path("data/example.tsv")
OUT_DIR = Path("data/pretrained")


def split_by_talk(
    frame: pd.DataFrame,
    seed: int,
    val_talks: int,
    test_talks: int,
    pinned_test: set[str] | None = None,
):
    """Split along talk boundaries so held-out text is genuinely unseen.

    Args:
        frame (pd.DataFrame): Corpus carrying a ``talk`` column.
        seed (int): Shuffle seed for choosing held-out talks.
        val_talks (int): How many talks to hold out for validation.
        test_talks (int): How many talks to hold out for test.
        pinned_test (set[str] | None): Use exactly these talks as the test set
            instead of drawing them. See ``--hold-out-talks-from``.

    Returns:
        tuple: Train, validation and test frames.
    """
    talks = sorted(frame["talk"].unique())
    if pinned_test is not None:
        missing = pinned_test - set(talks)
        if missing:
            raise SystemExit(
                f"{len(missing)} pinned test talks are not in this corpus, so the "
                f"comparison would not be like-for-like: {sorted(missing)[:3]}"
            )
        test = set(pinned_test)
        remaining = [talk for talk in talks if talk not in test]
        random.Random(seed).shuffle(remaining)
        val = set(remaining[:val_talks])
        train = set(remaining[val_talks:])
    else:
        random.Random(seed).shuffle(talks)
        test = set(talks[:test_talks])
        val = set(talks[test_talks : test_talks + val_talks])
        train = set(talks[test_talks + val_talks :])

    pick = lambda group: frame[frame["talk"].isin(group)]
    return pick(train), pick(val), pick(test)


def main() -> None:
    """Train, evaluate on held-out talks, and save what the tutorial needs."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--corpus", type=Path, default=CORPUS)
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    parser.add_argument("--vocab-size", type=int, default=3000)
    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--layers", type=int, default=3)
    parser.add_argument("--d-ff", type=int, default=512)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--max-words", type=int, default=25)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--hold-out-talks-from",
        type=Path,
        default=None,
        help="A TSV with a `talk` column whose talks become the test set, "
        "instead of drawing one by seed. Use this to score a new model on the "
        "same held-out text as an older one: the split is drawn from the talk "
        "list, so adding talks to the corpus silently changes which are held "
        "out, and two models measured on different test sets cannot be "
        "compared.",
    )
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    frame = pd.read_csv(args.corpus, sep="\t", dtype=str, keep_default_na=False)
    frame = frame[frame["kind"] == "transcript"]
    # Long sentences cost quadratic attention for little teaching value here.
    keep = (frame["src"].str.split().str.len() <= args.max_words) & (
        frame["tgt"].str.split().str.len() <= args.max_words
    )
    frame = frame[keep]

    pinned = None
    if args.hold_out_talks_from:
        previous = pd.read_csv(
            args.hold_out_talks_from, sep="\t", dtype=str, keep_default_na=False
        )
        pinned = set(previous["talk"].unique())
        print(
            f"holding out the {len(pinned)} talks from "
            f"{args.hold_out_talks_from}, so this model is scored on exactly "
            "the text the previous one was",
            flush=True,
        )

    train_df, val_df, test_df = split_by_talk(frame, args.seed, 20, 20, pinned)
    print(
        f"train {train_df['talk'].nunique()} talks / {len(train_df):,} pairs | "
        f"val {len(val_df):,} | test {len(test_df):,}",
        flush=True,
    )

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        train_file = tmp_path / "train.tsv"
        val_file = tmp_path / "val.tsv"
        train_df[["src", "tgt"]].to_csv(train_file, sep="\t", index=False)
        val_df[["src", "tgt"]].to_csv(val_file, sep="\t", index=False)
        # Keep the talk column on the held-out set. It is what lets anyone
        # verify the split was taken by talk rather than by sentence, which is
        # the claim the whole tutorial rests on.
        test_df[["src", "tgt", "talk"]].to_csv(
            args.out_dir / "test.tsv", sep="\t", index=False
        )

        # One joint model over both languages: en and es share an alphabet and
        # plenty of vocabulary, and one model is half the embedding rows.
        prefix = args.out_dir / "spm"
        train_sentencepiece(
            input_files=[train_file],
            model_prefix=str(prefix),
            vocab_size=args.vocab_size,
            model_type="bpe",
        )
        vocab = SentencePieceVocab(str(prefix) + ".model")
        print(f"sentencepiece vocab: {len(vocab):,}", flush=True)

        cfg = Config(batch_size=args.batch_size, learning_rate=3e-4)
        collate = partial(collate_fn, pad_idx=vocab.pad_idx)
        train_ds = NMTDataset(
            train_file, src_vocab=vocab, tgt_vocab=vocab, max_length=args.max_words * 3
        )
        val_ds = NMTDataset(
            val_file, src_vocab=vocab, tgt_vocab=vocab, max_length=args.max_words * 3
        )
        train_loader = DataLoader(
            train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate
        )
        val_loader = DataLoader(val_ds, batch_size=args.batch_size, collate_fn=collate)

        model = SimpleTransformer(
            src_vocab_size=len(vocab),
            tgt_vocab_size=len(vocab),
            d_model=args.d_model,
            n_heads=4,
            num_encoder_layers=args.layers,
            num_decoder_layers=args.layers,
            d_ff=args.d_ff,
            dropout=0.1,
            config=cfg,
        )
        params = sum(p.numel() for p in model.parameters())
        print(
            f"parameters: {params:,} ({params * 4 / 1e6:.1f} MB as float32)", flush=True
        )

        device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        started = time.time()
        result = train_model(
            model,
            train_loader,
            val_loader,
            num_epochs=args.epochs,
            optimizer=optim.Adam(model.parameters(), lr=3e-4, betas=(0.9, 0.98)),
            device=device,
            config=cfg,
            gradient_clip=1.0,
        )
        minutes = (time.time() - started) / 60

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "model_config": {
                "d_model": args.d_model,
                "n_heads": 4,
                "num_encoder_layers": args.layers,
                "num_decoder_layers": args.layers,
                "d_ff": args.d_ff,
            },
            "vocab_size": len(vocab),
            "train_losses": result.train_losses,
            "val_losses": result.val_losses,
            "trained_minutes": round(minutes, 1),
            "train_pairs": len(train_df),
        },
        args.out_dir / "model.pt",
    )

    size_mb = (args.out_dir / "model.pt").stat().st_size / 1e6
    print(
        f"\ntrained {args.epochs} epochs in {minutes:.1f} min; "
        f"final val loss {result.val_losses[-1]:.4f}",
        flush=True,
    )
    print(f"wrote {args.out_dir}/model.pt ({size_mb:.1f} MB) and spm.model", flush=True)


if __name__ == "__main__":
    main()
