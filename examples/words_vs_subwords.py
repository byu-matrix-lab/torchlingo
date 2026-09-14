"""Why the real corpus needs subwords, measured rather than asserted.

Run:
    python examples/words_vs_subwords.py

Every tutorial so far has used a toy corpus of a dozen phrases, where a
word-level vocabulary is obviously fine. This measures what happens when you
point the same approach at `data/example.tsv`, and then what SentencePiece
changes.

The question is not "which is better" in the abstract. It is three concrete
numbers a student can check:

1. **How big is the vocabulary?** Every entry costs an embedding row, so this is
   most of a small model's parameter budget.
2. **How often does a held-out sentence contain a word the model has never
   seen?** Those become `<unk>`, and a token the model cannot represent is a
   token it cannot translate.
3. **What does that cost in sequence length?** Subwords are not free: splitting
   rare words into pieces makes every sequence longer, and attention cost grows
   with the square of length.

The split is taken **by talk**, not by sentence. Consecutive sentences in a TED
transcript share a speaker, a topic and a vocabulary, so a random sentence split
leaks: the held-out set would be far easier than genuinely unseen text, and the
OOV rate would flatter the word-level approach for the wrong reason.
"""

import argparse
import collections
import random
import tempfile
from pathlib import Path

import pandas as pd

from torchlingo.data_processing.vocab import SentencePieceVocab
from torchlingo.preprocessing.sentencepiece import train_sentencepiece

CORPUS = Path("data/example.tsv")


def split_by_talk(frame: pd.DataFrame, seed: int = 0, held_out: float = 0.1):
    """Split into train and held-out sets along talk boundaries.

    Args:
        frame (pd.DataFrame): Corpus with a ``talk`` column.
        seed (int, optional): Shuffle seed.
        held_out (float, optional): Fraction of talks held out.

    Returns:
        tuple: ``(train, test)`` frames sharing no talk.
    """
    talks = sorted(frame["talk"].unique())
    random.Random(seed).shuffle(talks)
    cut = int(len(talks) * (1 - held_out))
    train_talks = set(talks[:cut])
    is_train = frame["talk"].isin(train_talks)
    return frame[is_train], frame[~is_train]


def word_stats(train: pd.DataFrame, test: pd.DataFrame, min_freq: int) -> dict:
    """Measure a word-level vocabulary built from the training split."""
    counts = collections.Counter(" ".join(train["src"]).lower().split())
    vocab = {word for word, n in counts.items() if n >= min_freq}

    tokens = unknown = 0
    for sentence in test["src"]:
        for word in sentence.lower().split():
            tokens += 1
            unknown += word not in vocab
    lengths = [len(s.split()) for s in test["src"]]
    return {
        "vocab": len(vocab),
        "oov_rate": unknown / max(tokens, 1),
        "mean_len": sum(lengths) / max(len(lengths), 1),
    }


def subword_stats(train: pd.DataFrame, test: pd.DataFrame, vocab_size: int) -> dict:
    """Train SentencePiece on the training split and measure the same things."""
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        train_file = tmp_path / "train.tsv"
        train[["src", "tgt"]].to_csv(train_file, sep="\t", index=False)

        train_sentencepiece(
            input_files=[train_file],
            model_prefix=str(tmp_path / "sp"),
            vocab_size=vocab_size,
            model_type="bpe",
        )
        vocab = SentencePieceVocab(str(tmp_path / "sp.model"))

        tokens = unknown = 0
        lengths = []
        for sentence in test["src"]:
            ids = vocab.encode(sentence, add_special_tokens=False)
            lengths.append(len(ids))
            tokens += len(ids)
            unknown += sum(1 for i in ids if i == vocab.unk_idx)

        sample = test["src"].iloc[0]
        pieces = [vocab.idx_to_token(i) for i in vocab.encode(sample, False)]

    return {
        "vocab": len(vocab),
        "oov_rate": unknown / max(tokens, 1),
        "mean_len": sum(lengths) / max(len(lengths), 1),
        "sample": sample,
        "pieces": pieces,
    }


def main() -> None:
    """Compare word-level and subword vocabularies on a held-out split."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--corpus", type=Path, default=CORPUS)
    parser.add_argument("--vocab-size", type=int, default=8000)
    parser.add_argument("--min-freq", type=int, default=2)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    frame = pd.read_csv(args.corpus, sep="\t", dtype=str, keep_default_na=False)
    frame = frame[frame["kind"] == "transcript"]
    train, test = split_by_talk(frame, seed=args.seed)

    print(
        f"{train['talk'].nunique()} talks / {len(train):,} pairs for training, "
        f"{test['talk'].nunique()} talks / {len(test):,} held out. "
        "No talk appears in both."
    )
    print()

    words = word_stats(train, test, args.min_freq)
    subwords = subword_stats(train, test, args.vocab_size)

    print(f"{'':22s} {'vocabulary':>12s} {'OOV rate':>10s} {'mean length':>12s}")
    print("-" * 60)
    print(
        f"{f'words (freq >= {args.min_freq})':22s} {words['vocab']:>12,} "
        f"{words['oov_rate']:>9.2%} {words['mean_len']:>12.1f}"
    )
    print(
        f"{'subwords (BPE)':22s} {subwords['vocab']:>12,} "
        f"{subwords['oov_rate']:>9.2%} {subwords['mean_len']:>12.1f}"
    )
    print()

    ratio = words["vocab"] / max(subwords["vocab"], 1)
    longer = subwords["mean_len"] / max(words["mean_len"], 1)
    print(f"Vocabulary is {ratio:.1f}x smaller, and sequences {longer:.1f}x longer.")
    print(
        f"On held-out text, {words['oov_rate']:.1%} of words are unknown to the "
        f"word-level model and become <unk>."
    )
    print("A token the model cannot represent is a token it cannot translate.")
    print()
    print("What the split actually looks like:")
    print(f"  {subwords['sample'][:78]}")
    print(f"  {' '.join(subwords['pieces'])[:78]}")
    print()
    print(
        "Notice that common words survive whole and only rare ones are broken up.\n"
        "That is the trade: a bounded vocabulary and no unknown tokens, paid for\n"
        "with longer sequences."
    )


if __name__ == "__main__":
    main()
