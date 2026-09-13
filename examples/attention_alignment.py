"""Show that LSTM attention learns the alignment it is supposed to learn.

Run:
    python examples/attention_alignment.py

This example trains three small models on the same toy translation task and
compares them:

1. ``attention=False`` -- the classic bottlenecked seq2seq.
2. ``attn_type="dot"`` -- Luong dot-product attention.
3. ``attn_type="additive"`` -- Bahdanau additive attention.

**The task.** Every source sentence is a random sequence of words. The target is
that sentence translated word-for-word through a fixed bilingual lexicon *and
reversed*. So the first target word comes from the last source word:

    source:  the cat sees a small bird
    target:  pajaro pequeno un ve gato el

Reversal is the point. It forces the decoder to reach across the whole sentence,
which is exactly the dependency a fixed-size hidden state handles badly, and it
means we **know the correct alignment in advance**: an anti-diagonal. That turns
"does attention work?" into a number we can actually measure, rather than a
heatmap we squint at and hope about.

The task is synthetic on purpose. The parallel corpus shipped in ``data/`` is not
reliably sentence-aligned, so it cannot support an alignment demo -- and real
corpora have no alignment ground truth to check against anyway.
"""

import math
import random

import torch
from torch import nn, optim
from torch.utils.data import DataLoader, TensorDataset

from torchlingo.config import get_default_config
from torchlingo.models import SimpleSeq2SeqLSTM
from torchlingo.training import train_model
from torchlingo.visualization import format_attention, plot_attention

CFG = get_default_config()
SEED = 0
N_TRAIN, N_VAL = 4000, 400
MIN_WORDS, MAX_WORDS = 4, 8
EPOCHS = 12

# A tiny bilingual lexicon. The "languages" are invented so that nothing about
# the task depends on knowing Spanish -- the mapping is one-to-one and total.
LEXICON = {
    "the": "el",
    "a": "un",
    "cat": "gato",
    "dog": "perro",
    "bird": "pajaro",
    "fish": "pez",
    "child": "nino",
    "woman": "mujer",
    "man": "hombre",
    "sees": "ve",
    "eats": "come",
    "finds": "halla",
    "wants": "quiere",
    "small": "pequeno",
    "big": "grande",
    "red": "rojo",
    "blue": "azul",
    "old": "viejo",
    "quick": "rapido",
    "quiet": "callado",
}


def build_vocabs() -> tuple[dict[str, int], dict[str, int], list[str], list[str]]:
    """Build source and target vocabularies with the standard special tokens.

    Returns:
        tuple: ``(src_stoi, tgt_stoi, src_itos, tgt_itos)``.
    """
    specials = [CFG.pad_token, CFG.unk_token, CFG.sos_token, CFG.eos_token]
    src_itos = specials + sorted(LEXICON)
    tgt_itos = specials + sorted(LEXICON.values())
    return (
        {tok: i for i, tok in enumerate(src_itos)},
        {tok: i for i, tok in enumerate(tgt_itos)},
        src_itos,
        tgt_itos,
    )


def make_pair(
    rng: random.Random,
    src_stoi: dict[str, int],
    tgt_stoi: dict[str, int],
) -> tuple[list[int], list[int]]:
    """Sample one (source, target) id pair for the reversal task.

    Args:
        rng (random.Random): Seeded generator, so runs are reproducible.
        src_stoi (dict): Source token-to-id mapping.
        tgt_stoi (dict): Target token-to-id mapping.

    Returns:
        tuple: Source and target id lists, each wrapped in SOS/EOS.
    """
    words = [
        rng.choice(list(LEXICON)) for _ in range(rng.randint(MIN_WORDS, MAX_WORDS))
    ]
    translated = [LEXICON[w] for w in reversed(words)]
    src = [CFG.sos_idx] + [src_stoi[w] for w in words] + [CFG.eos_idx]
    tgt = [CFG.sos_idx] + [tgt_stoi[w] for w in translated] + [CFG.eos_idx]
    return src, tgt


def make_dataset(
    n: int,
    rng: random.Random,
    src_stoi: dict[str, int],
    tgt_stoi: dict[str, int],
) -> TensorDataset:
    """Build a padded TensorDataset of ``n`` sampled pairs.

    Args:
        n (int): Number of pairs.
        rng (random.Random): Seeded generator.
        src_stoi (dict): Source token-to-id mapping.
        tgt_stoi (dict): Target token-to-id mapping.

    Returns:
        TensorDataset: Yields ``(src, tgt)`` id tensors, right-padded.
    """
    pairs = [make_pair(rng, src_stoi, tgt_stoi) for _ in range(n)]
    width = MAX_WORDS + 2

    def pad(ids: list[int]) -> list[int]:
        return ids + [CFG.pad_idx] * (width - len(ids))

    src = torch.tensor([pad(s) for s, _ in pairs])
    tgt = torch.tensor([pad(t) for _, t in pairs])
    return TensorDataset(src, tgt)


def alignment_accuracy(model: nn.Module, n_samples: int = 200) -> float:
    """Measure how often peak attention lands on the truly aligned source word.

    Because the target is the reversed source, the target token predicted at
    decoder row ``j`` of a length-``n`` sentence comes from source index
    ``n - j`` (source index 0 holds SOS). A model that has learned the task
    should put its attention peak exactly there.

    Args:
        model (nn.Module): A model built with attention enabled.
        n_samples (int, optional): Number of held-out sentences to score.

    Returns:
        float: Fraction of decoder positions whose argmax hits the true source
            position. Chance is roughly ``1 / src_len``.
    """
    rng = random.Random(SEED + 99)
    src_stoi, tgt_stoi, _, _ = build_vocabs()
    model.eval()
    hits = total = 0
    for _ in range(n_samples):
        src_ids, tgt_ids = make_pair(rng, src_stoi, tgt_stoi)
        n = len(src_ids) - 2  # content words, excluding SOS and EOS
        src = torch.tensor([src_ids])
        tgt_input = torch.tensor([tgt_ids[:-1]])
        with torch.no_grad():
            _logits, weights = model(src, tgt_input, return_attention=True)
        peaks = weights[0].argmax(dim=-1)
        for row in range(n):
            total += 1
            hits += int(peaks[row].item() == n - row)
    return hits / max(total, 1)


def train_one(label: str, attention: bool, attn_type: str, loaders) -> tuple:
    """Train a single configuration and report its validation loss.

    Args:
        label (str): Human-readable name for the run.
        attention (bool): Whether to enable attention.
        attn_type (str): Scorer to use when attention is enabled.
        loaders (tuple): ``(train_loader, val_loader)``.

    Returns:
        tuple: ``(model, final_val_loss)``.
    """
    torch.manual_seed(SEED)
    src_stoi, tgt_stoi, _, _ = build_vocabs()
    model = SimpleSeq2SeqLSTM(
        src_vocab_size=len(src_stoi),
        tgt_vocab_size=len(tgt_stoi),
        emb_dim=64,
        hidden_dim=64,
        num_layers=1,
        dropout=0.0,
        attention=attention,
        attn_type=attn_type,
    )
    train_loader, val_loader = loaders
    # The library default learning rate is tuned for real corpora; this toy task
    # converges much faster with a larger step.
    optimizer = optim.Adam(model.parameters(), lr=3e-3)
    print(f"\n=== {label} ===", flush=True)
    result = train_model(
        model, train_loader, val_loader, num_epochs=EPOCHS, optimizer=optimizer
    )
    return model, result.val_losses[-1]


def main() -> None:
    """Train the three configurations and print the comparison."""
    rng = random.Random(SEED)
    src_stoi, tgt_stoi, src_itos, tgt_itos = build_vocabs()
    train_ds = make_dataset(N_TRAIN, rng, src_stoi, tgt_stoi)
    val_ds = make_dataset(N_VAL, rng, src_stoi, tgt_stoi)
    loaders = (
        DataLoader(train_ds, batch_size=64, shuffle=True),
        DataLoader(val_ds, batch_size=64),
    )

    rows = []
    attended = {}
    for label, attention, attn_type in [
        ("no attention", False, "dot"),
        ("dot (Luong)", True, "dot"),
        ("additive (Bahdanau)", True, "additive"),
    ]:
        model, val_loss = train_one(label, attention, attn_type, loaders)
        accuracy = alignment_accuracy(model) if attention else float("nan")
        rows.append((label, val_loss, accuracy))
        if attention:
            attended[attn_type] = model

    print("\n" + "=" * 64)
    print(f"{'configuration':22s} {'val loss':>10s} {'alignment acc':>15s}")
    print("-" * 64)
    for label, val_loss, accuracy in rows:
        acc = "n/a" if math.isnan(accuracy) else f"{accuracy:.1%}"
        print(f"{label:22s} {val_loss:10.4f} {acc:>15s}")
    print("=" * 64)

    # Show one sentence in detail. The expected pattern is an anti-diagonal:
    # the first target word attends to the last source word, and so on.
    demo_rng = random.Random(SEED + 7)
    src_ids, tgt_ids = make_pair(demo_rng, src_stoi, tgt_stoi)
    src_tokens = [src_itos[i] for i in src_ids]
    tgt_tokens = [tgt_itos[i] for i in tgt_ids[:-1]]
    model = attended["dot"]
    with torch.no_grad():
        _logits, weights = model(
            torch.tensor([src_ids]), torch.tensor([tgt_ids[:-1]]), return_attention=True
        )
    print("\nsource:", " ".join(src_tokens[1:-1]))
    print("target:", " ".join(tgt_tokens[1:]))
    print("\nattention (rows = decoder step, columns = source token):\n")
    print(format_attention(weights, src_tokens, tgt_tokens))

    out = "attention_alignment.png"
    plot_attention(weights, src_tokens, tgt_tokens, title="Luong dot attention")
    import matplotlib.pyplot as plt

    plt.savefig(out, dpi=150)
    print(f"\nheatmap written to {out}")


if __name__ == "__main__":
    main()
