"""Score two checkpoints on the same held-out text, so the difference means something.

Retraining on more data raises an obvious question and a trap. The question is
whether the extra data helped. The trap is that `train_example_model.py` draws
its held-out talks by shuffling the talk list, so *adding talks to the corpus
changes which talks are held out*. Two models trained before and after such a
change are measured on different text, and the difference between their scores
says nothing about either.

Concretely: after the corpus grew by 98 talks, an unpinned split held out zero
of the original 20 test talks. Comparing those two BLEU numbers would have been
meaningless, and nothing about the numbers themselves would have shown it.

So `train_example_model.py` grew a `--hold-out-talks-from` flag, and this script
scores any two checkpoints on one test set with one set of decoding options. It
reports BLEU, and also the things BLEU hides: output length against the
reference, and how often the two models agree with each other.

Run:
    python scripts/compare_checkpoints.py \\
        --baseline old/model.pt --candidate data/pretrained/model.pt \\
        --test data/pretrained/test.tsv

Each checkpoint needs its tokenizer beside it as `spm.model`, which is how
`train_example_model.py` writes them.
"""

import argparse
import json
import sys
import time
from pathlib import Path

import pandas as pd
import torch

from torchlingo.data_processing.vocab import SentencePieceVocab
from torchlingo.inference import beam_search_decode, greedy_decode
from torchlingo.models import SimpleTransformer


def load(checkpoint_path: Path, device: torch.device):
    """Load a checkpoint and the tokenizer stored beside it.

    Args:
        checkpoint_path (Path): Path to ``model.pt``.
        device (torch.device): Where to place the model.

    Returns:
        tuple: The model, its vocabulary, and the checkpoint dict.
    """
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    spm_path = checkpoint_path.parent / "spm.model"
    if not spm_path.exists():
        raise SystemExit(f"no tokenizer at {spm_path}; it must sit beside the model")
    vocab = SentencePieceVocab(str(spm_path))
    model = SimpleTransformer(
        src_vocab_size=checkpoint["vocab_size"],
        tgt_vocab_size=checkpoint["vocab_size"],
        **checkpoint["model_config"],
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, vocab, checkpoint


def describe_setup(checkpoint: dict) -> dict:
    """Pull out everything that has to match for a comparison to mean anything.

    `epochs_run` rather than a requested count, because early stopping can end
    a run sooner, and because reading an epoch count off the wrong field is the
    specific mistake this function exists to catch.

    Older checkpoints predate these fields. A missing value reads as None and
    is reported as unknown rather than assumed equal, since "we cannot tell"
    and "they match" are different answers.

    Args:
        checkpoint (dict): A loaded checkpoint.

    Returns:
        dict: The comparable training settings.
    """
    epochs = checkpoint.get("epochs_run")
    if epochs is None and checkpoint.get("train_losses"):
        # Inferable: train_losses holds exactly one entry per epoch.
        epochs = len(checkpoint["train_losses"])
    return {
        "epochs": epochs,
        "train_pairs": checkpoint.get("train_pairs"),
        "seed": checkpoint.get("seed"),
        "model_config": checkpoint.get("model_config"),
        "vocab_size": checkpoint.get("vocab_size"),
    }


def compare_setups(baseline: dict, candidate: dict) -> list[str]:
    """List the training settings that differ between two checkpoints.

    A BLEU difference only tells you about the thing you changed if exactly one
    thing changed. This does not decide what "the thing" is -- that is the
    experimenter's intent, which no script can read -- it just refuses to let
    the other differences stay invisible.

    Args:
        baseline (dict): Output of :func:`describe_setup` for the baseline.
        candidate (dict): The same for the candidate.

    Returns:
        list[str]: One human-readable line per differing setting.
    """
    differences = []
    for key in ("epochs", "train_pairs", "seed", "vocab_size", "model_config"):
        before, after = baseline.get(key), candidate.get(key)
        if before is None or after is None:
            if before != after:
                differences.append(f"{key}: {before} vs {after} (one is unknown)")
            continue
        if before != after:
            differences.append(f"{key}: {before} vs {after}")
    return differences


def translate(
    sentences: list[str], model, vocab, device, max_len: int, beam_size: int | None
) -> list[str]:
    """Translate every sentence with one decoding configuration."""
    out = []
    for sentence in sentences:
        ids = vocab.encode(sentence, add_special_tokens=True)
        src = torch.tensor([ids]).to(device)
        with torch.no_grad():
            if beam_size is None:
                tokens = greedy_decode(model, src, max_len=max_len)[0]
            else:
                tokens = beam_search_decode(
                    model, src, beam_size=beam_size, max_len=max_len
                )
        out.append(vocab.decode(tokens, skip_special_tokens=True))
    return out


def score(hypotheses: list[str], references: list[str]) -> dict:
    """BLEU plus the things BLEU hides.

    Records the sacreBLEU signature alongside the score. BLEU is sensitive
    enough to tokenization that a bare number is not comparable to anyone
    else's, and this script exists to make a comparison trustworthy -- so
    omitting the one string that says how the number was produced would be the
    wrong place to save a line.
    """
    from sacrebleu.metrics import BLEU

    metric = BLEU()
    return {
        "bleu": round(metric.corpus_score(hypotheses, [references]).score, 2),
        "bleu_signature": str(metric.get_signature()),
        "mean_length": round(
            sum(len(h.split()) for h in hypotheses) / len(hypotheses), 2
        ),
        "empty_outputs": sum(1 for h in hypotheses if not h.strip()),
    }


def paired_bootstrap(
    baseline: list[str],
    candidate: list[str],
    references: list[str],
    resamples: int,
    seed: int,
) -> dict:
    """Test whether the BLEU difference survives resampling the test set.

    Both systems are scored on the *same* resampled sentences every time, so
    the comparison is paired and the sentence-level difficulty that dominates
    BLEU cancels out. What remains is whether the gap depends on which
    sentences happened to be in the test set.

    This is the same discipline the decoding sweep needed: a BLEU difference is
    not a result until you know it is bigger than the noise. There the gap
    turned out to be smaller.

    Args:
        baseline (list[str]): Baseline translations.
        candidate (list[str]): Candidate translations.
        references (list[str]): Reference translations.
        resamples (int): Bootstrap resamples.
        seed (int): RNG seed, so the answer is reproducible.

    Returns:
        dict: Mean difference, its standard error, a 95% interval, and the
            share of resamples in which the candidate won.
    """
    import random

    from sacrebleu.metrics import BLEU

    bleu = BLEU()
    rng = random.Random(seed)
    n = len(references)
    deltas = []
    for _ in range(resamples):
        idx = [rng.randrange(n) for _ in range(n)]
        refs = [[references[i] for i in idx]]
        base = bleu.corpus_score([baseline[i] for i in idx], refs).score
        cand = bleu.corpus_score([candidate[i] for i in idx], refs).score
        deltas.append(cand - base)

    deltas.sort()
    mean = sum(deltas) / len(deltas)
    variance = sum((d - mean) ** 2 for d in deltas) / (len(deltas) - 1)
    return {
        "resamples": resamples,
        "mean_delta": round(mean, 3),
        "stderr": round(variance**0.5, 3),
        "ci_low": round(deltas[int(0.025 * len(deltas))], 3),
        "ci_high": round(deltas[int(0.975 * len(deltas))], 3),
        "candidate_wins": round(sum(d > 0 for d in deltas) / len(deltas), 4),
    }


def main() -> int:
    """Score both checkpoints on the same test set and report the difference."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--test", type=Path, required=True)
    parser.add_argument("--max-len", type=int, default=60)
    parser.add_argument("--beam-size", type=int, default=None)
    parser.add_argument("--limit", type=int, default=0, help="0 uses the whole set.")
    parser.add_argument(
        "--bootstrap",
        type=int,
        default=300,
        help="Paired bootstrap resamples; 0 to skip. A BLEU difference is not "
        "a result until you know it is bigger than the noise.",
    )
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    device = torch.device("cpu")
    test = pd.read_csv(args.test, sep="\t", dtype=str, keep_default_na=False)
    if args.limit:
        test = test.head(args.limit)
    sources, references = list(test["src"]), list(test["tgt"])

    results = {
        "test_set": str(args.test),
        "pairs": len(test),
        "talks": int(test["talk"].nunique()) if "talk" in test else None,
        "beam_size": args.beam_size,
        "reference_mean_length": round(
            sum(len(r.split()) for r in references) / len(references), 2
        ),
    }

    outputs = {}
    setups = {}
    for label, path in (("baseline", args.baseline), ("candidate", args.candidate)):
        model, vocab, checkpoint = load(path, device)
        setups[label] = describe_setup(checkpoint)
        started = time.perf_counter()
        outputs[label] = translate(
            sources, model, vocab, device, args.max_len, args.beam_size
        )
        row = score(outputs[label], references)
        row["train_pairs"] = checkpoint.get("train_pairs")
        row["epochs"] = setups[label]["epochs"]
        row["final_val_loss"] = round(checkpoint["val_losses"][-1], 4)
        row["seconds"] = round(time.perf_counter() - started, 1)
        results[label] = row
        print(
            f"{label:<10} BLEU={row['bleu']:<7} len={row['mean_length']:<7} "
            f"train_pairs={row['train_pairs']:,} epochs={row['epochs']} "
            f"val_loss={row['final_val_loss']}",
            flush=True,
        )

    # Say what else moved. A BLEU difference is about the variable you changed
    # only if that is the variable you changed, and this script used to report
    # the delta without ever looking at how the two models were trained. It
    # once compared a model trained for 36 epochs against one trained for 20
    # and the writeup credited the corpus.
    differences = compare_setups(setups["baseline"], setups["candidate"])
    results["setup_differences"] = differences
    results["controlled"] = len(differences) <= 1

    print()
    if not differences:
        print("setups identical: nothing varies, so nothing is being measured.")
    elif len(differences) == 1:
        print(f"one variable differs -- {differences[0]}")
    else:
        print(f"WARNING: {len(differences)} settings differ, so this comparison")
        print("cannot attribute the result to any one of them:")
        for line in differences:
            print(f"  - {line}")
        print("Hold the others fixed and re-run before drawing a conclusion.")

    agree = sum(a == b for a, b in zip(outputs["baseline"], outputs["candidate"]))
    results["identical_outputs"] = round(agree / len(sources), 4)
    results["bleu_delta"] = round(
        results["candidate"]["bleu"] - results["baseline"]["bleu"], 2
    )

    print()
    print(f"BLEU change: {results['bleu_delta']:+.2f}")
    print(f"scored with: {results['baseline']['bleu_signature']}")
    print(f"identical outputs: {results['identical_outputs']:.1%}")
    print(
        "References average "
        f"{results['reference_mean_length']} tokens; baseline "
        f"{results['baseline']['mean_length']}, candidate "
        f"{results['candidate']['mean_length']}."
    )

    if args.bootstrap:
        print(f"\nresampling the test set {args.bootstrap} times ...", flush=True)
        boot = paired_bootstrap(
            outputs["baseline"], outputs["candidate"], references, args.bootstrap, 0
        )
        results["bootstrap"] = boot
        print(
            f"BLEU change {boot['mean_delta']:+.2f} ± {boot['stderr']:.2f}, "
            f"95% CI [{boot['ci_low']:+.2f}, {boot['ci_high']:+.2f}], "
            f"candidate wins {boot['candidate_wins']:.1%} of resamples"
        )

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(results, indent=2) + "\n")
        print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
