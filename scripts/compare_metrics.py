"""Score one set of translations under three metrics, so they can disagree in public.

Every number a student meets in this project is BLEU. That is one metric's
opinion, and the docs never say so. This script scores the *same* translations
from the tutorial 5 checkpoint under BLEU, chrF and TER, and records what each
one says.

Three things it is built to show, none of which survive a single-metric table:

- **They disagree about how good the model is.** Not by rounding -- BLEU and
  chrF can differ by tens of points on identical output, because one counts
  word n-grams and the other character n-grams.
- **TER runs the other way.** Lower is better, 0 is perfect. Put it in a column
  beside BLEU without saying so and every reader misreads it once.
- **They disagree about *ranking*, which is what actually matters.** A metric is
  usually used to choose between two systems. Scoring greedy against beam under
  all three shows whether the choice depends on the metric.

BLEU's collapse on short text is the sharpest single illustration and needs no
model at all, so it is computed here too, from the example in
``evaluation.py``'s module docstring: two translations, one exact and one close,
scoring 0.00 under BLEU and 100.00 under chrF.

Writes ``docs/docs/_generated/metric_comparison.json`` (the single source of
truth) and ``metric_comparison.md`` (generated from it, included into the
evaluation page). Nothing is hand-typed into the docs.

Run:
    python scripts/compare_metrics.py
"""

import argparse
import json
import sys
from pathlib import Path

import pandas as pd
import torch

from torchlingo.data_processing.vocab import SentencePieceVocab
from torchlingo.evaluation import compute_bleu, compute_chrf, compute_ter
from torchlingo.inference import beam_search_decode, greedy_decode
from torchlingo.models import SimpleTransformer

DEFAULT_OUT = Path("docs/docs/_generated")
PRETRAINED = Path("data/pretrained")

# The docstring example from evaluation.py: one exact translation and one close
# one, too short to contain a 4-gram.
SHORT_PREDICTIONS = ["Hello world", "How are you"]
SHORT_REFERENCES = ["Hello world", "How are you doing"]


def is_available(path: Path) -> bool:
    """Report whether a file holds real content rather than a Git LFS pointer.

    Args:
        path (Path): File to test.

    Returns:
        bool: True if the file exists and its content has been fetched.
    """
    if not path.exists():
        return False
    with path.open("rb") as handle:
        return not handle.read(23).startswith(b"version https://git-lfs")


def load_model(device: torch.device):
    """Load the tutorial 5 checkpoint and its tokenizer.

    Args:
        device (torch.device): Where to place the model.

    Returns:
        tuple: The model and its vocabulary.
    """
    checkpoint = torch.load(
        PRETRAINED / "model.pt", map_location=device, weights_only=False
    )
    vocab = SentencePieceVocab(str(PRETRAINED / "spm.model"))
    model = SimpleTransformer(
        src_vocab_size=checkpoint["vocab_size"],
        tgt_vocab_size=checkpoint["vocab_size"],
        **checkpoint["model_config"],
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, vocab


def translate_all(sentences, model, vocab, device, max_len, beam_size=None):
    """Translate every sentence with one decoding configuration.

    Args:
        sentences (list[str]): Source sentences.
        model: The translation model.
        vocab: Shared source and target vocabulary.
        device (torch.device): Where to run.
        max_len (int): Decoding length cap.
        beam_size (int | None): Beam width, or None for greedy.

    Returns:
        list[str]: One translation per input sentence.
    """
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


def score_all(hypotheses: list[str], references: list[str]) -> dict:
    """Score one set of translations under all three metrics.

    Args:
        hypotheses (list[str]): Translations to score.
        references (list[str]): Reference translations.

    Returns:
        dict: One entry per metric, each with its score and the direction it
            runs in, because TER runs the opposite way to the other two.
    """
    return {
        "bleu": {
            "score": round(compute_bleu(hypotheses, references).score, 2),
            "higher_is_better": True,
            "counts": "word n-grams, 1 to 4",
        },
        "chrf": {
            "score": round(compute_chrf(hypotheses, references).score, 2),
            "higher_is_better": True,
            "counts": "character n-grams, plus word n-grams up to order 2",
        },
        "ter": {
            "score": round(compute_ter(hypotheses, references).score, 2),
            "higher_is_better": False,
            "counts": "edits per reference word",
        },
    }


def short_text_demonstration() -> dict:
    """Score the two-sentence example that makes BLEU collapse.

    Returns:
        dict: The predictions, references and each metric's verdict.
    """
    return {
        "predictions": SHORT_PREDICTIONS,
        "references": SHORT_REFERENCES,
        "note": "one translation exact, one close; neither long enough for a 4-gram",
        "metrics": score_all(SHORT_PREDICTIONS, SHORT_REFERENCES),
    }


def render_markdown(results: dict) -> str:
    """Render the tables that the evaluation page includes.

    Args:
        results (dict): The structure written to metric_comparison.json.

    Returns:
        str: Markdown, generated rather than hand-typed.
    """
    short = results["short_text"]["metrics"]
    greedy = results["held_out"]["greedy"]
    beam = results["held_out"]["beam"]
    settings = results["settings"]

    def arrow(metric: str) -> str:
        return (
            "higher is better"
            if greedy[metric]["higher_is_better"]
            else "**lower** is better"
        )

    def rank(metric: str) -> str:
        g, b = greedy[metric]["score"], beam[metric]["score"]
        if g == b:
            return "tie"
        better_is_beam = (b > g) if greedy[metric]["higher_is_better"] else (b < g)
        return "beam" if better_is_beam else "greedy"

    winners = {rank(m) for m in ("bleu", "chrf", "ter")}
    verdict = (
        f"All three prefer **{winners.pop()}**."
        if len(winners) == 1
        else "**The metrics disagree about which decoder is better.**"
    )

    return f"""<!-- Generated by scripts/compare_metrics.py. Do not edit by hand. -->

**The same {settings["n_sentences"]} held-out sentences, scored three ways.**

| Metric | Greedy | Beam {settings["beam_size"]} | Direction | What it counts |
|---|---|---|---|---|
| BLEU | {greedy["bleu"]["score"]} | {beam["bleu"]["score"]} | {arrow("bleu")} | {greedy["bleu"]["counts"]} |
| chrF | {greedy["chrf"]["score"]} | {beam["chrf"]["score"]} | {arrow("chrf")} | {greedy["chrf"]["counts"]} |
| TER | {greedy["ter"]["score"]} | {beam["ter"]["score"]} | {arrow("ter")} | {greedy["ter"]["counts"]} |

{verdict}

**And on two short sentences, one of them a perfect translation:**

| Metric | Score |
|---|---|
| BLEU | {short["bleu"]["score"]} |
| chrF | {short["chrf"]["score"]} |
| TER | {short["ter"]["score"]} |

Reproduce with `python scripts/compare_metrics.py`. Signatures for every score
are in `metric_comparison.json`.
"""


def main() -> int:
    """Measure and write both artifacts.

    Returns:
        int: Process exit code.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sentences", type=int, default=200)
    parser.add_argument("--beam-size", type=int, default=5)
    parser.add_argument("--max-len", type=int, default=60)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    if not is_available(PRETRAINED / "model.pt"):
        print(
            "data/pretrained/model.pt is absent or an unfetched Git LFS pointer; "
            "run `git lfs pull` to fetch it.",
            file=sys.stderr,
        )
        return 1

    device = torch.device("cpu")
    model, vocab = load_model(device)
    test = pd.read_csv(
        PRETRAINED / "test.tsv", sep="\t", dtype=str, keep_default_na=False
    )
    subset = test.sample(args.sentences, random_state=args.seed)
    references = list(subset["tgt"])

    print(f"decoding {args.sentences} held-out sentences, greedy ...", flush=True)
    greedy_out = translate_all(list(subset["src"]), model, vocab, device, args.max_len)
    print(f"decoding the same {args.sentences}, beam {args.beam_size} ...", flush=True)
    beam_out = translate_all(
        list(subset["src"]), model, vocab, device, args.max_len, args.beam_size
    )

    results = {
        "settings": {
            "n_sentences": args.sentences,
            "beam_size": args.beam_size,
            "max_len": args.max_len,
            "seed": args.seed,
            "torch": torch.__version__,
        },
        "held_out": {
            "greedy": score_all(greedy_out, references),
            "beam": score_all(beam_out, references),
        },
        "short_text": short_text_demonstration(),
        "bleu_signature": str(
            compute_bleu(greedy_out, references).signature
            if hasattr(compute_bleu(greedy_out, references), "signature")
            else "not recorded"
        ),
    }

    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "metric_comparison.json").write_text(
        json.dumps(results, indent=2, ensure_ascii=False) + "\n"
    )
    (args.out_dir / "metric_comparison.md").write_text(render_markdown(results))
    print(f"\nwrote {args.out_dir}/metric_comparison.json and metric_comparison.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
