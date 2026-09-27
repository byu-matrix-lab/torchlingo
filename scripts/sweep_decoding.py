"""Measure what the decoding knobs actually do, on a model whose answers differ.

`docs/docs/concepts/decoding.md` explains how beam search works and what it
costs, and `scripts/bench_decode.py` supplies the cost numbers. Neither answers
the question a student asks next: **what do I get for it?**

The docs previously left that to prose, and tutorial 3 left it to a sweep over
`beam_size` in 1, 2, 3, 5, 10 that printed the same translation on every row.
Not because beam size does not matter, but because tutorial 3's model is a toy
trained on a handful of phrases and is decisive enough that widening the beam
changes nothing. A table of five identical rows teaches the opposite of the
intended lesson.

This script runs the same sweep against the pretrained model from tutorial 5,
which is trained on real data and is wrong often enough to be interesting, and
reports three things per configuration:

- **BLEU**, the quality number people actually report.
- **Mean output length**, because that is where `alpha` shows its effect, and
  the effect is hard to read off the formula.
- **Share of sentences whose output differs from greedy**, which is the honest
  measure of whether the extra computation bought anything at all. A beam that
  reproduces greedy on 97% of sentences is doing very little for 5x the calls.

Two sweeps, because the knobs are independent:

1. `beam_size` at fixed `alpha`.
2. `alpha` at fixed `beam_size` — the length effect, which connects to the open
   question in #4 about normalizing during pruning rather than only at the end.

Each runs across several seeds, where a seed draws a different held-out subset.
Decoding is deterministic, so the subset is the *only* source of run-to-run
variation, and the spread across seeds is an honest error bar. Every
configuration sees the same subsets, so the comparisons are paired: the
per-seed difference cancels the sampling variance that otherwise swamps them.
That pairing is not a detail. Unpaired, no beam size is distinguishable from
any other; paired, the greedy-to-beam gap is unmistakable and the
beam-to-beam gaps are still not there.

Unlike `bench_decode.py`, this needs the real checkpoint: the whole point is a
model whose answers move. It skips cleanly when the checkpoint is absent or is
an unfetched Git LFS pointer, which is the state in CI.

Run:
    python scripts/sweep_decoding.py

Writes `docs/docs/_generated/decoding_sweep.json` (the single source of truth)
and `docs/docs/_generated/decoding_sweep.md` (generated from it, included into
the decoding page). Nothing is hand-typed into the docs.
"""

import argparse
import itertools
import json
import sys
import time
from pathlib import Path

import pandas as pd
import torch

from torchlingo.data_processing.vocab import SentencePieceVocab
from torchlingo.inference import beam_search_decode, greedy_decode
from torchlingo.models import SimpleTransformer

DEFAULT_OUT = Path("docs/docs/_generated")
PRETRAINED = Path("data/pretrained")

# Sweeps. Beam sizes span the range a student would plausibly try, including
# 1 (which must reproduce greedy, and is the control that proves the harness is
# wired correctly). Alphas span no normalization through aggressive.
BEAM_SIZES = (1, 2, 3, 5, 10)
ALPHAS = (0.0, 0.3, 0.6, 1.0, 1.5)

# Fixed values for the other knob while one is swept.
FIXED_ALPHA = 0.6
FIXED_BEAM = 5


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
        tuple: The model, the vocabulary, and the checkpoint dict.
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
    return model, vocab, checkpoint


def translate_all(
    sentences: list[str],
    model,
    vocab,
    device: torch.device,
    max_len: int,
    beam_size: int | None,
    alpha: float,
) -> list[str]:
    """Translate every sentence with one decoding configuration.

    Args:
        sentences (list[str]): Source sentences.
        model: The translation model.
        vocab: Shared source and target vocabulary.
        device (torch.device): Where to run.
        max_len (int): Decoding length cap.
        beam_size (int | None): Beam width, or None for greedy.
        alpha (float): Length-normalization strength; ignored when greedy.

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
                    model, src, beam_size=beam_size, max_len=max_len, alpha=alpha
                )
        out.append(vocab.decode(tokens, skip_special_tokens=True))
    return out


def score(
    hypotheses: list[str], references: list[str], greedy_output: list[str]
) -> dict:
    """Summarize one configuration's output.

    Args:
        hypotheses (list[str]): This configuration's translations.
        references (list[str]): Reference translations.
        greedy_output (list[str]): Greedy's translations, for the agreement rate.

    Returns:
        dict: BLEU, mean output length in whitespace tokens, and the share of
            sentences identical to greedy.
    """
    from sacrebleu.metrics import BLEU

    bleu = BLEU()
    same = sum(h == g for h, g in zip(hypotheses, greedy_output))
    return {
        "bleu": round(bleu.corpus_score(hypotheses, [references]).score, 2),
        # Recorded per measurement rather than once per run, so a row lifted
        # out of this JSON still carries the settings that produced it.
        "bleu_signature": str(bleu.get_signature()),
        "mean_length": round(
            sum(len(h.split()) for h in hypotheses) / len(hypotheses), 2
        ),
        "same_as_greedy": round(same / len(hypotheses), 4),
    }


def sweep_one_seed(
    test: pd.DataFrame,
    n_sentences: int,
    max_len: int,
    seed: int,
    model,
    vocab,
    device: torch.device,
) -> dict:
    """Run both sweeps on the subset drawn by one seed.

    Args:
        test (pd.DataFrame): The full held-out set.
        n_sentences (int): How many held-out sentences to decode per setting.
        max_len (int): Decoding length cap.
        seed (int): Sampling seed for the held-out subset.
        model: The translation model.
        vocab: Shared source and target vocabulary.
        device (torch.device): Where to run.

    Returns:
        dict: The greedy baseline, both sweeps, and this subset's reference length.
    """
    subset = test.sample(n_sentences, random_state=seed)
    sources = list(subset["src"])
    references = list(subset["tgt"])

    started = time.perf_counter()
    greedy_output = translate_all(
        sources, model, vocab, device, max_len, None, FIXED_ALPHA
    )
    greedy_row = score(greedy_output, references, greedy_output)
    greedy_row["seconds"] = round(time.perf_counter() - started, 1)

    by_beam = {}
    for beam_size in BEAM_SIZES:
        started = time.perf_counter()
        hyps = translate_all(
            sources, model, vocab, device, max_len, beam_size, FIXED_ALPHA
        )
        row = score(hyps, references, greedy_output)
        row["seconds"] = round(time.perf_counter() - started, 1)
        by_beam[str(beam_size)] = row
        print(
            f"  seed {seed} beam={beam_size:<3} BLEU={row['bleu']:<6} "
            f"len={row['mean_length']:<6} same as greedy={row['same_as_greedy']:.0%}",
            flush=True,
        )

    by_alpha = {}
    for alpha in ALPHAS:
        hyps = translate_all(sources, model, vocab, device, max_len, FIXED_BEAM, alpha)
        by_alpha[str(alpha)] = score(hyps, references, greedy_output)
        print(
            f"  seed {seed} alpha={alpha:<4} BLEU={by_alpha[str(alpha)]['bleu']:<6} "
            f"len={by_alpha[str(alpha)]['mean_length']}",
            flush=True,
        )

    return {
        "greedy": greedy_row,
        "by_beam_size": by_beam,
        "by_alpha": by_alpha,
        "reference_mean_length": round(
            sum(len(r.split()) for r in references) / len(references), 2
        ),
    }


def collect_signatures(results) -> list[str]:
    """Every distinct sacreBLEU signature anywhere in the results.

    Walks the whole structure rather than reading one known location, because
    the signature is recorded per measurement: that way a row lifted out of the
    JSON still carries the settings that produced it, and a run that somehow
    mixed settings is detectable rather than silently averaged.

    Args:
        results: Any nested combination of dicts and lists.

    Returns:
        list[str]: Sorted distinct signatures; empty if none were recorded.
    """
    found: set[str] = set()

    def walk(node) -> None:
        if isinstance(node, dict):
            for key, value in node.items():
                if key == "bleu_signature" and isinstance(value, str):
                    found.add(value)
                else:
                    walk(value)
        elif isinstance(node, list):
            for item in node:
                walk(item)

    walk(results)
    return sorted(found)


def aggregate(values: list[float]) -> dict:
    """Reduce a metric across seeds to a mean and a standard error.

    The standard error is what decides whether a difference between two
    configurations is worth teaching. Without it, a one-point BLEU gap between
    adjacent beam sizes looks like a finding when it may be the luck of which
    sentences were sampled.

    Args:
        values (list[float]): One value per seed.

    Returns:
        dict: ``mean``, ``stderr`` and the raw ``values``.
    """
    n = len(values)
    mean = sum(values) / n
    if n < 2:
        return {"mean": round(mean, 2), "stderr": None, "values": values}
    variance = sum((v - mean) ** 2 for v in values) / (n - 1)
    return {
        "mean": round(mean, 2),
        "stderr": round((variance / n) ** 0.5, 2),
        "values": values,
    }


def paired_delta(per_seed: dict, a_path: list, b_path: list, metric: str) -> dict:
    """Compare two configurations on the seeds they both ran.

    Every configuration decodes the *same* subsets, so the comparison is paired
    and the per-seed difference cancels the subset-level variance. That matters
    here: the unpaired standard errors are dominated by which sentences were
    sampled, and comparing them would hide differences that are real.

    Args:
        per_seed (dict): Per-seed results.
        a_path (list): Key path to the baseline configuration.
        b_path (list): Key path to the configuration being compared.
        metric (str): Metric to difference.

    Returns:
        dict: Mean difference, its standard error, and ``distinguishable``,
            true when |t| clears the two-sided 5% point for this many seeds.
    """
    deltas = []
    for run in per_seed.values():
        values = []
        for path in (a_path, b_path):
            node = run
            for key in path:
                node = node[key]
            values.append(node[metric])
        deltas.append(values[1] - values[0])

    n = len(deltas)
    mean = sum(deltas) / n
    if n < 2:
        return {"delta": round(mean, 3), "stderr": None, "distinguishable": None}
    variance = sum((d - mean) ** 2 for d in deltas) / (n - 1)
    stderr = (variance / n) ** 0.5
    # Two-sided 5% points of Student's t. Beyond the table, 1.96 is close
    # enough and errs toward calling things distinguishable, so the table
    # stops where the seeds stop.
    critical = {2: 12.71, 3: 4.30, 4: 3.18, 5: 2.78, 6: 2.57, 7: 2.45}.get(n, 1.96)
    return {
        "delta": round(mean, 3),
        "stderr": round(stderr, 3),
        "distinguishable": bool(stderr > 0 and abs(mean / stderr) >= critical),
    }


def compare_all(per_seed: dict) -> dict:
    """Build the paired comparisons the docs actually make claims about.

    Args:
        per_seed (dict): Per-seed results.

    Returns:
        dict: Named comparisons, each a :func:`paired_delta` result.
    """
    comparisons = {}
    for beam in BEAM_SIZES[1:]:
        comparisons[f"greedy_to_beam_{beam}"] = paired_delta(
            per_seed, ["greedy"], ["by_beam_size", str(beam)], "bleu"
        )
    widths = list(BEAM_SIZES[1:])
    for smaller, larger in itertools.pairwise(widths):
        comparisons[f"beam_{smaller}_to_beam_{larger}"] = paired_delta(
            per_seed,
            ["by_beam_size", str(smaller)],
            ["by_beam_size", str(larger)],
            "bleu",
        )
    comparisons[f"beam_{widths[0]}_to_beam_{widths[-1]}"] = paired_delta(
        per_seed,
        ["by_beam_size", str(widths[0])],
        ["by_beam_size", str(widths[-1])],
        "bleu",
    )
    for alpha in ALPHAS:
        if alpha != FIXED_ALPHA:
            comparisons[f"alpha_{FIXED_ALPHA}_to_{alpha}"] = paired_delta(
                per_seed,
                ["by_alpha", str(FIXED_ALPHA)],
                ["by_alpha", str(alpha)],
                "bleu",
            )
    return comparisons


def sweep(
    n_sentences: int, max_len: int, seeds: list[int], device: torch.device
) -> dict:
    """Run both sweeps across several seeds and aggregate.

    Each seed draws a different subset of the held-out set. Decoding itself is
    deterministic, so the subset is the only source of run-to-run variation,
    which is what makes a standard error across seeds meaningful.

    Args:
        n_sentences (int): How many held-out sentences to decode per setting.
        max_len (int): Decoding length cap.
        seeds (list[int]): Sampling seeds.
        device (torch.device): Where to run.

    Returns:
        dict: Settings, per-seed results, and the aggregated tables.
    """
    model, vocab, checkpoint = load_model(device)
    test = pd.read_csv(
        PRETRAINED / "test.tsv", sep="\t", dtype=str, keep_default_na=False
    )

    per_seed = {}
    for seed in seeds:
        print(f"seed {seed} ...", flush=True)
        per_seed[str(seed)] = sweep_one_seed(
            test, n_sentences, max_len, seed, model, vocab, device
        )

    return summarize(
        per_seed,
        {
            "n_sentences": n_sentences,
            "max_len": max_len,
            "seeds": seeds,
            "fixed_alpha": FIXED_ALPHA,
            "fixed_beam": FIXED_BEAM,
            "torch": torch.__version__,
            "device": str(device),
            "train_pairs": checkpoint["train_pairs"],
        },
    )


def summarize(per_seed: dict, settings: dict) -> dict:
    """Aggregate per-seed results into the tables the docs quote.

    Split out from :func:`sweep` so the aggregation and the paired comparisons
    can be recomputed from a previous run's JSON without decoding again. The
    sweep takes the better part of an hour; re-rendering should not.

    Args:
        per_seed (dict): Per-seed results.
        settings (dict): Run settings to record alongside them.

    Returns:
        dict: Settings, aggregates, paired comparisons, and the per-seed data.
    """

    def across(path: list, metric: str) -> dict:
        """Collect one metric across seeds, walking `path` into each seed's dict."""
        values = []
        for run in per_seed.values():
            node = run
            for key in path:
                node = node[key]
            values.append(node[metric])
        return aggregate(values)

    greedy = {m: across(["greedy"], m) for m in ("bleu", "mean_length")}
    greedy["seconds"] = across(["greedy"], "seconds")["mean"]

    by_beam = {}
    for beam_size in BEAM_SIZES:
        key = str(beam_size)
        by_beam[key] = {
            m: across(["by_beam_size", key], m)
            for m in ("bleu", "mean_length", "same_as_greedy")
        }
        by_beam[key]["seconds"] = across(["by_beam_size", key], "seconds")["mean"]

    by_alpha = {
        str(a): {m: across(["by_alpha", str(a)], m) for m in ("bleu", "mean_length")}
        for a in ALPHAS
    }

    return {
        "settings": {
            **settings,
            "reference_mean_length": round(
                sum(r["reference_mean_length"] for r in per_seed.values())
                / len(per_seed),
                2,
            ),
        },
        "greedy": greedy,
        "by_beam_size": by_beam,
        "by_alpha": by_alpha,
        "paired": compare_all(per_seed),
        "per_seed": per_seed,
    }


def render_markdown(results: dict) -> str:
    """Generate the docs tables from the measurements.

    Args:
        results (dict): Output of :func:`sweep`.

    Returns:
        str: Markdown for inclusion into the decoding page.
    """
    s = results["settings"]
    greedy = results["greedy"]

    def pm(stat: dict) -> str:
        """Format a metric as mean ± standard error."""
        if stat["stderr"] is None:
            return f"{stat['mean']}"
        return f"{stat['mean']} ± {stat['stderr']}"

    def delta(stat: dict) -> str:
        """Format a paired difference, which has no error bar on one seed.

        A single seed gives one paired difference, and the variance of one
        number is undefined, so ``paired_delta`` reports ``stderr: None``. This
        used to be formatted unconditionally and crashed the whole run --
        including ``--seeds 1``, which is what the exercise in
        ``concepts/decoding.md`` tells a reader to run.
        """
        if stat["stderr"] is None:
            return f"{stat['delta']:+.2f} (one seed, no error bar)"
        return f"{stat['delta']:+.2f} ± {stat['stderr']:.2f}"

    beam_rows = "\n".join(
        f"| {k} | {pm(r['bleu'])} | {pm(r['mean_length'])} | "
        f"{r['same_as_greedy']['mean']:.0%} | {r['seconds']:.0f} |"
        for k, r in results["by_beam_size"].items()
    )
    alpha_rows = "\n".join(
        f"| {k} | {pm(r['bleu'])} | {pm(r['mean_length'])} |"
        for k, r in results["by_alpha"].items()
    )

    def label(key: str) -> str:
        """Turn a comparison key into something readable in a table."""
        return (
            key.replace("_to_", " → ")
            .replace("greedy", "greedy")
            .replace("beam_", "beam ")
            .replace("alpha_", "alpha ")
            .replace("_", " ")
        )

    paired_rows = "\n".join(
        f"| {label(k)} | {delta(v)} | "
        f"{'**yes**' if v['distinguishable'] else 'n/a' if v['distinguishable'] is None else 'no'} |"
        for k, v in results["paired"].items()
    )
    n_seeds = len(s["seeds"])

    # Every measurement records its own signature, so they should all agree.
    # If they ever do not, the table is mixing numbers that are not comparable,
    # and saying so is far more useful than quietly printing the first one.
    signatures = collect_signatures(results)
    if not signatures:
        bleu_signature = "not recorded"
    elif len(signatures) == 1:
        bleu_signature = signatures[0]
    else:
        bleu_signature = (
            "MIXED SETTINGS -- these numbers are not comparable:\n"
            + "\n".join(signatures)
        )

    return f"""<!-- Generated by scripts/sweep_decoding.py. Do not edit by hand. -->

Measured on {s["n_sentences"]} held-out sentences per run, across {n_seeds} runs
(seeds {", ".join(str(x) for x in s["seeds"])}), `max_len={s["max_len"]}`, torch
{s["torch"]}, {s["device"]}. The model is the tutorial 5 checkpoint, trained on
{s["train_pairs"]:,} real sentence pairs. The held-out references average
{s["reference_mean_length"]} whitespace tokens.

Decoding is deterministic, so the only thing that varies between runs is which
sentences are sampled. Figures are mean ± standard error across those runs.
**Read the error bars before reading the means:** several of the gaps below are
smaller than the noise, and that is itself the finding.

**Beam size**, at `alpha={s["fixed_alpha"]}`:

| `beam_size` | BLEU | mean length | same output as greedy | seconds |
|---|---|---|---|---|
| greedy | {pm(greedy["bleu"])} | {pm(greedy["mean_length"])} | 100% | {greedy["seconds"]:.0f} |
{beam_rows}

**Length normalization**, at `beam_size={s["fixed_beam"]}`:

| `alpha` | BLEU | mean length |
|---|---|---|
{alpha_rows}

Every configuration decodes the **same** subsets, so the comparisons are paired
and the per-run difference cancels the sampling variance that dominates the
error bars above. That is a more sensitive test than comparing the columns by
eye, and it is what the claims on this page rest on:

| Comparison | BLEU change | Distinguishable? |
|---|---|---|
{paired_rows}

Reproduce with `python scripts/sweep_decoding.py`. BLEU and lengths are
deterministic given the sampled subset; seconds are machine-dependent.

Every BLEU number above was produced with:

```
{bleu_signature}
```

That is the sacreBLEU signature. BLEU is sensitive enough to tokenization that
a score without one is not comparable to anyone else's, so it is recorded here
rather than left implicit.
"""


def main() -> int:
    """Run the sweep and write the JSON and the generated tables."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--sentences", type=int, default=200)
    parser.add_argument("--max-len", type=int, default=60)
    parser.add_argument(
        "--seeds",
        type=lambda s: [int(x) for x in s.split(",")],
        default=[1, 2, 3, 4, 5],
        help="Comma-separated sampling seeds. Each draws a different held-out "
        "subset; the spread across them is the standard error.",
    )
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument(
        "--rerender",
        action="store_true",
        help="Recompute the tables from the existing JSON instead of decoding "
        "again. Use after changing how results are aggregated or rendered.",
    )
    args = parser.parse_args()

    if args.rerender:
        existing = json.loads((args.out_dir / "decoding_sweep.json").read_text())
        results = summarize(existing["per_seed"], existing["settings"])
        args.out_dir.mkdir(parents=True, exist_ok=True)
        (args.out_dir / "decoding_sweep.json").write_text(
            json.dumps(results, indent=2) + "\n"
        )
        (args.out_dir / "decoding_sweep.md").write_text(render_markdown(results))
        print(f"re-rendered {args.out_dir}/decoding_sweep.json and decoding_sweep.md")
        return 0

    if not is_available(PRETRAINED / "model.pt"):
        print(
            "data/pretrained/model.pt is absent or an unfetched Git LFS pointer; "
            "skipping. Run `git lfs pull` to fetch it.",
            file=sys.stderr,
        )
        return 0

    device = torch.device("cpu")
    results = sweep(args.sentences, args.max_len, args.seeds, device)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "decoding_sweep.json").write_text(
        json.dumps(results, indent=2) + "\n"
    )
    (args.out_dir / "decoding_sweep.md").write_text(render_markdown(results))
    print(f"\nwrote {args.out_dir}/decoding_sweep.json and decoding_sweep.md")
    return 0


if __name__ == "__main__":
    sys.exit(main())
