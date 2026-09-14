"""Measure what decoding actually costs, so the docs can quote reproducible numbers.

`docs/docs/concepts/decoding.md` makes specific quantitative claims: how many
more times beam search calls the model than greedy does, how much arithmetic
that represents, and what batching the beams buys back. Those numbers were
originally produced by throwaway instrumentation, which meant a reader had no
way to check them and no way to see them change.

This script is that instrumentation, kept. It reports two different things,
because they do not move together and the gap between them is the lesson:

- **Call counts.** How many times the model is invoked, with how many sequences
  each time, and how many token positions are forwarded in total. Deterministic
  for a fixed seed, so these belong in the docs.
- **Wall clock.** How long it takes. Machine-dependent and noisy, so it is
  reported but should be read as a ratio rather than an absolute.

The model is small, untrained and seeded. That is deliberate: the point is to
count model calls and forwarded positions, which depend on the search structure
rather than on translation quality, and a synthetic model makes the whole
measurement reproducible anywhere with no data or checkpoint.

Run:
    python scripts/bench_decode.py

Writes `docs/docs/_generated/decode_bench.json` (the single source of truth) and
`docs/docs/_generated/decode_bench.md` (a table generated from it, included into
the decoding page). Nothing is hand-typed into the docs.
"""

import argparse
import json
import time
from contextlib import contextmanager
from pathlib import Path

import torch

from torchlingo import inference, inference_fast
from torchlingo.config import Config
from torchlingo.models import SimpleTransformer

DEFAULT_OUT = Path("docs/docs/_generated")

# Small on purpose. The measurement counts model calls and forwarded positions,
# which depend on the search structure rather than on vocabulary size, and a
# small vocabulary keeps the run to a few seconds.
VOCAB_SIZE = 200


class DecodeCounter:
    """Record how the model is called during a decode.

    Attributes:
        calls (int): Number of ``decode()`` invocations.
        sequences (int): Total sequences summed across calls.
        positions (int): Total token positions forwarded, summed across calls.
    """

    def __init__(self) -> None:
        self.calls = 0
        self.sequences = 0
        self.positions = 0

    @property
    def sequences_per_call(self) -> float:
        """Mean batch size per model call."""
        return self.sequences / self.calls if self.calls else 0.0

    def as_dict(self) -> dict:
        """Return the counters as plain JSON-serializable values."""
        return {
            "calls": self.calls,
            "sequences_per_call": round(self.sequences_per_call, 2),
            "positions": self.positions,
        }


@contextmanager
def counting(model: SimpleTransformer):
    """Temporarily instrument ``model.decode`` to count invocations.

    Args:
        model (SimpleTransformer): Model whose decode method is wrapped.

    Yields:
        DecodeCounter: Live counters, populated as decoding proceeds.
    """
    counter = DecodeCounter()
    original = model.decode

    def wrapped(tgt, memory, *args, **kwargs):
        counter.calls += 1
        counter.sequences += tgt.size(0)
        counter.positions += tgt.size(0) * tgt.size(1)
        return original(tgt, memory, *args, **kwargs)

    model.decode = wrapped
    try:
        yield counter
    finally:
        model.decode = original


def build_model(cfg: Config, seed: int) -> SimpleTransformer:
    """Build a small seeded model. Untrained: we are counting calls, not quality."""
    torch.manual_seed(seed)
    return SimpleTransformer(
        src_vocab_size=VOCAB_SIZE,
        tgt_vocab_size=VOCAB_SIZE,
        d_model=64,
        n_heads=4,
        num_encoder_layers=2,
        num_decoder_layers=2,
        d_ff=128,
        dropout=0.0,
    ).eval()


def build_sources(
    cfg: Config, n_sentences: int, src_len: int, seed: int
) -> torch.Tensor:
    """Build a padded batch of synthetic source sentences."""
    generator = torch.Generator().manual_seed(seed)
    body = torch.randint(4, VOCAB_SIZE, (n_sentences, src_len - 2), generator=generator)
    sos = torch.full((n_sentences, 1), cfg.sos_idx)
    eos = torch.full((n_sentences, 1), cfg.eos_idx)
    return torch.cat([sos, body, eos], dim=1)


def time_it(fn) -> tuple[float, object]:
    """Run ``fn`` and return elapsed seconds alongside its result."""
    start = time.perf_counter()
    result = fn()
    return time.perf_counter() - start, result


def benchmark(n_sentences: int, src_len: int, max_len: int, beam_size: int, seed: int):
    """Measure greedy, reference beam and fast beam on the same inputs.

    Returns:
        dict: Every measurement, keyed by variant, plus the settings used.
    """
    cfg = Config()
    model = build_model(cfg, seed)
    src = build_sources(cfg, n_sentences, src_len, seed)

    results = {}

    with counting(model) as counter:
        elapsed, _ = time_it(
            lambda: inference.greedy_decode(model, src, max_len=max_len, config=cfg)
        )
    results["greedy"] = {
        **counter.as_dict(),
        "seconds": elapsed,
        "ms_per_sentence": 1000 * elapsed / n_sentences,
    }

    # Both beam implementations take one sentence at a time, so the per-sentence
    # loop is part of what is being measured, not an artifact of the harness.
    for label, decode in (
        ("beam_reference", inference.beam_search_decode),
        ("beam_fast", inference_fast.beam_search_decode),
    ):
        with counting(model) as counter:
            elapsed, _ = time_it(
                lambda d=decode: [
                    d(
                        model,
                        src[i : i + 1],
                        beam_size=beam_size,
                        max_len=max_len,
                        config=cfg,
                    )
                    for i in range(n_sentences)
                ]
            )
        results[label] = {
            **counter.as_dict(),
            "seconds": elapsed,
            "ms_per_sentence": 1000 * elapsed / n_sentences,
        }

    greedy, reference, fast = (
        results["greedy"],
        results["beam_reference"],
        results["beam_fast"],
    )
    results["ratios"] = {
        "reference_beam_calls_vs_greedy": round(
            reference["calls"] / greedy["calls"], 1
        ),
        "reference_beam_positions_vs_greedy": round(
            reference["positions"] / greedy["positions"], 1
        ),
        "fast_beam_call_reduction": round(reference["calls"] / fast["calls"], 1),
        "fast_beam_speedup": round(
            reference["ms_per_sentence"] / fast["ms_per_sentence"], 1
        ),
    }
    results["settings"] = {
        "n_sentences": n_sentences,
        "src_len": src_len,
        "max_len": max_len,
        "beam_size": beam_size,
        "seed": seed,
        "torch": torch.__version__,
        "device": "cpu",
    }
    return results


def render_markdown(results: dict) -> str:
    """Generate the docs table from the measurements.

    Wall-clock figures are labelled as machine-dependent rather than presented
    alongside the deterministic counts without comment.
    """
    s = results["settings"]
    r = results["ratios"]
    greedy, reference, fast = (
        results["greedy"],
        results["beam_reference"],
        results["beam_fast"],
    )

    return f"""<!-- Generated by scripts/bench_decode.py. Do not edit by hand. -->

Measured on {s["n_sentences"]} sentences, `max_len={s["max_len"]}`,
`beam_size={s["beam_size"]}`, seed {s["seed"]}, torch {s["torch"]}, CPU.

| | `decode()` calls | sequences per call | positions forwarded |
|---|---|---|---|
| greedy | {greedy["calls"]:,} | {greedy["sequences_per_call"]} | {greedy["positions"]:,} |
| beam (reference) | {reference["calls"]:,} | **{reference["sequences_per_call"]}** | {reference["positions"]:,} |
| ratio | **{r["reference_beam_calls_vs_greedy"]}×** | | {r["reference_beam_positions_vs_greedy"]}× |

Batching the beams closes most of the call gap:

| | `decode()` calls | sequences per call | ms/sentence |
|---|---|---|---|
| beam (reference) | {reference["calls"]:,} | {reference["sequences_per_call"]} | {reference["ms_per_sentence"]:.1f} |
| beam (fast) | {fast["calls"]:,} | {fast["sequences_per_call"]} | {fast["ms_per_sentence"]:.1f} |
| improvement | **{r["fast_beam_call_reduction"]}× fewer** | | **{r["fast_beam_speedup"]}× faster** |

Call counts are deterministic for a fixed seed. Wall clock is machine-dependent
and should be read as a ratio, not an absolute. Reproduce with
`python scripts/bench_decode.py`.
"""


def main() -> None:
    """Run the benchmark and write the JSON and the generated table."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--sentences", type=int, default=8)
    parser.add_argument("--src-len", type=int, default=12)
    parser.add_argument("--max-len", type=int, default=25)
    parser.add_argument("--beam-size", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    results = benchmark(
        args.sentences, args.src_len, args.max_len, args.beam_size, args.seed
    )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "decode_bench.json").write_text(
        json.dumps(results, indent=2) + "\n"
    )
    (args.out_dir / "decode_bench.md").write_text(render_markdown(results))

    width = max(len(k) for k in results if k not in ("ratios", "settings"))
    for name in ("greedy", "beam_reference", "beam_fast"):
        row = results[name]
        print(
            f"{name:{width}s}  calls={row['calls']:>6,}  "
            f"seq/call={row['sequences_per_call']:>5}  "
            f"positions={row['positions']:>8,}  "
            f"{row['ms_per_sentence']:>7.1f} ms/sentence"
        )
    print()
    for key, value in results["ratios"].items():
        print(f"{key:38s} {value}")
    print(f"\nwrote {args.out_dir}/decode_bench.json and decode_bench.md")


if __name__ == "__main__":
    main()
