"""Keep the decoding page's numbers honest.

`docs/docs/concepts/decoding.md` quotes call counts and ratios, and the table it
shows is generated from `docs/docs/_generated/decode_bench.json`. Nothing would
notice if the code changed and the committed numbers went stale, which is the
same failure that produced #14, #16, #20 and #30.

These tests re-run the benchmark at the documented settings and compare against
the committed JSON. Call counts are deterministic for a fixed seed, so they can
be asserted exactly. Wall clock is not, and is deliberately not asserted.
"""

import json
import sys
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
BENCH_JSON = REPO / "docs" / "docs" / "_generated" / "decode_bench.json"

sys.path.insert(0, str(REPO / "scripts"))


def _load_committed() -> dict:
    return json.loads(BENCH_JSON.read_text())


@unittest.skipUnless(BENCH_JSON.exists(), "benchmark output not generated")
class TestDecodeBenchmarkMatchesDocs(unittest.TestCase):
    """The committed measurements must still describe the current code."""

    @classmethod
    def setUpClass(cls):
        from bench_decode import benchmark

        cls.committed = _load_committed()
        settings = cls.committed["settings"]
        cls.fresh = benchmark(
            n_sentences=settings["n_sentences"],
            src_len=settings["src_len"],
            max_len=settings["max_len"],
            beam_size=settings["beam_size"],
            seed=settings["seed"],
        )

    def test_call_counts_are_unchanged(self):
        """Deterministic for a fixed seed, so any drift is a real change."""
        for variant in ("greedy", "beam_reference", "beam_fast"):
            with self.subTest(variant=variant):
                self.assertEqual(
                    self.fresh[variant]["calls"],
                    self.committed[variant]["calls"],
                    f"{variant} call count changed; re-run scripts/bench_decode.py",
                )

    def test_positions_forwarded_are_unchanged(self):
        """Total work forwarded through the model should not drift silently."""
        for variant in ("greedy", "beam_reference", "beam_fast"):
            with self.subTest(variant=variant):
                self.assertEqual(
                    self.fresh[variant]["positions"],
                    self.committed[variant]["positions"],
                )

    def test_documented_ratios_hold(self):
        """The ratios the prose leans on must still be true."""
        for key in (
            "reference_beam_calls_vs_greedy",
            "reference_beam_positions_vs_greedy",
            "fast_beam_call_reduction",
        ):
            with self.subTest(ratio=key):
                self.assertEqual(
                    self.fresh["ratios"][key], self.committed["ratios"][key]
                )

    def test_fast_beam_does_the_same_work_in_fewer_calls(self):
        """The whole claim of the fast path, stated as an invariant.

        Same positions forwarded, strictly fewer calls. If this ever fails the
        two implementations have genuinely diverged.
        """
        reference, fast = self.fresh["beam_reference"], self.fresh["beam_fast"]
        self.assertEqual(reference["positions"], fast["positions"])
        self.assertLess(fast["calls"], reference["calls"])

    def test_wall_clock_is_recorded_but_not_asserted(self):
        """Timings are present and positive; their values are machine-dependent."""
        for variant in ("greedy", "beam_reference", "beam_fast"):
            with self.subTest(variant=variant):
                self.assertGreater(self.fresh[variant]["ms_per_sentence"], 0)


if __name__ == "__main__":
    unittest.main()
