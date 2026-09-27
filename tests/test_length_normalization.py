"""Where length normalization can and cannot change the result.

`beam_search_decode` applies `_rank_key` at two sites: pruning each step, and
choosing among finished hypotheses at the end. Only the second can be affected
by ``alpha``, because every candidate within a step has the same length and the
length divisor is therefore a shared constant.

That invariant is not obvious from the code and is easy to break. It holds only
because a hypothesis that emits EOS is moved out of ``beams`` and into
``completed``; if a finished beam were left in the live set, it would stop
growing while its siblings kept extending, candidates of different lengths would
meet in the same sort, and ``alpha`` would silently start steering the search.
The docstrings on `_rank_key` and `beam_search_decode` state the invariant, so
these tests exist to keep those statements true.
"""

import unittest

import torch

from torchlingo.config import Config
from torchlingo.inference import _rank_key, beam_search_decode

ALPHAS = (0.0, 0.3, 0.6, 1.0, 1.5, 5.0)


class HistorySensitiveDecoder(torch.nn.Module):
    """A decoder whose logits depend on the tokens so far, not just position.

    A position-only stub cannot exercise beam search: every beam would score
    identically and the search would never have a real choice to make. This is
    the same reason `tests/test_decoding_equivalence.py` carries its own
    history-sensitive fixture. Defined locally rather than imported from there,
    because `unittest discover` imports test modules top level and cross-module
    test imports have broken in this repo before.

    It exposes `encode`/`decode` because that is the interface
    `beam_search_decode` dispatches on for the Transformer path.
    """

    def __init__(self, vocab_size=10, eos_idx=2):
        super().__init__()
        self.vocab_size = vocab_size
        self.eos_idx = eos_idx
        self._weight = torch.nn.Parameter(torch.zeros(1))

    def _logits(self, tgt):
        batch, steps = tgt.shape
        logits = torch.zeros(batch, steps, self.vocab_size)
        for b in range(batch):
            for t in range(steps):
                history = int(tgt[b, : t + 1].sum().item())
                for v in range(self.vocab_size):
                    # Deterministic, history-dependent, and not monotone in v.
                    logits[b, t, v] = ((history * 7 + v * 13) % 11) / 3.0
                # Make EOS reachable but not dominant, so hypotheses finish at
                # a spread of different lengths -- which is what gives final
                # selection something for alpha to act on.
                logits[b, t, self.eos_idx] += 0.4 * t
        return logits

    def encode(self, src, src_key_padding_mask=None):
        return src

    def decode(self, tgt, memory, **kwargs):
        return self._logits(tgt)

    def forward(self, src, tgt):
        return self._logits(tgt)


def _config():
    return Config(pad_idx=0, sos_idx=1, eos_idx=2)


class RankKeyTests(unittest.TestCase):
    def test_equal_length_ordering_is_the_same_for_every_alpha(self):
        # The heart of it: a shared positive divisor cannot reorder.
        candidates = [
            ([1, 5, 7, 3], -4.0),
            ([1, 5, 7, 9], -2.5),
            ([1, 6, 2, 8], -9.25),
            ([1, 6, 2, 4], -0.5),
        ]
        baseline = None
        for alpha in ALPHAS:
            order = [
                t for t, _ in sorted(candidates, key=lambda i: _rank_key(*i, alpha))
            ]
            if baseline is None:
                baseline = order
            self.assertEqual(order, baseline, f"alpha={alpha} reordered equal lengths")

    def test_differing_lengths_can_be_reordered(self):
        # The contrast, so the test above is not vacuously true: a large enough
        # alpha does prefer the longer hypothesis with the worse raw score.
        short = ([1, 2, 3], -6.0)
        long = ([1, 2, 3, 4, 5, 6, 7], -10.0)
        at_zero = min([short, long], key=lambda i: _rank_key(*i, 0.0))
        at_high = min([short, long], key=lambda i: _rank_key(*i, 1.5))
        self.assertEqual(len(at_zero[0]), 3)
        self.assertEqual(len(at_high[0]), 7)

    def test_alpha_zero_ranks_by_raw_score(self):
        candidates = [([1, 4, 4], -3.0), ([1, 5, 5], -1.0), ([1, 6, 6], -2.0)]
        order = [s for _, s in sorted(candidates, key=lambda i: _rank_key(*i, 0.0))]
        self.assertEqual(order, [-1.0, -2.0, -3.0])


class PruningInvarianceTests(unittest.TestCase):
    """The invariant the docstrings rely on, checked on a real search."""

    def setUp(self):
        self.model = HistorySensitiveDecoder()
        self.src = torch.tensor([[1, 4, 5, 2]])
        self.config = _config()

    def _trace(self, alpha):
        trace = []
        beam_search_decode(
            self.model,
            self.src,
            beam_size=3,
            max_len=12,
            alpha=alpha,
            config=self.config,
            trace=trace,
        )
        return trace

    def test_every_candidate_in_a_step_has_the_same_length(self):
        for step in self._trace(0.6):
            lengths = {len(c.tokens) for c in step.candidates}
            self.assertEqual(
                len(lengths),
                1,
                f"step {step.step} mixed lengths {sorted(lengths)}; alpha would "
                "now steer pruning, which the docstrings say it cannot",
            )

    def test_which_candidates_survive_does_not_depend_on_alpha(self):
        baseline = [
            [c.tokens for c in step.candidates if c.kept] for step in self._trace(0.0)
        ]
        for alpha in ALPHAS:
            survivors = [
                [c.tokens for c in step.candidates if c.kept]
                for step in self._trace(alpha)
            ]
            self.assertEqual(
                survivors, baseline, f"alpha={alpha} changed which beams survived"
            )

    def test_the_search_explores_more_than_one_length_overall(self):
        # Guards the tests above from passing because the search is degenerate.
        steps = self._trace(0.6)
        self.assertGreater(len(steps), 1)
        lengths = {len(c.tokens) for step in steps for c in step.candidates}
        self.assertGreater(len(lengths), 1)


if __name__ == "__main__":
    unittest.main()
