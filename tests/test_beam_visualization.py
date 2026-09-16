"""Tests for the beam search trace and its renderers.

The trace exists to make pruning visible. These assert that it records what the
search actually did, and -- the property that matters most -- that asking for it
does not change the result.
"""

import unittest

import torch

from torchlingo.inference import BeamCandidate, BeamStep, beam_search_decode
from torchlingo.models import SimpleTransformer
from torchlingo.visualization import format_beam_search, plot_beam_search

ITOS = ["<pad>", "<unk>", "<s>", "</s>"] + [f"w{i}" for i in range(4, 24)]


def _model() -> SimpleTransformer:
    torch.manual_seed(3)
    return SimpleTransformer(
        src_vocab_size=24,
        tgt_vocab_size=24,
        d_model=32,
        n_heads=4,
        num_encoder_layers=1,
        num_decoder_layers=1,
        d_ff=32,
        dropout=0.0,
    ).eval()


def _src() -> torch.Tensor:
    return torch.tensor([[2, 9, 11, 3]])


class TestTraceDoesNotChangeResults(unittest.TestCase):
    """Tracing must be observation only."""

    def test_output_is_identical_with_and_without_a_trace(self):
        model, src = _model(), _src()
        without = beam_search_decode(model, src, beam_size=3, max_len=6)
        trace: list[BeamStep] = []
        with_trace = beam_search_decode(model, src, beam_size=3, max_len=6, trace=trace)
        self.assertEqual(without, with_trace)
        self.assertTrue(trace, "a trace was requested but nothing was recorded")

    def test_trace_is_optional(self):
        """Existing callers pass no trace and must be unaffected."""
        self.assertIsInstance(
            beam_search_decode(_model(), _src(), beam_size=2, max_len=4), list
        )


class TestTraceContents(unittest.TestCase):
    """The recorded steps must describe the search faithfully."""

    def setUp(self):
        self.trace: list[BeamStep] = []
        self.tokens = beam_search_decode(
            _model(), _src(), beam_size=3, max_len=6, trace=self.trace
        )

    def test_steps_are_numbered_in_order(self):
        self.assertEqual(
            [step.step for step in self.trace], list(range(len(self.trace)))
        )

    def test_candidates_are_ranked_best_first(self):
        """The display depends on the recorded order being the search's order."""
        for step in self.trace:
            scores = [candidate.normalized for candidate in step.candidates]
            self.assertEqual(scores, sorted(scores, reverse=True))

    def test_at_most_beam_size_candidates_are_kept(self):
        for step in self.trace:
            kept = [candidate for candidate in step.candidates if candidate.kept]
            self.assertLessEqual(len(kept), 3)

    def test_kept_candidates_are_the_top_ranked_ones(self):
        """kept must mean 'survived pruning', not something else."""
        for step in self.trace:
            flags = [candidate.kept for candidate in step.candidates]
            # All True values come first: no kept candidate ranks below a pruned one.
            self.assertEqual(flags, sorted(flags, reverse=True))

    def test_pruned_candidates_are_recorded_too(self):
        """Recording only survivors would hide the entire lesson."""
        pruned = sum(1 for step in self.trace for c in step.candidates if not c.kept)
        self.assertGreater(pruned, 0)

    def test_the_winner_appears_as_a_kept_candidate(self):
        """The returned sequence must be traceable through the recorded steps."""
        prefixes = {
            tuple(c.tokens) for step in self.trace for c in step.candidates if c.kept
        }
        self.assertIn(tuple(self.tokens[:2]), prefixes)


class TestFormatBeamSearch(unittest.TestCase):
    """The text renderer."""

    def setUp(self):
        self.trace: list[BeamStep] = []
        self.tokens = beam_search_decode(
            _model(), _src(), beam_size=3, max_len=6, trace=self.trace
        )

    def test_renders_a_line_per_shown_candidate(self):
        text = format_beam_search(self.trace, itos=ITOS, top=2, max_steps=2)
        lines = text.splitlines()
        self.assertEqual(sum(1 for line in lines if line.startswith("step ")), 2)

    def test_marks_the_winning_path(self):
        text = format_beam_search(self.trace, itos=ITOS, winner=self.tokens)
        self.assertIn(">", text)

    def test_marks_pruned_candidates(self):
        text = format_beam_search(self.trace, itos=ITOS, winner=self.tokens, top=8)
        self.assertIn("  . ", text)

    def test_uses_vocabulary_when_given(self):
        text = format_beam_search(self.trace, itos=ITOS, top=1, max_steps=1)
        self.assertIn("<s>", text)

    def test_falls_back_to_ids_without_a_vocabulary(self):
        text = format_beam_search(self.trace, top=1, max_steps=1)
        self.assertRegex(text, r"\d+")

    def test_reports_how_many_were_not_shown(self):
        text = format_beam_search(self.trace, itos=ITOS, top=1, max_steps=2)
        self.assertIn("more considered", text)

    def test_empty_trace_raises_with_guidance(self):
        with self.assertRaises(ValueError) as ctx:
            format_beam_search([])
        self.assertIn("trace=[]", str(ctx.exception))

    def test_handles_a_hand_built_trace(self):
        """The dataclasses are usable without running a decode."""
        step = BeamStep(0, [BeamCandidate([2, 7], -0.2, -0.2, True)])
        text = format_beam_search([step], itos=ITOS)
        self.assertIn("step 0", text)
        self.assertIn("-0.200", text)


class TestPlotBeamSearch(unittest.TestCase):
    """The matplotlib renderer."""

    def test_returns_labelled_axes(self):
        import matplotlib

        matplotlib.use("Agg")
        trace: list[BeamStep] = []
        tokens = beam_search_decode(
            _model(), _src(), beam_size=3, max_len=5, trace=trace
        )
        ax = plot_beam_search(trace, winner=tokens, title="beam search")
        self.assertEqual(ax.get_title(), "beam search")
        self.assertEqual(ax.get_xlabel(), "step")
        matplotlib.pyplot.close("all")

    def test_empty_trace_raises(self):
        with self.assertRaises(ValueError):
            plot_beam_search([])


if __name__ == "__main__":
    unittest.main()
