"""Tests for recovering attention over a sequence the model generated.

Both decoders compute cross-attention at every step and discard it. Rather than
thread weights through the search, `attention_for_sequence` re-runs the finished
sequence in one teacher-forced pass.

That design rests on a claim: **the second pass is exact, not an
approximation.** The decoder is causally masked, so the state at target
position *t* depends only on tokens up to *t*, and re-running over the whole
sequence reproduces each row as the incremental decode computed it. If that
claim were wrong, every attention picture in this library would be of a
computation nobody performed. `ExactnessTests` is the test that matters here;
the rest is shape and plumbing.
"""

import unittest

import torch

from torchlingo.inference import (
    attention_for_sequence,
    beam_search_decode,
    greedy_decode,
)
from torchlingo.models import SimpleSeq2SeqLSTM, SimpleTransformer

SRC = torch.tensor([[2, 5, 9, 11, 3]])


def _transformer(**kwargs) -> SimpleTransformer:
    torch.manual_seed(0)
    model = SimpleTransformer(
        src_vocab_size=30,
        tgt_vocab_size=30,
        d_model=16,
        n_heads=2,
        num_encoder_layers=1,
        num_decoder_layers=1,
        d_ff=32,
        **kwargs,
    )
    model.eval()
    return model


def _lstm(attention: bool = True) -> SimpleSeq2SeqLSTM:
    torch.manual_seed(0)
    model = SimpleSeq2SeqLSTM(
        src_vocab_size=30,
        tgt_vocab_size=30,
        emb_dim=16,
        hidden_dim=16,
        num_layers=1,
        attention=attention,
    )
    model.eval()
    return model


class ExactnessTests(unittest.TestCase):
    """The claim the whole approach rests on."""

    def test_one_pass_matches_step_by_step(self):
        """Re-running must reproduce what incremental decoding computed.

        Built by hand rather than by calling the decoder: feed the model
        progressively longer prefixes, keep the last attention row each time --
        which is the row for the token being generated at that step -- and
        compare the stack against a single pass over the whole sequence.
        """
        model = _transformer(dropout=0.0)
        sequence = [2, 7, 8, 12, 3]

        rows = []
        with torch.no_grad():
            for step in range(1, len(sequence)):
                _, weights = model(
                    SRC, torch.tensor([sequence[:step]]), return_attention=True
                )
                rows.append(weights[0, -1])
        incremental = torch.stack(rows)

        one_pass = attention_for_sequence(model, SRC, sequence)

        self.assertEqual(incremental.shape, one_pass.shape)
        self.assertTrue(
            torch.allclose(incremental, one_pass, atol=1e-6),
            f"max difference {(incremental - one_pass).abs().max().item():.2e}",
        )

    def test_decoding_is_unchanged_by_asking_for_attention(self):
        """The tokens must not move. Otherwise the picture is of another run."""
        model = _transformer()
        plain = greedy_decode(model, SRC, max_len=6)
        annotated, _ = greedy_decode(model, SRC, max_len=6, return_attention=True)
        self.assertEqual(plain, annotated)

        plain_beam = beam_search_decode(model, SRC, beam_size=3, max_len=6)
        annotated_beam, _ = beam_search_decode(
            model, SRC, beam_size=3, max_len=6, return_attention=True
        )
        self.assertEqual(plain_beam, annotated_beam)


class ShapeAndDistributionTests(unittest.TestCase):
    def test_one_row_per_generated_token(self):
        model = _transformer()
        tokens = greedy_decode(model, SRC, max_len=6)[0]
        weights = attention_for_sequence(model, SRC, tokens)
        self.assertEqual(weights.shape, (len(tokens) - 1, SRC.size(1)))

    def test_rows_are_distributions_over_source_positions(self):
        model = _transformer()
        tokens, weights = beam_search_decode(
            model, SRC, beam_size=3, max_len=6, return_attention=True
        )
        rows = weights.sum(-1)
        self.assertTrue(torch.allclose(rows, torch.ones_like(rows), atol=1e-5))
        self.assertTrue((weights >= 0).all())

    def test_accepts_unbatched_source(self):
        model = _transformer()
        tokens = greedy_decode(model, SRC, max_len=6)[0]
        batched = attention_for_sequence(model, SRC, tokens)
        flat = attention_for_sequence(model, SRC[0], tokens)
        self.assertTrue(torch.allclose(batched, flat, atol=1e-6))

    def test_works_on_the_lstm_too(self):
        """One call shape, either architecture."""
        model = _lstm(attention=True)
        tokens, weights = beam_search_decode(
            model, SRC, beam_size=3, max_len=6, return_attention=True
        )
        self.assertEqual(weights.shape, (len(tokens) - 1, SRC.size(1)))


class GreedyBatchTests(unittest.TestCase):
    """Greedy decodes a batch, and the sequences come out ragged."""

    def test_returns_one_tensor_per_sentence(self):
        model = _transformer()
        src = torch.tensor([[2, 5, 9, 3], [2, 6, 7, 3]])
        tokens, weights = greedy_decode(model, src, max_len=6, return_attention=True)
        self.assertEqual(len(weights), len(tokens))
        for sequence, weight in zip(tokens, weights):
            self.assertEqual(weight.shape, (len(sequence) - 1, src.size(1)))

    def test_default_return_is_unchanged(self):
        """Existing callers must not find a tuple where a list was."""
        model = _transformer()
        out = greedy_decode(model, SRC, max_len=6)
        self.assertIsInstance(out, list)
        self.assertIsInstance(out[0], list)


class RefusalTests(unittest.TestCase):
    """What it does when it cannot answer, rather than guessing."""

    def test_lstm_without_attention_is_refused(self):
        """An LSTM built with attention=False computes no cross-attention.

        Refusing beats returning zeros, which would plot as a picture of a
        model attending uniformly to nothing.
        """
        model = _lstm(attention=False)
        tokens = greedy_decode(model, SRC, max_len=6)[0]
        with self.assertRaises(ValueError) as caught:
            attention_for_sequence(model, SRC, tokens)
        self.assertIn("attention", str(caught.exception).lower())

    def test_batched_source_is_refused(self):
        model = _transformer()
        src = torch.tensor([[2, 5, 3], [2, 6, 3]])
        with self.assertRaises(ValueError) as caught:
            attention_for_sequence(model, src, [2, 7, 3])
        self.assertIn("one sentence at a time", str(caught.exception))

    def test_sequence_too_short_is_refused(self):
        model = _transformer()
        with self.assertRaises(ValueError) as caught:
            attention_for_sequence(model, SRC, [2])
        self.assertIn("at least one generated token", str(caught.exception))

    def test_training_mode_is_restored(self):
        """The helper calls eval() and must put the model back as it found it."""
        model = _transformer()
        model.train()
        attention_for_sequence(model, SRC, [2, 7, 8])
        self.assertTrue(model.training, "model was left in eval mode")


if __name__ == "__main__":
    unittest.main()
