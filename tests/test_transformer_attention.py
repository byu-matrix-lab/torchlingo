"""Tests for surfacing the Transformer's cross-attention weights.

Until now `SimpleTransformer` could not return its attention at all, so every
attention lesson in this library used the LSTM — including on tutorials whose
model is a Transformer. These tests cover the mechanism that closes that gap.

The mechanism is unusual enough to deserve testing carefully. PyTorch hardcodes
`need_weights=False` inside `TransformerDecoderLayer._mha_block` so it can take
a fused attention kernel, which means the weights are never computed rather than
merely hidden. Getting them requires temporarily replacing each layer's
`multihead_attn.forward`, and the two things that can go wrong are that the
replacement is not undone, and that what comes back is not what inference
actually used.
"""

import unittest

import torch

from torchlingo.models import SimpleTransformer
from torchlingo.models.transformer_simple import capture_cross_attention


def _model(layers: int = 2, **kwargs) -> SimpleTransformer:
    """Build a small deterministic model in eval mode."""
    torch.manual_seed(0)
    model = SimpleTransformer(
        src_vocab_size=20,
        tgt_vocab_size=20,
        d_model=16,
        n_heads=2,
        num_encoder_layers=1,
        num_decoder_layers=layers,
        d_ff=32,
        **kwargs,
    )
    model.eval()
    return model


SRC = torch.tensor([[2, 5, 9, 11, 3]])
TGT = torch.tensor([[2, 7, 8]])


class ReturnAttentionTests(unittest.TestCase):
    """The opt-in argument, which mirrors the LSTM's."""

    def test_default_return_is_unchanged(self):
        """Existing callers must not see a tuple appear under them."""
        out = _model()(SRC, TGT)
        self.assertIsInstance(out, torch.Tensor)

    def test_opting_in_returns_logits_and_weights(self):
        logits, weights = _model()(SRC, TGT, return_attention=True)
        self.assertEqual(logits.shape, (1, TGT.size(1), 20))
        self.assertEqual(weights.shape, (1, TGT.size(1), SRC.size(1)))

    def test_weights_are_a_distribution_over_source_positions(self):
        """Every target position spreads one unit of attention over the source."""
        _, weights = _model()(SRC, TGT, return_attention=True)
        self.assertTrue(
            torch.allclose(weights.sum(-1), torch.ones(1, TGT.size(1)), atol=1e-5)
        )
        self.assertTrue((weights >= 0).all())

    def test_logits_are_identical_either_way(self):
        """Asking for attention must not change the answer.

        The capture forces `need_weights=True`, which takes a different code
        path inside PyTorch. If that path computed anything differently, every
        attention visualization would be of a model that is not the one being
        used.
        """
        model = _model()
        with torch.no_grad():
            plain = model(SRC, TGT)
            withattn, _ = model(SRC, TGT, return_attention=True)
        self.assertTrue(torch.allclose(plain, withattn, atol=1e-6))

    def test_batch_of_more_than_one(self):
        src = torch.tensor([[2, 5, 9, 3], [2, 6, 7, 3]])
        tgt = torch.tensor([[2, 7, 8], [2, 9, 10]])
        _, weights = _model()(src, tgt, return_attention=True)
        self.assertEqual(weights.shape, (2, 3, 4))


class CaptureContextManagerTests(unittest.TestCase):
    """The lower-level mechanism, including that it cleans up after itself."""

    def test_one_entry_per_decoder_layer(self):
        model = _model(layers=3)
        with (
            capture_cross_attention(model.transformer.decoder) as weights,
            torch.no_grad(),
        ):
            model(SRC, TGT)
        self.assertEqual(len(weights), 3)
        for layer_weights in weights:
            self.assertEqual(layer_weights.shape, (1, TGT.size(1), SRC.size(1)))

    def test_forward_is_restored_afterwards(self):
        """The wrapper must come off, or the model stays slow forever.

        Asserted on whether `forward` is an instance attribute rather than by
        comparing bound methods: attribute access creates a fresh bound method
        each time, so identity comparison would fail even on a correct restore.
        The invariant that matters is that the object is left as it was found,
        with `forward` resolving to the class method.
        """
        model = _model()
        layers = model.transformer.decoder.layers
        self.assertFalse(
            any("forward" in layer.multihead_attn.__dict__ for layer in layers)
        )

        with capture_cross_attention(model.transformer.decoder):
            self.assertTrue(
                all("forward" in layer.multihead_attn.__dict__ for layer in layers),
                "expected every layer to be instrumented inside the block",
            )

        self.assertFalse(
            any("forward" in layer.multihead_attn.__dict__ for layer in layers),
            "the wrapper was left in place",
        )

    def test_forward_is_restored_after_an_exception(self):
        """A raise inside the block must not leave the model instrumented."""
        model = _model()
        layers = model.transformer.decoder.layers

        with (
            self.assertRaises(RuntimeError),
            capture_cross_attention(model.transformer.decoder),
        ):
            raise RuntimeError("boom")

        self.assertFalse(
            any("forward" in layer.multihead_attn.__dict__ for layer in layers)
        )

    def test_the_model_still_works_after_capturing(self):
        """The restore has to leave a working model, not just a clean __dict__."""
        model = _model()
        with capture_cross_attention(model.transformer.decoder), torch.no_grad():
            during = model(SRC, TGT)
        with torch.no_grad():
            after = model(SRC, TGT)
        self.assertTrue(torch.allclose(during, after, atol=1e-6))

    def test_forward_returns_last_layer(self):
        """`return_attention=True` documents that it gives the last layer."""
        model = _model(layers=3)
        with (
            capture_cross_attention(model.transformer.decoder) as weights,
            torch.no_grad(),
        ):
            model(SRC, TGT)
        with torch.no_grad():
            _, from_forward = model(SRC, TGT, return_attention=True)
        self.assertTrue(torch.allclose(from_forward, weights[-1], atol=1e-6))


class DropoutTests(unittest.TestCase):
    """The trap worth knowing about, asserted so it stays known."""

    def test_training_mode_weights_are_not_a_distribution(self):
        """Attention dropout zeroes and rescales, so rows stop summing to 1.

        A student who captures attention without calling `eval()` plots a map
        that inference never used. This is not a bug to fix -- dropout is
        supposed to do that -- but it is a result worth refusing to trust, so
        it is pinned here rather than left to surprise someone.
        """
        model = _model(dropout=0.5)
        model.train()
        torch.manual_seed(0)
        _, weights = model(SRC, TGT, return_attention=True)
        sums = weights.sum(-1)
        self.assertFalse(
            torch.allclose(sums, torch.ones_like(sums), atol=1e-3),
            "expected dropout to break the row sums in training mode",
        )

    def test_eval_mode_weights_are_a_distribution(self):
        model = _model(dropout=0.5)
        model.eval()
        _, weights = model(SRC, TGT, return_attention=True)
        sums = weights.sum(-1)
        self.assertTrue(torch.allclose(sums, torch.ones_like(sums), atol=1e-5))


class ArchitectureParityTests(unittest.TestCase):
    """The point of matching the LSTM's signature."""

    def test_same_call_works_on_both_architectures(self):
        """One call shape, either model, as with the decoders in #9."""
        from torchlingo.models import SimpleSeq2SeqLSTM

        transformer = _model()
        torch.manual_seed(0)
        lstm = SimpleSeq2SeqLSTM(
            src_vocab_size=20,
            tgt_vocab_size=20,
            emb_dim=16,
            hidden_dim=16,
            num_layers=1,
            attention=True,
        )
        lstm.eval()

        for model in (transformer, lstm):
            with torch.no_grad():
                logits, weights = model(SRC, TGT, return_attention=True)
            self.assertEqual(logits.shape[:2], (1, TGT.size(1)))
            self.assertEqual(weights.shape, (1, TGT.size(1), SRC.size(1)))


if __name__ == "__main__":
    unittest.main()
