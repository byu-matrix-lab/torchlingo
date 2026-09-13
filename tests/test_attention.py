import unittest

import torch

from torchlingo.config import ATTENTION_TYPES, Config, get_default_config
from torchlingo.inference import greedy_decode
from torchlingo.models import SimpleSeq2SeqLSTM
from torchlingo.models.attention import (
    AdditiveAttention,
    DotProductAttention,
    build_attention,
)
from torchlingo.visualization import format_attention, plot_attention

PAD = get_default_config().pad_idx


def make_model(attn_type=None, attention=True, **kwargs):
    """Build a small LSTM model with deterministic weights."""
    torch.manual_seed(0)
    return SimpleSeq2SeqLSTM(
        src_vocab_size=40,
        tgt_vocab_size=40,
        emb_dim=8,
        hidden_dim=8,
        num_layers=1,
        dropout=0.0,
        attention=attention,
        attn_type=attn_type,
        **kwargs,
    ).eval()


class TestAttentionModules(unittest.TestCase):
    """Test the attention mechanisms in isolation."""

    def test_shapes_for_every_scorer(self):
        """Each scorer should return context and weights of the right shape."""
        dec_out = torch.randn(2, 3, 8)
        enc_out = torch.randn(2, 5, 8)
        for attn_type in ATTENTION_TYPES:
            with self.subTest(attn_type=attn_type):
                attn = build_attention(attn_type, hidden_dim=8)
                context, weights = attn(dec_out, enc_out)
                self.assertEqual(context.shape, (2, 3, 8))
                self.assertEqual(weights.shape, (2, 3, 5))

    def test_weights_are_a_distribution(self):
        """Each row of weights should sum to 1 and be non-negative."""
        dec_out = torch.randn(2, 3, 8)
        enc_out = torch.randn(2, 5, 8)
        for attn_type in ATTENTION_TYPES:
            with self.subTest(attn_type=attn_type):
                _, weights = build_attention(attn_type, 8)(dec_out, enc_out)
                self.assertTrue(torch.all(weights >= 0))
                torch.testing.assert_close(weights.sum(-1), torch.ones(2, 3))

    def test_padding_receives_no_weight(self):
        """Masked source positions should get essentially zero weight."""
        dec_out = torch.randn(2, 3, 8)
        enc_out = torch.randn(2, 5, 8)
        mask = torch.zeros(2, 5, dtype=torch.bool)
        mask[:, 3:] = True  # last two positions are padding
        for attn_type in ATTENTION_TYPES:
            with self.subTest(attn_type=attn_type):
                _, weights = build_attention(attn_type, 8)(dec_out, enc_out, mask)
                self.assertTrue(torch.all(weights[:, :, 3:] < 1e-6))
                torch.testing.assert_close(weights[:, :, :3].sum(-1), torch.ones(2, 3))

    def test_masked_positions_cannot_change_the_context(self):
        """Changing a padded encoder output must not change the context."""
        dec_out = torch.randn(1, 2, 8)
        enc_out = torch.randn(1, 4, 8)
        mask = torch.tensor([[False, False, True, True]])
        other = enc_out.clone()
        other[:, 2:] = torch.randn(1, 2, 8) * 100  # garbage behind the mask
        for attn_type in ATTENTION_TYPES:
            with self.subTest(attn_type=attn_type):
                attn = build_attention(attn_type, 8)
                first, _ = attn(dec_out, enc_out, mask)
                second, _ = attn(dec_out, other, mask)
                torch.testing.assert_close(first, second)

    def test_fully_padded_row_does_not_produce_nan(self):
        """An entirely masked row should degrade gracefully, not to NaN."""
        dec_out = torch.randn(1, 2, 8)
        enc_out = torch.randn(1, 3, 8)
        mask = torch.ones(1, 3, dtype=torch.bool)
        for attn_type in ATTENTION_TYPES:
            with self.subTest(attn_type=attn_type):
                context, weights = build_attention(attn_type, 8)(dec_out, enc_out, mask)
                self.assertFalse(torch.isnan(weights).any())
                self.assertFalse(torch.isnan(context).any())

    def test_dot_product_rejects_mismatched_dims(self):
        """Dot-product attention needs matching hidden sizes."""
        with self.assertRaises(ValueError) as ctx:
            DotProductAttention()(torch.randn(1, 2, 8), torch.randn(1, 3, 16))
        self.assertIn("additive", str(ctx.exception))

    def test_additive_accepts_mismatched_dims(self):
        """Additive attention learns a projection, so dims may differ."""
        attn = AdditiveAttention(hidden_dim=8, enc_dim=16)
        context, weights = attn(torch.randn(1, 2, 8), torch.randn(1, 3, 16))
        self.assertEqual(context.shape, (1, 2, 16))
        self.assertEqual(weights.shape, (1, 2, 3))

    def test_dot_product_has_no_parameters(self):
        """Luong dot attention is parameter-free; additive is not."""
        self.assertEqual(len(list(DotProductAttention().parameters())), 0)
        self.assertGreater(len(list(AdditiveAttention(8).parameters())), 0)

    def test_build_attention_rejects_unknown_type(self):
        """An unknown scorer name should fail loudly."""
        with self.assertRaises(ValueError) as ctx:
            build_attention("luong", hidden_dim=8)
        self.assertIn("dot", str(ctx.exception))


class TestLSTMAttentionIntegration(unittest.TestCase):
    """Test attention wired into SimpleSeq2SeqLSTM."""

    def setUp(self):
        self.src = torch.randint(1, 40, (2, 6))
        self.tgt = torch.randint(1, 40, (2, 4))

    def test_attention_is_off_by_default(self):
        """The default model must keep the classic bottleneck."""
        model = SimpleSeq2SeqLSTM(40, 40, emb_dim=8, hidden_dim=8, num_layers=1)
        self.assertIsNone(model.attention)
        self.assertIsNone(model.attn_combine)
        self.assertFalse(get_default_config().lstm_attention)

    def test_forward_returns_bare_tensor_by_default(self):
        """Existing callers must keep getting a plain logits tensor."""
        for attention in (False, True):
            with self.subTest(attention=attention):
                model = make_model("dot", attention=attention)
                out = model(self.src, self.tgt)
                self.assertIsInstance(out, torch.Tensor)
                self.assertEqual(out.shape, (2, 4, 40))

    def test_return_attention_yields_weights(self):
        """return_attention should add a (batch, tgt_len, src_len) matrix."""
        model = make_model("dot")
        logits, weights = model(self.src, self.tgt, return_attention=True)
        self.assertEqual(logits.shape, (2, 4, 40))
        self.assertEqual(weights.shape, (2, 4, 6))

    def test_return_attention_is_none_without_attention(self):
        """A bottlenecked model reports None rather than fabricating weights."""
        model = make_model(attention=False)
        _, weights = model(self.src, self.tgt, return_attention=True)
        self.assertIsNone(weights)

    def test_disabled_attention_matches_the_original_computation(self):
        """attention=False must reproduce the pre-attention forward pass exactly."""
        model = make_model(attention=False)
        with torch.no_grad():
            expected_emb = model.src_embed(self.src)
            _enc_out, hidden = model.encoder(expected_emb)
            dec_out, _ = model.decoder(model.tgt_embed(self.tgt), hidden)
            expected = model.output(dec_out)
            actual = model(self.src, self.tgt)
        torch.testing.assert_close(actual, expected)

    def test_attention_changes_the_output(self):
        """Attention must actually affect logits, not sit inert."""
        src, tgt = self.src, self.tgt
        without = make_model(attention=False)(src, tgt)
        with_attn = make_model("dot")(src, tgt)
        self.assertFalse(torch.allclose(without, with_attn))

    def test_padding_does_not_affect_attended_output(self):
        """Trailing padding must not change results for the real tokens."""
        model = make_model("dot")
        short = torch.tensor([[5, 6, 7]])
        padded = torch.tensor([[5, 6, 7, PAD, PAD]])
        tgt = torch.tensor([[2, 9]])
        with torch.no_grad():
            a = model(short, tgt)
            b = model(padded, tgt)
        torch.testing.assert_close(a, b)

    def test_weights_depend_on_the_source(self):
        """Different sources should produce different alignments."""
        model = make_model("dot")
        with torch.no_grad():
            _, first = model(self.src, self.tgt, return_attention=True)
            other = self.src.clone()
            other[:, 0] = (other[:, 0] + 7) % 39 + 1
            _, second = model(other, self.tgt, return_attention=True)
        self.assertFalse(torch.allclose(first, second))

    def test_weights_depend_on_decoder_history(self):
        """Alignments should vary across target positions, not repeat."""
        model = make_model("dot")
        with torch.no_grad():
            _, weights = model(self.src, self.tgt, return_attention=True)
        first_row, second_row = weights[0, 0], weights[0, 1]
        self.assertFalse(torch.allclose(first_row, second_row))

    def test_gradients_reach_attention_parameters(self):
        """Additive attention's scoring network must actually train."""
        model = make_model("additive")
        model.train()
        logits = model(self.src, self.tgt)
        logits.sum().backward()
        for name, param in model.attention.named_parameters():
            with self.subTest(param=name):
                self.assertIsNotNone(param.grad)
                self.assertTrue(torch.any(param.grad != 0))

    def test_every_scorer_runs_end_to_end(self):
        """Both published scorers should work through the full model."""
        for attn_type in ATTENTION_TYPES:
            with self.subTest(attn_type=attn_type):
                model = make_model(attn_type)
                logits, weights = model(self.src, self.tgt, return_attention=True)
                self.assertEqual(logits.shape, (2, 4, 40))
                torch.testing.assert_close(weights.sum(-1), torch.ones(2, 4))


class TestDecodeStep(unittest.TestCase):
    """Test that the incremental path agrees with the batched forward pass."""

    def test_decode_step_matches_forward(self):
        """Stepping token by token must equal one teacher-forced forward pass."""
        src = torch.randint(1, 40, (2, 5))
        tgt = torch.randint(1, 40, (2, 4))
        for attention in (False, True):
            with self.subTest(attention=attention):
                model = make_model("dot", attention=attention)
                with torch.no_grad():
                    expected = model(src, tgt)
                    enc_out, hidden, mask = model.encode_source(src)
                    stepped = []
                    for position in range(tgt.size(1)):
                        logits, hidden, _ = model.decode_step(
                            tgt[:, position : position + 1], hidden, enc_out, mask
                        )
                        stepped.append(logits)
                    actual = torch.cat(stepped, dim=1)
                torch.testing.assert_close(actual, expected)

    def test_encode_source_reports_padding(self):
        """The mask returned by encode_source should mark pad positions."""
        model = make_model("dot")
        src = torch.tensor([[5, 6, PAD]])
        _enc_out, _hidden, mask = model.encode_source(src)
        self.assertEqual(mask.tolist(), [[False, False, True]])

    def test_greedy_decode_runs_with_attention(self):
        """greedy_decode must drive the attention path without error."""
        model = make_model("dot")
        src = torch.randint(1, 40, (2, 5))
        decoded = greedy_decode(model, src, max_len=6)
        self.assertEqual(len(decoded), 2)
        self.assertTrue(all(isinstance(ids, list) for ids in decoded))

    def test_greedy_decode_uses_the_encoder_outputs(self):
        """Attention must change greedy output, proving it reached inference."""
        src = torch.randint(1, 40, (2, 5))
        without = greedy_decode(make_model(attention=False), src, max_len=6)
        with_attn = greedy_decode(make_model("dot"), src, max_len=6)
        self.assertNotEqual(without, with_attn)


class TestAttentionConfig(unittest.TestCase):
    """Test the config fields backing attention."""

    def test_defaults(self):
        """Attention should default to off, with the dot scorer selected."""
        cfg = Config()
        self.assertFalse(cfg.lstm_attention)
        self.assertEqual(cfg.lstm_attn_type, "dot")

    def test_accepts_every_documented_scorer(self):
        """Config must accept exactly the scorers the models implement."""
        cfg = Config()
        for attn_type in ATTENTION_TYPES:
            with self.subTest(attn_type=attn_type):
                cfg.lstm_attn_type = attn_type
                self.assertEqual(cfg.lstm_attn_type, attn_type)

    def test_rejects_unknown_scorer(self):
        """An unrecognized scorer name should raise."""
        with self.assertRaises(ValueError):
            Config().lstm_attn_type = "luong"

    def test_rejects_non_string_scorer(self):
        """A non-string scorer should raise a TypeError."""
        with self.assertRaises(TypeError):
            Config().lstm_attn_type = 7

    def test_rejects_non_bool_toggle(self):
        """lstm_attention is a strict boolean."""
        with self.assertRaises(TypeError):
            Config().lstm_attention = "yes"

    def test_model_reads_config(self):
        """A model with no explicit flags should follow the config."""
        cfg = Config()
        cfg.lstm_attention = True
        cfg.lstm_attn_type = "additive"
        model = SimpleSeq2SeqLSTM(
            40, 40, emb_dim=8, hidden_dim=8, num_layers=1, config=cfg
        )
        self.assertIsInstance(model.attention, AdditiveAttention)

    def test_explicit_argument_beats_config(self):
        """Explicit parameters always win over config values."""
        cfg = Config()
        cfg.lstm_attention = True
        model = SimpleSeq2SeqLSTM(
            40,
            40,
            emb_dim=8,
            hidden_dim=8,
            num_layers=1,
            attention=False,
            config=cfg,
        )
        self.assertIsNone(model.attention)


class TestAttentionVisualization(unittest.TestCase):
    """Test the alignment renderers."""

    def setUp(self):
        self.weights = torch.tensor([[0.8, 0.1, 0.1], [0.1, 0.2, 0.7]])
        self.src_tokens = ["el", "gato", "duerme"]
        self.tgt_tokens = ["the", "sleeps"]

    def test_format_attention_lines_up(self):
        """Output should have one header row plus one row per target token."""
        text = format_attention(self.weights, self.src_tokens, self.tgt_tokens)
        lines = text.splitlines()
        self.assertEqual(len(lines), len(self.tgt_tokens) + 1)
        for token in self.src_tokens:
            self.assertIn(token, lines[0])

    def test_format_attention_shows_values(self):
        """show_values should print rounded percentages."""
        text = format_attention(
            self.weights, self.src_tokens, self.tgt_tokens, show_values=True
        )
        self.assertIn("80", text)
        self.assertIn("70", text)

    def test_format_attention_accepts_a_batch_of_one(self):
        """A (1, tgt, src) tensor should be unwrapped automatically."""
        batched = self.weights.unsqueeze(0)
        self.assertEqual(
            format_attention(batched, self.src_tokens, self.tgt_tokens),
            format_attention(self.weights, self.src_tokens, self.tgt_tokens),
        )

    def test_format_attention_rejects_a_real_batch(self):
        """A multi-sentence batch should ask the caller to index it."""
        batched = self.weights.unsqueeze(0).expand(3, -1, -1)
        with self.assertRaises(ValueError) as ctx:
            format_attention(batched, self.src_tokens, self.tgt_tokens)
        self.assertIn("Index the batch", str(ctx.exception))

    def test_format_attention_rejects_mismatched_labels(self):
        """Label counts must match the matrix shape."""
        with self.assertRaises(ValueError):
            format_attention(self.weights, ["only", "two"], self.tgt_tokens)
        with self.assertRaises(ValueError):
            format_attention(self.weights, self.src_tokens, ["one"])

    def test_plot_attention_returns_axes(self):
        """The matplotlib path should produce labelled axes."""
        import matplotlib

        matplotlib.use("Agg")
        ax = plot_attention(
            self.weights, self.src_tokens, self.tgt_tokens, title="alignment"
        )
        self.assertEqual(ax.get_title(), "alignment")
        self.assertEqual(
            [label.get_text() for label in ax.get_xticklabels()], self.src_tokens
        )
        matplotlib.pyplot.close("all")

    def test_model_weights_render(self):
        """Weights straight from a model should render without fuss."""
        model = make_model("dot")
        src = torch.randint(1, 40, (1, 3))
        tgt = torch.randint(1, 40, (1, 2))
        with torch.no_grad():
            _, weights = model(src, tgt, return_attention=True)
        text = format_attention(weights, ["a", "b", "c"], ["x", "y"])
        self.assertEqual(len(text.splitlines()), 3)


if __name__ == "__main__":
    unittest.main()
