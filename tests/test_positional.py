import math
import unittest

import torch

from torchlingo import config
from torchlingo.config import Config
from torchlingo.models.positional import SinusoidalPositionalEncoding


class TestSinusoidalPositionalEncoding(unittest.TestCase):
    """Tests for SinusoidalPositionalEncoding."""

    def test_init_builds_table(self):
        enc = SinusoidalPositionalEncoding(d_model=8, max_seq_len=16, dropout=0.0)
        self.assertEqual(enc.d_model, 8)
        self.assertEqual(enc.max_seq_len, 16)
        self.assertEqual(enc.pe.shape, (16, 8))

    def test_forward_preserves_shape(self):
        enc = SinusoidalPositionalEncoding(d_model=12, max_seq_len=32, dropout=0.0)
        x = torch.randn(2, 5, 12)
        y = enc(x)
        self.assertEqual(y.shape, x.shape)

    def test_forward_adds_encoding(self):
        enc = SinusoidalPositionalEncoding(d_model=8, max_seq_len=16, dropout=0.0)
        x = torch.zeros(1, 4, 8)
        y = enc(x)
        self.assertTrue(torch.allclose(y[0], enc.pe[:4]))

    def test_encoding_values_match_formula(self):
        d_model, seq_len = 8, 10
        enc = SinusoidalPositionalEncoding(
            d_model=d_model, max_seq_len=seq_len, dropout=0.0
        )
        pos, i = 3, 2
        angle = pos / (10000 ** (2 * i / d_model))
        self.assertAlmostEqual(enc.pe[pos, 2 * i].item(), math.sin(angle), places=5)
        self.assertAlmostEqual(enc.pe[pos, 2 * i + 1].item(), math.cos(angle), places=5)

    def test_distinct_positions_get_distinct_encodings(self):
        enc = SinusoidalPositionalEncoding(d_model=16, max_seq_len=32, dropout=0.0)
        self.assertFalse(torch.allclose(enc.pe[0], enc.pe[1]))

    def test_table_extends_for_longer_sequences(self):
        enc = SinusoidalPositionalEncoding(d_model=8, max_seq_len=8, dropout=0.0)
        x = torch.randn(2, 16, 8)
        y = enc(x)
        self.assertEqual(y.shape, x.shape)
        self.assertGreaterEqual(enc.pe.size(0), 16)

    def test_odd_d_model_supported(self):
        enc = SinusoidalPositionalEncoding(d_model=7, max_seq_len=8, dropout=0.0)
        x = torch.randn(1, 4, 7)
        y = enc(x)
        self.assertEqual(y.shape, x.shape)

    def test_dropout_disabled_in_eval_mode(self):
        enc = SinusoidalPositionalEncoding(d_model=8, max_seq_len=8, dropout=0.5)
        enc.eval()
        x = torch.zeros(1, 4, 8)
        self.assertTrue(torch.allclose(enc(x)[0], enc.pe[:4]))


class TestSinusoidalPositionalEncodingConfigOverride(unittest.TestCase):
    """Tests for config override behavior."""

    def test_uses_default_d_model(self):
        enc = SinusoidalPositionalEncoding()
        self.assertEqual(enc.d_model, config.D_MODEL)

    def test_explicit_d_model_overrides_default(self):
        enc = SinusoidalPositionalEncoding(d_model=128)
        self.assertEqual(enc.d_model, 128)

    def test_uses_passed_config_d_model(self):
        custom_cfg = Config(d_model=256, n_heads=8)
        enc = SinusoidalPositionalEncoding(config=custom_cfg)
        self.assertEqual(enc.d_model, 256)

    def test_explicit_d_model_overrides_passed_config(self):
        custom_cfg = Config(d_model=256, n_heads=8)
        enc = SinusoidalPositionalEncoding(d_model=64, config=custom_cfg)
        self.assertEqual(enc.d_model, 64)

    def test_uses_default_max_seq_len(self):
        enc = SinusoidalPositionalEncoding()
        self.assertEqual(enc.max_seq_len, config.MAX_SEQ_LENGTH)

    def test_explicit_max_seq_len_overrides_default(self):
        enc = SinusoidalPositionalEncoding(max_seq_len=1024)
        self.assertEqual(enc.max_seq_len, 1024)

    def test_uses_passed_config_max_seq_len(self):
        custom_cfg = Config(d_model=256, n_heads=8, max_seq_length=2048)
        enc = SinusoidalPositionalEncoding(config=custom_cfg)
        self.assertEqual(enc.max_seq_len, 2048)

    def test_explicit_max_seq_len_overrides_passed_config(self):
        custom_cfg = Config(d_model=256, n_heads=8, max_seq_length=2048)
        enc = SinusoidalPositionalEncoding(max_seq_len=128, config=custom_cfg)
        self.assertEqual(enc.max_seq_len, 128)


if __name__ == "__main__":
    unittest.main()
