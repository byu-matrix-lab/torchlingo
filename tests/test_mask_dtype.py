"""The decoder's masks must all be the same dtype.

PyTorch deprecated passing a float ``attn_mask`` alongside a boolean
``key_padding_mask``: it warns on every call and is scheduled for removal, at
which point every decode in this library would raise rather than warn.

The library had both conventions at once. The training path built its causal
mask with :func:`create_causal_mask`, which is boolean and matches the padding
masks. The decoders reached for ``nn.Transformer.generate_square_subsequent_mask``
instead, which returns float32 -- so training was silent and inference warned
six times in a six-token decode.

These tests pin the agreement rather than the absence of a particular warning
string, since the wording of a PyTorch warning is not ours to depend on.
"""

import unittest
import warnings

import torch

from torchlingo import inference_fast
from torchlingo.config import get_default_config
from torchlingo.inference import beam_search_decode, greedy_decode
from torchlingo.models import SimpleTransformer
from torchlingo.models.transformer_simple import (
    create_causal_mask,
    create_key_padding_mask,
)


def _model():
    torch.manual_seed(0)
    return SimpleTransformer(
        src_vocab_size=30,
        tgt_vocab_size=30,
        d_model=32,
        n_heads=2,
        num_encoder_layers=1,
        num_decoder_layers=1,
        d_ff=64,
        dropout=0.0,
    )


# Padding is what brings a key_padding_mask into play at all; without it there
# is nothing for the causal mask to disagree with.
PADDED_SRC = torch.tensor([[1, 5, 6, 7, 2, 0, 0]])


class MaskDtypeAgreementTests(unittest.TestCase):
    def test_causal_and_padding_masks_have_the_same_dtype(self):
        causal = create_causal_mask(5, torch.device("cpu"))
        padding = create_key_padding_mask(PADDED_SRC, pad_idx=0)
        self.assertEqual(causal.dtype, padding.dtype)
        self.assertEqual(causal.dtype, torch.bool)

    def test_causal_mask_masks_the_future_and_nothing_else(self):
        # Guards the dtype choice against being "fixed" into the wrong
        # semantics: for a boolean attn_mask, True means *not allowed to
        # attend*, so the upper triangle is masked and the diagonal is not.
        causal = create_causal_mask(4, torch.device("cpu"))
        self.assertTrue(bool(causal[0, 1]), "position 0 must not see position 1")
        self.assertFalse(bool(causal[1, 0]), "position 1 must see position 0")
        self.assertFalse(bool(causal[2, 2]), "a position must see itself")


class NoDeprecationWarningTests(unittest.TestCase):
    """Every Transformer decode path, on input that has padding."""

    def _assert_no_deprecation(self, call, label):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            call()
        offending = [
            str(w.message) for w in caught if "deprecated" in str(w.message).lower()
        ]
        self.assertEqual(offending, [], f"{label} raised deprecation warnings")

    def test_greedy_decode(self):
        model, cfg = _model(), get_default_config()
        self._assert_no_deprecation(
            lambda: greedy_decode(model, PADDED_SRC, max_len=6, config=cfg), "greedy"
        )

    def test_beam_search_decode(self):
        model, cfg = _model(), get_default_config()
        self._assert_no_deprecation(
            lambda: beam_search_decode(
                model, PADDED_SRC, beam_size=3, max_len=6, config=cfg
            ),
            "beam",
        )

    def test_fast_beam_search_decode(self):
        model, cfg = _model(), get_default_config()
        self._assert_no_deprecation(
            lambda: inference_fast.beam_search_decode(
                model, PADDED_SRC, beam_size=3, max_len=6, config=cfg
            ),
            "fast beam",
        )

    def test_training_forward_stays_clean(self):
        # It always was; pinned so a future change to create_causal_mask cannot
        # regress the path that was never broken.
        model = _model()
        tgt = torch.tensor([[1, 8, 9, 2, 0]])
        self._assert_no_deprecation(lambda: model(PADDED_SRC, tgt), "training forward")


class OutputUnchangedTests(unittest.TestCase):
    def test_boolean_and_float_causal_masks_decode_identically(self):
        """The dtype change must not move a single token.

        A boolean mask with True-means-masked and a float mask with -inf in the
        same positions express the same constraint, so this should hold -- but
        "should" is why it is asserted. ``tests/test_decoding_equivalence.py``
        covers the golden outputs; this covers the specific substitution made.
        """
        from torch import nn

        model, cfg = _model(), get_default_config()
        memory = model.encode(
            PADDED_SRC, src_key_padding_mask=PADDED_SRC.eq(cfg.pad_idx)
        )
        tgt = torch.tensor([[cfg.sos_idx, 7, 11]])

        def logits(mask):
            with torch.no_grad():
                return model.decode(
                    tgt,
                    memory,
                    src_key_padding_mask=PADDED_SRC.eq(cfg.pad_idx),
                    tgt_key_padding_mask=tgt.eq(cfg.pad_idx),
                    tgt_mask=mask,
                )

        boolean = logits(create_causal_mask(tgt.size(1), torch.device("cpu")))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            floating = logits(
                nn.Transformer.generate_square_subsequent_mask(tgt.size(1))
            )

        self.assertTrue(torch.allclose(boolean, floating, atol=1e-6))


if __name__ == "__main__":
    unittest.main()
