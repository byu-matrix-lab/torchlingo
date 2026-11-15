"""Decoding raises when a vocabulary and the config disagree on special-token ids.

The decoders start at ``config.sos_idx`` and stop at ``config.eos_idx``; the vocabulary
reads the output with its own ids. Until Nov 2026 a mismatch decoded every sentence to
``max_len`` and returned plausible-looking text with no error. A Final Project that brings
its own vocabulary is where this happens.
"""

import tempfile
import unittest
from pathlib import Path

import torch

from torchlingo import inference_fast
from torchlingo.config import Config
from torchlingo.data_processing.vocab import SimpleVocab
from torchlingo.inference import translate_batch, translate_to_file
from torchlingo.models import SimpleTransformer

CORPUS = ["el gato duerme", "the cat sleeps", "a dog runs", "un perro corre"]


def _vocab(**ids) -> SimpleVocab:
    vocab = SimpleVocab(min_freq=1, **ids)
    vocab.build_vocab(CORPUS)
    return vocab


def _model(size: int) -> SimpleTransformer:
    torch.manual_seed(0)
    model = SimpleTransformer(
        size,
        size,
        d_model=16,
        n_heads=2,
        num_encoder_layers=1,
        num_decoder_layers=1,
        d_ff=32,
    )
    return model.eval()


class SpecialIdMismatchTests(unittest.TestCase):
    def setUp(self):
        self.vocab = _vocab()
        self.model = _model(len(self.vocab))

    def test_matching_ids_decode(self):
        out = translate_batch(
            self.model, ["the cat sleeps"], self.vocab, self.vocab, max_len=5
        )
        self.assertEqual(len(out), 1)

    def test_a_config_with_other_ids_raises_naming_both_values(self):
        cfg = Config(pad_idx=3, unk_idx=0, sos_idx=1, eos_idx=2)
        with self.assertRaises(ValueError) as caught:
            translate_batch(
                self.model, ["the cat sleeps"], self.vocab, self.vocab, config=cfg
            )
        message = str(caught.exception)
        self.assertIn(
            f"src_vocab.pad_idx is {self.vocab.pad_idx} but config.pad_idx is 3",
            message,
        )
        self.assertIn("tgt_vocab.pad_idx", message)

    def test_only_the_target_vocabulary_disagreeing_still_raises(self):
        default = Config()
        tgt = _vocab(eos_idx=default.pad_idx, pad_idx=default.eos_idx)
        with self.assertRaises(ValueError) as caught:
            translate_batch(self.model, ["the cat sleeps"], self.vocab, tgt)
        self.assertIn("tgt_vocab.eos_idx", str(caught.exception))
        self.assertNotIn("src_vocab", str(caught.exception))

    def test_the_fast_path_and_translate_to_file_raise_too(self):
        cfg = Config(pad_idx=3, unk_idx=0, sos_idx=1, eos_idx=2)
        with self.assertRaises(ValueError):
            inference_fast.translate_batch(
                self.model, ["the cat sleeps"], self.vocab, self.vocab, config=cfg
            )
        out = Path(tempfile.mkdtemp()) / "test.hyp"
        with self.assertRaises(ValueError):
            translate_to_file(
                self.model, ["the cat sleeps"], self.vocab, self.vocab, out, config=cfg
            )
        self.assertFalse(out.exists(), "nothing is written before the check")

    def test_a_vocabulary_without_id_attributes_is_not_checked(self):
        class Bare:
            def __init__(self, inner):
                self.inner = inner

            def encode(self, text, add_special_tokens=True):
                return self.inner.encode(text, add_special_tokens=add_special_tokens)

            def decode(self, ids, skip_special_tokens=True):
                return self.inner.decode(ids, skip_special_tokens=skip_special_tokens)

        bare = Bare(self.vocab)
        out = translate_batch(self.model, ["the cat sleeps"], bare, bare, max_len=5)
        self.assertEqual(len(out), 1)


if __name__ == "__main__":
    unittest.main()
