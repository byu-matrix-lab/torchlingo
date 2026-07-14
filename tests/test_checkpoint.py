import tempfile
import unittest
from pathlib import Path

import torch

from torchlingo.checkpoint import (
    CHECKPOINT_FORMAT,
    load_checkpoint,
    save_checkpoint,
)
from torchlingo.models import SimpleTransformer


class TestCheckpointBundle(unittest.TestCase):
    """Save/load self-describing translation checkpoints."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)
        self.model_config = {
            "src_vocab_size": 40,
            "tgt_vocab_size": 40,
            "d_model": 32,
            "n_heads": 4,
            "num_encoder_layers": 1,
            "num_decoder_layers": 1,
            "d_ff": 64,
            "max_seq_length": 32,
        }
        self.model = SimpleTransformer(**self.model_config)
        # Stand-in "SentencePiece model" files (bytes are opaque to the bundler).
        self.src_sp = self.dir / "src.model"
        self.tgt_sp = self.dir / "tgt.model"
        self.src_sp.write_bytes(b"SRC-SP-MODEL-BYTES")
        self.tgt_sp.write_bytes(b"TGT-SP-MODEL-BYTES")

    def tearDown(self):
        self.tmp.cleanup()

    def test_roundtrip_preserves_weights_config_and_tokenizers(self):
        path = self.dir / "model.pt"
        save_checkpoint(
            path,
            self.model,
            model_config=self.model_config,
            src_sp_model=self.src_sp,
            tgt_sp_model=self.tgt_sp,
        )
        ckpt = load_checkpoint(path)

        self.assertEqual(ckpt["format"], CHECKPOINT_FORMAT)
        self.assertEqual(ckpt["model_config"], self.model_config)
        self.assertEqual(ckpt["src_sp_model"], b"SRC-SP-MODEL-BYTES")
        self.assertEqual(ckpt["tgt_sp_model"], b"TGT-SP-MODEL-BYTES")

        # Weights reload into a fresh model and match exactly.
        restored = SimpleTransformer(**self.model_config)
        restored.load_state_dict(ckpt["model_state_dict"])
        for (n1, p1), (n2, p2) in zip(
            self.model.named_parameters(), restored.named_parameters()
        ):
            self.assertEqual(n1, n2)
            self.assertTrue(torch.equal(p1, p2))

    def test_parent_directories_created(self):
        path = self.dir / "nested" / "deeper" / "model.pt"
        save_checkpoint(path, self.model, model_config=self.model_config)
        self.assertTrue(path.exists())

    def test_shared_tokenizer(self):
        path = self.dir / "shared.pt"
        save_checkpoint(
            path,
            self.model,
            model_config=self.model_config,
            src_sp_model=self.src_sp,
            tgt_sp_model=self.src_sp,
        )
        ckpt = load_checkpoint(path)
        self.assertEqual(ckpt["src_sp_model"], ckpt["tgt_sp_model"])

    def test_legacy_bare_state_dict_still_loads(self):
        # A pre-existing checkpoint saved as a bare state_dict.
        legacy = self.dir / "legacy.pt"
        torch.save(self.model.state_dict(), legacy)

        ckpt = load_checkpoint(legacy)
        self.assertEqual(ckpt["format"], "legacy")
        self.assertIsNone(ckpt["model_config"])
        self.assertNotIn("src_sp_model", ckpt)

        restored = SimpleTransformer(**self.model_config)
        restored.load_state_dict(ckpt["model_state_dict"])

    def test_legacy_training_state_dict_extracts_weights(self):
        # train_model() saves a dict with model/optimizer/scheduler state,
        # not a bare state_dict. load_checkpoint must still find the weights.
        legacy = self.dir / "training_state.pt"
        torch.save(
            {
                "epoch": 3,
                "global_step": 100,
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": {},
                "val_loss": 1.23,
            },
            legacy,
        )

        ckpt = load_checkpoint(legacy)
        self.assertEqual(ckpt["format"], "legacy")
        self.assertIsNone(ckpt["model_config"])

        restored = SimpleTransformer(**self.model_config)
        restored.load_state_dict(ckpt["model_state_dict"])
        for (n1, p1), (n2, p2) in zip(
            self.model.named_parameters(), restored.named_parameters()
        ):
            self.assertEqual(n1, n2)
            self.assertTrue(torch.equal(p1, p2))


if __name__ == "__main__":
    unittest.main()
