"""load_model: one call rebuilds a trained model and its vocabularies from its file.

train_model records the model's class, settings and vocabularies in every model file it
writes, so a new session needs only the path. Files written before that (on students' Drives
from A8 and A9) must still load as they always did, through load_checkpoint, and load_model
must refuse them with instructions rather than guess.
"""

import tempfile
import unittest
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader

from torchlingo.checkpoint import (
    CHECKPOINT_FORMAT,
    describe_model,
    load_checkpoint,
    load_model,
    save_checkpoint,
)
from torchlingo.config import Config
from torchlingo.data_processing import NMTDataset, collate_fn
from torchlingo.data_processing.vocab import CharVocab, SentencePieceVocab, SimpleVocab
from torchlingo.inference import translate_batch
from torchlingo.models import SimpleSeq2SeqLSTM, SimpleTransformer
from torchlingo.training import train_model

CPU = torch.device("cpu")
SPM = Path(__file__).resolve().parents[1] / "data" / "pretrained" / "spm.model"
PAIRS = pd.DataFrame(
    {
        "src": ["hello world", "thank you", "good morning", "see you soon"],
        "tgt": ["hola mundo", "gracias", "buenos dias", "hasta pronto"],
    }
)
TINY = {
    "d_model": 16,
    "n_heads": 2,
    "num_encoder_layers": 1,
    "num_decoder_layers": 1,
    "d_ff": 32,
}


def same_weights(a: torch.nn.Module, b: torch.nn.Module) -> bool:
    sa, sb = a.state_dict(), b.state_dict()
    return sa.keys() == sb.keys() and all(torch.equal(sa[k], sb[k]) for k in sa)


class TestTrainModelWritesWhatLoadModelNeeds(unittest.TestCase):
    """Train a toy model with save_dir, then rebuild it from the file alone."""

    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.dir = Path(cls.tmp.name)
        PAIRS.to_csv(cls.dir / "train.tsv", sep="\t", index=False)
        cls.config = Config(min_freq=1, batch_size=2, num_workers=0, **TINY)
        cls.data = NMTDataset(cls.dir / "train.tsv", config=cls.config)
        loader = DataLoader(cls.data, batch_size=2, collate_fn=collate_fn)
        torch.manual_seed(0)
        cls.model = SimpleTransformer(
            len(cls.data.src_vocab), len(cls.data.tgt_vocab), config=cls.config
        )
        train_model(
            cls.model,
            loader,
            loader,
            num_epochs=2,
            config=cls.config,
            save_dir=cls.dir / "best",
            device=CPU,
        )
        cls.path = cls.dir / "best" / "model_best.pt"

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_the_file_records_class_settings_and_vocabularies(self):
        raw = torch.load(self.path, weights_only=True)
        self.assertEqual(raw["format"], CHECKPOINT_FORMAT)
        self.assertEqual(raw["model_class"], "SimpleTransformer")
        self.assertEqual(raw["model_config"]["d_model"], 16)
        self.assertEqual(raw["src_vocab"]["token2idx"], self.data.src_vocab.token2idx)
        self.assertEqual(raw["tgt_vocab"]["token2idx"], self.data.tgt_vocab.token2idx)

    def test_load_model_rebuilds_the_same_model_and_vocabularies(self):
        model, src_vocab, tgt_vocab = load_model(self.path, device=CPU)
        best = torch.load(self.path, weights_only=True)["model_state_dict"]
        self.assertTrue(all(torch.equal(model.state_dict()[k], best[k]) for k in best))
        self.assertFalse(model.training, "returned in eval mode")
        self.assertEqual(src_vocab.token2idx, self.data.src_vocab.token2idx)
        self.assertEqual(tgt_vocab.token2idx, self.data.tgt_vocab.token2idx)

    def test_the_rebuilt_model_translates_as_the_original_does(self):
        loaded = load_model(self.path, device=CPU)
        original = SimpleTransformer(
            len(self.data.src_vocab), len(self.data.tgt_vocab), config=self.config
        )
        original.load_state_dict(load_checkpoint(self.path)["model_state_dict"])
        original.eval()
        sentences = list(PAIRS.src)
        expected = translate_batch(
            original, sentences, self.data.src_vocab, self.data.tgt_vocab, device=CPU
        )
        got = translate_batch(
            loaded.model, sentences, loaded.src_vocab, loaded.tgt_vocab, device=CPU
        )
        self.assertEqual(got, expected)

    def test_load_checkpoint_still_reads_it_as_the_notebooks_do(self):
        """The A8 and A9 notebooks: load_checkpoint, then load_state_dict into their model."""
        ckpt = load_checkpoint(self.path)
        rebuilt = SimpleTransformer(
            len(self.data.src_vocab), len(self.data.tgt_vocab), config=self.config
        )
        rebuilt.load_state_dict(ckpt["model_state_dict"])
        self.assertEqual(len(ckpt["train_losses"]), 2)


class TestOlderFiles(unittest.TestCase):
    """Files written before train_model recorded the model: students' A8 and A9 files."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)
        self.model = SimpleTransformer(12, 14, **TINY)

    def tearDown(self):
        self.tmp.cleanup()

    def old_train_model_file(self) -> Path:
        """Exactly the keys train_model wrote before this change."""
        path = self.dir / "model_best.pt"
        opt = torch.optim.AdamW(self.model.parameters())
        torch.save(
            {
                "epoch": 3,
                "global_step": 30,
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": opt.state_dict(),
                "scheduler_state_dict": {},
                "val_loss": 2.5,
                "train_losses": [3.0, 2.8, 2.6],
                "val_losses": [2.9, 2.7, 2.5],
            },
            path,
        )
        return path

    def test_an_old_file_loads_through_load_checkpoint_as_before(self):
        ckpt = load_checkpoint(self.old_train_model_file())
        self.assertEqual(ckpt["format"], "legacy")
        rebuilt = SimpleTransformer(12, 14, **TINY)
        rebuilt.load_state_dict(ckpt["model_state_dict"])
        self.assertTrue(same_weights(rebuilt, self.model))

    def test_load_model_refuses_an_old_file_and_says_what_to_do(self):
        for path in (self.old_train_model_file(), self.dir / "bare.pt"):
            if path.name == "bare.pt":
                torch.save(self.model.state_dict(), path)
            with self.subTest(path.name), self.assertRaises(ValueError) as refused:
                load_model(path, device=CPU)
            self.assertIn(
                "saved before TorchLingo recorded them", str(refused.exception)
            )
            self.assertIn("load_state_dict", str(refused.exception))
            self.assertIn("config=", str(refused.exception))

    def test_an_old_file_loads_given_its_config_and_vocabularies(self):
        src, tgt = self.vocab_of_size(12), self.vocab_of_size(14)
        for path in (self.old_train_model_file(), self.dir / "bare.pt"):
            if path.name == "bare.pt":
                torch.save(self.model.state_dict(), path)
            with self.subTest(path.name):
                loaded = load_model(
                    path, CPU, config=Config(**TINY), src_vocab=src, tgt_vocab=tgt
                )
                self.assertTrue(same_weights(loaded.model, self.model))
                self.assertFalse(loaded.model.training, "returned in eval mode")
                self.assertIs(loaded.src_vocab, src)
                self.assertIs(loaded.tgt_vocab, tgt)

    def test_an_old_file_with_the_wrong_config_says_so(self):
        src, tgt = self.vocab_of_size(12), self.vocab_of_size(14)
        wrong = Config(**{**TINY, "d_model": 32})
        with self.assertRaises(ValueError) as refused:
            load_model(
                self.old_train_model_file(),
                CPU,
                config=wrong,
                src_vocab=src,
                tgt_vocab=tgt,
            )
        self.assertIn("do not fit a SimpleTransformer", str(refused.exception))

    def test_an_old_file_still_needs_its_vocabularies(self):
        with self.assertRaises(ValueError) as refused:
            load_model(self.old_train_model_file(), CPU, config=Config(**TINY))
        self.assertIn("pass src_vocab=", str(refused.exception))

    @staticmethod
    def vocab_of_size(n: int) -> SimpleVocab:
        """A vocabulary of exactly n entries: the specials plus made-up words."""
        vocab = SimpleVocab(min_freq=1)
        vocab.build_vocab(["x"])
        extra = n - (len(vocab) - 1)  # the specials, without "x"
        vocab = SimpleVocab(min_freq=1)
        vocab.build_vocab([" ".join(f"w{i}" for i in range(extra))])
        assert len(vocab) == n, len(vocab)
        return vocab

    def test_the_pretrained_file_layout_loads_given_its_vocabulary(self):
        """data/pretrained/model.pt: one vocab_size, settings without sizes, spm.model apart."""
        vocab = SentencePieceVocab(str(SPM))
        model = SimpleTransformer(len(vocab), len(vocab), **TINY)
        path = self.dir / "model.pt"
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "model_config": dict(TINY),
                "vocab_size": len(vocab),
                "trained_minutes": 1.0,
            },
            path,
        )
        loaded = load_model(path, device=CPU, src_vocab=vocab)
        self.assertIs(loaded.src_vocab, vocab)
        self.assertIs(loaded.tgt_vocab, vocab, "one vocabulary serves both sides")
        self.assertTrue(same_weights(loaded.model, model))


class TestVocabularies(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.path = Path(self.tmp.name) / "model.pt"

    def tearDown(self):
        self.tmp.cleanup()

    def test_a_shared_sentencepiece_vocabulary_is_stored_once_and_comes_back_shared(
        self,
    ):
        vocab = SentencePieceVocab(str(SPM))
        model = SimpleTransformer(len(vocab), len(vocab), **TINY)
        save_checkpoint(self.path, model, src_vocab=vocab, tgt_vocab=vocab)
        self.assertEqual(
            torch.load(self.path, weights_only=True)["tgt_vocab"], {"same_as": "src"}
        )
        _, src_vocab, tgt_vocab = load_model(self.path, device=CPU)
        self.assertIs(src_vocab, tgt_vocab)
        sentence = "This report is our main public report."
        self.assertEqual(src_vocab.encode(sentence), vocab.encode(sentence))

    def test_a_character_vocabulary_round_trips(self):
        src, tgt = SimpleVocab(min_freq=1), CharVocab(min_freq=1)
        src.build_vocab(["I like cats"])
        tgt.build_vocab(["我喜欢猫。"])
        model = SimpleTransformer(len(src), len(tgt), **TINY)
        save_checkpoint(self.path, model, src_vocab=src, tgt_vocab=tgt)
        _, src_back, tgt_back = load_model(self.path, device=CPU)
        self.assertIsInstance(tgt_back, CharVocab)
        self.assertEqual(tgt_back.decode(tgt.encode("我喜欢猫")), "我喜欢猫")
        self.assertEqual(src_back.token2idx, src.token2idx)

    def test_an_lstm_round_trips(self):
        vocab = SimpleVocab(min_freq=1)
        vocab.build_vocab(["hello world"])
        model = SimpleSeq2SeqLSTM(
            len(vocab),
            len(vocab),
            emb_dim=8,
            hidden_dim=8,
            num_layers=1,
            attention=True,
        )
        save_checkpoint(self.path, model, src_vocab=vocab, tgt_vocab=vocab)
        loaded = load_model(self.path, device=CPU)
        self.assertIsInstance(loaded.model, SimpleSeq2SeqLSTM)
        self.assertTrue(same_weights(loaded.model, model))

    def test_a_vocabulary_of_the_wrong_size_is_refused(self):
        vocab = SimpleVocab(min_freq=1)
        vocab.build_vocab(["hello world"])
        model = SimpleTransformer(len(vocab), len(vocab), **TINY)
        save_checkpoint(self.path, model, src_vocab=vocab, tgt_vocab=vocab)
        other = SimpleVocab(min_freq=1)
        other.build_vocab(["a different corpus entirely"])
        with self.assertRaises(ValueError) as refused:
            load_model(self.path, device=CPU, src_vocab=other)
        self.assertIn(
            "not the vocabulary this model was trained with", str(refused.exception)
        )

    def test_a_file_without_vocabularies_asks_for_them(self):
        model = SimpleTransformer(10, 10, **TINY)
        save_checkpoint(self.path, model)
        with self.assertRaises(ValueError) as refused:
            load_model(self.path, device=CPU)
        self.assertIn("pass src_vocab=", str(refused.exception))

    def test_a_model_torchlingo_did_not_build_is_described_as_nothing(self):
        self.assertEqual(describe_model(torch.nn.Linear(2, 2)), {})


if __name__ == "__main__":
    unittest.main()
