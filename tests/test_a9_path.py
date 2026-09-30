"""Assignment 9's path, end to end: A8's split and training, then the same with subwords.

A9 retrains a student's A8 model with SentencePiece and compares the two BLEU scores. The
Lecture 9 notebook prints the change to make in the A8 kickoff notebook's Step 6:
``use_sentencepiece=True``, the model path, and a longer ``max_decode_length``. Nothing ran that
path until 2026-09-29, when it turned out that ``create_dataloaders`` with ``sp_model_path``
alone loaded the *target* vocabulary from a default path Colab does not have, so every student
would have stopped at A9's first training cell (fixed in PR #189). These tests run it the way the
kickoff does, on a corpus small enough to train in seconds on a CPU.

What they check is the path, not translation quality: a model trained for two epochs on a toy
corpus translates badly, and that is fine. What must hold is that both runs train, and that both
score **decoded text**: A9's comparison is meaningless if the subword run is scored on pieces.
"""

import random
import tempfile
import unittest
from pathlib import Path

import pandas as pd
import sacrebleu
import torch

from torchlingo.config import Config
from torchlingo.data_processing import create_dataloaders
from torchlingo.data_processing.vocab import SentencePieceVocab
from torchlingo.inference import translate_batch
from torchlingo.models import SimpleTransformer
from torchlingo.preprocessing import split_exact, train_sentencepiece
from torchlingo.training import train_model

SEED = 479
PIECE_MARK = "▁"  # SentencePiece's word-start marker; never in decoded text

NAMES = ["Maria", "Juan", "Ana", "Pedro", "Elena", "Diego"]
VERBS = {"sees": "ve", "wants": "quiere", "finds": "encuentra", "likes": "aprecia"}
NOUNS = {"cats": "gatos", "dogs": "perros", "birds": "pajaros", "books": "libros"}
ADJS = {"old": "viejos", "young": "jovenes", "big": "grandes", "small": "pequenos"}


def toy_corpus(n: int, seed: int) -> pd.DataFrame:
    """Word-for-word English/Spanish pairs, varied enough to split and to learn a little."""
    rng = random.Random(seed)
    rows = []
    for _ in range(n):
        name, verb = rng.choice(NAMES), rng.choice(list(VERBS))
        adj, noun = rng.choice(list(ADJS)), rng.choice(list(NOUNS))
        number = rng.randint(2, 99)
        rows.append(
            (
                f"{name} {verb} {number} {adj} {noun}",
                f"{name} {VERBS[verb]} {number} {ADJS[adj]} {NOUNS[noun]}",
            )
        )
    return pd.DataFrame(rows, columns=["src", "tgt"]).drop_duplicates("src")


class TestTheA9Path(unittest.TestCase):
    """A8 with words, then A9 with subwords, on the same files and the same split."""

    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.out = Path(cls.tmp.name)

        # The kickoff's Step 3 and Step 5: an exact, seeded split, written as TSVs.
        train, val, test = split_exact(
            toy_corpus(600, seed=0), n_val=40, n_test=40, seed=SEED
        )
        for name, frame in (("train", train), ("val", val), ("test", test)):
            frame.to_csv(cls.out / f"{name}.tsv", sep="\t", index=False)
        cls.test = test

        # The Lecture 9 notebook: SentencePiece on the training split only, and the decode
        # length measured from the longest training target in pieces.
        cls.spm = str(cls.out / "spm.model")
        train_sentencepiece(
            [cls.out / "train.tsv"], str(cls.out / "spm"), vocab_size=80
        )
        pieces = SentencePieceVocab(cls.spm)
        longest = max(
            len(pieces.encode(s, add_special_tokens=False)) for s in train["tgt"]
        )
        cls.max_decode = longest + 10

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    @classmethod
    def train_and_translate(cls, label, max_decode_length=None, **loader_settings):
        """Steps 6 and 7 and the scoring cell of the A8 kickoff, at toy size.

        Not named ``run``: that is ``unittest.TestCase.run``, which the runner calls.
        """
        config_settings = (
            {"max_decode_length": max_decode_length} if max_decode_length else {}
        )
        config = Config(
            d_model=32,
            n_heads=2,
            num_encoder_layers=1,
            num_decoder_layers=1,
            d_ff=64,
            batch_size=32,
            num_workers=0,
            **config_settings,
        )
        random.seed(SEED)
        train_loader, val_loader, src_vocab, tgt_vocab = create_dataloaders(
            cls.out / "train.tsv",
            cls.out / "val.tsv",
            use_bucketing=True,
            num_workers=0,
            config=config,
            **loader_settings,
        )
        torch.manual_seed(SEED)
        model = SimpleTransformer(
            src_vocab_size=len(src_vocab), tgt_vocab_size=len(tgt_vocab), config=config
        )
        train_model(
            model,
            train_loader,
            val_loader=val_loader,
            num_epochs=2,
            config=config,
            save_dir=cls.out / label,
            log_every=0,
        )
        model.eval()
        hypotheses = translate_batch(
            model,
            list(cls.test.src),
            src_vocab,
            tgt_vocab,
            decode_strategy="greedy",
            config=config,
        )
        return {
            "hypotheses": hypotheses,
            "src_vocab": src_vocab,
            "tgt_vocab": tgt_vocab,
            "config": config,
        }

    def assert_scores_decoded_text(self, run):
        hypotheses = run["hypotheses"]
        self.assertEqual(len(hypotheses), len(self.test))
        # Guards the next assertion against passing vacuously on empty output.
        self.assertTrue(
            any(h.strip() for h in hypotheses), "every translation is empty"
        )
        self.assertFalse(
            [h for h in hypotheses if PIECE_MARK in h],
            "translate_batch returned subword pieces; A9 would be scored on pieces",
        )
        bleu = sacrebleu.corpus_bleu(hypotheses, [list(self.test.tgt)])
        self.assertGreaterEqual(bleu.score, 0.0)

    # Each run lives in its own test, so a failure names the path that broke: before PR #189
    # only the last one fails, which is the point of the Lecture 9 notebook's printed form.

    def test_a8_word_run_scores_text(self):
        self.assert_scores_decoded_text(self.train_and_translate(label="a8-words"))

    def test_a9_as_the_lecture_9_notebook_prints_it(self):
        """The same model named for both sides: works on torchlingo 0.2.3, which students have."""
        run = self.train_and_translate(
            label="a9-printed",
            max_decode_length=self.max_decode,
            use_sentencepiece=True,
            sp_model_path=self.spm,
            sp_tgt_model_path=self.spm,
        )
        self.assertIs(
            run["tgt_vocab"], run["src_vocab"], "one subword model serves both sides"
        )
        self.assertEqual(len(run["src_vocab"]), len(SentencePieceVocab(self.spm)))
        self.assertEqual(run["config"].max_decode_length, self.max_decode)
        self.assert_scores_decoded_text(run)

    def test_a9_with_sp_model_path_alone(self):
        """The call that raised file-not-found for every student before PR #189."""
        run = self.train_and_translate(
            label="a9-alone",
            max_decode_length=self.max_decode,
            use_sentencepiece=True,
            sp_model_path=self.spm,
        )
        self.assertIs(
            run["tgt_vocab"], run["src_vocab"], "one subword model serves both sides"
        )
        self.assertEqual(len(run["src_vocab"]), len(SentencePieceVocab(self.spm)))
        self.assert_scores_decoded_text(run)


if __name__ == "__main__":
    unittest.main()
