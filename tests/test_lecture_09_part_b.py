"""Lecture 9's Part B (Assignment 9), executed from the notebook: Colab faked, toy size.

Part B declares ``colab``, so CI's notebook run skips it, and no one ran it before students did.
Its first run, by hand on 2026-10-07, found that scoring a day later raised ``NameError``:
``BEST_DIR`` was defined only in the training cell, which the notebook tells that student to skip.
These tests run Part B's cells from the notebook file itself, so a pull request that edits Part B
is tested on that pull request.

The fixture stands in for what A8 and Part A leave in ``CS479/assignment8`` on Drive: the split,
``spm.model``, ``a9-settings.json``, and an A8 model in ``best/``. What changes in the cells is
only the epochs and the model's size, the same for A8 and A9 so the saved A8 model still loads.
See ``notebook_cells`` for what is faked and why each swap must match the notebook.

What is checked is the path, not translation quality: one epoch on a toy corpus translates
badly, and that is fine.
"""

import json
import random
import tempfile
import unittest
from pathlib import Path

import sacrebleu
import torch
from notebook_cells import (
    COURSE,
    cell_containing,
    code_cells,
    colab_faked,
    run_cells,
    substitute,
)
from test_a9_path import PIECE_MARK, SEED, toy_corpus

from torchlingo.config import Config
from torchlingo.data_processing import create_dataloaders
from torchlingo.data_processing.vocab import SentencePieceVocab
from torchlingo.models import SimpleTransformer
from torchlingo.preprocessing import split_exact, train_sentencepiece
from torchlingo.training import train_model
from torchlingo.training_checkpoint import TrainingCheckpointer

# The version that replaces the live notebook once A9's deadline has passed (Eric, Oct 13); when
# it is renamed over lecture-09-subword-tokenization.ipynb, so is this. The live notebook is not
# edited in the meantime, so nothing here runs it.
NOTEBOOK = COURSE / "lecture-09-subword-tokenization-new.ipynb"

# The kickoff's model is d_model 512, 6 + 6 layers: minutes per epoch on a CPU even at toy size.
TOY_SIZE = {
    "d_model=512": "d_model=32",
    "n_heads=8": "n_heads=2",
    "num_encoder_layers=6": "num_encoder_layers=1",
    "num_decoder_layers=6": "num_decoder_layers=1",
    "d_ff=2048": "d_ff=64",
}
TOY_CONFIG = {
    "d_model": 32,
    "n_heads": 2,
    "num_encoder_layers": 1,
    "num_decoder_layers": 1,
    "d_ff": 64,
    "batch_size": 64,
    "num_workers": 0,
}

# Part B's cells, by what each one says.
CELL_1 = "# Part B, cell 1"
CONFIG_CELL = "max_decode_length=MAX_DECODE"
TRAINING_CELL = "result = train_model("
SCORING_CELL = "hyp_a9 = translate_all("
WRITE_UP_CELL = "=== For the A9 write-up ==="


# A8's run trains this many epochs in the fixture; Part B must read it from A8's checkpoint.
A8_EPOCHS = 2
READ_EPOCHS = 'EPOCHS = len(a8_state["train_losses"])'


def part_b(*markers: str, epochs: int | None = None) -> list[tuple[int, str]]:
    """The named Part B cells, in notebook order, at toy size.

    ``epochs``, when given, replaces the count Part B reads from A8's checkpoint.
    """
    cells = sorted(cell_containing(code_cells(NOTEBOOK), m) for m in markers)
    swaps = dict(TOY_SIZE)
    if epochs is not None:
        swaps[READ_EPOCHS] = f"EPOCHS = {epochs}"
    return substitute(cells, swaps)


def write_a8_and_part_a(a8_dir: Path, no_spaces: bool) -> int:
    """Write what A8 and Part A leave on Drive; return the number of test pairs.

    A8's run keeps a checkpoint under the name ``assignment-8``, as the kickoff's does. Outside
    Colab that lives in ``checkpoints/`` under the working directory, which ``colab_faked``
    makes the faked Drive, two levels above ``a8_dir``.
    """
    a8_dir.mkdir(parents=True)
    corpus = toy_corpus(600, seed=0)
    if no_spaces:
        corpus["tgt"] = corpus["tgt"].str.replace(" ", "", regex=False)
    train, val, test = split_exact(corpus, n_val=40, n_test=40, seed=SEED)
    for name, frame in (("train", train), ("val", val), ("test", test)):
        frame.to_csv(a8_dir / f"{name}.tsv", sep="\t", index=False)

    # Part A's Step 2 and Step 6
    train_sentencepiece([a8_dir / "train.tsv"], str(a8_dir / "spm"), vocab_size=80)
    pieces = SentencePieceVocab(str(a8_dir / "spm.model"))
    longest = max(len(pieces.encode(s, add_special_tokens=False)) for s in train["tgt"])
    settings = {
        "vocab_size": 80,
        "why": "toy corpus",
        "no_spaces": no_spaces,
        "max_decode_length": longest + 10,
        "piece_ratio": 1.5,
        "unk_percent_words_tgt": 1.0,
        "unk_count_pieces_tgt": 0,
        "expected_unk_percent": 2.0,
        "bleu_prediction": "up a little",
    }
    (a8_dir / "a9-settings.json").write_text(json.dumps(settings), encoding="utf-8")

    # A8's model, built as the kickoff builds it, saved where the kickoff saves it
    config = Config(**TOY_CONFIG)
    random.seed(SEED)
    train_loader, val_loader, src_vocab, tgt_vocab = create_dataloaders(
        a8_dir / "train.tsv",
        a8_dir / "val.tsv",
        use_bucketing=True,
        num_workers=0,
        config=config,
        target_units="characters" if no_spaces else "words",
    )
    torch.manual_seed(SEED)
    model = SimpleTransformer(
        src_vocab_size=len(src_vocab), tgt_vocab_size=len(tgt_vocab), config=config
    )
    drive = a8_dir.parents[1]
    train_model(
        model,
        train_loader,
        val_loader=val_loader,
        num_epochs=A8_EPOCHS,
        config=config,
        save_dir=a8_dir / "best",
        log_every=0,
        checkpointer=TrainingCheckpointer(
            experiment_name="assignment-8",
            checkpoint_dir=drive / "checkpoints" / "assignment-8",
            verbose=False,
        ),
    )
    return len(test)


class PartBStraightThrough:
    """Set up Drive, run Part B start to finish once, and check what it leaves."""

    no_spaces = False

    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.drive = Path(cls.tmp.name)
        cls.a8_dir = cls.drive / "CS479" / "assignment8"
        cls.n_test = write_a8_and_part_a(cls.a8_dir, cls.no_spaces)
        with colab_faked(cls.drive):
            cls.ns, cls.printed = run_cells(
                part_b(CELL_1, CONFIG_CELL, TRAINING_CELL, SCORING_CELL, WRITE_UP_CELL)
            )

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_leaves_a9_model_and_translations_beside_a8s(self):
        self.assertTrue((self.a8_dir / "best-a9" / "model_best.pt").exists())
        self.assertTrue(
            (self.a8_dir / "best" / "model_best.pt").exists(), "A8's model kept"
        )
        hyp = (self.a8_dir / "test-a9.hyp").read_text(encoding="utf-8").splitlines()
        self.assertEqual(len(hyp), self.n_test)

    def test_scores_both_systems_on_decoded_text(self):
        for name in ("hyp_a8", "hyp_a9"):
            hypotheses = self.ns[name]
            self.assertEqual(len(hypotheses), self.n_test, name)
            self.assertFalse(
                [h for h in hypotheses if PIECE_MARK in h], f"{name} has pieces"
            )

    def test_prints_the_write_up_block(self):
        self.assertIn("=== For the A9 write-up ===", self.printed)
        self.assertIn("A8 BLEU:", self.printed)

    def test_trains_for_as_many_epochs_as_a8(self):
        """Read from A8's checkpoint: a student who raised A8's epochs gets the same for A9."""
        self.assertEqual(self.ns["EPOCHS"], A8_EPOCHS)
        self.assertEqual(len(self.ns["result"].train_losses), A8_EPOCHS)


class TestPartB(PartBStraightThrough, unittest.TestCase):
    """A language written with spaces, and the two ways back into Part B in a new session."""

    def test_scoring_a_day_later_skips_training(self):
        """What the notebook tells a returning student to run; raised NameError before the fix."""
        with colab_faked(self.drive):
            ns, _ = run_cells(part_b(CELL_1, CONFIG_CELL, SCORING_CELL))
        self.assertEqual(len(ns["hyp_a9"]), self.n_test)

    def test_rerunning_training_resumes(self):
        """A dropped session: the training cell, run again, continues from the checkpoint."""
        with colab_faked(self.drive):
            ns, printed = run_cells(
                part_b(CELL_1, CONFIG_CELL, TRAINING_CELL, epochs=A8_EPOCHS + 1)
            )
        self.assertIn("resumed from epoch", printed)
        self.assertEqual(len(ns["result"].train_losses), A8_EPOCHS + 1)

    def test_without_an_a8_checkpoint_it_asks_for_the_epochs(self):
        cells = substitute(
            [cell_containing(code_cells(NOTEBOOK), CELL_1)],
            {'experiment_name="assignment-8"': 'experiment_name="no-such-run"'},
        )
        with colab_faked(self.drive), self.assertRaises(AssertionError) as stopped:
            run_cells(cells)
        self.assertIn("No A8 checkpoint", str(stopped.exception))


class TestPartBWithoutSpaces(PartBStraightThrough, unittest.TestCase):
    """A language written without spaces: A8 rebuilt in characters, scored with spBLEU."""

    no_spaces = True

    @classmethod
    def setUpClass(cls):
        try:  # spBLEU's tokenizer is downloaded on first use
            sacrebleu.corpus_bleu(["a"], [["a"]], tokenize="flores200")
        # Any failure here means offline, not a broken Part B.
        except Exception as error:
            raise unittest.SkipTest(
                f"flores200 tokenizer unavailable: {error}"
            ) from error
        super().setUpClass()

    def test_scores_with_spbleu(self):
        self.assertEqual(self.ns["TOKENIZE"], "flores200")
        self.assertIn("spBLEU (flores200)", self.printed)


if __name__ == "__main__":
    unittest.main()
