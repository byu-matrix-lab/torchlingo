"""Lecture 13 (Assignment 13), executed from the notebook: Colab faked, toy size.

The notebook declares ``colab``, so CI's notebook run skips it. These tests run its cells from the
notebook file itself, starting from what Lecture 9's Part B leaves on Drive, which the fixture
produces by running Part B's own cells (see ``test_lecture_09_part_b``). The install cell and the
COMET cell are left out: one is Colab's, and the other installs a package CI does not have.

Two cells are exercises: ``reverse`` and ``combine`` are the student's, and raise
``NotImplementedError`` until written. The tests write them in, one straightforward way each, and
a prediction, so what is tested is that the notebook's checks accept a right answer and that the
path after them works; further tests check that the notebook as shipped stops at the first
exercise, that it stops without a prediction or without ``mono.txt``, that the epochs come from
A9's checkpoint, and that the write-up runs from Drive alone after a dropped session.

What else changes in the cells is only the epochs and the model's size. What is checked is the
path: the split reversed, the monolingual text filtered against the split, the back-translations
written, the combined training set, and both systems scored on the same test set.
"""

import tempfile
import unittest
from pathlib import Path

import pandas as pd
from notebook_cells import COURSE, code_cells, colab_faked, run_cells, substitute
from test_a9_path import PIECE_MARK, toy_corpus
from test_lecture_09_part_b import (
    A8_EPOCHS,
    CELL_1,
    CONFIG_CELL,
    SCORING_CELL,
    TOY_SIZE,
    TRAINING_CELL,
    part_b,
    write_a8_and_part_a,
)

NOTEBOOK = COURSE / "lecture-13-back-translation.ipynb"
SKIPPED = ('%pip install --quiet "torchlingo', "comet-score")

# A student's answers to the two exercises, in place of the line that raises.
ANSWERS = {
    'raise NotImplementedError("write reverse, then run this cell again")': (
        'return frame.rename(columns={"src": "tgt", "tgt": "src"})[["src", "tgt"]]'
    ),
    'raise NotImplementedError("write combine, then run this cell again")': (
        'new = pd.DataFrame({"src": list(back), "tgt": list(mono)})\n'
        '    new = new[(new.src.str.strip() != "") & ~new.src.isin(held_out)]\n'
        "    return pd.concat([original, new], ignore_index=True)"
    ),
}
PREDICTED = {'PREDICTION = ""': 'PREDICTION = "back-translation helps a little"'}
READ_EPOCHS = 'EPOCHS = len(a9_state["train_losses"])'
SETUP_CELL = "def train("
MONO_CELL = 'MONO_FILE = A8_DIR / "mono.txt"'
PREDICTION_CELL = 'PREDICTION = ""'
WRITE_UP_CELL = "=== For the A13 write-up ==="


def lecture_13_cells(
    answered: bool = True, predicted: bool = True, one_epoch: bool = True
) -> list[tuple[int, str]]:
    """Every code cell but the install and COMET, at toy size and, unless told not to, one epoch."""
    cells = [c for c in code_cells(NOTEBOOK) if not any(m in c[1] for m in SKIPPED)]
    swaps = {
        **TOY_SIZE,
        **({READ_EPOCHS: "EPOCHS = 1"} if one_epoch else {}),
        **(ANSWERS if answered else {}),
        **(PREDICTED if predicted else {}),
    }
    return substitute(cells, swaps)


def cells_with(cells: list[tuple[int, str]], *markers: str) -> list[tuple[int, str]]:
    """The cells containing each marker, in notebook order."""
    return sorted(next(c for c in cells if m in c[1]) for m in markers)


class TestLecture13(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.drive = Path(cls.tmp.name)
        cls.a8_dir = cls.drive / "CS479" / "assignment8"
        cls.n_test = write_a8_and_part_a(cls.a8_dir, no_spaces=False)
        with colab_faked(cls.drive):
            run_cells(part_b(CELL_1, CONFIG_CELL, TRAINING_CELL, SCORING_CELL))

        # The student's monolingual file: new sentences, plus a repeat, a blank line and a
        # sentence from the test set, which the notebook must all leave out.
        cls.train = pd.read_csv(cls.a8_dir / "train.tsv", sep="\t", dtype=str)
        cls.test = pd.read_csv(cls.a8_dir / "test.tsv", sep="\t", dtype=str)
        in_split = set(cls.train.tgt) | set(cls.test.tgt)
        new = [s for s in toy_corpus(2000, seed=7).tgt if s not in in_split]
        cls.mono_lines = new + [new[0], "", cls.test.tgt.iloc[0]]
        (cls.a8_dir / "mono.txt").write_text(
            "\n".join(cls.mono_lines) + "\n", encoding="utf-8"
        )

        with colab_faked(cls.drive):
            cls.ns, cls.printed = run_cells(lecture_13_cells())

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_monolingual_text_is_new_to_the_split(self):
        used = (self.a8_dir / "mono-used.txt").read_text(encoding="utf-8").splitlines()
        self.assertEqual(len(used), len(self.train), "as many as the training pairs")
        self.assertEqual(len(used), len(set(used)), "no repeats")
        self.assertNotIn(self.test.tgt.iloc[0], used)
        self.assertNotIn("", used)

    def test_reverses_the_split(self):
        reversed_train = pd.read_csv(
            self.a8_dir / "train-reversed.tsv", sep="\t", dtype=str
        )
        self.assertEqual(list(reversed_train.src), list(self.train.tgt))
        self.assertEqual(list(reversed_train.tgt), list(self.train.src))
        self.assertTrue((self.a8_dir / "best-a13-reverse" / "model_best.pt").exists())

    def test_back_translates_every_sentence_into_text(self):
        back = (
            (self.a8_dir / "mono-used.bt.en")
            .read_text(encoding="utf-8")
            .split("\n")[:-1]
        )
        self.assertEqual(len(back), len(self.train))
        self.assertFalse([b for b in back if PIECE_MARK in b], "decoded, not pieces")

    def test_trains_on_original_plus_synthetic_pairs(self):
        combined = pd.read_csv(
            self.a8_dir / "train-a13.tsv", sep="\t", dtype=str, keep_default_na=False
        )
        self.assertEqual(list(combined.tgt[: len(self.train)]), list(self.train.tgt))
        used = (self.a8_dir / "mono-used.txt").read_text(encoding="utf-8").splitlines()
        self.assertTrue(set(combined.tgt[len(self.train) :]) <= set(used))
        self.assertTrue((self.a8_dir / "best-a13" / "model_best.pt").exists())

    def test_scores_both_systems_on_the_same_test_set(self):
        hyp = (self.a8_dir / "test-a13.hyp").read_text(encoding="utf-8").splitlines()
        self.assertEqual(len(hyp), self.n_test)
        self.assertEqual(len(self.ns["hyp_a9"]), self.n_test)
        for name in ("test.src.txt", "test.ref.txt"):
            lines = (self.a8_dir / name).read_text(encoding="utf-8").splitlines()
            self.assertEqual(len(lines), self.n_test, name)

    def test_prints_the_write_up_block(self):
        self.assertIn("=== For the A13 write-up ===", self.printed)
        self.assertIn("A9 BLEU:", self.printed)
        self.assertIn(
            "Prediction, written before training: back-translation helps a little",
            self.printed,
        )

    def test_the_write_up_runs_after_a_dropped_session(self):
        """A new session runs the setup cell, then the write-up: nothing else in memory."""
        with colab_faked(self.drive):
            _, printed = run_cells(
                cells_with(lecture_13_cells(), SETUP_CELL, WRITE_UP_CELL)
            )
        self.assertIn("A13 BLEU:", printed)
        self.assertIn("Back-translator (your language to English) BLEU:", printed)
        self.assertIn("back-translation helps a little", printed)

    def test_epochs_come_from_a9s_checkpoint(self):
        with colab_faked(self.drive):
            ns, _ = run_cells(cells_with(lecture_13_cells(one_epoch=False), SETUP_CELL))
        self.assertEqual(ns["EPOCHS"], A8_EPOCHS, "A9 trained as many epochs as A8")

    def test_without_a_prediction_it_stops(self):
        cells = cells_with(
            lecture_13_cells(predicted=False), SETUP_CELL, PREDICTION_CELL
        )
        with colab_faked(self.drive), self.assertRaises(AssertionError) as stopped:
            run_cells(cells)
        self.assertIn("Write your prediction", str(stopped.exception))

    def test_without_mono_txt_it_says_what_to_do(self):
        cells = substitute(
            cells_with(lecture_13_cells(), SETUP_CELL, MONO_CELL),
            {'A8_DIR / "mono.txt"': 'A8_DIR / "no-such-mono.txt"'},
        )
        with colab_faked(self.drive), self.assertRaises(SystemExit) as stopped:
            run_cells(cells)
        self.assertIn(
            "Put a file of sentences in your language", str(stopped.exception)
        )

    def test_as_shipped_it_stops_at_the_first_exercise(self):
        with colab_faked(self.drive), self.assertRaises(NotImplementedError) as stopped:
            run_cells(lecture_13_cells(answered=False))
        self.assertIn("write reverse", str(stopped.exception))

    def test_combine_check_rejects_an_answer_that_keeps_held_out_sentences(self):
        """The toy case fails a combine that forgets held_out, with the message to fix it."""
        cells = lecture_13_cells()
        setup_cell = next(c for c in cells if SETUP_CELL in c[1])
        combine_cell = next(c for c in cells if "def combine(" in c[1])
        forgetful = combine_cell[1].replace(" & ~new.src.isin(held_out)", "")
        with colab_faked(self.drive), self.assertRaises(AssertionError) as stopped:
            run_cells([setup_cell, (combine_cell[0], forgetful)])
        self.assertIn("combine is not right yet", str(stopped.exception))


if __name__ == "__main__":
    unittest.main()
