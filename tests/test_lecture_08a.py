"""Lecture 8a (the Assignment 8 kickoff), executed from the notebook: Colab faked, toy size.

The notebook declares ``colab``, so CI's notebook run skips it: it mounts Drive and needs a GPU.
These tests run its cells from the notebook file itself on a toy corpus written where Step 0
looks for it, with the model shrunk, two epochs, split sizes a toy corpus can meet, a prediction
and a first logged loss filled in. The install cell is left out.

What is checked is the path: the nine split files on Drive, a training run, the loss curves read
back from the checkpoint, the test set scored, and the write-up block. Further tests check that
the notebook stops without a prediction, without a first logged loss, or when the input files
are not where SRC_FILE and TGT_FILE say, and that the cells after training run in a later
session that skips the training cell.
"""

import tempfile
import unittest
from pathlib import Path

from notebook_cells import COURSE, code_cells, colab_faked, run_cells, substitute
from test_a9_path import toy_corpus
from test_lecture_09_part_b import TOY_SIZE

NOTEBOOK = COURSE / "lecture-08a-a8-kickoff.ipynb"
SKIPPED = ("%pip install",)

TOY_RUN = {
    **TOY_SIZE,
    "VAL_SIZE    = 2_000": "VAL_SIZE    = 50",
    "TEST_SIZE   = 2_000": "TEST_SIZE   = 50",
    "EPOCHS      = 35": "EPOCHS      = 2",
    "plt.show()": "plt.close()",
}
PREDICTED = {'PREDICTION = ""': 'PREDICTION = "still falling: the corpus is small"'}
FIRST_LOGGED = {"FIRST_LOSS = None": "FIRST_LOSS = 9.5"}
TRAINING_CELL = "result = train_model("


def kickoff_cells(
    predicted: bool = True, first_logged: bool = True
) -> list[tuple[int, str]]:
    """Every code cell but the install, at toy size, with the student's two entries unless told."""
    cells = [c for c in code_cells(NOTEBOOK) if not any(m in c[1] for m in SKIPPED)]
    swaps = {
        **TOY_RUN,
        **(PREDICTED if predicted else {}),
        **(FIRST_LOGGED if first_logged else {}),
    }
    return substitute(cells, swaps)


def write_corpus(drive: Path) -> None:
    """The two Assignment 5 files, where Step 0's default paths point."""
    corpus = toy_corpus(1200, seed=479)
    folder = drive / "CS479"
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "cleaned.en.txt").write_text(
        "\n".join(corpus.src) + "\n", encoding="utf-8"
    )
    (folder / "cleaned.xx.txt").write_text(
        "\n".join(corpus.tgt) + "\n", encoding="utf-8"
    )


class TestLecture08a(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.drive = Path(cls.tmp.name)
        cls.out_dir = cls.drive / "CS479" / "assignment8"
        write_corpus(cls.drive)
        with colab_faked(cls.drive):
            cls.ns, cls.printed = run_cells(kickoff_cells())

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_writes_the_split_the_handout_asks_for(self):
        for split in ("train", "val", "test"):
            for ext in ("tsv", "src", "tgt"):
                self.assertTrue(
                    (self.out_dir / f"{split}.{ext}").exists(), f"{split}.{ext}"
                )
        test_lines = (
            (self.out_dir / "test.src").read_text(encoding="utf-8").splitlines()
        )
        self.assertEqual(len(test_lines), 50)

    def test_plots_training_against_validation_from_the_checkpoint(self):
        self.assertTrue((self.out_dir / "loss-curves.png").exists())
        self.assertEqual(len(self.ns["train_losses"]), 2)
        self.assertEqual(len(self.ns["val_losses"]), 2)
        self.assertIn(
            "Your prediction: still falling: the corpus is small", self.printed
        )

    def test_scores_the_test_set_and_prints_the_write_up(self):
        hyp = (self.out_dir / "test.hyp").read_text(encoding="utf-8").splitlines()
        self.assertEqual(len(hyp), 50)
        self.assertIn("FOR YOUR WRITE-UP", self.printed)
        self.assertIn("Decoder:  greedy", self.printed)
        self.assertIn("first logged loss 9.500", self.printed)
        self.assertIn("What limits the output:", self.printed)

    def test_the_cells_after_training_run_in_a_later_session(self):
        """A new session runs every cell but the training cell, as the notebook says to."""
        cells = [c for c in kickoff_cells() if TRAINING_CELL not in c[1]]
        with colab_faked(self.drive):
            _, printed = run_cells(cells)
        self.assertIn("FOR YOUR WRITE-UP", printed)
        self.assertIn("2 epochs. Validation loss was lowest at epoch", printed)

    def test_without_a_prediction_it_stops(self):
        with colab_faked(self.drive), self.assertRaises(AssertionError) as stopped:
            run_cells(kickoff_cells(predicted=False))
        self.assertIn("Write your prediction", str(stopped.exception))

    def test_without_a_first_logged_loss_it_stops(self):
        cells = [
            c for c in kickoff_cells(first_logged=False) if TRAINING_CELL not in c[1]
        ]
        with colab_faked(self.drive), self.assertRaises(AssertionError) as stopped:
            run_cells(cells)
        self.assertIn("Copy your first logged loss", str(stopped.exception))

    def test_missing_input_files_say_which_lines_to_edit(self):
        cells = substitute(
            kickoff_cells(), {'"cleaned.en.txt"': '"no-such-file.en.txt"'}
        )
        with colab_faked(self.drive), self.assertRaises(SystemExit) as stopped:
            run_cells(cells)
        self.assertIn("Edit SRC_FILE and TGT_FILE", str(stopped.exception))


if __name__ == "__main__":
    unittest.main()
