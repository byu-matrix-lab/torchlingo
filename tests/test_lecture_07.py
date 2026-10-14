"""Lecture 7's toy model, executed from the notebook with Part B's two blanks filled in.

The notebook declares ``exercises``, so CI's notebook run skips it: as shipped, Part B stops
until a student chooses ``HELD_OUT`` and writes ``PREDICTION``. These tests run its cells from
the notebook file itself, in a scratch directory standing in for Drive, with both written in and
the training shortened, and check the path: the two held-out runs, the contrast of the other
kind, the resume after the faked disconnect, and the report. Two more check that the notebook
as shipped stops at the choice, and that a choice without a prediction stops too.

The resume demo keeps its own 20 epochs: it drops in epoch 12, so it cannot be shortened below
that, and on this model it takes seconds.
"""

import tempfile
import unittest
from pathlib import Path

from notebook_cells import COURSE, code_cells, colab_faked, run_cells, substitute

NOTEBOOK = COURSE / "lecture-07-toy-model.ipynb"

SHORTER = {"EPOCHS = 40": "EPOCHS = 3", "plt.show()": "plt.close()"}
CHOSEN = {"HELD_OUT = None": 'HELD_OUT = ("The dog sleeps", "El perro duerme")'}
PREDICTED = {
    'PREDICTION = ""': 'PREDICTION = "Right: every word of it is in the other phrases."'
}


def lecture_07_cells(
    chosen: bool = True, predicted: bool = True
) -> list[tuple[int, str]]:
    """Every code cell but the install, shortened, with the blanks filled in as asked."""
    cells = [c for c in code_cells(NOTEBOOK) if "%pip" not in c[1]]
    swaps = {
        **SHORTER,
        **(CHOSEN if chosen else {}),
        **(PREDICTED if predicted else {}),
    }
    return substitute(cells, swaps)


class TestLecture07(unittest.TestCase):
    def run_notebook(self, **cell_options) -> tuple[dict, str]:
        scratch = tempfile.TemporaryDirectory()
        self.addCleanup(scratch.cleanup)
        with colab_faked(Path(scratch.name)):
            return run_cells(lecture_07_cells(**cell_options))

    def test_runs_end_to_end_with_both_blanks_filled_in(self):
        namespace, printed = self.run_notebook()

        self.assertIn("Holding out 'The dog sleeps'. Prediction recorded.", printed)
        # A phrase that shares its words is contrasted with one that has a word of its own.
        self.assertEqual(namespace["contrast"]["pair"], ("Good night", "Buenas noches"))
        for run in (namespace["mine"], namespace["contrast"]):
            self.assertIsInstance(run["unseen_hyp"], str)
            self.assertLessEqual(run["exact_seen"], 11)

        # The resume demo: dropped in epoch 12, resumed from the last epoch-end save.
        self.assertIn("the session dropped", printed)
        self.assertIn("resumed from epoch 10", printed)
        self.assertEqual(len(namespace["demo"].train_losses), 20)

        self.assertIn("=== Lecture 7 toy model: report ===", printed)
        self.assertIn(
            "Prediction: Right: every word of it is in the other phrases.", printed
        )

    def test_as_shipped_it_stops_at_the_choice(self):
        with self.assertRaises(AssertionError) as stopped:
            self.run_notebook(chosen=False, predicted=False)
        self.assertIn("Set HELD_OUT", str(stopped.exception))

    def test_without_a_prediction_it_stops_before_training(self):
        with self.assertRaises(AssertionError) as stopped:
            self.run_notebook(predicted=False)
        self.assertIn("Write your prediction", str(stopped.exception))


if __name__ == "__main__":
    unittest.main()
