"""Lecture 8b, executed from the notebook: the student's attention step against the model's own.

The notebook declares ``exercises``, so CI's notebook run skips it: as shipped it stops at
``attention`` until a student writes it. These tests run its cells from the notebook file itself
with an answer written in, on the pretrained model, and check that the step matches the model's
four heads, the map prints and the prediction is marked. The rest check that the notebook as
shipped stops at the exercise, that it stops at the prediction while it is empty, holds other than
three pieces or names a piece that is not there, and that an answer missing the √d_k scale is rejected by the paper case.

The pretrained model lives in Git LFS, which CI does not fetch, so the class skips there.
"""

import os
import unittest
from pathlib import Path

from notebook_cells import COURSE, code_cells, run_cells, substitute

NOTEBOOK = COURSE / "lecture-08b-cross-attention.ipynb"
REPO = Path(__file__).resolve().parents[1]
MODEL = REPO / "data" / "pretrained" / "model.pt"
SKIPPED = ("%pip install",)

RAISE = 'raise NotImplementedError("write attention, then run this cell again")'
ANSWER = {
    RAISE: (
        "weights = torch.softmax(q @ k.T / math.sqrt(k.shape[-1]), dim=-1)\n"
        "    return weights, weights @ v"
    ),
}


def is_available(path: Path) -> bool:
    """True if the file exists and is not a Git LFS pointer."""
    if not path.exists():
        return False
    with path.open("rb") as handle:
        return not handle.read(23).startswith(b"version https://git-lfs")


PREDICTED = {
    "PREDICTION = {}": (
        'PREDICTION = {"▁nuestra": "▁our", "▁Esta": "▁is", "▁públ": "▁public"}'
    )
}


def lecture_08b_cells(
    answer: dict[str, str] | None = None, predicted: bool = True
) -> list[tuple[int, str]]:
    cells = [c for c in code_cells(NOTEBOOK) if not any(m in c[1] for m in SKIPPED)]
    cells = substitute(cells, ANSWER if answer is None else answer)
    return substitute(cells, PREDICTED) if predicted else cells


@unittest.skipUnless(is_available(MODEL), "pretrained model not fetched (Git LFS)")
class TestLecture08b(unittest.TestCase):
    def setUp(self):
        # The notebook reads data/pretrained relative to where it runs, and setup() downloads
        # whatever is missing there: run from the repository root, or it fetches a second copy.
        self.addCleanup(os.chdir, Path.cwd())
        os.chdir(REPO)

    def test_the_step_is_the_models_in_every_head(self):
        namespace, printed = run_cells(lecture_08b_cells())
        self.assertIn("attention passes its checks.", printed)
        self.assertIn("Your attention step is the model's, in all four heads.", printed)
        self.assertIn("What missed: (none written)", printed)
        self.assertEqual(len(namespace["yours"]), 4)

    def test_a_prediction_is_marked(self):
        _, printed = run_cells(lecture_08b_cells())
        self.assertIn(
            "'▁nuestra': you predicted '▁our'; it looked most at '▁our' (right)",
            printed,
        )
        self.assertIn(
            "'▁Esta': you predicted '▁is'; it looked most at '▁This' (not this time)",
            printed,
        )
        self.assertIn(
            "'▁públ': you predicted '▁public'; it looked most at '▁public' (right)",
            printed,
        )

    def test_as_shipped_it_stops_at_the_exercise(self):
        with self.assertRaises(NotImplementedError) as stopped:
            run_cells(lecture_08b_cells(answer={}))
        self.assertIn("write attention", str(stopped.exception))

    def test_without_a_prediction_it_stops_before_the_map(self):
        with self.assertRaises(AssertionError) as stopped:
            run_cells(lecture_08b_cells(predicted=False))
        self.assertIn("Write your prediction", str(stopped.exception))

    def test_a_prediction_naming_no_output_piece_stops(self):
        cells = substitute(
            lecture_08b_cells(predicted=False),
            {
                "PREDICTION = {}": (
                    'PREDICTION = {"nuestra": "▁our", "▁Esta": "▁This", "▁públ": "▁public"}'
                )
            },
        )
        with self.assertRaises(AssertionError) as stopped:
            run_cells(cells)
        self.assertIn("is not an output piece", str(stopped.exception))

    def test_a_prediction_of_two_pieces_stops(self):
        cells = substitute(
            lecture_08b_cells(predicted=False),
            {"PREDICTION = {}": 'PREDICTION = {"▁nuestra": "▁our", "▁Esta": "▁This"}'},
        )
        with self.assertRaises(AssertionError) as stopped:
            run_cells(cells)
        self.assertIn("Choose three output pieces; you have 2.", str(stopped.exception))

    def test_an_answer_without_the_scale_is_rejected(self):
        unscaled = {RAISE: ANSWER[RAISE].replace(" / math.sqrt(k.shape[-1])", "")}
        with self.assertRaises(AssertionError) as stopped:
            run_cells(lecture_08b_cells(answer=unscaled))
        self.assertIn("divide the scores by √d_k", str(stopped.exception))


if __name__ == "__main__":
    unittest.main()
