"""Lecture 8c, executed from the notebook: the student's heads against the model's own.

The notebook declares ``exercises``, so CI's notebook run skips it: as shipped it stops at
``split_heads`` until a student writes it. These tests run its cells from the notebook file
itself with both answers written in, on the pretrained model, and check that the heads match the
model in every encoder layer and the sharpness table prints. Two more check that the notebook as
shipped stops at the first exercise, and that a ``split_heads`` which reshapes without moving
the head dimension, a likely first attempt, is rejected by the eye-readable case.

The pretrained model lives in Git LFS, which CI does not fetch, so the class skips there.
"""

import os
import unittest
from pathlib import Path

from notebook_cells import COURSE, code_cells, run_cells, substitute
from test_lecture_08b import MODEL, REPO, is_available

NOTEBOOK = COURSE / "lecture-08c-attention-heads.ipynb"
SKIPPED = ("%pip install",)

SPLIT = 'raise NotImplementedError("write split_heads, then run this cell again")'
SHARP = 'raise NotImplementedError("write sharpness, then run this cell again")'
ANSWERS = {
    SPLIT: (
        "pieces, d_model = x.shape\n"
        "    return x.reshape(pieces, n_heads, d_model // n_heads).transpose(0, 1)"
    ),
    SHARP: "return weights.max(dim=-1).values.mean(dim=-1)",
}


def lecture_08c_cells(answers: dict[str, str] | None = None) -> list[tuple[int, str]]:
    cells = [c for c in code_cells(NOTEBOOK) if not any(m in c[1] for m in SKIPPED)]
    return substitute(cells, ANSWERS if answers is None else answers)


@unittest.skipUnless(is_available(MODEL), "pretrained model not fetched (Git LFS)")
class TestLecture08c(unittest.TestCase):
    def setUp(self):
        # As in test_lecture_08b: run from the repository root, or setup() fetches a second copy.
        self.addCleanup(os.chdir, Path.cwd())
        os.chdir(REPO)

    def test_the_heads_are_the_models_in_every_layer(self):
        namespace, printed = run_cells(lecture_08c_cells())
        self.assertIn("split_heads passes its checks.", printed)
        self.assertIn("Your heads are the model's, in every layer.", printed)
        self.assertIn("sharpness passes its checks.", printed)
        self.assertIn("Your prediction: (none written)", printed)
        self.assertEqual(len(namespace["layer_weights"]), 3)
        # The first layer's heads are sharp and their average is not: what the reading claims.
        first = namespace["sharpness"](namespace["layer_weights"][0])
        averaged = namespace["sharpness"](namespace["layer_weights"][0].mean(dim=0))
        self.assertGreater(float(first.min()), 0.85)
        self.assertLess(float(averaged), 0.5)

    def test_as_shipped_it_stops_at_the_first_exercise(self):
        with self.assertRaises(NotImplementedError) as stopped:
            run_cells(lecture_08c_cells(answers={}))
        self.assertIn("write split_heads", str(stopped.exception))

    def test_a_reshape_that_keeps_pieces_first_is_rejected(self):
        careless = {**ANSWERS, SPLIT: "return x.reshape(n_heads, x.shape[0], -1)"}
        with self.assertRaises(AssertionError) as stopped:
            run_cells(lecture_08c_cells(answers=careless))
        self.assertIn("head 1 should be columns 0 and 1", str(stopped.exception))

    def test_a_sharpness_over_columns_is_rejected(self):
        careless = {**ANSWERS, SHARP: "return weights.max(dim=-2).values.mean(dim=-1)"}
        with self.assertRaises(AssertionError) as stopped:
            run_cells(lecture_08c_cells(answers=careless))
        self.assertIn("not in each column", str(stopped.exception))


if __name__ == "__main__":
    unittest.main()
