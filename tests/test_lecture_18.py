"""Lecture 18, executed from the notebook: IBM Model 1 by EM, set beside a Transformer's attention.

The notebook declares ``exercises``, so CI's notebook run skips it: as shipped it stops at the
prediction until a student writes one. These tests run its cells from the notebook file itself
with a prediction written in, and check what its markdown says of the results, so the prose
cannot drift from them: one round leaves ``house`` between ``casa`` and ``la``, ten settle it,
``the`` and ``red`` stay split by gender, both alignments put ``el`` on ``the`` while Model 1
keeps a share for ``car``, and attention spreads ``coche`` and ``rojo`` where Model 1 is sure.
Another checks the "your turn" step: adding ``the boy / el niño`` moves ``el`` towards ``the``.

The pretrained model lives in Git LFS, which CI does not fetch, so the class skips there.
"""

import os
import unittest
from pathlib import Path

from notebook_cells import COURSE, code_cells, run_cells, substitute

NOTEBOOK = COURSE / "lecture-18-ibm-model-1.ipynb"
REPO = Path(__file__).resolve().parents[1]
MODEL = REPO / "data" / "pretrained" / "model.pt"
SKIPPED = ("%pip install",)

PREDICTED = {
    'PREDICTION = ""': 'PREDICTION = "casa, but la is in most of the same pairs"'
}
BOY = {
    '    ("a big book", "un libro grande"),\n': (
        '    ("a big book", "un libro grande"),\n    ("the boy", "el niño"),\n'
    )
}


def is_available(path: Path) -> bool:
    """True if the file exists and is not a Git LFS pointer."""
    if not path.exists():
        return False
    with path.open("rb") as handle:
        return not handle.read(23).startswith(b"version https://git-lfs")


def lecture_18_cells(predicted: bool = True, extra: dict | None = None):
    cells = [c for c in code_cells(NOTEBOOK) if not any(m in c[1] for m in SKIPPED)]
    if predicted:
        cells = substitute(cells, PREDICTED)
    return substitute(cells, extra) if extra else cells


def ranked(t, e):
    """Spanish words for English word e, likeliest first, as (word, probability)."""
    return sorted(((s, p) for (s, w), p in t.items() if w == e), key=lambda x: -x[1])


@unittest.skipUnless(is_available(MODEL), "pretrained model not fetched (Git LFS)")
class TestLecture18(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # setup() downloads whatever is missing relative to where it runs: run from the root.
        cwd = Path.cwd()
        os.chdir(REPO)
        try:
            cls.namespace, cls.printed = run_cells(lecture_18_cells())
        finally:
            os.chdir(cwd)

    def test_as_shipped_it_stops_at_the_prediction(self):
        cwd = Path.cwd()
        self.addCleanup(os.chdir, cwd)
        os.chdir(REPO)
        with self.assertRaises(AssertionError) as stopped:
            run_cells(lecture_18_cells(predicted=False))
        self.assertIn("Write your prediction", str(stopped.exception))

    def test_one_round_leaves_house_between_casa_and_la(self):
        line = next(
            x for x in self.printed.splitlines() if x.startswith("round 1  house:")
        )
        words = line.split(":", 1)[1].split()[::2]
        self.assertEqual(words[:2], ["casa", "la"], line)
        self.assertIn("You predicted: casa, but la", self.printed)

    def test_ten_rounds_settle_each_noun_and_leave_gender_split(self):
        t = self.namespace["t"]
        for english, spanish in [
            ("house", "casa"),
            ("door", "puerta"),
            ("car", "coche"),
            ("book", "libro"),
        ]:
            self.assertEqual(ranked(t, english)[0][0], spanish)
        self.assertGreater(ranked(t, "house")[0][1], 0.8)
        self.assertEqual({s for s, _ in ranked(t, "the")[:2]}, {"la", "el"})
        self.assertEqual({s for s, _ in ranked(t, "red")[:2]}, {"roja", "rojo"})

    def test_both_put_el_on_the_and_model_1_keeps_a_share_for_car(self):
        model1, attention = self.namespace["model1"], self.namespace["attention"]
        # Rows el, coche, rojo; columns the, red, car.
        self.assertEqual(int(model1[0].argmax()), 0)
        self.assertEqual(int(attention[0].argmax()), 0)
        self.assertGreater(model1[0, 2].item(), 0.2)
        self.assertGreater(attention[0, 0].item(), model1[0, 0].item())

    def test_attention_spreads_coche_and_rojo_where_model_1_is_sure(self):
        model1, attention = self.namespace["model1"], self.namespace["attention"]
        self.assertGreater(model1[1, 2].item(), 0.9)  # coche on car
        self.assertGreater(model1[2, 1].item(), 0.9)  # rojo on red
        for row in (1, 2):
            self.assertLess(attention[row].max().item(), 0.7)
            self.assertGreater(attention[row, 1:].sum().item(), 0.7)  # on red car

    def test_rows_sum_to_one(self):
        for name in ("model1", "attention"):
            sums = self.namespace[name].sum(dim=1)
            for s in sums.tolist():
                self.assertAlmostEqual(s, 1.0, places=5, msg=name)

    def test_adding_the_boy_moves_el_towards_the(self):
        cwd = Path.cwd()
        self.addCleanup(os.chdir, cwd)
        os.chdir(REPO)
        namespace, _ = run_cells(lecture_18_cells(extra=BOY))
        self.assertGreater(
            namespace["model1"][0, 0].item(),
            self.namespace["model1"][0, 0].item() + 0.05,
        )


if __name__ == "__main__":
    unittest.main()
