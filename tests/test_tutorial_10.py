"""Tutorial 10, executed from the notebook: IBM Model 2 against Model 1.

The notebook declares ``exercises``, so CI's notebook run skips it: as shipped it stops at the
first prediction until a student writes one, and then at the second. These tests run its cells
from the notebook file itself with both predictions written in, and check what its markdown says of the results, so the prose
cannot drift from them: Model 1 cannot tell the two ``le`` apart and splits the fox and the owl
evenly; Model 2 learns a diagonal for six-word pairs and a swap for four-word ones, settles the
fox and gets the owl wrong, by learning that *chouette* means *grey*; and two more owls fix
Model 1 but not Model 2.
"""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from notebook_cells import code_cells, run_cells, substitute

NOTEBOOK = (
    Path(__file__).resolve().parent.parent
    / "docs"
    / "docs"
    / "tutorials"
    / "10-ibm-model-2.ipynb"
)
SKIPPED = ("%pip install",)
PREDICTED = {
    'PREDICTION = ""': 'PREDICTION = "No: the diagonal puts chouette on grey."'
}
PREDICTED_MORE = {
    'PREDICTION_MORE = ""': 'PREDICTION_MORE = "Model 1 only: the positions are certain."'
}

FOX = ("the young fox sees the cat", "le jeune renard voit le chat")
OWL = ("the grey owl sees the cat", "la chouette grise voit le chat")


def tutorial_10_cells(predicted: bool = True, predicted_more: bool = True):
    cells = [c for c in code_cells(NOTEBOOK) if not any(m in c[1] for m in SKIPPED)]
    if predicted:
        cells = substitute(cells, PREDICTED)
    if predicted_more:
        cells = substitute(cells, PREDICTED_MORE)
    return cells


class TestTutorial10(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.ns, cls.printed = run_cells(tutorial_10_cells())

    def rows(self, t, pair, a=None):
        """One pair's alignment table, with its English and French words."""
        english, french = pair[0].split(), pair[1].split()
        table = self.ns["alignment"](t, english, french, a)
        return table, english, french

    def share(self, t, pair, f, e, a=None):
        """The share of French word ``f`` that goes to English word ``e`` (first of each)."""
        table, english, french = self.rows(t, pair, a)
        return table[french.index(f), english.index(e)].item()

    def test_as_shipped_it_stops_at_the_prediction(self):
        with self.assertRaises(AssertionError) as stopped:
            run_cells(tutorial_10_cells(predicted=False))
        self.assertIn("Write your prediction in PREDICTION,", str(stopped.exception))

    def test_then_it_stops_before_the_two_more_owls(self):
        with self.assertRaises(AssertionError) as stopped:
            run_cells(tutorial_10_cells(predicted_more=False))
        self.assertIn(
            "Write your prediction in PREDICTION_MORE", str(stopped.exception)
        )

    def test_fifteen_pairs_five_of_six_words_four_in_english_order(self):
        """Step 3: 'Of the five six-word pairs, four keep English order'."""
        pairs = self.ns["PAIRS"]
        self.assertEqual(len(pairs), 15)
        six = [p for p in pairs if len(p[0].split()) == 6 == len(p[1].split())]
        self.assertEqual(len(six), 5)
        reordered = [
            p for p in six if "grise" in p[1] or "noir" in p[1] or "blanc" in p[1]
        ]
        self.assertEqual(reordered, [OWL])
        four = [p for p in pairs if len(p[0].split()) == 4 == len(p[1].split())]
        self.assertEqual({p[1].split()[2] for p in four}, {"noir", "blanc"})

    def test_model_1_gives_both_le_the_same_row(self):
        table, _, french = self.rows(
            self.ns["t1"], ("the cat sees the dog", "le chat voit le chien")
        )
        self.assertTrue(table[0].allclose(table[3]))
        self.assertAlmostEqual(table[0, 0].item(), table[0, 3].item())

    def test_model_1_splits_the_fox_and_the_owl_evenly(self):
        t1 = self.ns["t1"]
        for pair, (f, e1, e2) in [
            (FOX, ("jeune", "young", "fox")),
            (OWL, ("chouette", "grey", "owl")),
        ]:
            with self.subTest(f=f):
                self.assertAlmostEqual(self.share(t1, pair, f, e1), 0.5, places=6)
                self.assertAlmostEqual(self.share(t1, pair, f, e2), 0.5, places=6)

    def test_model_2_learns_a_diagonal_at_six_words_and_a_swap_at_four(self):
        a = self.ns["a2"]
        for i in range(6):
            self.assertGreater(a[i, i, 6, 6], 0.98)
        self.assertGreater(a[2, 1, 4, 4], 0.99)  # French word 2 from English word 3
        self.assertGreater(a[1, 2, 4, 4], 0.99)

    def test_model_2_settles_both_le_and_the_fox(self):
        t2, a2 = self.ns["t2"], self.ns["a2"]
        table, _, _ = self.rows(
            t2, ("the cat sees the dog", "le chat voit le chien"), a2
        )
        self.assertGreater(table[0, 0].item(), 0.95)  # the first le on the first the
        self.assertGreater(table[3, 3].item(), 0.95)  # the second on the second
        self.assertGreater(self.share(t2, FOX, "jeune", "young", a2), 0.99)
        self.assertGreater(self.share(t2, FOX, "renard", "fox", a2), 0.99)

    def test_model_2_gets_the_owl_wrong_and_believes_chouette_means_grey(self):
        t2, a2 = self.ns["t2"], self.ns["a2"]
        self.assertGreater(self.share(t2, OWL, "chouette", "grey", a2), 0.99)
        self.assertGreater(self.share(t2, OWL, "grise", "owl", a2), 0.99)
        # Step 4: 'Model 2 now believes that chouette means grey', from its t alone.
        self.assertGreater(self.share(t2, OWL, "chouette", "grey"), 0.99)
        self.assertGreater(t2["chouette", "grey"], 0.99)

    def test_two_more_owls_fix_model_1_and_not_model_2(self):
        """Step 5: Model 1 right; Model 2 still wrong though its own t knows the word."""
        t1, t2, a2 = self.ns["t1_more"], self.ns["t2_more"], self.ns["a2_more"]
        self.assertGreater(self.share(t1, OWL, "chouette", "owl"), 0.85)
        self.assertGreater(self.share(t1, OWL, "grise", "grey"), 0.99)
        self.assertGreater(self.share(t2, OWL, "chouette", "grey", a2), 0.99)
        self.assertGreater(t2["chouette", "owl"], t2["chouette", "grey"])
        self.assertGreater(a2[1, 1, 6, 6], 0.99)

    def test_both_predictions_are_printed_back(self):
        self.assertIn(
            "You predicted: No: the diagonal puts chouette on grey.", self.printed
        )
        self.assertIn(
            "You predicted: Model 1 only: the positions are certain.", self.printed
        )


if __name__ == "__main__":
    unittest.main()
