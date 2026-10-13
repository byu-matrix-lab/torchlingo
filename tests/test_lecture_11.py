"""Lecture 11 (Assignment 11), executed from the notebook: Colab faked, COMET stubbed.

The notebook declares ``colab`` and ``pip``, so CI's notebook run skips it, and CI has no
``unbabel-comet``. These tests run its cells from the notebook file itself on a toy A6 folder in a
faked Drive: the loading, the two exercises and their checks, the scoring loop with BLEU and chrF
for real and COMET stubbed, and the report A11's summary is written from.

The stub scores a translation by how many of its words the reference has, or, without a
reference, by length; what is checked is the path, not COMET. Answers to the two exercises are
written in, and two further tests check that the notebook as shipped stops at the first, and that
an agreement count that includes ties is rejected.
"""

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from notebook_cells import COURSE, code_cells, colab_faked, run_cells, substitute
from test_lecture_10 import ANSWER as COMET_INPUTS_ANSWER

NOTEBOOK = COURSE / "lecture-11-comet-on-a6.ipynb"
SKIPPED = ("%pip install",)

ANSWERS = {
    'raise NotImplementedError("paste comet_inputs from A10, then run this cell again")': (
        COMET_INPUTS_ANSWER[
            'raise NotImplementedError("write comet_inputs, then run this cell again")'
        ]
    ),
    'raise NotImplementedError("write pairwise_agreement, then run this cell again")': (
        "agree = counted = 0\n"
        "    for x1, x2, y1, y2 in zip(a_1, a_2, b_1, b_2):\n"
        "        if x1 == x2 or y1 == y2:\n"
        "            continue\n"
        "        counted += 1\n"
        "        agree += (x1 > x2) == (y1 > y2)\n"
        "    return agree, counted"
    ),
}
STUB_COMET = {
    "from comet import download_model, load_from_checkpoint": (
        "from test_lecture_11 import download_model, load_from_checkpoint"
    ),
    'HUMAN_AVERAGE_RANK = {"system 1": None, "system 2": None}': (
        'HUMAN_AVERAGE_RANK = {"system 1": 1.4, "system 2": 1.6}'
    ),
}


def download_model(name: str) -> str:
    """Stand-in for COMET's download: the model's name is its path."""
    return name


class _StubModel:
    def predict(self, records, **kwargs):
        def score(r):
            if "ref" in r:
                ref = set(r["ref"].split())
                return len(ref & set(r["mt"].split())) / max(len(ref), 1)
            return len(r["mt"]) / 100

        scores = [score(r) for r in records]
        return SimpleNamespace(scores=scores, system_score=sum(scores) / len(scores))


def load_from_checkpoint(path: str) -> _StubModel:
    """Stand-in for COMET's loader."""
    return _StubModel()


def lecture_11_cells(answers: dict) -> list[tuple[int, str]]:
    cells = [c for c in code_cells(NOTEBOOK) if not any(m in c[1] for m in SKIPPED)]
    return substitute(cells, {**STUB_COMET, **answers})


class TestLecture11(unittest.TestCase):
    def setUp(self):
        scratch = tempfile.TemporaryDirectory()
        self.addCleanup(scratch.cleanup)
        self.drive = Path(scratch.name)
        a6 = self.drive / "CS479" / "assignment6"
        a6.mkdir(parents=True)
        n = 30
        refs = [f"the cat number {i} sat on the mat" for i in range(n)]
        files = {
            "source.txt": [f"el gato número {i} se sentó" for i in range(n)],
            "reference.txt": refs,
            "system1.txt": [
                r if i % 3 else f"a cat {i} sat" for i, r in enumerate(refs)
            ],
            "system2.txt": [f"the cat {i} is on a mat" for i in range(n)],
        }
        for name, lines in files.items():
            (a6 / name).write_text("\n".join(lines) + "\n", encoding="utf-8")

    def test_runs_to_the_report(self):
        with colab_faked(self.drive):
            namespace, printed = run_cells(lecture_11_cells(ANSWERS))
        self.assertIn("your files are aligned", printed)
        self.assertIn("=== For the A11 summary ===", printed)
        self.assertIn("COMET and chrF:", printed)
        self.assertIn("human rank", printed)
        self.assertIn("Your prediction: (none written)", printed)
        for label in ("BLEU", "chrF", "COMET", "COMET-QE-DA"):
            self.assertIn((label, "system 1"), namespace["system_scores"])

    def test_as_shipped_it_stops_at_the_first_exercise(self):
        with colab_faked(self.drive), self.assertRaises(NotImplementedError) as stopped:
            run_cells(lecture_11_cells({}))
        self.assertIn("paste comet_inputs", str(stopped.exception))

    def test_an_agreement_count_that_includes_ties_is_rejected(self):
        careless = dict(ANSWERS)
        key = 'raise NotImplementedError("write pairwise_agreement, then run this cell again")'
        careless[key] = ANSWERS[key].replace(
            "if x1 == x2 or y1 == y2:\n            continue\n        ", ""
        )
        with colab_faked(self.drive), self.assertRaises(AssertionError) as stopped:
            run_cells(lecture_11_cells(careless))
        self.assertIn("2 agreements out of 3", str(stopped.exception))


if __name__ == "__main__":
    unittest.main()
