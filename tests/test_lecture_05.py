"""Lecture 5 (sentence alignment), executed from the notebook.

The notebook declares ``pip`` and ``exercises``, so neither CI's notebook run nor the student-path
run executes it: one cell installs NLTK, and the ``WHY_MERGE`` answer stops while empty. Until it
declared ``exercises``, the student-path run executed it unanswered and failed at that stop on
every push to ``course``.

These tests run its cells from the notebook file itself, without the install cell (NLTK comes from
the dev extras) and with an answer written in, and check that the shipped notebook stops at the
answer. With no A4 files beside it, the last part runs on the notebook's own sample.
"""

import os
import tempfile
import unittest

from notebook_cells import COURSE, code_cells, run_cells, substitute

NOTEBOOK = COURSE / "lecture-05-sentence-alignment.ipynb"
SKIPPED = ("pip install",)
ANSWERED = {
    'WHY_MERGE = ""': 'WHY_MERGE = "the second sentence was joined to the first, not dropped"',
}


def lecture_05_cells(answered: bool = True) -> list[tuple[int, str]]:
    cells = [c for c in code_cells(NOTEBOOK) if not any(m in c[1] for m in SKIPPED)]
    return substitute(cells, ANSWERED) if answered else cells


class TestLecture05(unittest.TestCase):
    def setUp(self):
        # The A4 cell looks for its files in the working directory; run where there are none.
        self.previous = os.getcwd()
        self.scratch = tempfile.TemporaryDirectory()
        os.chdir(self.scratch.name)

    def tearDown(self):
        os.chdir(self.previous)
        self.scratch.cleanup()

    def test_runs_to_the_end_with_an_answer(self):
        _, printed = run_cells(lecture_05_cells())
        self.assertIn("precision", printed)

    def test_as_shipped_it_stops_at_the_answer(self):
        with self.assertRaises(AssertionError) as stopped:
            run_cells(lecture_05_cells(answered=False))
        self.assertIn("Write your answer in WHY_MERGE", str(stopped.exception))


if __name__ == "__main__":
    unittest.main()
