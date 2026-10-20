"""Lecture 11 (Assignment 11), executed from the notebook: Colab faked, COMET stubbed.

The notebook declares ``colab``, ``pip``, ``download`` and ``exercises``, so CI's notebook run skips it, and CI has no
``unbabel-comet``. These tests run its cells from the notebook file itself on a toy A6 folder in a
faked Drive: the loading, the MTEval results file and its join to the sources, the two exercises
and their checks, the scoring loop with BLEU and chrF for real and COMET stubbed, and the report
A11's summary is written from.

The toy MTEval file is saved as a Windows user might leave it, with a byte-order mark and CRLF
line endings; one ranked sentence differs from its source in spacing only, one is a tie, and one is
in no source file, so each of those paths runs.

The stub scores a translation by how many of its words the reference has, or, without a
reference, by length; what is checked is the path, not COMET. Answers to the two exercises and a
prediction are written in, and three further tests check that the notebook as shipped stops at the
first exercise, that it stops without a prediction, and that an agreement count that includes ties
is rejected.
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
}
BOM = chr(0xFEFF)
PREDICTED = {'PREDICTION = ""': 'PREDICTION = "COMET agrees with chrF"'}


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


def lecture_11_cells(answers: dict, predicted: bool = True) -> list[tuple[int, str]]:
    cells = [c for c in code_cells(NOTEBOOK) if not any(m in c[1] for m in SKIPPED)]
    return substitute(
        cells, {**STUB_COMET, **answers, **(PREDICTED if predicted else {})}
    )


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
        self.mteval = a6 / "mteval_results.tsv"
        self.write_mteval("source\tsystem 1\tsystem 2", self.ranked_rows())

    @staticmethod
    def ranked_rows() -> list[str]:
        """Sources 0 to 9 ranked: system 1 better on even, system 2 on odd, 4 a tie; 3 with
        extra spaces; then a sentence in no source file."""
        rows = []
        for i in range(10):
            ranks = "1\t1" if i == 4 else "1\t2" if i % 2 == 0 else "2\t1"
            source = (
                f"el gato  número {i} se sentó "
                if i == 3
                else f"el gato número {i} se sentó"
            )
            rows.append(f"{source}\t{ranks}")
        return rows + ["una frase que no está\t2\t1"]

    def write_mteval(self, header: str, rows: list[str]) -> None:
        # As saved on Windows: a byte-order mark, CRLF, and no final newline.
        text = BOM + "\r\n".join([header, *rows])
        self.mteval.write_bytes(text.encode("utf-8"))

    def test_runs_to_the_report(self):
        with colab_faked(self.drive):
            namespace, printed = run_cells(lecture_11_cells(ANSWERS))
        self.assertIn("your files are aligned", printed)
        self.assertIn("11 sentences you ranked; 10 found in source.txt", printed)
        self.assertIn("not found: una frase que no está", printed)
        self.assertIn("1 of 11 ranked as a tie", printed)
        self.assertEqual(
            namespace["HUMAN_AVERAGE_RANK"], {"system 1": 17 / 11, "system 2": 15 / 11}
        )
        self.assertIn("=== For the A11 summary ===", printed)
        self.assertIn("COMET and chrF:", printed)
        self.assertIn("human rank", printed)
        # The stub COMET prefers system 2 on sources 0, 3, 6 and 9, system 1 on the rest; the
        # ranks prefer system 1 on 0, 2, 6 and 8 and system 2 on the odd ones: of the nine that
        # are not a tie, they agree on 2, 3, 8 and 9.
        self.assertIn("1 you ranked as a tie are not counted", printed)
        self.assertIn("COMET and you: 4 of 9", printed)
        self.assertIn("Your prediction: COMET agrees with chrF", printed)
        for label in ("BLEU", "chrF", "COMET", "COMET-QE-DA"):
            self.assertIn("system 1", namespace["system_scores"][label])

    def test_the_report_survives_a_dropped_session(self):
        with colab_faked(self.drive):
            _, before = run_cells(lecture_11_cells(ANSWERS))
            # A fresh session: Step 1 and the cells above it, Step 3, and the report; no
            # comet_inputs, no prediction, no scoring.
            scoring = ("def comet_inputs", 'PREDICTION = "', "load_from_checkpoint(")
            cells = [
                c
                for c in lecture_11_cells(ANSWERS)
                if not any(s in c[1] for s in scoring)
            ]
            namespace, after = run_cells(cells)
        self.assertNotIn("Scored with COMET", after)
        self.assertEqual(
            before[before.index("=== For the A11 summary ===") :],
            after[after.index("=== For the A11 summary ===") :],
        )
        self.assertIn("Your prediction: COMET agrees with chrF", after)

    def test_a_failed_check_says_what_the_function_returned(self):
        wrong = dict(ANSWERS)
        key = 'raise NotImplementedError("write pairwise_agreement, then run this cell again")'
        wrong[key] = "return 2, 5"
        with colab_faked(self.drive), self.assertRaises(AssertionError) as stopped:
            run_cells(lecture_11_cells(wrong))
        self.assertIn("got (2, 5)", str(stopped.exception))

    def test_system_names_must_match_the_mteval_header(self):
        self.write_mteval("source\tmicrosoft\tnllb", self.ranked_rows())
        with colab_faked(self.drive), self.assertRaises(SystemExit) as stopped:
            run_cells(lecture_11_cells(ANSWERS))
        self.assertIn("['source', 'microsoft', 'nllb']", str(stopped.exception))
        self.assertIn("use the header's two names as its keys", str(stopped.exception))

    def test_a_ranks_file_matching_no_source_stops(self):
        self.write_mteval("source\tsystem 1\tsystem 2", ["something else\t1\t2"])
        with colab_faked(self.drive), self.assertRaises(SystemExit) as stopped:
            run_cells(lecture_11_cells(ANSWERS))
        self.assertIn(
            "None of the sentences you ranked is in source.txt", str(stopped.exception)
        )

    def test_without_a_prediction_it_stops_before_scoring(self):
        with colab_faked(self.drive), self.assertRaises(AssertionError) as stopped:
            run_cells(lecture_11_cells(ANSWERS, predicted=False))
        self.assertIn("Write your prediction", str(stopped.exception))

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
