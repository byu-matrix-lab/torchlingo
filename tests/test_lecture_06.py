"""Lecture 6's two notebooks, executed from the notebook files with the student's parts written in.

The activity declares ``exercises``: as shipped it stops at its first prediction, and then at
``bleu_from_parts`` until a student writes it, so CI's notebook run skips it. The homework
declares ``colab``: it uploads the student's own files through the Colab picker. These tests run
both from the notebook files themselves, with an answer, predictions, files, translations and a
ranking written in, and check that each stop stops with a message that says what to do.
"""

import contextlib
import os
import tempfile
import unittest
from pathlib import Path

from notebook_cells import COURSE, code_cells, run_cells, substitute

ACTIVITY = COURSE / "lecture-06-mt-evaluation.ipynb"
HOMEWORK = COURSE / "lecture-06-mt-evaluation-homework.ipynb"

RAISE = 'raise NotImplementedError("write bleu_from_parts, then run this cell again")'
ANSWER = (
    "if min(precisions) == 0:\n"
    "        return 0.0\n"
    "    return bp * math.exp(sum(math.log(p / 100) for p in precisions) / 4) * 100"
)
PART_1 = 'PREDICTION = ""   # <- the BLEU score, 0 to 100'
PART_2 = ('YOU_PREFER = ""', 'BLEU_PREFERS = ""')
PART_3 = 'PREDICTION = ""   # <- BLEU for System B'
PREDICTIONS = {
    PART_1: 'PREDICTION = "about 70"   # <- the BLEU score, 0 to 100',
    PART_2[0]: 'YOU_PREFER = "a"',
    PART_2[1]: 'BLEU_PREFERS = "B"',
    PART_3: 'PREDICTION = "near zero"   # <- BLEU for System B',
}


def activity_cells(
    answer: str | None = ANSWER, unwritten: tuple[str, ...] = ()
) -> list[tuple[int, str]]:
    """Every code cell but the install, with ``answer`` and every prediction not ``unwritten``."""
    cells = [c for c in code_cells(ACTIVITY) if "%pip" not in c[1]]
    swaps = {k: v for k, v in PREDICTIONS.items() if k not in unwritten}
    if answer is not None:
        swaps[RAISE] = answer
    return substitute(cells, swaps)


class TestActivity(unittest.TestCase):
    def test_runs_end_to_end_with_the_answer_written_in(self):
        namespace, printed = run_cells(activity_cells())
        self.assertIn("your prediction: about 70", printed)
        self.assertIn("bleu_from_parts passes its checks.", printed)
        self.assertIn("you would rank higher: System A", printed)
        self.assertIn("you predicted BLEU prefers: System B", printed)
        self.assertIn("your prediction for three words kept: near zero", printed)
        # The table's own column beside BLEU's: the five rows agree.
        rows = [line.split() for line in printed.splitlines()]
        rows = [r for r in rows if len(r) == 5 and r[0] in {"3", "5", "8", "12", "100"}]
        self.assertEqual(len(rows), 5)
        for row in rows:
            self.assertEqual(row[1], row[2], row)

    def test_as_shipped_it_stops_at_the_first_prediction(self):
        with self.assertRaises(AssertionError) as stopped:
            run_cells(activity_cells(answer=None, unwritten=tuple(PREDICTIONS)))
        self.assertIn("Write your prediction in PREDICTION", str(stopped.exception))

    def test_with_predictions_it_stops_at_the_exercise(self):
        with self.assertRaises(NotImplementedError) as stopped:
            run_cells(activity_cells(answer=None))
        self.assertIn("write bleu_from_parts", str(stopped.exception))

    def test_part_2_stops_until_both_choices_are_written(self):
        for missing in PART_2:
            with self.subTest(missing=missing):
                with self.assertRaises(AssertionError) as stopped:
                    run_cells(activity_cells(unwritten=(missing,)))
                self.assertIn('Write "A" or "B"', str(stopped.exception))

    def test_part_3_stops_until_its_prediction_is_written(self):
        with self.assertRaises(AssertionError) as stopped:
            run_cells(activity_cells(unwritten=(PART_3,)))
        self.assertIn("Write your prediction in PREDICTION", str(stopped.exception))

    def test_an_answer_without_the_penalty_is_rejected(self):
        without_bp = ANSWER.replace("return bp * ", "return ")
        with self.assertRaises(AssertionError) as stopped:
            run_cells(activity_cells(answer=without_bp))
        message = str(stopped.exception)
        self.assertIn("the penalty multiplies", message)
        self.assertIn("got 35.35", message)

    def test_an_answer_that_logs_a_zero_is_rejected(self):
        unguarded = ANSWER.split("\n", 2)[2].strip()
        with self.assertRaises(AssertionError) as stopped:
            run_cells(activity_cells(answer=unguarded))
        self.assertIn("a precision of 0 must give BLEU 0", str(stopped.exception))


# The student's corpus: thirty distinct aligned pairs, and an MT system that gets most of a
# sentence right.
SOURCES = [f"source sentence number {i} has a few more words in it" for i in range(30)]
TARGETS = [f"target sentence number {i} has a few more words in it" for i in range(30)]


def mt(reference: str) -> str:
    return reference.replace("more", "other")


class TestHomework(unittest.TestCase):
    def setUp(self):
        scratch = tempfile.TemporaryDirectory()
        self.addCleanup(scratch.cleanup)
        self.dir = Path(scratch.name)
        (self.dir / "a5.en").write_text("\n".join(SOURCES) + "\n", encoding="utf-8")
        (self.dir / "a5.xx").write_text("\n".join(TARGETS) + "\n", encoding="utf-8")

    def homework_cells(
        self,
        files: bool = True,
        ranking: str | None = "[3, 1, 2, 4, 5, 6, 7, 8, 9, 10]",
    ) -> list[tuple[int, str]]:
        """Every code cell but the install and the picker, with the student's parts filled in.

        The translations depend on which sentences the seeded sample picked, so they are
        written in from a first run of the cells up to the sampling cell.
        """
        cells = [
            c
            for c in code_cells(HOMEWORK)
            if "%pip" not in c[1] and "files.upload()" not in c[1]
        ]
        named = {
            "SRC_FILE = ''": "SRC_FILE = 'a5.en'",
            "TGT_FILE = ''": "TGT_FILE = 'a5.xx'",
        }
        cells = substitute(cells, named) if files else cells
        if not files:
            return cells
        sample_at = next(i for i, src in cells if "random.sample(" in src)
        with self.in_dir():
            namespace, _ = run_cells([c for c in cells if c[0] <= sample_at])
        pasted = "\n".join(mt(r) for r in namespace["sample_ref"])
        swaps = {'"""\n\n"""': f'"""\n{pasted}\n"""'}
        if ranking is not None:
            swaps["MY_RANKING = []"] = f"MY_RANKING = {ranking}"
        return substitute(cells, swaps)

    @contextlib.contextmanager
    def in_dir(self):
        previous = Path.cwd()
        os.chdir(self.dir)
        try:
            yield
        finally:
            os.chdir(previous)

    def test_runs_end_to_end(self):
        with self.in_dir():
            _, printed = run_cells(self.homework_cells())
        self.assertIn("30 aligned pairs", printed)
        self.assertIn("your ranking, best first: [3, 1, 2", printed)
        self.assertIn("BLEU", printed)
        self.assertIn("TER ", printed)
        self.assertIn("chrF rank", printed)

    def test_no_score_prints_before_the_ranking(self):
        """The finding this notebook was reordered for: rank first, then see the numbers."""
        cells = code_cells(HOMEWORK)
        ranking = next(i for i, src in cells if "MY_RANKING = []" in src)
        scoring = [i for i, src in cells if "compute_" in src and "import" not in src]
        self.assertTrue(scoring)
        self.assertLess(ranking, min(scoring))

    def test_empty_file_names_stop_with_a_message(self):
        with self.in_dir(), self.assertRaises(AssertionError) as stopped:
            run_cells(self.homework_cells(files=False))
        message = str(stopped.exception)
        self.assertIn("Set SRC_FILE to your source-language file", message)
        self.assertIn("TGT_FILE", message)

    def test_an_empty_ranking_stops(self):
        with self.in_dir(), self.assertRaises(AssertionError) as stopped:
            run_cells(self.homework_cells(ranking=None))
        self.assertIn("Write your ranking in MY_RANKING", str(stopped.exception))

    def test_a_ranking_missing_a_sentence_is_rejected(self):
        with self.in_dir(), self.assertRaises(AssertionError) as stopped:
            run_cells(self.homework_cells(ranking="[1, 1, 2, 3, 4, 5, 6, 7, 8, 9]"))
        self.assertIn(
            "each of the numbers 1 to 10 exactly once", str(stopped.exception)
        )


if __name__ == "__main__":
    unittest.main()
