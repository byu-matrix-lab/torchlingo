"""Lecture 11 (Assignment 11), executed from the notebook: Colab faked, COMET stubbed.

The notebook declares ``colab``, ``pip``, ``download`` and ``exercises``, so CI's notebook run skips it, and CI has no
``unbabel-comet``. These tests run its cells from the notebook file itself on a toy A6 folder in a
faked Drive: the loading, the MTEval results file and its join to the sources, the two exercises
and their checks, the scoring loop with BLEU and chrF for real and COMET stubbed, and the report
A11's summary is written from. Step 6 runs too, on a toy A8 folder beside it: the student's own A8
and A9 translations of their test set, scored the same way and saved to their own file; its tests
cover a missing A9 file, a file of the wrong length, A9's re-translation of A8 preferred over
A8's own, and the step run alone in a later session.

The toy MTEval file is saved as a Windows user might leave it, with a byte-order mark and CRLF
line endings; one ranked sentence differs from its source in spacing only, one is a tie, and one is
in no source file, so each of those paths runs.

The stub scores a translation by how many of its words the reference has, or, without a
reference, by length; what is checked is the path, not COMET. Answers to the two exercises and a
prediction are written in, and three further tests check that the notebook as shipped stops at the
first exercise, that it stops without a prediction, and that an agreement count that includes ties
is rejected.

The Hugging Face login cell is left out of the run and tested on its own, with Colab's secrets
faked. The gated download's failure, and a notes file saved before the switch to COMETKiwi, each
have a test.
"""

import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest import mock

from notebook_cells import (
    COURSE,
    cell_containing,
    code_cells,
    colab_faked,
    run_cells,
    substitute,
)
from test_lecture_10 import ANSWER as COMET_INPUTS_ANSWER
from test_lecture_10 import (
    refuse_download,  # noqa: F401  (the license test imports it from here)
)

NOTEBOOK = COURSE / "lecture-11-comet-on-a6.ipynb"
SKIPPED = ("%pip install", "userdata.get")

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
        self.write_a8_a9()

    def write_a8_a9(self, n: int = 12) -> None:
        """Step 6's inputs, as A8 and A9 leave them: a test set and each model's translations.

        On even sentences A8 has the reference's words in reverse order and A9 the reference with
        one word misspelled, so the stub COMET (word overlap) prefers A8 and chrF prefers A9; on
        odd ones A8 is the reference and A9 its first three words, so both prefer A8.
        """
        self.a8 = self.drive / "CS479" / "assignment8"
        self.a8.mkdir(parents=True, exist_ok=True)
        refs = [f"the old man walked his small dog {i} times" for i in range(n)]
        a8 = [
            " ".join(reversed(r.split())) if i % 2 == 0 else r
            for i, r in enumerate(refs)
        ]
        a9 = [
            r.replace("walked", "walkd") if i % 2 == 0 else " ".join(r.split()[:3])
            for i, r in enumerate(refs)
        ]
        rows = [f"el viejo paseó a su perro {i} veces\t{r}" for i, r in enumerate(refs)]
        (self.a8 / "test.tsv").write_text(
            "\n".join(["src\ttgt", *rows]) + "\n", encoding="utf-8"
        )
        (self.a8 / "test.hyp").write_text("\n".join(a8) + "\n", encoding="utf-8")
        (self.a8 / "test-a9.hyp").write_text("\n".join(a9) + "\n", encoding="utf-8")

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
        self.assertIn("COMETKiwi and COMET:", printed)
        for label in ("BLEU", "chrF", "COMET", "COMETKiwi"):
            self.assertIn("system 1", namespace["system_scores"][label])

    def test_scores_the_students_own_a8_and_a9(self):
        with colab_faked(self.drive):
            namespace, printed = run_cells(lecture_11_cells(ANSWERS))
        own = printed[printed.index("=== For the A11 summary: your own systems ===") :]
        self.assertIn("12 test sentences; A8 from test.hyp, A9 from test-a9.hyp", own)
        self.assertIn("A8: test.hyp, from the A8 notebook", printed)
        for label in ("BLEU", "chrF", "COMET", "COMETKiwi"):
            self.assertIn(label, namespace["own_system"])
        # Disagreeing on the six even sentences, agreeing on the six odd ones.
        self.assertIn("COMET and chrF: 6 of 12 (50%)", own)
        self.assertEqual(own.count("\nSRC "), 3)
        self.assertIn("REF the old man walked his small dog 0 times", own)
        self.assertIn("A9  the old man walkd his small dog 0 times", own)

    def test_own_scores_go_to_their_own_file(self):
        with colab_faked(self.drive):
            run_cells(lecture_11_cells(ANSWERS))
        own = self.drive / "CS479" / "assignment11" / "a11-own-notes.json"
        self.assertEqual(
            json.loads(own.read_text(encoding="utf-8"))["files"],
            {"A8": "test.hyp", "A9": "test-a9.hyp"},
        )
        a6_notes = self.drive / "CS479" / "assignment6" / "a11-notes.json"
        self.assertNotIn(
            "A8",
            json.loads(a6_notes.read_text(encoding="utf-8"))["system_scores"]["COMET"],
        )

    def test_prefers_a8_as_a9_rescored_it(self):
        (self.a8 / "test-a8-again.hyp").write_bytes((self.a8 / "test.hyp").read_bytes())
        with colab_faked(self.drive):
            _, printed = run_cells(lecture_11_cells(ANSWERS))
        self.assertIn("A8 from test-a8-again.hyp", printed)
        self.assertNotIn("A8: test.hyp, from the A8 notebook", printed)

    def test_step_6_alone_in_a_later_session(self):
        """Steps 1 to 5 done earlier: the top cells, Steps 1 to 3, then Step 6 alone."""
        with colab_faked(self.drive):
            skipped = ('PREDICTION = "', "NOTES.write_text", "notes = json.loads(NOTES")
            cells = [
                c
                for c in lecture_11_cells(ANSWERS)
                if not any(s in c[1] for s in skipped)
            ]
            _, printed = run_cells(cells)
        self.assertNotIn("=== For the A11 summary ===\n", printed)
        self.assertIn("=== For the A11 summary: your own systems ===", printed)

    def test_a_missing_a9_file_names_a9(self):
        (self.a8 / "test-a9.hyp").unlink()
        with colab_faked(self.drive), self.assertRaises(SystemExit) as stopped:
            run_cells(lecture_11_cells(ANSWERS))
        self.assertIn("test-a9.hyp", str(stopped.exception))
        self.assertIn("Part B of the A9 notebook", str(stopped.exception))

    def test_a_translation_file_of_the_wrong_length_stops(self):
        lines = (self.a8 / "test-a9.hyp").read_text(encoding="utf-8").splitlines()
        (self.a8 / "test-a9.hyp").write_text(
            "\n".join(lines[:-1]) + "\n", encoding="utf-8"
        )
        with colab_faked(self.drive), self.assertRaises(SystemExit) as stopped:
            run_cells(lecture_11_cells(ANSWERS))
        self.assertIn(
            "test-a9.hyp has 11 lines and test.tsv has 12", str(stopped.exception)
        )
        self.assertIn("your A9 model", str(stopped.exception))

    def test_an_empty_last_translation_is_still_a_line(self):
        lines = (self.a8 / "test-a9.hyp").read_text(encoding="utf-8").splitlines()
        (self.a8 / "test-a9.hyp").write_text(
            "\n".join([*lines[:-1], ""]) + "\n", encoding="utf-8"
        )
        with colab_faked(self.drive):
            namespace, _ = run_cells(lecture_11_cells(ANSWERS))
        self.assertEqual(namespace["own_hyps"]["A9"][-1], "")

    def test_the_summary_asks_five_questions(self):
        nb = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
        summary = next(
            "".join(c["source"])
            for c in nb["cells"]
            if "".join(c["source"]).startswith("## Step 7: the summary")
        )
        self.assertIn("these five questions", summary)
        self.assertIn("5. **Your own systems.**", summary)
        self.assertNotIn("6. **", summary)

    def test_a_missing_token_names_notebook_access(self):
        """Notebook access is per notebook, so a student whose A10 logged in can still hit this."""
        _, login_cell = cell_containing(code_cells(NOTEBOOK), "userdata.get")
        colab = ModuleType("google.colab")
        colab.userdata = SimpleNamespace(get=lambda name: None)
        google = ModuleType("google")
        google.colab = colab
        hub = ModuleType(
            "huggingface_hub"
        )  # not installed here; the cell must stop first
        hub.login = mock.Mock()
        modules = {"google": google, "google.colab": colab, "huggingface_hub": hub}
        with (
            mock.patch.dict(sys.modules, modules),
            self.assertRaises(RuntimeError) as stopped,
        ):
            run_cells([(0, login_cell)])
        self.assertIn("'Notebook access'", str(stopped.exception))
        hub.login.assert_not_called()

    def test_an_unaccepted_license_says_what_to_do(self):
        cells = substitute(
            lecture_11_cells(ANSWERS),
            {"import download_model,": "import refuse_download as download_model,"},
        )
        with colab_faked(self.drive), self.assertRaises(RuntimeError) as stopped:
            run_cells(cells)
        self.assertIn("has not accepted COMETKiwi's license", str(stopped.exception))
        self.assertIn(
            "huggingface.co/Unbabel/wmt22-cometkiwi-da", str(stopped.exception)
        )

    def test_notes_saved_before_the_switch_ask_for_a_rescore(self):
        with colab_faked(self.drive):
            run_cells(lecture_11_cells(ANSWERS))
            notes_file = self.drive / "CS479" / "assignment6" / "a11-notes.json"
            notes = json.loads(notes_file.read_text(encoding="utf-8"))
            for scores in (notes["system_scores"], notes["sentence_scores"]):
                scores["COMET-QE-DA"] = scores.pop("COMETKiwi")
            notes_file.write_text(json.dumps(notes), encoding="utf-8")
            report = [
                c
                for c in lecture_11_cells(ANSWERS)
                if "load_from_checkpoint(" not in c[1]
            ]
            with self.assertRaises(SystemExit) as stopped:
                run_cells(report)
        self.assertIn("saved before A11 switched to COMETKiwi", str(stopped.exception))

    def test_the_report_survives_a_dropped_session(self):
        with colab_faked(self.drive):
            _, before = run_cells(lecture_11_cells(ANSWERS))
            # A fresh session: Step 1 and the cells above it, Step 3, and the two reports; no
            # comet_inputs, no prediction, no scoring.
            scoring = ("def comet_inputs", 'PREDICTION = "', "load_from_checkpoint(")
            cells = [
                c
                for c in lecture_11_cells(ANSWERS)
                if not any(s in c[1] for s in scoring)
            ]
            namespace, after = run_cells(cells)
        self.assertNotIn("Scored with COMET", after)
        self.assertNotIn("Scored A8 and A9", after)

        def reports(printed):
            """Step 5's report, through the prediction, and Step 6's, to the end."""
            a6 = printed.index("=== For the A11 summary ===")
            own = printed.index("=== For the A11 summary: your own systems ===")
            return printed[
                a6 : printed.index("\n", printed.index("Your prediction:"))
            ], printed[own:]

        self.assertEqual(reports(before), reports(after))
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
