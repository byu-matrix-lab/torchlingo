"""Lecture 10 (Assignment 10), executed from the notebook: COMET stubbed.

The notebook declares ``pip``, ``colab`` and ``hf-token``, so CI's notebook run skips it, and CI
has neither ``unbabel-comet`` nor a Hugging Face token. What these tests run is everything else,
from the notebook file itself: the example, the student's ``comet_inputs`` and its checks, the
scoring loop and the summary A10 grades.

Two cells are left out (the install, the Colab login) and two imports changed (COMET's, and
huggingface_hub's ``GatedRepoError``), and a stub model stands in for COMET: it scores a
translation 1.0 when it equals the reference and 0.5 otherwise, or, without a reference, by
length. What is checked is the path, not COMET.
"""

import unittest
from types import SimpleNamespace

from notebook_cells import COURSE, code_cells, run_cells, substitute

NOTEBOOK = COURSE / "lecture-10-comet-install.ipynb"
SKIPPED = ("%pip install", "userdata.get")

ANSWER = {
    'raise NotImplementedError("write comet_inputs, then run this cell again")': (
        "for other in (translations, references):\n"
        "        if other is not None and len(other) != len(sources):\n"
        '            raise ValueError("lists of different lengths")\n'
        "    if references is None:\n"
        '        return [{"src": s, "mt": t} for s, t in zip(sources, translations)]\n'
        '    return [{"src": s, "mt": t, "ref": r} for s, t, r in zip(sources, translations, references)]'
    ),
}
PREDICTED = {
    'PREDICTION = ""': 'PREDICTION = "both prefer system 2"',
}
STUB_COMET = {
    "from comet import download_model, load_from_checkpoint": (
        "from test_lecture_10 import download_model, load_from_checkpoint"
    ),
}


def download_model(name: str) -> str:
    """Stand-in for COMET's download: the model's name is its path."""
    return name


class _StubModel:
    def predict(self, records, **kwargs):
        scores = [
            (1.0 if r["mt"] == r["ref"] else 0.5) if "ref" in r else len(r["mt"]) / 100
            for r in records
        ]
        return SimpleNamespace(scores=scores, system_score=sum(scores) / len(scores))


def load_from_checkpoint(path: str) -> _StubModel:
    """Stand-in for COMET's loader."""
    return _StubModel()


def lecture_10_cells(
    answered: bool = True, predicted: bool = True
) -> list[tuple[int, str]]:
    cells = [c for c in code_cells(NOTEBOOK) if not any(m in c[1] for m in SKIPPED)]
    return substitute(
        cells,
        {
            **STUB_COMET,
            **(ANSWER if answered else {}),
            **(PREDICTED if predicted else {}),
        },
    )


def refuse_download(name: str) -> str:
    """COMET's download for an account that has not accepted COMETKiwi's license.

    COMET 2.2.7 catches the gated-repository error and raises this ``KeyError`` instead.
    """
    if "cometkiwi" in name:
        raise KeyError(f"Model '{name}' not supported by COMET.")
    return name


class TestLecture10(unittest.TestCase):
    def test_runs_to_the_summary_a10_grades(self):
        namespace, printed = run_cells(lecture_10_cells())
        self.assertIn("comet_inputs passes its checks.", printed)
        self.assertIn("=== For A10 ===", printed)
        self.assertIn("Your prediction: both prefer system 2", printed)
        self.assertEqual(set(namespace["with_reference"]), {"system 1", "system 2"})
        self.assertEqual(set(namespace["without_reference"]), {"system 1", "system 2"})

    def test_as_shipped_it_stops_at_the_exercise(self):
        with self.assertRaises(NotImplementedError) as stopped:
            run_cells(lecture_10_cells(answered=False))
        self.assertIn("write comet_inputs", str(stopped.exception))

    def test_without_a_prediction_it_stops_before_scoring(self):
        with self.assertRaises(AssertionError) as stopped:
            run_cells(lecture_10_cells(predicted=False))
        self.assertIn("Write your prediction", str(stopped.exception))

    def test_an_unaccepted_license_says_what_to_do(self):
        """The gated download fails even when logged in; the cell names the missing step."""
        cells = substitute(
            lecture_10_cells(),
            {
                "import download_model, load_from_checkpoint": (
                    "import refuse_download as download_model, load_from_checkpoint"
                ),
            },
        )
        with self.assertRaises(RuntimeError) as stopped:
            run_cells(cells)
        self.assertIn("has not accepted COMETKiwi's license", str(stopped.exception))

    def test_an_answer_that_ignores_lengths_is_rejected(self):
        careless = {
            old: new.replace('raise ValueError("lists of different lengths")', "pass")
            for old, new in ANSWER.items()
        }
        cells = [c for c in code_cells(NOTEBOOK) if not any(m in c[1] for m in SKIPPED)]
        with self.assertRaises(AssertionError) as stopped:
            run_cells(substitute(cells, {**STUB_COMET, **PREDICTED, **careless}))
        self.assertIn("must refuse lists of different lengths", str(stopped.exception))


if __name__ == "__main__":
    unittest.main()
