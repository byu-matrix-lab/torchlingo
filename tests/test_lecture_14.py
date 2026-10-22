"""Lecture 14 (Assignment 14), executed from the notebook: Colab faked, toy size.

The notebook declares ``colab``, so CI's notebook run skips it. These tests run its cells from the
notebook file itself, starting from what A9 and A13 leave on Drive, which the fixture produces by
running Lecture 9's Part B and the A13 notebook's own cells (see ``test_lecture_13``). The install
cell is left out: it is Colab's.

Two cells are exercises: ``tag`` and ``both_directions`` are the student's, and raise
``NotImplementedError`` until written. The tests write them in, one straightforward way each, and
a hypothesis and a prediction, so what is tested is that the notebook's checks accept a right
answer and that the path after them works: the tags' pieces, a training set holding every pair
once each way, intermingled, with both languages' test sentences kept out of it, one model
trained on it, each direction decoded greedily and with beam and compared with its one-way model
by the paired bootstrap, the wrong-tag outputs, and the write-up. Further tests check that the
notebook as shipped stops at the first exercise, that it stops without a hypothesis in
"If ..., then ..." form, that a both_directions which does not shuffle is caught, that the
write-up runs from Drive alone after a dropped session, that the comparison into English uses
NLLB (stubbed: CI has no ``transformers``) when A13 used it, and that with nothing from A13 the
notebook compares English to X only.

What else changes in the cells is only the epochs and the model's size.
"""

import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import pandas as pd
from notebook_cells import COURSE, code_cells, colab_faked, run_cells, substitute
from test_lecture_09_part_b import TOY_SIZE
from test_lecture_13 import _FakeNLLB, build_drive, lecture_13_cells

import torchlingo_harness.nllb

NOTEBOOK = COURSE / "lecture-14-multilingual.ipynb"
SKIPPED = ('%pip install --quiet "torchlingo',)

# A student's answers to the two exercises, in place of the line that raises.
ANSWERS = {
    'raise NotImplementedError("write tag, then run this cell again")': (
        'return frame.assign(src=tag + " " + frame.src)[["src", "tgt"]]'
    ),
    'raise NotImplementedError("write both_directions, then run this cell again")': (
        'turned = frame.rename(columns={"src": "tgt", "tgt": "src"})[["src", "tgt"]]\n'
        "    both = pd.concat([tag(frame, TO_X), tag(turned, TO_EN)], ignore_index=True)\n"
        "    return both.sample(frac=1, random_state=SEED).reset_index(drop=True)"
    ),
}
HYPOTHESIS = "If one model learns both directions, then BLEU will fall a little"
PREDICTED = {
    'HYPOTHESIS = ""': f"HYPOTHESIS = {HYPOTHESIS!r}",
    'PREDICTION = ""': 'PREDICTION = "about one BLEU lower each way"',
}
READ_EPOCHS = 'EPOCHS = len(a9_state["train_losses"])'
SETUP_CELL = "def into_english("
TAGS_CELL = 'TO_X, TO_EN = "<2x>", "<2en>"'
EXERCISE_CELL = "def both_directions("
WRITE_UP_CELL = "=== For the A14 write-up ==="


def lecture_14_cells(
    answered: bool = True, predicted: bool = True, swaps: dict | None = None
) -> list[tuple[int, str]]:
    """Every code cell but the install, at toy size and one epoch."""
    cells = [c for c in code_cells(NOTEBOOK) if not any(m in c[1] for m in SKIPPED)]
    return substitute(
        cells,
        {
            **TOY_SIZE,
            READ_EPOCHS: "EPOCHS = 1",
            **(ANSWERS if answered else {}),
            **(PREDICTED if predicted else {}),
            **(swaps or {}),
        },
    )


def cells_with(cells: list[tuple[int, str]], *markers: str) -> list[tuple[int, str]]:
    """The cells containing each marker, in notebook order."""
    return sorted(next(c for c in cells if m in c[1]) for m in markers)


def after_a13(drive: Path, swaps: dict | None = None) -> tuple[Path, int]:
    """A9's files and what the A13 notebook leaves beside them, run at toy size."""
    a8_dir, n_test, _ = build_drive(drive)
    with colab_faked(drive):
        run_cells(lecture_13_cells(swaps=swaps))
    return a8_dir, n_test


def read(a8_dir: Path, name: str) -> pd.DataFrame:
    return pd.read_csv(a8_dir / name, sep="\t", dtype=str, keep_default_na=False)


class TestLecture14(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.drive = Path(cls.tmp.name)
        cls.a8_dir, cls.n_test = after_a13(cls.drive)
        with colab_faked(cls.drive):
            cls.ns, cls.printed = run_cells(lecture_14_cells())

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def notes(self) -> dict:
        return json.loads((self.a8_dir / "a14-notes.json").read_text(encoding="utf-8"))

    def test_the_tags_become_different_pieces(self):
        pieces = self.notes()["tag_pieces"]
        self.assertEqual(set(pieces), {"<2x>", "<2en>"})
        self.assertNotEqual(pieces["<2x>"], pieces["<2en>"])
        self.assertIn("<2x>    becomes", self.printed)

    def test_every_pair_once_each_way_intermingled(self):
        train = read(self.a8_dir, "train.tsv")
        combined = read(self.a8_dir, "train-a14.tsv")
        self.assertEqual(len(combined), 2 * len(train))
        to_x = combined[combined.src.str.startswith("<2x> ")]
        to_en = combined[combined.src.str.startswith("<2en> ")]
        self.assertEqual(len(to_x), len(train))
        self.assertEqual(sorted(to_x.tgt), sorted(train.tgt))
        self.assertEqual(sorted(to_en.tgt), sorted(train.src))
        first_half = combined.src.iloc[: len(combined) // 2].str.startswith("<2x> ")
        self.assertTrue(0.2 < first_half.mean() < 0.8, "intermingled")
        self.assertEqual(self.notes()["training_pairs"], len(combined))

    def test_one_test_set_per_direction_and_none_in_training(self):
        test = read(self.a8_dir, "test.tsv")
        to_x = read(self.a8_dir, "test-a14-to-x.tsv")
        to_en = read(self.a8_dir, "test-a14-to-en.tsv")
        self.assertEqual(list(to_x.src), [f"<2x> {s}" for s in test.src])
        self.assertEqual(list(to_en.src), [f"<2en> {s}" for s in test.tgt])
        self.assertEqual(list(to_en.tgt), list(test.src))
        self.assertEqual(self.printed.count("[PASS] test, "), 2)

    def test_one_model_trained(self):
        self.assertTrue((self.a8_dir / "best-a14" / "model_best.pt").exists())
        self.assertIn("Epochs: 1", self.printed)

    def test_each_direction_decoded_both_ways_against_its_one_way_model(self):
        for direction in ("to-x", "to-en"):
            for decoder in ("greedy", "beam"):
                hyp = self.a8_dir / f"test-a14-{direction}-{decoder}.hyp"
                self.assertEqual(
                    len(hyp.read_text(encoding="utf-8").splitlines()), self.n_test
                )
        for decoder in ("greedy", "beam"):
            for name in (f"test-a9-{decoder}.hyp", f"test-a13-reverse-{decoder}.hyp"):
                self.assertIn(name, self.notes()["hyp_files"])
        scores = self.notes()["scores"]
        self.assertEqual(list(scores["to-x"]["greedy"]), ["A9", "A14"])
        self.assertEqual(list(scores["to-en"]["beam"]), ["A13 reverse", "A14"])
        self.assertIsNone(scores["to-x"]["greedy"]["A9"]["bleu_p"])
        self.assertTrue(0 < scores["to-x"]["greedy"]["A14"]["bleu_p"] <= 1)
        self.assertEqual(self.notes()["into_english"], "A13 reverse")
        # A9's greedy decode here is the one A9 itself reported.
        a9 = (self.a8_dir / "test-a9.hyp").read_text(encoding="utf-8").splitlines()
        again = (self.a8_dir / "test-a9-greedy.hyp").read_text(encoding="utf-8")
        self.assertEqual(again.splitlines(), a9)

    def test_the_wrong_tag_outputs_are_saved(self):
        wrong = self.notes()["wrong_tag"]
        self.assertEqual(len(wrong), 2)
        test = read(self.a8_dir, "test.tsv")
        english = wrong["English, tagged into English"]
        self.assertEqual([row[0] for row in english], list(test.src[:5]))

    def test_prints_the_write_up_block_and_the_rerun_record(self):
        self.assertIn(WRITE_UP_CELL, self.printed)
        self.assertIn(
            f"Hypothesis, written before training: {HYPOTHESIS}", self.printed
        )
        self.assertIn("into English, beam:", self.printed)
        self.assertIn("A13 reverse", self.printed)
        self.assertIn(
            "=== The wrong tag: English, tagged into English ===", self.printed
        )
        rerun = json.loads((self.a8_dir / "a14-rerun.json").read_text(encoding="utf-8"))
        self.assertEqual(rerun["tags"], ["<2x>", "<2en>"])
        self.assertEqual(rerun["seed"], 479)
        self.assertEqual(rerun["into_english_compared_with"], "A13 reverse")

    def test_the_write_up_runs_after_a_dropped_session(self):
        """A new session runs the setup cell, then the write-up: nothing else in memory."""
        with colab_faked(self.drive):
            _, printed = run_cells(
                cells_with(lecture_14_cells(), SETUP_CELL, WRITE_UP_CELL)
            )
        self.assertIn(HYPOTHESIS, printed)
        self.assertIn("into your language, greedy:", printed)

    def test_rerunning_the_training_cell_after_it_finishes_trains_nothing(self):
        model = self.a8_dir / "best-a14" / "model_best.pt"
        before = model.stat().st_mtime_ns
        with colab_faked(self.drive):
            _, printed = run_cells(
                cells_with(lecture_14_cells(), SETUP_CELL, 'train("train-a14.tsv"')
            )
        self.assertEqual(
            model.stat().st_mtime_ns, before, "the best model was not rewritten"
        )

    def test_as_shipped_it_stops_at_the_first_exercise(self):
        cells = cells_with(
            lecture_14_cells(answered=False), SETUP_CELL, TAGS_CELL, EXERCISE_CELL
        )
        with colab_faked(self.drive), self.assertRaises(NotImplementedError) as stopped:
            run_cells(cells)
        self.assertIn("write tag", str(stopped.exception))

    def test_without_a_hypothesis_it_stops(self):
        cells = cells_with(
            lecture_14_cells(predicted=False), SETUP_CELL, 'HYPOTHESIS = ""'
        )
        with colab_faked(self.drive), self.assertRaises(AssertionError) as stopped:
            run_cells(cells)
        self.assertIn("Write your hypothesis", str(stopped.exception))

    def test_a_hypothesis_not_in_if_then_form_is_refused(self):
        bad = {
            'HYPOTHESIS = ""': 'HYPOTHESIS = "one model will be worse"',
            'PREDICTION = ""': 'PREDICTION = "a little"',
        }
        cells = cells_with(
            lecture_14_cells(predicted=False, swaps=bad),
            SETUP_CELL,
            "save_note(hypothesis=",
        )
        with colab_faked(self.drive), self.assertRaises(AssertionError) as stopped:
            run_cells(cells)
        self.assertIn('"If ..., then ..."', str(stopped.exception))

    def test_directions_not_intermingled_are_caught(self):
        """Both directions concatenated, unshuffled: the known case passes, the split does not."""
        unshuffled = {
            "    return both.sample(frac=1, random_state=SEED).reset_index(drop=True)": (
                "    return both"
            )
        }
        answered = cells_with(lecture_14_cells(), SETUP_CELL, TAGS_CELL, EXERCISE_CELL)
        cells = substitute(answered, unshuffled)  # after the answer is in, not before
        with colab_faked(self.drive), self.assertRaises(AssertionError) as stopped:
            run_cells(cells)
        self.assertIn("not intermingled", str(stopped.exception))


class TestNLLBInPlaceOfAReverseModel(unittest.TestCase):
    """A13 back-translated with NLLB: A14 compares into English with NLLB, stubbed here."""

    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.drive = Path(cls.tmp.name)
        cls.loaded = []

        def load_nllb(src, tgt, device=None):
            cls.loaded.append((src, tgt))
            return _FakeNLLB()

        nllb = {
            'BACK_TRANSLATOR = "own"': 'BACK_TRANSLATOR = "nllb"',
            "NLLB_CODE = None": 'NLLB_CODE = "tel_Telu"',
        }
        with mock.patch.object(torchlingo_harness.nllb, "load_nllb", load_nllb):
            cls.a8_dir, _ = after_a13(cls.drive, swaps=nllb)
            cls.loaded.clear()
            with colab_faked(cls.drive):
                _, cls.printed = run_cells(lecture_14_cells())

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_compares_into_english_with_nllb(self):
        self.assertEqual(self.loaded, [("tel_Telu", "eng_Latn")])
        notes = json.loads((self.a8_dir / "a14-notes.json").read_text(encoding="utf-8"))
        self.assertEqual(list(notes["scores"]["to-en"]["greedy"]), ["NLLB-200", "A14"])
        hyp = (self.a8_dir / "test-a13-reverse-beam.hyp").read_text(encoding="utf-8")
        self.assertTrue(hyp.startswith("BEAM "))


class TestNothingFromA13(unittest.TestCase):
    """No reverse model and no NLLB: English to X is compared, into English is scored alone."""

    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.drive = Path(cls.tmp.name)
        cls.a8_dir, _, _ = build_drive(cls.drive)
        with colab_faked(cls.drive):
            _, cls.printed = run_cells(lecture_14_cells())

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_into_english_is_scored_alone(self):
        notes = json.loads((self.a8_dir / "a14-notes.json").read_text(encoding="utf-8"))
        self.assertIsNone(notes["into_english"])
        self.assertEqual(list(notes["scores"]["to-en"]["greedy"]), ["A14"])
        self.assertIsNone(notes["scores"]["to-en"]["greedy"]["A14"]["bleu_ci"])
        self.assertEqual(list(notes["scores"]["to-x"]["beam"]), ["A9", "A14"])
        self.assertIn("compared with: nothing", self.printed)


if __name__ == "__main__":
    unittest.main()
