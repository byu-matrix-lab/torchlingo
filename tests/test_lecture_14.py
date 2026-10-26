"""Lecture 14 (Assignment 14, fine-tuning NLLB both ways), executed from the notebook.

The notebook declares ``colab``, so CI's notebook run skips it. These tests run its cells from the
notebook file itself, starting from what A9 and A13 leave on Drive, which the fixture produces by
running Lecture 9's Part B and the A13 notebook's own cells at toy size (see
``test_lecture_13``). The install cell and COMET's are left out: they are Colab's.

CI has no ``transformers`` and should not download 2.4 GB, so NLLB is a stand-in: a tokenizer
that builds NLLB's sequences from words, a one-weight model whose loss training lowers, and a
translator that writes the reference once the model has trained and the source before. The
fine-tuning itself is the library's (``torchlingo_harness.nllb.fine_tune``) with an SGD in
Adafactor's place; COMET is a stand-in too, scoring exact matches. What is tested is the path:
the code table and a borrowed code, the student's tag and both_directions, the training set and
the test sets, every stop, the three systems per direction with the bootstrap, COMET on every
translation, the wrong code and the third language, and the write-up from Drive alone.
"""

import contextlib
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from typing import ClassVar
from unittest import mock

import pandas as pd
import torch
from notebook_cells import COURSE, code_cells, colab_faked, run_cells, substitute
from test_lecture_09_part_b import TOY_SIZE
from test_lecture_13 import build_drive, lecture_13_cells

import torchlingo.evaluation
import torchlingo_harness.nllb

NOTEBOOK = COURSE / "lecture-14-multilingual.ipynb"
CODES = ["eng_Latn", "deu_Latn", "fra_Latn", "tgl_Latn", "smo_Latn"]

# A student's answers to the two exercises, in place of the line that raises.
ANSWERS = {
    'raise NotImplementedError("write tag, then run this cell again")': (
        'return frame.assign(src_code=src_code, tgt_code=tgt_code)[["src", "tgt", "src_code", "tgt_code"]]'
    ),
    'raise NotImplementedError("write both_directions, then run this cell again")': (
        'turned = frame.rename(columns={"src": "tgt", "tgt": "src"})[["src", "tgt"]]\n'
        "    both = pd.concat([tag(frame, ENG, X_CODE), tag(turned, X_CODE, ENG)], ignore_index=True)\n"
        "    return both.sample(frac=1, random_state=SEED).reset_index(drop=True)"
    ),
}
HYPOTHESIS = "If I fine-tune NLLB both ways, then BLEU will beat A9"
DECIDED = {
    'LANGUAGE = ""': 'LANGUAGE = "German"',
    'HYPOTHESIS = ""': f"HYPOTHESIS = {HYPOTHESIS!r}",
    'PREDICTION = ""': 'PREDICTION = "well above A9, a little above off the shelf"',
    "STEPS = 3000": "STEPS = 6",
    'THIRD = ""': 'THIRD = "fra_Latn"',
    'EXPECT = ""': 'EXPECT = "copies, copies, and French that keeps the meaning"',
    # Read off the table: the answer whatever the toy run's p-values come out as.
    "REAL = {}": (
        'REAL = {f"{n}: A14 against {s}": v["bleu_p"] < 0.05 for d, n in '
        '(("to-x", "into your language"), ("to-en", "into English")) '
        'for s, v in load_notes()["scores"][d]["greedy"].items() if s != "A14"}'
    ),
}
REAL_CELL = "Copy these into REAL"
SETUP_CELL = "def scratch_translate("
NLLB_CELL = "def use_fine_tuned("
LANGUAGE_CELL = "COURSE_CODES = {"
BORROW_CELL = "X_CODE = BORROW"
EXERCISE_CELL = "def both_directions("
OFF_THE_SHELF_CELL = "test-nllb-"
HYPOTHESIS_CELL = "HYPOTHESIS = "
TRAINING_CELL = "fine_tune(model"
WRITE_UP_CELL = "=== For the A14 write-up ==="


class StubTokenizer:
    """NLLB's tokenizer in miniature: words for pieces, its codes as special tokens."""

    pad_token_id, eos_token_id = 1, 2

    def __init__(self):
        self.additional_special_tokens = list(CODES)
        self.src_lang = "eng_Latn"
        self.ids = {code: 10 + i for i, code in enumerate(CODES)}

    def tokenize(self, text):
        return text.split()

    def convert_tokens_to_ids(self, tokens):
        if isinstance(tokens, str):
            return self.ids.setdefault(tokens, 100 + len(self.ids))
        return [self.convert_tokens_to_ids(t) for t in tokens]


class StubModel(torch.nn.Module):
    """One weight, whose loss training lowers; it says whether it has been trained."""

    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(1))

    def forward(self, input_ids, attention_mask, labels):
        return SimpleNamespace(loss=(self.weight**2).sum())

    @property
    def trained(self):
        return self.weight.item() < 1.0


class FakeAdafactor(torch.optim.SGD):
    """Adafactor's arguments, SGD's update."""

    def __init__(self, params, lr, **options):
        super().__init__(params, lr=lr)


class FakeTranslator:
    """NLLB decoding: the reference once trained, the source tagged with its code before."""

    references: ClassVar[dict] = {}

    def __init__(self, model, tokenizer, tgt_lang, device, name):
        self.model, self.tgt_lang = model, tgt_lang

    def __call__(self, sentences, **options):
        if self.model.trained:
            return [self.references.get((s, self.tgt_lang), s) for s in sentences]
        return [f"[{self.tgt_lang}] {s}" for s in sentences]

    def settings(self, **decoding):
        return {"translator": "fake-nllb", "tgt_lang": self.tgt_lang, **decoding}


class FakeComet:
    def predict(self, data, **options):
        scores = [float(d["mt"] == d["ref"]) for d in data]
        return SimpleNamespace(scores=scores, system_score=sum(scores) / len(scores))


@contextlib.contextmanager
def nllb_and_comet_stubbed():
    """transformers, Adafactor, NLLB's translator and COMET, for the cells that ask for them.

    Only these sys.modules entries change, and are put back as they were: restoring all of
    sys.modules would unload the parts of torch an optimizer step loads.
    """
    fakes = {
        "transformers": SimpleNamespace(
            AutoTokenizer=SimpleNamespace(
                from_pretrained=lambda name, **kw: StubTokenizer()
            ),
            AutoModelForSeq2SeqLM=SimpleNamespace(
                from_pretrained=lambda name, **kw: StubModel()
            ),
        ),
        "transformers.optimization": SimpleNamespace(Adafactor=FakeAdafactor),
        "comet": SimpleNamespace(
            download_model=lambda name: "fake-comet",
            load_from_checkpoint=lambda path: FakeComet(),
        ),
    }
    originals = {name: sys.modules.get(name) for name in fakes}
    sys.modules.update(fakes)
    try:
        # comet_score keeps a loaded model for the session; this one must not outlive the test.
        with (
            mock.patch.object(
                torchlingo_harness.nllb, "NLLBTranslator", FakeTranslator
            ),
            mock.patch.dict(torchlingo.evaluation._comet_models, clear=True),
        ):
            yield
    finally:
        for name, original in originals.items():
            if original is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = original


def lecture_14_cells(
    answered: bool = True, decided: bool = True, swaps: dict | None = None
) -> list[tuple[int, str]]:
    """Every code cell but the installs, at toy size, with six steps of training."""
    cells = [c for c in code_cells(NOTEBOOK) if "%pip" not in c[1]]
    return substitute(
        cells,
        {
            **TOY_SIZE,
            **(ANSWERS if answered else {}),
            **(DECIDED if decided else {}),
            **(swaps or {}),
        },
    )


def cells_with(cells: list[tuple[int, str]], *markers: str) -> list[tuple[int, str]]:
    """The cells containing each marker, in notebook order."""
    return sorted(next(c for c in cells if m in c[1]) for m in markers)


def read(a8_dir: Path, name: str) -> pd.DataFrame:
    return pd.read_csv(a8_dir / name, sep="\t", dtype=str, keep_default_na=False)


def a9_and_a13(drive: Path, with_a13: bool = True) -> Path:
    """A9's files, and what the A13 notebook leaves beside them, run at toy size."""
    a8_dir, _, _ = build_drive(drive)
    if with_a13:
        with colab_faked(drive):
            run_cells(lecture_13_cells())
    test = read(a8_dir, "test.tsv")
    FakeTranslator.references = {
        **{(s, "deu_Latn"): t for s, t in zip(test.src, test.tgt)},
        **{(t, "eng_Latn"): s for s, t in zip(test.src, test.tgt)},
    }
    return a8_dir


def run(drive: Path, cells) -> tuple[dict, str]:
    with colab_faked(drive), nllb_and_comet_stubbed():
        return run_cells(cells)


class TestLecture14(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.drive = Path(cls.tmp.name)
        cls.a8_dir = a9_and_a13(cls.drive)
        cls.ns, cls.printed = run(cls.drive, lecture_14_cells())

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def notes(self) -> dict:
        return json.loads((self.a8_dir / "a14-notes.json").read_text(encoding="utf-8"))

    def test_the_code_comes_from_the_table(self):
        self.assertEqual(self.ns["X_CODE"], "deu_Latn")
        self.assertIn("NLLB covers German: its code is deu_Latn", self.printed)
        self.assertEqual(self.notes()["x_code"], "deu_Latn")

    def test_every_pair_once_each_way_with_its_codes_intermingled(self):
        train = read(self.a8_dir, "train.tsv")
        combined = read(self.a8_dir, "train-a14.tsv")
        self.assertEqual(list(combined.columns), ["src", "tgt", "src_code", "tgt_code"])
        self.assertEqual(len(combined), 2 * len(train))
        to_x = combined[combined.tgt_code == "deu_Latn"]
        to_en = combined[combined.tgt_code == "eng_Latn"]
        self.assertEqual(set(to_x.src_code), {"eng_Latn"})
        self.assertEqual(sorted(to_x.tgt), sorted(train.tgt))
        self.assertEqual(sorted(to_en.tgt), sorted(train.src))
        first_half = (combined.tgt_code.iloc[: len(combined) // 2] == "deu_Latn").mean()
        self.assertTrue(0.2 < first_half < 0.8, "intermingled")

    def test_one_test_set_per_direction_and_none_in_training(self):
        test = read(self.a8_dir, "test.tsv")
        to_x = read(self.a8_dir, "test-a14-to-x.tsv")
        to_en = read(self.a8_dir, "test-a14-to-en.tsv")
        self.assertEqual(list(to_x.src), list(test.src))
        self.assertEqual(set(to_x.tgt_code), {"deu_Latn"})
        self.assertEqual(list(to_en.src), list(test.tgt))
        self.assertEqual(set(to_en.tgt_code), {"eng_Latn"})
        self.assertEqual(self.printed.count("[PASS] test, "), 2)

    def test_fine_tuned_by_the_library_with_the_students_budget(self):
        self.assertTrue((self.a8_dir / "a14-checkpoints" / "best.pt").is_file())
        self.assertEqual(self.notes()["steps_taken"], 6)
        self.assertTrue(self.ns["FINE_TUNED"])

    def test_three_systems_each_direction_and_p_against_a14(self):
        scores = self.notes()["scores"]
        self.assertEqual(
            list(scores["to-x"]["greedy"]), ["A14", "NLLB off the shelf", "A9"]
        )
        self.assertEqual(
            list(scores["to-en"]["beam"]), ["A14", "NLLB off the shelf", "A13 reverse"]
        )
        rows = scores["to-x"]["greedy"]
        self.assertIsNone(rows["A14"]["bleu_p"])
        self.assertTrue(0 < rows["A9"]["bleu_p"] <= 1)
        self.assertGreater(rows["A14"]["chrf"], rows["NLLB off the shelf"]["chrf"])
        files = self.notes()["hyp_files"]
        for decoder in ("greedy", "beam"):
            for name in (
                f"test-nllb-to-x-{decoder}.hyp",
                f"test-a14-to-en-{decoder}.hyp",
                f"test-a9-{decoder}.hyp",
                f"test-a13-reverse-{decoder}.hyp",
            ):
                self.assertIn(name, files.values())
        # A9's greedy decode here is the one A9 itself reported.
        a9 = (self.a8_dir / "test-a9.hyp").read_text(encoding="utf-8").splitlines()
        again = (self.a8_dir / "test-a9-greedy.hyp").read_text(encoding="utf-8")
        self.assertEqual(again.splitlines(), a9)

    def test_comet_scores_every_translation(self):
        comet = self.notes()["comet"]
        for direction in ("to-x", "to-en"):
            for decoder in ("greedy", "beam"):
                self.assertEqual(len(comet[direction][decoder]), 3)
        self.assertEqual(comet["to-x"]["greedy"]["A14"], 1.0)
        self.assertEqual(comet["to-x"]["greedy"]["NLLB off the shelf"], 0.0)

    def test_the_wrong_code_and_the_third_language(self):
        wrong = self.notes()["wrong_code"]
        self.assertEqual(
            list(wrong),
            [
                "English, asked for English",
                "your language, asked for your language",
                "your language, asked for fra_Latn",
            ],
        )
        test = read(self.a8_dir, "test.tsv")
        self.assertEqual(
            [r[0] for r in wrong["English, asked for English"]], list(test.src[:5])
        )
        self.assertEqual(self.notes()["third"], "fra_Latn")

    def test_prints_the_write_up_block_and_the_rerun_record(self):
        self.assertIn(WRITE_UP_CELL, self.printed)
        self.assertIn(
            f"Hypothesis, written before training: {HYPOTHESIS}", self.printed
        )
        self.assertIn("into English, beam:", self.printed)
        self.assertIn("COMET 1.0000", self.printed)
        rerun = json.loads((self.a8_dir / "a14-rerun.json").read_text(encoding="utf-8"))
        self.assertEqual(rerun["codes"], {"English": "eng_Latn", "German": "deu_Latn"})
        self.assertEqual((rerun["seed"], rerun["steps"]), (479, 6))
        self.assertEqual(rerun["into_english_one_way"], "A13 reverse")
        self.assertIn(
            "(", rerun["versions"]["torchlingo"], "which code, not only 0.2.x"
        )
        self.assertIn("sacrebleu", rerun["versions"])

    def test_an_a9_model_saved_before_settings_were_recorded_decodes_the_same(self):
        """A9 models saved before Oct 14 hold only weights; the notebook's Config builds them."""
        raw = torch.load(self.a8_dir / "best-a9" / "model_best.pt", weights_only=True)
        old = self.a8_dir / "best-old-a9" / "model_best.pt"
        old.parent.mkdir()
        torch.save({"epoch": 1, "model_state_dict": raw["model_state_dict"]}, old)
        translate = self.ns["scratch_translate"]
        sentences = list(self.ns["test_df"].src[:3])
        with colab_faked(
            self.drive
        ):  # the notebook's A8_DIR is relative to the faked Drive
            self.assertEqual(
                translate("best-old-a9", sentences, self.a8_dir / "old.hyp", "greedy"),
                translate("best-a9", sentences, self.a8_dir / "new.hyp", "greedy"),
            )

    def test_the_write_up_runs_after_a_dropped_session(self):
        """A new session runs the setup cells, then the write-up: nothing else in memory."""
        _, printed = run(
            self.drive,
            cells_with(lecture_14_cells(), SETUP_CELL, NLLB_CELL, WRITE_UP_CELL),
        )
        self.assertIn(HYPOTHESIS, printed)
        self.assertIn("into your language, greedy:", printed)

    def test_rerunning_the_training_after_it_finished_trains_nothing(self):
        best = self.a8_dir / "a14-checkpoints" / "best.pt"
        before = best.stat().st_mtime_ns
        _, printed = run(
            self.drive,
            cells_with(
                lecture_14_cells(),
                SETUP_CELL,
                NLLB_CELL,
                HYPOTHESIS_CELL,
                TRAINING_CELL,
            ),
        )
        self.assertEqual(
            best.stat().st_mtime_ns, before, "the best model was not rewritten"
        )
        self.assertIn("Fine-tuned for 6 steps", printed)

    def test_the_student_reads_which_differences_are_real(self):
        self.assertEqual(
            len(self.notes()["real"]), 4
        )  # two systems against A14, each way

    def test_reading_a_p_value_wrong_is_caught_after_a_dropped_session(self):
        """The cell reads the scores from Drive, so it needs only the setup cell before it."""
        scores = self.notes()["scores"]
        key = "into your language: A14 against A9"
        wrong = not (scores["to-x"]["greedy"]["A9"]["bleu_p"] < 0.05)
        others = {
            f"{n}: A14 against {s}": v["bleu_p"] < 0.05
            for d, n in (("to-x", "into your language"), ("to-en", "into English"))
            for s, v in scores[d]["greedy"].items()
            if s != "A14"
        }
        answer = {**others, key: wrong}
        cells = cells_with(
            substitute(
                lecture_14_cells(decided=False), {"REAL = {}": f"REAL = {answer!r}"}
            ),
            SETUP_CELL,
            REAL_CELL,
        )
        with self.assertRaises(AssertionError) as stopped:
            run(self.drive, cells)
        self.assertIn(f"For {key} you said", str(stopped.exception))

    def test_off_the_shelf_cannot_be_decoded_by_a_fine_tuned_model(self):
        """After training, in the same session, a missing off-the-shelf file is refused."""
        hyp = self.a8_dir / "test-nllb-to-x-beam.hyp"
        hyp.rename(hyp.with_suffix(".kept"))
        self.addCleanup(hyp.with_suffix(".kept").rename, hyp)
        cells = cells_with(
            lecture_14_cells(),
            SETUP_CELL,
            NLLB_CELL,
            LANGUAGE_CELL,
            BORROW_CELL,
            OFF_THE_SHELF_CELL,
        )
        *before, last = cells
        with self.assertRaises(AssertionError) as stopped:
            run(self.drive, [*before, (last[0], "FINE_TUNED = True\n" + last[1])])
        self.assertIn("fine-tuned now", str(stopped.exception))


class TestTheStops(unittest.TestCase):
    """Each place the notebook waits for the student, and the checks on their answers."""

    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.drive = Path(cls.tmp.name)
        cls.a8_dir = a9_and_a13(cls.drive, with_a13=False)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def stopped(self, cells, error=AssertionError) -> str:
        with self.assertRaises(error) as stopped:
            run(self.drive, cells)
        return str(stopped.exception)

    def test_as_shipped_it_stops_for_the_language(self):
        cells = cells_with(
            lecture_14_cells(answered=False, decided=False),
            SETUP_CELL,
            NLLB_CELL,
            LANGUAGE_CELL,
        )
        self.assertIn("Set LANGUAGE", self.stopped(cells))

    def test_then_it_stops_at_the_first_exercise(self):
        cells = cells_with(
            lecture_14_cells(answered=False),
            SETUP_CELL,
            NLLB_CELL,
            LANGUAGE_CELL,
            BORROW_CELL,
            EXERCISE_CELL,
        )
        self.assertIn("write tag", self.stopped(cells, NotImplementedError))

    def test_directions_not_intermingled_are_caught(self):
        """Both directions concatenated, unshuffled: the known case passes, the split does not."""
        cells = cells_with(
            lecture_14_cells(),
            SETUP_CELL,
            NLLB_CELL,
            LANGUAGE_CELL,
            BORROW_CELL,
            EXERCISE_CELL,
        )
        unshuffled = substitute(
            cells,
            {
                "    return both.sample(frac=1, random_state=SEED).reset_index(drop=True)": (
                    "    return both"
                )
            },
        )
        self.assertIn("not intermingled", self.stopped(unshuffled))

    def test_a_hypothesis_not_in_if_then_form_is_refused(self):
        bad = {'HYPOTHESIS = ""': 'HYPOTHESIS = "it will be better"'}
        cells = cells_with(
            substitute(
                lecture_14_cells(decided=False),
                {**bad, 'LANGUAGE = ""': 'LANGUAGE = "German"'},
            ),
            SETUP_CELL,
            HYPOTHESIS_CELL,
        )
        self.assertIn('"If ..., then ..."', self.stopped(cells))

    def test_the_third_language_must_be_a_third_one(self):
        swaps = {'THIRD = "fra_Latn"': 'THIRD = "eng_Latn"'}
        cells = cells_with(
            substitute(lecture_14_cells(), swaps),
            SETUP_CELL,
            NLLB_CELL,
            LANGUAGE_CELL,
            BORROW_CELL,
            'THIRD = "eng_Latn"',
        )
        self.assertIn("third language you can read", self.stopped(cells))

    def test_it_stops_until_the_student_says_what_they_expect(self):
        swaps = {
            'EXPECT = "copies, copies, and French that keeps the meaning"': 'EXPECT = ""'
        }
        cells = cells_with(
            substitute(lecture_14_cells(), swaps),
            SETUP_CELL,
            NLLB_CELL,
            LANGUAGE_CELL,
            BORROW_CELL,
            "assert EXPECT.strip()",
        )
        self.assertIn("Write what you expect in EXPECT", self.stopped(cells))

    def test_an_uncovered_language_stops_for_candidates_then_borrows(self):
        hmong = {'LANGUAGE = "German"': 'LANGUAGE = "Hmong"'}
        cells = cells_with(
            substitute(lecture_14_cells(), hmong), SETUP_CELL, NLLB_CELL, LANGUAGE_CELL
        )
        self.assertIn("NLLB does not cover Hmong", self.stopped(cells))

        chosen = {
            **hmong,
            "CANDIDATES = []": 'CANDIDATES = ["tgl_Latn", "smo_Latn"]',
            'BORROW = ""': 'BORROW = "tgl_Latn"',
            'REASON = ""': 'REASON = "Austronesian contact, Latin script"',
        }
        cells = cells_with(
            substitute(lecture_14_cells(), chosen),
            SETUP_CELL,
            NLLB_CELL,
            LANGUAGE_CELL,
            BORROW_CELL,
        )
        namespace, printed = run(self.drive, cells)
        self.assertEqual(namespace["X_CODE"], "tgl_Latn")
        self.assertIn("tgl_Latn   chrF", printed)
        self.assertIn("smo_Latn   chrF", printed)


class TestNoReverseModel(unittest.TestCase):
    """No A13 reverse model: into English compares A14 with NLLB off the shelf only."""

    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.drive = Path(cls.tmp.name)
        cls.a8_dir = a9_and_a13(cls.drive, with_a13=False)
        _, cls.printed = run(cls.drive, lecture_14_cells())

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_into_english_has_two_systems(self):
        notes = json.loads((self.a8_dir / "a14-notes.json").read_text(encoding="utf-8"))
        self.assertIsNone(notes["into_english"])
        self.assertEqual(
            list(notes["scores"]["to-en"]["greedy"]), ["A14", "NLLB off the shelf"]
        )
        self.assertEqual(
            list(notes["scores"]["to-x"]["beam"]), ["A14", "NLLB off the shelf", "A9"]
        )
        self.assertIn("none (no A13 reverse model)", self.printed)


if __name__ == "__main__":
    unittest.main()
