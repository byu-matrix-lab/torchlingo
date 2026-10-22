"""Lecture 13 (Assignment 13), executed from the notebook: Colab faked, toy size.

The notebook declares ``colab``, so CI's notebook run skips it. These tests run its cells from the
notebook file itself, starting from what Lecture 9's Part B leaves on Drive, which the fixture
produces by running Part B's own cells (see ``test_lecture_09_part_b``). The install cell and the
COMET cell are left out: one is Colab's, and the other installs a package CI does not have.

Two cells are exercises: ``reverse`` and ``combine`` are the student's, and raise
``NotImplementedError`` until written. The tests write them in, one straightforward way each, and
a hypothesis and a prediction, so what is tested is that the notebook's checks accept a right
answer and that the path after them works: the split reversed, the monolingual text filtered
against the split, back-translations written two ways, a training set and a model for each, all
three systems decoded greedily and with beam and compared with the paired bootstrap, the MTEval
files and the agreement count, and the write-up. Further tests check that the notebook as shipped
stops at the first exercise, that it stops without a hypothesis in "If ..., then ..." form or
without a monolingual file, that the Church TM's text is used before the public text's, that the
epochs come from A9's checkpoint, that the write-up runs from Drive alone after a dropped
session, and that the NLLB path trains no reverse model and back-translates through the harness,
which is stubbed: CI has no ``transformers``.

What else changes in the cells is only the epochs and the model's size.
"""

import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import pandas as pd
from notebook_cells import COURSE, code_cells, colab_faked, run_cells, substitute
from test_a9_path import PIECE_MARK, toy_corpus
from test_lecture_09_part_b import (
    A8_EPOCHS,
    CELL_1,
    CONFIG_CELL,
    SCORING_CELL,
    TOY_SIZE,
    TRAINING_CELL,
    part_b,
    write_a8_and_part_a,
)

import torchlingo_harness.nllb

NOTEBOOK = COURSE / "lecture-13-back-translation.ipynb"
SKIPPED = ('%pip install --quiet "torchlingo', "unbabel-comet", "comet-score")
METHODS = ("greedy", "sample")

# A student's answers to the two exercises, in place of the line that raises.
ANSWERS = {
    'raise NotImplementedError("write reverse, then run this cell again")': (
        'return frame.rename(columns={"src": "tgt", "tgt": "src"})[["src", "tgt"]]'
    ),
    'raise NotImplementedError("write combine, then run this cell again")': (
        'new = pd.DataFrame({"src": list(back), "tgt": list(mono)})\n'
        '    new = new[(new.src.str.strip() != "") & ~new.src.isin(held_out)]\n'
        "    return pd.concat([original, new], ignore_index=True)"
    ),
}
HYPOTHESIS = "If I add back-translated data, then BLEU will rise a little"
PREDICTED = {
    'HYPOTHESIS = ""': f"HYPOTHESIS = {HYPOTHESIS!r}",
    'PREDICTION = ""': 'PREDICTION = "sampling helps more"',
}
READ_EPOCHS = 'EPOCHS = len(a9_state["train_losses"])'
SETUP_CELL = "def train("
MONO_CELL = "MONO_FILES = {"
HYPOTHESIS_CELL = 'HYPOTHESIS = ""'
AGREEMENT_CELL = "MTEVAL_MINE ="
WRITE_UP_CELL = "=== For the A13 write-up ==="


def lecture_13_cells(
    answered: bool = True,
    predicted: bool = True,
    one_epoch: bool = True,
    swaps: dict | None = None,
) -> list[tuple[int, str]]:
    """Every code cell but the install and COMET, at toy size and, unless told not to, one epoch."""
    cells = [c for c in code_cells(NOTEBOOK) if not any(m in c[1] for m in SKIPPED)]
    return substitute(
        cells,
        {
            **TOY_SIZE,
            **({READ_EPOCHS: "EPOCHS = 1"} if one_epoch else {}),
            **(ANSWERS if answered else {}),
            **(PREDICTED if predicted else {}),
            **(swaps or {}),
        },
    )


def cells_with(cells: list[tuple[int, str]], *markers: str) -> list[tuple[int, str]]:
    """The cells containing each marker, in notebook order."""
    return sorted(next(c for c in cells if m in c[1]) for m in markers)


def write_mteval(path: Path, sentences, ranks) -> None:
    """An MTEval results file: `source`, the two systems, then a row of ranks per sentence."""
    rows = ["source\t" + "\t".join(METHODS)]
    rows += [f"{s}\t{a}\t{b}" for s, (a, b) in zip(sentences, ranks)]
    path.write_text("\n".join(rows) + "\n", encoding="utf-8")


def build_drive(drive: Path) -> tuple[Path, int, dict]:
    """A9's files as Part B leaves them, and the student's two monolingual files.

    The Church TM file holds half as many new sentences as the training pairs, so the public file
    tops it up. Each also has a repeat, a blank line and a sentence from the test set, and the
    public file repeats one of the TM's sentences: the notebook must leave all of those out.
    """
    a8_dir = drive / "CS479" / "assignment8"
    n_test = write_a8_and_part_a(a8_dir, no_spaces=False)
    with colab_faked(drive):
        run_cells(part_b(CELL_1, CONFIG_CELL, TRAINING_CELL, SCORING_CELL))
    train = pd.read_csv(a8_dir / "train.tsv", sep="\t", dtype=str)
    test = pd.read_csv(a8_dir / "test.tsv", sep="\t", dtype=str)
    val = pd.read_csv(a8_dir / "val.tsv", sep="\t", dtype=str)
    in_split = set(train.tgt) | set(val.tgt) | set(test.tgt)
    new = list(
        dict.fromkeys(s for s in toy_corpus(2000, seed=7).tgt if s not in in_split)
    )
    half = len(train) // 2
    tm_new = new[:half]
    unwanted = ["", test.tgt.iloc[0]]
    for name, lines in (
        ("mono-tm.txt", tm_new + [tm_new[1]] + unwanted),
        ("mono-public.txt", new[half:] + [new[-1], tm_new[0]] + unwanted),
    ):
        (a8_dir / name).write_text("\n".join(lines) + "\n", encoding="utf-8")
    return (
        a8_dir,
        n_test,
        {"train": train, "test": test, "half": half, "tm_new": tm_new},
    )


class TestLecture13(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.drive = Path(cls.tmp.name)
        cls.a8_dir, cls.n_test, fixture = build_drive(cls.drive)
        cls.train, cls.test = fixture["train"], fixture["test"]
        cls.half, cls.tm_new = fixture["half"], fixture["tm_new"]
        with colab_faked(cls.drive):
            cls.ns, cls.printed = run_cells(lecture_13_cells())

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def notes(self) -> dict:
        return json.loads((self.a8_dir / "a13-notes.json").read_text(encoding="utf-8"))

    def test_monolingual_text_is_new_to_the_split(self):
        used = (self.a8_dir / "mono-used.txt").read_text(encoding="utf-8").splitlines()
        self.assertEqual(len(used), len(self.train), "as many as the training pairs")
        self.assertEqual(len(used), len(set(used)), "no repeats")
        self.assertNotIn(self.test.tgt.iloc[0], used)
        self.assertNotIn("", used)

    def test_the_church_tm_comes_first_and_public_text_tops_it_up(self):
        used = (self.a8_dir / "mono-used.txt").read_text(encoding="utf-8").splitlines()
        self.assertEqual(set(used[: self.half]), set(self.tm_new))
        self.assertEqual(
            self.notes()["mono_sources"],
            {"Church TM": self.half, "public text": len(self.train) - self.half},
        )
        self.assertIn("Monolingual sentences used, by source: Church TM", self.printed)

    def test_reverses_the_split_and_trains_the_reverse_model(self):
        reversed_train = pd.read_csv(
            self.a8_dir / "train-reversed.tsv", sep="\t", dtype=str
        )
        self.assertEqual(list(reversed_train.src), list(self.train.tgt))
        self.assertEqual(list(reversed_train.tgt), list(self.train.src))
        self.assertTrue((self.a8_dir / "best-a13-reverse" / "model_best.pt").exists())
        self.assertIn("Back-translator: your own reverse model", self.printed)

    def test_back_translates_every_sentence_two_ways(self):
        back = {}
        for m in METHODS:
            path = self.a8_dir / f"mono-used.bt-{m}.en"
            back[m] = path.read_text(encoding="utf-8").split("\n")[:-1]
            self.assertEqual(len(back[m]), len(self.train), m)
            self.assertFalse(
                [b for b in back[m] if PIECE_MARK in b], "decoded, not pieces"
            )
        self.assertNotEqual(back["greedy"], back["sample"], "sampling is not greedy")
        record = json.loads(
            (self.a8_dir / "mono-used.bt-sample.en.progress.json").read_text()
        )
        self.assertEqual((record["decode_strategy"], record["seed"]), ("sample", 479))

    def test_a_training_set_and_a_model_per_method(self):
        used = (self.a8_dir / "mono-used.txt").read_text(encoding="utf-8").splitlines()
        for m in METHODS:
            combined = pd.read_csv(
                self.a8_dir / f"train-a13-{m}.tsv",
                sep="\t",
                dtype=str,
                keep_default_na=False,
            )
            self.assertEqual(
                list(combined.tgt[: len(self.train)]), list(self.train.tgt)
            )
            self.assertTrue(set(combined.tgt[len(self.train) :]) <= set(used))
            self.assertTrue((self.a8_dir / f"best-a13-{m}" / "model_best.pt").exists())

    def test_three_systems_decoded_both_ways_and_bootstrapped(self):
        names = ["a9", *(f"a13-{m}" for m in METHODS)]
        for decoder in ("greedy", "beam"):
            for name in names:
                hyp = self.a8_dir / f"test-{name}-{decoder}.hyp"
                self.assertEqual(
                    len(hyp.read_text(encoding="utf-8").splitlines()), self.n_test
                )
        notes = self.notes()
        self.assertEqual(
            notes["hyp_files"],
            [f"test-{n}-{d}.hyp" for d in ("greedy", "beam") for n in names],
        )
        scores = notes["scores"]["greedy"]
        self.assertEqual(list(scores), ["A9", "A13 greedy", "A13 sample"])
        self.assertIsNone(
            scores["A9"]["bleu_p"], "the baseline has no p against itself"
        )
        a9_greedy = (self.a8_dir / "test-a9-greedy.hyp").read_text(encoding="utf-8")
        for system in ("A13 greedy", "A13 sample"):
            self.assertTrue(0 < scores[system]["bleu_p"] <= 1)
            self.assertGreaterEqual(scores[system]["chrf_ci"], 0)
            name = system.replace("A13 ", "a13-")
            mine = (self.a8_dir / f"test-{name}-greedy.hyp").read_text(encoding="utf-8")
            if mine == a9_greedy:
                # sacreBLEU alone would call identical systems different, at p = 0.001.
                self.assertEqual(scores[system]["bleu_p"], 1.0, system)
        self.assertEqual(set(notes["methods_p"]), {"greedy", "beam"})
        # A9's greedy decode here is the one A9 itself reported.
        a9 = (self.a8_dir / "test-a9.hyp").read_text(encoding="utf-8").splitlines()
        again = (self.a8_dir / "test-a9-greedy.hyp").read_text(encoding="utf-8")
        self.assertEqual(again.splitlines(), a9)
        for name in ("test.src.txt", "test.ref.txt"):
            lines = (self.a8_dir / name).read_text(encoding="utf-8").splitlines()
            self.assertEqual(len(lines), self.n_test, name)

    def test_writes_the_mteval_files(self):
        folder = self.a8_dir / "a13-mteval"
        reverse_test = pd.read_csv(
            self.a8_dir / "test-reversed.tsv", sep="\t", dtype=str
        )
        source = (folder / "source.txt").read_text(encoding="utf-8").splitlines()
        reference = (folder / "reference.txt").read_text(encoding="utf-8").splitlines()
        self.assertEqual(len(source), 25)
        self.assertTrue(set(source) <= set(reverse_test.src), "in your language")
        self.assertTrue(set(reference) <= set(reverse_test.tgt), "English")
        for m in METHODS:
            lines = (folder / f"{m}.txt").read_text(encoding="utf-8").splitlines()
            self.assertEqual(len(lines), 25, m)
        self.assertIn("Not counted yet", self.printed)

    def test_counts_agreement_from_both_mteval_files(self):
        """Agree, disagree, a tie for one rater, agree: 2 of 3 counted, 4 ranked by both."""
        folder = self.a8_dir / "a13-mteval"
        source = (folder / "source.txt").read_text(encoding="utf-8").splitlines()[:5]
        write_mteval(
            folder / "mteval-mine.tsv", source[:4], [(1, 2), (1, 2), (1, 1), (2, 1)]
        )
        write_mteval(
            folder / "mteval-partner.tsv",
            source,
            [(1, 2), (2, 1), (1, 2), (2, 1), (1, 2)],
        )
        try:
            with colab_faked(self.drive):
                _, printed = run_cells(
                    cells_with(lecture_13_cells(), SETUP_CELL, AGREEMENT_CELL)
                )
            self.assertIn("you agreed on 2 (67%)", printed)
            self.assertEqual(
                self.notes()["rater_agreement"], {"agree": 2, "counted": 3, "ranked": 4}
            )
        finally:
            for name in ("mteval-mine.tsv", "mteval-partner.tsv"):
                (folder / name).unlink()

    def test_agreement_refuses_files_naming_other_systems(self):
        folder = self.a8_dir / "a13-mteval"
        source = (folder / "source.txt").read_text(encoding="utf-8").splitlines()[:2]
        for name in ("mteval-mine.tsv", "mteval-partner.tsv"):
            (folder / name).write_text(
                "source\tsystem 1\tsystem 2\n"
                + "".join(f"{s}\t1\t2\n" for s in source),
                encoding="utf-8",
            )
        try:
            with colab_faked(self.drive), self.assertRaises(SystemExit) as stopped:
                run_cells(cells_with(lecture_13_cells(), SETUP_CELL, AGREEMENT_CELL))
            self.assertIn("must use your two methods' names", str(stopped.exception))
        finally:
            for name in ("mteval-mine.tsv", "mteval-partner.tsv"):
                (folder / name).unlink()

    def test_prints_the_write_up_block_and_the_rerun_record(self):
        self.assertIn("=== For the A13 write-up ===", self.printed)
        self.assertIn(
            f"Hypothesis, written before training: {HYPOTHESIS}", self.printed
        )
        self.assertIn("greedy  A13 sample", self.printed)
        self.assertIn("beam    A9", self.printed)
        self.assertIn("helped most, by chrF", self.printed)
        self.assertIn("hurt most, by chrF", self.printed)
        rerun = json.loads((self.a8_dir / "a13-rerun.json").read_text(encoding="utf-8"))
        self.assertEqual(rerun["bt_methods"], list(METHODS))
        self.assertEqual(rerun["back_translator"], "your reverse model")
        self.assertEqual(rerun["seed"], 479)

    def test_the_write_up_runs_after_a_dropped_session(self):
        """A new session runs the setup cell, then the write-up: nothing else in memory."""
        with colab_faked(self.drive):
            _, printed = run_cells(
                cells_with(lecture_13_cells(), SETUP_CELL, WRITE_UP_CELL)
            )
        self.assertIn("A13 greedy", printed)
        self.assertIn("its BLEU into English on your test set", printed)
        self.assertIn(HYPOTHESIS, printed)

    def test_rerunning_the_training_cell_after_both_finish_trains_nothing(self):
        """The notebook says a finished training is skipped when the cell is run again."""
        model = self.a8_dir / "best-a13-sample" / "model_best.pt"
        before = model.stat().st_mtime_ns
        with colab_faked(self.drive):
            _, printed = run_cells(
                cells_with(
                    lecture_13_cells(),
                    SETUP_CELL,
                    'for m in load_notes()["bt_methods"]:\n    print(f"=== English',
                )
            )
        self.assertEqual(
            model.stat().st_mtime_ns, before, "the best model was not rewritten"
        )
        self.assertEqual(printed.count("Epochs: 1"), 2)

    def test_epochs_come_from_a9s_checkpoint(self):
        with colab_faked(self.drive):
            ns, _ = run_cells(cells_with(lecture_13_cells(one_epoch=False), SETUP_CELL))
        self.assertEqual(ns["EPOCHS"], A8_EPOCHS, "A9 trained as many epochs as A8")

    def test_without_a_prediction_it_stops(self):
        cells = cells_with(
            lecture_13_cells(predicted=False), SETUP_CELL, HYPOTHESIS_CELL
        )
        with colab_faked(self.drive), self.assertRaises(AssertionError) as stopped:
            run_cells(cells)
        self.assertIn("Write your hypothesis", str(stopped.exception))

    def test_a_hypothesis_not_in_if_then_form_is_refused(self):
        bad = {
            'HYPOTHESIS = ""': 'HYPOTHESIS = "back-translation will help"',
            'PREDICTION = ""': 'PREDICTION = "sampling helps more"',
        }
        cells = cells_with(
            lecture_13_cells(predicted=False, swaps=bad),
            SETUP_CELL,
            "save_note(hypothesis=",
        )
        with colab_faked(self.drive), self.assertRaises(AssertionError) as stopped:
            run_cells(cells)
        self.assertIn('"If ..., then ..."', str(stopped.exception))

    def test_without_a_monolingual_file_it_says_what_to_do(self):
        cells = substitute(
            cells_with(lecture_13_cells(), SETUP_CELL, MONO_CELL),
            {
                'A8_DIR / "mono-tm.txt"': 'A8_DIR / "no-such-tm.txt"',
                'A8_DIR / "mono-public.txt"': 'A8_DIR / "no-such-public.txt"',
            },
        )
        with colab_faked(self.drive), self.assertRaises(SystemExit) as stopped:
            run_cells(cells)
        self.assertIn(
            "Put a file of sentences in your language", str(stopped.exception)
        )

    def test_as_shipped_it_stops_at_the_first_exercise(self):
        with colab_faked(self.drive), self.assertRaises(NotImplementedError) as stopped:
            run_cells(lecture_13_cells(answered=False))
        self.assertIn("write reverse", str(stopped.exception))

    def test_combine_check_rejects_an_answer_that_keeps_held_out_sentences(self):
        """The toy case fails a combine that forgets held_out, with the message to fix it."""
        cells = lecture_13_cells()
        setup_cell = next(c for c in cells if SETUP_CELL in c[1])
        combine_cell = next(c for c in cells if "def combine(" in c[1])
        forgetful = combine_cell[1].replace(" & ~new.src.isin(held_out)", "")
        with colab_faked(self.drive), self.assertRaises(AssertionError) as stopped:
            run_cells([setup_cell, (combine_cell[0], forgetful)])
        self.assertIn("combine is not right yet", str(stopped.exception))


class _FakeNLLB:
    """Stands in for torchlingo_harness.nllb's translator: English made by capitalizing."""

    def __call__(self, sentences, decode_strategy="greedy", seed=0, **options):
        return [f"{decode_strategy.upper()} {s}" for s in sentences]

    def settings(self, **decoding):
        return {"translator": "fake-nllb", **decoding}


class TestNLLBPath(unittest.TestCase):
    """BACK_TRANSLATOR = "nllb": no reverse model, back-translations from the harness."""

    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.drive = Path(cls.tmp.name)
        cls.a8_dir, _, _ = build_drive(cls.drive)
        swaps = {
            'BACK_TRANSLATOR = "own"': 'BACK_TRANSLATOR = "nllb"',
            "NLLB_CODE = None": 'NLLB_CODE = "tel_Telu"',
        }
        cls.loaded = []

        def load_nllb(src, tgt, device=None):
            cls.loaded.append((src, tgt))
            return _FakeNLLB()

        with (
            colab_faked(cls.drive),
            mock.patch.object(torchlingo_harness.nllb, "load_nllb", load_nllb),
        ):
            cls.ns, cls.printed = run_cells(lecture_13_cells(swaps=swaps))

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_trains_no_reverse_model(self):
        self.assertIn(
            "NLLB back-translates, so there is no reverse model", self.printed
        )
        self.assertFalse((self.a8_dir / "best-a13-reverse").exists())

    def test_loads_nllb_once_from_your_language_into_english(self):
        self.assertEqual(self.loaded, [("tel_Telu", "eng_Latn")])

    def test_back_translations_come_from_nllb_and_resume_like_any_other(self):
        back = (self.a8_dir / "mono-used.bt-sample.en").read_text(encoding="utf-8")
        self.assertTrue(back.startswith("SAMPLE "))
        record = json.loads(
            (self.a8_dir / "mono-used.bt-sample.en.progress.json").read_text()
        )
        self.assertEqual(
            (record["translator"], record["decode_strategy"]), ("fake-nllb", "sample")
        )

    def test_the_write_up_names_nllb(self):
        self.assertIn("Back-translator: NLLB-200 (tel_Telu)", self.printed)
        rerun = json.loads((self.a8_dir / "a13-rerun.json").read_text(encoding="utf-8"))
        self.assertEqual(rerun["back_translator"], "NLLB-200 (tel_Telu)")


if __name__ == "__main__":
    unittest.main()
