"""Lecture 9's Part A, executed from the notebook: Colab faked, toy size.

Part A is the in-class half of the Lecture 9 notebook, and the A9 numbers start here: a wrong
``unknown_rate`` would put wrong ``<unk>`` rates in every student's A9 write-up without an error.
These tests run Part A's cells from the notebook file itself, on a toy split standing in for what
the A8 kickoff leaves in ``CS479/assignment8``, with the student's answer and choices written in.

They check that a correct ``unknown_rate`` passes the notebook's checks and a likely wrong one is
told which number is off; that the notebook as shipped stops at the exercise; that Step 6 shows
the A8 BLEU from ``test.hyp`` before the prediction; and that the settings are not saved until
a prediction is written.

The vocabulary sizes on offer (4,000 to 16,000 pieces) are larger than a toy corpus allows, so
the size check is swapped for an 80-piece vocabulary; see ``notebook_cells`` for why each swap
must match the notebook.
"""

import json
import tempfile
import unittest
from pathlib import Path

from notebook_cells import (
    COURSE,
    cell_containing,
    code_cells,
    colab_faked,
    run_cells,
    substitute,
)
from test_a9_path import SEED, toy_corpus

from torchlingo.preprocessing import split_exact

NOTEBOOK = COURSE / "lecture-09-subword-tokenization.ipynb"

RAISE = 'raise NotImplementedError("write unknown_rate, then run this cell again")'
ANSWER = (
    "unknown = total = 0\n"
    "    for sentence in sentences:\n"
    "        ids = vocab.encode(sentence, add_special_tokens=False)\n"
    "        unknown += sum(i == vocab.unk_idx for i in ids)\n"
    "        total += len(ids)\n"
    "    return unknown, total"
)
CHOICES = {
    RAISE: ANSWER,
    "EXPECTED_UNK_PERCENT = None": "EXPECTED_UNK_PERCENT = 5.0",
}
VOCAB_CHOICES = {
    "VOCAB_SIZE = 8_000": "VOCAB_SIZE = 80",
    'assert VOCAB_SIZE in (4_000, 8_000, 16_000), "pick one of the three sizes"': "",
    'WHY = ""': 'WHY = "a toy corpus"',
}
PREDICTION = {'BLEU_PREDICTION = ""': 'BLEU_PREDICTION = "up a little"'}

# Part A's cells, by what each one says, in notebook order.
DRIVE_CELL = "the A8 kickoff notebook wrote your split"
COUNT_CELL = "def unknown_rate"
VOCAB_CELL = "train_sentencepiece([A8_DIR"
PIECES_CELL = "piece_unk = {}"
MEASURE_CELL = "MAX_DECODE = int(piece_len.max())"
PREDICT_CELL = "BLEU_PREDICTION ="


def part_a(*markers: str, swaps: dict[str, str]) -> list[tuple[int, str]]:
    cells = sorted(cell_containing(code_cells(NOTEBOOK), m) for m in markers)
    return substitute(cells, swaps)


class TestPartA(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.drive = Path(cls.tmp.name)
        cls.a8_dir = cls.drive / "CS479" / "assignment8"
        cls.a8_dir.mkdir(parents=True)
        train, val, test = split_exact(
            toy_corpus(600, seed=0), n_val=40, n_test=40, seed=SEED
        )
        for name, frame in (("train", train), ("val", val), ("test", test)):
            frame.to_csv(cls.a8_dir / f"{name}.tsv", sep="\t", index=False)
        # A8's translations: the references themselves, so A8's BLEU is exactly 100.
        (cls.a8_dir / "test.hyp").write_text(
            "\n".join(test.tgt) + "\n", encoding="utf-8"
        )

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def run_part_a(self, *markers: str, swaps: dict[str, str]):
        with colab_faked(self.drive):
            return run_cells(part_a(*markers, swaps=swaps))

    def test_a_correct_count_passes_and_reports(self):
        ns, printed = self.run_part_a(DRIVE_CELL, COUNT_CELL, swaps=CHOICES)
        self.assertIn("You guessed 5.0% on the target side.", printed)
        self.assertEqual(
            ns["unknown_rate"](ns["toy"], ["red qzxvw fish", "qzxvw"]), (2, 4)
        )

    def test_as_shipped_it_stops_at_the_exercise(self):
        with self.assertRaises(NotImplementedError) as stopped:
            self.run_part_a(DRIVE_CELL, COUNT_CELL, swaps={})
        self.assertIn("write unknown_rate", str(stopped.exception))

    def test_counting_sentences_is_told_the_second_number_is_off(self):
        careless = {**CHOICES, RAISE: ANSWER.replace("total += len(ids)", "total += 1")}
        with self.assertRaises(AssertionError) as stopped:
            self.run_part_a(DRIVE_CELL, COUNT_CELL, swaps=careless)
        self.assertIn("The second number counts every token", str(stopped.exception))

    def test_counting_special_tokens_is_caught(self):
        careless = {**CHOICES, RAISE: ANSWER.replace("add_special_tokens=False", "")}
        with self.assertRaises(AssertionError) as stopped:
            self.run_part_a(DRIVE_CELL, COUNT_CELL, swaps=careless)
        self.assertIn("without <sos> and <eos>", str(stopped.exception))

    def test_step_6_shows_a8_bleu_before_the_prediction_and_saves_after_it(self):
        cells = (DRIVE_CELL, COUNT_CELL, VOCAB_CELL, PIECES_CELL, MEASURE_CELL)
        choices = {**CHOICES, **VOCAB_CHOICES}
        _, printed = self.run_part_a(*cells, swaps=choices)
        self.assertIn("Your A8 BLEU, from test.hyp: 100.00", printed)

        with self.assertRaises(AssertionError) as stopped:
            self.run_part_a(*cells, PREDICT_CELL, swaps=choices)
        self.assertIn("write your prediction", str(stopped.exception))

        self.run_part_a(*cells, PREDICT_CELL, swaps={**choices, **PREDICTION})
        settings = json.loads(
            (self.a8_dir / "a9-settings.json").read_text(encoding="utf-8")
        )
        self.assertEqual(settings["bleu_prediction"], "up a little")
        self.assertEqual(settings["vocab_size"], 80)


if __name__ == "__main__":
    unittest.main()
