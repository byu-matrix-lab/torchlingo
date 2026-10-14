"""Lecture 4's two notebooks, executed from the notebook files with answers written in.

Both declare ``exercises``, so CI's notebook run skips them: each stops until a student writes
something. These tests write it in and run the rest.

**The TMX cleaning activity** also installs Translate Toolkit, which CI does not have. Its
``tmxfile`` is stubbed with ``xml.etree``, which gives the same ten pairs on the notebook's sample
(checked against Translate Toolkit itself when this test was written), so what is tested is the
notebook's own path: the prediction that stops while empty, the ``PATTERN`` that stops while
``None``, the naive write that misaligns, the repair that realigns, and the checker before and
after two cleaners.

**The regular-expression reference** stops at each exercise until its pattern is written, then
checks the answer; the tests check that a sound answer to every exercise passes, that the
notebook as shipped stops at the first, and that a wrong pattern is refused with what it matched.
"""

import os
import sys
import tempfile
import types
import unittest
import xml.etree.ElementTree as ET
from unittest import mock

from notebook_cells import COURSE, cell_containing, code_cells, run_cells, substitute

TMX_NOTEBOOK = COURSE / "lecture-04-tmx-cleaning.ipynb"
REGEX_NOTEBOOK = COURSE / "lecture-04-regex-refresher.ipynb"

PREDICTED = {'MY_PREDICTION = ""': 'MY_PREDICTION = "10 and 10"'}
LINE_BREAKS = {"PATTERN = None": 'PATTERN = r"[\\r\\n]+"'}
CLEANERS = {
    "def clean_one(s):\n    return s": (
        'def clean_one(s):\n    return re.sub(r"\\s+", " ", s).strip()'
    ),
    "def clean_two(s):\n    return s": (
        'def clean_two(s):\n    return re.sub(r"<[^>]+>", "", s)'
    ),
}


def stub_translate_toolkit() -> dict:
    """``translate.storage.tmx`` as modules, with ``tmxfile`` parsed by ``xml.etree``."""

    class tmxfile:
        def __init__(self, f, source_lang, target_lang):
            self.pairs = [
                tuple(tuv.find("seg").text or "" for tuv in tu.findall("tuv"))
                for tu in ET.parse(f).getroot().iter("tu")
            ]

        def unit_iter(self):
            for source, target in self.pairs:
                yield types.SimpleNamespace(source=source, target=target)

    tmx = types.ModuleType("translate.storage.tmx")
    tmx.tmxfile = tmxfile
    return {
        "translate": types.ModuleType("translate"),
        "translate.storage": types.ModuleType("translate.storage"),
        "translate.storage.tmx": tmx,
    }


class TestTmxCleaning(unittest.TestCase):
    def run_tmx(self, swaps: dict) -> tuple[dict, str]:
        """Every code cell but the install, in a scratch directory, with ``swaps`` applied."""
        cells = [c for c in code_cells(TMX_NOTEBOOK) if "pip install" not in c[1]]
        scratch = tempfile.TemporaryDirectory()
        self.addCleanup(scratch.cleanup)
        previous = os.getcwd()
        os.chdir(scratch.name)
        self.addCleanup(os.chdir, previous)
        with mock.patch.dict(sys.modules, stub_translate_toolkit()):
            return run_cells(substitute(cells, swaps) if swaps else cells)

    def test_runs_to_the_end_with_answers_written_in(self):
        namespace, printed = self.run_tmx({**PREDICTED, **LINE_BREAKS, **CLEANERS})

        self.assertEqual(len(namespace["pairs"]), 10)
        self.assertIn("12 lines in english.txt", printed)  # the naive write misaligns
        self.assertIn("aligned? False", printed)
        self.assertIn("you predicted: 10 and 10", printed)
        self.assertEqual((len(namespace["en"]), len(namespace["tx"])), (10, 10))
        self.assertIn("aligned? True", printed)
        self.assertIn("wanted: 10 lines in each file", printed)

        before = {step for _, step, _ in namespace["inspect"](namespace["repaired"])}
        after = {step for _, step, _ in namespace["remaining"]}
        self.assertEqual(before, {"step 2", "step 4", "step 6", "step 10", "step 11"})
        # Two string cleaners fix steps 4 and 6; dropping pairs is not theirs to do.
        self.assertEqual(after, {"step 2", "step 10", "step 11"})
        self.assertIn("10 pairs, no third-party package involved", printed)

    def test_as_shipped_it_stops_at_the_prediction(self):
        with self.assertRaises(AssertionError) as stopped:
            self.run_tmx({})
        self.assertIn("MY_PREDICTION", str(stopped.exception))

    def test_without_a_pattern_it_stops_at_the_repair(self):
        with self.assertRaises(NotImplementedError) as stopped:
            self.run_tmx(PREDICTED)
        self.assertIn("PATTERN is still None", str(stopped.exception))

    def test_a_pattern_missing_the_bare_carriage_return_stays_misaligned(self):
        """Unit 9's bare \\r is why the hint says the two units differ."""
        newline_only = {"PATTERN = None": 'PATTERN = r"\\n+"'}
        namespace, _ = self.run_tmx({**PREDICTED, **newline_only})
        self.assertEqual(len(namespace["tx"]), 10)
        self.assertNotEqual(len(namespace["en"]), 10)


REGEX_ANSWERS = {
    "pattern_1 = None": 'pattern_1 = r"fox"',
    "pattern_2a = None": 'pattern_2a = r"d.g"',
    "pattern_2b = None": 'pattern_2b = r"[bfd]o"',
    "start_pattern = None": 'start_pattern = r"^The\\s"',
    "digit_pattern = None": 'digit_pattern = r"\\d\\d"',
    "digit_seq_pattern = None": 'digit_seq_pattern = r"\\d{3,}"',
    "url_pattern = None": 'url_pattern = r"(https?)://([\\w.-]+)"',
    "space_pattern = None": 'space_pattern = r"\\s+"',
    "tag_pattern = None": 'tag_pattern = r"<[^>]+>"',
}


class TestRegexReference(unittest.TestCase):
    def run_regex(self, answers: dict) -> tuple[dict, str]:
        cells = code_cells(REGEX_NOTEBOOK)
        return run_cells(substitute(cells, answers) if answers else cells)

    def test_every_exercise_passes_with_a_sound_answer(self):
        namespace, printed = self.run_regex(REGEX_ANSWERS)
        self.assertIn("starting at index 16 and ending at index 19", printed)
        self.assertEqual(namespace["collapsed"], "Study the word. Then pray.")
        self.assertEqual(namespace["stripped"], "Press Save, then Close.")

    def test_as_shipped_it_stops_at_the_first_exercise(self):
        cells = code_cells(REGEX_NOTEBOOK)
        first = cells.index(cell_containing(cells, "pattern_1 = None"))
        run_cells(cells[:first])  # everything before it runs
        with self.assertRaises(NotImplementedError) as stopped:
            run_cells(cells[: first + 1])
        self.assertIn("Replace None with your pattern", str(stopped.exception))

    def test_a_greedy_tag_pattern_is_refused_with_what_it_matched(self):
        greedy = {**REGEX_ANSWERS, "tag_pattern = None": 'tag_pattern = r"<.*>"'}
        with self.assertRaises(AssertionError) as stopped:
            self.run_regex(greedy)
        self.assertIn(
            "your pattern matched ['<b>Save</b>, then <i>Close</i>']",
            str(stopped.exception),
        )

    def test_an_unanchored_start_pattern_is_refused(self):
        loose = {**REGEX_ANSWERS, "start_pattern = None": 'start_pattern = r"The\\s"'}
        with self.assertRaises(AssertionError) as stopped:
            self.run_regex(loose)
        self.assertIn("tie it to the start", str(stopped.exception))


if __name__ == "__main__":
    unittest.main()
