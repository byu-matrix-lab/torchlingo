"""Tests for the parallel-corpus alignment checks.

The important test here is not that the numbers are right on well-formed input.
It is that the checks **separate** aligned data from misaligned data, because
that separation is the entire reason the module exists and the entire reason a
student can trust the result on their own corpus.
"""

import unittest

import pandas as pd

from torchlingo.preprocessing.alignment import (
    anchor_agreement,
    anchors,
    diagnose_alignment,
    length_correlation,
    shuffle_target_side,
)

_NAMES = [
    "Stephen",
    "Maria",
    "Jonas",
    "Priya",
    "Ahmed",
    "Lucia",
    "Kenji",
    "Olena",
    "Diego",
    "Fatima",
]


def _parallel_corpus(n: int = 60) -> pd.DataFrame:
    """Build a synthetic corpus that is genuinely parallel.

    Lengths vary row to row and the two sides track each other, and each row
    carries a *distinctive* name and year. The distinctiveness matters: a token
    that appears in every row is shared by every pairing, correct or not, and
    would make the anchor check report perfect agreement on scrambled data.
    See ``test_anchor_check_is_blind_to_a_corpus_wide_name``.
    """
    rows = []
    for i in range(n):
        words = "word " * (i % 12 + 1)
        name = _NAMES[i % len(_NAMES)]
        rows.append(
            {
                "src": f"{name} said {words}in {2000 + i}.",
                "tgt": f"{name} dijo {words}en {2000 + i}.",
            }
        )
    return pd.DataFrame(rows)


class AnchorsTests(unittest.TestCase):
    def test_finds_names_and_numbers(self):
        self.assertEqual(anchors("Stephen spoke in 2010."), {"Stephen", "2010"})

    def test_skips_short_capitalized_words(self):
        """ "The" capitalizes by grammar, not because it is a name."""
        self.assertEqual(anchors("The cat sat."), set())

    def test_empty_text_has_no_anchors(self):
        self.assertEqual(anchors(""), set())


class LengthCorrelationTests(unittest.TestCase):
    def test_parallel_text_correlates_strongly(self):
        self.assertGreater(length_correlation(_parallel_corpus()), 0.9)

    def test_ignores_rows_with_an_empty_side(self):
        """An empty side carries no length signal and must not count as short."""
        frame = _parallel_corpus(20)
        frame.loc[0, "tgt"] = ""
        self.assertGreater(length_correlation(frame), 0.9)

    def test_returns_zero_when_nothing_is_scorable(self):
        frame = pd.DataFrame({"src": [""], "tgt": [""]})
        self.assertEqual(length_correlation(frame), 0.0)

    def test_constant_lengths_do_not_crash(self):
        """Zero variance makes Pearson undefined; it must not propagate NaN."""
        frame = pd.DataFrame({"src": ["a b"] * 5, "tgt": ["x y"] * 5})
        self.assertEqual(length_correlation(frame), 0.0)


class AnchorAgreementTests(unittest.TestCase):
    def test_aligned_rows_agree(self):
        agreement, scorable = anchor_agreement(_parallel_corpus())
        self.assertEqual(agreement, 1.0)
        self.assertEqual(scorable, 60)

    def test_rows_without_anchors_are_not_scored(self):
        """A row with nothing checkable is not evidence either way."""
        frame = pd.DataFrame({"src": ["the cat sat"], "tgt": ["el gato"]})
        agreement, scorable = anchor_agreement(frame)
        self.assertEqual(scorable, 0)
        self.assertEqual(agreement, 0.0)

    def test_respects_the_sample_cap(self):
        _, scorable = anchor_agreement(_parallel_corpus(100), sample=10)
        self.assertEqual(scorable, 10)


class DiagnoseAlignmentTests(unittest.TestCase):
    """The separation property, which is what the module is for."""

    def test_aligned_corpus_passes_both_checks(self):
        report = diagnose_alignment(_parallel_corpus())
        self.assertTrue(report.looks_aligned())

    def test_misaligned_corpus_fails(self):
        """The regression that shipped once and must never ship again.

        `shuffle_target_side` reproduces it exactly: two intact columns, paired
        up wrongly. If this ever passes, the checks have stopped working and
        the corpus tests that depend on them are decorative.
        """
        broken = shuffle_target_side(_parallel_corpus())
        report = diagnose_alignment(broken)
        self.assertFalse(report.looks_aligned())
        self.assertLess(report.anchor_agreement, 0.25)

    def test_misaligned_corpus_still_looks_fine_superficially(self):
        """Why the checks are needed at all.

        The broken corpus has the same shape, the same row count, no empty
        cells and plausible text on both sides. Every cheap structural check
        passes. Only the two alignment probes notice.
        """
        broken = shuffle_target_side(_parallel_corpus())
        self.assertEqual(len(broken), 60)
        self.assertEqual(list(broken.columns), ["src", "tgt"])
        self.assertFalse((broken["src"].str.strip() == "").any())
        self.assertFalse((broken["tgt"].str.strip() == "").any())

    def test_report_carries_row_counts(self):
        report = diagnose_alignment(_parallel_corpus(30))
        self.assertEqual(report.rows, 30)
        self.assertEqual(report.scorable_rows, 30)

    def test_thresholds_are_adjustable(self):
        report = diagnose_alignment(_parallel_corpus())
        self.assertTrue(report.looks_aligned())
        self.assertFalse(report.looks_aligned(min_agreement=1.01))

    def test_anchor_check_is_blind_to_a_corpus_wide_name(self):
        """A known limitation, asserted so it stays known.

        The anchor check compares *which* names and numbers the two sides
        share. A token appearing in every row is shared by every pairing,
        right or wrong, so it carries no alignment signal at all. A corpus of
        one speaker's talks, where that speaker is named in most lines, is
        exactly this case.

        Here the length check still catches the scrambling. That is the
        argument for running both rather than picking one.
        """
        frame = pd.DataFrame(
            [
                {
                    "src": f"Stephen said {'word ' * (i % 9 + 1)}today.",
                    "tgt": f"Stephen dijo {'word ' * (i % 9 + 1)}hoy.",
                }
                for i in range(40)
            ]
        )
        broken = shuffle_target_side(frame)
        agreement, _ = anchor_agreement(broken)
        self.assertEqual(agreement, 1.0, "the shared name defeats the anchor check")
        self.assertLess(
            length_correlation(broken),
            length_correlation(frame),
            "the length check is what still notices",
        )


class ShuffleTargetSideTests(unittest.TestCase):
    def test_no_row_keeps_its_own_translation(self):
        frame = _parallel_corpus(10)
        broken = shuffle_target_side(frame)
        self.assertFalse((frame["tgt"] == broken["tgt"]).any())

    def test_source_side_is_untouched(self):
        frame = _parallel_corpus(10)
        broken = shuffle_target_side(frame)
        self.assertTrue((frame["src"] == broken["src"]).all())

    def test_the_same_sentences_are_still_present(self):
        """Each column stays individually intact; only the pairing breaks."""
        frame = _parallel_corpus(10)
        broken = shuffle_target_side(frame)
        self.assertEqual(sorted(frame["tgt"]), sorted(broken["tgt"]))

    def test_input_is_not_mutated(self):
        frame = _parallel_corpus(5)
        before = list(frame["tgt"])
        shuffle_target_side(frame)
        self.assertEqual(list(frame["tgt"]), before)


if __name__ == "__main__":
    unittest.main()
