"""Tests for the parallel-corpus alignment checks.

The important test here is not that the numbers are right on well-formed input.
It is that the checks **separate** aligned data from misaligned data, because
that separation is the entire reason the module exists and the entire reason a
student can trust the result on their own corpus.
"""

import unittest

import pandas as pd

from torchlingo.preprocessing.alignment import (
    align_one_to_one,
    anchor_agreement,
    anchors,
    diagnose_alignment,
    gale_church_align,
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


def _sentences(lengths: list[int], word: str = "word") -> list[str]:
    """Build sentences of controlled character length."""
    return [" ".join([word] * n) for n in lengths]


class GaleChurchAlignTests(unittest.TestCase):
    """The aligner's job is to survive slight segmentation drift."""

    def test_identical_segmentation_aligns_one_to_one(self):
        src = _sentences([3, 9, 4, 12, 6])
        tgt = _sentences([3, 9, 4, 12, 6])
        beads = gale_church_align(src, tgt)
        self.assertEqual(beads, [([i], [i]) for i in range(5)])

    def test_recovers_after_a_dropped_sentence(self):
        """The case that actually occurs: one side lost a line.

        Everything after the gap is off by one. A positional zip would pair
        every later sentence with its neighbour and stay wrong to the end of
        the talk. The aligner must absorb the gap and resynchronize, which is
        the whole reason 13k rows are recoverable at all.
        """
        src = _sentences([12, 3, 14, 30, 6, 40])
        tgt = _sentences([12, 14, 30, 6, 40])  # the short sentence is missing
        pairs = align_one_to_one(src, tgt)
        # Everything after the gap pairs with its own partner, not its neighbour.
        self.assertIn((src[3], tgt[2]), pairs)
        self.assertIn((src[4], tgt[3]), pairs)
        self.assertIn((src[5], tgt[4]), pairs)

    def test_a_long_dropped_sentence_is_handled_worse(self):
        """A known limitation, asserted so it stays known.

        Gale-Church scores a deletion by both a fixed penalty *and* the
        improbability of the missing length, so dropping a long sentence looks
        so unlikely that a poor one-to-one plus a merge can score better. The
        aligner still resynchronizes afterwards, but it emits one bad pair
        around the gap.

        This is tolerable because it is rare and because the corpus-level
        checks are the real gate: the recovered rows have to clear
        `looks_aligned` in aggregate, which they do. Worth knowing before
        trusting the aligner on a corpus with many long omissions.
        """
        src = _sentences([3, 20, 4, 30, 6, 40])
        tgt = _sentences([3, 4, 30, 6, 40])  # the LONG sentence is missing
        pairs = align_one_to_one(src, tgt)
        # It recovers: the tail still pairs correctly.
        self.assertIn((src[4], tgt[3]), pairs)
        self.assertIn((src[5], tgt[4]), pairs)
        # But it mispairs at the gap rather than emitting a deletion.
        self.assertIn((src[1], tgt[1]), pairs)

    def test_a_split_sentence_becomes_a_one_to_two_bead(self):
        src = _sentences([4, 30, 5])
        tgt = _sentences([4, 15, 15, 5])  # the long one was split in two
        beads = gale_church_align(src, tgt)
        shapes = [(len(s), len(t)) for s, t in beads]
        self.assertIn((1, 2), shapes)

    def test_one_to_two_beads_are_not_returned_as_pairs(self):
        """Confident pairings only. A split sentence is dropped, not guessed."""
        src = _sentences([4, 30, 5])
        tgt = _sentences([4, 15, 15, 5])
        pairs = align_one_to_one(src, tgt)
        self.assertNotIn(src[1], [p[0] for p in pairs])

    def test_every_index_is_used_exactly_once(self):
        """A bead path must consume both sides completely and without overlap."""
        src = _sentences([3, 9, 4, 12, 6, 7])
        tgt = _sentences([3, 9, 16, 6, 7])
        beads = gale_church_align(src, tgt)
        used_src = [i for s, _ in beads for i in s]
        used_tgt = [j for _, t in beads for j in t]
        self.assertEqual(used_src, list(range(len(src))))
        self.assertEqual(used_tgt, list(range(len(tgt))))

    def test_alignment_is_monotonic(self):
        """Sentences cannot be reordered, only merged, split or dropped."""
        src = _sentences([5, 10, 3, 8, 12])
        tgt = _sentences([5, 10, 11, 12])
        beads = gale_church_align(src, tgt)
        starts = [s[0] for s, _ in beads if s]
        self.assertEqual(starts, sorted(starts))

    def test_handles_an_empty_side(self):
        self.assertEqual(align_one_to_one([], []), [])
        self.assertEqual(align_one_to_one(["only source"], []), [])

    def test_scales_to_a_realistic_talk(self):
        """Talks run to hundreds of sentences; the DP must stay tractable."""
        lengths = [(i % 17) + 3 for i in range(300)]
        src = _sentences(lengths)
        tgt = _sentences(lengths[:150] + lengths[151:])  # drop one in the middle
        pairs = align_one_to_one(src, tgt)
        self.assertGreater(len(pairs), 250)

    def test_beats_positional_pairing_on_drifted_input(self):
        """The comparison that justifies the algorithm.

        Zipping drifted sentences positionally yields *more* rows than the
        aligner does, and they are wrong. This is the same trade the corpus
        itself presents: fewer correct pairs beat more uncertain ones.
        """
        # Distinctive lengths so a mispairing is unmistakable, and a short
        # omission, which is what segmentation drift actually looks like.
        lengths = [20, 3, 25, 30, 35, 40, 45]
        src = _sentences(lengths)
        tgt = _sentences(lengths[:1] + lengths[2:])  # the short one is dropped

        naive = list(zip(src, tgt))
        aligned = align_one_to_one(src, tgt)

        def mean_ratio(pairs):
            return sum(
                min(len(a), len(b)) / max(len(a), len(b)) for a, b in pairs
            ) / len(pairs)

        self.assertGreater(len(naive), len(aligned))
        self.assertGreater(mean_ratio(aligned), mean_ratio(naive))
        self.assertGreater(mean_ratio(aligned), 0.95)


if __name__ == "__main__":
    unittest.main()
