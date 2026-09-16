"""Guard the shipped example corpus against silent misalignment.

`data/example.tsv` once shipped as a two-column TSV that looked parallel and was
not: the two columns were independent documents zipped together, so the rows
were not translations of each other. Nothing caught it, and a student training
on it would have seen a model that refused to learn with no way to tell bad data
from their own mistake.

These tests are the check that was missing. They are cheap and they fail loudly.
"""

import re
import unittest
from pathlib import Path

import pandas as pd

CORPUS = Path(__file__).resolve().parent.parent / "data" / "example.tsv"

# Both thresholds sit well below the values the repaired corpus achieves
# (0.97 correlation, 0.39 agreement) and far above what the broken one managed
# (0.001 and 0.014). Anything landing between the two is a real regression.
MIN_LENGTH_CORRELATION = 0.80
MIN_ANCHOR_AGREEMENT = 0.25
MIN_ROWS = 50000

_NAMES = re.compile(r"\b[A-Z][a-z]{3,}\b")
_DIGITS = re.compile(r"\d+")


def _anchors(text: str) -> set[str]:
    """Extract tokens that should survive translation."""
    return set(_NAMES.findall(text)) | set(_DIGITS.findall(text))


@unittest.skipUnless(CORPUS.exists(), "example corpus not present in this checkout")
class TestExampleCorpusAlignment(unittest.TestCase):
    """The shipped corpus must actually be parallel."""

    @classmethod
    def setUpClass(cls):
        cls.frame = pd.read_csv(CORPUS, sep="\t", dtype=str, keep_default_na=False)

    def test_has_expected_columns(self):
        """src and tgt come first so positional readers keep working.

        `talk` and `kind` are additive metadata: which talk a pair came from,
        and whether it is transcript, title or description.
        """
        self.assertEqual(list(self.frame.columns), ["src", "tgt", "talk", "kind"])

    def test_talk_ids_are_present_and_plural(self):
        """Without talk ids a held-out split can only be taken by sentence.

        Consecutive sentences in a transcript share a speaker, topic and
        vocabulary, so a random sentence split leaks and flatters any model
        evaluated on it.
        """
        self.assertGreater(self.frame["talk"].nunique(), 100)
        self.assertFalse((self.frame["talk"].str.strip() == "").any())

    def test_kinds_are_known_values(self):
        """Anything else means the generator changed without the tests noticing."""
        self.assertEqual(
            set(self.frame["kind"].unique()), {"transcript", "title", "description"}
        )

    def test_transcript_dominates(self):
        """Titles and descriptions are two rows per talk; speech is the corpus."""
        share = (self.frame["kind"] == "transcript").mean()
        self.assertGreater(share, 0.95)

    def test_a_talk_split_leaks_almost_nothing(self):
        """The property that makes talk ids worth carrying.

        Splitting by talk should leave very few held-out sentences that also
        appear in training. What remains is short formulaic speech -- "Thank
        you." and the like -- not topical overlap.
        """
        talks = sorted(self.frame["talk"].unique())
        cut = int(len(talks) * 0.9)
        train = self.frame[self.frame["talk"].isin(set(talks[:cut]))]
        held_out = self.frame[~self.frame["talk"].isin(set(talks[:cut]))]
        overlap = held_out["src"].isin(set(train["src"])).mean()
        self.assertLess(overlap, 0.05)

    def test_has_enough_rows_to_train_on(self):
        """A corpus this small would not support the tutorials."""
        self.assertGreaterEqual(len(self.frame), MIN_ROWS)

    def test_no_empty_rows(self):
        """Empty sides would silently become NaN and poison a batch."""
        self.assertFalse((self.frame["src"].str.strip() == "").any())
        self.assertFalse((self.frame["tgt"].str.strip() == "").any())

    def test_sentence_lengths_correlate(self):
        """Parallel text has strongly correlated sentence lengths.

        A genuinely aligned corpus scores near 0.97 here. The misaligned version
        of this file scored 0.001 — the signature of unrelated sentence pairs.
        """
        src_len = self.frame["src"].str.split().str.len()
        tgt_len = self.frame["tgt"].str.split().str.len()
        usable = (src_len > 0) & (tgt_len > 0)
        correlation = src_len[usable].corr(tgt_len[usable])
        self.assertGreater(correlation, MIN_LENGTH_CORRELATION)

    def test_rows_share_names_and_numbers(self):
        """Proper nouns and numbers should survive translation.

        Scored only on rows whose source side actually contains such a token.
        The misaligned version scored 0.014 here.
        """
        hits = total = 0
        for src, tgt in zip(self.frame["src"], self.frame["tgt"]):
            source_anchors = _anchors(src)
            if not source_anchors:
                continue
            total += 1
            hits += bool(source_anchors & _anchors(tgt))
        self.assertGreater(total, 0, "no rows carried a checkable anchor token")
        self.assertGreater(hits / total, MIN_ANCHOR_AGREEMENT)

    def test_sides_are_not_identical(self):
        """Catch a corpus accidentally rebuilt with one language twice."""
        identical = (self.frame["src"] == self.frame["tgt"]).mean()
        self.assertLess(identical, 0.10)


if __name__ == "__main__":
    unittest.main()
