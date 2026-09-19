"""Guard the shipped example corpus against silent misalignment.

`data/example.tsv` once shipped as a two-column TSV that looked parallel and was
not: the two columns were independent documents zipped together, so the rows
were not translations of each other. Nothing caught it, and a student training
on it would have seen a model that refused to learn with no way to tell bad data
from their own mistake.

These tests are the check that was missing. They are cheap and they fail loudly.
"""

import unittest
from pathlib import Path

import pandas as pd

from torchlingo.preprocessing.alignment import (
    anchor_agreement,
    diagnose_alignment,
    length_correlation,
    shuffle_target_side,
)

CORPUS = Path(__file__).resolve().parent.parent / "data" / "example.tsv"


def is_available(path: Path) -> bool:
    """Report whether a file holds real content rather than a Git LFS pointer.

    `data/example.tsv` and `data/pretrained/model.pt` are tracked in Git LFS,
    and CI checks out *without* it on purpose: they are tens of megabytes, they
    change rarely, and fetching them on every run costs time and bandwidth quota
    for the jobs that never read them.

    In that checkout the files still exist and are still readable. They just
    hold about 130 bytes of pointer text starting `version https://git-lfs...`.
    Handing that to `pd.read_csv` fails with a complaint about column counts,
    which tells whoever reads the CI log nothing about the real cause. So these
    tests ask first and skip. A normal clone has the bytes and runs them all.

    Args:
        path (Path): File to test.

    Returns:
        bool: True if the file exists and its content has been fetched.
    """
    if not path.exists():
        return False
    with path.open("rb") as handle:
        return not handle.read(23).startswith(b"version https://git-lfs")


# Both thresholds sit well below the values the repaired corpus achieves
# (0.97 correlation, 0.39 agreement) and far above what the broken one managed
# (0.001 and 0.014). Anything landing between the two is a real regression.
MIN_LENGTH_CORRELATION = 0.80
MIN_ANCHOR_AGREEMENT = 0.25
MIN_ROWS = 50000


@unittest.skipUnless(
    is_available(CORPUS), "example corpus not fetched in this checkout (Git LFS)"
)
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
        self.assertGreater(length_correlation(self.frame), MIN_LENGTH_CORRELATION)

    def test_rows_share_names_and_numbers(self):
        """Proper nouns and numbers should survive translation.

        Scored only on rows whose source side actually contains such a token.
        The misaligned version scored 0.014 here.
        """
        agreement, scorable = anchor_agreement(self.frame, sample=len(self.frame))
        self.assertGreater(scorable, 0, "no rows carried a checkable anchor token")
        self.assertGreater(agreement, MIN_ANCHOR_AGREEMENT)

    def test_scrambling_this_corpus_would_be_caught(self):
        """The checks above must actually discriminate, not just pass.

        A threshold that the shipped corpus clears proves nothing on its own if
        a broken corpus would clear it too. Rotating the target side by one row
        leaves every structural property intact and breaks every pairing, and
        the checks have to notice.
        """
        report = diagnose_alignment(
            shuffle_target_side(self.frame), sample=len(self.frame)
        )
        self.assertFalse(
            report.looks_aligned(MIN_LENGTH_CORRELATION, MIN_ANCHOR_AGREEMENT)
        )

    def test_sides_are_not_identical(self):
        """Catch a corpus accidentally rebuilt with one language twice."""
        identical = (self.frame["src"] == self.frame["tgt"]).mean()
        self.assertLess(identical, 0.10)


PRETRAINED = Path(__file__).resolve().parent.parent / "data" / "pretrained"


@unittest.skipUnless(
    is_available(PRETRAINED / "model.pt"),
    "pretrained checkpoint not fetched in this checkout (Git LFS)",
)
class TestPretrainedArtifacts(unittest.TestCase):
    """Tutorial 5 loads these; if they go missing it fails in CI, not silently."""

    def test_checkpoint_and_tokenizer_are_present(self):
        self.assertTrue((PRETRAINED / "model.pt").exists())
        self.assertTrue((PRETRAINED / "spm.model").exists())
        self.assertTrue((PRETRAINED / "test.tsv").exists())

    def test_checkpoint_stays_small_enough_to_commit(self):
        """A checkpoint in git is forever. Keep it modest or do not ship it."""
        megabytes = (PRETRAINED / "model.pt").stat().st_size / 1e6
        self.assertLess(megabytes, 20)

    @unittest.skipUnless(
        is_available(CORPUS), "example corpus not fetched in this checkout (Git LFS)"
    )
    def test_held_out_set_came_from_whole_talks(self):
        """The property the whole tutorial rests on.

        The held-out set must be a handful of complete talks, not sentences
        scattered across the corpus. If it ever becomes the latter, the
        translations the tutorial shows are of effectively memorized text and
        the lesson inverts without anything looking wrong.

        Asserted on the ``talk`` column rather than by matching sentence text,
        because short formulaic lines like "Thank you." appear in hundreds of
        talks and make text matching meaningless.
        """
        held_out = pd.read_csv(
            PRETRAINED / "test.tsv", sep="\t", dtype=str, keep_default_na=False
        )
        self.assertIn("talk", held_out.columns)
        talks = held_out["talk"].nunique()
        self.assertLess(talks, 40, "held-out set is spread over too many talks")
        self.assertGreater(talks, 5, "too few talks to be a meaningful test set")

        # And those talks must be complete: every sentence of a held-out talk is
        # held out, which is what "split by talk" means.
        corpus = pd.read_csv(CORPUS, sep="\t", dtype=str, keep_default_na=False)
        in_test_talks = corpus[corpus["talk"].isin(set(held_out["talk"]))]
        self.assertGreater(len(in_test_talks), len(held_out) * 0.5)


if __name__ == "__main__":
    unittest.main()
