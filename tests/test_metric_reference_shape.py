"""The three corpus metrics must agree with sacreBLEU, and with each other.

sacreBLEU's ``corpus_*`` functions take references as *streams*: one list per
reference slot, each running the length of the corpus. The natural way to hold
references is the transpose of that, one list per sentence. Pass the natural
shape and sacreBLEU does not complain -- it reads N sentences as N separate
reference streams of one sentence each, scores against that, and returns a
plausible number.

``compute_bleu`` transposed correctly. ``compute_chrf`` and ``compute_ter`` did
not, so on a three-sentence corpus chrF read 54.85 where the truth was 67.91,
and TER read 50.00 where the truth was 25.00. Nothing caught it because nothing
exercised them: neither function appeared in a test, a tutorial, or a docs page.

These tests compare each wrapper against sacreBLEU called directly at matching
parameters. Getting that comparison right matters as much as the fix -- chrF
defaults to ``word_order=2`` here and ``0`` in sacreBLEU, so a careless baseline
makes a correct implementation look broken by 0.45 points.
"""

import unittest

import sacrebleu

from torchlingo.evaluation import (
    _as_reference_streams,
    compute_bleu,
    compute_chrf,
    compute_ter,
)

PREDICTIONS = ["The cat sat on the mat", "Hello world", "How are you"]
SINGLE_REFS = ["A cat sat on a mat", "Hello world", "How are you doing"]

# Two references per sentence, in the per-sentence shape a caller would write.
MULTI_REFS = [
    ["A cat sat on a mat", "The cat sat on the mat"],
    ["Hello world", "Hi world"],
    ["How are you doing", "How are you"],
]
MULTI_STREAMS = [[r[0] for r in MULTI_REFS], [r[1] for r in MULTI_REFS]]


class ReferenceStreamShapeTests(unittest.TestCase):
    def test_one_reference_per_sentence_becomes_one_stream(self):
        self.assertEqual(_as_reference_streams(["a", "b", "c"]), [["a", "b", "c"]])

    def test_per_sentence_lists_are_transposed(self):
        self.assertEqual(
            _as_reference_streams([["a1", "a2"], ["b1", "b2"]]),
            [["a1", "b1"], ["a2", "b2"]],
        )

    def test_ragged_reference_counts_raise(self):
        # sacreBLEU cannot represent this and would silently truncate.
        with self.assertRaises(ValueError):
            _as_reference_streams([["a1", "a2"], ["b1"]])

    def test_empty_is_handled(self):
        self.assertEqual(_as_reference_streams([]), [[]])


class AgreementWithSacrebleuTests(unittest.TestCase):
    """Each wrapper against the library it wraps, at matching parameters."""

    def test_bleu_single_reference(self):
        self.assertAlmostEqual(
            compute_bleu(PREDICTIONS, SINGLE_REFS).score,
            sacrebleu.corpus_bleu(PREDICTIONS, [SINGLE_REFS]).score,
            places=9,
        )

    def test_chrf_single_reference(self):
        # word_order must match: compute_chrf defaults to 2 (chrF++),
        # sacrebleu.corpus_chrf to 0.
        self.assertAlmostEqual(
            compute_chrf(PREDICTIONS, SINGLE_REFS).score,
            sacrebleu.corpus_chrf(PREDICTIONS, [SINGLE_REFS], word_order=2).score,
            places=9,
        )

    def test_ter_single_reference(self):
        self.assertAlmostEqual(
            compute_ter(PREDICTIONS, SINGLE_REFS).score,
            sacrebleu.corpus_ter(PREDICTIONS, [SINGLE_REFS]).score,
            places=9,
        )

    def test_bleu_multiple_references(self):
        self.assertAlmostEqual(
            compute_bleu(PREDICTIONS, MULTI_REFS).score,
            sacrebleu.corpus_bleu(PREDICTIONS, MULTI_STREAMS).score,
            places=9,
        )

    def test_chrf_multiple_references(self):
        self.assertAlmostEqual(
            compute_chrf(PREDICTIONS, MULTI_REFS).score,
            sacrebleu.corpus_chrf(PREDICTIONS, MULTI_STREAMS, word_order=2).score,
            places=9,
        )

    def test_ter_multiple_references(self):
        self.assertAlmostEqual(
            compute_ter(PREDICTIONS, MULTI_REFS).score,
            sacrebleu.corpus_ter(PREDICTIONS, MULTI_STREAMS).score,
            places=9,
        )


class CorpusSizeSensitivityTests(unittest.TestCase):
    """The property the old bug violated.

    Scoring a corpus must not equal scoring its sentences as if each were its
    own corpus. If it does, the references were misread as separate streams --
    which is exactly what produced chrF 54.85 and TER 50.00.
    """

    def test_corpus_score_is_not_the_degenerate_per_sentence_reading(self):
        wrong_shape = [[ref] for ref in SINGLE_REFS]
        for name, fn, direct in (
            ("chrF", compute_chrf, sacrebleu.corpus_chrf),
            ("TER", compute_ter, sacrebleu.corpus_ter),
        ):
            with self.subTest(metric=name):
                degenerate = direct(PREDICTIONS, wrong_shape).score
                correct = fn(PREDICTIONS, SINGLE_REFS).score
                self.assertNotAlmostEqual(
                    correct,
                    degenerate,
                    places=2,
                    msg=f"{name} still reads references as one stream per sentence",
                )


class MetricDirectionTests(unittest.TestCase):
    """Worth pinning before teaching them: TER runs the other way."""

    def test_bleu_and_chrf_reward_a_perfect_translation(self):
        perfect = compute_bleu(SINGLE_REFS, SINGLE_REFS).score
        self.assertAlmostEqual(perfect, 100.0, places=4)
        self.assertAlmostEqual(
            compute_chrf(SINGLE_REFS, SINGLE_REFS).score, 100.0, places=4
        )

    def test_ter_scores_a_perfect_translation_as_zero(self):
        self.assertAlmostEqual(
            compute_ter(SINGLE_REFS, SINGLE_REFS).score, 0.0, places=4
        )


if __name__ == "__main__":
    unittest.main()
