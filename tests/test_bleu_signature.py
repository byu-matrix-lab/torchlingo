"""Every score carries the settings that produced it.

sacreBLEU keeps the score and the signature on separate objects: the metric
knows its settings, the score does not. A caller holding only the result cannot
say how it was produced, which is the exact failure the signature exists to
prevent. `compute_bleu` attaches it, and these tests keep that true.

The case worth the most attention is character tokenization. TorchLingo does it
*before* sacreBLEU sees the text and then passes `tokenize="none"`, so
sacreBLEU's own signature would honestly report `tok:none` and thereby omit the
tokenization that actually happened. A signature that hides a tokenization
decision is worse than no signature, because it invites a comparison that is
not valid.
"""

import unittest

from torchlingo.evaluation import compute_bleu, compute_chrf, compute_ter

PREDICTIONS = ["The cat sat on the mat", "Hello world"]
REFERENCES = ["A cat sat on a mat", "Hello world"]


class SignaturePresenceTests(unittest.TestCase):
    def test_result_carries_a_signature(self):
        result = compute_bleu(PREDICTIONS, REFERENCES)
        self.assertTrue(hasattr(result, "signature"))
        self.assertIsInstance(result.signature, str)
        self.assertTrue(result.signature)

    def test_signature_names_the_fields_that_make_bleu_comparable(self):
        signature = compute_bleu(PREDICTIONS, REFERENCES).signature
        for field in ("nrefs", "case", "tok", "smooth", "version"):
            self.assertIn(f"{field}:", signature, f"{field} missing from {signature}")

    def test_score_still_works(self):
        # The signature is added alongside, not in place of, the existing API.
        result = compute_bleu(PREDICTIONS, REFERENCES)
        self.assertGreater(result.score, 0.0)
        self.assertLessEqual(result.score, 100.0)


class SignatureReflectsSettingsTests(unittest.TestCase):
    def test_tokenizer_choice_appears(self):
        thirteen_a = compute_bleu(PREDICTIONS, REFERENCES, tokenize="13a").signature
        intl = compute_bleu(PREDICTIONS, REFERENCES, tokenize="intl").signature
        self.assertIn("tok:13a", thirteen_a)
        self.assertIn("tok:intl", intl)
        self.assertNotEqual(thirteen_a, intl)

    def test_lowercasing_appears(self):
        cased = compute_bleu(PREDICTIONS, REFERENCES, lowercase=False).signature
        lowered = compute_bleu(PREDICTIONS, REFERENCES, lowercase=True).signature
        self.assertIn("case:mixed", cased)
        self.assertIn("case:lc", lowered)

    def test_char_tokenization_is_recorded_not_hidden(self):
        # The load-bearing one. Without the explicit field this would read
        # `tok:none`, which is true of what sacreBLEU did and false about what
        # was scored.
        result = compute_bleu(["你好世界"], ["你好世界"], tokenization="char")
        self.assertIn("tokenization:char", result.signature)

    def test_word_tokenization_does_not_claim_char(self):
        result = compute_bleu(PREDICTIONS, REFERENCES, tokenization="word")
        self.assertNotIn("tokenization:char", result.signature)


class SignatureDistinguishesIncomparableScoresTests(unittest.TestCase):
    def test_two_settings_that_change_the_score_also_change_the_signature(self):
        """The property that makes the signature worth reporting.

        If two settings can produce different scores from identical text, they
        must be distinguishable from the signature alone -- otherwise someone
        comparing the two numbers has no way to know they should not.
        """
        # Long enough to have 4-grams; BLEU is 0 on very short sentences
        # whatever the settings, which would make this pass vacuously.
        prediction = ["The Cat Sat On The Mat Today And Slept"]
        reference = ["the cat sat on the mat today and slept"]

        cased = compute_bleu(prediction, reference, lowercase=False)
        lowered = compute_bleu(prediction, reference, lowercase=True)

        self.assertEqual(cased.score, 0.0, "case-sensitive should find no match")
        self.assertAlmostEqual(lowered.score, 100.0, places=4)
        self.assertNotEqual(cased.signature, lowered.signature)


class EveryMetricCarriesASignatureTests(unittest.TestCase):
    """Not only BLEU. Task #85.

    `compute_bleu` had a signature and `compute_chrf` and `compute_ter` did not, because
    those two used sacreBLEU's functional API — which returns a score with no route back to
    the metric that produced it, so `get_signature()` was unreachable. All three use the
    class API now.
    """

    def test_all_three_metrics_attach_one(self):
        for name, fn in (
            ("BLEU", compute_bleu),
            ("chrF", compute_chrf),
            ("TER", compute_ter),
        ):
            result = fn(PREDICTIONS, REFERENCES)
            self.assertTrue(
                getattr(result, "signature", None), f"{name} has no signature"
            )
            self.assertIn("version:", result.signature, name)

    def test_chrf_declares_word_order(self):
        """The case that proves signatures are not cosmetic.

        `compute_chrf` defaults to `word_order=2` (chrF++) where sacreBLEU's own
        `corpus_chrf` defaults to `0`. That one undeclared parameter caused two separate
        confusions in this repository: a correct implementation judged broken by 0.45
        points against the wrong baseline, and a bug note stating the right value beside a
        snippet returning the wrong one.
        """
        self.assertIn("nw:2", compute_chrf(PREDICTIONS, REFERENCES).signature)
        self.assertIn(
            "nw:0", compute_chrf(PREDICTIONS, REFERENCES, word_order=0).signature
        )

    def test_word_order_changes_both_the_score_and_the_signature(self):
        """Neither alone is enough: a setting that moves the number must move the label."""
        plus = compute_chrf(PREDICTIONS, REFERENCES)
        chars_only = compute_chrf(PREDICTIONS, REFERENCES, word_order=0)
        self.assertNotAlmostEqual(plus.score, chars_only.score, places=4)
        self.assertNotEqual(plus.signature, chars_only.signature)

    def test_ter_records_that_it_lowercases_and_bleu_does_not(self):
        """Quoted side by side, these two were computed on differently-cased text.

        Nothing in either number says so, and a reader comparing them has no way to find
        out except the signature.
        """
        self.assertIn("case:lc", compute_ter(PREDICTIONS, REFERENCES).signature)
        self.assertIn("case:mixed", compute_bleu(PREDICTIONS, REFERENCES).signature)

    def test_ter_normalization_appears(self):
        plain = compute_ter(PREDICTIONS, REFERENCES)
        normalized = compute_ter(PREDICTIONS, REFERENCES, normalized=True)
        self.assertIn("norm:no", plain.signature)
        self.assertIn("norm:yes", normalized.signature)

    def test_switching_to_the_class_api_did_not_move_any_score(self):
        """The refactor had to be numerically invisible, so it is checked against
        sacreBLEU's functional API directly rather than against a remembered number."""
        import sacrebleu

        streams = [REFERENCES]
        self.assertAlmostEqual(
            compute_chrf(PREDICTIONS, REFERENCES).score,
            sacrebleu.corpus_chrf(PREDICTIONS, streams, word_order=2).score,
            places=10,
        )
        self.assertAlmostEqual(
            compute_ter(PREDICTIONS, REFERENCES).score,
            sacrebleu.corpus_ter(PREDICTIONS, streams).score,
            places=10,
        )


if __name__ == "__main__":
    unittest.main()
