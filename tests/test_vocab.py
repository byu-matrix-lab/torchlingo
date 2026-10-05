import unittest
from pathlib import Path

import torch

from torchlingo import config
from torchlingo.config import Config
from torchlingo.data_processing.vocab import BaseVocab, SimpleVocab


class BaseVocabContractTests(unittest.TestCase):
    """Contract checks for the abstract base vocabulary and alias."""

    def test_base_is_abstract(self):
        """BaseVocab cannot be instantiated directly."""
        with self.assertRaises(TypeError):
            BaseVocab()

    def test_simple_vocab_is_subclass(self):
        """SimpleVocab must subclass BaseVocab for polymorphic use."""
        self.assertTrue(issubclass(SimpleVocab, BaseVocab))


class SimpleVocabInitTests(unittest.TestCase):
    """Initialization behavior for SimpleVocab defaults and overrides."""

    def test_uses_config_defaults(self):
        """Defaults come from global config when no args are provided."""
        vocab = SimpleVocab()
        self.assertEqual(vocab.pad_token, config.PAD_TOKEN)
        self.assertEqual(vocab.unk_token, config.UNK_TOKEN)
        self.assertEqual(vocab.sos_token, config.SOS_TOKEN)
        self.assertEqual(vocab.eos_token, config.EOS_TOKEN)
        self.assertEqual(vocab.pad_idx, config.PAD_IDX)
        self.assertEqual(vocab.unk_idx, config.UNK_IDX)
        self.assertEqual(vocab.sos_idx, config.SOS_IDX)
        self.assertEqual(vocab.eos_idx, config.EOS_IDX)
        self.assertEqual(vocab.min_freq, config.MIN_FREQ)
        self.assertEqual(len(vocab), 4)

    def test_explicit_overrides(self):
        """Explicit kwargs override both defaults and config."""
        vocab = SimpleVocab(
            min_freq=5,
            pad_token="[P]",
            unk_token="[U]",
            sos_token="<S>",
            eos_token="</S>",
            pad_idx=10,
            unk_idx=11,
            sos_idx=12,
            eos_idx=13,
        )
        self.assertEqual(vocab.min_freq, 5)
        self.assertEqual(vocab.pad_token, "[P]")
        self.assertEqual(vocab.unk_idx, 11)

    def test_config_object_overrides_defaults(self):
        """Provided Config object can override default tokens/indices."""
        cfg = Config(min_freq=7, pad_token="<PADDING>", pad_idx=9)
        vocab = SimpleVocab(config=cfg)
        self.assertEqual(vocab.min_freq, 7)
        self.assertEqual(vocab.pad_token, "<PADDING>")
        self.assertEqual(vocab.pad_idx, 9)


class SimpleVocabBuildTests(unittest.TestCase):
    """Vocabulary construction edge cases and frequency handling."""

    def test_counts_tokens_and_respects_min_freq(self):
        """Tokens below min_freq are excluded; counts are tracked."""
        sentences = [
            "hello world from the torch playground today",
            "hello pytorch from the bright sunny lab",
            "hello there from the quiet reading room",
        ]
        vocab = SimpleVocab(min_freq=2)
        vocab.build_vocab(sentences)

        self.assertIn("hello", vocab.token2idx)
        self.assertNotIn("world", vocab.token2idx)
        self.assertEqual(vocab.token_freqs["hello"], 3)

    def test_len_includes_special_tokens(self):
        """Length grows after build but starts with special tokens."""
        vocab = SimpleVocab(min_freq=1)
        initial_len = len(vocab)
        vocab.build_vocab(
            [
                "alpha beta gamma delta epsilon",
                "alpha beta gamma delta zeta",
            ]
        )
        self.assertGreater(len(vocab), initial_len)

    def test_build_is_noop_on_empty(self):
        """Empty input leaves only the four special tokens present."""
        vocab = SimpleVocab(min_freq=1)
        vocab.build_vocab([])
        self.assertEqual(len(vocab), 4)


class SimpleVocabMostlyUnknownWarningTests(unittest.TestCase):
    """A vocabulary that turns most of its corpus into <unk> says so.

    Two A8 students on Asian languages got empty translations: their text has no
    spaces, so each sentence was one "word", nearly none repeated, and the model
    learned to write <unk>, which decoding then dropped. It is shown now (see
    SimpleVocabDecodeKeepsUnknownTests), and this warning comes before training.
    """

    UNSPACED = tuple(f"我今天看了第{i}本书。" for i in range(300))
    SPACED_RARE = tuple(f"w{i} x{i} y{i} z{i}" for i in range(300))

    @staticmethod
    def _ordinary(n: int = 300) -> list[str]:
        """Space-separated sentences over a small vocabulary, so words repeat."""
        import random

        rng = random.Random(0)
        words = [f"word{k}" for k in range(50)]
        return [" ".join(rng.choices(words, k=8)) for _ in range(n)]

    def test_unspaced_text_warns_and_names_the_cause(self):
        from torchlingo.data_processing.vocab import MostlyUnknownWarning

        with self.assertWarns(MostlyUnknownWarning) as caught:
            SimpleVocab(min_freq=2).build_vocab(self.UNSPACED)
        message = str(caught.warning)
        self.assertIn("without spaces", message)
        self.assertIn("translations will be mostly <unk>", message)
        self.assertIn("use_sentencepiece=True", message)

    def test_the_warning_matches_what_encoding_does(self):
        """The failure it warns about: a training sentence encodes to <unk> alone."""
        vocab = SimpleVocab(min_freq=2)
        with self.assertWarns(UserWarning):
            vocab.build_vocab(self.UNSPACED)
        ids = vocab.encode(self.UNSPACED[0])
        self.assertEqual(ids, [vocab.sos_idx, vocab.unk_idx, vocab.eos_idx])
        # Shown, not dropped: an empty line hid this from two students.
        self.assertEqual(vocab.decode(ids), "<unk>")

    def test_spaced_but_rare_warns_without_blaming_spaces(self):
        from torchlingo.data_processing.vocab import MostlyUnknownWarning

        with self.assertWarns(MostlyUnknownWarning) as caught:
            SimpleVocab(min_freq=2).build_vocab(self.SPACED_RARE)
        self.assertNotIn("without spaces", str(caught.warning))

    def test_ordinary_text_is_quiet(self):
        import warnings

        with warnings.catch_warnings():
            warnings.simplefilter("error")
            SimpleVocab(min_freq=2).build_vocab(self._ordinary())

    def test_small_corpus_is_quiet(self):
        """Toy corpora are mostly rare words by nature; tutorials must not warn."""
        import warnings

        with warnings.catch_warnings():
            warnings.simplefilter("error")
            SimpleVocab(min_freq=2).build_vocab(self.UNSPACED[:199])

    def test_min_freq_one_is_quiet(self):
        """Nothing falls below min_freq=1, so nothing becomes <unk>."""
        import warnings

        with warnings.catch_warnings():
            warnings.simplefilter("error")
            SimpleVocab(min_freq=1).build_vocab(self.UNSPACED)


class SimpleVocabConversionTests(unittest.TestCase):
    """Token/index conversion helpers including unknown handling."""

    def setUp(self) -> None:
        self.vocab = SimpleVocab(min_freq=1)
        self.sentence1 = "hello world from the torch playground today"
        self.sentence2 = "good morning from the bright python universe"
        self.vocab.build_vocab([self.sentence1, self.sentence2])

    def test_token_and_index_roundtrip(self):
        """Tokens map to indices and back without loss."""
        idx = self.vocab.token_to_idx("hello")
        self.assertEqual(self.vocab.idx_to_token(idx), "hello")

    def test_unknown_token_and_index(self):
        """Unknown tokens/indices fall back to UNK defaults."""
        self.assertEqual(self.vocab.token_to_idx("<missing>"), config.UNK_IDX)
        self.assertEqual(self.vocab.idx_to_token(999), config.UNK_TOKEN)

    def test_tokens_to_indices_and_back(self):
        """Batch conversions preserve known tokens and replace unknowns."""
        tokens = ["hello", "world", "<missing>"]
        ids = self.vocab.tokens_to_indices(tokens)
        self.assertEqual(ids[2], config.UNK_IDX)
        roundtrip = self.vocab.indices_to_tokens(ids)
        self.assertEqual(roundtrip[0], "hello")
        self.assertEqual(roundtrip[2], config.UNK_TOKEN)


class SimpleVocabDecodeKeepsUnknownTests(unittest.TestCase):
    """Decoding drops the framing tokens and shows <unk>.

    Dropping <unk> too made a model that writes nothing but <unk> print empty
    lines, which is how two A8 students' failure went unexplained.
    """

    def setUp(self) -> None:
        self.vocab = SimpleVocab(min_freq=1)
        self.vocab.build_vocab(["el gato"])

    def test_unknown_is_shown(self):
        v = self.vocab
        ids = [v.sos_idx, v.token_to_idx("el"), v.unk_idx, v.eos_idx, v.pad_idx]
        self.assertEqual(v.decode(ids), "el <unk>")

    def test_all_unknown_is_not_empty(self):
        v = self.vocab
        self.assertEqual(
            v.decode([v.sos_idx, v.unk_idx, v.unk_idx, v.eos_idx]), "<unk> <unk>"
        )

    def test_without_skipping_everything_is_shown(self):
        v = self.vocab
        self.assertEqual(
            v.decode([v.sos_idx, v.unk_idx, v.eos_idx], skip_special_tokens=False),
            "<sos> <unk> <eos>",
        )


class CharVocabTests(unittest.TestCase):
    """One token per character, for languages written without spaces between words."""

    def setUp(self) -> None:
        from torchlingo.data_processing.vocab import CharVocab

        self.vocab = CharVocab(min_freq=1)
        self.vocab.build_vocab(["我今天 想要船。", "ฉันไป โรงเรียน"])

    def test_round_trip_keeps_real_spaces(self):
        for sentence in ("我今天 想要船。", "ฉันไป โรงเรียน"):
            self.assertEqual(self.vocab.decode(self.vocab.encode(sentence)), sentence)

    def test_one_token_per_character(self):
        ids = self.vocab.encode("我今天", add_special_tokens=False)
        self.assertEqual(len(ids), 3)
        self.assertNotIn(self.vocab.unk_idx, ids)

    def test_an_unseen_character_is_unknown_and_shown(self):
        # 要 was seen (in 想要船); 鱼 was not.
        self.assertEqual(self.vocab.decode(self.vocab.encode("我要鱼")), "我要<unk>")

    def test_batches_decode_to_a_list(self):
        batch = [self.vocab.encode("我今天"), self.vocab.encode("想要船")]
        self.assertEqual(self.vocab.decode(batch), ["我今天", "想要船"])

    def test_min_freq_applies_to_characters(self):
        from torchlingo.data_processing.vocab import CharVocab

        vocab = CharVocab(min_freq=2)
        vocab.build_vocab(["我我", "你"])
        self.assertNotEqual(vocab.token_to_idx("我"), vocab.unk_idx)
        self.assertEqual(vocab.token_to_idx("你"), vocab.unk_idx)

    def test_unspaced_text_does_not_trigger_the_unknown_warning(self):
        """The fix for MostlyUnknownWarning's usual cause must not raise it."""
        import warnings

        from torchlingo.data_processing.vocab import CharVocab

        with warnings.catch_warnings():
            warnings.simplefilter("error")
            CharVocab(min_freq=2).build_vocab(
                SimpleVocabMostlyUnknownWarningTests.UNSPACED
            )


class SentencePieceVocabDecodeKeepsUnknownTests(unittest.TestCase):
    """The same for SentencePiece, which would otherwise render <unk> as "⁇"."""

    SPM = Path(__file__).resolve().parents[1] / "data" / "pretrained" / "spm.model"

    def setUp(self) -> None:
        if not self.SPM.exists():
            self.skipTest("data/pretrained/spm.model is not present")
        from torchlingo.data_processing.vocab import SentencePieceVocab

        self.vocab = SentencePieceVocab(str(self.SPM))

    def test_unknown_is_shown_as_the_unknown_token(self):
        v = self.vocab
        pieces = v.encode("This is public.", add_special_tokens=False)
        ids = [v.sos_idx, *pieces[:1], v.unk_idx, *pieces[1:], v.eos_idx]
        decoded = v.decode(ids)
        self.assertIn("<unk>", decoded)
        self.assertNotIn("⁇", decoded)
        self.assertEqual(
            decoded.replace("<unk>", "").split(), ["This", "is", "public."]
        )

    def test_all_unknown_is_not_empty(self):
        v = self.vocab
        self.assertEqual(v.decode([v.sos_idx, v.unk_idx, v.eos_idx]), "<unk>")


class SimpleVocabEncodeDecodeTests(unittest.TestCase):
    """End-to-end encode/decode flows for sentences and batches."""

    def setUp(self) -> None:
        self.vocab = SimpleVocab(min_freq=1)
        self.sentence1 = "hello world from the torch playground today"
        self.sentence2 = "good morning from the bright python universe"
        self.vocab.build_vocab([self.sentence1, self.sentence2])

    def test_encode_appends_special_tokens(self):
        """encode adds SOS/EOS when requested."""
        encoded = self.vocab.encode(self.sentence1)
        self.assertEqual(encoded[0], config.SOS_IDX)
        self.assertEqual(encoded[-1], config.EOS_IDX)
        self.assertEqual(len(encoded), len(self.sentence1.split()) + 2)

    def test_encode_without_special_tokens(self):
        """encode can omit special tokens when flagged."""
        encoded = self.vocab.encode(self.sentence1, add_special_tokens=False)
        self.assertEqual(len(encoded), len(self.sentence1.split()))
        self.assertNotIn(config.SOS_IDX, encoded)

    def test_decode_skips_special_tokens(self):
        """decode drops special tokens by default."""
        encoded = [config.SOS_IDX, self.vocab.token_to_idx("hello"), config.EOS_IDX]
        decoded = self.vocab.decode(encoded)
        self.assertEqual(decoded, "hello")

    def test_decode_keeps_special_tokens_when_requested(self):
        """decode can retain special markers when desired."""
        encoded = [config.SOS_IDX, self.vocab.token_to_idx("hello"), config.EOS_IDX]
        decoded = self.vocab.decode(encoded, skip_special_tokens=False)
        self.assertIn(config.SOS_TOKEN, decoded)
        self.assertIn(config.EOS_TOKEN, decoded)

    def test_decode_batch_from_list(self):
        """List-of-lists decode returns list of strings."""
        seq1 = self.vocab.encode(self.sentence1, add_special_tokens=False)
        seq2 = self.vocab.encode(self.sentence2, add_special_tokens=False)
        decoded = self.vocab.decode([seq1, seq2])
        self.assertEqual(decoded, [self.sentence1, self.sentence2])

    def test_decode_batch_from_tensor(self):
        """Tensor batch decode mirrors list batch behavior."""
        seq1 = self.vocab.encode(self.sentence1)
        seq2 = self.vocab.encode(self.sentence2)

        max_len = max(len(seq1), len(seq2))
        pad = self.vocab.pad_idx
        padded = [
            seq1 + [pad] * (max_len - len(seq1)),
            seq2 + [pad] * (max_len - len(seq2)),
        ]

        batch = torch.tensor(padded, dtype=torch.long)
        decoded = self.vocab.decode(batch)

        self.assertEqual(decoded, [self.sentence1, self.sentence2])

    def test_roundtrip_with_empty_sentence(self):
        """Empty string roundtrips to only SOS/EOS and back to blank text."""
        encoded = self.vocab.encode("", add_special_tokens=True)
        self.assertEqual(encoded, [config.SOS_IDX, config.EOS_IDX])
        decoded = self.vocab.decode(encoded)
        self.assertEqual(decoded, "")


# -----------------------------------------------------------------------------
# MeCabVocab Tests (Japanese)
# -----------------------------------------------------------------------------
try:
    import fugashi  # noqa: F401

    FUGASHI_AVAILABLE = True
except ImportError:
    FUGASHI_AVAILABLE = False


@unittest.skipUnless(FUGASHI_AVAILABLE, "fugashi not installed")
class MeCabVocabInitTests(unittest.TestCase):
    """Initialization tests for MeCabVocab."""

    def test_uses_config_defaults(self):
        """Defaults come from global config when no args are provided."""
        from torchlingo.data_processing.vocab import MeCabVocab

        vocab = MeCabVocab()
        self.assertEqual(vocab.pad_token, config.PAD_TOKEN)
        self.assertEqual(vocab.unk_token, config.UNK_TOKEN)
        self.assertEqual(vocab.sos_token, config.SOS_TOKEN)
        self.assertEqual(vocab.eos_token, config.EOS_TOKEN)
        self.assertEqual(vocab.min_freq, config.MIN_FREQ)
        self.assertEqual(len(vocab), 4)

    def test_explicit_overrides(self):
        """Explicit kwargs override defaults."""
        from torchlingo.data_processing.vocab import MeCabVocab

        vocab = MeCabVocab(min_freq=5, pad_token="[P]", pad_idx=10)
        self.assertEqual(vocab.min_freq, 5)
        self.assertEqual(vocab.pad_token, "[P]")
        self.assertEqual(vocab.pad_idx, 10)


@unittest.skipUnless(FUGASHI_AVAILABLE, "fugashi not installed")
class MeCabVocabBuildTests(unittest.TestCase):
    """Vocabulary construction tests for MeCabVocab."""

    def test_tokenizes_japanese_correctly(self):
        """MeCab correctly segments Japanese sentences."""
        from torchlingo.data_processing.vocab import MeCabVocab

        vocab = MeCabVocab(min_freq=1)
        # "私は学生です" = "I am a student"
        # "彼は先生です" = "He is a teacher"
        sentences = ["私は学生です", "彼は先生です"]
        vocab.build_vocab(sentences)

        # Should have special tokens + Japanese morphemes
        self.assertGreater(len(vocab), 4)
        # Common particle "は" should be in vocab (appears twice)
        self.assertIn("は", vocab.token2idx)

    def test_respects_min_freq(self):
        """Tokens below min_freq are excluded."""
        from torchlingo.data_processing.vocab import MeCabVocab

        vocab = MeCabVocab(min_freq=2)
        sentences = ["私は学生です", "彼は先生です", "彼女は医者です"]
        vocab.build_vocab(sentences)

        # "は" appears 3 times, should be in vocab
        self.assertIn("は", vocab.token2idx)
        # "学生" appears once, should not be in vocab
        self.assertNotIn("学生", vocab.token2idx)


@unittest.skipUnless(FUGASHI_AVAILABLE, "fugashi not installed")
class MeCabVocabEncodeDecodeTests(unittest.TestCase):
    """Encode/decode tests for MeCabVocab."""

    def setUp(self):
        from torchlingo.data_processing.vocab import MeCabVocab

        self.vocab = MeCabVocab(min_freq=1)
        self.sentences = ["私は学生です", "彼は先生です"]
        self.vocab.build_vocab(self.sentences)

    def test_encode_adds_special_tokens(self):
        """encode adds SOS/EOS when requested."""
        encoded = self.vocab.encode("私は学生です")
        self.assertEqual(encoded[0], config.SOS_IDX)
        self.assertEqual(encoded[-1], config.EOS_IDX)

    def test_encode_without_special_tokens(self):
        """encode can omit special tokens."""
        encoded = self.vocab.encode("私は学生です", add_special_tokens=False)
        self.assertNotEqual(encoded[0], config.SOS_IDX)

    def test_decode_reconstructs_japanese(self):
        """decode reconstructs Japanese text without spaces."""
        sentence = "私は学生です"
        encoded = self.vocab.encode(sentence, add_special_tokens=False)
        decoded = self.vocab.decode(encoded)
        # Should reconstruct without spaces (Japanese convention)
        self.assertEqual(decoded, sentence)

    def test_roundtrip_with_special_tokens(self):
        """Full roundtrip with special tokens."""
        sentence = "私は学生です"
        encoded = self.vocab.encode(sentence, add_special_tokens=True)
        decoded = self.vocab.decode(encoded, skip_special_tokens=True)
        self.assertEqual(decoded, sentence)

    def test_decode_batch(self):
        """Batch decode returns list of strings."""
        seq1 = self.vocab.encode(self.sentences[0], add_special_tokens=False)
        seq2 = self.vocab.encode(self.sentences[1], add_special_tokens=False)
        decoded = self.vocab.decode([seq1, seq2])
        self.assertEqual(decoded, self.sentences)


# -----------------------------------------------------------------------------
# JiebaVocab Tests (Chinese)
# -----------------------------------------------------------------------------
try:
    import jieba  # noqa: F401

    JIEBA_AVAILABLE = True
except ImportError:
    JIEBA_AVAILABLE = False


@unittest.skipUnless(JIEBA_AVAILABLE, "jieba not installed")
class JiebaVocabInitTests(unittest.TestCase):
    """Initialization tests for JiebaVocab."""

    def test_uses_config_defaults(self):
        """Defaults come from global config when no args are provided."""
        from torchlingo.data_processing.vocab import JiebaVocab

        vocab = JiebaVocab()
        self.assertEqual(vocab.pad_token, config.PAD_TOKEN)
        self.assertEqual(vocab.unk_token, config.UNK_TOKEN)
        self.assertEqual(vocab.sos_token, config.SOS_TOKEN)
        self.assertEqual(vocab.eos_token, config.EOS_TOKEN)
        self.assertEqual(vocab.min_freq, config.MIN_FREQ)
        self.assertEqual(len(vocab), 4)

    def test_explicit_overrides(self):
        """Explicit kwargs override defaults."""
        from torchlingo.data_processing.vocab import JiebaVocab

        vocab = JiebaVocab(min_freq=5, pad_token="[P]", pad_idx=10)
        self.assertEqual(vocab.min_freq, 5)
        self.assertEqual(vocab.pad_token, "[P]")
        self.assertEqual(vocab.pad_idx, 10)

    def test_cut_all_mode(self):
        """cut_all parameter is stored correctly."""
        from torchlingo.data_processing.vocab import JiebaVocab

        vocab = JiebaVocab(cut_all=True)
        self.assertTrue(vocab.cut_all)


@unittest.skipUnless(JIEBA_AVAILABLE, "jieba not installed")
class JiebaVocabBuildTests(unittest.TestCase):
    """Vocabulary construction tests for JiebaVocab."""

    def test_tokenizes_chinese_correctly(self):
        """Jieba correctly segments Chinese sentences."""
        from torchlingo.data_processing.vocab import JiebaVocab

        vocab = JiebaVocab(min_freq=1)
        # "我是学生" = "I am a student"
        # "他是老师" = "He is a teacher"
        sentences = ["我是学生", "他是老师"]
        vocab.build_vocab(sentences)

        # Should have special tokens + Chinese words
        self.assertGreater(len(vocab), 4)
        # "是" (is) should be in vocab (appears twice)
        self.assertIn("是", vocab.token2idx)

    def test_respects_min_freq(self):
        """Tokens below min_freq are excluded."""
        from torchlingo.data_processing.vocab import JiebaVocab

        vocab = JiebaVocab(min_freq=2)
        sentences = ["我是学生", "他是老师", "她是医生"]
        vocab.build_vocab(sentences)

        # "是" appears 3 times, should be in vocab
        self.assertIn("是", vocab.token2idx)
        # "学生" appears once, should not be in vocab
        self.assertNotIn("学生", vocab.token2idx)


@unittest.skipUnless(JIEBA_AVAILABLE, "jieba not installed")
class JiebaVocabEncodeDecodeTests(unittest.TestCase):
    """Encode/decode tests for JiebaVocab."""

    def setUp(self):
        from torchlingo.data_processing.vocab import JiebaVocab

        self.vocab = JiebaVocab(min_freq=1)
        self.sentences = ["我是学生", "他是老师"]
        self.vocab.build_vocab(self.sentences)

    def test_encode_adds_special_tokens(self):
        """encode adds SOS/EOS when requested."""
        encoded = self.vocab.encode("我是学生")
        self.assertEqual(encoded[0], config.SOS_IDX)
        self.assertEqual(encoded[-1], config.EOS_IDX)

    def test_encode_without_special_tokens(self):
        """encode can omit special tokens."""
        encoded = self.vocab.encode("我是学生", add_special_tokens=False)
        self.assertNotEqual(encoded[0], config.SOS_IDX)

    def test_decode_reconstructs_chinese(self):
        """decode reconstructs Chinese text without spaces."""
        sentence = "我是学生"
        encoded = self.vocab.encode(sentence, add_special_tokens=False)
        decoded = self.vocab.decode(encoded)
        # Should reconstruct without spaces (Chinese convention)
        self.assertEqual(decoded, sentence)

    def test_roundtrip_with_special_tokens(self):
        """Full roundtrip with special tokens."""
        sentence = "我是学生"
        encoded = self.vocab.encode(sentence, add_special_tokens=True)
        decoded = self.vocab.decode(encoded, skip_special_tokens=True)
        self.assertEqual(decoded, sentence)

    def test_decode_batch(self):
        """Batch decode returns list of strings."""
        seq1 = self.vocab.encode(self.sentences[0], add_special_tokens=False)
        seq2 = self.vocab.encode(self.sentences[1], add_special_tokens=False)
        decoded = self.vocab.decode([seq1, seq2])
        self.assertEqual(decoded, self.sentences)


# -----------------------------------------------------------------------------
# Import Error Tests
# -----------------------------------------------------------------------------
class VocabImportErrorTests(unittest.TestCase):
    """Tests for import error handling when optional dependencies are missing."""

    def test_mecab_vocab_import_message(self):
        """MeCabVocab provides helpful error message when fugashi missing."""
        # This test is informational - if fugashi is installed, we just verify
        # the class exists. The actual import error is tested implicitly.
        from torchlingo.data_processing.vocab import MeCabVocab

        self.assertTrue(callable(MeCabVocab))

    def test_jieba_vocab_import_message(self):
        """JiebaVocab provides helpful error message when jieba missing."""
        from torchlingo.data_processing.vocab import JiebaVocab

        self.assertTrue(callable(JiebaVocab))


if __name__ == "__main__":
    unittest.main()
