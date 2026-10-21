"""Sampled and noised-beam decoding, for back-translation (Task #267).

Edunov et al. (2018) found that back-translations made by sampling, or by beam search
with noise added to its output, train a better forward model than greedy or beam ones in
all but resource-poor settings. Assignment 13 lets a student choose, so both must be
reproducible from a seed, must not depend on how a 100,000-sentence job is batched or
chunked, and must resume like every other bulk decode.
"""

import json
import tempfile
import unittest
from pathlib import Path

import torch

from torchlingo import inference_fast
from torchlingo.config import Config
from torchlingo.data_processing.vocab import SimpleVocab
from torchlingo.inference import (
    DECODE_STRATEGIES,
    NOISE_FILLER,
    add_noise,
    greedy_decode,
    sample_decode,
    translate_batch,
    translate_to_file,
)
from torchlingo.models import SimpleSeq2SeqLSTM, SimpleTransformer

WORDS = [f"w{i}" for i in range(40)]


def _transformer(vocab_size=50):
    torch.manual_seed(0)
    return SimpleTransformer(
        vocab_size,
        vocab_size,
        d_model=16,
        n_heads=2,
        num_encoder_layers=1,
        num_decoder_layers=1,
        d_ff=32,
    ).eval()


def _lstm(vocab_size=50):
    torch.manual_seed(0)
    return SimpleSeq2SeqLSTM(
        vocab_size, vocab_size, emb_dim=16, hidden_dim=16, num_layers=1
    ).eval()


def _src(rows=12):
    torch.manual_seed(1)
    return torch.randint(4, 50, (rows, 6))


class TestSampleDecode(unittest.TestCase):
    """An untrained model spreads its probability widely, so samples vary a lot."""

    def setUp(self):
        self.src = _src()

    def test_a_fixed_seed_is_deterministic(self):
        for model in (_transformer(), _lstm()):
            first = sample_decode(model, self.src, seed=5, max_len=8)
            self.assertEqual(first, sample_decode(model, self.src, seed=5, max_len=8))

    def test_different_seeds_differ(self):
        model = _transformer()
        self.assertNotEqual(
            sample_decode(model, self.src, seed=5, max_len=8),
            sample_decode(model, self.src, seed=6, max_len=8),
        )

    def test_temperature_near_zero_is_greedy(self):
        # How near is near depends on the closest call: the untrained LSTM has two
        # logits within about 1e-3 somewhere here, so at 1e-4 the runner-up keeps a
        # real share and is sometimes drawn. That is temperature working, not a bug.
        for model in (_transformer(), _lstm()):
            self.assertEqual(
                sample_decode(model, self.src, temperature=1e-6, max_len=8),
                greedy_decode(model, self.src, max_len=8),
            )

    def test_top_k_of_one_is_greedy(self):
        model = _transformer()
        self.assertEqual(
            sample_decode(model, self.src, top_k=1, max_len=8, seed=3),
            greedy_decode(model, self.src, max_len=8),
        )

    def test_top_k_only_draws_among_the_k_most_likely(self):
        """Every step's token is among that step's top k, given the prefix actually drawn."""
        model = _transformer()
        src = self.src[:1]
        tokens = sample_decode(model, src, top_k=3, max_len=8, seed=11)[0]
        with torch.no_grad():
            logits = model(src, torch.tensor([tokens[:-1]]))[0]
        for step, token in enumerate(tokens[1:]):
            self.assertIn(token, logits[step].topk(3).indices.tolist(), step)

    def test_a_sentence_does_not_depend_on_its_batch(self):
        """A sentence's sample is the same alone, in a small batch or in a large one."""
        model = _transformer()
        whole = sample_decode(
            model, self.src, seed=9, max_len=8, config=Config(batch_size=64)
        )
        chunked = sample_decode(
            model, self.src, seed=9, max_len=8, config=Config(batch_size=5)
        )
        self.assertEqual(whole, chunked)

    def test_sampling_varies_and_follows_the_distribution(self):
        """Many draws of one first token land in proportion to its probability."""
        model = _transformer()
        src = self.src[:1].repeat(2000, 1)
        firsts = [s[1] for s in sample_decode(model, src, max_len=1, seed=0)]
        with torch.no_grad():
            logits = model(self.src[:1], torch.tensor([[Config().sos_idx]]))[0, 0]
        probs = logits.softmax(-1)
        top = int(probs.argmax())
        self.assertGreater(len(set(firsts)), 5)
        self.assertAlmostEqual(
            firsts.count(top) / len(firsts), float(probs[top]), delta=0.03
        )

    def test_bad_arguments_are_refused(self):
        model = _transformer()
        with self.assertRaisesRegex(ValueError, "temperature"):
            sample_decode(model, self.src, temperature=0)
        with self.assertRaisesRegex(ValueError, "top_k"):
            sample_decode(model, self.src, top_k=0)
        with self.assertRaisesRegex(ValueError, "seeds"):
            sample_decode(model, self.src, seed=[1, 2])


class TestAddNoise(unittest.TestCase):
    LONG = " ".join(f"w{i}" for i in range(20000))

    def test_rates_match_the_paper_on_a_long_input(self):
        words = self.LONG.split()
        noisy = add_noise(self.LONG, seed=0).split()
        deleted = 1 - len(noisy) / len(words)
        replaced = noisy.count(NOISE_FILLER) / len(noisy)
        self.assertAlmostEqual(deleted, 0.1, delta=0.01)
        self.assertAlmostEqual(replaced, 0.1, delta=0.01)

    def test_no_word_moves_more_than_three_places(self):
        words = self.LONG.split()
        noisy = add_noise(self.LONG, seed=0, delete_prob=0, replace_prob=0).split()
        self.assertEqual(sorted(noisy), sorted(words))
        moves = [abs(pos - int(word[1:])) for pos, word in enumerate(noisy)]
        self.assertEqual(max(moves), 3)

    def test_seeded_and_never_empty(self):
        self.assertEqual(add_noise("a b c d", seed=4), add_noise("a b c d", seed=4))
        self.assertEqual(
            add_noise("solo", seed=0, delete_prob=1.0, replace_prob=0.0), "solo"
        )
        self.assertEqual(add_noise("", seed=0), "")


class _Corpus:
    """Eleven five-word sentences over a forty-word vocabulary, and an untrained model."""

    def setUp(self):
        self.vocab = SimpleVocab(min_freq=1)
        self.vocab.build_vocab([" ".join(WORDS)])
        self.model = _transformer(len(self.vocab))
        self.sentences = [" ".join(WORDS[i : i + 5]) for i in range(0, 33, 3)]

    def batch(self, translate=translate_batch, **kwargs):
        return translate(
            self.model, self.sentences, self.vocab, self.vocab, max_len=8, **kwargs
        )


class TestTranslateWithRandomStrategies(_Corpus, unittest.TestCase):
    def test_unknown_strategy_is_refused(self):
        """It used to decode greedily under whatever name was asked for."""
        for translate in (translate_batch, inference_fast.translate_batch):
            with self.assertRaisesRegex(ValueError, "Unknown decode_strategy"):
                self.batch(translate, decode_strategy="Beam")

    def test_fast_path_matches_the_reference(self):
        for strategy in DECODE_STRATEGIES:
            self.assertEqual(
                self.batch(
                    inference_fast.translate_batch,
                    decode_strategy=strategy,
                    beam_size=3,
                    seed=2,
                ),
                self.batch(decode_strategy=strategy, beam_size=3, seed=2),
                strategy,
            )

    def test_beam_plus_noise_is_beam_then_noise(self):
        beam = self.batch(decode_strategy="beam", beam_size=3)
        noisy = self.batch(decode_strategy="beam+noise", beam_size=3, seed=2)
        self.assertNotEqual(noisy, beam)
        for clean, dirty in zip(beam, noisy):
            kept = [w for w in dirty.split() if w != NOISE_FILLER]
            self.assertTrue(set(kept) <= set(clean.split()), (clean, dirty))

    def test_sampling_ignores_batch_size(self):
        self.assertEqual(
            self.batch(decode_strategy="sample", config=Config(batch_size=2)),
            self.batch(decode_strategy="sample", config=Config(batch_size=64)),
        )


class TestTranslateToFileWithRandomStrategies(_Corpus, unittest.TestCase):
    def setUp(self):
        super().setUp()
        self.out = Path(tempfile.mkdtemp()) / "mono.bt"

    def to_file(self, **kwargs):
        kwargs.setdefault("chunk_size", 4)
        return translate_to_file(
            self.model,
            self.sentences,
            self.vocab,
            self.vocab,
            self.out,
            max_len=8,
            verbose=False,
            **kwargs,
        )

    def record(self):
        return json.loads(Path(str(self.out) + ".progress.json").read_text())

    def test_same_output_as_translate_batch_whatever_the_chunk_size(self):
        for strategy in ("sample", "beam+noise"):
            expected = self.batch(decode_strategy=strategy, beam_size=3, seed=4)
            for chunk_size in (1, 4, 100):
                got = self.to_file(
                    decode_strategy=strategy,
                    beam_size=3,
                    seed=4,
                    chunk_size=chunk_size,
                    overwrite=True,
                )
                self.assertEqual(got, expected, (strategy, chunk_size))

    def test_the_record_holds_the_strategy_and_seed(self):
        self.to_file(decode_strategy="sample", seed=4, temperature=0.7, top_k=10)
        record = self.record()
        self.assertEqual(record["decode_strategy"], "sample")
        self.assertEqual(
            (record["seed"], record["temperature"], record["top_k"]), (4, 0.7, 10)
        )
        self.to_file(decode_strategy="beam+noise", beam_size=3, seed=4, overwrite=True)
        record = self.record()
        self.assertEqual((record["seed"], record["beam_size"]), (4, 3))

    def test_greedy_and_beam_records_are_unchanged(self):
        """A decode interrupted before sampling existed must still resume."""
        self.to_file()
        self.assertEqual(
            set(self.record()),
            {"inputs_sha256", "inputs", "decode_strategy", "beam_size", "max_len"},
        )

    def test_another_seed_is_a_different_job(self):
        self.to_file(decode_strategy="sample", seed=4)
        with self.assertRaisesRegex(ValueError, "different job"):
            self.to_file(decode_strategy="sample", seed=5)

    def test_an_interrupted_sampled_run_resumes_to_the_same_file(self):
        expected = self.to_file(decode_strategy="sample", seed=4)
        lines = self.out.read_text(encoding="utf-8").splitlines(keepends=True)
        self.out.write_text("".join(lines[:5]), encoding="utf-8")
        self.assertEqual(self.to_file(decode_strategy="sample", seed=4), expected)

    def test_unknown_strategy_is_refused_before_any_file_is_written(self):
        with self.assertRaisesRegex(ValueError, "Unknown decode_strategy"):
            self.to_file(decode_strategy="sampling")
        self.assertFalse(self.out.exists())


if __name__ == "__main__":
    unittest.main()
