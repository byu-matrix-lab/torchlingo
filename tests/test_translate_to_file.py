"""translate_to_file: a bulk decode that survives an interruption (Task #102).

Assignment 13 back-translates 100,000 sentences or more. A decode that dies partway must
resume where it stopped, must never redo finished work, and must never append to a file that
belongs to a different job: a misaligned hypothesis file scores badly and says nothing about
why.
"""

import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import torch

from torchlingo import inference_fast
from torchlingo.data_processing.vocab import SimpleVocab
from torchlingo.inference import translate_batch, translate_to_file
from torchlingo.models import SimpleTransformer


def _setup():
    vocab = SimpleVocab(min_freq=1)
    vocab.build_vocab(
        ["el gato duerme", "the cat sleeps", "a dog runs", "un perro corre"]
    )
    torch.manual_seed(0)
    model = SimpleTransformer(
        len(vocab),
        len(vocab),
        d_model=16,
        n_heads=2,
        num_encoder_layers=1,
        num_decoder_layers=1,
        d_ff=32,
    )
    model.eval()
    sentences = [f"the cat sleeps {'a dog runs ' * (i % 3)}".strip() for i in range(11)]
    return model, vocab, sentences


class _Interrupt(Exception):
    """Stands in for a Colab disconnect partway through a decode."""


class TestTranslateToFile(unittest.TestCase):
    def setUp(self):
        self.model, self.vocab, self.sentences = _setup()
        self.dir = Path(tempfile.mkdtemp())
        self.out = self.dir / "test.hyp"
        self.real = inference_fast.translate_batch
        self.calls = 0

    def translate(self, **kwargs):
        return translate_to_file(
            self.model,
            self.sentences,
            self.vocab,
            self.vocab,
            self.out,
            max_len=6,
            chunk_size=3,
            verbose=False,
            **kwargs,
        )

    def counting(self, fail_on_call=None):
        """A stand-in for the fast decoder that counts calls and can fail on one of them."""

        def wrapped(*args, **kwargs):
            self.calls += 1
            if self.calls == fail_on_call:
                raise _Interrupt
            return self.real(*args, **kwargs)

        return mock.patch.object(inference_fast, "translate_batch", wrapped)

    def test_same_output_as_translate_batch(self):
        """Same translations as the reference decoder, one line each, in order."""
        for strategy in ("greedy", "beam"):
            expected = translate_batch(
                self.model,
                self.sentences,
                self.vocab,
                self.vocab,
                decode_strategy=strategy,
                beam_size=3,
                max_len=6,
            )
            got = self.translate(decode_strategy=strategy, beam_size=3, overwrite=True)
            self.assertEqual(got, expected, strategy)
            self.assertEqual(
                self.out.read_text(encoding="utf-8"),
                "".join(t + "\n" for t in expected),
            )

    def test_an_interrupted_run_resumes_without_redoing_work(self):
        """Killed during the third chunk; the rerun decodes only the chunks not yet written."""
        expected = translate_batch(
            self.model, self.sentences, self.vocab, self.vocab, max_len=6
        )

        with self.counting(fail_on_call=3), self.assertRaises(_Interrupt):
            self.translate()
        self.assertEqual(len(self.out.read_text(encoding="utf-8").splitlines()), 6)

        self.calls = 0
        with self.counting():
            got = self.translate()
        # 11 sentences in chunks of 3: 4 chunks in all, 2 already done before the failure.
        self.assertEqual(self.calls, 2)
        self.assertEqual(got, expected)

    def test_a_finished_job_decodes_nothing_on_a_rerun(self):
        self.translate()
        with self.counting():
            self.translate()
        self.assertEqual(self.calls, 0)

    def test_a_partly_written_last_line_is_redone(self):
        """A crash mid-write leaves a fragment with no newline; it is cut off and decoded again."""
        expected = self.translate()
        complete = "".join(t + "\n" for t in expected[:4])
        self.out.write_text(complete + "half a transla", encoding="utf-8")
        self.assertEqual(self.translate(), expected)

    def test_a_different_job_is_refused(self):
        """Other sentences, other settings or no record: never append, which would misalign."""
        self.translate()
        original = self.sentences
        self.sentences = list(reversed(original))
        with self.assertRaisesRegex(ValueError, "different job"):
            self.translate()
        self.sentences = original
        with self.assertRaisesRegex(ValueError, "different job"):
            self.translate(decode_strategy="beam")
        Path(str(self.out) + ".progress.json").unlink()
        with self.assertRaisesRegex(ValueError, "missing"):
            self.translate()

    def test_overwrite_starts_again(self):
        self.translate()
        self.sentences = self.sentences[:5]
        self.assertEqual(len(self.translate(overwrite=True)), 5)

    def test_more_lines_than_sentences_is_refused(self):
        self.translate()
        progress = Path(str(self.out) + ".progress.json")
        record = json.loads(progress.read_text(encoding="utf-8"))
        with self.out.open("a", encoding="utf-8") as handle:
            handle.write("an extra line\n")
        progress.write_text(json.dumps(record), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "only 11 sentences"):
            self.translate()

    def test_a_newline_inside_a_translation_stays_on_one_line(self):
        def two_lines(model, chunk, *args, **kwargs):
            return ["line one\nline two"] * len(chunk)

        with mock.patch.object(inference_fast, "translate_batch", two_lines):
            got = self.translate()
        self.assertEqual(len(got), len(self.sentences))
        self.assertEqual(got[0], "line one line two")


if __name__ == "__main__":
    unittest.main()
