"""torchlingo_harness.nllb: NLLB-200 decoded the ways TorchLingo decodes (A12, A13).

CI has no ``transformers`` and should not download 2.4 GB, so these run the translator against a
stand-in model and tokenizer that record what ``generate`` was asked for. What is checked is
the mapping from TorchLingo's four strategies to Hugging Face's arguments, the noise, and the
record a resumable job keeps; not NLLB's translations.
"""

import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import torch

from torchlingo.inference import add_noise, translate_lines_to_file
from torchlingo_harness.nllb import NLLBTranslator


class _Batch(dict):
    def to(self, device):
        return self


class _Tokenizer:
    src_lang = "tel_Telu"
    unk_token_id = 3

    def convert_tokens_to_ids(self, token):
        return 256047 if token == "eng_Latn" else self.unk_token_id

    def __call__(self, sentences, return_tensors, padding):
        return _Batch(input_ids=torch.zeros(len(sentences), 2, dtype=torch.long))

    def batch_decode(self, ids, skip_special_tokens):
        # Each "translation" is the row number within the batch, plus a draw if sampled.
        return [" ".join(f"w{int(t)}" for t in row) for row in ids]


class _Model:
    def __init__(self):
        self.calls = []

    def generate(self, input_ids, **options):
        self.calls.append(options)
        rows = input_ids.size(0)
        if options["do_sample"]:
            return torch.randint(0, 1000, (rows, 6))
        return torch.arange(rows * 6).reshape(rows, 6)


def _translator():
    return NLLBTranslator(
        _Model(), _Tokenizer(), "eng_Latn", torch.device("cpu"), "stub"
    )


class TestStrategies(unittest.TestCase):
    sentences = tuple(f"sentence {i}" for i in range(5))

    def test_greedy_and_beam(self):
        nllb = _translator()
        nllb(self.sentences, decode_strategy="greedy", batch_size=16)
        nllb(self.sentences, decode_strategy="beam", beam_size=4, batch_size=16)
        greedy, beam = nllb.model.calls
        self.assertEqual((greedy["num_beams"], greedy["do_sample"]), (1, False))
        self.assertEqual((beam["num_beams"], beam["do_sample"]), (4, False))
        self.assertEqual(greedy["forced_bos_token_id"], 256047)

    def test_sampling_samples_the_whole_distribution_and_is_seeded(self):
        nllb = _translator()
        first = nllb(self.sentences, decode_strategy="sample", seed=3, batch_size=2)
        again = nllb(self.sentences, decode_strategy="sample", seed=3, batch_size=2)
        other = nllb(self.sentences, decode_strategy="sample", seed=4, batch_size=2)
        self.assertEqual(first, again)
        self.assertNotEqual(first, other)
        options = nllb.model.calls[0]
        # Hugging Face samples among the top 50 unless told otherwise.
        self.assertEqual((options["do_sample"], options["top_k"]), (True, 0))
        nllb(self.sentences, decode_strategy="sample", top_k=10, temperature=0.5)
        self.assertEqual(
            (nllb.model.calls[-1]["top_k"], nllb.model.calls[-1]["temperature"]),
            (10, 0.5),
        )

    def test_beam_plus_noise_is_beam_then_add_noise(self):
        nllb = _translator()
        beam = nllb(self.sentences, decode_strategy="beam")
        noisy = nllb(self.sentences, decode_strategy="beam+noise", seed=[7] * 5)
        self.assertEqual(noisy, [add_noise(text, seed=7) for text in beam])

    def test_unknown_strategy_is_refused(self):
        with self.assertRaisesRegex(ValueError, "Unknown decode_strategy"):
            _translator()(self.sentences, decode_strategy="nucleus")

    def test_a_resumable_job_records_the_translator(self):
        nllb = _translator()
        out = Path(tempfile.mkdtemp()) / "mono.bt.en"
        settings = nllb.settings(decode_strategy="greedy")
        self.assertEqual(
            settings,
            {
                "translator": "stub",
                "src_lang": "tel_Telu",
                "tgt_lang": "eng_Latn",
                "decode_strategy": "greedy",
            },
        )

        def translate(chunk, first):
            return nllb(chunk, decode_strategy="greedy")

        lines = translate_lines_to_file(
            translate, self.sentences, out, settings, chunk_size=2, verbose=False
        )
        self.assertEqual(len(lines), 5)
        with self.assertRaisesRegex(ValueError, "different job"):
            translate_lines_to_file(
                translate,
                self.sentences,
                out,
                nllb.settings(decode_strategy="beam"),
                verbose=False,
            )


class TestTheHarnessStaysOutside(unittest.TestCase):
    def test_importing_torchlingo_does_not_load_the_harness(self):
        """TorchLingo stays lean: the accessory is loaded only when asked for (Eric, Oct 21)."""
        code = "import sys, torchlingo; print(any(m.startswith('torchlingo_harness') for m in sys.modules))"
        result = subprocess.run(
            [sys.executable, "-c", code], capture_output=True, text=True, check=True
        )
        self.assertEqual(result.stdout.strip(), "False")


class TestTranslateLinesToFile(unittest.TestCase):
    def test_a_translator_that_drops_a_sentence_is_stopped(self):
        out = Path(tempfile.mkdtemp()) / "x.txt"
        with self.assertRaisesRegex(ValueError, "exactly one per sentence"):
            translate_lines_to_file(
                lambda chunk, first: chunk[:-1], ["a", "b"], out, {}, verbose=False
            )

    def test_first_is_the_chunk_s_position_in_the_job(self):
        out = Path(tempfile.mkdtemp()) / "x.txt"
        lines = translate_lines_to_file(
            lambda chunk, first: [f"{first}:{s}" for s in chunk],
            ["a", "b", "c"],
            out,
            {},
            chunk_size=2,
            verbose=False,
        )
        self.assertEqual(lines, ["0:a", "0:b", "2:c"])


if __name__ == "__main__":
    unittest.main()
