"""torchlingo_harness.nllb: NLLB-200 decoded the ways TorchLingo decodes (A12, A13), and
fine-tuned on a student's pairs (A14).

CI has no ``transformers`` and should not download 2.4 GB, so these run the translator against a
stand-in model and tokenizer that record what ``generate`` was asked for. What is checked is
the mapping from TorchLingo's four strategies to Hugging Face's arguments, the noise, and the
record a resumable job keeps; not NLLB's translations. Fine-tuning is checked the same way: the
sequences it trains on, the steps, the checkpoints and the stop when Drive is too full.
"""

import contextlib
import io
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

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


class _WordTokenizer:
    """NLLB's tokenizer in miniature: words for pieces, codes as their own ids."""

    pad_token_id, eos_token_id = 1, 2

    def __init__(self):
        self.ids = {"eng_Latn": 10, "spa_Latn": 11}

    def tokenize(self, text):
        return text.split()

    def convert_tokens_to_ids(self, tokens):
        if isinstance(tokens, str):
            return self.ids.setdefault(tokens, 100 + len(self.ids))
        return [self.convert_tokens_to_ids(t) for t in tokens]


class _Learner(torch.nn.Module):
    """One weight, whose loss training lowers."""

    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(1))

    def forward(self, input_ids, attention_mask, labels):
        return SimpleNamespace(loss=(self.weight**2).sum())


class _Adafactor(torch.optim.SGD):
    """Adafactor's arguments, SGD's update: CI has no transformers."""

    def __init__(self, params, lr, **options):
        super().__init__(params, lr=lr)


def _pairs(n):
    import pandas as pd

    rows = [(f"word{i} go", f"palabra{i}", "eng_Latn", "spa_Latn") for i in range(n)]
    rows += [(f"palabra{i}", f"word{i} go", "spa_Latn", "eng_Latn") for i in range(n)]
    return pd.DataFrame(rows, columns=["src", "tgt", "src_code", "tgt_code"])


class TestFineTune(unittest.TestCase):
    """fine_tune and coded_batch, for A14: what NLLB trains on, and a resumable run."""

    def setUp(self):
        # Only these two entries, put back as they were; restoring all of sys.modules would
        # unload the parts of torch an optimizer step loads.
        optimization = SimpleNamespace(Adafactor=_Adafactor)
        for name, fake in (
            ("transformers", SimpleNamespace(optimization=optimization)),
            ("transformers.optimization", optimization),
        ):
            self.addCleanup(self._restore, name, sys.modules.get(name))
            sys.modules[name] = fake
        self.folder = Path(tempfile.mkdtemp())

    @staticmethod
    def _restore(name, original):
        if original is None:
            sys.modules.pop(name, None)
        else:
            sys.modules[name] = original

    def run_fine_tune(self, steps=6):
        from torchlingo.training_checkpoint import TrainingCheckpointer
        from torchlingo_harness.nllb import fine_tune

        model = _Learner()
        checkpointer = TrainingCheckpointer(
            "t", checkpoint_dir=self.folder, verbose=False
        )
        with contextlib.redirect_stdout(io.StringIO()) as printed:
            state = fine_tune(
                model,
                _WordTokenizer(),
                _pairs(20),
                _pairs(5),
                checkpointer,
                steps=steps,
                device=torch.device("cpu"),
                batch_size=4,
                eval_every=3,
            )
        return model, state, printed.getvalue()

    def test_each_row_carries_its_own_codes(self):
        from torchlingo_harness.nllb import coded_batch

        tok = _WordTokenizer()
        batch = coded_batch(tok, _pairs(1), torch.device("cpu"))
        word, go, palabra = tok.convert_tokens_to_ids(["word0", "go", "palabra0"])
        self.assertEqual(
            batch["input_ids"].tolist(), [[10, word, go, 2], [11, palabra, 2, 1]]
        )
        self.assertEqual(
            batch["labels"].tolist(), [[11, palabra, 2, -100], [10, word, go, 2]]
        )
        self.assertEqual(batch["attention_mask"].tolist(), [[1, 1, 1, 1], [1, 1, 1, 0]])

    def test_trains_the_steps_asked_and_keeps_the_best(self):
        model, state, printed = self.run_fine_tune()
        self.assertEqual(state.global_step, 6)
        self.assertEqual(len(state.val_losses), 2)  # at steps 3 and 6
        self.assertLess(model.weight.item(), 1.0)
        self.assertTrue((self.folder / "best.pt").is_file())
        self.assertIn("step      3", printed)

    def test_a_finished_run_is_not_repeated(self):
        self.run_fine_tune()
        _, state, printed = self.run_fine_tune()
        self.assertEqual(state.global_step, 6)
        self.assertNotIn("step      3", printed)

    def test_it_stops_before_training_when_the_folder_is_too_full(self):
        full = mock.patch("shutil.disk_usage", return_value=SimpleNamespace(free=2e9))
        with full, self.assertRaises(SystemExit) as stopped:
            self.run_fine_tune()
        self.assertIn("2.0 GB free", str(stopped.exception))


if __name__ == "__main__":
    unittest.main()
