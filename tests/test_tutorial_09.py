"""Tutorial 9 runs from its own cells, with NLLB stubbed.

The tutorial fine-tunes NLLB-200 on a GPU in Colab, so CI cannot execute it: CI has no GPU, and
the repository's environment has no Hugging Face ``transformers``. These tests run every cell
but the install and the one that loads NLLB, with a tokenizer and a model small enough to train
in a second. The stub tokenizer builds sequences the way NLLB's does (the language code, the
pieces, then ``</s>``), so the exercise's check against "NLLB's tokenizer" means the same thing
here; the stub translator writes the reference once the model has trained and the English
before, so fine-tuning has a known effect on the report.
"""

import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from typing import ClassVar
from unittest import mock

import pandas as pd
import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parent))

from notebook_cells import code_cells, colab_faked, run_cells, substitute

NOTEBOOK = (
    Path(__file__).resolve().parent.parent
    / "docs/docs/tutorials/09-fine-tuning-nllb.ipynb"
)
MODEL_CELL = "AutoModelForSeq2SeqLM"
WITH_CODES = 'raise NotImplementedError("write with_codes, then run this cell again")'
ANSWER = "return ([source_code_id, *source_ids, end_id], [target_code_id, *target_ids, end_id])"
PREDICTED = {'PREDICTION = ""': 'PREDICTION = "a few BLEU over off the shelf"'}
SMALL = {
    "STEPS = 2000": "STEPS = 6",
    "EVAL_EVERY = 250": "EVAL_EVERY = 3",
    "BATCH_SIZE = 16": "BATCH_SIZE = 4",
}
CODES = ["eng_Latn", "spa_Latn", "deu_Latn", "tgl_Latn", "smo_Latn", "fij_Latn"]


class Batch(dict):
    """What a Hugging Face tokenizer returns: a dict that moves to a device."""

    def to(self, device):
        return Batch({k: v.to(device) for k, v in self.items()})


class StubTokenizer:
    """NLLB's tokenizer in miniature: a code, the pieces (here words), then ``</s>``."""

    pad_token_id, eos_token_id = 1, 2

    def __init__(self):
        self.additional_special_tokens = list(CODES)
        self.src_lang, self.tgt_lang = "eng_Latn", "spa_Latn"
        self.vocab = {"<pad>": 1, "</s>": 2, **{c: 10 + i for i, c in enumerate(CODES)}}

    def tokenize(self, text):
        return text.split()

    def convert_tokens_to_ids(self, tokens):
        if isinstance(tokens, str):
            return self.vocab.setdefault(tokens, 100 + len(self.vocab))
        return [self.convert_tokens_to_ids(t) for t in tokens]

    def convert_ids_to_tokens(self, ids):
        names = {i: t for t, i in self.vocab.items()}
        return [names[i] for i in ids]

    def encode(self, text, code):
        return [
            self.convert_tokens_to_ids(code),
            *self.convert_tokens_to_ids(text.split()),
            2,
        ]

    def __call__(self, text, text_target=None, return_tensors=None, **_):
        if isinstance(text, str):
            return {
                "input_ids": self.encode(text, self.src_lang),
                "labels": self.encode(text_target, self.tgt_lang),
            }

        def padded(rows):
            width = max(map(len, rows))
            return torch.tensor(
                [r + [self.pad_token_id] * (width - len(r)) for r in rows]
            )

        ids = padded([self.encode(t, self.src_lang) for t in text])
        return Batch(
            input_ids=ids,
            attention_mask=(ids != self.pad_token_id).long(),
            labels=padded([self.encode(t, self.tgt_lang) for t in text_target]),
        )


class StubModel(nn.Module):
    """One weight, whose loss training lowers; it says whether the model has been trained."""

    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(1))

    def forward(self, input_ids, attention_mask, labels):
        return SimpleNamespace(
            loss=(self.weight**2).sum() * (labels != -100).float().mean()
        )

    @property
    def trained(self):
        # Six steps at the tutorial's learning rate move the weight by about 3e-4.
        return self.weight.item() < 1.0


def stub_namespace(namespace: dict) -> dict:
    """What the NLLB cell defines, with a translator whose output fine-tuning changes."""
    model = StubModel()
    by_source = {}

    class Translator:
        def __init__(self, model, tokenizer, tgt_lang, device, name):
            self.model, self.tgt_lang = model, tgt_lang

        def __call__(self, sentences, **_):
            if not by_source:
                test = namespace["test"]
                by_source.update(zip(test.src, test.tgt))
            if self.model.trained:
                return [by_source.get(s, s) for s in sentences]
            return [f"{self.tgt_lang} {s}" for s in sentences]

    namespace.update(
        tokenizer=StubTokenizer(),
        model=model,
        NLLB_MODEL="stub-nllb",
        NLLBTranslator=Translator,
        NLLB_CODES=set(CODES),
    )
    return namespace


def pairs(n: int) -> list[tuple[str, str]]:
    """n distinct English and made-up target sentences, five to eight words long."""
    return [
        (
            f"sentence {i} is in English" + " today" * (i % 4),
            f"frase {i} esta en lengua" + " hoy" * (i % 4),
        )
        for i in range(n)
    ]


class FakeAdafactor(torch.optim.SGD):
    """Adafactor's arguments, SGD's update: the training cell imports it from transformers."""

    made_with: ClassVar[dict] = {}

    def __init__(self, params, lr, **options):
        FakeAdafactor.made_with = {"lr": lr, **options}
        super().__init__(params, lr=lr)


class NotebookCase(unittest.TestCase):
    def setUp(self):
        # Only these two entries, put back as they were: restoring all of sys.modules would
        # unload the parts of torch an optimizer step loads, and torch cannot load them twice.
        optimization = SimpleNamespace(Adafactor=FakeAdafactor)
        fakes = {
            "transformers": SimpleNamespace(optimization=optimization),
            "transformers.optimization": optimization,
        }
        for name, fake in fakes.items():
            self.addCleanup(self._restore_module, name, sys.modules.get(name))
            sys.modules[name] = fake
        scratch = tempfile.TemporaryDirectory()
        self.addCleanup(scratch.cleanup)
        self.drive = Path(scratch.name)
        (self.drive / "data").mkdir()
        rows = [(s, t, "talk", "transcript") for s, t in pairs(4500)]
        pd.DataFrame(rows, columns=["src", "tgt", "talk", "kind"]).to_csv(
            self.drive / "data" / "example.tsv", sep="\t", index=False
        )
        self.out = self.drive / "CS479" / "tutorial9"

    @staticmethod
    def _restore_module(name, original):
        if original is None:
            sys.modules.pop(name, None)
        else:
            sys.modules[name] = original

    def cells(self, swaps: dict) -> list[tuple[int, str]]:
        return substitute(
            [c for c in code_cells(NOTEBOOK) if "%pip" not in c[1]], swaps
        )

    def run_notebook(self, swaps=None, answered=True, predicted=True):
        """Run the cells in the faked Drive, NLLB stubbed; return (namespace, printed)."""
        swaps = {**SMALL, **(swaps or {})}
        if answered:
            swaps[WITH_CODES] = ANSWER
        if predicted:
            swaps.update(PREDICTED)
        cells = self.cells(swaps)
        model_at = next(i for i, src in cells if MODEL_CELL in src)
        before = [c for c in cells if c[0] < model_at]
        after = [c for c in cells if c[0] > model_at]
        with colab_faked(self.drive):
            namespace, printed = run_cells(before)
            namespace, more = run_cells(after, stub_namespace(namespace))
        return namespace, printed + more

    def a8_split(self, language: str, with_a9: bool = False) -> dict:
        """A student's A8 folder on the faked Drive; the swaps that point the notebook at it."""
        split = self.drive / "CS479" / "assignment8"
        split.mkdir(parents=True)
        data = pd.DataFrame(pairs(300), columns=["src", "tgt"])
        for name, rows in (
            ("train", data[:200]),
            ("val", data[200:250]),
            ("test", data[250:]),
        ):
            rows.to_csv(split / f"{name}.tsv", sep="\t", index=False)
        if with_a9:
            (split / "test-a9.hyp").write_text(
                "\n".join(data[250:].tgt) + "\n", encoding="utf-8"
            )
        return {
            'LANGUAGE = "Spanish"': f"LANGUAGE = {language!r}",
            "A8_SPLIT = None": f"A8_SPLIT = {str(split)!r}",
        }


class TestEndToEnd(NotebookCase):
    def test_on_the_sample_fine_tuning_beats_the_model_off_the_shelf(self):
        namespace, printed = self.run_notebook()
        self.assertIn("The built-in English-Spanish sample", printed)
        self.assertIn("Yours match NLLB's tokenizer", printed)
        self.assertEqual(namespace["TARGET_CODE"], "spa_Latn")

        scores = pd.read_csv(self.out / "scores.tsv", sep="\t")
        self.assertEqual(sorted(set(scores.system)), ["fine-tuned", "off the shelf"])
        self.assertEqual(sorted(set(scores.decoding)), ["beam 5", "greedy"])
        by = scores.set_index(["system", "decoding"])
        for how in ("greedy", "beam 5"):
            self.assertGreater(
                by.chrf["fine-tuned", how], by.chrf["off the shelf", how]
            )
        for name in (
            "off-the-shelf-greedy",
            "off-the-shelf-beam5",
            "fine-tuned-greedy",
            "fine-tuned-beam5",
        ):
            self.assertTrue((self.out / f"{name}.hyp").is_file(), name)

        run = json.loads((self.out / "t9-run.json").read_text(encoding="utf-8"))
        self.assertEqual((run["steps"], run["target_code"]), (6, "spa_Latn"))
        self.assertEqual(len(namespace["state"].val_losses), 2)  # at steps 3 and 6
        self.assertIn("of 200 greedy translations changed", printed)

    def test_adafactor_with_a_fixed_rate_so_checkpoints_fit_a_drive(self):
        """AdamW's state made each checkpoint 6.9 GB, two of them on a 15 GB Drive (Oct 25)."""
        self.run_notebook()
        self.assertEqual(
            FakeAdafactor.made_with,
            {
                "lr": 1e-4,
                "scale_parameter": False,
                "relative_step": False,
                "warmup_init": False,
            },
        )

    def test_it_stops_before_training_when_drive_is_too_full(self):
        full = mock.patch("shutil.disk_usage", return_value=SimpleNamespace(free=2e9))
        with full, self.assertRaises(SystemExit) as stopped:
            self.run_notebook()
        self.assertIn("2.0 GB free", str(stopped.exception))

    def test_a_rerun_resumes_from_the_checkpoint_rather_than_training_again(self):
        self.run_notebook()
        checkpoints = self.out / "checkpoints"
        self.assertTrue(any(checkpoints.rglob("best.pt")))
        namespace, printed = self.run_notebook()
        self.assertEqual(namespace["state"].global_step, 6)
        self.assertNotIn("s a step", printed)  # no step trained, so no estimate

    def test_the_students_split_and_a9_translations_are_read(self):
        namespace, printed = self.run_notebook(self.a8_split("German", with_a9=True))
        self.assertEqual((len(namespace["train"]), len(namespace["test"])), (200, 50))
        self.assertEqual(namespace["TARGET_CODE"], "deu_Latn")
        scores = pd.read_csv(self.out / "scores.tsv", sep="\t")
        self.assertIn("A9, scratch", set(scores.system))


class TestTheStops(NotebookCase):
    def test_as_shipped_it_stops_at_the_exercise(self):
        with self.assertRaises(NotImplementedError):
            self.run_notebook(answered=False)

    def test_a_wrong_with_codes_fails_with_what_it_returned(self):
        wrong = "return ([*source_ids, source_code_id, end_id], [*target_ids, target_code_id, end_id])"
        with self.assertRaises(AssertionError) as stopped:
            self.run_notebook({WITH_CODES: wrong}, answered=False)
        self.assertIn("yours returned", str(stopped.exception))

    def test_it_stops_while_the_prediction_is_empty(self):
        with self.assertRaises(AssertionError) as stopped:
            self.run_notebook(predicted=False)
        self.assertIn("Write your prediction", str(stopped.exception))

    def test_language_and_split_go_together(self):
        with self.assertRaises(SystemExit) as stopped:
            self.run_notebook({'LANGUAGE = "Spanish"': 'LANGUAGE = "German"'})
        self.assertIn("together", str(stopped.exception))


class TestAnUncoveredLanguage(NotebookCase):
    def test_it_stops_until_candidates_are_listed(self):
        with self.assertRaises(AssertionError) as stopped:
            self.run_notebook(self.a8_split("Hmong"))
        self.assertIn("NLLB does not cover Hmong", str(stopped.exception))

    def test_a_candidate_must_be_an_nllb_code(self):
        swaps = {
            **self.a8_split("Hmong"),
            "CANDIDATES = []": 'CANDIDATES = ["hmn_Latn"]',
        }
        with self.assertRaises(AssertionError) as stopped:
            self.run_notebook(swaps)
        self.assertIn("not NLLB codes", str(stopped.exception))

    def test_it_stops_until_a_borrowed_code_and_a_reason_are_given(self):
        swaps = {
            **self.a8_split("Hmong"),
            "CANDIDATES = []": 'CANDIDATES = ["tgl_Latn", "smo_Latn"]',
        }
        with self.assertRaises(AssertionError) as stopped:
            self.run_notebook(swaps)
        self.assertIn("Set BORROW", str(stopped.exception))

    def test_the_borrowed_code_becomes_the_target_and_each_candidate_is_scored(self):
        swaps = {
            **self.a8_split("Hmong"),
            "CANDIDATES = []": 'CANDIDATES = ["tgl_Latn", "smo_Latn"]',
            'BORROW = ""': 'BORROW = "smo_Latn"',
            'REASON = ""': 'REASON = "a test reason"',
        }
        namespace, printed = self.run_notebook(swaps)
        self.assertEqual(namespace["TARGET_CODE"], "smo_Latn")
        self.assertIn("tgl_Latn   chrF", printed)
        self.assertIn("smo_Latn   chrF", printed)
        run = json.loads((self.out / "t9-run.json").read_text(encoding="utf-8"))
        self.assertEqual(run["reason"], "a test reason")


class TestTheCodeTable(unittest.TestCase):
    """Checked by hand against NLLB-200's tokenizer on 2026-10-25: these are its codes."""

    def test_every_listed_code_is_written_as_nllb_writes_one(self):
        source = next(
            src for _, src in code_cells(NOTEBOOK) if "COURSE_CODES = {" in src
        )
        namespace: dict = {}
        table = source.split("assert LANGUAGE")[0].split("compute_chrf\n", 1)[1]
        exec(table, namespace)  # noqa: S102
        codes = namespace["COURSE_CODES"]
        self.assertEqual(
            sorted(k for k, v in codes.items() if v is None),
            ["Efik", "Hmong", "Kosraean", "Palauan", "Pohnpeian", "Yapese"],
        )
        for language, code in codes.items():
            if code is not None:
                self.assertRegex(code, r"^[a-z]{3}_[A-Z][a-z]{3}$", language)


if __name__ == "__main__":
    unittest.main()
