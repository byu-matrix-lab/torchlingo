"""Tutorial 9 (how much fine-tuning is enough, and what it costs) runs from its own cells.

The tutorial fine-tunes NLLB-200 on a GPU in Colab, so CI cannot execute it: CI has no GPU, and
the repository's environment has no Hugging Face ``transformers``. These tests run every cell
but the install, with stand-ins for ``transformers`` (a tokenizer that builds NLLB's sequences
from words, a one-weight model whose loss training lowers, and an SGD in Adafactor's place) and
for NLLB's translator, which writes the reference once the model has trained and the source
before. The fine-tuning itself is the library's, ``torchlingo_harness.nllb.fine_tune``. What
is tested is the path: the prediction stop, the curve resumed through three budgets, the
forgetting checks, the high learning rate, results kept on Drive, and the settings' checks.
"""

import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from notebook_cells import code_cells, colab_faked, run_cells, substitute

import torchlingo_harness.nllb

NOTEBOOK = (
    Path(__file__).resolve().parent.parent
    / "docs/docs/tutorials/09-fine-tuning-nllb.ipynb"
)
CODES = ["eng_Latn", "spa_Latn", "deu_Latn", "fra_Latn"]
PREDICTED = {
    "PREDICTED = {100: None, 300: None, 1000: None}": "PREDICTED = {100: 40, 300: 45, 1000: 47}",
    'FORGETTING = ""': 'FORGETTING = "into English drops a little; French barely moves"',
}


class StubTokenizer:
    """NLLB's tokenizer in miniature: words for pieces, its codes as special tokens."""

    pad_token_id, eos_token_id = 1, 2

    def __init__(self):
        self.additional_special_tokens = list(CODES)
        self.src_lang = "eng_Latn"
        self.ids = {code: 10 + i for i, code in enumerate(CODES)}

    def tokenize(self, text):
        return text.split()

    def convert_tokens_to_ids(self, tokens):
        if isinstance(tokens, str):
            return self.ids.setdefault(tokens, 100 + len(self.ids))
        return [self.convert_tokens_to_ids(t) for t in tokens]


class StubModel(torch.nn.Module):
    """One weight, whose loss training lowers; it says whether it has been trained."""

    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(1))

    def forward(self, input_ids, attention_mask, labels):
        return SimpleNamespace(loss=(self.weight**2).sum())

    @property
    def trained(self):
        return self.weight.item() < 1.0


class FakeAdafactor(torch.optim.SGD):
    """Adafactor's arguments, SGD's update."""

    def __init__(self, params, lr, **options):
        super().__init__(params, lr=lr)


class FakeTranslator:
    """NLLB decoding: the reference once trained, the source tagged with its code before."""

    references: dict = {}  # noqa: RUF012 (filled per test, read by every instance)

    def __init__(self, model, tokenizer, tgt_lang, device, name):
        self.model, self.tgt_lang = model, tgt_lang

    def __call__(self, sentences, **options):
        if self.model.trained:
            return [
                self.references.get((s, self.tgt_lang), f"[{self.tgt_lang}*] {s}")
                for s in sentences
            ]
        return [f"[{self.tgt_lang}] {s}" for s in sentences]


def pairs(n: int) -> list[tuple[str, str]]:
    """n distinct English and made-up target sentences, five to eight words long."""
    return [
        (
            f"sentence {i} is in English" + " today" * (i % 4),
            f"frase {i} esta en lengua" + " hoy" * (i % 4),
        )
        for i in range(n)
    ]


class NotebookCase(unittest.TestCase):
    def setUp(self):
        # Only these sys.modules entries change, and they are put back as they were: restoring
        # all of sys.modules would unload the parts of torch an optimizer step loads.
        fakes = {
            "transformers": SimpleNamespace(
                AutoTokenizer=SimpleNamespace(
                    from_pretrained=lambda name, **kw: StubTokenizer()
                ),
                AutoModelForSeq2SeqLM=SimpleNamespace(
                    from_pretrained=lambda name, **kw: StubModel()
                ),
            ),
            "transformers.optimization": SimpleNamespace(Adafactor=FakeAdafactor),
        }
        for name, fake in fakes.items():
            self.addCleanup(self._restore, name, sys.modules.get(name))
            sys.modules[name] = fake
        patched = mock.patch.object(
            torchlingo_harness.nllb, "NLLBTranslator", FakeTranslator
        )
        patched.start()
        self.addCleanup(patched.stop)

        scratch = tempfile.TemporaryDirectory()
        self.addCleanup(scratch.cleanup)
        self.drive = Path(scratch.name)
        (self.drive / "data").mkdir()
        rows = [(s, t, "talk", "transcript") for s, t in pairs(4500)]
        pd.DataFrame(rows, columns=["src", "tgt", "talk", "kind"]).to_csv(
            self.drive / "data" / "example.tsv", sep="\t", index=False
        )
        FakeTranslator.references = {
            **{(s, "spa_Latn"): t for s, t in pairs(4500)},
            **{(t, "eng_Latn"): s for s, t in pairs(4500)},
        }
        self.out = self.drive / "CS479" / "tutorial9"

    @staticmethod
    def _restore(name, original):
        if original is None:
            sys.modules.pop(name, None)
        else:
            sys.modules[name] = original

    def run_notebook(self, swaps=None, predicted=True):
        """Run every cell but the install in the faked Drive; return (namespace, printed)."""
        cells = [c for c in code_cells(NOTEBOOK) if "%pip" not in c[1]]
        cells = substitute(cells, {**(PREDICTED if predicted else {}), **(swaps or {})})
        with colab_faked(self.drive):
            return run_cells(cells)

    def results(self) -> dict:
        return json.loads((self.out / "t9-results.json").read_text(encoding="utf-8"))


class TestTheCurveAndTheCosts(NotebookCase):
    def test_runs_end_to_end_on_the_sample(self):
        _, printed = self.run_notebook()
        results = self.results()
        self.assertEqual(sorted(results["curve"]), ["100", "1000", "300"])
        before_x, before_en = results["before"]["chrf"]
        after_x, after_en = results["curve"]["1000"]["chrf"]
        self.assertGreater(after_x, before_x, "fine-tuning raised chrF into Spanish")
        self.assertEqual(len(results["before"]["third"]), 5)
        self.assertEqual(len(results["third_after"]), 5)
        self.assertIn("chrF into English, the direction not trained", printed)
        self.assertIn("After 100 steps at 3e-3", printed)
        self.assertIn("You expected: into English drops a little", printed)
        self.assertTrue(any((self.out / "checkpoints").rglob("best.pt")))

    def test_one_run_resumed_through_the_three_budgets(self):
        self.run_notebook()
        state = torch.load(
            self.out / "checkpoints" / "latest.pt",
            map_location="cpu",
            weights_only=False,
        )["state"]
        self.assertEqual(state["global_step"], 1000)
        self.assertEqual(len(state["val_losses"]), 10)  # every 100 steps, one run

    def test_a_rerun_reuses_the_saved_results_instead_of_training(self):
        self.run_notebook()
        _, printed = self.run_notebook()
        # Between the prediction and the forgetting check is the curve's cell: it trains nothing.
        curve = printed.split("Predictions recorded.")[1].split(
            "the direction not trained"
        )[0]
        self.assertNotIn("step    100", curve)
        self.assertIn(
            "  1000", curve
        )  # the table still printed, from the saved results

    def test_the_students_split_and_code(self):
        split = self.drive / "CS479" / "assignment8"
        split.mkdir(parents=True)
        data = pd.DataFrame(pairs(400), columns=["src", "tgt"])
        for name, rows in (
            ("train", data[:300]),
            ("val", data[300:350]),
            ("test", data[350:]),
        ):
            rows.to_csv(split / f"{name}.tsv", sep="\t", index=False)
        FakeTranslator.references.update({(s, "deu_Latn"): t for s, t in pairs(400)})
        namespace, _ = self.run_notebook(
            {
                "A8_SPLIT = None": f"A8_SPLIT = {str(split)!r}",
                'CODE = "spa_Latn"': 'CODE = "deu_Latn"',
            }
        )
        self.assertEqual(len(namespace["test"]), 50)
        self.assertEqual(set(namespace["train"].tgt_code), {"deu_Latn"})


class TestTheStops(NotebookCase):
    def test_it_stops_until_the_predictions_are_written(self):
        with self.assertRaises(AssertionError) as stopped:
            self.run_notebook(predicted=False)
        self.assertIn("Write a chrF for each budget", str(stopped.exception))

    def test_the_sample_is_spanish(self):
        with self.assertRaises(AssertionError) as stopped:
            self.run_notebook({'CODE = "spa_Latn"': 'CODE = "deu_Latn"'})
        self.assertIn("the built-in sample is Spanish", str(stopped.exception))

    def test_the_third_language_must_be_a_third_one(self):
        with self.assertRaises(AssertionError) as stopped:
            self.run_notebook({'THIRD = "fra_Latn"': 'THIRD = "eng_Latn"'})
        self.assertIn("THIRD neither English nor yours", str(stopped.exception))


if __name__ == "__main__":
    unittest.main()
