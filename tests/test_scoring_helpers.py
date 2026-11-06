"""comet_score, paired_bootstrap and format_scores: the scoring A13 and A14 wrote by hand.

COMET itself is not installed here (it is heavy, and pins transformers), so its tests use a
stand-in model with COMET's predict() and, for the import, a fake `comet` module set in
sys.modules and removed again. comet_scores_in_new_process starts a real process, which finds a
fake `comet` package on PYTHONPATH.
"""

import contextlib
import io
import os
import sys
import tempfile
import textwrap
import types
import unittest
from pathlib import Path
from unittest import mock

import sacrebleu

from torchlingo import evaluation
from torchlingo.evaluation import (
    CometScores,
    comet_score,
    comet_scores_in_new_process,
    compute_bleu,
    compute_chrf,
    format_scores,
    paired_bootstrap,
    sentence_chrf,
)

REFS = [
    "the cat sat on the mat",
    "a dog ran in the park",
    "she reads a book every night",
    "we walked home in the rain",
] * 10
WORSE = ["the cat sat", "a dog ran", "she reads", "we walked home"] * 10


class FakeComet:
    """COMET's predict(): one score per sentence, and their mean as the system score."""

    def __init__(self):
        self.calls = []

    def predict(self, data, batch_size, gpus, num_workers, progress_bar):
        self.calls.append((data, batch_size, gpus))
        self.num_workers = num_workers
        scores = [1.0 if row["mt"] == row["ref"] else 0.5 for row in data]
        return types.SimpleNamespace(
            scores=scores, system_score=sum(scores) / len(scores)
        )


class TestCometScore(unittest.TestCase):
    def test_it_scores_each_sentence_and_the_system(self):
        model = FakeComet()
        got = comet_score(["s1", "s2"], ["a", "b"], ["a", "c"], model=model, gpus=0)
        self.assertEqual(got, CometScores(0.75, [1.0, 0.5]))
        data, _, gpus = model.calls[0]
        self.assertEqual(data[1], {"src": "s2", "mt": "b", "ref": "c"})
        self.assertEqual(gpus, 0)

    def test_lists_of_different_lengths_are_refused(self):
        with self.assertRaises(ValueError) as refused:
            comet_score(["s1", "s2"], ["a"], ["a", "b"], model=FakeComet())
        self.assertIn("1 translations", str(refused.exception))

    def test_a_model_name_is_loaded_once(self):
        loaded = FakeComet()
        fake = types.ModuleType("comet")
        fake.download_model = mock.Mock(return_value="/cache/model.ckpt")
        fake.load_from_checkpoint = mock.Mock(return_value=loaded)
        saved = sys.modules.get("comet")
        sys.modules["comet"] = fake
        try:
            with mock.patch.dict(evaluation._comet_models, clear=True):
                for _ in range(2):
                    comet_score(["s"], ["a"], ["a"], gpus=0)
        finally:
            if saved is None:
                del sys.modules["comet"]
            else:
                sys.modules["comet"] = saved
        fake.download_model.assert_called_once_with("Unbabel/wmt22-comet-da")
        fake.load_from_checkpoint.assert_called_once_with("/cache/model.ckpt")
        self.assertEqual(len(loaded.calls), 2)

    def test_without_comet_it_says_how_to_install_it(self):
        saved = sys.modules.get("comet")
        sys.modules["comet"] = None  # makes `import comet` raise ImportError
        try:
            with (
                mock.patch.dict(evaluation._comet_models, clear=True),
                self.assertRaises(ImportError) as refused,
            ):
                comet_score(["s"], ["a"], ["a"])
        finally:
            if saved is None:
                del sys.modules["comet"]
            else:
                sys.modules["comet"] = saved
        self.assertIn("--no-deps unbabel-comet", str(refused.exception))

    def test_installed_but_unloadable_does_not_say_install_it(self):
        # What A14 hit: COMET on disk, and an import that fails against the transformers in
        # memory. Told "not installed", a student reinstalls, which cannot help.
        saved = sys.modules.get("comet")
        sys.modules["comet"] = None
        try:
            with (
                mock.patch.dict(evaluation._comet_models, clear=True),
                mock.patch.object(evaluation.importlib.util, "find_spec"),
                self.assertRaises(ImportError) as refused,
            ):
                comet_score(["s"], ["a"], ["a"])
        finally:
            if saved is None:
                del sys.modules["comet"]
            else:
                sys.modules["comet"] = saved
        self.assertIn("comet_scores_in_new_process", str(refused.exception))
        self.assertNotIn("is not installed", str(refused.exception))

    def test_it_always_gives_comet_workers(self):
        # With gpus=0 COMET defaults to no workers, which it cannot run with on an Apple GPU.
        for gpus, workers in ((0, 2), (1, 2), (2, 4)):
            model = FakeComet()
            comet_score(["s"], ["a"], ["a"], model=model, gpus=gpus)
            self.assertEqual(model.num_workers, workers)


FAKE_COMET_PACKAGE = textwrap.dedent(
    """\
    import types


    class Model:
        def predict(self, data, batch_size, gpus, num_workers, progress_bar):
            scores = [1.0 if row["mt"] == row["ref"] else 0.5 for row in data]
            return types.SimpleNamespace(
                scores=scores, system_score=sum(scores) / len(scores)
            )


    LOADS = []


    def download_model(name):
        return name


    def load_from_checkpoint(path):
        LOADS.append(path)
        assert len(LOADS) == 1, "the model should load once, whatever the number of jobs"
        return Model()
    """
)


class TestCometScoresInNewProcess(unittest.TestCase):
    def setUp(self):
        folder = tempfile.TemporaryDirectory()
        self.addCleanup(folder.cleanup)
        self.package = Path(folder.name, "comet")
        self.package.mkdir()
        (self.package / "__init__.py").write_text(FAKE_COMET_PACKAGE, encoding="utf-8")
        path = os.pathsep.join(
            filter(None, [folder.name, os.environ.get("PYTHONPATH")])
        )
        patched = mock.patch.dict(os.environ, {"PYTHONPATH": path})
        patched.start()
        self.addCleanup(patched.stop)

    def test_it_scores_every_job_and_prints_each_as_it_finishes(self):
        jobs = {
            "A14|to-x|greedy": (["s1", "s2"], ["a", "b"], ["a", "c"]),
            "A9|to-x|greedy": (["señal"], ["日本語"], ["日本語"]),
        }
        printed = io.StringIO()
        with contextlib.redirect_stdout(printed):
            got = comet_scores_in_new_process(jobs, gpus=0)
        self.assertEqual(got["A14|to-x|greedy"], CometScores(0.75, [1.0, 0.5]))
        self.assertEqual(got["A9|to-x|greedy"], CometScores(1.0, [1.0]))
        self.assertEqual(
            printed.getvalue().splitlines(),
            [
                "COMET scored A14|to-x|greedy: 0.7500",
                "COMET scored A9|to-x|greedy: 1.0000",
            ],
        )

    def test_a_failure_there_says_so_here_with_its_output(self):
        (self.package / "__init__.py").write_text(
            'raise ImportError("cannot import name _torch_version")\n', encoding="utf-8"
        )
        with self.assertRaises(RuntimeError) as failed:
            comet_scores_in_new_process({"job": (["s"], ["a"], ["a"])}, gpus=0)
        self.assertIn("new process", str(failed.exception))
        self.assertIn("will not load in this session", str(failed.exception))

    def test_lists_of_different_lengths_are_refused_before_it_starts(self):
        with (
            mock.patch.object(evaluation.subprocess, "Popen") as popen,
            self.assertRaises(ValueError) as refused,
        ):
            comet_scores_in_new_process({"A14": (["s1", "s2"], ["a"], ["a", "b"])})
        popen.assert_not_called()
        self.assertIn("A14", str(refused.exception))


class TestOneChrfOneBleu(unittest.TestCase):
    """Every path that reports a metric reports the same metric.

    `compute_chrf` was chrF++ and `paired_bootstrap` plain chrF until Nov 2026, so A14 printed
    both under one name.
    """

    def test_every_chrf_is_sacrebleus_default(self):
        self.assertEqual(evaluation.CHRF_WORD_ORDER, 0)
        corpus = compute_chrf(WORSE, REFS).score
        self.assertAlmostEqual(
            corpus, sacrebleu.corpus_chrf(WORSE, [REFS]).score, places=9
        )
        table = paired_bootstrap({"a": WORSE, "b": REFS}, REFS, n_samples=10)
        self.assertAlmostEqual(table["a"]["chrf"], corpus, places=3)

    def test_sentence_chrf_is_the_same_chrf_one_sentence_at_a_time(self):
        scores = sentence_chrf(WORSE[:4], REFS[:4])
        for score, hypothesis, reference in zip(scores, WORSE, REFS):
            self.assertAlmostEqual(
                score, compute_chrf([hypothesis], [reference]).score, places=9
            )
            self.assertAlmostEqual(
                score, sacrebleu.sentence_chrf(hypothesis, [reference]).score, places=9
            )

    def test_sentence_chrf_needs_one_reference_each(self):
        with self.assertRaises(ValueError) as refused:
            sentence_chrf(["a", "b"], ["a"])
        self.assertIn("2 translations and 1 references", str(refused.exception))

    def test_bleu_is_the_same_in_the_table_and_alone(self):
        table = paired_bootstrap({"a": WORSE, "b": REFS}, REFS, n_samples=10)
        self.assertAlmostEqual(
            table["a"]["bleu"], compute_bleu(WORSE, REFS).score, places=3
        )


class TestPairedBootstrap(unittest.TestCase):
    def test_a_real_difference_has_a_small_p_and_the_baseline_none(self):
        scores = paired_bootstrap({"good": REFS, "worse": WORSE}, REFS, n_samples=200)
        self.assertEqual(list(scores), ["good", "worse"])
        self.assertIsNone(scores["good"]["bleu_p"])
        self.assertIsNone(scores["good"]["chrf_p"])
        self.assertEqual(scores["good"]["bleu"], 100.0)
        self.assertLess(scores["worse"]["bleu"], scores["good"]["bleu"])
        self.assertLess(scores["worse"]["bleu_p"], 0.05)
        self.assertLess(scores["worse"]["chrf_p"], 0.05)

    def test_identical_outputs_have_p_one(self):
        """sacreBLEU alone gives identical systems its smallest p, which reads as real."""
        scores = paired_bootstrap(
            {"A14": WORSE, "copy": list(WORSE)}, REFS, n_samples=200
        )
        self.assertEqual(scores["copy"]["bleu_p"], 1.0)
        self.assertEqual(scores["copy"]["chrf_p"], 1.0)

    def test_the_baseline_comes_first_whatever_the_order(self):
        scores = paired_bootstrap(
            {"worse": WORSE, "good": REFS}, REFS, baseline="good", n_samples=100
        )
        self.assertEqual(list(scores), ["good", "worse"])
        self.assertIsNone(scores["good"]["bleu_p"])

    def test_the_same_inputs_give_the_same_numbers(self):
        def run():
            return paired_bootstrap({"good": REFS, "worse": WORSE}, REFS, n_samples=100)

        self.assertEqual(run(), run())

    def test_the_tokenizer_reaches_bleu(self):
        refs = ["我喜欢猫", "他们在公园里跑步"] * 10
        hyps = ["我喜欢狗", "他们在公园里跑步"] * 10
        spaced = paired_bootstrap({"r": refs, "h": hyps}, refs, n_samples=50)
        flores = paired_bootstrap(
            {"r": refs, "h": hyps}, refs, tokenize="flores200", n_samples=50
        )
        self.assertNotEqual(spaced["h"]["bleu"], flores["h"]["bleu"])

    def test_bad_inputs_are_refused(self):
        cases = {
            "at least two": ({"only": REFS}, None),
            "is not one of the systems": ({"a": REFS, "b": WORSE}, "c"),
            "translations for": ({"a": REFS, "b": WORSE[:3]}, None),
        }
        for message, (systems, baseline) in cases.items():
            with self.subTest(message), self.assertRaises(ValueError) as refused:
                paired_bootstrap(systems, REFS, baseline=baseline, n_samples=10)
            self.assertIn(message, str(refused.exception))


class TestFormatScores(unittest.TestCase):
    def test_a_table_from_paired_bootstrap(self):
        scores = paired_bootstrap({"good": REFS, "worse": WORSE}, REFS, n_samples=100)
        lines = format_scores(scores, comet={"good": 0.91}).splitlines()
        self.assertEqual(len(lines), 2)
        self.assertTrue(lines[0].startswith("good "))
        self.assertIn("COMET 0.9100", lines[0])
        self.assertNotIn("p =", lines[0])
        self.assertIn("COMET not scored", lines[1])
        self.assertIn("p = ", lines[1])

    def test_without_comet_there_is_no_comet_column(self):
        scores = paired_bootstrap({"good": REFS, "worse": WORSE}, REFS, n_samples=50)
        self.assertNotIn("COMET", format_scores(scores))


if __name__ == "__main__":
    unittest.main()
