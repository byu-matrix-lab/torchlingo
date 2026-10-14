"""Tests for the helpers the notebooks call instead of carrying their own plumbing.

``torchlingo.colab`` (``setup``, ``fetch_data`` and ``course_dir``), ``preprocessing.split_exact`` and ``diagnostics.padding_report``
replace code every course notebook used to write out by hand, and ``evaluate_model`` is the
scoring function callers use (Task #86: it had no test of its own). The network and Colab
are faked, since CI has neither; the real Colab path is exercised by
``scripts/student_path.sh``.
"""

import os
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from torchlingo import colab
from torchlingo.config import Config
from torchlingo.data_processing import BucketBatchSampler, NMTDataset, collate_fn
from torchlingo.data_processing.vocab import SimpleVocab
from torchlingo.diagnostics import padding_report
from torchlingo.evaluation import (
    compute_bleu,
    compute_chrf,
    compute_ter,
    evaluate_model,
    save_translations,
)
from torchlingo.inference import translate_batch
from torchlingo.models import SimpleTransformer
from torchlingo.preprocessing import split_exact

POINTER = b"version https://git-lfs.github.com/spec/v1\noid sha256:abc\nsize 3\n"


class TestFetchData(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)

    def tearDown(self):
        self._tmp.cleanup()

    def fetch(self, responses, paths=("data/x.bin",)):
        """Run fetch_data with each URL answered from ``responses`` by host."""
        calls = []

        def fake_download(url):
            calls.append(url)
            host = "lfs" if "media.githubusercontent.com" in url else "raw"
            return responses[host]

        with mock.patch.object(colab, "_download", side_effect=fake_download):
            result = colab.fetch_data(paths, root=self.root)
        return result, calls

    def test_an_ordinary_file_comes_from_the_raw_host(self):
        (path,), calls = self.fetch({"raw": b"content"})
        self.assertEqual(path.read_bytes(), b"content")
        self.assertEqual(len(calls), 1)
        self.assertIn("raw.githubusercontent.com", calls[0])

    def test_an_lfs_file_is_fetched_from_the_lfs_host(self):
        """The raw host answers an LFS file with its pointer; that must not be what lands."""
        (path,), calls = self.fetch({"raw": POINTER, "lfs": b"real bytes"})
        self.assertEqual(path.read_bytes(), b"real bytes")
        self.assertIn("media.githubusercontent.com", calls[1])

    def test_a_present_file_is_left_alone(self):
        target = self.root / "data" / "x.bin"
        target.parent.mkdir()
        target.write_bytes(b"mine")
        (path,), calls = self.fetch({"raw": b"theirs"})
        self.assertEqual(path.read_bytes(), b"mine")
        self.assertEqual(calls, [])

    def test_a_pointer_left_by_a_clone_without_lfs_is_replaced(self):
        target = self.root / "data" / "x.bin"
        target.parent.mkdir()
        target.write_bytes(POINTER)
        (path,), _ = self.fetch({"raw": POINTER, "lfs": b"real bytes"})
        self.assertEqual(path.read_bytes(), b"real bytes")

    def test_no_partial_file_is_left_behind(self):
        self.fetch({"raw": b"content"})
        self.assertEqual(list(self.root.rglob("*.partial")), [])


class TestSetup(unittest.TestCase):
    def test_off_colab_it_reports_the_cpu_and_needs_nothing(self):
        with mock.patch("torch.cuda.is_available", return_value=False):
            env = colab.setup()
        self.assertEqual(env.device.type, "cpu")
        self.assertFalse(env.in_colab)
        self.assertIsNone(env.drive_dir)

    def test_requiring_a_gpu_without_one_stops_with_instructions(self):
        with (
            mock.patch("torch.cuda.is_available", return_value=False),
            self.assertRaisesRegex(RuntimeError, "Change runtime type"),
        ):
            colab.setup(gpu=True)

    def test_drive_off_colab_is_the_current_directory(self):
        with mock.patch("torch.cuda.is_available", return_value=False):
            env = colab.setup(drive=True)
        self.assertEqual(env.drive_dir, Path("."))

    def test_drive_in_colab_mounts_and_returns_mydrive(self):
        """The Colab path, with google.colab faked and the mount point made writable."""
        with tempfile.TemporaryDirectory() as tmp:
            mounted = []

            def fake_mount(path, **kwargs):
                mounted.append(path)
                (Path(path) / "MyDrive").mkdir(parents=True)

            fake = types.ModuleType("google.colab")
            fake.drive = types.SimpleNamespace(mount=fake_mount)
            google = sys.modules.get("google") or types.ModuleType("google")
            with (
                mock.patch.dict(sys.modules, {"google": google, "google.colab": fake}),
                mock.patch.object(google, "colab", fake, create=True),
                mock.patch.dict(os.environ, {"TORCHLINGO_DRIVE_MOUNT": tmp}),
                mock.patch("torch.cuda.is_available", return_value=False),
            ):
                env = colab.setup(drive=True)
            self.assertTrue(env.in_colab)
            self.assertEqual(env.drive_dir, Path(tmp) / "MyDrive")
            self.assertEqual(mounted, [tmp])


class TestCourseDir(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.drive = Path(self.tmp.name)
        self.a8 = self.drive / "CS479" / "assignment8"

    def tearDown(self):
        self.tmp.cleanup()

    def test_found_files_return_the_folder(self):
        (self.a8 / "best-a9").mkdir(parents=True)
        (self.a8 / "train.tsv").write_text("src\ttgt\n")
        (self.a8 / "best-a9" / "model_best.pt").write_bytes(b"")
        folder = colab.course_dir(
            self.drive, "assignment8", needs=["train.tsv", "best-a9/model_best.pt"]
        )
        self.assertEqual(folder, self.a8)

    def test_every_missing_file_is_named_with_the_folder_and_hint(self):
        self.a8.mkdir(parents=True)
        (self.a8 / "train.tsv").write_text("src\ttgt\n")
        with self.assertRaises(SystemExit) as stopped:
            colab.course_dir(
                self.drive,
                "assignment8",
                needs=["train.tsv", "val.tsv", "test.tsv"],
                hint="Step 5 of the A8 notebook writes them.",
            )
        message = str(stopped.exception)
        self.assertIn(str(self.a8), message)
        self.assertIn("val.tsv, test.tsv.", message)
        self.assertNotIn("train.tsv", message)
        self.assertTrue(message.endswith("Step 5 of the A8 notebook writes them."))
        self.assertNotIn("does not exist", message)

    def test_a_missing_folder_says_so(self):
        with self.assertRaisesRegex(SystemExit, "folder itself does not exist"):
            colab.course_dir(self.drive, "assignment8", needs=["train.tsv"])

    def test_create_makes_the_folder_for_a_notebook_that_writes(self):
        folder = colab.course_dir(self.drive, "assignment12", create=True)
        self.assertTrue(folder.is_dir())
        self.assertEqual(folder, self.drive / "CS479" / "assignment12")

    def test_without_an_assignment_it_is_the_course_folder(self):
        self.assertEqual(colab.course_dir(self.drive), self.drive / "CS479")

    def test_without_drive_it_says_to_ask_setup_for_it(self):
        with self.assertRaisesRegex(ValueError, r"setup\(drive=True\)"):
            colab.course_dir(None, "assignment8")

    def test_as_a_notebook_calls_it_with_colab_faked(self):
        """The cell the notebooks would write: setup's drive_dir, then the folder."""
        from notebook_cells import colab_faked

        self.a8.mkdir(parents=True)
        (self.a8 / "test.tsv").write_text("src\ttgt\n")
        with (
            colab_faked(self.drive),
            mock.patch("torch.cuda.is_available", return_value=False),
        ):
            env = colab.setup(drive=True)
            folder = colab.course_dir(env.drive_dir, "assignment8", needs=["test.tsv"])
            self.assertEqual(folder.resolve(), self.a8.resolve())


class TestSplitExact(unittest.TestCase):
    def setUp(self):
        self.frame = pd.DataFrame(
            {"src": [f"s{i}" for i in range(50)], "tgt": [f"t{i}" for i in range(50)]}
        )

    def test_sizes_are_exact_and_the_sets_disjoint(self):
        train, val, test = split_exact(self.frame, n_val=5, n_test=7, seed=1)
        self.assertEqual((len(train), len(val), len(test)), (38, 5, 7))
        self.assertEqual(
            len(set(train.src) | set(val.src) | set(test.src)), len(self.frame)
        )

    def test_the_seed_fixes_the_test_set(self):
        self.assertEqual(
            list(split_exact(self.frame, 5, 7, seed=3)[2].src),
            list(split_exact(self.frame, 5, 7, seed=3)[2].src),
        )
        self.assertNotEqual(
            list(split_exact(self.frame, 5, 7, seed=3)[2].src),
            list(split_exact(self.frame, 5, 7, seed=4)[2].src),
        )

    def test_it_matches_the_split_the_a8_notebook_wrote_by_hand(self):
        """A student's test set must not change when the notebook switches to this."""
        order = np.random.default_rng(479).permutation(len(self.frame))
        by_hand = self.frame.iloc[order[:7]].reset_index(drop=True)
        self.assertTrue(by_hand.equals(split_exact(self.frame, 5, 7, seed=479)[2]))

    def test_nothing_left_for_training_is_an_error(self):
        with self.assertRaises(ValueError):
            split_exact(self.frame, n_val=25, n_test=25, seed=0)


class TestPaddingReport(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        rng = np.random.default_rng(0)
        lengths = np.concatenate([rng.integers(2, 5, 200), rng.integers(40, 60, 200)])
        rows = [" ".join(f"w{j}" for j in range(n)) for n in lengths]
        path = Path(self._tmp.name) / "train.tsv"
        pd.DataFrame({"src": rows, "tgt": rows}).to_csv(path, sep="\t", index=False)
        self.dataset = NMTDataset(path, config=Config(num_workers=0))

    def tearDown(self):
        self._tmp.cleanup()

    def test_bucketing_carries_less_padding_than_a_shuffle(self):
        loader = DataLoader(
            self.dataset,
            batch_sampler=BucketBatchSampler(self.dataset, batch_size=16),
            collate_fn=collate_fn,
        )
        report = padding_report(loader)
        self.assertLess(report.padded_tokens, report.shuffled_tokens / 2)
        self.assertEqual(report.dropped, len(self.dataset) - 16 * len(loader))
        self.assertIn("less", str(report))

    def test_a_plain_shuffle_drops_nothing_and_saves_nothing_much(self):
        loader = DataLoader(
            self.dataset, batch_size=16, shuffle=True, collate_fn=collate_fn
        )
        report = padding_report(loader)
        self.assertEqual(report.dropped, 0)
        self.assertGreaterEqual(report.padded_tokens, report.real_tokens)


class TestEvaluateModel(unittest.TestCase):
    """Task #86: the aggregator must agree with the metrics it aggregates."""

    @classmethod
    def setUpClass(cls):
        cls.src = ["the cat sleeps", "the dog runs", "a cat runs", "the dog sleeps"]
        cls.tgt = [
            "el gato duerme",
            "el perro corre",
            "un gato corre",
            "el perro duerme",
        ]
        cls.src_vocab, cls.tgt_vocab = SimpleVocab(min_freq=1), SimpleVocab(min_freq=1)
        cls.src_vocab.build_vocab(cls.src)
        cls.tgt_vocab.build_vocab(cls.tgt)
        torch.manual_seed(0)
        cls.config = Config(
            d_model=16, n_heads=2, num_encoder_layers=1, num_decoder_layers=1, d_ff=32
        )
        cls.model = SimpleTransformer(
            src_vocab_size=len(cls.src_vocab),
            tgt_vocab_size=len(cls.tgt_vocab),
            config=cls.config,
        ).eval()
        cls.predictions = translate_batch(
            cls.model,
            cls.src,
            cls.src_vocab,
            cls.tgt_vocab,
            decode_strategy="greedy",
            device=torch.device("cpu"),
            config=cls.config,
        )

    def evaluate(self, **kwargs):
        return evaluate_model(
            self.model,
            src_vocab=self.src_vocab,
            tgt_vocab=self.tgt_vocab,
            device=torch.device("cpu"),
            src_sentences=self.src,
            tgt_sentences=self.tgt,
            config=self.config,
            **kwargs,
        )

    def test_default_scores_match_the_wrappers(self):
        results = self.evaluate()
        self.assertAlmostEqual(
            results["bleu"], compute_bleu(self.predictions, self.tgt).score
        )
        self.assertAlmostEqual(
            results["chrf"], compute_chrf(self.predictions, self.tgt).score
        )
        self.assertNotIn("ter", results)

    def test_ter_matches_the_wrapper_when_asked_for(self):
        results = self.evaluate(compute_ter_score=True)
        self.assertAlmostEqual(
            results["ter"], compute_ter(self.predictions, self.tgt).score
        )

    def test_save_translations_writes_pairs_and_metrics(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "out" / "translations.txt"
            with mock.patch("builtins.print"):
                save_translations(self.predictions, self.tgt, output_path=path)
            text = path.read_text(encoding="utf-8")
        self.assertIn(f"Prediction 1: {self.predictions[0]}", text)
        self.assertIn(f"Reference 1:  {self.tgt[0]}", text)
        self.assertIn("BLEU:", text)
        self.assertIn("chrF:", text)


if __name__ == "__main__":
    unittest.main()
