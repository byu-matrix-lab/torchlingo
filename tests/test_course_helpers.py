"""What notebooks share: reading line files, opening the baseline, and the COMET install.

read_lines and write_lines replace four spellings of one read, one of which split sentences at
U+2028. load_baseline replaces thirty lines A13 and A14 each repeated, and the question that
matters is whether the configuration really is A9's. The install test exists because a COMET fix
on Nov 5, 2026 reached one notebook of four.
"""

import json
import re
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import pandas as pd
import torch

from torchlingo import colab
from torchlingo.checkpoint import CHECKPOINT_FORMAT
from torchlingo.colab import BASELINE_SHAPE, load_baseline
from torchlingo.preprocessing import (
    parallel_txt_to_dataframe,
    read_lines,
    write_lines,
    written_without_spaces,
)

COURSE = Path(__file__).resolve().parents[1] / "docs" / "docs" / "course"


class TestReadLines(unittest.TestCase):
    def setUp(self):
        folder = tempfile.TemporaryDirectory()
        self.addCleanup(folder.cleanup)
        self.path = Path(folder.name, "lines.txt")

    def test_a_byte_order_mark_does_not_join_the_first_sentence(self):
        self.path.write_bytes("﻿hello\nworld\n".encode())
        self.assertEqual(read_lines(self.path), ["hello", "world"])

    def test_it_splits_on_the_newline_alone(self):
        self.path.write_text("one two\nthree\u0085four\n", encoding="utf-8")
        self.assertEqual(read_lines(self.path), ["one two", "three\u0085four"])

    def test_windows_line_endings(self):
        self.path.write_bytes(b"one\r\ntwo\r\n")
        self.assertEqual(read_lines(self.path), ["one", "two"])

    def test_a_blank_last_translation_is_still_a_line(self):
        self.path.write_text("one\n\n", encoding="utf-8")
        self.assertEqual(read_lines(self.path), ["one", ""])

    def test_no_final_newline(self):
        self.path.write_text("one\ntwo", encoding="utf-8")
        self.assertEqual(read_lines(self.path), ["one", "two"])

    def test_an_empty_file_has_no_lines(self):
        self.path.write_text("", encoding="utf-8")
        self.assertEqual(read_lines(self.path), [])

    def test_n_keeps_the_first_lines(self):
        self.path.write_text("a\nb\nc\n", encoding="utf-8")
        self.assertEqual(read_lines(self.path, 2), ["a", "b"])

    def test_writing_then_reading_gives_the_same_lines(self):
        lines = ["señal", "", "one two", "日本語"]
        write_lines(self.path.parent / "new" / "out.txt", lines)
        self.assertEqual(read_lines(self.path.parent / "new" / "out.txt"), lines)

    def test_write_takes_a_pandas_column(self):
        write_lines(self.path, pd.Series(["a", "b"]))
        self.assertEqual(read_lines(self.path), ["a", "b"])

    def test_parallel_files_lose_the_byte_order_mark_too(self):
        source = self.path.with_suffix(".src")
        source.write_bytes("﻿hello\n".encode())
        target = self.path.with_suffix(".tgt")
        target.write_text("hola\n", encoding="utf-8")
        frame = parallel_txt_to_dataframe(source, target)
        self.assertEqual(frame.src.iloc[0], "hello")


class TestWrittenWithoutSpaces(unittest.TestCase):
    def test_the_line_is_three_words_a_sentence(self):
        self.assertTrue(written_without_spaces(["one two", "three four"]))
        self.assertFalse(written_without_spaces(["one two three", "four five six"]))

    def test_it_takes_a_pandas_column(self):
        self.assertTrue(written_without_spaces(pd.Series(["今天天气很好"])))


class TestLoadBaseline(unittest.TestCase):
    """The real SentencePiece model and checkpoint machinery, on a small saved model."""

    def setUp(self):
        folder = tempfile.TemporaryDirectory()
        self.addCleanup(folder.cleanup)
        self.drive = Path(folder.name)
        self.dir = self.drive / "CS479" / "assignment8"
        (self.dir / "best-a9").mkdir(parents=True)
        for name in ("train", "val", "test"):
            (self.dir / f"{name}.tsv").write_text(
                "src\ttgt\nhello\tNA\n", encoding="utf-8"
            )
        (self.dir / "a9-settings.json").write_text(
            json.dumps({"no_spaces": True, "max_decode_length": 77}), encoding="utf-8"
        )
        (self.dir / "spm.model").write_text("", encoding="utf-8")
        self.env = SimpleNamespace(drive_dir=self.drive, device="cpu")
        patched = mock.patch(
            "torchlingo.data_processing.vocab.SentencePieceVocab", return_value="vocab"
        )
        patched.start()
        self.addCleanup(patched.stop)

    def save(self, model_config=None):
        saved = {"model_state_dict": {}}
        if model_config:
            saved.update(format=CHECKPOINT_FORMAT, model_config=model_config)
        torch.save(saved, self.dir / "best-a9" / "model_best.pt")

    def test_the_shape_comes_from_the_checkpoint(self):
        self.save({"d_model": 32, "n_heads": 2, "num_encoder_layers": 1,
                   "num_decoder_layers": 1, "d_ff": 64})  # fmt: skip
        config = load_baseline(self.env).config
        self.assertEqual((config.d_model, config.n_heads, config.d_ff), (32, 2, 64))

    def test_a_file_too_old_to_say_gets_a9s_shape(self):
        self.save()
        config = load_baseline(self.env).config
        for key, value in BASELINE_SHAPE.items():
            self.assertEqual(getattr(config, key), value, key)

    def test_it_reads_what_a9_left(self):
        self.save()
        baseline = load_baseline(self.env)
        self.assertEqual(baseline.dir, self.dir)
        self.assertEqual(baseline.config.max_decode_length, 77)
        self.assertEqual(baseline.config.num_workers, 0)
        self.assertTrue(baseline.no_spaces)
        self.assertEqual(baseline.tokenize, "flores200")
        self.assertEqual(baseline.bleu_name, "spBLEU (flores200)")
        self.assertEqual(baseline.spm_path, str(self.dir / "spm.model"))

    def test_the_split_is_text(self):
        self.save()
        baseline = load_baseline(self.env)
        self.assertEqual(baseline.test.tgt.iloc[0], "NA")

    def test_a_missing_file_stops_naming_it(self):
        self.save()
        with self.assertRaises(SystemExit) as stopped:
            load_baseline(self.env, needs=["test-a9.hyp"])
        self.assertIn("test-a9.hyp", str(stopped.exception))
        self.assertIn("finish A9 first", str(stopped.exception))

    def test_every_required_file_is_listed(self):
        self.assertIn("best-a9/model_best.pt", colab.BASELINE_FILES)


class TestCometInstalls(unittest.TestCase):
    def test_every_notebook_installs_the_same_comet_requirements(self):
        # The next COMET break should be fixed once. Each line is pinned to Colab's Python.
        lines = {}
        for path in COURSE.glob("*.ipynb"):
            for cell in json.loads(path.read_text(encoding="utf-8"))["cells"]:
                for line in "".join(cell["source"]).splitlines():
                    if "pip install" in line and "jsonargparse" in line:
                        lines[path.stem] = re.sub(r"^\s*", "", line)
        self.assertEqual(
            sorted(lines),
            [
                "lecture-10-comet-install",
                "lecture-11-comet-on-a6",
                "lecture-13-back-translation",
                "lecture-14-multilingual",
            ],
        )
        self.assertEqual(len(set(lines.values())), 1, lines)


if __name__ == "__main__":
    unittest.main()
