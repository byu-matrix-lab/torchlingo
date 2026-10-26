"""Notes and save_rerun: what A12 to A14 save to Drive so a dropped session loses nothing."""

import importlib.metadata
import json
import tempfile
import unittest
from pathlib import Path

import torch

import torchlingo
from torchlingo.colab import Notes, save_rerun


class TestNotes(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.path = Path(self.tmp.name) / "a14-notes.json"

    def tearDown(self):
        self.tmp.cleanup()

    def test_nothing_saved_yet_is_empty(self):
        self.assertEqual(Notes(self.path).load(), {})

    def test_each_save_adds_and_a_new_session_reads_them_all(self):
        notes = Notes(self.path)
        notes.save(prediction="better", language="Efik")
        returned = notes.save(scores={"A14": {"bleu": 30.1}}, language="Ibibio")
        back = Notes(self.path).load()  # as after a dropped session
        self.assertEqual(
            back,
            {
                "prediction": "better",
                "language": "Ibibio",
                "scores": {"A14": {"bleu": 30.1}},
            },
        )
        self.assertEqual(returned, back)

    def test_text_is_stored_as_written(self):
        Notes(self.path).save(output="我喜欢猫, Ẹ̀dịk")
        self.assertIn("我喜欢猫, Ẹ̀dịk", self.path.read_text(encoding="utf-8"))

    def test_a_folder_not_made_yet_is_made(self):
        deeper = Path(self.tmp.name) / "assignment8" / "notes.json"
        Notes(deeper).save(x=1)
        self.assertEqual(Notes(deeper).load(), {"x": 1})

    def test_a_value_json_cannot_hold_leaves_the_notes_as_they_were(self):
        notes = Notes(self.path)
        notes.save(kept=True)
        with self.assertRaises(TypeError):
            notes.save(model=torch.nn.Linear(1, 1))
        self.assertEqual(notes.load(), {"kept": True})


class TestSaveRerun(unittest.TestCase):
    def test_settings_and_versions_are_written(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "a14-rerun.json"
            record = save_rerun(path, model="nllb-200-distilled-600M", seed=479)
            back = json.loads(path.read_text(encoding="utf-8"))
        self.assertEqual(back, record)
        self.assertEqual(back["seed"], 479)
        self.assertEqual(back["versions"]["torchlingo"], torchlingo.version_label())
        self.assertEqual(back["versions"]["torch"], importlib.metadata.version("torch"))
        self.assertEqual(
            set(back["versions"]), {"torchlingo", "torch", "sacrebleu", "transformers"}
        )

    def test_a_package_not_installed_says_so(self):
        with tempfile.TemporaryDirectory() as tmp:
            record = save_rerun(Path(tmp) / "r.json", packages=["no-such-package-xyz"])
        self.assertEqual(record["versions"]["no-such-package-xyz"], "not installed")


if __name__ == "__main__":
    unittest.main()
