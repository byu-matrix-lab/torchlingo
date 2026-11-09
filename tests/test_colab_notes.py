"""Notes, StageTimes and save_rerun: what A12 to A14 save to Drive against a dropped session."""

import importlib.metadata
import json
import subprocess
import sys
import tempfile
import textwrap
import time
import unittest
from pathlib import Path
from unittest import mock

import torch

import torchlingo
from torchlingo.colab import Notes, StageTimes, save_rerun


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


class TestStageTimes(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.path = Path(self.tmp.name) / "a14-times.json"

    def tearDown(self):
        self.tmp.cleanup()

    def test_a_finished_stage_has_its_time_one_run_and_the_device(self):
        times = StageTimes(self.path)
        with times.stage("Step 3"):
            time.sleep(0.15)
        stage = StageTimes(self.path).load()["Step 3"]  # as in a new session
        self.assertGreaterEqual(stage["seconds"], 0.1)
        self.assertEqual((stage["runs"], stage["gpus"]), (1, ["cpu"]))

    def test_a_stage_run_again_adds_to_its_time_and_counts_the_run(self):
        StageTimes(self.path).notes.save(
            **{"Step 4": {"seconds": 100.0, "runs": 1, "gpus": ["A100"]}}
        )
        with StageTimes(self.path).stage("Step 4"):
            pass
        stage = StageTimes(self.path).load()["Step 4"]
        self.assertGreaterEqual(stage["seconds"], 100.0)
        self.assertLess(stage["seconds"], 101.0)
        self.assertEqual((stage["runs"], stage["gpus"]), (2, ["A100", "cpu"]))

    def test_a_stage_that_stops_with_an_error_keeps_its_time_and_the_error(self):
        times = StageTimes(self.path)
        # The student stopped the cell.
        with self.assertRaises(KeyboardInterrupt), times.stage("Step 4"):
            time.sleep(0.15)
            raise KeyboardInterrupt
        self.assertGreaterEqual(times.load()["Step 4"]["seconds"], 0.1)

    def test_a_session_that_dies_inside_a_stage_keeps_the_time_so_far(self):
        # A real drop: the process ends with no chance to run the end of the stage.
        script = textwrap.dedent(f"""
            import os, time
            from torchlingo.colab import StageTimes
            with StageTimes({str(self.path)!r}, every=0.05).stage("Step 4"):
                time.sleep(0.5)
                os._exit(1)
        """)
        subprocess.run([sys.executable, "-c", script], check=False, timeout=120)
        stage = StageTimes(self.path).load()["Step 4"]
        self.assertGreater(stage["seconds"], 0.2)
        self.assertEqual(stage["runs"], 1)

    def test_a_failed_save_does_not_stop_the_stage(self):
        times = StageTimes(self.path, every=0.01)
        finished = False
        with (
            mock.patch("torchlingo.colab._write_json", side_effect=OSError("Drive")),
            times.stage("Step 5"),
        ):
            time.sleep(0.05)
            finished = True
        self.assertTrue(finished)
        with times.stage("Step 5"):  # and the next save catches up
            pass
        self.assertEqual(times.load()["Step 5"]["runs"], 1)

    def test_the_table_lines_up_totals_and_names_each_gpu(self):
        StageTimes(self.path).notes.save(
            **{
                "Step 3: NLLB off the shelf": {
                    "seconds": 59.6,
                    "runs": 1,
                    "gpus": ["A100"],
                },
                "Step 4: fine-tuning": {
                    "seconds": 7325.0,
                    "runs": 3,
                    "gpus": ["A100", "L4"],
                },
            }
        )
        self.assertEqual(
            StageTimes(self.path).table().splitlines(),
            [
                "Stage                       Time     Runs  GPU",
                "Step 3: NLLB off the shelf  0:01:00  1     A100",
                "Step 4: fine-tuning         2:02:05  3     A100 + L4",
                "Total                       2:03:05",
            ],
        )

    def test_the_table_before_any_stage_says_so(self):
        self.assertIn("No stage", StageTimes(self.path).table())


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
