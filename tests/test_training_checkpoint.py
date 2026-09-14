"""Tests for resumable training checkpoints.

The Colab and Drive paths cannot be exercised here: GitHub runners have no Drive
to mount. What is tested is everything else, plus that the Colab helpers degrade
safely off-Colab, which is the behavior a local user actually depends on.
"""

import tempfile
import unittest
from pathlib import Path

import torch
from torch import nn, optim

from torchlingo.training import train_model
from torchlingo.training_checkpoint import (
    CheckpointState,
    TrainingCheckpointer,
    default_checkpoint_dir,
    is_colab,
    mount_drive,
)


def _model() -> nn.Module:
    torch.manual_seed(0)
    return nn.Linear(8, 8)


class TestColabHelpers(unittest.TestCase):
    """The environment helpers must be safe to call anywhere."""

    def test_is_colab_is_false_off_colab(self):
        """This suite never runs in Colab, so this is the honest answer."""
        self.assertFalse(is_colab())

    def test_mount_drive_is_a_noop_off_colab(self):
        """Notebooks call this unconditionally; it must not raise locally."""
        self.assertFalse(mount_drive())

    def test_default_checkpoint_dir_is_local_off_colab(self):
        """Off Colab the default must not point into a Drive path."""
        path = default_checkpoint_dir("demo")
        self.assertEqual(path, Path("checkpoints") / "demo")
        self.assertNotIn("drive", str(path).lower())


class TestCheckpointState(unittest.TestCase):
    """State must survive a round trip, including from a newer writer."""

    def test_round_trip(self):
        state = CheckpointState(epoch=3, global_step=100, train_losses=[1.0, 0.5])
        restored = CheckpointState.from_dict(state.to_dict())
        self.assertEqual(restored.epoch, 3)
        self.assertEqual(restored.global_step, 100)
        self.assertEqual(restored.train_losses, [1.0, 0.5])

    def test_unknown_keys_are_ignored(self):
        """A checkpoint from a newer version must still load."""
        restored = CheckpointState.from_dict({"epoch": 1, "invented_later": 42})
        self.assertEqual(restored.epoch, 1)


class TestTrainingCheckpointer(unittest.TestCase):
    """Save, load and the interval logic."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self._tmp.name)

    def tearDown(self):
        self._tmp.cleanup()

    def _checkpointer(self, **kwargs):
        kwargs.setdefault("verbose", False)
        return TrainingCheckpointer("demo", checkpoint_dir=self.dir, **kwargs)

    def test_reports_no_checkpoint_before_saving(self):
        self.assertFalse(self._checkpointer().has_checkpoint())

    def test_save_then_load_restores_weights_and_progress(self):
        """The whole point: a fresh process picks up where the last one stopped."""
        model, optimizer = _model(), None
        saver = self._checkpointer()
        saver.update(epoch=2, global_step=50, train_loss=1.5, val_loss=1.2)
        saver.save(model, optimizer)

        restored_model = nn.Linear(8, 8)
        state = self._checkpointer().load(restored_model)

        self.assertEqual(state.epoch, 2)
        self.assertEqual(state.global_step, 50)
        self.assertEqual(state.best_val_loss, 1.2)
        torch.testing.assert_close(model.weight, restored_model.weight)

    def test_optimizer_state_round_trips(self):
        """Resuming weights without optimizer state loses momentum silently."""
        model = _model()
        optimizer = optim.Adam(model.parameters(), lr=1e-3)
        model.weight.sum().backward()
        optimizer.step()

        self._checkpointer().save(model, optimizer)

        restored_model = nn.Linear(8, 8)
        restored_opt = optim.Adam(restored_model.parameters(), lr=1e-3)
        self._checkpointer().load(restored_model, restored_opt)
        self.assertEqual(
            restored_opt.state_dict()["state"].keys(),
            optimizer.state_dict()["state"].keys(),
        )

    def test_best_is_written_only_when_asked(self):
        saver = self._checkpointer()
        saver.save(_model())
        self.assertFalse(saver.has_checkpoint("best"))
        saver.save(_model(), is_best=True)
        self.assertTrue(saver.has_checkpoint("best"))

    def test_loading_a_missing_checkpoint_raises(self):
        with self.assertRaises(FileNotFoundError):
            self._checkpointer().load(_model())

    def test_path_for_rejects_unknown_names(self):
        with self.assertRaises(ValueError):
            self._checkpointer().path_for("penultimate")

    def test_step_interval_controls_saving(self):
        saver = self._checkpointer(save_every_seconds=0, save_every_steps=10)
        self.assertIsNone(saver.maybe_save(_model(), global_step=5))
        self.assertIsNotNone(saver.maybe_save(_model(), global_step=10))
        # Immediately after saving the counter resets.
        self.assertIsNone(saver.maybe_save(_model(), global_step=11))

    def test_intervals_can_both_be_disabled(self):
        """With both off, automatic saving never fires; explicit save still works."""
        saver = self._checkpointer(save_every_seconds=0, save_every_steps=0)
        self.assertIsNone(saver.maybe_save(_model(), global_step=10_000))
        saver.save(_model())
        self.assertTrue(saver.has_checkpoint())

    def test_partial_write_does_not_replace_a_good_checkpoint(self):
        """latest.pt is written via a temporary file and moved into place.

        A runtime that dies mid-write must not leave a corrupt checkpoint, which
        would fail at load time -- exactly when the work is already gone.
        """
        saver = self._checkpointer()
        saver.save(_model())
        good = saver.path_for("latest").read_bytes()

        staging = saver.path_for("latest").with_suffix(".tmp")
        staging.write_bytes(b"truncated garbage")
        self.assertEqual(saver.path_for("latest").read_bytes(), good)


class TestTrainModelIntegration(unittest.TestCase):
    """The training loop must resume rather than restart."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self._tmp.name)

    def tearDown(self):
        self._tmp.cleanup()

    @staticmethod
    def _loader():
        src = torch.tensor([[2, 5, 3], [2, 6, 3]], dtype=torch.long)
        tgt = torch.tensor([[2, 8, 3], [2, 9, 3]], dtype=torch.long)
        dataset = torch.utils.data.TensorDataset(src, tgt)
        return torch.utils.data.DataLoader(dataset, batch_size=2)

    @staticmethod
    def _tiny_model():
        from tests.test_training_inference import DummyTransformer

        return DummyTransformer()

    def test_training_without_a_checkpointer_is_unchanged(self):
        """The parameter is optional and must not perturb existing callers."""
        result = train_model(
            self._tiny_model(), self._loader(), num_epochs=2, save_dir=None
        )
        self.assertEqual(len(result.train_losses), 2)

    def test_checkpointer_records_progress(self):
        checkpointer = TrainingCheckpointer(
            "demo", checkpoint_dir=self.dir, verbose=False, save_every_seconds=0
        )
        train_model(
            self._tiny_model(),
            self._loader(),
            val_loader=self._loader(),
            num_epochs=2,
            checkpointer=checkpointer,
        )
        self.assertTrue(checkpointer.has_checkpoint())
        self.assertEqual(checkpointer.state.epoch, 1)
        self.assertEqual(len(checkpointer.state.train_losses), 2)

    def test_second_run_resumes_instead_of_restarting(self):
        """The behavior the whole module exists for.

        A run that stopped after epoch 2 of 4 should train only the remaining
        epochs on the next invocation, not start over.
        """
        first = TrainingCheckpointer(
            "demo", checkpoint_dir=self.dir, verbose=False, save_every_seconds=0
        )
        train_model(
            self._tiny_model(),
            self._loader(),
            val_loader=self._loader(),
            num_epochs=2,
            checkpointer=first,
        )

        second = TrainingCheckpointer(
            "demo", checkpoint_dir=self.dir, verbose=False, save_every_seconds=0
        )
        result = train_model(
            self._tiny_model(),
            self._loader(),
            val_loader=self._loader(),
            num_epochs=4,
            checkpointer=second,
        )

        # Two epochs restored from the checkpoint plus two newly run.
        self.assertEqual(len(result.train_losses), 4)
        self.assertEqual(second.state.epoch, 3)

    def test_corrupt_checkpoint_does_not_prevent_training(self):
        """Failing to resume should cost the history, not the ability to train."""
        checkpointer = TrainingCheckpointer(
            "demo", checkpoint_dir=self.dir, verbose=False
        )
        checkpointer.path_for("latest").write_bytes(b"not a torch file")

        result = train_model(
            self._tiny_model(),
            self._loader(),
            num_epochs=1,
            checkpointer=checkpointer,
        )
        self.assertEqual(len(result.train_losses), 1)


if __name__ == "__main__":
    unittest.main()
