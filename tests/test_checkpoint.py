"""Unit tests for torchlingo.checkpoint module.

Tests the LocalCheckpointer and ColabCheckpointer classes, as well as related
utilities for auto-checkpointing during training.
"""

import shutil
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
from torch import nn

from torchlingo.checkpoint import (
    CheckpointState,
    LocalCheckpointer,
    create_checkpointer_from_config,
    get_drive_free_space,
    is_colab,
    is_drive_mounted,
    save_checkpoint_locally,
    save_checkpoint_to_drive,
)


class SimpleModel(nn.Module):
    """Simple model for testing checkpoint save/load."""

    def __init__(self, input_size: int = 10, hidden_size: int = 20):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, input_size)

    def forward(self, x):
        return self.fc2(torch.relu(self.fc1(x)))


class TestEnvironmentDetection(unittest.TestCase):
    """Tests for environment detection utilities."""

    def test_is_colab_returns_bool(self):
        """is_colab should return a boolean."""
        result = is_colab()
        self.assertIsInstance(result, bool)

    def test_is_colab_false_locally(self):
        """is_colab should return False when not in Colab."""
        # When running locally, should be False
        self.assertFalse(is_colab())

    def test_is_drive_mounted_returns_bool(self):
        """is_drive_mounted should return a boolean."""
        result = is_drive_mounted()
        self.assertIsInstance(result, bool)

    def test_is_drive_mounted_false_locally(self):
        """is_drive_mounted should return False when not in Colab."""
        # /content/drive shouldn't exist locally
        self.assertFalse(is_drive_mounted())

    def test_get_drive_free_space_returns_int_or_none(self):
        """get_drive_free_space should return int or None."""
        result = get_drive_free_space(Path("."))
        self.assertTrue(result is None or isinstance(result, int))

    def test_get_drive_free_space_for_valid_path(self):
        """get_drive_free_space should return positive int for valid path."""
        result = get_drive_free_space(Path("."))
        # Should be able to get space for current directory
        self.assertIsNotNone(result)
        self.assertGreater(result, 0)


class TestCheckpointState(unittest.TestCase):
    """Tests for CheckpointState dataclass."""

    def test_default_values(self):
        """CheckpointState should have sensible defaults."""
        state = CheckpointState()
        self.assertEqual(state.epoch, 0)
        self.assertEqual(state.global_step, 0)
        self.assertEqual(state.best_val_loss, float("inf"))
        self.assertEqual(state.train_losses, [])
        self.assertEqual(state.val_losses, [])
        self.assertEqual(state.metrics, {})

    def test_to_dict(self):
        """to_dict should return all state as a dictionary."""
        state = CheckpointState(
            epoch=5,
            global_step=1000,
            best_val_loss=0.5,
            train_losses=[1.0, 0.8, 0.6],
            val_losses=[0.9, 0.7, 0.5],
            metrics={"accuracy": 0.95},
            experiment_name="test_exp",
        )
        d = state.to_dict()

        self.assertEqual(d["epoch"], 5)
        self.assertEqual(d["global_step"], 1000)
        self.assertEqual(d["best_val_loss"], 0.5)
        self.assertEqual(d["train_losses"], [1.0, 0.8, 0.6])
        self.assertEqual(d["val_losses"], [0.9, 0.7, 0.5])
        self.assertEqual(d["metrics"], {"accuracy": 0.95})
        self.assertEqual(d["experiment_name"], "test_exp")

    def test_from_dict(self):
        """from_dict should reconstruct state from dictionary."""
        data = {
            "epoch": 10,
            "global_step": 5000,
            "best_val_loss": 0.3,
            "train_losses": [0.5, 0.4, 0.3],
            "val_losses": [0.6, 0.5, 0.4],
            "metrics": {"bleu": 25.0},
            "experiment_name": "restored_exp",
        }
        state = CheckpointState.from_dict(data)

        self.assertEqual(state.epoch, 10)
        self.assertEqual(state.global_step, 5000)
        self.assertEqual(state.best_val_loss, 0.3)
        self.assertEqual(state.train_losses, [0.5, 0.4, 0.3])
        self.assertEqual(state.experiment_name, "restored_exp")

    def test_from_dict_missing_keys(self):
        """from_dict should handle missing keys with defaults."""
        data = {"epoch": 3}
        state = CheckpointState.from_dict(data)

        self.assertEqual(state.epoch, 3)
        self.assertEqual(state.global_step, 0)
        self.assertEqual(state.best_val_loss, float("inf"))
        self.assertEqual(state.train_losses, [])


class TestLocalCheckpointer(unittest.TestCase):
    """Tests for LocalCheckpointer class."""

    def setUp(self):
        """Create temporary directory for checkpoints."""
        self.temp_dir = tempfile.mkdtemp()
        self.checkpoint_dir = Path(self.temp_dir) / "checkpoints" / "test_exp"
        self.model = SimpleModel()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)

    def tearDown(self):
        """Clean up temporary directory."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_init_creates_checkpoint_dir(self):
        """LocalCheckpointer should create checkpoint directory."""
        checkpointer = LocalCheckpointer(
            experiment_name="test_exp",
            checkpoint_dir=self.checkpoint_dir,
            verbose=False,
        )

        self.assertTrue(self.checkpoint_dir.exists())
        self.assertTrue(self.checkpoint_dir.is_dir())
        self.assertEqual(checkpointer.experiment_name, "test_exp")

    def test_checkpoint_dir_property(self):
        """checkpoint_dir property should return correct path."""
        checkpointer = LocalCheckpointer(
            experiment_name="test_exp",
            checkpoint_dir=self.checkpoint_dir,
            verbose=False,
        )

        self.assertEqual(checkpointer.checkpoint_dir, self.checkpoint_dir)

    def test_save_creates_checkpoint_file(self):
        """save() should create a checkpoint file."""
        checkpointer = LocalCheckpointer(
            experiment_name="test_exp",
            checkpoint_dir=self.checkpoint_dir,
            verbose=False,
        )

        save_path = checkpointer.save(
            model=self.model,
            optimizer=self.optimizer,
            epoch=5,
            step=1000,
            metrics={"train_loss": 0.5},
        )

        self.assertTrue(save_path.exists())
        self.assertIn("checkpoint_step_00001000.pt", str(save_path))

    def test_save_creates_latest_checkpoint(self):
        """save() should also create checkpoint_latest.pt."""
        checkpointer = LocalCheckpointer(
            experiment_name="test_exp",
            checkpoint_dir=self.checkpoint_dir,
            verbose=False,
        )

        checkpointer.save(self.model, self.optimizer, epoch=1, step=100)

        latest_path = self.checkpoint_dir / "checkpoint_latest.pt"
        self.assertTrue(latest_path.exists())

    def test_save_with_is_best_creates_best_checkpoint(self):
        """save() with is_best=True should create checkpoint_best.pt."""
        checkpointer = LocalCheckpointer(
            experiment_name="test_exp",
            checkpoint_dir=self.checkpoint_dir,
            verbose=False,
        )

        checkpointer.save(
            self.model,
            self.optimizer,
            epoch=1,
            step=100,
            is_best=True,
        )

        best_path = self.checkpoint_dir / "checkpoint_best.pt"
        self.assertTrue(best_path.exists())

    def test_load_restores_model_state(self):
        """load() should restore model state correctly."""
        checkpointer = LocalCheckpointer(
            experiment_name="test_exp",
            checkpoint_dir=self.checkpoint_dir,
            verbose=False,
        )

        # Save checkpoint
        checkpointer.save(self.model, self.optimizer, epoch=5, step=1000)

        # Modify model weights
        with torch.no_grad():
            for param in self.model.parameters():
                param.fill_(999.0)

        # Load checkpoint
        state = checkpointer.load(self.model, self.optimizer)

        # Weights should be restored (not 999)
        for param in self.model.parameters():
            self.assertFalse(torch.all(param == 999.0))

        self.assertEqual(state.epoch, 5)
        self.assertEqual(state.global_step, 1000)

    def test_has_checkpoint_returns_false_when_no_checkpoint(self):
        """has_checkpoint() should return False when no checkpoint exists."""
        checkpointer = LocalCheckpointer(
            experiment_name="test_exp",
            checkpoint_dir=self.checkpoint_dir,
            verbose=False,
        )

        self.assertFalse(checkpointer.has_checkpoint())

    def test_has_checkpoint_returns_true_after_save(self):
        """has_checkpoint() should return True after saving."""
        checkpointer = LocalCheckpointer(
            experiment_name="test_exp",
            checkpoint_dir=self.checkpoint_dir,
            verbose=False,
        )

        checkpointer.save(self.model, epoch=1, step=100)

        self.assertTrue(checkpointer.has_checkpoint())

    def test_should_save_time_based(self):
        """should_save() should return True after time interval."""
        checkpointer = LocalCheckpointer(
            experiment_name="test_exp",
            checkpoint_dir=self.checkpoint_dir,
            save_interval_minutes=0.001,  # Very short for testing
            save_interval_steps=0,
            verbose=False,
        )

        # Initially should not save
        self.assertFalse(checkpointer.should_save(0))

        # Wait a bit
        time.sleep(0.1)

        # Now should save
        self.assertTrue(checkpointer.should_save(0))

    def test_should_save_step_based(self):
        """should_save() should return True after step interval."""
        checkpointer = LocalCheckpointer(
            experiment_name="test_exp",
            checkpoint_dir=self.checkpoint_dir,
            save_interval_minutes=0,
            save_interval_steps=100,
            verbose=False,
        )

        # Initially should not save
        self.assertFalse(checkpointer.should_save(50))

        # After interval steps should save
        self.assertTrue(checkpointer.should_save(100))
        self.assertTrue(checkpointer.should_save(200))

    def test_cleanup_old_checkpoints(self):
        """Old step checkpoints should be automatically cleaned up."""
        checkpointer = LocalCheckpointer(
            experiment_name="test_exp",
            checkpoint_dir=self.checkpoint_dir,
            keep_n_checkpoints=2,
            verbose=False,
        )

        # Save multiple checkpoints
        for i in range(5):
            checkpointer.save(self.model, epoch=i, step=i * 100)
            time.sleep(0.01)  # Ensure different timestamps

        # Should only keep 2 step checkpoints
        step_checkpoints = list(self.checkpoint_dir.glob("checkpoint_step_*.pt"))
        self.assertEqual(len(step_checkpoints), 2)

    def test_step_callback_saves_when_needed(self):
        """step_callback() should save when should_save() returns True."""
        checkpointer = LocalCheckpointer(
            experiment_name="test_exp",
            checkpoint_dir=self.checkpoint_dir,
            save_interval_minutes=0,
            save_interval_steps=10,
            verbose=False,
        )

        # First few steps should not save
        for step in range(9):
            result = checkpointer.step_callback(self.model, step=step)
            self.assertIsNone(result)

        # Step 10 should save
        result = checkpointer.step_callback(self.model, step=10)
        self.assertIsNotNone(result)
        self.assertTrue(result.exists())

    def test_update_state_appends_losses(self):
        """update_state() should append losses to state."""
        checkpointer = LocalCheckpointer(
            experiment_name="test_exp",
            checkpoint_dir=self.checkpoint_dir,
            verbose=False,
        )

        checkpointer.update_state(train_loss=1.0)
        checkpointer.update_state(train_loss=0.8, val_loss=0.9)
        checkpointer.update_state(train_loss=0.6, val_loss=0.7)

        state = checkpointer.get_state()
        self.assertEqual(state.train_losses, [1.0, 0.8, 0.6])
        self.assertEqual(state.val_losses, [0.9, 0.7])
        self.assertEqual(state.best_val_loss, 0.7)

    def test_load_raises_on_missing_checkpoint(self):
        """load() should raise FileNotFoundError if no checkpoint exists."""
        checkpointer = LocalCheckpointer(
            experiment_name="nonexistent_exp",
            checkpoint_dir=Path(self.temp_dir) / "nonexistent",
            verbose=False,
        )

        with self.assertRaises(FileNotFoundError):
            checkpointer.load(self.model)

    def test_default_checkpoint_dir(self):
        """Default checkpoint_dir should be ./checkpoints/<experiment>."""
        checkpointer = LocalCheckpointer(
            experiment_name="my_experiment",
            verbose=False,
        )

        expected_dir = Path("./checkpoints/my_experiment")
        self.assertEqual(checkpointer.checkpoint_dir, expected_dir)

        # Cleanup
        if expected_dir.exists():
            shutil.rmtree(expected_dir.parent, ignore_errors=True)


class TestColabCheckpointer(unittest.TestCase):
    """Tests for ColabCheckpointer class."""

    def test_colab_checkpointer_raises_outside_colab(self):
        """ColabCheckpointer should raise RuntimeError outside of Colab."""
        from torchlingo.checkpoint import ColabCheckpointer

        with self.assertRaises(RuntimeError) as ctx:
            ColabCheckpointer(experiment_name="test")

        self.assertIn("Colab", str(ctx.exception))

    @patch("torchlingo.checkpoint.is_colab", return_value=True)
    @patch("torchlingo.checkpoint.mount_drive")
    @patch("torchlingo.checkpoint.is_drive_mounted", return_value=False)
    @patch("pathlib.Path.mkdir")
    @patch("torchlingo.checkpoint.check_drive_space", return_value=True)
    def test_colab_checkpointer_warns_when_drive_not_mounted(
        self, mock_space, mock_mkdir, mock_mounted, mock_mount, mock_colab
    ):
        """ColabCheckpointer should warn when Drive is not mounted."""
        from torchlingo.checkpoint import ColabCheckpointer

        with self.assertWarns(UserWarning):
            checkpointer = ColabCheckpointer(
                experiment_name="test_exp",
                verbose=True,
            )

        # Should fall back to local directory
        self.assertIn("content/checkpoints", str(checkpointer.checkpoint_dir))


class TestConvenienceFunctions(unittest.TestCase):
    """Tests for convenience functions."""

    def setUp(self):
        """Create temporary directory."""
        self.temp_dir = tempfile.mkdtemp()
        self.model = SimpleModel()

    def tearDown(self):
        """Clean up."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
        # Clean up any created checkpoint dirs
        default_dir = Path("./checkpoints")
        if default_dir.exists():
            shutil.rmtree(default_dir, ignore_errors=True)

    def test_save_checkpoint_locally(self):
        """save_checkpoint_locally should create a checkpoint file."""
        path = save_checkpoint_locally(
            model=self.model,
            experiment_name="test_local_save",
            epoch=5,
            step=100,
        )

        self.assertTrue(path.exists())

    def test_save_checkpoint_to_drive_locally(self):
        """save_checkpoint_to_drive should work locally with warning."""
        with self.assertWarns(UserWarning):
            # Should warn when not in Colab
            path = save_checkpoint_to_drive(
                model=self.model,
                experiment_name="test_drive_save",
                epoch=1,
            )

        # Should still save (falls back to local)
        self.assertIsNotNone(path)


class TestCreateCheckpointerFromConfig(unittest.TestCase):
    """Tests for create_checkpointer_from_config factory function."""

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)
        # Clean up any created checkpoint dirs
        default_dir = Path("./checkpoints")
        if default_dir.exists():
            shutil.rmtree(default_dir, ignore_errors=True)

    def test_returns_none_when_disabled(self):
        """Should return None when use_colab_checkpointing is False."""
        from torchlingo.config import Config

        cfg = Config(use_colab_checkpointing=False)
        result = create_checkpointer_from_config(cfg)
        self.assertIsNone(result)

    def test_returns_local_checkpointer_when_not_in_colab(self):
        """Should return LocalCheckpointer when enabled but not in Colab."""
        from torchlingo.config import Config

        cfg = Config(
            use_colab_checkpointing=True,
            experiment_name="test_config_exp",
            colab_checkpoint_interval_minutes=5.0,
            colab_keep_n_checkpoints=5,
        )
        result = create_checkpointer_from_config(cfg)

        self.assertIsInstance(result, LocalCheckpointer)
        self.assertEqual(result.experiment_name, "test_config_exp")
        self.assertEqual(result.save_interval_minutes, 5.0)
        self.assertEqual(result.keep_n_checkpoints, 5)


class TestIntegrationWithScheduler(unittest.TestCase):
    """Test checkpoint save/load with LR scheduler."""

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.checkpoint_dir = Path(self.temp_dir) / "checkpoints"
        self.model = SimpleModel()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.01)
        self.scheduler = torch.optim.lr_scheduler.StepLR(
            self.optimizer, step_size=10, gamma=0.1
        )

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_save_and_load_scheduler_state(self):
        """Scheduler state should be saved and restored correctly."""
        checkpointer = LocalCheckpointer(
            experiment_name="test_scheduler",
            checkpoint_dir=self.checkpoint_dir,
            verbose=False,
        )

        # Step scheduler a few times
        for _ in range(15):
            self.scheduler.step()

        original_lr = self.optimizer.param_groups[0]["lr"]

        # Save checkpoint
        checkpointer.save(
            self.model,
            self.optimizer,
            self.scheduler,
            epoch=1,
            step=100,
        )

        # Create new scheduler and load
        new_scheduler = torch.optim.lr_scheduler.StepLR(
            self.optimizer, step_size=10, gamma=0.1
        )

        checkpointer.load(self.model, self.optimizer, new_scheduler)

        # LR should be restored
        restored_lr = self.optimizer.param_groups[0]["lr"]
        self.assertAlmostEqual(original_lr, restored_lr, places=10)


if __name__ == "__main__":
    unittest.main()
