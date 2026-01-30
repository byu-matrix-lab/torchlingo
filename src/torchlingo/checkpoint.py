"""Checkpointing utilities with automatic persistence for training.

This module provides auto-checkpointing utilities for training runs.
LocalCheckpointer saves to local disk, while ColabCheckpointer extends it
to save to Google Drive for Colab environments where runtime disconnects
can cause loss of training progress.

Quick Start (Colab):
    from torchlingo.checkpoint import ColabCheckpointer

    # Initialize checkpointer (will mount Google Drive automatically)
    checkpointer = ColabCheckpointer(experiment_name="my_translation_model")

    # Train model - checkpoints are auto-saved every 10 minutes
    result = train_model(
        model, train_loader,
        checkpointer=checkpointer,
        num_epochs=10
    )

Quick Start (Local):
    from torchlingo.checkpoint import LocalCheckpointer

    # Initialize local checkpointer
    checkpointer = LocalCheckpointer(experiment_name="my_model")

    # Same API as ColabCheckpointer
    checkpointer.save(model, optimizer, epoch=5, step=1000)
    state = checkpointer.load(model, optimizer)

Config-based (recommended):
    from torchlingo.config import Config

    # Enable checkpointing in config
    cfg = Config(
        use_colab_checkpointing=True,  # or False for local
        experiment_name="hw3_translation"
    )

    # train_model will auto-create the appropriate checkpointer
    result = train_model(model, train_loader, config=cfg)
"""

from __future__ import annotations

import os
import shutil
import time
import warnings
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Union

import torch
from torch import nn, optim

if TYPE_CHECKING:
    from .config import Config


# ============================================================================
# CONSTANTS
# ============================================================================

# Minimum recommended free space in bytes (500 MB)
MIN_RECOMMENDED_SPACE_BYTES = 500 * 1024 * 1024

# Typical checkpoint size estimate (100 MB for a medium model)
ESTIMATED_CHECKPOINT_SIZE_BYTES = 100 * 1024 * 1024


# ============================================================================
# ENVIRONMENT DETECTION
# ============================================================================


def is_colab() -> bool:
    """Detect if running in Google Colab environment.

    Returns:
        True if running in Google Colab, False otherwise.

    Examples:
        >>> if is_colab():
        ...     print("Running in Colab!")
    """
    try:
        import google.colab  # type: ignore  # noqa: F401

        return True
    except ImportError:
        return False


def is_drive_mounted(mount_point: str = "/content/drive") -> bool:
    """Check if Google Drive is already mounted.

    Args:
        mount_point: Path where Drive should be mounted.

    Returns:
        True if Drive is mounted at the specified path.
    """
    return os.path.exists(mount_point) and os.path.ismount(mount_point)


def mount_drive(
    mount_point: str = "/content/drive", force_remount: bool = False
) -> bool:
    """Mount Google Drive in Colab environment.

    This function will prompt the user for authentication if needed.

    Args:
        mount_point: Path where Drive should be mounted.
        force_remount: If True, unmount and remount even if already mounted.

    Returns:
        True if Drive was successfully mounted, False if not in Colab.

    Raises:
        RuntimeError: If mounting fails in Colab.

    Examples:
        >>> from torchlingo.checkpoint import mount_drive
        >>> mount_drive()  # Will prompt for auth in Colab
        True
    """
    if not is_colab():
        warnings.warn(
            "mount_drive() called outside of Colab environment. "
            "Use LocalCheckpointer instead of ColabCheckpointer."
        )
        return False

    try:
        from google.colab import drive  # type: ignore

        drive.mount(mount_point, force_remount=force_remount)
        return True
    except Exception as e:
        raise RuntimeError(f"Failed to mount Google Drive: {e}") from e


def flush_drive() -> None:
    """Flush pending writes to Google Drive.

    Call this after saving important checkpoints to ensure data is synced
    to Drive before the runtime potentially disconnects.

    Examples:
        >>> checkpointer.save(model, optimizer, epoch=10)
        >>> flush_drive()  # Ensure checkpoint is synced
    """
    if not is_colab():
        return

    try:
        from google.colab import drive  # type: ignore

        drive.flush_and_unmount()
        drive.mount("/content/drive")
    except Exception:
        # Flush not critical, just continue
        pass


def get_drive_free_space(drive_path: Path) -> Optional[int]:
    """Get available free space on the drive containing the given path.

    This uses the fast OS-based method (shutil.disk_usage) which is suitable
    for repeated checks during training. For the most accurate quota info,
    use get_true_drive_free_space() instead.

    Args:
        drive_path: Path on the drive to check.

    Returns:
        Free space in bytes, or None if unable to determine.

    Note:
        This method relies on the file system driver and may sometimes lag
        behind recent changes (e.g., if you just deleted files).
    """
    try:
        import shutil

        total, used, free = shutil.disk_usage(drive_path)
        return free
    except (OSError, AttributeError):
        return None


def get_true_drive_free_space() -> Optional[float]:
    """Query Google Drive API for exact account storage quota.

    This is the "Gold Standard" method that queries Google's servers directly.
    It is the most accurate way to find free space as it accounts for storage
    shared across Google Drive, Gmail, and Google Photos.

    This method does NOT require mounting Google Drive.

    Returns:
        Free space in GB, float('inf') for unlimited storage, or None on error.

    Note:
        Use get_drive_free_space() for repeated checks during training loops
        (it's faster). Use this method at the start of a notebook to verify
        account state before heavy work.

    Examples:
        >>> free_gb = get_true_drive_free_space()
        >>> if free_gb == float('inf'):
        ...     print("Unlimited storage!")
        >>> elif free_gb is not None:
        ...     print(f"You have {free_gb:.2f} GB free")
    """
    if not is_colab():
        return None

    try:
        from google.colab import auth  # type: ignore
        from googleapiclient.discovery import build  # type: ignore

        # Authenticate (may open popup if not recently authenticated)
        auth.authenticate_user()

        # Build the Drive service
        service = build("drive", "v3")

        # Request storage quota metadata
        about = service.about().get(fields="storageQuota").execute()
        quota = about.get("storageQuota", {})

        limit = int(quota.get("limit", 0))  # Total storage limit
        usage = int(quota.get("usage", 0))  # Total storage used

        # Check for unlimited storage (common in Edu/Enterprise plans)
        if limit == 0:
            return float("inf")

        free_bytes = limit - usage
        free_gb = free_bytes / (2**30)

        return round(free_gb, 2)

    except Exception:
        return None


def check_drive_space(
    drive_path: Path,
    min_space_bytes: int = MIN_RECOMMENDED_SPACE_BYTES,
    verbose: bool = True,
    use_api: bool = False,
) -> bool:
    """Check if there's sufficient space for checkpoints.

    Args:
        drive_path: Path to the checkpoint directory.
        min_space_bytes: Minimum recommended free space in bytes.
        verbose: Whether to print warnings.
        use_api: If True, use the Google Drive API for accurate quota info.
            This is slower but more accurate. Default False uses the fast
            OS-based method suitable for repeated checks during training.

    Returns:
        True if sufficient space is available, False otherwise.
    """
    if use_api and is_colab():
        # Use the accurate API method
        free_gb = get_true_drive_free_space()
        if free_gb == float("inf"):
            return True  # Unlimited storage
        if free_gb is not None:
            free_space = int(free_gb * (2**30))  # Convert GB to bytes
        else:
            free_space = None
    else:
        # Use the fast OS-based method
        free_space = get_drive_free_space(drive_path)

    if free_space is None:
        if verbose:
            warnings.warn(
                "⚠ Could not determine available disk space. "
                "Ensure you have enough space for checkpoints."
            )
        return True  # Assume OK if we can't check

    free_space_mb = free_space / (1024 * 1024)
    min_space_mb = min_space_bytes / (1024 * 1024)

    if free_space < min_space_bytes:
        if verbose:
            warnings.warn(
                f"⚠ LOW DISK SPACE WARNING!\n"
                f"  Available: {free_space_mb:.1f} MB\n"
                f"  Recommended minimum: {min_space_mb:.1f} MB\n"
                f"  Checkpoints may fail to save. Free up space on your Drive!"
            )
        return False

    return True


# ============================================================================
# CHECKPOINT STATE
# ============================================================================


@dataclass
class CheckpointState:
    """Container for checkpoint metadata and training state.

    Attributes:
        epoch: Current epoch number (0-indexed).
        global_step: Total training steps completed.
        best_val_loss: Best validation loss seen so far.
        train_losses: List of training losses per epoch.
        val_losses: List of validation losses per epoch.
        metrics: Additional custom metrics dict.
        timestamp: ISO format timestamp when checkpoint was created.
        experiment_name: Name of the experiment.
    """

    epoch: int = 0
    global_step: int = 0
    best_val_loss: float = float("inf")
    train_losses: List[float] = field(default_factory=list)
    val_losses: List[float] = field(default_factory=list)
    metrics: Dict[str, Any] = field(default_factory=dict)
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    experiment_name: str = "torchlingo_experiment"

    def to_dict(self) -> Dict[str, Any]:
        """Convert state to dictionary for serialization."""
        return {
            "epoch": self.epoch,
            "global_step": self.global_step,
            "best_val_loss": self.best_val_loss,
            "train_losses": self.train_losses,
            "val_losses": self.val_losses,
            "metrics": self.metrics,
            "timestamp": self.timestamp,
            "experiment_name": self.experiment_name,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CheckpointState":
        """Create state from dictionary."""
        return cls(
            epoch=data.get("epoch", 0),
            global_step=data.get("global_step", 0),
            best_val_loss=data.get("best_val_loss", float("inf")),
            train_losses=data.get("train_losses", []),
            val_losses=data.get("val_losses", []),
            metrics=data.get("metrics", {}),
            timestamp=data.get("timestamp", datetime.now().isoformat()),
            experiment_name=data.get("experiment_name", "torchlingo_experiment"),
        )


# ============================================================================
# LOCAL CHECKPOINTER
# ============================================================================


class LocalCheckpointer:
    """Local checkpointing manager for saving training progress to disk.

    This class manages saving and loading checkpoints to a local directory,
    providing automatic periodic saves during training.

    Key Features:
        - Configurable auto-save interval (time or step based)
        - Full training state resumption (model, optimizer, scheduler, epoch)
        - Keeps multiple checkpoint versions with automatic cleanup
        - Base class for ColabCheckpointer

    Args:
        experiment_name: Unique name for this experiment/run.
        checkpoint_dir: Directory for checkpoints. Default: "./checkpoints/<experiment>".
        save_interval_minutes: Auto-save every N minutes during training.
            Default: 10 minutes. Set to 0 to disable time-based saves.
        save_interval_steps: Auto-save every N training steps.
            Default: 0 (disabled). Set to positive int to enable.
        keep_n_checkpoints: Number of recent checkpoints to keep. Default: 3.
        save_optimizer: Whether to save optimizer state. Default: True.
        save_scheduler: Whether to save scheduler state. Default: True.
        verbose: Print status messages. Default: True.

    Examples:
        >>> # Basic usage
        >>> checkpointer = LocalCheckpointer("my_translation_exp")

        >>> # Save a checkpoint
        >>> checkpointer.save(
        ...     model=model,
        ...     optimizer=optimizer,
        ...     epoch=5,
        ...     step=10000,
        ...     metrics={"train_loss": 0.5, "val_loss": 0.4}
        ... )

        >>> # Resume from checkpoint
        >>> state = checkpointer.load(model, optimizer)
        >>> print(f"Resuming from epoch {state.epoch}, step {state.global_step}")
    """

    def __init__(
        self,
        experiment_name: str,
        checkpoint_dir: Optional[Union[str, Path]] = None,
        save_interval_minutes: float = 10.0,
        save_interval_steps: int = 0,
        keep_n_checkpoints: int = 3,
        save_optimizer: bool = True,
        save_scheduler: bool = True,
        verbose: bool = True,
    ):
        self.experiment_name = experiment_name
        self.save_interval_minutes = save_interval_minutes
        self.save_interval_steps = save_interval_steps
        self.keep_n_checkpoints = keep_n_checkpoints
        self.save_optimizer = save_optimizer
        self.save_scheduler = save_scheduler
        self.verbose = verbose

        self._last_save_time = time.time()
        self._last_save_step = 0

        # Set up checkpoint directory
        if checkpoint_dir is None:
            checkpoint_dir = Path("./checkpoints") / experiment_name
        self._checkpoint_dir = Path(checkpoint_dir)
        self._checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Check disk space
        check_drive_space(self._checkpoint_dir, verbose=verbose)

        if self.verbose:
            print(f"✓ Checkpoints will be saved to: {self._checkpoint_dir}")

        # Track current state
        self._current_state = CheckpointState(experiment_name=experiment_name)

    def _log(self, message: str) -> None:
        """Print message if verbose mode is enabled."""
        if self.verbose:
            print(message)

    @property
    def checkpoint_dir(self) -> Path:
        """Get the checkpoint directory."""
        return self._checkpoint_dir

    def _list_checkpoints(self) -> List[Path]:
        """List all checkpoint files in checkpoint directory, sorted by time."""
        pattern = "checkpoint_step_*.pt"
        checkpoints = list(self.checkpoint_dir.glob(pattern))
        # Sort by modification time (newest first)
        checkpoints.sort(key=lambda p: p.stat().st_mtime, reverse=True)
        return checkpoints

    def _cleanup_old_checkpoints(self) -> None:
        """Remove old step checkpoints, keeping only the most recent N."""
        if self.keep_n_checkpoints <= 0:
            return

        checkpoints = self._list_checkpoints()
        # Keep best and latest, plus N step checkpoints
        to_remove = checkpoints[self.keep_n_checkpoints :]

        for ckpt in to_remove:
            try:
                ckpt.unlink()
                self._log(f"  Removed old checkpoint: {ckpt.name}")
            except Exception:
                pass

    def save(
        self,
        model: nn.Module,
        optimizer: Optional[optim.Optimizer] = None,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
        epoch: int = 0,
        step: int = 0,
        metrics: Optional[Dict[str, Any]] = None,
        is_best: bool = False,
        checkpoint_type: str = "step",
    ) -> Path:
        """Save a checkpoint to disk.

        Args:
            model: The model to save.
            optimizer: Optional optimizer to save state.
            scheduler: Optional LR scheduler to save state.
            epoch: Current epoch number.
            step: Current global step.
            metrics: Dictionary of metrics (loss, etc.) to save.
            is_best: If True, also save as "best" checkpoint.
            checkpoint_type: One of "step" (includes step number), "latest", "best".

        Returns:
            Path to the saved checkpoint file.

        Examples:
            >>> path = checkpointer.save(
            ...     model, optimizer,
            ...     epoch=5, step=10000,
            ...     metrics={"val_loss": 0.3},
            ...     is_best=True
            ... )
        """
        metrics = metrics or {}

        # Update internal state
        self._current_state.epoch = epoch
        self._current_state.global_step = step
        self._current_state.metrics = metrics
        self._current_state.timestamp = datetime.now().isoformat()

        if "val_loss" in metrics and metrics["val_loss"] is not None:
            val_loss = metrics["val_loss"]
            if val_loss < self._current_state.best_val_loss:
                self._current_state.best_val_loss = val_loss
                is_best = True

        # Build checkpoint dict
        checkpoint = {
            "model_state_dict": model.state_dict(),
            "state": self._current_state.to_dict(),
        }

        if optimizer is not None and self.save_optimizer:
            checkpoint["optimizer_state_dict"] = optimizer.state_dict()

        if scheduler is not None and self.save_scheduler:
            checkpoint["scheduler_state_dict"] = scheduler.state_dict()

        # Determine filename
        if checkpoint_type == "step":
            filename = f"checkpoint_step_{step:08d}.pt"
        elif checkpoint_type == "best":
            filename = "checkpoint_best.pt"
        else:
            filename = "checkpoint_latest.pt"

        # Save to checkpoint directory
        save_path = self.checkpoint_dir / filename
        torch.save(checkpoint, save_path)
        self._log(f"✓ Saved checkpoint: {save_path}")

        # Also save as "latest" for easy resumption
        if checkpoint_type == "step":
            latest_path = self.checkpoint_dir / "checkpoint_latest.pt"
            shutil.copy2(save_path, latest_path)

        # Save as "best" if applicable
        if is_best:
            best_path = self.checkpoint_dir / "checkpoint_best.pt"
            shutil.copy2(save_path, best_path)
            self._log(
                f"  ★ New best checkpoint (val_loss={metrics.get('val_loss', 'N/A')})"
            )

        # Cleanup old checkpoints
        self._cleanup_old_checkpoints()

        # Update tracking
        self._last_save_time = time.time()
        self._last_save_step = step

        return save_path

    def load(
        self,
        model: nn.Module,
        optimizer: Optional[optim.Optimizer] = None,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
        checkpoint_type: str = "latest",
        checkpoint_path: Optional[Union[str, Path]] = None,
        map_location: Optional[torch.device] = None,
    ) -> CheckpointState:
        """Load a checkpoint and restore model/optimizer state.

        Args:
            model: Model to load state into.
            optimizer: Optional optimizer to restore state.
            scheduler: Optional scheduler to restore state.
            checkpoint_type: One of "latest", "best", or a specific step number.
            checkpoint_path: Explicit path to checkpoint file (overrides type).
            map_location: Device to map tensors to.

        Returns:
            CheckpointState with training metadata.

        Raises:
            FileNotFoundError: If no checkpoint exists.

        Examples:
            >>> state = checkpointer.load(model, optimizer)
            >>> print(f"Resuming from epoch {state.epoch}")

            >>> # Load best checkpoint
            >>> state = checkpointer.load(model, checkpoint_type="best")
        """
        if checkpoint_path is not None:
            load_path = Path(checkpoint_path)
        else:
            if checkpoint_type == "latest":
                filename = "checkpoint_latest.pt"
            elif checkpoint_type == "best":
                filename = "checkpoint_best.pt"
            else:
                # Assume it's a step number
                filename = f"checkpoint_step_{int(checkpoint_type):08d}.pt"

            load_path = self.checkpoint_dir / filename

        if not load_path.exists():
            raise FileNotFoundError(f"No checkpoint found at {load_path}")

        self._log(f"Loading checkpoint from {load_path}...")

        checkpoint = torch.load(
            load_path, map_location=map_location, weights_only=False
        )

        # Restore model
        model.load_state_dict(checkpoint["model_state_dict"])

        # Restore optimizer if available
        if optimizer is not None and "optimizer_state_dict" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        # Restore scheduler if available
        if scheduler is not None and "scheduler_state_dict" in checkpoint:
            scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

        # Restore state
        state_dict = checkpoint.get("state", {})
        self._current_state = CheckpointState.from_dict(state_dict)

        self._log(
            f"✓ Restored from epoch {self._current_state.epoch}, "
            f"step {self._current_state.global_step}"
        )

        return self._current_state

    def has_checkpoint(self, checkpoint_type: str = "latest") -> bool:
        """Check if a checkpoint exists.

        Args:
            checkpoint_type: "latest", "best", or step number.

        Returns:
            True if checkpoint exists.

        Examples:
            >>> if checkpointer.has_checkpoint():
            ...     state = checkpointer.load(model, optimizer)
        """
        if checkpoint_type == "latest":
            filename = "checkpoint_latest.pt"
        elif checkpoint_type == "best":
            filename = "checkpoint_best.pt"
        else:
            filename = f"checkpoint_step_{int(checkpoint_type):08d}.pt"

        return (self.checkpoint_dir / filename).exists()

    def should_save(self, current_step: int) -> bool:
        """Check if it's time for an auto-save based on configured intervals.

        Args:
            current_step: Current training step.

        Returns:
            True if a save should be triggered.
        """
        # Check step-based interval
        if self.save_interval_steps > 0:
            steps_since_save = current_step - self._last_save_step
            if steps_since_save >= self.save_interval_steps:
                return True

        # Check time-based interval
        if self.save_interval_minutes > 0:
            minutes_since_save = (time.time() - self._last_save_time) / 60
            if minutes_since_save >= self.save_interval_minutes:
                return True

        return False

    def step_callback(
        self,
        model: nn.Module,
        optimizer: Optional[optim.Optimizer] = None,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
        step: int = 0,
        epoch: int = 0,
        metrics: Optional[Dict[str, Any]] = None,
    ) -> Optional[Path]:
        """Callback to be called each training step for auto-save logic.

        This method checks if it's time to save based on the configured
        intervals and saves if needed.

        Args:
            model: Current model.
            optimizer: Current optimizer.
            scheduler: Current scheduler.
            step: Current global step.
            epoch: Current epoch.
            metrics: Current metrics dict.

        Returns:
            Path to saved checkpoint if save occurred, None otherwise.

        Examples:
            >>> for step, batch in enumerate(train_loader):
            ...     loss = train_step(model, batch)
            ...     checkpointer.step_callback(
            ...         model, optimizer, step=step,
            ...         metrics={"train_loss": loss.item()}
            ...     )
        """
        if self.should_save(step):
            return self.save(
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                epoch=epoch,
                step=step,
                metrics=metrics,
                checkpoint_type="step",
            )
        return None

    def get_state(self) -> CheckpointState:
        """Get the current checkpoint state.

        Returns:
            Current CheckpointState object.
        """
        return self._current_state

    def update_state(
        self,
        train_loss: Optional[float] = None,
        val_loss: Optional[float] = None,
        **kwargs: Any,
    ) -> None:
        """Update the current state with new metrics.

        Args:
            train_loss: Training loss to append.
            val_loss: Validation loss to append.
            **kwargs: Additional metrics to update.
        """
        if train_loss is not None:
            self._current_state.train_losses.append(train_loss)

        if val_loss is not None:
            self._current_state.val_losses.append(val_loss)
            if val_loss < self._current_state.best_val_loss:
                self._current_state.best_val_loss = val_loss

        self._current_state.metrics.update(kwargs)


# ============================================================================
# COLAB CHECKPOINTER
# ============================================================================


class ColabCheckpointer(LocalCheckpointer):
    """Auto-checkpointing manager with Google Drive persistence for Colab.

    This class extends LocalCheckpointer to save checkpoints to Google Drive,
    providing automatic persistence even if the Colab runtime disconnects.

    Key Features:
        - Automatic Google Drive mounting and setup
        - Drive space checking with early warnings
        - Configurable auto-save interval (time or step based)
        - Full training state resumption (model, optimizer, scheduler, epoch)
        - Keeps multiple checkpoint versions with automatic cleanup
        - Falls back to local storage if Drive is unavailable

    Args:
        experiment_name: Unique name for this experiment/run.
        drive_path: Path within Google Drive for checkpoints.
            Default: "My Drive/torchlingo_checkpoints".
        mount_point: Where to mount Google Drive. Default: "/content/drive".
        auto_mount: If True, automatically mount Drive on init. Default: True.
        save_interval_minutes: Auto-save every N minutes during training.
            Default: 10 minutes. Set to 0 to disable time-based saves.
        save_interval_steps: Auto-save every N training steps.
            Default: 0 (disabled). Set to positive int to enable.
        keep_n_checkpoints: Number of recent checkpoints to keep. Default: 3.
        save_optimizer: Whether to save optimizer state. Default: True.
        save_scheduler: Whether to save scheduler state. Default: True.
        verbose: Print status messages. Default: True.

    Examples:
        >>> # Basic usage in Colab
        >>> checkpointer = ColabCheckpointer("my_translation_exp")
        >>> # Will auto-mount Drive and create checkpoint directory

        >>> # Save a checkpoint
        >>> checkpointer.save(
        ...     model=model,
        ...     optimizer=optimizer,
        ...     epoch=5,
        ...     step=10000,
        ...     metrics={"train_loss": 0.5, "val_loss": 0.4}
        ... )

        >>> # Resume from checkpoint
        >>> state = checkpointer.load(model, optimizer)
        >>> print(f"Resuming from epoch {state.epoch}, step {state.global_step}")

    Raises:
        RuntimeError: If not running in Google Colab. Use LocalCheckpointer instead.
    """

    DEFAULT_DRIVE_PATH = "My Drive/torchlingo_checkpoints"

    def __init__(
        self,
        experiment_name: str,
        drive_path: str = DEFAULT_DRIVE_PATH,
        mount_point: str = "/content/drive",
        auto_mount: bool = True,
        save_interval_minutes: float = 10.0,
        save_interval_steps: int = 0,
        keep_n_checkpoints: int = 3,
        save_optimizer: bool = True,
        save_scheduler: bool = True,
        verbose: bool = True,
    ):
        self.mount_point = mount_point
        self._drive_mounted = False

        # Check if in Colab
        if not is_colab():
            raise RuntimeError(
                "ColabCheckpointer requires Google Colab environment. "
                "Use LocalCheckpointer for local development."
            )

        # Try to mount Drive
        if auto_mount:
            self._mount_drive(verbose)

        # Determine checkpoint directory
        if is_drive_mounted(mount_point):
            checkpoint_dir = Path(mount_point) / drive_path / experiment_name
            self._drive_mounted = True
        else:
            if verbose:
                warnings.warn(
                    "⚠ Google Drive not mounted. "
                    "Checkpoints will only be saved locally to /content/checkpoints. "
                    "These will be LOST when the runtime disconnects!"
                )
            checkpoint_dir = Path("/content/checkpoints") / experiment_name

        # Initialize parent class
        super().__init__(
            experiment_name=experiment_name,
            checkpoint_dir=checkpoint_dir,
            save_interval_minutes=save_interval_minutes,
            save_interval_steps=save_interval_steps,
            keep_n_checkpoints=keep_n_checkpoints,
            save_optimizer=save_optimizer,
            save_scheduler=save_scheduler,
            verbose=verbose,
        )

        # Additional drive space check with more aggressive warning
        if self._drive_mounted:
            free_space = get_drive_free_space(checkpoint_dir)
            if free_space is not None:
                free_space_gb = free_space / (1024 * 1024 * 1024)
                if verbose:
                    print(f"  Google Drive free space: {free_space_gb:.2f} GB")

    def _mount_drive(self, verbose: bool = True) -> None:
        """Attempt to mount Google Drive."""
        try:
            mount_drive(self.mount_point)
            self._drive_mounted = True
            if verbose:
                print("✓ Google Drive mounted successfully")
        except Exception as e:
            if verbose:
                warnings.warn(f"⚠ Could not mount Drive: {e}")

    @property
    def is_drive_mounted(self) -> bool:
        """Check if Google Drive is currently mounted."""
        return self._drive_mounted and is_drive_mounted(self.mount_point)

    def save(
        self,
        model: nn.Module,
        optimizer: Optional[optim.Optimizer] = None,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
        epoch: int = 0,
        step: int = 0,
        metrics: Optional[Dict[str, Any]] = None,
        is_best: bool = False,
        checkpoint_type: str = "step",
    ) -> Path:
        """Save a checkpoint to Google Drive.

        Extends LocalCheckpointer.save() with Drive-specific checks.
        """
        # Check drive space before saving
        if self._drive_mounted:
            check_drive_space(
                self.checkpoint_dir,
                min_space_bytes=ESTIMATED_CHECKPOINT_SIZE_BYTES * 2,
                verbose=self.verbose,
            )

        return super().save(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            epoch=epoch,
            step=step,
            metrics=metrics,
            is_best=is_best,
            checkpoint_type=checkpoint_type,
        )


# ============================================================================
# FACTORY FUNCTION
# ============================================================================


def create_checkpointer_from_config(
    config: "Config",
) -> Optional[Union[LocalCheckpointer, ColabCheckpointer]]:
    """Create the appropriate checkpointer based on config settings.

    This factory function creates either a LocalCheckpointer or ColabCheckpointer
    based on the config settings and the runtime environment.

    Args:
        config: TorchLingo Config object with checkpoint settings.

    Returns:
        LocalCheckpointer or ColabCheckpointer instance, or None if
        checkpointing is disabled in config.

    Examples:
        >>> from torchlingo.config import Config
        >>> cfg = Config(use_colab_checkpointing=True, experiment_name="hw3")
        >>> checkpointer = create_checkpointer_from_config(cfg)
    """
    if not config.use_colab_checkpointing:
        return None

    # Determine which checkpointer to use
    if is_colab():
        return ColabCheckpointer(
            experiment_name=config.experiment_name,
            drive_path=config.colab_drive_path,
            save_interval_minutes=config.colab_checkpoint_interval_minutes,
            save_interval_steps=config.colab_checkpoint_interval_steps,
            keep_n_checkpoints=config.colab_keep_n_checkpoints,
            verbose=True,
        )
    else:
        # Use LocalCheckpointer when not in Colab
        return LocalCheckpointer(
            experiment_name=config.experiment_name,
            save_interval_minutes=config.colab_checkpoint_interval_minutes,
            save_interval_steps=config.colab_checkpoint_interval_steps,
            keep_n_checkpoints=config.colab_keep_n_checkpoints,
            verbose=True,
        )


# ============================================================================
# CONVENIENCE FUNCTIONS
# ============================================================================


def setup_colab_checkpointing(
    experiment_name: str,
    auto_resume: bool = True,
    **kwargs: Any,
) -> ColabCheckpointer:
    """Quick setup for Colab checkpointing with sensible defaults.

    This is a convenience function that creates a ColabCheckpointer with
    good defaults for student use in Google Colab.

    Args:
        experiment_name: Name for your experiment (e.g., "hw3_translation").
        auto_resume: If True, will prompt to resume from existing checkpoint.
        **kwargs: Additional arguments passed to ColabCheckpointer.

    Returns:
        Configured ColabCheckpointer instance.

    Raises:
        RuntimeError: If not running in Google Colab.

    Examples:
        >>> from torchlingo.checkpoint import setup_colab_checkpointing
        >>> checkpointer = setup_colab_checkpointing("hw3_en_es_translation")
        >>> # If previous checkpoint exists, you'll be asked to resume
    """
    checkpointer = ColabCheckpointer(experiment_name=experiment_name, **kwargs)

    if auto_resume and checkpointer.has_checkpoint():
        print(f"\n📁 Found existing checkpoint for '{experiment_name}'")
        print("   Run checkpointer.load(model, optimizer) to resume training.")

    return checkpointer


def setup_local_checkpointing(
    experiment_name: str,
    auto_resume: bool = True,
    **kwargs: Any,
) -> LocalCheckpointer:
    """Quick setup for local checkpointing.

    This is a convenience function that creates a LocalCheckpointer with
    good defaults for local development.

    Args:
        experiment_name: Name for your experiment.
        auto_resume: If True, will prompt to resume from existing checkpoint.
        **kwargs: Additional arguments passed to LocalCheckpointer.

    Returns:
        Configured LocalCheckpointer instance.

    Examples:
        >>> from torchlingo.checkpoint import setup_local_checkpointing
        >>> checkpointer = setup_local_checkpointing("my_experiment")
    """
    checkpointer = LocalCheckpointer(experiment_name=experiment_name, **kwargs)

    if auto_resume and checkpointer.has_checkpoint():
        print(f"\n📁 Found existing checkpoint for '{experiment_name}'")
        print("   Run checkpointer.load(model, optimizer) to resume training.")

    return checkpointer


def save_checkpoint_to_drive(
    model: nn.Module,
    experiment_name: str,
    optimizer: Optional[optim.Optimizer] = None,
    epoch: int = 0,
    step: int = 0,
    metrics: Optional[Dict[str, Any]] = None,
) -> Optional[Path]:
    """One-liner to save a checkpoint to Google Drive.

    This is a simple convenience function for quick saves without setting
    up a full ColabCheckpointer.

    Args:
        model: Model to save.
        experiment_name: Name for the experiment.
        optimizer: Optional optimizer to save.
        epoch: Current epoch.
        step: Current step.
        metrics: Metrics dictionary.

    Returns:
        Path to saved checkpoint, or None if not in Colab.

    Examples:
        >>> save_checkpoint_to_drive(model, "my_exp", epoch=5)
    """
    if not is_colab():
        warnings.warn(
            "Not in Colab - use LocalCheckpointer or save_checkpoint_locally()."
        )
        return save_checkpoint_locally(
            model=model,
            experiment_name=experiment_name,
            optimizer=optimizer,
            epoch=epoch,
            step=step,
            metrics=metrics,
        )

    checkpointer = ColabCheckpointer(
        experiment_name=experiment_name,
        verbose=False,
        keep_n_checkpoints=1,
    )

    return checkpointer.save(
        model=model,
        optimizer=optimizer,
        epoch=epoch,
        step=step,
        metrics=metrics,
    )


def save_checkpoint_locally(
    model: nn.Module,
    experiment_name: str,
    optimizer: Optional[optim.Optimizer] = None,
    epoch: int = 0,
    step: int = 0,
    metrics: Optional[Dict[str, Any]] = None,
) -> Path:
    """One-liner to save a checkpoint locally.

    Args:
        model: Model to save.
        experiment_name: Name for the experiment.
        optimizer: Optional optimizer to save.
        epoch: Current epoch.
        step: Current step.
        metrics: Metrics dictionary.

    Returns:
        Path to saved checkpoint.

    Examples:
        >>> save_checkpoint_locally(model, "my_exp", epoch=5)
    """
    checkpointer = LocalCheckpointer(
        experiment_name=experiment_name,
        verbose=False,
        keep_n_checkpoints=1,
    )

    return checkpointer.save(
        model=model,
        optimizer=optimizer,
        epoch=epoch,
        step=step,
        metrics=metrics,
    )
