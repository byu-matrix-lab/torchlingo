"""Resumable training checkpoints, and surviving a Colab disconnect.

A Colab session ends when the browser tab closes, the runtime idles out, or the
laptop lid shuts. Without periodic checkpoints that takes the whole training run
with it, which is a bad first experience for a student and an avoidable one.

**This is not the same thing as** :mod:`torchlingo.checkpoint`. That module saves
a *finished* model for inference: weights, architecture config and tokenizers in
one portable file. This module saves *training state* so a run can pick up where
it stopped: optimizer, scheduler, epoch, step, and the loss history. The two
answer different questions and deliberately stay separate.

Typical usage in Colab:
    >>> from torchlingo.training_checkpoint import (
    ...     TrainingCheckpointer, default_checkpoint_dir, mount_drive
    ... )
    >>> mount_drive()                                  # no-op outside Colab
    >>> checkpointer = TrainingCheckpointer(
    ...     "my-experiment", checkpoint_dir=default_checkpoint_dir("my-experiment")
    ... )
    >>> result = train_model(model, loader, checkpointer=checkpointer)

Re-running that same cell after a disconnect resumes from the last save.

Note:
    The Colab and Google Drive paths here cannot be exercised by CI — GitHub
    runners have no Drive to mount. Everything else is tested; the Drive
    integration has been written carefully and reviewed, but it has not been run
    in a live Colab session. Treat that part as unverified until someone does.
"""

from __future__ import annotations

import shutil
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
from torch import nn, optim

DRIVE_MOUNT_POINT = Path("/content/drive")
DRIVE_ROOT = DRIVE_MOUNT_POINT / "MyDrive"

__all__ = [
    "CheckpointState",
    "TrainingCheckpointer",
    "default_checkpoint_dir",
    "is_colab",
    "mount_drive",
]


def is_colab() -> bool:
    """Report whether the current process is running inside Google Colab.

    Returns:
        bool: True when the Colab runtime module is importable.

    Examples:
        >>> is_colab()
        False
    """
    try:
        import google.colab  # noqa: F401
    except ImportError:
        return False
    return True


def mount_drive(mount_point: Path | str = DRIVE_MOUNT_POINT) -> bool:
    """Mount Google Drive, if running in Colab and not already mounted.

    Safe to call unconditionally: outside Colab it does nothing and returns
    False, so notebooks do not need to branch on the environment.

    Args:
        mount_point (Path | str, optional): Where Drive should be mounted.

    Returns:
        bool: True if Drive is mounted when this returns, False otherwise.
    """
    mount_point = Path(mount_point)
    if not is_colab():
        return False
    if (mount_point / "MyDrive").exists():
        return True

    from google.colab import drive

    drive.mount(str(mount_point))
    return (mount_point / "MyDrive").exists()


def default_checkpoint_dir(experiment_name: str) -> Path:
    """Pick a sensible checkpoint directory for the current environment.

    In Colab this is a folder on Drive, so checkpoints outlive the runtime.
    Anywhere else it is a local directory.

    Args:
        experiment_name (str): Used as the final path segment.

    Returns:
        Path: Directory to write checkpoints into. Not created here.
    """
    root = DRIVE_ROOT / "torchlingo" if is_colab() else Path("checkpoints")
    return root / experiment_name


def free_space_mb(path: Path) -> float | None:
    """Report free space at ``path`` in megabytes.

    Args:
        path (Path): Any existing directory.

    Returns:
        float | None: Free megabytes, or None if it could not be determined.
    """
    try:
        return shutil.disk_usage(path).free / (1024 * 1024)
    except OSError:
        return None


@dataclass
class CheckpointState:
    """Training progress, everything needed to resume a run.

    Attributes:
        epoch (int): Last completed epoch, 0-indexed.
        global_step (int): Total optimizer steps taken.
        best_val_loss (float): Best validation loss seen so far.
        train_losses (list[float]): Training loss per epoch.
        val_losses (list[float]): Validation loss per epoch.
        metrics (dict): Any extra values the caller wants carried along.
        timestamp (str): UTC ISO-8601 time the state was created.
        experiment_name (str): Name the checkpoints are filed under.
    """

    epoch: int = 0
    global_step: int = 0
    best_val_loss: float = float("inf")
    train_losses: list[float] = field(default_factory=list)
    val_losses: list[float] = field(default_factory=list)
    metrics: dict[str, Any] = field(default_factory=dict)
    timestamp: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    experiment_name: str = "torchlingo"

    def to_dict(self) -> dict[str, Any]:
        """Return the state as a plain dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> CheckpointState:
        """Rebuild state from a dictionary, ignoring unknown keys.

        Unknown keys are dropped rather than raising, so a checkpoint written by
        a newer version still loads.
        """
        known = {f for f in cls.__dataclass_fields__}
        return cls(**{k: v for k, v in data.items() if k in known})


class TrainingCheckpointer:
    """Save and restore training state, periodically and on demand.

    Writes two files: ``latest.pt``, overwritten each save, and ``best.pt``,
    written only when the caller says this is the best result so far. Keeping
    exactly two is deliberate — on Drive, a growing pile of numbered checkpoints
    is how a student silently fills their quota.

    Args:
        experiment_name (str): Name the checkpoints are filed under.
        checkpoint_dir (Path | str, optional): Where to write. Defaults to
            :func:`default_checkpoint_dir`, which is a Drive folder in Colab.
        save_every_seconds (float, optional): Minimum wall-clock gap between
            automatic saves. Set to 0 to disable time-based saving.
        save_every_steps (int, optional): Minimum step gap between automatic
            saves. Set to 0 to disable step-based saving.
        verbose (bool, optional): Print when saving, loading and resuming.

    Attributes:
        checkpoint_dir (Path): Resolved directory, created on construction.
        state (CheckpointState): Current training progress.

    Examples:
        >>> import tempfile
        >>> with tempfile.TemporaryDirectory() as tmp:
        ...     ckpt = TrainingCheckpointer("demo", checkpoint_dir=tmp, verbose=False)
        ...     ckpt.has_checkpoint()
        False
    """

    LATEST = "latest.pt"
    BEST = "best.pt"

    def __init__(
        self,
        experiment_name: str,
        checkpoint_dir: Path | str | None = None,
        save_every_seconds: float = 600.0,
        save_every_steps: int = 0,
        verbose: bool = True,
    ) -> None:
        self.experiment_name = experiment_name
        self.save_every_seconds = save_every_seconds
        self.save_every_steps = save_every_steps
        self.verbose = verbose

        directory = (
            Path(checkpoint_dir)
            if checkpoint_dir is not None
            else default_checkpoint_dir(experiment_name)
        )
        directory.mkdir(parents=True, exist_ok=True)
        self.checkpoint_dir = directory

        self.state = CheckpointState(experiment_name=experiment_name)
        self._last_save_time = time.monotonic()
        self._last_save_step = 0

        if self.verbose:
            free = free_space_mb(self.checkpoint_dir)
            space = f", {free:,.0f} MB free" if free is not None else ""
            self._log(f"checkpoints -> {self.checkpoint_dir}{space}")

    def _log(self, message: str) -> None:
        """Print ``message`` when verbose."""
        if self.verbose:
            print(f"[checkpoint] {message}")

    def path_for(self, which: str = "latest") -> Path:
        """Return the path of the ``latest`` or ``best`` checkpoint."""
        if which not in ("latest", "best"):
            raise ValueError(f"which must be 'latest' or 'best', got {which!r}")
        return self.checkpoint_dir / (self.LATEST if which == "latest" else self.BEST)

    def has_checkpoint(self, which: str = "latest") -> bool:
        """Report whether a checkpoint exists to resume from."""
        return self.path_for(which).exists()

    def save(
        self,
        model: nn.Module,
        optimizer: optim.Optimizer | None = None,
        scheduler: Any = None,
        *,
        is_best: bool = False,
    ) -> Path:
        """Write the current training state.

        Args:
            model (nn.Module): Model whose weights are saved.
            optimizer (optim.Optimizer, optional): Saved so the run resumes with
                its momentum and step counts intact, not just its weights.
            scheduler (optional): Learning-rate scheduler, saved for the same reason.
            is_best (bool, optional): Also write ``best.pt``.

        Returns:
            Path: The path written.
        """
        payload = {
            "state": self.state.to_dict(),
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict() if optimizer is not None else None,
            "scheduler": scheduler.state_dict() if scheduler is not None else None,
        }

        # Write to a temporary file first, then move it into place. A Colab
        # runtime can die mid-write, and a half-written latest.pt is worse than
        # no checkpoint at all: it fails at load time, when the work is gone.
        target = self.path_for("latest")
        staging = target.with_suffix(".tmp")
        torch.save(payload, staging)
        staging.replace(target)

        if is_best:
            shutil.copyfile(target, self.path_for("best"))

        self._last_save_time = time.monotonic()
        self._last_save_step = self.state.global_step
        self._log(
            f"saved epoch {self.state.epoch}, step {self.state.global_step}"
            f"{' (best)' if is_best else ''}"
        )
        return target

    def load(
        self,
        model: nn.Module,
        optimizer: optim.Optimizer | None = None,
        scheduler: Any = None,
        which: str = "latest",
        map_location: Any = "cpu",
    ) -> CheckpointState:
        """Restore training state in place and return it.

        Args:
            model (nn.Module): Model to load weights into.
            optimizer (optim.Optimizer, optional): Restored when present in the file.
            scheduler (optional): Restored when present in the file.
            which (str, optional): ``"latest"`` or ``"best"``.
            map_location (optional): Passed through to ``torch.load``.

        Returns:
            CheckpointState: The restored progress.

        Raises:
            FileNotFoundError: If no such checkpoint exists.
        """
        path = self.path_for(which)
        if not path.exists():
            raise FileNotFoundError(f"No {which} checkpoint at {path}")

        # weights_only=False: the payload holds the state dataclass alongside
        # tensors. Only load checkpoints you or your own runs produced.
        payload = torch.load(path, map_location=map_location, weights_only=False)

        model.load_state_dict(payload["model"])
        if optimizer is not None and payload.get("optimizer") is not None:
            optimizer.load_state_dict(payload["optimizer"])
        if scheduler is not None and payload.get("scheduler") is not None:
            scheduler.load_state_dict(payload["scheduler"])

        self.state = CheckpointState.from_dict(payload["state"])
        self._log(
            f"resumed from epoch {self.state.epoch}, step {self.state.global_step}"
        )
        return self.state

    def should_save(self) -> bool:
        """Report whether enough time or steps have passed to save again."""
        steps_since = self.state.global_step - self._last_save_step
        if self.save_every_steps and steps_since >= self.save_every_steps:
            return True
        seconds_since = time.monotonic() - self._last_save_time
        return bool(
            self.save_every_seconds and seconds_since >= self.save_every_seconds
        )

    def update(
        self,
        *,
        epoch: int | None = None,
        global_step: int | None = None,
        train_loss: float | None = None,
        val_loss: float | None = None,
        metrics: dict[str, Any] | None = None,
    ) -> None:
        """Record progress without writing anything to disk."""
        if epoch is not None:
            self.state.epoch = epoch
        if global_step is not None:
            self.state.global_step = global_step
        if train_loss is not None:
            self.state.train_losses.append(train_loss)
        if val_loss is not None:
            self.state.val_losses.append(val_loss)
            self.state.best_val_loss = min(self.state.best_val_loss, val_loss)
        if metrics:
            self.state.metrics.update(metrics)

    def maybe_save(
        self,
        model: nn.Module,
        optimizer: optim.Optimizer | None = None,
        scheduler: Any = None,
        *,
        global_step: int | None = None,
    ) -> Path | None:
        """Save only if :meth:`should_save` says it is time.

        This is what the training loop calls every step; the interval logic
        lives here so the loop stays readable.

        Returns:
            Path | None: The path written, or None if nothing was due.
        """
        if global_step is not None:
            self.state.global_step = global_step
        if not self.should_save():
            return None
        return self.save(model, optimizer, scheduler)
