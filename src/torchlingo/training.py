"""High-level training helper for TorchLingo models.

Provides a lightweight, single-GPU training loop that works for both
Transformer and LSTM seq2seq models. Decoding and translation utilities live
in `torchlingo.inference` to keep responsibilities separated.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from pathlib import Path

import torch
from torch import nn, optim
from torch.nn.utils import clip_grad_norm_

try:
    from tqdm.auto import tqdm
except ImportError:  # pragma: no cover - optional dependency
    # Fallback: simple identity when tqdm is not available (keeps behavior testable)
    def tqdm(x, **kwargs):
        return x


try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:  # pragma: no cover - optional dependency
    SummaryWriter = None


from .config import Config, get_default_config


@dataclass
class TrainResult:
    """Container for training metrics and artifacts.

    Attributes:
        train_losses: Mean training loss per epoch.
        val_losses: Mean validation loss per epoch (empty when no val_loader).
        best_checkpoint: Path to the best checkpoint file if saved, else None.
    """

    train_losses: list[float]
    val_losses: list[float]
    best_checkpoint: Path | None


def _resolve_device(device: torch.device | None) -> torch.device:
    return (
        device
        if device is not None
        else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    )


def get_transformer_scheduler(
    optimizer: torch.optim.Optimizer,
    d_model: int = 512,
    warmup_steps: int = 4000,
) -> torch.optim.lr_scheduler.LambdaLR:
    """Create the learning rate scheduler from 'Attention Is All You Need'.

    Implements the learning rate schedule from Vaswani et al. (2017):
        lr = d_model^(-0.5) * min(step^(-0.5), step * warmup_steps^(-1.5))

    This schedule:
    1. Increases learning rate linearly during warmup
    2. Decreases proportionally to the inverse square root of the step number afterward

    This helps stabilize training by starting with smaller learning rates and
    prevents the model from converging too quickly to suboptimal solutions.

    Args:
        optimizer: The optimizer to schedule.
        d_model: Model dimension (typically 512). Used for scaling.
        warmup_steps: Number of warmup steps (typically 4000).

    Returns:
        LambdaLR scheduler that adjusts learning rate according to the Transformer schedule.

    Example:
        >>> opt = torch.optim.Adam(model.parameters(), lr=1.0)
        >>> scheduler = get_transformer_scheduler(opt, d_model=512, warmup_steps=4000)
        >>> for epoch in range(num_epochs):
        >>>     ...
        >>>     scheduler.step()
    """

    def lr_lambda(step: int) -> float:
        # Avoid division by zero on first step
        if step == 0:
            step = 1
        # Transformer schedule: scale by d_model and apply warmup + inverse sqrt decay
        return d_model ** (-0.5) * min(step ** (-0.5), step * warmup_steps ** (-1.5))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def get_cosine_annealing_scheduler(
    optimizer: torch.optim.Optimizer,
    warmup_steps: int = 4000,
    total_steps: int = 100000,
    min_lr_ratio: float = 0.1,
) -> torch.optim.lr_scheduler.LambdaLR:
    """Create a cosine annealing scheduler with linear warmup.

    This scheduler is now standard in modern deep learning and often outperforms
    the inverse square root schedule. It:
    1. Increases learning rate linearly during warmup
    2. Decreases smoothly following a cosine curve from peak to min_lr
    3. Maintains a reasonable learning rate throughout training (unlike inverse sqrt)

    Args:
        optimizer: The optimizer to schedule.
        warmup_steps: Number of warmup steps (typically 4000-8000).
        total_steps: Total number of training steps (num_epochs * steps_per_epoch).
        min_lr_ratio: Minimum learning rate as ratio of peak (default 0.1 = 10% of peak).

    Returns:
        LambdaLR scheduler that adjusts learning rate with cosine annealing.

    Example:
        >>> opt = torch.optim.Adam(model.parameters(), lr=1e-3)
        >>> scheduler = get_cosine_annealing_scheduler(opt, warmup_steps=8000, total_steps=200000)
        >>> for epoch in range(num_epochs):
        >>>     for batch in train_loader:
        >>>         ...
        >>>         scheduler.step()
    """
    import math

    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            # Linear warmup
            return float(step) / float(max(1, warmup_steps))
        else:
            # Cosine annealing
            progress = float(step - warmup_steps) / float(
                max(1, total_steps - warmup_steps)
            )
            progress = min(
                progress, 1.0
            )  # Cap at 1.0 if training goes over total_steps
            cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))
            return min_lr_ratio + (1.0 - min_lr_ratio) * cosine_decay

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def train_model(
    model: nn.Module,
    train_loader: torch.utils.data.DataLoader,
    val_loader: torch.utils.data.DataLoader | None = None,
    num_epochs: int = 10,
    optimizer: optim.Optimizer | None = None,
    criterion: nn.Module | None = None,
    scheduler: torch.optim.lr_scheduler._LRScheduler | None = None,
    gradient_clip: float | None = None,
    device: torch.device | None = None,
    config: Config | None = None,
    save_dir: Path | None = None,
    use_amp: bool = False,
    log_every: int = 0,
    accumulation_steps: int = 1,
) -> TrainResult:
    """Train a seq2seq model with optional validation and checkpointing.

    Args:
        model: Seq2seq model (Transformer or LSTM) supporting forward(src, tgt).
        train_loader: DataLoader yielding (src, tgt) tensors.
        val_loader: Optional DataLoader for validation loss.
        num_epochs: Number of training epochs to run.
        optimizer: Optimizer instance. Defaults to Adam with config.learning_rate.
        criterion: Loss function. Defaults to CrossEntropyLoss with padding ignore.
        scheduler: Optional LR scheduler stepped once per optimizer step.
        gradient_clip: If provided, max gradient norm applied each step.
        device: Torch device. Defaults to CUDA if available else CPU.
        config: TorchLingo Config providing pad_idx, label_smoothing, etc.
        save_dir: If provided, best model (lowest val loss) is saved here.
        use_amp: Enable automatic mixed precision for speed on GPUs. Uses
            bfloat16 where supported (more numerically stable for transformers),
            otherwise float16 with gradient scaling.
        log_every: If >0, prints/logs the mean batch loss over the last
            log_every steps (a smooth, epoch-continuous "train/loss" curve).
        accumulation_steps: Number of micro-batches to accumulate gradients over
            before each optimizer step. The *effective* batch size is
            ``train_loader.batch_size * accumulation_steps``, at the memory cost
            of a single micro-batch — useful for fitting a large effective batch
            on a small GPU. Defaults to 1 (an optimizer step every batch).
            Counters keyed on "steps" (num_steps, step_limit, val_interval,
            save_interval) count optimizer steps, not micro-batches.

    Returns:
        TrainResult containing per-epoch losses and optional checkpoint path.

    Raises:
        ValueError: If accumulation_steps < 1.

    Warns:
        RuntimeWarning: If an epoch completes with 0 training steps because
            train_loader yielded no batches (e.g., batch_size larger than the
            dataset with bucketing enabled, which drops incomplete batches).
    """

    if accumulation_steps < 1:
        raise ValueError("accumulation_steps must be >= 1")

    cfg = config if config is not None else get_default_config()
    device = _resolve_device(device)
    model = model.to(device)

    opt = (
        optimizer
        if optimizer is not None
        else optim.AdamW(
            model.parameters(),
            lr=cfg.learning_rate,
            betas=cfg.adam_betas,
            eps=cfg.adam_eps,
            weight_decay=cfg.weight_decay,
        )
    )
    # Select learning rate scheduler based on config if no scheduler provided
    # is_plateau_scheduler tracks whether sched needs val loss (not per-step)
    is_plateau_scheduler = False
    if scheduler is not None:
        sched = scheduler
        is_plateau_scheduler = isinstance(
            scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau
        )
    else:
        scheduler_type = getattr(cfg, "scheduler_type", "cosine").lower()
        if scheduler_type == "plateau":
            sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
                opt,
                mode="min",
                factor=0.5,
                patience=getattr(cfg, "scheduler_patience", 3),
                min_lr=1e-7,
            )
            is_plateau_scheduler = True
        elif scheduler_type == "cosine":
            total_steps = num_epochs * len(train_loader)
            sched = get_cosine_annealing_scheduler(
                opt,
                warmup_steps=cfg.warmup_steps,
                total_steps=total_steps,
                min_lr_ratio=0.1,
            )
        elif scheduler_type in ("transformer", "noam"):
            sched = get_transformer_scheduler(
                opt,
                d_model=getattr(cfg, "d_model", 512),
                warmup_steps=cfg.warmup_steps,
            )
        elif scheduler_type == "none":
            # No scheduler - constant learning rate
            sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda step: 1.0)
        else:
            raise ValueError(
                f"Unknown scheduler_type: '{scheduler_type}'. "
                f"Must be 'plateau', 'cosine', 'transformer', 'noam', or 'none'."
            )
    loss_fn = (
        criterion
        if criterion is not None
        else nn.CrossEntropyLoss(
            ignore_index=cfg.pad_idx, label_smoothing=cfg.label_smoothing
        )
    )

    # Mixed-precision dtype. Prefer bfloat16 where supported: it has the same
    # exponent range as float32, so activations/attention scores can't overflow
    # to inf the way they can in float16 (a common cause of transformer training
    # suddenly diverging to NaN). float16 needs loss scaling (GradScaler);
    # bfloat16 does not, so the scaler is only enabled for the float16 path.
    if not use_amp:
        amp_dtype = torch.float16  # unused; autocast is disabled
    elif device.type == "cuda":
        amp_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    else:
        amp_dtype = torch.bfloat16  # CPU autocast supports bfloat16
    use_scaler = use_amp and amp_dtype == torch.float16

    scaler = torch.amp.GradScaler(device=device, enabled=use_scaler)
    skipped_steps = 0
    best_val = float("inf")
    best_path: Path | None = None
    train_losses: list[float] = []
    val_losses: list[float] = []
    global_step = 0
    stop_training = False
    no_improve_steps = 0
    # Accumulates loss over the current logging window. Unlike the per-epoch
    # totals below, this is reset at each log point (not at each epoch), so the
    # logged "train/loss" curve is continuous across epoch boundaries instead of
    # showing a sawtooth drop whenever a per-epoch running average resets.
    window_loss_sum = 0.0
    window_loss_count = 0

    # Initialize TensorBoard writer if enabled
    writer = None
    if cfg.use_tensorboard and SummaryWriter is not None:
        tb_dir = cfg.tensorboard_dir / cfg.experiment_name
        tb_dir.mkdir(parents=True, exist_ok=True)
        writer = SummaryWriter(log_dir=str(tb_dir))

    for epoch in range(num_epochs):
        model.train()
        total_train = 0.0
        steps_in_epoch = 0
        num_batches = len(train_loader)
        # Gradients accumulate across micro-batches, so zero them once before the
        # accumulation window rather than before every batch.
        opt.zero_grad()
        # Use tqdm to present a progress bar for steps in the epoch
        for step, (src, tgt) in enumerate(
            tqdm(train_loader, desc=f"Epoch {epoch + 1}", total=num_batches),
            start=1,
        ):
            src = src.to(device)
            tgt = tgt.to(device)
            tgt_input = tgt[:, :-1]
            tgt_output = tgt[:, 1:]

            with torch.amp.autocast(
                device_type=device.type, dtype=amp_dtype, enabled=use_amp
            ):
                logits = model(src, tgt_input)
                loss = loss_fn(
                    logits.reshape(-1, logits.size(-1)), tgt_output.reshape(-1)
                )

            # Report the true per-batch loss, but divide before backward so that
            # the gradients summed over accumulation_steps micro-batches equal the
            # gradient of one effective batch.
            batch_loss = loss.item()
            scaler.scale(loss / accumulation_steps).backward()

            total_train += batch_loss
            window_loss_sum += batch_loss
            window_loss_count += 1
            steps_in_epoch += 1

            # Step the optimizer once per accumulation window, and always on the
            # final batch so no accumulated gradients are left unapplied.
            took_step = step % accumulation_steps == 0 or step == num_batches
            if took_step:
                grad_finite = True
                if gradient_clip is not None:
                    scaler.unscale_(opt)
                    grad_norm = clip_grad_norm_(model.parameters(), gradient_clip)
                    # A non-finite gradient (transient overflow or a bad batch)
                    # would otherwise corrupt the weights and turn the whole run
                    # into NaN. Skip the update instead. The float16 GradScaler
                    # also skips internally; this additionally protects bf16/fp32.
                    grad_finite = bool(torch.isfinite(grad_norm))
                if grad_finite:
                    scaler.step(opt)
                else:
                    skipped_steps += 1
                scaler.update()
                opt.zero_grad()

                # Step the learning rate scheduler (skip for plateau - it steps on val loss)
                if not is_plateau_scheduler:
                    sched.step()

                global_step += 1
                # stop when either explicit step_limit or config.num_steps reached
                if (
                    getattr(cfg, "step_limit", None) is not None
                    and global_step >= cfg.step_limit
                ):
                    stop_training = True
                if (
                    getattr(cfg, "num_steps", None) is not None
                    and cfg.num_steps
                    and global_step >= cfg.num_steps
                ):
                    stop_training = True

                # Periodic validation (counted in optimizer steps)
                if (
                    val_loader is not None
                    and getattr(cfg, "val_interval", None)
                    and global_step % cfg.val_interval == 0
                ):
                    model.eval()
                    total_val = 0.0
                    with torch.no_grad():
                        for v_src, v_tgt in val_loader:
                            v_src = v_src.to(device)
                            v_tgt = v_tgt.to(device)
                            v_logits = model(v_src, v_tgt[:, :-1])
                            v_loss = loss_fn(
                                v_logits.reshape(-1, v_logits.size(-1)),
                                v_tgt[:, 1:].reshape(-1),
                            )
                            total_val += v_loss.item()
                    avg_val = total_val / max(1, len(val_loader))
                    val_losses.append(avg_val)
                    # Step plateau scheduler on validation loss
                    if is_plateau_scheduler:
                        sched.step(avg_val)
                    # early stopping logic (patience)
                    if avg_val < best_val:
                        best_val = avg_val
                        no_improve_steps = 0
                        # save best checkpoint with full training state
                        state = {
                            "epoch": epoch,
                            "global_step": global_step,
                            "model_state_dict": model.state_dict(),
                            "optimizer_state_dict": opt.state_dict(),
                            "scheduler_state_dict": sched.state_dict(),
                            "val_loss": avg_val,
                            "train_losses": train_losses,
                            "val_losses": val_losses,
                        }
                        if save_dir is not None:
                            save_dir.mkdir(parents=True, exist_ok=True)
                            best_path = Path(save_dir) / "model_best.pt"
                            torch.save(state, best_path)
                        else:
                            cfg.checkpoint_path.parent.mkdir(
                                parents=True, exist_ok=True
                            )
                            torch.save(state, cfg.checkpoint_path)
                    else:
                        no_improve_steps += 1
                        if no_improve_steps >= getattr(cfg, "patience", 0):
                            stop_training = True
                    if writer is not None:
                        writer.add_scalar("val/loss", avg_val, global_step)
                    model.train()

                # Periodic save of last checkpoint with full training state
                if (
                    getattr(cfg, "save_interval", None)
                    and cfg.save_interval
                    and global_step % cfg.save_interval == 0
                ):
                    state = {
                        "epoch": epoch,
                        "global_step": global_step,
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": opt.state_dict(),
                        "scheduler_state_dict": sched.state_dict(),
                        "train_losses": train_losses,
                        "val_losses": val_losses,
                    }
                    if save_dir is not None:
                        save_dir.mkdir(parents=True, exist_ok=True)
                        last_path = Path(save_dir) / "model_last.pt"
                        torch.save(state, last_path)
                        best_path = last_path if best_path is None else best_path
                    else:
                        cfg.last_checkpoint_path.parent.mkdir(
                            parents=True, exist_ok=True
                        )
                        torch.save(state, cfg.last_checkpoint_path)

            if stop_training:
                break

            # logging: prefer explicit function arg, fall back to config
            effective_log = log_every if log_every else getattr(cfg, "log_interval", 0)
            if effective_log and step % effective_log == 0:
                # Mean batch loss over the steps since the last log. This window
                # spans epoch boundaries, so the curve reflects the true recent
                # training loss rather than an epoch-cumulative average.
                window_loss = window_loss_sum / max(1, window_loss_count)
                window_loss_sum = 0.0
                window_loss_count = 0
                print(
                    f"Epoch {epoch + 1} Step {step}/{num_batches} | Train Loss: {window_loss:.4f}"
                )
                if writer is not None:
                    writer.add_scalar("train/loss", window_loss, global_step)
                    # Log learning rate from optimizer
                    for param_group in opt.param_groups:
                        writer.add_scalar(
                            "train/learning_rate", param_group["lr"], global_step
                        )

        if steps_in_epoch == 0:
            warnings.warn(
                f"Epoch {epoch + 1} ran 0 training steps because train_loader "
                f"yielded no batches. This usually means batch_size is larger "
                f"than the dataset (incomplete batches are dropped when "
                f"bucketing is enabled). Use a smaller batch_size or provide "
                f"more training data.",
                RuntimeWarning,
                stacklevel=2,
            )

        # Average over steps actually run in this epoch
        avg_train = total_train / max(1, steps_in_epoch)
        train_losses.append(avg_train)
        if writer is not None:
            writer.add_scalar("train/epoch_loss", avg_train, epoch)

        if val_loader is None:
            print(f"Epoch {epoch + 1}/{num_epochs} | Train: {avg_train:.4f}")
            if stop_training:
                break
            continue

        # Epoch-end validation (also used to update early-stopping state)
        model.eval()
        total_val = 0.0
        with torch.no_grad():
            for src, tgt in val_loader:
                src = src.to(device)
                tgt = tgt.to(device)
                logits = model(src, tgt[:, :-1])
                loss = loss_fn(
                    logits.reshape(-1, logits.size(-1)), tgt[:, 1:].reshape(-1)
                )
                total_val += loss.item()
        avg_val = total_val / max(1, len(val_loader))
        val_losses.append(avg_val)

        print(
            f"Epoch {epoch + 1}/{num_epochs} | Train: {avg_train:.4f} | Val: {avg_val:.4f}"
        )
        if writer is not None:
            writer.add_scalar("val/epoch_loss", avg_val, epoch)

        # Step plateau scheduler on epoch-end validation loss
        if is_plateau_scheduler:
            sched.step(avg_val)

        if avg_val < best_val:
            best_val = avg_val
            no_improve_steps = 0
            # save best checkpoint with full training state
            if save_dir is not None:
                save_dir.mkdir(parents=True, exist_ok=True)
                best_path = Path(save_dir) / "model_best.pt"
                torch.save(
                    {
                        "epoch": epoch,
                        "global_step": global_step,
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": opt.state_dict(),
                        "scheduler_state_dict": sched.state_dict(),
                        "val_loss": avg_val,
                        "train_losses": train_losses,
                        "val_losses": val_losses,
                    },
                    best_path,
                )
                print("  -> Saved best checkpoint")
            else:
                cfg.checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
                torch.save(
                    {
                        "epoch": epoch,
                        "global_step": global_step,
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": opt.state_dict(),
                        "scheduler_state_dict": sched.state_dict(),
                        "val_loss": avg_val,
                    },
                    cfg.checkpoint_path,
                )
        else:
            no_improve_steps += 1
            if no_improve_steps >= getattr(cfg, "patience", 0):
                stop_training = True

        if stop_training:
            print(
                "Reached stopping condition (step limit or patience). Stopping training."
            )
            break

    if writer is not None:
        writer.close()

    if skipped_steps:
        warnings.warn(
            f"Skipped {skipped_steps} optimizer step(s) due to non-finite "
            f"gradients. Training continued, but repeated skips indicate "
            f"instability — consider lowering the learning rate, increasing "
            f"warmup, or reducing the model size.",
            RuntimeWarning,
            stacklevel=2,
        )

    return TrainResult(
        train_losses=train_losses, val_losses=val_losses, best_checkpoint=best_path
    )


__all__ = ["TrainResult", "train_model"]
