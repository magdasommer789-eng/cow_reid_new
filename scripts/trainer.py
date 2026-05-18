"""
Training Loop for Video-Based Cow Re-Identification
====================================================

Implements metric learning training with Batch Hard Triplet Loss.

Training strategy:
  - P×K sampled batches (P identities × K clips each)
  - Batch Hard Triplet Loss on L2-normalised embeddings
  - Cosine annealing learning rate schedule with warmup
  - Periodic validation using a simplified re-ID distance check
  - TensorBoard logging of loss, distance statistics, and learning rate

Educational Note — Metric Learning Training Loop
-------------------------------------------------
Unlike classification, there is no "accuracy" to track during training.
Instead we monitor:
  1. Triplet loss value (lower is better, but 0 means all triplets are
     already satisfied — not necessarily a good sign if margin is too small)
  2. mean_pos_d:  mean distance to hardest positive  (should decrease)
  3. mean_neg_d:  mean distance to hardest negative  (should increase)
  4. frac_active: fraction of non-zero loss triplets (ideally 20-50%)
  5. Rank-1 accuracy on a small validation set (if available)
"""

import os
import time
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, BatchSampler
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from pathlib import Path
from typing import Dict, Optional, Tuple

from .losses import BatchHardTripletLoss
from .dataset import PKSampler


# ─────────────────────────────────────────────────────────────────────────────
# Learning-rate warmup + cosine annealing scheduler
# ─────────────────────────────────────────────────────────────────────────────

class WarmupCosineScheduler:
    """
    Linear warmup for the first `warmup_epochs` epochs, then cosine decay.

    During warmup the lr increases linearly from 0 to `base_lr`.
    After warmup it follows a cosine curve down to `min_lr`.

    Args:
        optimizer:      PyTorch optimiser.
        base_lr:        Peak learning rate (reached at end of warmup).
        warmup_epochs:  Number of warmup epochs.
        total_epochs:   Total training epochs.
        min_lr:         Minimum LR at end of cosine decay.
    """

    def __init__(
        self,
        optimizer:     optim.Optimizer,
        base_lr:       float,
        warmup_epochs: int,
        total_epochs:  int,
        min_lr:        float = 1e-6,
    ):
        self.optimizer     = optimizer
        self.base_lr       = base_lr
        self.warmup_epochs = warmup_epochs
        self.total_epochs  = total_epochs
        self.min_lr        = min_lr

    def step(self, epoch: int):
        """Call once per epoch (0-indexed)."""
        import math
        if epoch < self.warmup_epochs:
            lr = self.base_lr * (epoch + 1) / self.warmup_epochs
        else:
            progress = (epoch - self.warmup_epochs) / max(
                1, self.total_epochs - self.warmup_epochs
            )
            lr = self.min_lr + 0.5 * (self.base_lr - self.min_lr) * (
                1.0 + math.cos(math.pi * progress)
            )
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = lr
        return lr


# ─────────────────────────────────────────────────────────────────────────────
# Trainer
# ─────────────────────────────────────────────────────────────────────────────

class ReIDTrainer:
    """
    Full training + validation loop for one re-ID model.

    Args:
        model:          The embedding model (any of C3D/X3D/Swin/ViViT).
        train_dataset:  VideoClipDataset for the training cows.
        config:         Dictionary of training hyperparameters (from config.yaml).
        model_name:     String identifier for logging/checkpointing.
        device:         "cuda" or "cpu".
        log_dir:        TensorBoard log directory.
        checkpoint_dir: Directory for saving .pt checkpoints.
    """

    def __init__(
        self,
        model:          nn.Module,
        train_dataset,
        config:         Dict,
        model_name:     str,
        device:         str = "cuda",
        log_dir:        str = "./logs",
        checkpoint_dir: str = "./checkpoints",
    ):
        self.model          = model.to(device)
        self.train_dataset  = train_dataset
        self.config         = config
        self.model_name     = model_name
        self.device         = device
        self.log_dir        = Path(log_dir) / model_name
        self.checkpoint_dir = Path(checkpoint_dir)

        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        tc = config["training"]
        self.num_epochs    = tc["num_epochs"]
        self.P             = tc["P"]
        self.K             = tc["K"]
        self.save_interval = config["logging"]["save_interval"]
        self.patience      = tc["early_stopping_patience"]

        # Loss
        self.criterion = BatchHardTripletLoss(
            margin      = tc["triplet_margin"],
            distance    = "euclidean",
            soft_margin = False,
        )

        # Optimiser
        self.optimizer = optim.Adam(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr           = tc["learning_rate"],
            weight_decay = tc["weight_decay"],
        )

        # LR scheduler
        self.scheduler = WarmupCosineScheduler(
            optimizer     = self.optimizer,
            base_lr       = tc["learning_rate"],
            warmup_epochs = tc["warmup_epochs"],
            total_epochs  = self.num_epochs,
        )

        # TensorBoard writer
        self.writer = SummaryWriter(log_dir=str(self.log_dir))

        # State tracking
        self.best_loss  = float("inf")
        self.no_improve = 0
        self.history    = []

    def _build_loader(self) -> DataLoader:
        """Build a DataLoader with PK batch sampler."""
        # PK sampler yields P*K indices per batch
        num_batches = max(
            50,
            len(self.train_dataset) // (self.P * self.K),
        )
        pk_sampler = PKSampler(
            label_to_indices = self.train_dataset.label_to_indices,
            P                = self.P,
            K                = self.K,
            num_batches      = num_batches,
        )
        # BatchSampler wraps pk_sampler: each "batch" is already P*K elements
        batch_sampler = BatchSampler(pk_sampler, batch_size=self.P * self.K,
                                     drop_last=False)
        return DataLoader(
            self.train_dataset,
            batch_sampler = batch_sampler,
            num_workers   = 0,  # numpy 2.x: num_workers>0 breaks in worker subprocesses
            pin_memory    = (self.device == "cuda"),
        )

    def _train_one_epoch(self, loader: DataLoader, epoch: int) -> Dict:
        """Run one full training epoch and return aggregated metrics."""
        self.model.train()

        total_loss    = 0.0
        total_pos_d   = 0.0
        total_neg_d   = 0.0
        total_active  = 0.0
        num_batches   = 0

        pbar = tqdm(loader, desc=f"[{self.model_name}] Epoch {epoch+1:03d}",
                    leave=False)

        for clips, labels in pbar:
            clips  = clips.to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True)

            self.optimizer.zero_grad()

            embeddings = self.model(clips)                       # (N, D)
            loss, info = self.criterion(embeddings, labels)

            loss.backward()
            # Gradient clipping prevents exploding gradients in transformers
            nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()

            total_loss   += info["loss"]
            total_pos_d  += info["mean_pos_d"]
            total_neg_d  += info["mean_neg_d"]
            total_active += info["frac_active"]
            num_batches  += 1

            pbar.set_postfix(
                loss     = f"{info['loss']:.4f}",
                pos_d    = f"{info['mean_pos_d']:.3f}",
                neg_d    = f"{info['mean_neg_d']:.3f}",
                active   = f"{info['frac_active']:.0%}",
            )

        n = max(1, num_batches)
        return {
            "loss":       total_loss   / n,
            "mean_pos_d": total_pos_d  / n,
            "mean_neg_d": total_neg_d  / n,
            "frac_active":total_active / n,
        }

    def _log_epoch(self, metrics: Dict, epoch: int, lr: float):
        """Write metrics to TensorBoard."""
        for k, v in metrics.items():
            self.writer.add_scalar(f"train/{k}", v, epoch)
        self.writer.add_scalar("train/lr", lr, epoch)

    def _save_checkpoint(self, epoch: int, metrics: Dict, is_best: bool = False):
        """Save model weights to disk."""
        state = {
            "epoch":      epoch,
            "model_name": self.model_name,
            "model_state":self.model.state_dict(),
            "optim_state":self.optimizer.state_dict(),
            "metrics":    metrics,
        }
        path = self.checkpoint_dir / f"{self.model_name}_epoch_{epoch:03d}.pt"
        torch.save(state, path)
        if is_best:
            best_path = self.checkpoint_dir / f"{self.model_name}_best.pt"
            torch.save(state, best_path)
            print(f"  ✓ Best checkpoint saved: {best_path}")

    def train(self) -> Dict:
        """
        Run the full training loop.

        Returns:
            Dictionary of training history (one entry per epoch).
        """
        print(f"\n{'='*60}")
        print(f"Training {self.model_name.upper()}")
        print(f"  Epochs:  {self.num_epochs}")
        print(f"  P×K:     {self.P}×{self.K}  (batch size = {self.P*self.K})")
        print(f"  Device:  {self.device}")
        print(f"{'='*60}\n")

        loader = self._build_loader()

        for epoch in range(self.num_epochs):
            t0 = time.time()

            # Update learning rate
            lr = self.scheduler.step(epoch)

            # Training epoch
            metrics = self._train_one_epoch(loader, epoch)
            metrics["epoch"] = epoch
            metrics["lr"]    = lr

            elapsed = time.time() - t0
            print(
                f"Epoch {epoch+1:03d}/{self.num_epochs}  "
                f"loss={metrics['loss']:.4f}  "
                f"pos_d={metrics['mean_pos_d']:.3f}  "
                f"neg_d={metrics['mean_neg_d']:.3f}  "
                f"active={metrics['frac_active']:.0%}  "
                f"lr={lr:.2e}  "
                f"({elapsed:.1f}s)"
            )

            # Logging
            self._log_epoch(metrics, epoch, lr)
            self.history.append(metrics)

            # Checkpoint
            is_best = metrics["loss"] < self.best_loss
            if is_best:
                self.best_loss = metrics["loss"]
                self.no_improve = 0
            else:
                self.no_improve += 1

            if (epoch + 1) % self.save_interval == 0 or is_best:
                self._save_checkpoint(epoch, metrics, is_best=is_best)

            # Early stopping
            if self.no_improve >= self.patience:
                print(f"\nEarly stopping at epoch {epoch+1} "
                      f"(no improvement for {self.patience} epochs).")
                break

        self.writer.close()

        # Save training history as JSON for later analysis
        history_path = self.checkpoint_dir / f"{self.model_name}_history.json"
        with open(history_path, "w") as f:
            json.dump(self.history, f, indent=2)
        print(f"\nHistory saved to {history_path}")

        return self.history


# ─────────────────────────────────────────────────────────────────────────────
# Convenience: build a trainer from a config dict
# ─────────────────────────────────────────────────────────────────────────────

def build_trainer(
    model:         nn.Module,
    train_dataset,
    config:        Dict,
    model_name:    str,
) -> ReIDTrainer:
    """
    Instantiate a ReIDTrainer with settings from config.yaml.

    Args:
        model:         The embedding model.
        train_dataset: Training VideoClipDataset.
        config:        Parsed config.yaml dictionary.
        model_name:    Identifier string (e.g. "c3d").

    Returns:
        ReIDTrainer ready to call .train() on.
    """
    device = config["training"]["device"]
    return ReIDTrainer(
        model          = model,
        train_dataset  = train_dataset,
        config         = config,
        model_name     = model_name,
        device         = device,
        log_dir        = config["logging"]["log_dir"],
        checkpoint_dir = config["logging"]["checkpoint_dir"],
    )
