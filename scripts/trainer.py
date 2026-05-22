"""
Trainer v2 — Video-Based Cow Re-Identification
===============================================

Trains one model with metric learning (Batch Hard Triplet Loss).
After every epoch the val set is evaluated with the same CMC/re-ID protocol
used at test time: first clip per cow = query, rest = gallery.
Val Rank-1 drives early stopping and best-checkpoint selection.

WandB: one run per training session.  Pass an active wandb.Run object
       (or None to skip logging) via the *wandb_run* argument.

Learning curves: saved as PNG to results/{model}_learning_curve.png.

Only ONE checkpoint is saved per model — the epoch with highest val Rank-1.
"""

import json
import math
import os
import time
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib
matplotlib.use("Agg")          # headless backend — no display needed
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from .dataset  import PKSampler, ReIDEvalDataset
from .losses   import BatchHardTripletLoss


# ─────────────────────────────────────────────────────────────────────────────
# Lightweight val re-ID evaluator  (called by trainer AND hpo.py)
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def run_reid_eval(
    model:    nn.Module,
    eval_ds:  ReIDEvalDataset,
    device:   str = "cuda",
    batch_sz: int = 8,
) -> float:
    """
    Run CMC evaluation and return Rank-1 accuracy.

    Protocol (identical for val and test):
      - Query  clips: first clip per cow  (role == "query")
      - Gallery clips: all remaining clips (role == "gallery")
      - Distance metric: Euclidean on L2-normalised embeddings.

    Returns val Rank-1 in [0, 1].
    """
    model.eval()
    loader = DataLoader(eval_ds, batch_size=batch_sz,
                        shuffle=False, num_workers=0)

    all_embs:     List[np.ndarray] = []
    all_cow_ids:  List[str]        = []
    all_roles:    List[str]        = []

    for clips, cow_ids, roles in loader:
        clips = clips.to(device, non_blocking=True)
        embs  = model(clips).cpu().float()
        all_embs.append(np.array(embs.tolist(), dtype=np.float32))
        all_cow_ids.extend(list(cow_ids))
        all_roles.extend(list(roles))

    model.train()

    if not all_embs:
        return 0.0

    embeddings = np.concatenate(all_embs, axis=0)   # (N, D)
    roles_arr  = np.array(all_roles)
    ids_arr    = np.array(all_cow_ids)

    # Gallery: mean-pool all gallery clips per cow → one embedding per identity
    g_mask   = roles_arr == "gallery"
    g_embs   = embeddings[g_mask]
    g_ids    = ids_arr[g_mask]
    unique_g = sorted(set(g_ids.tolist()))

    if not unique_g:
        return 0.0

    agg_embs = []
    for cid in unique_g:
        idx      = g_ids == cid
        mean_emb = g_embs[idx].mean(axis=0)
        mean_emb = mean_emb / (np.linalg.norm(mean_emb) + 1e-12)
        agg_embs.append(mean_emb)
    gallery_embs = np.stack(agg_embs, axis=0)   # (G, D)

    # Query embeddings
    q_mask   = roles_arr == "query"
    q_embs   = embeddings[q_mask]
    q_ids    = ids_arr[q_mask].tolist()

    if len(q_ids) == 0:
        return 0.0

    # Euclidean distance matrix  (Q, G)
    q_sq  = (q_embs ** 2).sum(1, keepdims=True)
    g_sq  = (gallery_embs ** 2).sum(1, keepdims=True).T
    dot   = q_embs @ gallery_embs.T
    dist  = np.sqrt(np.clip(q_sq + g_sq - 2 * dot, 0, None))

    # Rank-1 accuracy
    gallery_arr = np.array(unique_g)
    rank1_hits  = 0
    for i, qid in enumerate(q_ids):
        top1_id = gallery_arr[np.argmin(dist[i])]
        if top1_id == qid:
            rank1_hits += 1

    return rank1_hits / len(q_ids)


# ─────────────────────────────────────────────────────────────────────────────
# Learning-rate scheduler (warmup → cosine)
# ─────────────────────────────────────────────────────────────────────────────

class WarmupCosineScheduler:
    def __init__(
        self,
        optimizer:     torch.optim.Optimizer,
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

    def step(self, epoch: int) -> float:
        if epoch < self.warmup_epochs:
            lr = self.base_lr * (epoch + 1) / max(1, self.warmup_epochs)
        else:
            progress = (epoch - self.warmup_epochs) / max(
                1, self.total_epochs - self.warmup_epochs
            )
            lr = self.min_lr + 0.5 * (self.base_lr - self.min_lr) * (
                1.0 + math.cos(math.pi * progress)
            )
        for pg in self.optimizer.param_groups:
            pg["lr"] = lr
        return lr


# ─────────────────────────────────────────────────────────────────────────────
# Main trainer
# ─────────────────────────────────────────────────────────────────────────────

class ReIDTrainer:
    """
    Full training loop for one re-ID model.

    Args:
        model:          Embedding model (C3D / X3D / Swin / ViViT).
        train_ds:       VideoClipDataset.
        val_ds:         ReIDEvalDataset for val evaluation after each epoch.
        cfg:            Config dict.
        model_name:     "c3d" | "x3d" | "swin" | "vivit".
        hparams:        Hyperparameter dict (overrides cfg defaults if given).
        wandb_run:      Active wandb.Run or None.
        is_final_train: If True (train+val combined) use loss plateau stopping.
    """

    def __init__(
        self,
        model:          nn.Module,
        train_ds,
        val_ds:         Optional[ReIDEvalDataset],
        cfg:            Dict,
        model_name:     str,
        hparams:        Optional[Dict] = None,
        wandb_run       = None,
        is_final_train: bool = False,
    ):
        self.model          = model
        self.train_ds       = train_ds
        self.val_ds         = val_ds
        self.cfg            = cfg
        self.model_name     = model_name
        self.wandb_run      = wandb_run
        self.is_final_train = is_final_train

        tc   = cfg["training"]
        hp   = hparams or {}

        self.lr           = hp.get("lr",           tc["learning_rate"])
        self.weight_decay = hp.get("weight_decay",  tc["weight_decay"])
        self.P            = hp.get("P",             tc["P"])
        self.K            = hp.get("K",             tc["K"])
        self.margin       = hp.get("triplet_margin",tc["triplet_margin"])
        self.embed_dim    = hp.get("embedding_dim", cfg["model"]["embedding_dim"])
        self.device       = tc["device"]
        self.num_epochs   = tc["num_epochs"]
        self.patience     = tc["early_stopping_patience"]
        self.warmup       = tc.get("warmup_epochs", 3)
        self.batch_ep     = tc.get("batches_per_epoch", 30)

        self.ckpt_dir  = Path(cfg["logging"]["checkpoint_dir"])
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)
        self.res_dir   = Path(cfg["logging"]["results_dir"])
        self.res_dir.mkdir(parents=True, exist_ok=True)

        # Build training infrastructure
        self.criterion = BatchHardTripletLoss(margin=self.margin)
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.lr, weight_decay=self.weight_decay,
        )
        self.scheduler = WarmupCosineScheduler(
            self.optimizer,
            base_lr       = self.lr,
            warmup_epochs = self.warmup,
            total_epochs  = self.num_epochs,
        )

        sampler    = PKSampler(
            labels            = train_ds.labels,
            P                 = self.P,
            K                 = self.K,
            batches_per_epoch = self.batch_ep,
        )
        self.loader = DataLoader(
            train_ds,
            batch_sampler = sampler,
            num_workers   = 2,
            pin_memory    = (self.device == "cuda"),
        )

        # History for learning-curve plot
        self.history: Dict[str, List] = {
            "epoch":      [],
            "train_loss": [],
            "val_rank1":  [],
            "lr":         [],
        }
        self.best_val_rank1 = 0.0
        self.best_epoch     = 0

    # ── Training loop ─────────────────────────────────────────────────────────

    def train(self) -> Dict:
        """Run the full training loop.  Returns the best history snapshot."""
        print(f"\n{'─'*60}")
        print(f"  Training {self.model_name.upper()}")
        print(f"  lr={self.lr:.2e}  wd={self.weight_decay:.2e}  "
              f"margin={self.margin:.2f}  P={self.P}  K={self.K}")
        print(f"{'─'*60}")

        no_improve = 0

        for epoch in range(self.num_epochs):
            lr = self.scheduler.step(epoch)

            # ── epoch ──────────────────────────────────────────────────────
            train_loss = self._train_epoch()

            # ── val eval ───────────────────────────────────────────────────
            val_rank1 = 0.0
            if self.val_ds is not None:
                val_rank1 = run_reid_eval(
                    self.model, self.val_ds,
                    device=self.device, batch_sz=self.cfg["training"]["batch_size"]
                )

            # ── record ─────────────────────────────────────────────────────
            self.history["epoch"].append(epoch)
            self.history["train_loss"].append(train_loss)
            self.history["val_rank1"].append(val_rank1)
            self.history["lr"].append(lr)

            print(f"  Epoch {epoch+1:3d}/{self.num_epochs}  "
                  f"loss={train_loss:.4f}  "
                  f"val_rank1={val_rank1:.4f}  lr={lr:.2e}")

            # ── WandB log ───────────────────────────────────────────────────
            if self.wandb_run is not None:
                import wandb
                wandb.log({
                    "epoch":      epoch,
                    "train/loss": train_loss,
                    "val/rank1":  val_rank1,
                    "lr":         lr,
                })

            # ── best checkpoint ─────────────────────────────────────────────
            # Prefer val Rank-1 whenever it is available (val_ds provided).
            # Fall back to training loss only when there is no val set at all.
            has_val_signal = self.val_ds is not None
            if has_val_signal:
                # Use val Rank-1 for both final and regular training
                improved = val_rank1 > self.best_val_rank1
            else:
                # Truly no val: save on training-loss improvement
                improved = (epoch == 0 or
                            train_loss < min(self.history["train_loss"][:-1]))

            if improved:
                self.best_val_rank1 = val_rank1
                self.best_epoch     = epoch
                self._save_checkpoint(epoch, val_rank1, train_loss)
                no_improve = 0
            else:
                no_improve += 1

            # ── early stopping ──────────────────────────────────────────────
            # For final training: stop on loss plateau (val Rank-1 hits ceiling
            # quickly with only 6 val cows, so it's not a reliable stop signal).
            if self.is_final_train:
                loss_plateau = (epoch > 0 and
                                train_loss >= min(self.history["train_loss"][:-1]))
                if loss_plateau:
                    no_improve_loss = getattr(self, "_no_improve_loss", 0) + 1
                else:
                    no_improve_loss = 0
                self._no_improve_loss = no_improve_loss
                if no_improve_loss >= self.patience:
                    print(f"  Early stopping at epoch {epoch+1} "
                          f"(loss plateau for {self.patience} epochs).")
                    break
            else:
                if no_improve >= self.patience:
                    print(f"  Early stopping at epoch {epoch+1} "
                          f"(val Rank-1 no improvement for {self.patience} epochs).")
                    break

        self._plot_learning_curve()

        # Save training history
        hist_path = self.res_dir / f"{self.model_name}_history.json"
        with open(hist_path, "w") as f:
            json.dump(self.history, f, indent=2)

        print(f"\n  Best epoch: {self.best_epoch+1}  "
              f"val_rank1={self.best_val_rank1:.4f}")
        return self.history

    # ── Single epoch ─────────────────────────────────────────────────────────

    def _train_epoch(self) -> float:
        self.model.train()
        total_loss = 0.0
        n_batches  = 0

        for clips, labels in self.loader:
            clips  = clips.to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True)

            self.optimizer.zero_grad()
            embs = self.model(clips)
            loss, _ = self.criterion(embs, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 5.0)
            self.optimizer.step()

            total_loss += loss.item()
            n_batches  += 1

        return total_loss / max(n_batches, 1)

    # ── Checkpoint (best only) ────────────────────────────────────────────────

    def _save_checkpoint(self, epoch: int, val_rank1: float, train_loss: float):
        ckpt_path = self.ckpt_dir / f"{self.model_name}_best.pt"
        torch.save({
            "epoch":       epoch,
            "model_state": self.model.state_dict(),
            "val_rank1":   val_rank1,
            "train_loss":  train_loss,
            "hparams": {
                "lr":            self.lr,
                "weight_decay":  self.weight_decay,
                "P":             self.P,
                "K":             self.K,
                "margin":        self.margin,
                "embedding_dim": self.embed_dim,
            },
        }, ckpt_path)
        print(f"    ✓ Saved best checkpoint (epoch {epoch+1}, "
              f"val_rank1={val_rank1:.4f})")

    # ── Learning curve ────────────────────────────────────────────────────────

    def _plot_learning_curve(self):
        epochs     = self.history["epoch"]
        train_loss = self.history["train_loss"]
        val_rank1  = self.history["val_rank1"]
        if not epochs:
            return

        has_val = any(v > 0 for v in val_rank1)

        fig, ax1 = plt.subplots(figsize=(10, 5))
        title = (f"{self.model_name.upper()} — Learning Curve"
                 f"  (final train)" if self.is_final_train else
                 f"{self.model_name.upper()} — Learning Curve")
        fig.suptitle(title, fontsize=13)

        # ── Training loss on left y-axis ──────────────────────────────────────
        color_loss = "#2196F3"   # blue
        ax1.set_xlabel("Epoch")
        ax1.set_ylabel("Train Loss (Triplet)", color=color_loss)
        ax1.plot(epochs, train_loss, color=color_loss,
                 linewidth=2, marker="o", markersize=3, label="Train loss")
        ax1.tick_params(axis="y", labelcolor=color_loss)
        ax1.grid(True, alpha=0.25)

        # ── Val Rank-1 on right y-axis ────────────────────────────────────────
        if has_val:
            color_val = "#F44336"    # red
            ax2 = ax1.twinx()
            ax2.set_ylabel("Val Rank-1 (CMC@1)", color=color_val)
            ax2.plot(epochs, val_rank1, color=color_val,
                     linewidth=2, marker="s", markersize=3, label="Val Rank-1")
            ax2.set_ylim(0, 1.05)
            ax2.tick_params(axis="y", labelcolor=color_val)

            # Mark the optimal epoch (best val Rank-1 = lowest loss if final)
            best_ep = self.history["epoch"][self.best_epoch]
            best_v  = val_rank1[self.best_epoch]
            ax2.axvline(x=best_ep, color="green", linestyle="--",
                        linewidth=1.5, alpha=0.8,
                        label=f"Best epoch {best_ep+1} (Rank-1={best_v:.2f})")

            # Combined legend
            lines1, labels1 = ax1.get_legend_handles_labels()
            lines2, labels2 = ax2.get_legend_handles_labels()
            ax1.legend(lines1 + lines2, labels1 + labels2,
                       loc="upper right", fontsize=9)
        else:
            # Final training with no val — mark lowest-loss epoch
            best_ep  = self.history["epoch"][self.best_epoch]
            best_lo  = train_loss[self.best_epoch]
            ax1.axvline(x=best_ep, color="green", linestyle="--",
                        linewidth=1.5, alpha=0.8,
                        label=f"Best epoch {best_ep+1} (loss={best_lo:.4f})")
            ax1.legend(loc="upper right", fontsize=9)
            ax1.annotate("No val set — trained on train+val combined",
                         xy=(0.5, 0.96), xycoords="axes fraction",
                         ha="center", fontsize=9, color="gray")

        plt.tight_layout()
        out = self.res_dir / f"{self.model_name}_learning_curve.png"
        plt.savefig(out, dpi=130, bbox_inches="tight")
        plt.close(fig)
        print(f"  Learning curve → {out}")


# ─────────────────────────────────────────────────────────────────────────────
# Factory
# ─────────────────────────────────────────────────────────────────────────────

def build_trainer(
    model:          nn.Module,
    train_ds,
    val_ds:         Optional[ReIDEvalDataset],
    cfg:            Dict,
    model_name:     str,
    hparams:        Optional[Dict] = None,
    wandb_run       = None,
    is_final_train: bool = False,
) -> ReIDTrainer:
    return ReIDTrainer(
        model          = model,
        train_ds       = train_ds,
        val_ds         = val_ds,
        cfg            = cfg,
        model_name     = model_name,
        hparams        = hparams,
        wandb_run      = wandb_run,
        is_final_train = is_final_train,
    )
