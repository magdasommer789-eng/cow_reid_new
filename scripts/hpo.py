"""
Hyperparameter Optimisation — Optuna + WandB
=============================================

For each model (C3D / X3D / Swin / ViViT) an Optuna study searches the
hyperparameter space with 30 trials.  Each trial trains for up to 20 epochs
with early stopping, evaluates on the val set using the same re-ID CMC
protocol as the final test, and returns val Rank-1 as the objective.

WandB integration:
  • One run per trial → group "hpo_{model_name}"
  • All 4 model HPO runs go to project "cow-reid"
  • Best params saved to results/{model}_best_hparams.json

Usage:
  python -m scripts.hpo --model c3d
  python -m scripts.hpo --all
"""

import argparse
import json
import os
from pathlib import Path
from typing import Dict, Optional

import optuna
import torch
import yaml

# suppress optuna info logs for cleaner output
optuna.logging.set_verbosity(optuna.logging.WARNING)


# ─────────────────────────────────────────────────────────────────────────────
# Hyperparameter search space
# ─────────────────────────────────────────────────────────────────────────────

def sample_hparams(trial: optuna.Trial, cfg: Dict, model_name: str) -> Dict:
    """Sample one set of hyperparameters from the search space."""
    h = cfg["hpo"]
    mc = cfg["model"][model_name]

    dropout_base = float(mc["dropout_rate"])
    dropout_choices = h.get("dropout_choices", [0.1, 0.2, 0.3, 0.5])

    return {
        "lr":           trial.suggest_float("lr",
                            h["lr_min"], h["lr_max"], log=True),
        "weight_decay": trial.suggest_float("weight_decay",
                            h["wd_min"], h["wd_max"], log=True),
        "dropout_rate": trial.suggest_categorical("dropout_rate",
                            dropout_choices),
        "triplet_margin": trial.suggest_float("triplet_margin",
                            h["margin_min"], h["margin_max"]),
        "P":              trial.suggest_categorical("P",   h["P_choices"]),
        "K":              trial.suggest_categorical("K",   h["K_choices"]),
        "embedding_dim":  trial.suggest_categorical("embedding_dim",
                            h["embed_dim_choices"]),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Build model with given hparams
# ─────────────────────────────────────────────────────────────────────────────

def build_model_with_hparams(
    model_name: str,
    hparams:    Dict,
    cfg:        Dict,
    device:     str,
):
    """Instantiate model using *hparams* overrides."""
    mc   = cfg["model"]
    edim = hparams["embedding_dim"]
    dr   = hparams["dropout_rate"]

    if model_name in ("c3d", "x3d"):
        from .models_cnn import create_cnn_model
        return create_cnn_model(
            model_name      = model_name,
            embedding_dim   = edim,
            pretrained      = mc[model_name]["pretrained"],
            freeze_backbone = mc[model_name]["freeze_backbone"],
            dropout_rate    = dr,
            device          = device,
            model_size      = mc["x3d"]["model_size"] if model_name == "x3d" else "m",
        )

    if model_name == "swin":
        from .models_transformer import create_transformer_model
        return create_transformer_model(
            model_name      = "swin",
            embedding_dim   = edim,
            pretrained      = mc["swin"]["pretrained"],
            freeze_backbone = mc["swin"]["freeze_backbone"],
            dropout_rate    = dr,
            device          = device,
            swin_variant    = mc["swin"]["model_name"],
        )

    if model_name == "vivit":
        from .models_transformer import create_transformer_model
        return create_transformer_model(
            model_name      = "vivit",
            embedding_dim   = edim,
            pretrained      = mc["vivit"]["pretrained"],
            freeze_backbone = mc["vivit"]["freeze_backbone"],
            dropout_rate    = dr,
            device          = device,
            num_frames      = cfg["data"].get("clip_frames", 16),
            tubelet_size    = mc["vivit"]["tubelet_size"],
        )

    raise ValueError(f"Unknown model: {model_name}")


# ─────────────────────────────────────────────────────────────────────────────
# Objective function (one Optuna trial)
# ─────────────────────────────────────────────────────────────────────────────

def make_objective(
    model_name: str,
    metadata:   Dict,
    cfg:        Dict,
    device:     str,
):
    """Return an Optuna objective function for *model_name*."""

    from .dataset    import build_train_dataset, build_eval_dataset, PKSampler
    from .losses     import BatchHardTripletLoss
    from .trainer    import run_reid_eval
    from torch.utils.data import DataLoader

    train_ds  = build_train_dataset(metadata, temporal_jitter=True)
    val_ds    = build_eval_dataset(metadata, split="val")

    use_wandb  = cfg["logging"].get("use_wandb", False)
    project    = cfg["logging"].get("wandb_project", "cow-reid")
    entity     = cfg["logging"].get("wandb_entity", None)
    results_dir = Path(cfg["logging"]["results_dir"])
    results_dir.mkdir(parents=True, exist_ok=True)

    def objective(trial: optuna.Trial) -> float:
        hparams = sample_hparams(trial, cfg, model_name)
        P       = hparams["P"]
        K       = hparams["K"]

        run = None
        if use_wandb:
            import wandb
            run = wandb.init(
                project  = project,
                entity   = entity,
                group    = f"hpo_{model_name}",
                name     = f"trial_{trial.number:03d}",
                config   = {**hparams, "model": model_name, "stage": "hpo"},
                reinit   = "finish_previous",
            )

        try:
            model = build_model_with_hparams(model_name, hparams, cfg, device)

            sampler = PKSampler(
                labels            = train_ds.labels,
                P                 = P,
                K                 = K,
                batches_per_epoch = cfg["training"].get("batches_per_epoch", 30),
            )
            loader = DataLoader(
                train_ds,
                batch_sampler = sampler,
                num_workers   = 2,
                pin_memory    = (device == "cuda"),
            )

            criterion = BatchHardTripletLoss(margin=hparams["triplet_margin"])
            optimizer = torch.optim.AdamW(
                model.parameters(),
                lr           = hparams["lr"],
                weight_decay = hparams["weight_decay"],
            )

            max_epochs = cfg["hpo"]["max_epochs_per_trial"]
            patience   = cfg["training"].get("early_stopping_patience", 5)
            best_rank1 = 0.0
            no_improve = 0

            for epoch in range(max_epochs):
                # ── train one epoch ──────────────────────────────────────────
                model.train()
                epoch_loss = 0.0
                for clips, labels in loader:
                    clips  = clips.to(device, non_blocking=True)
                    labels = labels.to(device, non_blocking=True)
                    optimizer.zero_grad()
                    embs = model(clips)
                    loss, _ = criterion(embs, labels)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                    optimizer.step()
                    epoch_loss += loss.item()

                avg_loss = epoch_loss / len(loader)

                # ── val CMC eval ─────────────────────────────────────────────
                val_rank1 = run_reid_eval(
                    model    = model,
                    eval_ds  = val_ds,
                    device   = device,
                    batch_sz = cfg["training"].get("batch_size", 8),
                )

                if run is not None:
                    import wandb
                    wandb.log({
                        "epoch":      epoch,
                        "train/loss": avg_loss,
                        "val/rank1":  val_rank1,
                    })

                # ── Optuna pruning ───────────────────────────────────────────
                trial.report(val_rank1, epoch)
                if trial.should_prune():
                    raise optuna.TrialPruned()

                # ── early stopping ───────────────────────────────────────────
                if val_rank1 > best_rank1:
                    best_rank1 = val_rank1
                    no_improve = 0
                else:
                    no_improve += 1
                    if no_improve >= patience:
                        break

            if run is not None:
                import wandb
                wandb.summary["best_val_rank1"] = best_rank1
                run.finish()

            # Free GPU memory between trials
            del model
            if device == "cuda":
                torch.cuda.empty_cache()

            return best_rank1

        except torch.cuda.OutOfMemoryError:
            # OOM: skip this trial cleanly so the study can continue
            if run is not None:
                run.finish()
            try:
                del model
            except Exception:
                pass
            torch.cuda.empty_cache()
            print(f"  OOM on trial {trial.number} (P={hparams.get('P')}, "
                  f"K={hparams.get('K')}) — skipping, returning 0.")
            return 0.0
        except optuna.TrialPruned:
            if run is not None:
                run.finish()
            raise
        except Exception as exc:
            if run is not None:
                run.finish()
            raise exc

    return objective


# ─────────────────────────────────────────────────────────────────────────────
# Run HPO for one model
# ─────────────────────────────────────────────────────────────────────────────

def run_hpo(
    model_name: str,
    metadata:   Dict,
    cfg:        Dict,
    device:     str,
) -> Dict:
    """
    Run Optuna HPO for *model_name*.  Saves best params to
    results/{model_name}_best_hparams.json and returns them.
    """
    h         = cfg["hpo"]
    n_trials  = h["n_trials"]
    results_dir = Path(cfg["logging"]["results_dir"])
    results_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"HPO: {model_name.upper()}  ({n_trials} trials)")
    print(f"{'='*60}")

    pruner = optuna.pruners.MedianPruner(
        n_startup_trials = h["pruner_startup_trials"],
        n_warmup_steps   = h["pruner_warmup_steps"],
    )
    study = optuna.create_study(
        direction  = "maximize",
        pruner     = pruner,
        study_name = f"cow_reid_{model_name}",
    )

    objective  = make_objective(model_name, metadata, cfg, device)
    out_path   = results_dir / f"{model_name}_best_hparams.json"

    def _save_best_callback(study: optuna.Study, trial: optuna.trial.FrozenTrial):
        """Save best params to disk after every improvement."""
        try:
            if study.best_trial.number == trial.number:
                bp = {**study.best_trial.params, "best_val_rank1": study.best_value}
                with open(out_path, "w") as fh:
                    json.dump(bp, fh, indent=2)
                print(f"  ✓ New best: trial {trial.number}  "
                      f"val_rank1={study.best_value:.4f}  → {out_path}")
        except Exception:
            pass   # don't let callback errors kill the study

    try:
        study.optimize(
            objective,
            n_trials          = n_trials,
            callbacks         = [_save_best_callback],
            show_progress_bar = True,
        )
    except Exception as e:
        print(f"  Study interrupted: {e}")
        print("  Saving best params found so far …")

    if not study.trials:
        print("  No trials completed — cannot save best params.")
        return {}

    best        = study.best_trial
    best_params = {**best.params, "best_val_rank1": best.value}

    with open(out_path, "w") as f:
        json.dump(best_params, f, indent=2)

    print(f"\n{'─'*50}")
    print(f"  Best trial #{best.number}  |  val Rank-1 = {best.value:.4f}")
    for k, v in best.params.items():
        print(f"    {k}: {v}")
    print(f"  Saved → {out_path}")

    # WandB summary run
    if cfg["logging"].get("use_wandb", False):
        try:
            import wandb
            run = wandb.init(
                project = cfg["logging"].get("wandb_project", "cow-reid"),
                entity  = cfg["logging"].get("wandb_entity", None),
                group   = f"hpo_{model_name}",
                name    = f"best_summary_{model_name}",
                config  = best_params,
                reinit  = "finish_previous",
            )
            wandb.summary.update({
                "best_trial":     best.number,
                "best_val_rank1": best.value,
                **best.params,
            })
            run.finish()
        except Exception as e:
            print(f"  WandB summary upload failed: {e}")

    return best_params


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def _parse_args():
    p = argparse.ArgumentParser(description="HPO for cow re-ID models (Optuna + WandB).")
    p.add_argument("--model",   choices=["c3d", "x3d", "swin", "vivit"],
                   help="Single model to optimise.")
    p.add_argument("--all",     action="store_true",
                   help="Run HPO for all enabled models.")
    p.add_argument("--config",  default="./configs/config.yaml")
    p.add_argument("--device",  default=None)
    return p.parse_args()


if __name__ == "__main__":
    import json as _json

    args = _parse_args()
    cfg  = yaml.safe_load(open(args.config))

    device = args.device or cfg["training"]["device"]
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"
    print(f"Using device: {device}")

    # Load metadata
    meta_path = Path(cfg["data"]["processed_dir"]) / "dataset_metadata.json"
    with open(meta_path) as f:
        metadata = _json.load(f)

    models_to_run = (
        [m for m, on in cfg["experiment"]["models_to_run"].items() if on]
        if args.all else [args.model]
    )

    all_best = {}
    for model_name in models_to_run:
        best = run_hpo(model_name, metadata, cfg, device)
        all_best[model_name] = best

    print("\n\nHPO complete. Best params summary:")
    for mn, bp in all_best.items():
        print(f"  {mn.upper()}: rank1={bp.get('best_val_rank1', '?'):.4f}  "
              f"lr={bp.get('lr', '?'):.2e}")
