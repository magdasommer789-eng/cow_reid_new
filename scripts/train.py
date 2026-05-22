"""
Main Training Script v2 — Video-Based Cow Re-Identification
============================================================

Entry point for:
  --prepare_data   : Run data preparation (clip extraction).
  --hpo            : Hyperparameter optimisation with Optuna + WandB.
  --model / --all  : Train a single model (or all) with optional best hparams.
  --eval_only      : Skip training, evaluate from checkpoint.
  --final          : Train on combined train+val with best hparams.

Typical workflow:
  # Step 1 — extract clips
  python -m scripts.train --prepare_data

  # Step 2 — HPO (finds best hparams per model)
  python -m scripts.train --hpo --all

  # Step 3 — final training on train+val, then test evaluation
  python -m scripts.train --final --all
"""

import argparse
import json
import sys
from pathlib import Path

import torch
import yaml


# ─────────────────────────────────────────────────────────────────────────────
# Utilities
# ─────────────────────────────────────────────────────────────────────────────

def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def load_metadata(cfg: dict) -> dict:
    meta_path = Path(cfg["data"]["processed_dir"]) / "dataset_metadata.json"
    if not meta_path.exists():
        raise FileNotFoundError(
            f"Metadata not found at {meta_path}. Run --prepare_data first."
        )
    with open(meta_path) as f:
        return json.load(f)


def load_best_hparams(model_name: str, cfg: dict) -> dict:
    """Load saved best hparams or return empty dict (use config defaults)."""
    path = Path(cfg["logging"]["results_dir"]) / f"{model_name}_best_hparams.json"
    if path.exists():
        with open(path) as f:
            hp = json.load(f)
        # Remove summary key that isn't a hyperparameter
        hp.pop("best_val_rank1", None)
        print(f"  Loaded best hparams from {path}")
        return hp
    print(f"  No saved hparams for {model_name} — using config defaults.")
    return {}


def build_model(model_name: str, cfg: dict, hparams: dict, device: str):
    mc  = cfg["model"]
    edim = hparams.get("embedding_dim", mc["embedding_dim"])
    dr   = hparams.get("dropout_rate",  mc[model_name]["dropout_rate"])

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
# Train one model (train split only, val for evaluation)
# ─────────────────────────────────────────────────────────────────────────────

def train_model(
    model_name: str,
    cfg:        dict,
    metadata:   dict,
    device:     str,
    hparams:    dict,
    eval_only:  bool = False,
    checkpoint: str  = None,
) -> dict:
    from .dataset   import build_train_dataset, build_eval_dataset
    from .trainer   import build_trainer
    from .evaluate  import evaluate_model

    model = build_model(model_name, cfg, hparams, device)

    if not eval_only:
        train_ds = build_train_dataset(metadata, temporal_jitter=True)
        val_ds   = build_eval_dataset(metadata, split="val")

        wandb_run = None
        if cfg["logging"].get("use_wandb", False):
            import wandb
            wandb_run = wandb.init(
                project = cfg["logging"].get("wandb_project", "cow-reid"),
                entity  = cfg["logging"].get("wandb_entity", None),
                group   = "hpo_train",
                name    = f"train_{model_name}",
                config  = {**hparams, "model": model_name, "stage": "train"},
                reinit  = "finish_previous",
            )

        trainer = build_trainer(
            model      = model,
            train_ds   = train_ds,
            val_ds     = val_ds,
            cfg        = cfg,
            model_name = model_name,
            hparams    = hparams,
            wandb_run  = wandb_run,
        )
        trainer.train()

        if wandb_run is not None:
            wandb_run.finish()

        best_ckpt = Path(cfg["logging"]["checkpoint_dir"]) / f"{model_name}_best.pt"
        if best_ckpt.exists():
            ckpt = torch.load(best_ckpt, map_location=device)
            model.load_state_dict(ckpt["model_state"])
            print(f"  Loaded best checkpoint (epoch {ckpt['epoch']+1})")
    else:
        if checkpoint is None:
            raise ValueError("--eval_only requires --checkpoint <path>")
        ckpt = torch.load(checkpoint, map_location=device)
        model.load_state_dict(ckpt["model_state"])

    # Evaluate on TEST set (do NOT look at test before final training)
    print(f"\n  NOTE: Use --final for final training + test evaluation.")
    return {}


# ─────────────────────────────────────────────────────────────────────────────
# Argument parsing
# ─────────────────────────────────────────────────────────────────────────────

def _parse_args():
    p = argparse.ArgumentParser(
        description="Cow re-ID training pipeline v2."
    )
    p.add_argument("--model",    choices=["c3d", "x3d", "swin", "vivit"])
    p.add_argument("--all",      action="store_true",
                   help="Run for all enabled models.")
    p.add_argument("--config",   default="./configs/config.yaml")
    p.add_argument("--device",   default=None)

    p.add_argument("--prepare_data", action="store_true",
                   help="Run data preparation (extract clips).")
    p.add_argument("--hpo",      action="store_true",
                   help="Run Optuna HPO on the val set.")
    p.add_argument("--final",    action="store_true",
                   help="Final training on train+val + test evaluation.")
    p.add_argument("--eval_only",action="store_true",
                   help="Skip training, evaluate from checkpoint.")
    p.add_argument("--checkpoint",default=None)
    return p.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    args = _parse_args()
    cfg  = load_config(args.config)

    device = args.device or cfg["training"]["device"]
    if device == "cuda" and not torch.cuda.is_available():
        print("WARNING: CUDA not available — using CPU.")
        device = "cpu"
    print(f"Device: {device}")

    # ── Data preparation ──────────────────────────────────────────────────────
    if args.prepare_data:
        from .data_preparation import prepare_dataset
        prepare_dataset(
            video_dir      = cfg["data"]["video_dir"],
            clips_dir      = cfg["data"]["clips_dir"],
            processed_dir  = cfg["data"]["processed_dir"],
            num_train_cows = cfg["data"]["num_train_cows"],
            num_val_cows   = cfg["data"]["num_val_cows"],
            clip_seconds   = cfg["data"]["clip_seconds"],
            seed           = cfg["data"]["random_seed"],
        )
        return

    metadata = load_metadata(cfg)

    # Determine which models to run
    if args.all:
        to_run = [m for m, on in cfg["experiment"]["models_to_run"].items() if on]
    elif args.model:
        to_run = [args.model]
    else:
        print("Specify --model <name> or --all.  See --help.")
        sys.exit(1)

    # ── HPO ───────────────────────────────────────────────────────────────────
    if args.hpo:
        from .hpo import run_hpo
        for model_name in to_run:
            run_hpo(model_name, metadata, cfg, device)
        return

    # ── Final training on train+val then test eval ────────────────────────────
    if args.final:
        from .train_final import run_final
        all_results = []
        for model_name in to_run:
            hparams = load_best_hparams(model_name, cfg)
            result  = run_final(model_name, metadata, cfg, device, hparams)
            all_results.append(result)
        if len(all_results) > 1:
            from .evaluate import build_results_table
            build_results_table(
                all_results = all_results,
                results_dir = cfg["logging"]["results_dir"],
                cmc_ranks   = cfg["evaluation"]["cmc_ranks"],
            )
        return

    # ── Train on train split (HPO training or standalone) ────────────────────
    for model_name in to_run:
        hparams = load_best_hparams(model_name, cfg)
        print(f"\n{'#'*60}\n# MODEL: {model_name.upper()}\n{'#'*60}")
        train_model(
            model_name = model_name,
            cfg        = cfg,
            metadata   = metadata,
            device     = device,
            hparams    = hparams,
            eval_only  = args.eval_only,
            checkpoint = args.checkpoint,
        )


if __name__ == "__main__":
    main()
