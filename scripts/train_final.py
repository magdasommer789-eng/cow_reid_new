"""
Final Training Script — Train on train+val, then evaluate on test
=================================================================

After HPO identifies the best hyperparameters for each model, this script:
  1. Trains the model on the COMBINED train+val set with those hyperparameters.
  2. Early stopping via training-loss plateau (no held-out val any more).
  3. Loads the best checkpoint and evaluates on the TEST set.
  4. Reports CMC@1/5/10 and mAP; saves per-query ranking table.
  5. Saves a final learning curve PNG.

The test set is NOT touched before this step.

Usage:
  python -m scripts.train_final --model c3d
  python -m scripts.train_final --all
"""

import argparse
import json
from pathlib import Path
from typing import Dict

import torch
import yaml


# ─────────────────────────────────────────────────────────────────────────────
# Run final training + test evaluation for one model
# ─────────────────────────────────────────────────────────────────────────────

def run_final(
    model_name: str,
    metadata:   Dict,
    cfg:        Dict,
    device:     str,
    hparams:    Dict,
) -> Dict:
    """
    Train on train+val, evaluate on test, return results dict.
    """
    from .dataset   import build_combined_train_val_dataset, build_eval_dataset
    from .evaluate  import evaluate_model
    from .trainer   import build_trainer

    print(f"\n{'='*60}")
    print(f"  Final training: {model_name.upper()}")
    print(f"  Hparams: {hparams}")
    print(f"{'='*60}")

    # ── Build model ───────────────────────────────────────────────────────────
    mc   = cfg["model"]
    edim = hparams.get("embedding_dim", mc["embedding_dim"])
    dr   = hparams.get("dropout_rate",  mc[model_name]["dropout_rate"])

    if model_name in ("c3d", "x3d"):
        from .models_cnn import create_cnn_model
        model = create_cnn_model(
            model_name      = model_name,
            embedding_dim   = edim,
            pretrained      = mc[model_name]["pretrained"],
            freeze_backbone = mc[model_name]["freeze_backbone"],
            dropout_rate    = dr,
            device          = device,
            model_size      = mc["x3d"]["model_size"] if model_name == "x3d" else "m",
        )
    elif model_name == "swin":
        from .models_transformer import create_transformer_model
        model = create_transformer_model(
            model_name      = "swin",
            embedding_dim   = edim,
            pretrained      = mc["swin"]["pretrained"],
            freeze_backbone = mc["swin"]["freeze_backbone"],
            dropout_rate    = dr,
            device          = device,
            swin_variant    = mc["swin"]["model_name"],
        )
    elif model_name == "vivit":
        from .models_transformer import create_transformer_model
        model = create_transformer_model(
            model_name      = "vivit",
            embedding_dim   = edim,
            pretrained      = mc["vivit"]["pretrained"],
            freeze_backbone = mc["vivit"]["freeze_backbone"],
            dropout_rate    = dr,
            device          = device,
            num_frames      = cfg["data"].get("clip_frames", 16),
            tubelet_size    = mc["vivit"]["tubelet_size"],
        )
    else:
        raise ValueError(f"Unknown model: {model_name}")

    # ── Datasets ──────────────────────────────────────────────────────────────
    train_val_ds = build_combined_train_val_dataset(metadata, temporal_jitter=True)
    test_ds      = build_eval_dataset(metadata, split="test")
    # Pass val set for monitoring (CMC logged each epoch for the learning curve).
    # Early stopping still uses loss plateau — val is NOT used to select best epoch.
    val_ds_monitor = build_eval_dataset(metadata, split="val")

    print(f"  Train+val dataset: {len(train_val_ds)} clips  "
          f"({len(train_val_ds.label_to_idx)} identities)")

    # ── WandB run ─────────────────────────────────────────────────────────────
    wandb_run = None
    if cfg["logging"].get("use_wandb", False):
        import wandb
        wandb_run = wandb.init(
            project = cfg["logging"].get("wandb_project", "cow-reid"),
            entity  = cfg["logging"].get("wandb_entity", None),
            group   = "final_training",
            name    = f"final_{model_name}",
            config  = {**hparams, "model": model_name, "stage": "final"},
            reinit  = "finish_previous",
        )

    # ── Train ─────────────────────────────────────────────────────────────────
    # val_ds_monitor is used ONLY for logging/curves — NOT for early stopping.
    # is_final_train=True keeps early stopping on loss plateau.
    trainer = build_trainer(
        model          = model,
        train_ds       = train_val_ds,
        val_ds         = val_ds_monitor,
        cfg            = cfg,
        model_name     = model_name,
        hparams        = hparams,
        wandb_run      = wandb_run,
        is_final_train = True,
    )
    trainer.train()

    if wandb_run is not None:
        wandb_run.finish()

    # ── Load best checkpoint ──────────────────────────────────────────────────
    best_ckpt = Path(cfg["logging"]["checkpoint_dir"]) / f"{model_name}_best.pt"
    if best_ckpt.exists():
        ckpt = torch.load(best_ckpt, map_location=device)
        model.load_state_dict(ckpt["model_state"])
        print(f"  Loaded best checkpoint from epoch {ckpt['epoch']+1}")

    # ── Test evaluation ───────────────────────────────────────────────────────
    results = evaluate_model(
        model        = model,
        eval_dataset = test_ds,
        model_name   = model_name,
        device       = device,
        batch_size   = cfg["training"]["batch_size"],
        cmc_ranks    = cfg["evaluation"]["cmc_ranks"],
        results_dir  = cfg["logging"]["results_dir"],
        verbose      = True,
    )

    # Log test results to WandB if available
    if cfg["logging"].get("use_wandb", False):
        try:
            import wandb
            run = wandb.init(
                project = cfg["logging"].get("wandb_project", "cow-reid"),
                entity  = cfg["logging"].get("wandb_entity", None),
                group   = "final_training",
                name    = f"test_{model_name}",
                config  = {"model": model_name, "stage": "test"},
                reinit  = "finish_previous",
            )
            wandb.summary.update({
                "test/mAP":    results["mAP"],
                "test/rank1":  results.get("rank1", 0),
                "test/rank5":  results.get("rank5", 0),
                "test/rank10": results.get("rank10", 0),
            })
            run.finish()
        except Exception as e:
            print(f"  WandB test upload failed: {e}")

    return results


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def _parse_args():
    p = argparse.ArgumentParser(
        description="Final training (train+val) + test evaluation."
    )
    p.add_argument("--model",  choices=["c3d", "x3d", "swin", "vivit"])
    p.add_argument("--all",    action="store_true")
    p.add_argument("--config", default="./configs/config.yaml")
    p.add_argument("--device", default=None)
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    cfg  = yaml.safe_load(open(args.config))

    device = args.device or cfg["training"]["device"]
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"
    print(f"Device: {device}")

    meta_path = Path(cfg["data"]["processed_dir"]) / "dataset_metadata.json"
    with open(meta_path) as f:
        metadata = json.load(f)

    models_to_run = (
        [m for m, on in cfg["experiment"]["models_to_run"].items() if on]
        if args.all else [args.model]
    )

    all_results = []
    for model_name in models_to_run:
        hparams_path = (
            Path(cfg["logging"]["results_dir"]) /
            f"{model_name}_best_hparams.json"
        )
        hparams = {}
        if hparams_path.exists():
            with open(hparams_path) as f:
                hparams = json.load(f)
            hparams.pop("best_val_rank1", None)

        result = run_final(model_name, metadata, cfg, device, hparams)
        all_results.append(result)

    if len(all_results) > 1:
        from scripts.evaluate import build_results_table
        build_results_table(
            all_results = all_results,
            results_dir = cfg["logging"]["results_dir"],
            cmc_ranks   = cfg["evaluation"]["cmc_ranks"],
        )
