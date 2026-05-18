"""
Main Training Script — Video-Based Cow Re-Identification
=========================================================

Entry point that orchestrates:
  1. Dataset preparation (if not already done)
  2. Model creation (C3D / X3D / Video Swin / ViViT)
  3. Metric-learning training with Batch Hard Triplet Loss
  4. Evaluation on gallery/query test set (CMC + mAP)
  5. Results table across all models

Usage examples:
  # Single model
  python -m scripts.train --model c3d
  python -m scripts.train --model x3d
  python -m scripts.train --model swin
  python -m scripts.train --model vivit

  # All models sequentially
  python -m scripts.train --all

  # Skip training, only evaluate from existing checkpoint
  python -m scripts.train --model c3d --eval_only \
      --checkpoint checkpoints/c3d_best.pt

  # Custom config
  python -m scripts.train --model swin --config configs/config.yaml
"""

import argparse
import json
import os
import sys
from pathlib import Path

import torch
import yaml


# ─────────────────────────────────────────────────────────────────────────────
# Argument parsing
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description="Train and evaluate video-based cow re-ID models."
    )
    parser.add_argument(
        "--model",
        choices=["c3d", "x3d", "swin", "vivit"],
        help="Which model to train/evaluate.",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Train and evaluate all models sequentially.",
    )
    parser.add_argument(
        "--config",
        default="./configs/config.yaml",
        help="Path to YAML config file.",
    )
    parser.add_argument(
        "--eval_only",
        action="store_true",
        help="Skip training, only run evaluation.",
    )
    parser.add_argument(
        "--checkpoint",
        default=None,
        help="Checkpoint path for --eval_only mode.",
    )
    parser.add_argument(
        "--prepare_data",
        action="store_true",
        help="Force re-run of data preparation even if metadata exists.",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Override device (cuda / cpu).",
    )
    return parser.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
# Config loader
# ─────────────────────────────────────────────────────────────────────────────

def load_config(config_path: str) -> dict:
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    return cfg


# ─────────────────────────────────────────────────────────────────────────────
# Data preparation
# ─────────────────────────────────────────────────────────────────────────────

def ensure_data_prepared(cfg: dict, force: bool = False) -> dict:
    """
    Run data_preparation.prepare_dataset() if metadata JSON doesn't exist yet.

    Args:
        cfg:   Config dict.
        force: If True, re-run preparation even if metadata already exists.

    Returns:
        Loaded metadata dict.
    """
    from .data_preparation import prepare_dataset

    meta_path = Path(cfg["data"]["processed_dir"]) / "dataset_metadata.json"

    if meta_path.exists() and not force:
        print(f"Dataset metadata found at {meta_path} — skipping preparation.")
        with open(meta_path) as f:
            return json.load(f)

    print("Running data preparation...")
    metadata = prepare_dataset(
        video_dir       = cfg["data"]["video_dir"],
        processed_dir   = cfg["data"]["processed_dir"],
        num_train_cows  = cfg["data"]["num_train_cows"],
        gallery_seconds = cfg["data"]["gallery_seconds"],
        clip_frames     = cfg["data"]["clip_frames"],
        clip_stride     = cfg["data"]["clip_stride"],
        seed            = cfg["data"]["random_seed"],
    )
    return metadata


# ─────────────────────────────────────────────────────────────────────────────
# Model factory (dispatches to CNN or Transformer factory)
# ─────────────────────────────────────────────────────────────────────────────

def build_model(model_name: str, cfg: dict, device: str):
    """
    Instantiate the requested embedding model with settings from config.

    Args:
        model_name: "c3d" | "x3d" | "swin" | "vivit"
        cfg:        Config dict.
        device:     "cuda" or "cpu".

    Returns:
        torch.nn.Module
    """
    mc = cfg["model"]
    embed_dim = mc["embedding_dim"]

    if model_name == "c3d":
        from .models_cnn import create_cnn_model
        return create_cnn_model(
            model_name      = "c3d",
            embedding_dim   = embed_dim,
            pretrained      = mc["c3d"]["pretrained"],
            freeze_backbone = mc["c3d"]["freeze_backbone"],
            dropout_rate    = mc["c3d"]["dropout_rate"],
            device          = device,
        )

    if model_name == "x3d":
        from .models_cnn import create_cnn_model
        return create_cnn_model(
            model_name      = "x3d",
            embedding_dim   = embed_dim,
            pretrained      = mc["x3d"]["pretrained"],
            freeze_backbone = mc["x3d"]["freeze_backbone"],
            dropout_rate    = mc["x3d"]["dropout_rate"],
            device          = device,
            model_size      = mc["x3d"]["model_size"],
        )

    if model_name == "swin":
        from .models_transformer import create_transformer_model
        return create_transformer_model(
            model_name      = "swin",
            embedding_dim   = embed_dim,
            pretrained      = mc["swin"]["pretrained"],
            freeze_backbone = mc["swin"]["freeze_backbone"],
            dropout_rate    = mc["swin"]["dropout_rate"],
            device          = device,
            swin_variant    = mc["swin"]["model_name"],
        )

    if model_name == "vivit":
        from .models_transformer import create_transformer_model
        return create_transformer_model(
            model_name      = "vivit",
            embedding_dim   = embed_dim,
            pretrained      = mc["vivit"]["pretrained"],
            freeze_backbone = mc["vivit"]["freeze_backbone"],
            dropout_rate    = mc["vivit"]["dropout_rate"],
            device          = device,
            num_frames      = cfg["data"]["clip_frames"],
            tubelet_size    = mc["vivit"]["tubelet_size"],
        )

    raise ValueError(f"Unknown model: {model_name}")


# ─────────────────────────────────────────────────────────────────────────────
# Single-model train + evaluate
# ─────────────────────────────────────────────────────────────────────────────

def run_model(
    model_name: str,
    cfg:        dict,
    metadata:   dict,
    device:     str,
    eval_only:  bool = False,
    checkpoint: str  = None,
) -> dict:
    """
    Full pipeline for one model: (optionally train) then evaluate.

    Args:
        model_name:  Which model to run.
        cfg:         Config dict.
        metadata:    Dataset metadata from data_preparation.
        device:      Compute device.
        eval_only:   If True, skip training and load `checkpoint`.
        checkpoint:  Path to .pt file for eval_only mode.

    Returns:
        Evaluation results dict.
    """
    from .dataset import build_train_dataset, build_eval_dataset
    from .evaluate import evaluate_model

    img_size = cfg["data"]["target_size"][0]

    # ── Build model ───────────────────────────────────────────────────────────
    model = build_model(model_name, cfg, device)

    # ── Train ─────────────────────────────────────────────────────────────────
    if not eval_only:
        from .trainer import build_trainer

        train_ds = build_train_dataset(
            metadata        = metadata,
            img_size        = img_size,
            temporal_jitter = cfg["augmentation"]["train"]["temporal_jitter"],
        )

        trainer = build_trainer(model, train_ds, cfg, model_name)
        trainer.train()

        # After training, load the best checkpoint for evaluation
        best_ckpt = Path(cfg["logging"]["checkpoint_dir"]) / f"{model_name}_best.pt"
        if best_ckpt.exists():
            ckpt = torch.load(best_ckpt, map_location=device)
            model.load_state_dict(ckpt["model_state"])
            print(f"Loaded best checkpoint: {best_ckpt}")

    else:
        # Eval-only: load the provided checkpoint
        if checkpoint is None:
            raise ValueError("--eval_only requires --checkpoint <path>")
        ckpt = torch.load(checkpoint, map_location=device)
        model.load_state_dict(ckpt["model_state"])
        print(f"Loaded checkpoint: {checkpoint}")

    # ── Evaluate ──────────────────────────────────────────────────────────────
    eval_ds = build_eval_dataset(metadata, img_size=img_size)

    results = evaluate_model(
        model        = model,
        eval_dataset = eval_ds,
        model_name   = model_name,
        device       = device,
        batch_size   = cfg["training"]["batch_size"],
        cmc_ranks    = cfg["evaluation"]["cmc_ranks"],
        results_dir  = cfg["logging"]["results_dir"],
        verbose      = True,
    )

    return results


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    cfg  = load_config(args.config)

    # Device override
    if args.device:
        cfg["training"]["device"] = args.device
    device = cfg["training"]["device"]
    if device == "cuda" and not torch.cuda.is_available():
        print("WARNING: CUDA not available — falling back to CPU.")
        device = "cpu"
        cfg["training"]["device"] = "cpu"
    print(f"Using device: {device}")

    # Data preparation
    metadata = ensure_data_prepared(cfg, force=args.prepare_data)

    # Determine which models to run
    if args.all:
        exp_cfg   = cfg["experiment"]["models_to_run"]
        to_run    = [m for m, enabled in exp_cfg.items() if enabled]
    elif args.model:
        to_run    = [args.model]
    else:
        print("ERROR: specify --model <name> or --all")
        sys.exit(1)

    print(f"\nModels to run: {to_run}")

    # Run each model
    all_results = []
    for model_name in to_run:
        print(f"\n{'#'*60}")
        print(f"# MODEL: {model_name.upper()}")
        print(f"{'#'*60}")

        results = run_model(
            model_name = model_name,
            cfg        = cfg,
            metadata   = metadata,
            device     = device,
            eval_only  = args.eval_only,
            checkpoint = args.checkpoint,
        )
        all_results.append(results)

    # ── Final results table ───────────────────────────────────────────────────
    if len(all_results) > 1:
        from .evaluate import build_results_table
        build_results_table(
            all_results = all_results,
            results_dir = cfg["logging"]["results_dir"],
            cmc_ranks   = cfg["evaluation"]["cmc_ranks"],
        )
    elif len(all_results) == 1:
        r = all_results[0]
        print(f"\nFinal result for {r['model_name'].upper()}:")
        print(f"  mAP={r['mAP']:.4f}  "
              f"Rank-1={r.get('rank1', 0):.4f}  "
              f"Rank-5={r.get('rank5', 0):.4f}  "
              f"Rank-10={r.get('rank10', 0):.4f}")

    print("\nDone.")


if __name__ == "__main__":
    main()
