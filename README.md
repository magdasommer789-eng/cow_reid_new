# Cow Re-Identification — Video-Based Metric Learning

**Educational deep learning project** comparing four video model architectures
on an open-set animal re-identification task.

## Problem Statement

Given 31 cows, each filmed for ~1 minute, **identify a cow from new video footage
without ever training a classifier for that specific cow**.

This is the *open-set re-identification* problem:
- Training cows and test cows are **completely disjoint** (21 train / 10 test)
- The model learns an **embedding space** where clips of the same cow cluster together
- At test time, a 10-second *gallery* clip is compared against *query* clips using cosine distance

## Models Compared

| Model | Type | Params (approx) | Transfer Learning |
|-------|------|-----------------|-------------------|
| **C3D** | 3D CNN | 78M | Sports-1M |
| **X3D-M** | Efficient 3D CNN | 3.8M | Kinetics-400 (ImageNet init) |
| **Video Swin-T** | 3D Transformer | 28M | Kinetics-400 (ImageNet-22K init) |
| **ViViT** | Pure Transformer | 86M | ImageNet-21K |

## Results

| Model | mAP | Rank-1 | Rank-5 | Rank-10 |
|-------|-----|--------|--------|---------|
| C3D | — | — | — | — |
| X3D-M | — | — | — | — |
| Video Swin-T | — | — | — | — |
| ViViT | — | — | — | — |

*Table populated after training runs on the remote server.*

## Project Structure

```
cow_reid_new/
├── configs/config.yaml          # All hyperparameters
├── scripts/
│   ├── data_preparation.py      # Split cows, create gallery/query metadata
│   ├── dataset.py               # VideoClipDataset + PKSampler + GalleryQueryDataset
│   ├── models_cnn.py            # C3D + X3D with embedding heads  [Session 2]
│   ├── models_transformer.py    # Video Swin + ViViT               [Session 3]
│   ├── losses.py                # Batch Hard Triplet Loss           [Session 2]
│   ├── trainer.py               # Training loop                     [Session 2]
│   ├── evaluate.py              # CMC, mAP, rank-k evaluation       [Session 4]
│   └── train.py                 # Main entry point                  [Session 3]
├── data/                        # Symlink → /data (videos)
├── checkpoints/                 # Saved model weights
├── logs/                        # TensorBoard logs
└── results/                     # Final results tables
```

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Prepare dataset (split cows, create gallery/query metadata)
python -m scripts.data_preparation \
    --video_dir /data \
    --processed_dir ./data/processed \
    --num_train_cows 21 \
    --gallery_seconds 10

# 3. Train a model
python -m scripts.train --model c3d        # or x3d | swin | vivit

# 4. Evaluate
python -m scripts.evaluate --model c3d --checkpoint checkpoints/c3d_best.pt

# 5. Run all models and generate results table
python -m scripts.train --all
```

## Evaluation Protocol

- **Gallery**: First 10 seconds of each test cow video → 1 embedding per cow (averaged over clips)
- **Query**: Remaining video → multiple clips per cow, each ranked against the gallery
- **Metric**: CMC curve (Rank-1/5/10) and mAP computed over all query clips

## Educational References

- **Re-ID as metric learning**: Hermans et al., "In Defense of the Triplet Loss for Person Re-Identification" (2017)
- **C3D**: Tran et al., "Learning Spatiotemporal Features with 3D CNNs" (2015)
- **X3D**: Feichtenhofer, "X3D: Expanding Architectures for Efficient Video Recognition" (2020)
- **Video Swin**: Liu et al., "Video Swin Transformer" (2021)
- **ViViT**: Arnab et al., "ViViT: A Video Vision Transformer" (2021)
