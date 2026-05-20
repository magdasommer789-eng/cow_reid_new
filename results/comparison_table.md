# Cow Re-Identification — Model Comparison Results

## Test Protocol
- **Gallery**: First 10 seconds of each test cow video (10 cows, 29 gallery clips/cow)
- **Query**: Remaining video (non-overlapping with gallery), 1932 total query clips
- **Train / Test split**: 21 train cows / 10 test cows (open-set — no identity overlap)
- **Metric learning**: Batch Hard Triplet Loss, 512-d L2-normalised embeddings
- **Transfer learning**: Kinetics-400 (ImageNet-init) for X3D & Swin; ImageNet-21K for ViViT; random init for C3D

## Results

| Model | Epochs | mAP | Rank-1 | Rank-5 | Rank-10 |
|-------|--------|-----|--------|--------|---------|
| C3D | 20 | 39.4% | 16.5% | 72.6% | 100.0% |
| X3D-M | 20 | 58.3% | 37.2% | 90.3% | 100.0% |
| Video Swin-T | 9 | 53.8% | 30.1% | 90.8% | 100.0% |
| ViViT | 5 | 56.0% | 34.6% | 94.6% | 100.0% |

*mAP = mean Average Precision; Rank-k = CMC at rank k*

## Notes
- **Swin**: trained ~9 epochs before disk-full crash; best checkpoint used for evaluation
- **ViViT**: early stopping after 5 epochs (patience=3, no improvement)
- **C3D**: trained from random init (no public ImageNet-pretrained C3D weights exist)
- All other models use Kinetics-400 pretrained weights (backbone initialised from ImageNet)
