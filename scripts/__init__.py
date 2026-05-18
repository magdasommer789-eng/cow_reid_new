"""
cow_reid_new — Video-Based Cow Re-Identification
================================================

Package exposing the core modules for training and evaluation.
"""

from .data_preparation import prepare_dataset, discover_videos, split_cows
from .dataset import (
    VideoClipDataset,
    GalleryQueryDataset,
    PKSampler,
    build_train_dataset,
    build_eval_dataset,
    get_frame_transform,
)

__all__ = [
    "prepare_dataset",
    "discover_videos",
    "split_cows",
    "VideoClipDataset",
    "GalleryQueryDataset",
    "PKSampler",
    "build_train_dataset",
    "build_eval_dataset",
    "get_frame_transform",
]
