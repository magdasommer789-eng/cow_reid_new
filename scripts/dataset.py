"""
PyTorch Dataset Module for Video-Based Cow Re-Identification
=============================================================

Two dataset classes are defined here:

1. VideoClipDataset
   Used during TRAINING. Supports P×K sampling (P identities, K clips each)
   which is required for Batch Hard Triplet Loss.

2. GalleryQueryDataset
   Used during EVALUATION. Returns clips labelled as "gallery" or "query"
   so the evaluator can build the distance matrix for CMC/mAP computation.

Educational Note — Why P×K sampling?
--------------------------------------
Triplet loss needs at least one positive pair (same ID) and one negative pair
(different ID) per anchor. Random batches rarely have enough positives.
P×K sampling guarantees P*(K-1) valid positive pairs per batch, which makes
gradient signals much more stable and training far more efficient.
"""

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, Sampler
from torchvision import transforms
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from collections import defaultdict
import random


# ─────────────────────────────────────────────────────────────────────────────
# Frame-level video reader
# ─────────────────────────────────────────────────────────────────────────────

def load_video_clip(
    video_path: str,
    start_frame: int,
    num_frames: int,
    target_size: Tuple[int, int] = (224, 224),
) -> np.ndarray:
    """
    Load a fixed-length clip from a video file using OpenCV.

    Args:
        video_path:   Path to the video file.
        start_frame:  Index of the first frame to read.
        num_frames:   Number of consecutive frames to load.
        target_size:  (H, W) to resize each frame.

    Returns:
        np.ndarray of shape (num_frames, H, W, 3), dtype uint8, RGB order.
    """
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    frames = []
    for _ in range(num_frames):
        ret, frame = cap.read()
        if not ret:
            # Pad with the last valid frame if the video ends early
            if frames:
                frames.append(frames[-1].copy())
            else:
                frames.append(np.zeros((*target_size, 3), dtype=np.uint8))
            continue
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = cv2.resize(frame, (target_size[1], target_size[0]))
        frames.append(frame)

    cap.release()
    return np.stack(frames, axis=0)   # (T, H, W, 3)


def frames_to_tensor(frames: np.ndarray, transform=None) -> torch.Tensor:
    """
    Convert (T, H, W, 3) uint8 numpy array to a model-ready tensor.

    Two layout conventions exist in the video deep-learning world:
      - (C, T, H, W): used by C3D, X3D, Video Swin
      - (T, C, H, W): used by some transformer implementations

    This function produces (C, T, H, W) by default (most common for 3D-CNNs).
    The transform is applied per-frame before stacking.

    Args:
        frames:    (T, H, W, 3) uint8 numpy array.
        transform: Optional torchvision transform applied to each frame PIL image.

    Returns:
        Tensor of shape (C, T, H, W), float32, ImageNet-normalised.
    """
    from PIL import Image

    if transform is not None:
        # Apply spatial augmentation frame-by-frame (keeps temporal coherence)
        tensors = []
        for t in range(frames.shape[0]):
            img = Image.fromarray(frames[t])
            tensors.append(transform(img))               # (C, H, W)
        clip = torch.stack(tensors, dim=1)               # (C, T, H, W)
    else:
        # Fast path: manual normalisation without per-frame PIL overhead
        clip = torch.from_numpy(frames).permute(3, 0, 1, 2).float() / 255.0
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1, 1)
        std  = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1, 1)
        clip = (clip - mean) / std

    return clip   # (C, T, H, W)


# ─────────────────────────────────────────────────────────────────────────────
# Transforms
# ─────────────────────────────────────────────────────────────────────────────

def get_frame_transform(split: str = "train", img_size: int = 224) -> transforms.Compose:
    """
    Spatial augmentation applied independently to each frame.

    Training uses colour jitter + random horizontal flip.
    Test uses only deterministic resize (no augmentation).

    Args:
        split:    "train" or "test".
        img_size: Target spatial size (H = W).

    Returns:
        torchvision.transforms.Compose
    """
    if split == "train":
        return transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.3, contrast=0.3,
                                   saturation=0.3, hue=0.0),
            transforms.RandomRotation(degrees=10),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])
    else:
        return transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])


# ─────────────────────────────────────────────────────────────────────────────
# Training dataset
# ─────────────────────────────────────────────────────────────────────────────

class VideoClipDataset(Dataset):
    """
    Dataset for metric learning training.

    Each item is a (clip_tensor, label) pair where label is the integer
    cow identity index.  The P×K sampler (PKSampler below) controls
    how batches are assembled to ensure positive pairs.

    Args:
        clip_list:    List of clip descriptor dicts from data_preparation.py.
        id_to_label:  Mapping {cow_id (str) → integer label}.
        transform:    Per-frame spatial transform (torchvision).
        clip_frames:  Number of frames to load per clip.
        img_size:     Spatial size (H = W).
        temporal_jitter: If True, randomly shift the clip start by ±4 frames.
    """

    def __init__(
        self,
        clip_list:       List[Dict],
        id_to_label:     Dict[str, int],
        transform:       Optional[transforms.Compose] = None,
        clip_frames:     int = 16,
        img_size:        int = 224,
        temporal_jitter: bool = False,
    ):
        self.clips          = clip_list
        self.id_to_label    = id_to_label
        self.transform      = transform
        self.clip_frames    = clip_frames
        self.img_size       = img_size
        self.temporal_jitter = temporal_jitter

        # Build an index from label → list of clip indices (for PK sampler)
        self.label_to_indices: Dict[int, List[int]] = defaultdict(list)
        for idx, clip in enumerate(self.clips):
            label = id_to_label[clip["cow_id"]]
            self.label_to_indices[label].append(idx)

        self.unique_labels = sorted(self.label_to_indices.keys())

    def __len__(self) -> int:
        return len(self.clips)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        clip_info = self.clips[idx]

        start = clip_info["start_frame"]

        # Optional temporal jitter: randomly shift start frame by up to 4 frames
        if self.temporal_jitter:
            jitter = random.randint(-4, 4)
            start = max(0, start + jitter)

        frames = load_video_clip(
            video_path  = clip_info["video_path"],
            start_frame = start,
            num_frames  = self.clip_frames,
            target_size = (self.img_size, self.img_size),
        )

        clip_tensor = frames_to_tensor(frames, transform=self.transform)
        label       = self.id_to_label[clip_info["cow_id"]]

        return clip_tensor, label


# ─────────────────────────────────────────────────────────────────────────────
# P×K (PK) Batch Sampler
# ─────────────────────────────────────────────────────────────────────────────

class PKSampler(Sampler):
    """
    P×K batch sampler for metric learning.

    Each batch contains exactly P unique identities, each represented by
    exactly K randomly sampled clips.  Batch size = P × K.

    This guarantees that every batch contains valid positive pairs for
    Batch Hard Triplet mining, making training much more stable.

    Args:
        label_to_indices: Dict mapping integer label → list of dataset indices.
        P:  Number of identities per batch.
        K:  Number of clips per identity per batch.
        num_batches: How many batches to yield per epoch.
    """

    def __init__(
        self,
        label_to_indices: Dict[int, List[int]],
        P: int = 4,
        K: int = 2,
        num_batches: int = 100,
    ):
        self.label_to_indices = label_to_indices
        self.labels           = list(label_to_indices.keys())
        self.P                = P
        self.K                = K
        self.num_batches      = num_batches

        if len(self.labels) < P:
            raise ValueError(
                f"PKSampler: only {len(self.labels)} identities but P={P}."
            )

    def __len__(self) -> int:
        return self.num_batches * self.P * self.K

    def __iter__(self):
        for _ in range(self.num_batches):
            selected_labels = random.sample(self.labels, self.P)
            batch_indices = []
            for label in selected_labels:
                pool = self.label_to_indices[label]
                # Sample K indices with replacement if pool is smaller than K
                chosen = random.choices(pool, k=self.K)
                batch_indices.extend(chosen)
            yield from batch_indices


# ─────────────────────────────────────────────────────────────────────────────
# Evaluation dataset (gallery + query)
# ─────────────────────────────────────────────────────────────────────────────

class GalleryQueryDataset(Dataset):
    """
    Dataset used at evaluation time.

    Returns each clip with:
      - clip_tensor: (C, T, H, W) float tensor
      - cow_id:      string identity label
      - role:        "gallery" or "query"

    The evaluator computes pairwise distances between all gallery embeddings
    and all query embeddings, then ranks gallery identities for each query.

    Args:
        clip_list:   Combined list of gallery + query clip descriptors.
        transform:   Spatial transform (should be the TEST transform — no augment).
        clip_frames: Frames per clip.
        img_size:    Spatial size.
    """

    def __init__(
        self,
        clip_list:   List[Dict],
        transform:   Optional[transforms.Compose] = None,
        clip_frames: int = 16,
        img_size:    int = 224,
    ):
        self.clips      = clip_list
        self.transform  = transform
        self.clip_frames = clip_frames
        self.img_size   = img_size

    def __len__(self) -> int:
        return len(self.clips)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, str, str]:
        clip_info = self.clips[idx]

        frames = load_video_clip(
            video_path  = clip_info["video_path"],
            start_frame = clip_info["start_frame"],
            num_frames  = self.clip_frames,
            target_size = (self.img_size, self.img_size),
        )

        clip_tensor = frames_to_tensor(frames, transform=self.transform)

        return clip_tensor, clip_info["cow_id"], clip_info["role"]


# ─────────────────────────────────────────────────────────────────────────────
# Factory helpers
# ─────────────────────────────────────────────────────────────────────────────

def build_train_dataset(
    metadata: Dict,
    img_size: int = 224,
    temporal_jitter: bool = True,
) -> VideoClipDataset:
    """
    Build the training VideoClipDataset from the prepared metadata dict.

    Args:
        metadata:        Output of data_preparation.prepare_dataset().
        img_size:        Spatial size for all frames.
        temporal_jitter: Whether to apply random temporal shift.

    Returns:
        VideoClipDataset ready for the DataLoader.
    """
    transform = get_frame_transform(split="train", img_size=img_size)
    return VideoClipDataset(
        clip_list       = metadata["splits"]["train"]["clips"],
        id_to_label     = metadata["train_id_to_idx"],
        transform       = transform,
        clip_frames     = metadata["clip_frames"],
        img_size        = img_size,
        temporal_jitter = temporal_jitter,
    )


def build_eval_dataset(metadata: Dict, img_size: int = 224) -> GalleryQueryDataset:
    """
    Build the evaluation GalleryQueryDataset from the prepared metadata dict.

    Merges gallery and query clip lists — the dataset returns the role field
    so the evaluator can separate them.

    Args:
        metadata: Output of data_preparation.prepare_dataset().
        img_size: Spatial size for all frames.

    Returns:
        GalleryQueryDataset ready for the DataLoader.
    """
    transform  = get_frame_transform(split="test", img_size=img_size)
    all_clips  = (metadata["splits"]["gallery"]["clips"] +
                  metadata["splits"]["query"]["clips"])
    return GalleryQueryDataset(
        clip_list   = all_clips,
        transform   = transform,
        clip_frames = metadata["clip_frames"],
        img_size    = img_size,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Quick self-test (run: python -m scripts.dataset)
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import json

    meta_path = "./data/processed/dataset_metadata.json"
    print(f"Loading metadata from {meta_path}")

    with open(meta_path) as f:
        meta = json.load(f)

    # Training dataset
    train_ds = build_train_dataset(meta, img_size=224)
    print(f"\nTrain dataset: {len(train_ds)} clips, "
          f"{len(train_ds.unique_labels)} identities")

    clip, label = train_ds[0]
    print(f"  clip shape: {clip.shape}  label: {label}")

    # Evaluation dataset
    eval_ds = build_eval_dataset(meta, img_size=224)
    gallery_count = sum(1 for c in eval_ds.clips if c["role"] == "gallery")
    query_count   = sum(1 for c in eval_ds.clips if c["role"] == "query")
    print(f"\nEval dataset:  {len(eval_ds)} clips total  "
          f"(gallery={gallery_count}, query={query_count})")

    clip, cow_id, role = eval_ds[0]
    print(f"  clip shape: {clip.shape}  cow_id: {cow_id}  role: {role}")

    # PK sampler
    sampler = PKSampler(train_ds.label_to_indices, P=4, K=2, num_batches=10)
    from torch.utils.data import DataLoader, BatchSampler
    loader = DataLoader(train_ds, batch_sampler=BatchSampler(sampler, batch_size=8,
                                                              drop_last=False))
    for clips_batch, labels_batch in loader:
        print(f"\nPK batch — clips: {clips_batch.shape}  labels: {labels_batch.tolist()}")
        break
