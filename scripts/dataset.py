"""
Dataset Module v2 — Video-Based Cow Re-Identification
======================================================

Loads from *physical* 10-second clip files (written by data_preparation.py).
Two dataset classes are defined here, shared by training, val, and test.

  VideoClipDataset   — training only; P×K batch sampler for triplet loss.
  ReIDEvalDataset    — val AND test; returns (tensor, cow_id, role).
                       The same class and the same CMC protocol are used
                       for both validation and final test evaluation.

Val/Test re-ID protocol (matching what the user requested):
  • First clip per cow  → role = "query"  (the 10-second probe)
  • Remaining clips      → role = "gallery" (no temporal overlap with query)
  This is already encoded in the metadata by data_preparation.py.
"""

import random
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset, Sampler
from torchvision import transforms


# ─────────────────────────────────────────────────────────────────────────────
# Frame / tensor helpers  (PyTorch 2.2 + NumPy 2.x safe)
# ─────────────────────────────────────────────────────────────────────────────

_IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
_IMAGENET_STD  = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)


def _pil_to_float_tensor(img: Image.Image) -> torch.Tensor:
    """PIL RGB → (C, H, W) float32 without going through NumPy.

    PyTorch 2.2 + NumPy 2.x are binary-incompatible at the C-API level;
    torch.from_numpy() raises at runtime.  frombuffer() bypasses that.
    """
    W, H = img.size
    raw = img.convert("RGB").tobytes()
    t   = torch.frombuffer(bytearray(raw), dtype=torch.uint8)
    return t.view(H, W, 3).permute(2, 0, 1).float() / 255.0  # (C, H, W)


def frames_to_tensor(
    frames:    np.ndarray,
    transform: Optional[object] = None,
) -> torch.Tensor:
    """
    Convert (T, H, W, 3) uint8 array → (C, T, H, W) ImageNet-normalised tensor.

    *transform* must be a PIL-only transform (no ToTensor / Normalize).
    """
    tensors = []
    for t in range(frames.shape[0]):
        img = Image.fromarray(frames[t])
        if transform is not None:
            img = transform(img)
        tensor = _pil_to_float_tensor(img)
        tensor = (tensor - _IMAGENET_MEAN) / _IMAGENET_STD
        tensors.append(tensor)
    return torch.stack(tensors, dim=1)   # (C, T, H, W)


# ─────────────────────────────────────────────────────────────────────────────
# Clip file loader
# ─────────────────────────────────────────────────────────────────────────────

def load_clip_file(
    clip_path:       str,
    num_frames:      int             = 16,
    target_size:     Tuple[int, int] = (224, 224),
    temporal_jitter: bool            = False,
) -> np.ndarray:
    """
    Load *num_frames* from a physical 10-second clip file.

    With temporal_jitter=True (training): choose a random contiguous window
    of *num_frames* frames anywhere within the clip.
    Without jitter (eval): sample uniformly across the full clip.

    Returns (T, H, W, 3) uint8 RGB array.
    """
    cap   = cv2.VideoCapture(clip_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total <= 0:
        total = 250   # 10 s @ 25 fps fallback

    if temporal_jitter and total > num_frames:
        start          = random.randint(0, total - num_frames)
        sample_indices = list(range(start, start + num_frames))
    else:
        sample_indices = np.linspace(0, total - 1, num_frames, dtype=int).tolist()

    frames    = []
    prev_idx  = -1
    for idx in sample_indices:
        idx = int(idx)
        if idx != prev_idx + 1:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = cv2.resize(frame, (target_size[1], target_size[0]))
        else:
            frame = (frames[-1].copy() if frames
                     else np.zeros((*target_size, 3), dtype=np.uint8))
        frames.append(frame)
        prev_idx = idx

    cap.release()
    return np.stack(frames, axis=0)   # (T, H, W, 3)


# ─────────────────────────────────────────────────────────────────────────────
# Transforms
# ─────────────────────────────────────────────────────────────────────────────

def get_frame_transform(split: str = "train", img_size: int = 224):
    """
    PIL-only transform applied independently to each frame.
    Clips from ffmpeg are 256×256, so random/centre crop brings them to img_size.
    """
    if split == "train":
        return transforms.Compose([
            transforms.RandomCrop(img_size),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3),
            transforms.RandomRotation(degrees=10),
        ])
    return transforms.Compose([
        transforms.CenterCrop(img_size),
    ])


# ─────────────────────────────────────────────────────────────────────────────
# Training dataset
# ─────────────────────────────────────────────────────────────────────────────

class VideoClipDataset(Dataset):
    """
    Loads physical clip files for training.

    Each item: (clip_tensor, label)
      clip_tensor: (C=3, T, H, W) float32, ImageNet-normalised.
      label:       int — cow identity index (0 .. N_train-1).
    """

    def __init__(
        self,
        clip_entries:    List[Dict],
        clip_frames:     int             = 16,
        target_size:     Tuple[int, int] = (224, 224),
        temporal_jitter: bool            = True,
    ):
        self.entries         = clip_entries
        self.clip_frames     = clip_frames
        self.target_size     = target_size
        self.temporal_jitter = temporal_jitter
        self.transform       = get_frame_transform("train", target_size[0])

        self.label_to_idx: Dict[int, List[int]] = defaultdict(list)
        for i, e in enumerate(self.entries):
            self.label_to_idx[e["label"]].append(i)
        self.labels = [e["label"] for e in self.entries]

    def __len__(self) -> int:
        return len(self.entries)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        entry  = self.entries[idx]
        frames = load_clip_file(
            entry["clip_path"],
            num_frames      = self.clip_frames,
            target_size     = self.target_size,
            temporal_jitter = self.temporal_jitter,
        )
        tensor = frames_to_tensor(frames, self.transform)
        return tensor, int(entry["label"])


# ─────────────────────────────────────────────────────────────────────────────
# Val / Test evaluation dataset  (shared protocol)
# ─────────────────────────────────────────────────────────────────────────────

class ReIDEvalDataset(Dataset):
    """
    Shared dataset for *both* validation and test evaluation.

    Val and test use the exact same re-ID protocol:
      - First clip of each cow → role = "query"   (the 10-second probe)
      - Remaining clips         → role = "gallery" (non-overlapping)

    Each item: (clip_tensor, cow_id, role)
      clip_tensor: (C, T, H, W) — no augmentation, centre-cropped.
      cow_id:      str  e.g. "07487"
      role:        "query" | "gallery"
    """

    def __init__(
        self,
        clip_entries: List[Dict],
        clip_frames:  int             = 16,
        target_size:  Tuple[int, int] = (224, 224),
    ):
        self.entries     = clip_entries
        self.clip_frames = clip_frames
        self.target_size = target_size
        self.transform   = get_frame_transform("eval", target_size[0])

    def __len__(self) -> int:
        return len(self.entries)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, str, str]:
        entry  = self.entries[idx]
        frames = load_clip_file(
            entry["clip_path"],
            num_frames      = self.clip_frames,
            target_size     = self.target_size,
            temporal_jitter = False,
        )
        tensor = frames_to_tensor(frames, self.transform)
        return tensor, entry["cow_id"], entry["role"]


# ─────────────────────────────────────────────────────────────────────────────
# P×K batch sampler
# ─────────────────────────────────────────────────────────────────────────────

class PKSampler(Sampler):
    """
    Yields batch-index lists of P*K items (P identities × K clips each).

    Sampling is with replacement so we can produce *batches_per_epoch*
    gradient steps even when the training set is small (~90 clips / 15 cows).
    """

    def __init__(
        self,
        labels:            List[int],
        P:                 int = 4,
        K:                 int = 2,
        batches_per_epoch: int = 30,
    ):
        self.P                 = P
        self.K                 = K
        self.batches_per_epoch = batches_per_epoch

        id_to_idx: Dict[int, List[int]] = defaultdict(list)
        for i, lbl in enumerate(labels):
            id_to_idx[lbl].append(i)
        self.id_to_idx  = dict(id_to_idx)
        self.unique_ids = sorted(self.id_to_idx.keys())

        if len(self.unique_ids) < P:
            raise ValueError(
                f"PKSampler: only {len(self.unique_ids)} identities but P={P}."
            )

    def __iter__(self):
        for _ in range(self.batches_per_epoch):
            ids   = random.sample(self.unique_ids, self.P)
            batch: List[int] = []
            for id_ in ids:
                avail  = self.id_to_idx[id_]
                chosen = (random.sample(avail, self.K)
                          if len(avail) >= self.K
                          else random.choices(avail, k=self.K))
                batch.extend(chosen)
            yield batch

    def __len__(self) -> int:
        return self.batches_per_epoch


# ─────────────────────────────────────────────────────────────────────────────
# Dataset factories
# ─────────────────────────────────────────────────────────────────────────────

def build_train_dataset(metadata: Dict, temporal_jitter: bool = True) -> VideoClipDataset:
    """Training dataset from metadata."""
    train_clips = [c for c in metadata["clips"] if c["split"] == "train"]
    return VideoClipDataset(
        clip_entries    = train_clips,
        clip_frames     = metadata.get("clip_frames", 16),
        temporal_jitter = temporal_jitter,
    )


def build_combined_train_val_dataset(
    metadata:        Dict,
    temporal_jitter: bool = True,
) -> VideoClipDataset:
    """
    Combined train + val dataset for final training.

    All train clips keep their original labels.
    ALL val clips (query AND gallery) are added with new integer labels
    so each val cow has multiple clips — ensuring K>=2 for PKSampler.
    """
    train_clips    = [c for c in metadata["clips"] if c["split"] == "train"]
    n_train_labels = len(metadata["train_id_to_label"])
    val_cow_ids    = metadata["cow_splits"]["val"]
    val_label_map  = {cid: n_train_labels + i
                      for i, cid in enumerate(sorted(val_cow_ids))}

    augmented_val = []
    for c in metadata["clips"]:
        if c["split"] == "val":          # include ALL val clips (query + gallery)
            c2          = dict(c)
            c2["role"]  = "train"
            c2["label"] = val_label_map[c["cow_id"]]
            augmented_val.append(c2)

    return VideoClipDataset(
        clip_entries    = train_clips + augmented_val,
        clip_frames     = metadata.get("clip_frames", 16),
        temporal_jitter = temporal_jitter,
    )


def build_eval_dataset(metadata: Dict, split: str = "test") -> ReIDEvalDataset:
    """Val or test evaluation dataset from metadata."""
    clips = [c for c in metadata["clips"] if c["split"] == split]
    return ReIDEvalDataset(
        clip_entries = clips,
        clip_frames  = metadata.get("clip_frames", 16),
    )
