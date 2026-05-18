"""
Data Preparation for Video-Based Cow Re-Identification
=======================================================

This script handles all preprocessing before training:

1. Discovers all cow videos in /data (one video = one cow identity)
2. Splits cows into train (21) and test (10) sets — split by identity,
   so the model never sees test cow identities during training.
3. For test cows, creates a strict gallery/query split:
   - Gallery: first 10 seconds (used as the "known" reference)
   - Query:   everything after 10 seconds (used to search against gallery)
4. Saves a metadata JSON that the dataset loader reads at runtime.

Educational Note:
  Open-set Re-ID vs. Closed-set Re-ID
  ------------------------------------
  Closed-set: all identities are known at training time (classification works).
  Open-set:   test identities are UNSEEN during training → we need embeddings.
  This project is open-set: train and test cow IDs never overlap.
"""

import os
import json
import random
import argparse
import cv2
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple


# ─────────────────────────────────────────────────────────────────────────────
# Video discovery
# ─────────────────────────────────────────────────────────────────────────────

VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".wmv", ".flv", ".MOV"}


def discover_videos(video_dir: str) -> Dict[str, str]:
    """
    Scan video_dir and return a mapping {cow_id: video_path}.

    The cow identity is derived from the video filename (without extension),
    matching the project convention: "The id of the cow is always the name
    of the video."

    Args:
        video_dir: Directory containing one video file per cow.

    Returns:
        Dictionary mapping cow_id (str) → absolute video path (str).
    """
    video_dir = Path(video_dir)
    if not video_dir.exists():
        raise FileNotFoundError(f"Video directory not found: {video_dir}")

    cow_videos: Dict[str, str] = {}
    for path in sorted(video_dir.iterdir()):
        if path.suffix in VIDEO_EXTENSIONS:
            cow_id = path.stem          # filename without extension = cow ID
            cow_videos[cow_id] = str(path.resolve())

    print(f"Found {len(cow_videos)} cow videos in {video_dir}")
    return cow_videos


# ─────────────────────────────────────────────────────────────────────────────
# Train / Test split (by identity)
# ─────────────────────────────────────────────────────────────────────────────

def split_cows(
    cow_ids: List[str],
    num_train: int = 21,
    seed: int = 42
) -> Tuple[List[str], List[str]]:
    """
    Randomly partition cow identities into train and test sets.

    The split is deterministic (seeded) so experiments are reproducible.
    Train and test cow IDs are DISJOINT — this is the core open-set property.

    Args:
        cow_ids:   Sorted list of all cow identity strings.
        num_train: How many cows go into the training set (default 21).
        seed:      Random seed for reproducibility.

    Returns:
        (train_ids, test_ids) — two disjoint lists of cow identity strings.
    """
    if len(cow_ids) < num_train:
        raise ValueError(
            f"Not enough cows ({len(cow_ids)}) for {num_train} train identities."
        )

    ids = sorted(cow_ids)           # Sort first so the seed is truly deterministic
    rng = random.Random(seed)
    rng.shuffle(ids)

    train_ids = sorted(ids[:num_train])
    test_ids  = sorted(ids[num_train:])

    print(f"Train cows ({len(train_ids)}): {train_ids}")
    print(f"Test  cows ({len(test_ids)}):  {test_ids}")
    return train_ids, test_ids


# ─────────────────────────────────────────────────────────────────────────────
# Video metadata helpers
# ─────────────────────────────────────────────────────────────────────────────

def get_video_metadata(video_path: str) -> Dict:
    """
    Read basic metadata (fps, total frames, duration) from a video file.

    Args:
        video_path: Path to the video file.

    Returns:
        Dict with keys: fps, total_frames, duration_seconds.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"Cannot open video: {video_path}")

    fps           = cap.get(cv2.CAP_PROP_FPS)
    total_frames  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration      = total_frames / fps if fps > 0 else 0.0
    cap.release()

    return {
        "fps":              fps,
        "total_frames":     total_frames,
        "duration_seconds": duration,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Clip index generation
# ─────────────────────────────────────────────────────────────────────────────

def generate_clip_indices(
    start_frame: int,
    end_frame: int,
    clip_frames: int,
    clip_stride: int,
) -> List[Tuple[int, int]]:
    """
    Generate (start, end) frame index pairs for non-overlapping clips.

    Args:
        start_frame: First frame index to consider (inclusive).
        end_frame:   Last frame index to consider (exclusive).
        clip_frames: Number of frames in each clip.
        clip_stride: Step between consecutive clip start frames.

    Returns:
        List of (clip_start, clip_end) tuples.
    """
    clips = []
    pos = start_frame
    while pos + clip_frames <= end_frame:
        clips.append((pos, pos + clip_frames))
        pos += clip_stride
    return clips


def build_clip_list(
    cow_id: str,
    video_path: str,
    start_frame: int,
    end_frame: int,
    clip_frames: int,
    clip_stride: int,
    split: str,
    role: str = "train",        # "train" | "gallery" | "query"
) -> List[Dict]:
    """
    Build a list of clip descriptors for one cow's video segment.

    Each descriptor is a lightweight dict — no frames are loaded here;
    loading happens in the dataset at training/evaluation time.

    Args:
        cow_id:      Identity label (filename stem).
        video_path:  Absolute path to the video file.
        start_frame: First usable frame index.
        end_frame:   Last usable frame index (exclusive).
        clip_frames: Frames per clip.
        clip_stride: Stride between clip start frames.
        split:       "train" or "test".
        role:        "train" | "gallery" | "query".

    Returns:
        List of clip descriptor dicts.
    """
    indices = generate_clip_indices(start_frame, end_frame, clip_frames, clip_stride)
    clips = []
    for start, end in indices:
        clips.append({
            "cow_id":      cow_id,
            "video_path":  video_path,
            "start_frame": start,
            "end_frame":   end,
            "clip_frames": clip_frames,
            "split":       split,
            "role":        role,
        })
    return clips


# ─────────────────────────────────────────────────────────────────────────────
# Gallery / Query split for test cows
# ─────────────────────────────────────────────────────────────────────────────

def create_gallery_query_split(
    cow_id: str,
    video_path: str,
    gallery_seconds: float,
    fps: float,
    total_frames: int,
    clip_frames: int,
    clip_stride: int,
) -> Tuple[List[Dict], List[Dict]]:
    """
    Split a test cow's video into gallery and query segments.

    Gallery: frames [0, gallery_end)   — first gallery_seconds of video
    Query:   frames [gallery_end, end) — strict temporal separation

    The boundary is chosen so no frame appears in both gallery and query.

    Args:
        cow_id:          Cow identity.
        video_path:      Path to video.
        gallery_seconds: Duration (in seconds) allocated to the gallery.
        fps:             Frames per second of the video.
        total_frames:    Total frame count.
        clip_frames:     Frames per clip.
        clip_stride:     Stride between clip starts.

    Returns:
        (gallery_clips, query_clips)
    """
    gallery_end = int(gallery_seconds * fps)
    # Ensure gallery_end is a multiple of clip_frames for clean clips
    gallery_end = (gallery_end // clip_frames) * clip_frames
    gallery_end = min(gallery_end, total_frames)

    if gallery_end <= 0:
        raise ValueError(
            f"Cow {cow_id}: gallery_end={gallery_end} is invalid "
            f"(fps={fps:.1f}, total_frames={total_frames})"
        )

    query_start = gallery_end          # Strictly after gallery — no overlap

    gallery_clips = build_clip_list(
        cow_id, video_path,
        start_frame=0,
        end_frame=gallery_end,
        clip_frames=clip_frames,
        clip_stride=clip_stride,
        split="test",
        role="gallery",
    )

    query_clips = build_clip_list(
        cow_id, video_path,
        start_frame=query_start,
        end_frame=total_frames,
        clip_frames=clip_frames,
        clip_stride=clip_stride,
        split="test",
        role="query",
    )

    if not gallery_clips:
        print(f"  WARNING: Cow {cow_id} has no gallery clips — video too short?")
    if not query_clips:
        print(f"  WARNING: Cow {cow_id} has no query clips — extend recording?")

    return gallery_clips, query_clips


# ─────────────────────────────────────────────────────────────────────────────
# Main preparation pipeline
# ─────────────────────────────────────────────────────────────────────────────

def prepare_dataset(
    video_dir: str,
    processed_dir: str,
    num_train_cows: int = 21,
    gallery_seconds: float = 10.0,
    clip_frames: int = 16,
    clip_stride: int = 8,
    seed: int = 42,
) -> Dict:
    """
    Full preparation pipeline.

    Steps:
      1. Discover all cow videos.
      2. Split cow IDs into train / test.
      3. Build training clip list (all frames, all train cows).
      4. Build gallery and query clip lists for test cows.
      5. Assemble and save metadata JSON.

    Args:
        video_dir:       Directory with raw cow videos.
        processed_dir:   Output directory for metadata JSON.
        num_train_cows:  Number of cows in the training set.
        gallery_seconds: Seconds of each test video used as gallery.
        clip_frames:     Frames per video clip.
        clip_stride:     Stride between clip starts.
        seed:            Random seed for train/test split.

    Returns:
        Metadata dictionary (also saved to disk as JSON).
    """
    processed_dir = Path(processed_dir)
    processed_dir.mkdir(parents=True, exist_ok=True)

    # ── Step 1: discover ──────────────────────────────────────────────────────
    cow_videos = discover_videos(video_dir)
    all_cow_ids = sorted(cow_videos.keys())
    total_cows = len(all_cow_ids)
    print(f"\nTotal cows: {total_cows}")

    # ── Step 2: split ─────────────────────────────────────────────────────────
    train_ids, test_ids = split_cows(all_cow_ids, num_train=num_train_cows, seed=seed)

    # Assign integer class indices to TRAIN identities only.
    # Test identities are deliberately excluded from the index
    # because the model never classifies them — it ranks by embedding distance.
    train_id_to_idx = {cid: i for i, cid in enumerate(sorted(train_ids))}

    # ── Step 3: training clips ────────────────────────────────────────────────
    print("\nBuilding training clip list...")
    train_clips: List[Dict] = []
    for cow_id in train_ids:
        video_path = cow_videos[cow_id]
        meta = get_video_metadata(video_path)
        clips = build_clip_list(
            cow_id=cow_id,
            video_path=video_path,
            start_frame=0,
            end_frame=meta["total_frames"],
            clip_frames=clip_frames,
            clip_stride=clip_stride,
            split="train",
            role="train",
        )
        # Attach integer label for triplet sampling
        for c in clips:
            c["label"] = train_id_to_idx[cow_id]
        train_clips.extend(clips)
        print(f"  {cow_id}: {meta['duration_seconds']:.1f}s  "
              f"→ {len(clips)} clips  (label={train_id_to_idx[cow_id]})")

    # ── Step 4: test gallery + query ──────────────────────────────────────────
    print("\nBuilding gallery and query clip lists...")
    gallery_clips: List[Dict] = []
    query_clips:   List[Dict] = []

    for cow_id in test_ids:
        video_path = cow_videos[cow_id]
        meta = get_video_metadata(video_path)

        g_clips, q_clips = create_gallery_query_split(
            cow_id=cow_id,
            video_path=video_path,
            gallery_seconds=gallery_seconds,
            fps=meta["fps"],
            total_frames=meta["total_frames"],
            clip_frames=clip_frames,
            clip_stride=clip_stride,
        )

        gallery_clips.extend(g_clips)
        query_clips.extend(q_clips)
        print(f"  {cow_id}: {meta['duration_seconds']:.1f}s  "
              f"→ gallery={len(g_clips)} clips, query={len(q_clips)} clips")

    # ── Step 5: assemble and save ─────────────────────────────────────────────
    metadata = {
        "version":           "1.0",
        "video_dir":         str(Path(video_dir).resolve()),
        "total_cows":        total_cows,
        "num_train_cows":    len(train_ids),
        "num_test_cows":     len(test_ids),
        "train_cow_ids":     train_ids,
        "test_cow_ids":      test_ids,
        "train_id_to_idx":   train_id_to_idx,
        "clip_frames":       clip_frames,
        "clip_stride":       clip_stride,
        "gallery_seconds":   gallery_seconds,
        "splits": {
            "train":   {"clips": train_clips,   "num_clips": len(train_clips)},
            "gallery": {"clips": gallery_clips, "num_clips": len(gallery_clips)},
            "query":   {"clips": query_clips,   "num_clips": len(query_clips)},
        },
    }

    out_path = processed_dir / "dataset_metadata.json"
    with open(out_path, "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"\nDataset metadata saved to {out_path}")
    print(f"  Train clips:   {len(train_clips)}")
    print(f"  Gallery clips: {len(gallery_clips)}")
    print(f"  Query clips:   {len(query_clips)}")

    return metadata


# ─────────────────────────────────────────────────────────────────────────────
# CLI entry point
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description="Prepare the cow re-ID dataset (split, gallery/query creation)."
    )
    parser.add_argument("--video_dir",       default="/data",
                        help="Directory containing one .mp4 per cow.")
    parser.add_argument("--processed_dir",   default="./data/processed",
                        help="Output directory for metadata JSON.")
    parser.add_argument("--num_train_cows",  type=int, default=21)
    parser.add_argument("--gallery_seconds", type=float, default=10.0)
    parser.add_argument("--clip_frames",     type=int, default=16)
    parser.add_argument("--clip_stride",     type=int, default=8)
    parser.add_argument("--seed",            type=int, default=42)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    prepare_dataset(
        video_dir       = args.video_dir,
        processed_dir   = args.processed_dir,
        num_train_cows  = args.num_train_cows,
        gallery_seconds = args.gallery_seconds,
        clip_frames     = args.clip_frames,
        clip_stride     = args.clip_stride,
        seed            = args.seed,
    )
