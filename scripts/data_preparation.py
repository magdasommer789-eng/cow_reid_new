"""
Data Preparation v2 — Video-Based Cow Re-Identification
=========================================================

Pipeline:
  1. Check free disk space on /local1 before writing anything.
  2. Discover all 31 cow videos in cow_videos/.
  3. Deterministic 3-way split: 15 train / 6 val / 10 test (seeded by identity).
  4. Extract non-overlapping 10-second MP4 clips via ffmpeg:
       /local1/cow_clips/{split}/{cow_id}/{cow_id}_{N:03d}.mp4
  5. For val and test cows the first clip (N=001) is the *query*;
     all remaining clips are *gallery*.  Train clips have role="train".
  6. Save a flat metadata JSON: data/processed/dataset_metadata.json.

Usage:
  python -m scripts.data_preparation
  python -m scripts.data_preparation --video_dir /path/to/videos
"""

import argparse
import json
import os
import random
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2


def _get_ffmpeg() -> str:
    """Return path to ffmpeg binary (system or bundled via imageio-ffmpeg)."""
    import shutil as _shutil
    system_ff = _shutil.which("ffmpeg")
    if system_ff:
        return system_ff
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except ImportError:
        raise RuntimeError(
            "ffmpeg not found. Install it (apt install ffmpeg) or "
            "pip install imageio-ffmpeg."
        )


FFMPEG = _get_ffmpeg()


# ─────────────────────────────────────────────────────────────────────────────
# Safety: disk-space guard
# ─────────────────────────────────────────────────────────────────────────────

def check_disk_space(path: str, min_gb: float = 5.0) -> None:
    """Raise RuntimeError if free space at *path* is below *min_gb* GB."""
    stat = shutil.disk_usage(path)
    free_gb  = stat.free  / (1024 ** 3)
    total_gb = stat.total / (1024 ** 3)
    print(f"  Disk [{path}]: {free_gb:.1f} GB free / {total_gb:.1f} GB total")
    if free_gb < min_gb:
        raise RuntimeError(
            f"Disk space too low at {path}: {free_gb:.1f} GB free "
            f"(need ≥ {min_gb:.1f} GB). Aborting to prevent system crash."
        )


# ─────────────────────────────────────────────────────────────────────────────
# Video discovery
# ─────────────────────────────────────────────────────────────────────────────

_VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".wmv", ".MP4", ".AVI", ".MOV", ".MKV"}


def discover_videos(video_dir: str) -> Dict[str, str]:
    """Return {cow_id: absolute_path} for every video in *video_dir*."""
    p = Path(video_dir)
    if not p.exists():
        raise FileNotFoundError(f"Video directory not found: {p}")
    result = {}
    for f in sorted(p.iterdir()):
        if f.suffix in _VIDEO_EXTS:
            result[f.stem] = str(f.resolve())
    print(f"Found {len(result)} cow videos in {p}")
    return result


# ─────────────────────────────────────────────────────────────────────────────
# 3-way split
# ─────────────────────────────────────────────────────────────────────────────

def split_cows(
    cow_ids:       List[str],
    num_train:     int,
    num_val:       int,
    seed:          int = 42,
) -> Tuple[List[str], List[str], List[str]]:
    """
    Randomly partition cow IDs into train / val / test (test = remainder).

    Sorting before shuffle ensures the split is deterministic across platforms.
    """
    ids = sorted(cow_ids)
    rng = random.Random(seed)
    rng.shuffle(ids)

    train_ids = sorted(ids[:num_train])
    val_ids   = sorted(ids[num_train : num_train + num_val])
    test_ids  = sorted(ids[num_train + num_val :])

    print(f"\nSplit (seed={seed}):")
    print(f"  Train ({len(train_ids):2d}): {train_ids}")
    print(f"  Val   ({len(val_ids):2d}):   {val_ids}")
    print(f"  Test  ({len(test_ids):2d}):  {test_ids}")
    return train_ids, val_ids, test_ids


# ─────────────────────────────────────────────────────────────────────────────
# Video metadata
# ─────────────────────────────────────────────────────────────────────────────

def get_video_duration(video_path: str) -> Tuple[float, float]:
    """Return (fps, duration_seconds) using OpenCV."""
    cap = cv2.VideoCapture(video_path)
    fps    = cap.get(cv2.CAP_PROP_FPS)
    frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    duration = frames / fps if fps > 0 else 0.0
    return fps, duration


# ─────────────────────────────────────────────────────────────────────────────
# Clip extraction
# ─────────────────────────────────────────────────────────────────────────────

def extract_clips(
    video_path:   str,
    cow_id:       str,
    split:        str,
    clips_dir:    str,
    clip_seconds: float = 10.0,
) -> List[Dict]:
    """
    Cut *video_path* into non-overlapping 10-second clips using ffmpeg.

    Output path: {clips_dir}/{split}/{cow_id}/{cow_id}_{N:03d}.mp4
    Clips are re-encoded at 256×256 / libx264 / ultrafast / crf28 (≈1–2 MB each).
    Already-existing clips are skipped to allow re-runs without re-encoding.

    Returns a list of clip-info dicts (path, number, cow_id, split).
    """
    out_dir = Path(clips_dir) / split / cow_id
    out_dir.mkdir(parents=True, exist_ok=True)

    fps, duration = get_video_duration(video_path)
    n_clips = int(duration // clip_seconds)

    if n_clips == 0:
        print(f"  WARNING: {cow_id} is only {duration:.1f}s — skipping (too short).")
        return []

    clips = []
    for i in range(n_clips):
        start_s      = i * clip_seconds
        clip_number  = i + 1
        out_path     = out_dir / f"{cow_id}_{clip_number:03d}.mp4"

        if not out_path.exists():
            cmd = [
                FFMPEG, "-y",
                "-ss", str(start_s),       # fast input-seek (keyframe accurate)
                "-i", video_path,
                "-t", str(clip_seconds),
                "-vf", "scale=256:256",    # resize; model will crop to 224
                "-c:v", "libx264",
                "-preset", "ultrafast",
                "-crf", "28",
                "-an",                     # drop audio track
                str(out_path),
            ]
            proc = subprocess.run(cmd, capture_output=True, text=True)
            if proc.returncode != 0:
                print(f"  WARNING: ffmpeg failed for {out_path.name}:\n"
                      f"    {proc.stderr[-300:]}")
                continue

        clips.append({
            "cow_id":      cow_id,
            "split":       split,
            "clip_path":   str(out_path),
            "clip_number": clip_number,
        })

    return clips


# ─────────────────────────────────────────────────────────────────────────────
# Main pipeline
# ─────────────────────────────────────────────────────────────────────────────

def prepare_dataset(
    video_dir:      str  = "/home/hswts124607/cow_videos",
    clips_dir:      str  = "/local1/cow_clips",
    processed_dir:  str  = "./data/processed",
    num_train_cows: int  = 15,
    num_val_cows:   int  = 6,
    clip_seconds:   float = 10.0,
    seed:           int  = 42,
) -> Dict:
    """
    Full preparation pipeline. Safe to re-run: clips already on disk are skipped.
    """
    print("=" * 60)
    print("Cow Re-ID — Data Preparation v2")
    print("=" * 60)

    # 1. Disk-space guard BEFORE creating any files
    print("\n[1/5] Checking disk space …")
    # Resolve mount point: /local1/cow_clips → /local1
    mount = "/" + clips_dir.lstrip("/").split("/")[0]
    check_disk_space(mount, min_gb=5.0)

    Path(processed_dir).mkdir(parents=True, exist_ok=True)

    # 2. Discover videos
    print("\n[2/5] Discovering videos …")
    cow_videos = discover_videos(video_dir)
    all_ids    = sorted(cow_videos.keys())
    if len(all_ids) != 31:
        print(f"  NOTE: expected 31 cows, found {len(all_ids)}")

    # 3. Split
    print("\n[3/5] Splitting cows …")
    train_ids, val_ids, test_ids = split_cows(
        all_ids, num_train_cows, num_val_cows, seed
    )
    train_label = {cid: i for i, cid in enumerate(sorted(train_ids))}

    # 4. Extract clips
    print("\n[4/5] Extracting clips (ffmpeg) …")
    all_clips: List[Dict] = []

    for split_name, cow_ids in [
        ("train", train_ids),
        ("val",   val_ids),
        ("test",  test_ids),
    ]:
        print(f"\n  [{split_name.upper()}]")
        for cow_id in cow_ids:
            clips = extract_clips(
                cow_videos[cow_id], cow_id, split_name, clips_dir, clip_seconds
            )
            # Assign roles: train clips all get role="train";
            # for val/test the FIRST clip is the query, rest are gallery.
            for j, c in enumerate(clips):
                if split_name == "train":
                    c["role"]  = "train"
                    c["label"] = train_label[cow_id]
                else:
                    c["role"]  = "query" if j == 0 else "gallery"
                    c["label"] = None
            all_clips.extend(clips)
            role_str = (f"{len(clips)} train"
                        if split_name == "train"
                        else f"1 query + {len(clips)-1} gallery")
            print(f"    {cow_id}: {len(clips)} clips  ({role_str})")

    # 5. Build and save metadata
    print("\n[5/5] Saving metadata …")

    def _count(sp, rl=None):
        return sum(
            1 for c in all_clips
            if c["split"] == sp and (rl is None or c["role"] == rl)
        )

    metadata = {
        "version":         "2.0",
        "video_dir":       str(Path(video_dir).resolve()),
        "clips_dir":       str(Path(clips_dir).resolve()),
        "seed":            seed,
        "clip_seconds":    clip_seconds,
        "cow_splits": {
            "train": train_ids,
            "val":   val_ids,
            "test":  test_ids,
        },
        "train_id_to_label": train_label,
        "summary": {
            "train": {
                "n_cows":   len(train_ids),
                "n_clips":  _count("train"),
            },
            "val": {
                "n_cows":   len(val_ids),
                "n_query":  _count("val",  "query"),
                "n_gallery":_count("val",  "gallery"),
                "n_clips":  _count("val"),
            },
            "test": {
                "n_cows":   len(test_ids),
                "n_query":  _count("test", "query"),
                "n_gallery":_count("test", "gallery"),
                "n_clips":  _count("test"),
            },
        },
        "clips": all_clips,
    }

    out_path = Path(processed_dir) / "dataset_metadata.json"
    with open(out_path, "w") as f:
        json.dump(metadata, f, indent=2)

    s = metadata["summary"]
    print(f"\nDataset metadata → {out_path}")
    print(f"  Train : {s['train']['n_cows']} cows, {s['train']['n_clips']} clips")
    print(f"  Val   : {s['val']['n_cows']} cows, "
          f"{s['val']['n_query']} queries + {s['val']['n_gallery']} gallery")
    print(f"  Test  : {s['test']['n_cows']} cows, "
          f"{s['test']['n_query']} queries + {s['test']['n_gallery']} gallery")

    return metadata


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def _parse_args():
    p = argparse.ArgumentParser(description="Prepare cow re-ID dataset (v2).")
    p.add_argument("--video_dir",      default="/home/hswts124607/cow_videos")
    p.add_argument("--clips_dir",      default="/local1/cow_clips")
    p.add_argument("--processed_dir",  default="./data/processed")
    p.add_argument("--num_train_cows", type=int,   default=15)
    p.add_argument("--num_val_cows",   type=int,   default=6)
    p.add_argument("--clip_seconds",   type=float, default=10.0)
    p.add_argument("--seed",           type=int,   default=42)
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    prepare_dataset(
        video_dir      = args.video_dir,
        clips_dir      = args.clips_dir,
        processed_dir  = args.processed_dir,
        num_train_cows = args.num_train_cows,
        num_val_cows   = args.num_val_cows,
        clip_seconds   = args.clip_seconds,
        seed           = args.seed,
    )
