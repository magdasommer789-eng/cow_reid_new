"""
Evaluation Module for Video-Based Cow Re-Identification
========================================================

Implements the full re-ID evaluation pipeline:

  1. Extract embeddings for all gallery and query clips.
  2. Aggregate clip-level embeddings into a single per-cow gallery embedding
     (mean pooling over gallery clips).
  3. Build a pairwise distance matrix: query clips vs. gallery identities.
  4. Rank gallery identities by distance for each query.
  5. Compute CMC (Cumulative Match Characteristic) and mAP.
  6. Print / save results table.

Educational Note — CMC vs. mAP
-------------------------------
CMC @ Rank-k: "Is the correct identity in the top-k results?"
  → Answers: "How often does the system find the right cow somewhere in
    the top k candidates?" Optimistic metric — any single hit counts.

mAP (mean Average Precision): measures precision at every rank position
  and averages it.  Rewards models that put the correct match at rank 1
  AND retrieve all other matching clips early.  More rigorous than CMC.

For single-gallery re-ID (one gallery clip per identity, as in this project),
mAP and CMC@1 convey similar information, but mAP remains more informative
when multiple query clips exist for the same identity.

Test Protocol:
  - Gallery:  1 embedding per test cow (averaged over gallery clips)
  - Query:    N clips per test cow (each ranked independently)
  - Ranks reported: 1, 5, 10
"""

import json
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from pathlib import Path
from typing import Dict, List, Tuple
from tqdm import tqdm


# ─────────────────────────────────────────────────────────────────────────────
# Embedding extraction
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def extract_embeddings(
    model:      torch.nn.Module,
    dataset,                          # GalleryQueryDataset
    batch_size: int = 8,
    device:     str = "cuda",
) -> Tuple[np.ndarray, List[str], List[str]]:
    """
    Run the model over the entire gallery+query dataset and collect embeddings.

    Args:
        model:      Trained embedding model in eval mode.
        dataset:    GalleryQueryDataset (returns clip, cow_id, role).
        batch_size: Clips per forward pass.
        device:     "cuda" or "cpu".

    Returns:
        embeddings: (N, D) numpy array of L2-normalised embeddings.
        cow_ids:    List[str] of cow identity for each clip.
        roles:      List[str] — "gallery" or "query" for each clip.
    """
    model.eval()
    model.to(device)

    loader = DataLoader(
        dataset,
        batch_size  = batch_size,
        shuffle     = False,
        num_workers = 0,
        pin_memory  = (device == "cuda"),
    )

    all_embeddings = []
    all_cow_ids    = []
    all_roles      = []

    for clips, cow_ids, roles in tqdm(loader, desc="Extracting embeddings"):
        clips = clips.to(device, non_blocking=True)
        embs  = model(clips)                                     # (B, D)
        # tensor.numpy() fails with PyTorch 2.2 + NumPy 2.x (binary incompatible).
        # tolist() → np.array() copies via pure Python, bypassing the C API bridge.
        import numpy as _np
        all_embeddings.append(_np.array(embs.cpu().float().tolist(), dtype=_np.float32))
        all_cow_ids.extend(list(cow_ids))
        all_roles.extend(list(roles))

    embeddings = np.concatenate(all_embeddings, axis=0)         # (N, D)
    return embeddings, all_cow_ids, all_roles


# ─────────────────────────────────────────────────────────────────────────────
# Gallery aggregation
# ─────────────────────────────────────────────────────────────────────────────

def aggregate_gallery(
    embeddings: np.ndarray,
    cow_ids:    List[str],
    roles:      List[str],
) -> Tuple[np.ndarray, List[str]]:
    """
    Compute one gallery embedding per cow identity by mean-pooling all
    gallery clips for that cow, then re-normalising.

    Args:
        embeddings: (N, D) array from extract_embeddings.
        cow_ids:    Cow identity per row.
        roles:      "gallery" or "query" per row.

    Returns:
        gallery_embs: (G, D) array — one row per unique gallery cow.
        gallery_ids:  List[str] of length G — corresponding cow identity.
    """
    gallery_mask  = np.array(roles) == "gallery"
    g_embeddings  = embeddings[gallery_mask]
    g_cow_ids     = [cid for cid, r in zip(cow_ids, roles) if r == "gallery"]

    unique_ids = sorted(set(g_cow_ids))
    agg_embs   = []

    for cid in unique_ids:
        idxs = [i for i, c in enumerate(g_cow_ids) if c == cid]
        mean_emb = g_embeddings[idxs].mean(axis=0)
        # Re-normalise after mean pooling
        mean_emb = mean_emb / (np.linalg.norm(mean_emb) + 1e-12)
        agg_embs.append(mean_emb)

    return np.stack(agg_embs, axis=0), unique_ids


# ─────────────────────────────────────────────────────────────────────────────
# Distance matrix
# ─────────────────────────────────────────────────────────────────────────────

def compute_distance_matrix(
    query_embs:   np.ndarray,    # (Q, D)
    gallery_embs: np.ndarray,    # (G, D)
    metric:       str = "euclidean",
) -> np.ndarray:
    """
    Compute pairwise distances between every query and every gallery embedding.

    Args:
        query_embs:   (Q, D) query embedding matrix.
        gallery_embs: (G, D) gallery embedding matrix.
        metric:       "euclidean" | "cosine"

    Returns:
        dist_mat: (Q, G) distance matrix — dist_mat[i, j] = d(query_i, gallery_j)
    """
    if metric == "cosine":
        # Cosine distance = 1 - cosine_similarity (embeddings already L2-normalised)
        sim  = query_embs @ gallery_embs.T                    # (Q, G)
        return 1.0 - sim

    # Euclidean: ||a - b||^2 = ||a||^2 + ||b||^2 - 2*a·b
    q_sq = (query_embs ** 2).sum(axis=1, keepdims=True)       # (Q, 1)
    g_sq = (gallery_embs ** 2).sum(axis=1, keepdims=True).T   # (1, G)
    dot  = query_embs @ gallery_embs.T                        # (Q, G)
    dist_sq = q_sq + g_sq - 2.0 * dot
    dist_sq = np.clip(dist_sq, 0, None)
    return np.sqrt(dist_sq)


# ─────────────────────────────────────────────────────────────────────────────
# CMC and mAP
# ─────────────────────────────────────────────────────────────────────────────

def compute_cmc_map(
    dist_mat:    np.ndarray,        # (Q, G)
    query_ids:   List[str],         # length Q
    gallery_ids: List[str],         # length G
    max_rank:    int = 10,
) -> Tuple[np.ndarray, float, List[Dict]]:
    """
    Compute CMC curve (up to max_rank) and mAP.

    For single-gallery re-ID (one identity per gallery slot), the CMC@k
    is simply: fraction of queries where the correct gallery entry appears
    in the top-k ranked results.

    mAP is computed per query as the average precision over the ranked list,
    then averaged across all queries.

    Args:
        dist_mat:    (Q, G) pairwise distance matrix.
        query_ids:   Cow identity label for each query clip.
        gallery_ids: Cow identity label for each gallery slot.
        max_rank:    Maximum rank to evaluate.

    Returns:
        cmc:         np.ndarray of shape (max_rank,) — CMC values at ranks 1..max_rank
        mAP:         float — mean average precision
        per_query:   List of dicts with per-query results (for inspection)
    """
    num_query   = len(query_ids)
    gallery_arr = np.array(gallery_ids)
    max_rank    = min(max_rank, len(gallery_ids))

    cmc_counts  = np.zeros(max_rank, dtype=np.float64)
    ap_scores   = []
    per_query   = []

    for q_idx in range(num_query):
        q_id      = query_ids[q_idx]
        distances = dist_mat[q_idx]                          # (G,)

        # Sort gallery by ascending distance
        sorted_indices  = np.argsort(distances)
        sorted_ids      = gallery_arr[sorted_indices]        # (G,) ranked IDs
        sorted_dists    = distances[sorted_indices]

        # Binary match indicator: 1 if gallery ID == query ID
        matches = (sorted_ids == q_id).astype(np.float64)   # (G,)

        # CMC: first match position
        first_match = np.argmax(matches)                     # index of first hit
        for k in range(first_match, max_rank):
            cmc_counts[k] += 1.0

        # Average Precision
        # AP = sum_k [ Precision@k * rel(k) ] / num_relevant
        num_relevant = matches.sum()
        if num_relevant == 0:
            ap = 0.0
        else:
            precision_at_k = np.cumsum(matches) / np.arange(1, len(matches) + 1)
            ap = (precision_at_k * matches).sum() / num_relevant
        ap_scores.append(ap)

        # Per-query record for verbose reporting
        per_query.append({
            "query_id":     q_id,
            "rank1_id":     sorted_ids[0] if len(sorted_ids) > 0 else "",
            "rank5_id":     sorted_ids[4] if len(sorted_ids) > 4 else "",
            "rank10_id":    sorted_ids[9] if len(sorted_ids) > 9 else "",
            "rank1_correct":  bool(matches[0]),
            "rank5_correct":  bool(matches[:5].max()) if len(matches) >= 5 else False,
            "rank10_correct": bool(matches[:10].max()) if len(matches) >= 10 else False,
            "ap":           float(ap),
            "first_match_rank": int(first_match + 1),
            "ranked_ids":   sorted_ids[:max_rank].tolist(),
            "ranked_dists": sorted_dists[:max_rank].tolist(),
        })

    cmc = cmc_counts / num_query                             # normalise to [0,1]
    mAP = float(np.mean(ap_scores))

    return cmc, mAP, per_query


# ─────────────────────────────────────────────────────────────────────────────
# Full evaluation pipeline
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_model(
    model:           torch.nn.Module,
    eval_dataset,
    model_name:      str,
    device:          str  = "cuda",
    batch_size:      int  = 8,
    cmc_ranks:       List[int] = [1, 5, 10],
    results_dir:     str  = "./results",
    verbose:         bool = True,
) -> Dict:
    """
    Complete evaluation: extract embeddings → build distance matrix →
    compute CMC + mAP → print and save results.

    Args:
        model:        Trained embedding model.
        eval_dataset: GalleryQueryDataset.
        model_name:   String for logging/filenames.
        device:       "cuda" or "cpu".
        batch_size:   Clips per forward pass during extraction.
        cmc_ranks:    Which rank values to report.
        results_dir:  Directory for JSON + CSV output.
        verbose:      Print per-query breakdown.

    Returns:
        results dict with keys: model_name, mAP, rank1, rank5, rank10, cmc.
    """
    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    # ── Step 1: extract embeddings ────────────────────────────────────────────
    print(f"\n[{model_name}] Extracting embeddings...")
    embeddings, cow_ids, roles = extract_embeddings(
        model, eval_dataset, batch_size=batch_size, device=device
    )

    gallery_count = roles.count("gallery")
    query_count   = roles.count("query")
    print(f"  Gallery clips: {gallery_count}  |  Query clips: {query_count}")

    # ── Step 2: aggregate gallery to one embedding per cow ────────────────────
    gallery_embs, gallery_ids = aggregate_gallery(embeddings, cow_ids, roles)
    print(f"  Gallery identities: {len(gallery_ids)}  →  {gallery_ids}")

    # ── Step 3: separate query embeddings and IDs ─────────────────────────────
    query_mask  = np.array(roles) == "query"
    query_embs  = embeddings[query_mask]
    query_ids   = [cid for cid, r in zip(cow_ids, roles) if r == "query"]
    print(f"  Query clips total: {len(query_ids)}")

    # ── Step 4: distance matrix ───────────────────────────────────────────────
    dist_mat = compute_distance_matrix(query_embs, gallery_embs)

    # ── Step 5: CMC + mAP ─────────────────────────────────────────────────────
    max_rank = max(cmc_ranks)
    cmc, mAP, per_query = compute_cmc_map(
        dist_mat, query_ids, gallery_ids, max_rank=max_rank
    )

    rank_vals = {f"rank{k}": float(cmc[k - 1]) for k in cmc_ranks}

    results = {
        "model_name": model_name,
        "mAP":        round(mAP, 4),
        **{k: round(v, 4) for k, v in rank_vals.items()},
        "cmc":        cmc.tolist(),
    }

    # ── Step 6: print results ─────────────────────────────────────────────────
    print(f"\n{'─'*55}")
    print(f"  Results for: {model_name.upper()}")
    print(f"{'─'*55}")
    print(f"  mAP:    {mAP:.4f}  ({mAP*100:.2f}%)")
    for k in cmc_ranks:
        print(f"  Rank-{k:<3}: {cmc[k-1]:.4f}  ({cmc[k-1]*100:.2f}%)")
    print(f"{'─'*55}")

    if verbose:
        print(f"\n  Per-query breakdown (first 10 shown):")
        print(f"  {'Query':<15} {'Rank-1':>8} {'R1 ID':<15} "
              f"{'R5 ID':<15} {'R10 ID':<15} {'AP':>6}")
        print(f"  {'-'*75}")
        for q in per_query[:10]:
            correct = "✓" if q["rank1_correct"] else "✗"
            print(f"  {q['query_id']:<15} {correct:>8}  "
                  f"{q['rank1_id']:<15} {q['rank5_id']:<15} "
                  f"{q['rank10_id']:<15} {q['ap']:>6.4f}")

    # ── Step 7: save outputs ──────────────────────────────────────────────────
    json_path = results_dir / f"{model_name}_results.json"
    with open(json_path, "w") as f:
        json.dump({"summary": results, "per_query": per_query}, f, indent=2)

    return results


# ─────────────────────────────────────────────────────────────────────────────
# Results table builder
# ─────────────────────────────────────────────────────────────────────────────

def build_results_table(
    all_results: List[Dict],
    results_dir: str = "./results",
    cmc_ranks:   List[int] = [1, 5, 10],
) -> pd.DataFrame:
    """
    Combine results from all models into a single comparison table.

    Saves both a CSV and a Markdown table to results_dir.

    Args:
        all_results: List of result dicts from evaluate_model().
        results_dir: Output directory.
        cmc_ranks:   Rank values to include.

    Returns:
        pandas DataFrame with one row per model.
    """
    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for r in all_results:
        row = {
            "Model":   r["model_name"].upper(),
            "mAP":     f"{r['mAP']*100:.2f}%",
        }
        for k in cmc_ranks:
            row[f"Rank-{k}"] = f"{r.get(f'rank{k}', 0)*100:.2f}%"
        rows.append(row)

    df = pd.DataFrame(rows)

    # CSV
    csv_path = results_dir / "comparison_table.csv"
    df.to_csv(csv_path, index=False)

    # Markdown table
    md_path  = results_dir / "comparison_table.md"
    with open(md_path, "w") as f:
        f.write("# Cow Re-Identification — Model Comparison Results\n\n")
        f.write("## Test Protocol\n")
        f.write("- **Gallery**: First 10 seconds of each test cow video "
                "(averaged clip embeddings)\n")
        f.write("- **Query**: Remaining video clips (non-overlapping with gallery)\n")
        f.write("- **Test cows**: 10 (unseen during training)\n")
        f.write("- **Train cows**: 21\n\n")
        f.write("## Results\n\n")
        f.write(df.to_markdown(index=False))
        f.write("\n\n*mAP = mean Average Precision;  "
                "Rank-k = Cumulative Match Characteristic at rank k*\n")

    print(f"\nResults table saved:")
    print(f"  CSV:      {csv_path}")
    print(f"  Markdown: {md_path}")
    print(f"\n{df.to_markdown(index=False)}")

    return df


# ─────────────────────────────────────────────────────────────────────────────
# CLI entry point
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate a trained cow re-ID model (CMC + mAP)."
    )
    parser.add_argument("--model",      required=True,
                        choices=["c3d", "x3d", "swin", "vivit"],
                        help="Model architecture to evaluate.")
    parser.add_argument("--checkpoint", required=True,
                        help="Path to the .pt checkpoint file.")
    parser.add_argument("--config",     default="./configs/config.yaml")
    parser.add_argument("--device",     default="cuda")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--results_dir",default="./results")
    parser.add_argument("--no_verbose", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    import yaml
    from .dataset import build_eval_dataset
    from .models_cnn import create_cnn_model
    from .models_transformer import create_transformer_model

    args   = parse_args()
    config = yaml.safe_load(open(args.config))

    # Load dataset metadata
    meta_path = Path(config["data"]["processed_dir"]) / "dataset_metadata.json"
    with open(meta_path) as f:
        metadata = json.load(f)

    img_size   = config["data"]["target_size"][0]
    eval_ds    = build_eval_dataset(metadata, img_size=img_size)

    # Load model
    mc = config["model"]
    if args.model in ("c3d", "x3d"):
        model = create_cnn_model(
            args.model,
            embedding_dim   = mc["embedding_dim"],
            pretrained      = False,
            device          = args.device,
            model_size      = mc["x3d"]["model_size"] if args.model == "x3d" else "m",
        )
    else:
        model = create_transformer_model(
            args.model,
            embedding_dim   = mc["embedding_dim"],
            pretrained      = False,
            device          = args.device,
            num_frames      = config["data"]["clip_frames"],
        )

    # Load checkpoint weights
    ckpt = torch.load(args.checkpoint, map_location=args.device)
    model.load_state_dict(ckpt["model_state"])
    print(f"Loaded checkpoint from epoch {ckpt.get('epoch', '?')}")

    evaluate_model(
        model        = model,
        eval_dataset = eval_ds,
        model_name   = args.model,
        device       = args.device,
        batch_size   = args.batch_size,
        cmc_ranks    = config["evaluation"]["cmc_ranks"],
        results_dir  = args.results_dir,
        verbose      = not args.no_verbose,
    )
