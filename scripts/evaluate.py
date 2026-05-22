"""
Evaluation Module v2 — Video-Based Cow Re-Identification
=========================================================

Implements the full re-ID evaluation pipeline for BOTH val and test.
Val and test share the exact same protocol (same function, different split).

Protocol:
  Query  : first 10-second clip per cow   (role == "query")
  Gallery: all remaining clips            (role == "gallery", no overlap)

For each query clip the gallery is sorted by embedding distance.
Output shows: cow ID at Rank-1, Rank-5, Rank-10 and whether it is correct.

Metrics reported: CMC@1, CMC@5, CMC@10, mAP.

Final comparison table is printed and saved as CSV + Markdown.
"""

import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from .dataset import ReIDEvalDataset


# ─────────────────────────────────────────────────────────────────────────────
# Embedding extraction
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def extract_embeddings(
    model:      nn.Module,
    dataset:    ReIDEvalDataset,
    batch_size: int = 8,
    device:     str = "cuda",
) -> Tuple[np.ndarray, List[str], List[str]]:
    """
    Run the model over the full eval dataset and collect embeddings.

    Returns:
        embeddings: (N, D) float32 array, L2-normalised.
        cow_ids:    List[str] — one per clip.
        roles:      List[str] — "query" or "gallery" per clip.
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

    all_embs    = []
    all_cow_ids = []
    all_roles   = []

    for clips, cow_ids, roles in tqdm(loader, desc="  Extracting embeddings", leave=False):
        clips = clips.to(device, non_blocking=True)
        embs  = model(clips).cpu().float()
        all_embs.append(np.array(embs.tolist(), dtype=np.float32))
        all_cow_ids.extend(list(cow_ids))
        all_roles.extend(list(roles))

    return np.concatenate(all_embs, axis=0), all_cow_ids, all_roles


# ─────────────────────────────────────────────────────────────────────────────
# Gallery aggregation  (mean-pool clips per identity → one embedding)
# ─────────────────────────────────────────────────────────────────────────────

def aggregate_gallery(
    embeddings: np.ndarray,
    cow_ids:    List[str],
    roles:      List[str],
) -> Tuple[np.ndarray, List[str]]:
    """
    Mean-pool all gallery clips for each cow into one embedding.

    Returns:
        gallery_embs: (G, D) — one row per gallery cow identity.
        gallery_ids:  List[str] of length G.
    """
    g_mask   = np.array(roles) == "gallery"
    g_embs   = embeddings[g_mask]
    g_ids    = [cid for cid, r in zip(cow_ids, roles) if r == "gallery"]

    unique_ids = sorted(set(g_ids))
    agg        = []
    for cid in unique_ids:
        idx  = [i for i, c in enumerate(g_ids) if c == cid]
        mean = g_embs[idx].mean(axis=0)
        mean = mean / (np.linalg.norm(mean) + 1e-12)
        agg.append(mean)

    return np.stack(agg, axis=0), unique_ids


# ─────────────────────────────────────────────────────────────────────────────
# CMC + mAP
# ─────────────────────────────────────────────────────────────────────────────

def compute_cmc_map(
    dist_mat:    np.ndarray,   # (Q, G)
    query_ids:   List[str],
    gallery_ids: List[str],
    max_rank:    int = 10,
) -> Tuple[np.ndarray, float, List[Dict]]:
    """
    Compute CMC curve (up to max_rank) and mAP.

    per_query records contain the ranked gallery cow IDs so the caller can
    print "Rank-1 ID: 07487 (correct: yes)" style output.
    """
    num_query   = len(query_ids)
    g_arr       = np.array(gallery_ids)
    max_rank    = min(max_rank, len(gallery_ids))

    cmc_counts  = np.zeros(max_rank, dtype=np.float64)
    ap_scores   = []
    per_query   = []

    for qi in range(num_query):
        qid     = query_ids[qi]
        dists   = dist_mat[qi]

        sorted_idx  = np.argsort(dists)
        ranked_ids  = g_arr[sorted_idx]
        ranked_dist = dists[sorted_idx]

        matches = (ranked_ids == qid).astype(float)

        # CMC
        first_hit = int(np.argmax(matches)) if matches.sum() > 0 else max_rank
        for k in range(min(first_hit, max_rank), max_rank):
            cmc_counts[k] += 1.0

        # AP
        n_rel = matches.sum()
        if n_rel == 0:
            ap = 0.0
        else:
            prec = np.cumsum(matches) / np.arange(1, len(matches) + 1)
            ap   = float((prec * matches).sum() / n_rel)
        ap_scores.append(ap)

        # Per-query info (ranks 1, 5, 10)
        def _id_at(rank):
            return ranked_ids[rank - 1] if len(ranked_ids) >= rank else "—"
        def _correct(rank):
            return ranked_ids[rank - 1] == qid if len(ranked_ids) >= rank else False

        per_query.append({
            "query_id":       qid,
            "rank1_id":       _id_at(1),
            "rank1_correct":  bool(_correct(1)),
            "rank5_id":       _id_at(5),
            "rank5_correct":  bool(_correct(5)),
            "rank10_id":      _id_at(10),
            "rank10_correct": bool(_correct(10)),
            "first_match_rank": int(first_hit + 1),
            "ap":             round(ap, 4),
            "top10_ranked_ids": ranked_ids[:10].tolist(),
            "top10_distances":  ranked_dist[:10].tolist(),
        })

    cmc = cmc_counts / num_query
    mAP = float(np.mean(ap_scores))
    return cmc, mAP, per_query


# ─────────────────────────────────────────────────────────────────────────────
# Full evaluation pipeline
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_model(
    model:        nn.Module,
    eval_dataset: ReIDEvalDataset,
    model_name:   str,
    device:       str  = "cuda",
    batch_size:   int  = 8,
    cmc_ranks:    List[int] = [1, 5, 10],
    results_dir:  str  = "./results",
    verbose:      bool = True,
) -> Dict:
    """
    Full evaluation: extract → aggregate → distance matrix → CMC + mAP.

    Returns results dict: model_name, mAP, rank1, rank5, rank10, cmc.
    """
    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'─'*60}")
    print(f"  Evaluating: {model_name.upper()}")

    # 1. Extract embeddings
    embeddings, cow_ids, roles = extract_embeddings(
        model, eval_dataset, batch_size, device
    )
    n_q = roles.count("query")
    n_g = roles.count("gallery")
    print(f"  Queries: {n_q}  |  Gallery clips: {n_g}")

    # 2. Aggregate gallery
    gallery_embs, gallery_ids = aggregate_gallery(embeddings, cow_ids, roles)
    print(f"  Gallery identities: {len(gallery_ids)}  → {gallery_ids}")

    # 3. Query embeddings
    q_mask  = np.array(roles) == "query"
    q_embs  = embeddings[q_mask]
    q_ids   = [cid for cid, r in zip(cow_ids, roles) if r == "query"]

    # 4. Distance matrix (Euclidean on L2-normalised embeddings)
    q_sq   = (q_embs ** 2).sum(1, keepdims=True)
    g_sq   = (gallery_embs ** 2).sum(1, keepdims=True).T
    dot    = q_embs @ gallery_embs.T
    dist   = np.sqrt(np.clip(q_sq + g_sq - 2 * dot, 0, None))

    # 5. CMC + mAP
    max_rank = max(cmc_ranks)
    cmc, mAP, per_query = compute_cmc_map(dist, q_ids, gallery_ids, max_rank)
    rank_vals = {f"rank{k}": float(cmc[k - 1]) for k in cmc_ranks}

    results = {
        "model_name": model_name,
        "mAP":        round(mAP, 4),
        **{k: round(v, 4) for k, v in rank_vals.items()},
        "cmc":        cmc.tolist(),
    }

    # 6. Print results
    print(f"\n  {'─'*50}")
    print(f"  Model: {model_name.upper()}")
    print(f"  {'─'*50}")
    print(f"  mAP:     {mAP:.4f}  ({mAP*100:.1f}%)")
    for k in cmc_ranks:
        print(f"  Rank-{k:<3}: {cmc[k-1]:.4f}  ({cmc[k-1]*100:.1f}%)")
    print(f"  {'─'*50}")

    if verbose:
        header = (f"  {'Query':>8}  {'R1 ID':>8}  {'R1✓':>4}  "
                  f"{'R5 ID':>8}  {'R5✓':>4}  "
                  f"{'R10 ID':>8}  {'R10✓':>5}  {'AP':>6}")
        print(f"\n  Per-query ranking:")
        print(header)
        print(f"  {'─'*70}")
        for q in per_query:
            r1  = "✓" if q["rank1_correct"]  else "✗"
            r5  = "✓" if q["rank5_correct"]  else "✗"
            r10 = "✓" if q["rank10_correct"] else "✗"
            print(
                f"  {q['query_id']:>8}  "
                f"{q['rank1_id']:>8}  {r1:>4}  "
                f"{q['rank5_id']:>8}  {r5:>4}  "
                f"{q['rank10_id']:>8}  {r10:>5}  "
                f"{q['ap']:>6.4f}"
            )

        # Full top-10 gallery ranking per query
        print(f"\n  Full top-10 ranked gallery per query:")
        for q in per_query:
            correct_mark = "✓" if q["rank1_correct"] else "✗"
            print(f"\n  Query: {q['query_id']}  (Rank-1: {q['rank1_id']} {correct_mark})")
            for rank, (rid, dist_val) in enumerate(
                zip(q["top10_ranked_ids"], q["top10_distances"]), start=1
            ):
                match = "← CORRECT" if rid == q["query_id"] else ""
                print(f"    Rank {rank:2d}: {rid}  (dist={dist_val:.4f}) {match}")

    # 7. Save JSON
    out_json = results_dir / f"{model_name}_results.json"
    with open(out_json, "w") as f:
        json.dump({"summary": results, "per_query": per_query}, f, indent=2)
    print(f"\n  Results saved → {out_json}")

    return results


# ─────────────────────────────────────────────────────────────────────────────
# Multi-model comparison table
# ─────────────────────────────────────────────────────────────────────────────

def build_results_table(
    all_results: List[Dict],
    results_dir: str = "./results",
    cmc_ranks:   List[int] = [1, 5, 10],
) -> "pd.DataFrame":
    """Save a CSV + Markdown comparison table for all models."""
    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for r in all_results:
        row = {"Model": r["model_name"].upper(), "mAP": f"{r['mAP']*100:.2f}%"}
        for k in cmc_ranks:
            row[f"Rank-{k}"] = f"{r.get(f'rank{k}', 0)*100:.2f}%"
        rows.append(row)

    df = pd.DataFrame(rows)

    csv_path = results_dir / "comparison_table.csv"
    df.to_csv(csv_path, index=False)

    md_path = results_dir / "comparison_table.md"
    with open(md_path, "w") as f:
        f.write("# Cow Re-Identification — Final Results\n\n")
        f.write("## Test Protocol\n")
        f.write("- **Query**: first 10-second clip per test cow\n")
        f.write("- **Gallery**: all remaining clips from all test cows "
                "(non-overlapping with query)\n")
        f.write("- **Test cows**: 10  |  **Train cows**: 15  |  **Val cows**: 6\n")
        f.write("- **Transfer learning**: ImageNet-pretrained backbones\n\n")
        f.write("## Results\n\n")
        f.write(df.to_markdown(index=False))
        f.write("\n\n*mAP = mean Average Precision; "
                "Rank-k = CMC@k (fraction of queries with correct ID in top-k)*\n")

    print(f"\n{'='*60}")
    print("  FINAL COMPARISON TABLE")
    print(f"{'='*60}")
    print(df.to_markdown(index=False))
    print(f"\n  CSV  → {csv_path}")
    print(f"  MD   → {md_path}")

    return df
