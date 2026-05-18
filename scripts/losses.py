"""
Metric Learning Losses for Video-Based Cow Re-Identification
=============================================================

Implements Batch Hard Triplet Loss — the standard loss for Re-ID tasks.

Educational Note — Why Triplet Loss for Re-ID?
------------------------------------------------
Cross-entropy loss trains a classifier: it optimises class probabilities.
At test time the final classification layer is discarded and only the
intermediate embedding is used, which it was never directly optimised for.

Triplet loss directly optimises the embedding space:
  "Pull same-identity embeddings together, push different-identity apart."

The key insight: we don't need identity labels at inference.  The model
learns a universal distance metric — whichever two clips are "closest" in
embedding space are the most likely to show the same animal.

Batch Hard Mining:
  For each anchor A in the batch, choose:
    - Hardest positive  P*  = max_{p: y_p == y_a} d(A, P)
    - Hardest negative  N*  = min_{n: y_n != y_a} d(A, N)
  Loss = max(0, d(A,P*) - d(A,N*) + margin)

This focuses learning on the most informative triplets and converges faster
than random triplet sampling.

Reference: Hermans et al., "In Defense of the Triplet Loss for Person Re-ID"
           https://arxiv.org/abs/1703.07737
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


# ─────────────────────────────────────────────────────────────────────────────
# Distance utilities
# ─────────────────────────────────────────────────────────────────────────────

def euclidean_distance_matrix(embeddings: torch.Tensor) -> torch.Tensor:
    """
    Compute pairwise Euclidean distance matrix.

    For a batch of N L2-normalised embeddings this is equivalent to
    2 - 2*cosine_similarity (since ||e|| = 1 for all e).

    Args:
        embeddings: (N, D) float tensor of L2-normalised embeddings.

    Returns:
        (N, N) distance matrix, dist[i, j] = ||e_i - e_j||_2
    """
    # Numerically stable: ||a-b||^2 = ||a||^2 + ||b||^2 - 2*a·b
    dot_product = torch.mm(embeddings, embeddings.t())        # (N, N)
    sq_norm     = torch.diag(dot_product)                     # (N,)
    dist_sq     = sq_norm.unsqueeze(1) + sq_norm.unsqueeze(0) - 2.0 * dot_product
    dist_sq     = dist_sq.clamp(min=1e-12)                    # numerical safety
    return torch.sqrt(dist_sq)                                # (N, N)


def cosine_distance_matrix(embeddings: torch.Tensor) -> torch.Tensor:
    """
    Compute pairwise cosine distance matrix (= 1 − cosine_similarity).

    Useful as an alternative to Euclidean distance.  For L2-normalised
    embeddings both metrics are monotonically related, so they produce
    identical rankings.

    Args:
        embeddings: (N, D) float tensor (need not be normalised).

    Returns:
        (N, N) cosine distance matrix in [0, 2].
    """
    normed = F.normalize(embeddings, p=2, dim=1)
    sim    = torch.mm(normed, normed.t()).clamp(-1.0, 1.0)
    return 1.0 - sim


# ─────────────────────────────────────────────────────────────────────────────
# Batch Hard Triplet Loss
# ─────────────────────────────────────────────────────────────────────────────

class BatchHardTripletLoss(nn.Module):
    """
    Batch Hard Triplet Loss with L2 distance (Hermans et al. 2017).

    Steps per forward pass:
      1. L2-normalise all embeddings.
      2. Build (N, N) pairwise distance matrix.
      3. Build positive mask  (same label) and negative mask (different label).
      4. For each anchor:  find max distance among positives (hardest positive)
                                find min distance among negatives (hardest negative)
      5. Loss = mean over anchors of max(0, d_pos - d_neg + margin).

    Args:
        margin:       Triplet margin α (default 0.3).
        distance:     "euclidean" | "cosine"
        soft_margin:  If True, use softplus instead of hinge (smoother gradient).
    """

    def __init__(
        self,
        margin:      float = 0.3,
        distance:    str   = "euclidean",
        soft_margin: bool  = False,
    ):
        super().__init__()
        self.margin      = margin
        self.distance    = distance
        self.soft_margin = soft_margin

    def forward(
        self,
        embeddings: torch.Tensor,   # (N, D)
        labels:     torch.Tensor,   # (N,) integer identity labels
    ) -> Tuple[torch.Tensor, dict]:
        """
        Args:
            embeddings: Raw (un-normalised) embedding vectors from the model.
            labels:     Integer cow identity labels for each sample in the batch.

        Returns:
            (loss_scalar, info_dict) where info_dict contains diagnostic values.
        """
        N = embeddings.size(0)

        # L2 normalise so distances live in [0, 2]
        emb_norm = F.normalize(embeddings, p=2, dim=1)

        # Pairwise distance matrix
        if self.distance == "cosine":
            dist_mat = cosine_distance_matrix(emb_norm)
        else:
            dist_mat = euclidean_distance_matrix(emb_norm)

        # Boolean masks
        labels_row = labels.unsqueeze(1).expand(N, N)
        labels_col = labels.unsqueeze(0).expand(N, N)

        pos_mask = labels_row.eq(labels_col)              # same identity
        neg_mask = ~pos_mask                              # different identity
        # Exclude diagonal (anchor == anchor)
        eye = torch.eye(N, dtype=torch.bool, device=embeddings.device)
        pos_mask = pos_mask & ~eye
        neg_mask = neg_mask & ~eye

        # Hardest positive: max distance among positives
        # Fill non-positive entries with 0 before taking max
        pos_dist = dist_mat * pos_mask.float()
        hardest_pos, _ = pos_dist.max(dim=1)             # (N,)

        # Hardest negative: min distance among negatives
        # Fill non-negative entries with a large value before taking min
        neg_dist = dist_mat.clone()
        neg_dist[~neg_mask] = float("inf")
        hardest_neg, _ = neg_dist.min(dim=1)             # (N,)

        # Triplet loss
        if self.soft_margin:
            loss_per_anchor = F.softplus(hardest_pos - hardest_neg)
        else:
            loss_per_anchor = F.relu(hardest_pos - hardest_neg + self.margin)

        # Only count anchors that have at least one valid positive AND negative
        valid = pos_mask.any(dim=1) & neg_mask.any(dim=1)
        if valid.sum() == 0:
            loss = torch.tensor(0.0, device=embeddings.device, requires_grad=True)
        else:
            loss = loss_per_anchor[valid].mean()

        info = {
            "loss":        loss.item(),
            "mean_pos_d":  hardest_pos[valid].mean().item() if valid.any() else 0.0,
            "mean_neg_d":  hardest_neg[valid].mean().item() if valid.any() else 0.0,
            "frac_active": (loss_per_anchor[valid] > 0).float().mean().item()
                           if valid.any() else 0.0,
        }
        return loss, info


# ─────────────────────────────────────────────────────────────────────────────
# Quick self-test
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    torch.manual_seed(0)

    # Simulate a P=4, K=2 batch (8 samples, 128-dim embeddings)
    P, K, D = 4, 2, 128
    labels = torch.arange(P).repeat_interleave(K)      # [0,0,1,1,2,2,3,3]
    embs   = torch.randn(P * K, D)

    criterion = BatchHardTripletLoss(margin=0.3)
    loss, info = criterion(embs, labels)

    print(f"Loss:          {info['loss']:.4f}")
    print(f"Mean pos dist: {info['mean_pos_d']:.4f}")
    print(f"Mean neg dist: {info['mean_neg_d']:.4f}")
    print(f"Frac active:   {info['frac_active']:.2%}")
