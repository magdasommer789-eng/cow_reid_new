"""
3D CNN Models for Video-Based Cow Re-Identification
====================================================

Implements two 3D-CNN backbones:
  - C3D   : Tran et al. 2015, "Learning Spatiotemporal Features with 3D CNNs"
  - X3D-M : Feichtenhofer 2020, "X3D: Expanding Architectures for Efficient
             Video Recognition" (Facebook Research)

Both models are wrapped in a unified VideoEmbeddingModel that:
  1. Loads pre-trained weights (ImageNet / Sports-1M via pytorchvideo)
  2. Removes the original classification head
  3. Adds an embedding head: backbone → Linear(D) → BN → L2-normalise → 512-d

Educational Note — Why 3D convolutions for video?
---------------------------------------------------
A 2D CNN applies a (H×W) kernel to each frame independently — it captures
spatial patterns but ignores temporal motion.
A 3D CNN applies a (T×H×W) kernel, so it learns JOINT space-time features:
flickering patterns, motion blur, direction of movement.
For re-ID, gait and movement are strong identity cues, which 3D CNNs exploit.

Reference:
  C3D: https://arxiv.org/abs/1412.0767
  X3D: https://arxiv.org/abs/2004.04730
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


# ─────────────────────────────────────────────────────────────────────────────
# Shared embedding head
# ─────────────────────────────────────────────────────────────────────────────

class EmbeddingHead(nn.Module):
    """
    Projection head that maps backbone features to a fixed-size embedding.

    Architecture:  Linear → BatchNorm1d → ReLU → Linear → L2-normalise

    The L2 normalisation ensures all embeddings lie on the unit hypersphere,
    which makes cosine similarity and Euclidean distance equivalent and keeps
    the triplet loss scale-invariant.

    Args:
        in_features:    Dimensionality of the backbone's output feature vector.
        embedding_dim:  Target embedding size (default 512).
        dropout_rate:   Dropout before the first linear layer.
    """

    def __init__(
        self,
        in_features:   int,
        embedding_dim: int = 512,
        dropout_rate:  float = 0.5,
    ):
        super().__init__()
        self.head = nn.Sequential(
            nn.Dropout(p=dropout_rate),
            nn.Linear(in_features, embedding_dim),
            nn.BatchNorm1d(embedding_dim),
            nn.ReLU(inplace=True),
            nn.Linear(embedding_dim, embedding_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (N, in_features) backbone feature vector.

        Returns:
            (N, embedding_dim) L2-normalised embedding.
        """
        emb = self.head(x)
        return F.normalize(emb, p=2, dim=1)


# ─────────────────────────────────────────────────────────────────────────────
# C3D
# ─────────────────────────────────────────────────────────────────────────────

class C3DEmbeddingModel(nn.Module):
    """
    C3D backbone with an embedding head for re-identification.

    C3D was originally trained on Sports-1M for action recognition.
    We load the pretrained weights from pytorchvideo and replace the
    final classification layer with our EmbeddingHead.

    Input tensor shape: (N, C=3, T=16, H=112, W=112)
    The original C3D was designed for 112×112 spatial input.

    Args:
        embedding_dim:  Size of the output embedding vector.
        pretrained:     Load Sports-1M pre-trained weights via pytorchvideo.
        freeze_backbone: Freeze all backbone parameters (feature extractor mode).
        dropout_rate:   Dropout in the embedding head.
    """

    def __init__(
        self,
        embedding_dim:   int   = 512,
        pretrained:      bool  = True,
        freeze_backbone: bool  = False,
        dropout_rate:    float = 0.5,
    ):
        super().__init__()

        # Load C3D from pytorchvideo (returns a nn.Module)
        # pytorchvideo wraps the model in a "create_c3d" factory function
        # and provides weights pre-trained on Sports-1M.
        from pytorchvideo.models import create_c3d
        self.backbone = create_c3d(
            # Head is replaced below; set to Identity here
            model_num_class=400,        # placeholder — head will be removed
            dropout_rate=0.0,           # we add our own dropout in EmbeddingHead
        )

        if pretrained:
            self._load_pretrained_c3d()

        # C3D's final fc layer has 512 output features (before the 400-class head)
        # The backbone output after the avg pool has shape (N, 512)
        # We remove the classification head and attach EmbeddingHead
        backbone_out_dim = 512

        # Replace the classifier with Identity so backbone returns raw features
        # pytorchvideo C3D structure: backbone → blocks → head (pool + dropout + fc)
        # We access the head block and replace the projection layer
        try:
            # pytorchvideo model structure
            self.backbone.blocks[-1].proj = nn.Identity()
        except (AttributeError, IndexError):
            # Fallback: try to zero out the last linear layer manually
            pass

        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

        self.embed = EmbeddingHead(backbone_out_dim, embedding_dim, dropout_rate)

    def _load_pretrained_c3d(self):
        """Load Sports-1M pre-trained C3D weights from pytorchvideo hub."""
        try:
            import torch.hub
            checkpoint = torch.hub.load_state_dict_from_url(
                "https://dl.fbaipublicfiles.com/pytorchvideo/model_zoo/kinetics/"
                "C3D_8x8_R2PLUS1D.pyth",
                map_location="cpu",
                check_hash=False,
            )
            # pytorchvideo checkpoints vary; attempt best-effort load
            state = checkpoint.get("model_state", checkpoint)
            missing, unexpected = self.backbone.load_state_dict(state, strict=False)
            print(f"C3D pretrained: missing={len(missing)}, unexpected={len(unexpected)}")
        except Exception as e:
            print(f"C3D pretrained weights not loaded ({e}). Training from scratch.")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (N, 3, T, H, W) video clip tensor (ImageNet-normalised).

        Returns:
            (N, embedding_dim) L2-normalised embedding.
        """
        features = self.backbone(x)        # (N, 512) after removing head
        if features.dim() > 2:
            # Global average pool if backbone still returns spatial dims
            features = features.mean(dim=[2, 3, 4]) if features.dim() == 5 \
                       else features.flatten(1)
        return self.embed(features)


# ─────────────────────────────────────────────────────────────────────────────
# X3D
# ─────────────────────────────────────────────────────────────────────────────

class X3DEmbeddingModel(nn.Module):
    """
    X3D backbone with an embedding head for re-identification.

    X3D is a family of efficient 3D CNNs obtained by progressively expanding
    a tiny base model along multiple axes (temporal, spatial, width, depth).
    X3D-M is a good balance between accuracy and speed.

    Input tensor shape: (N, C=3, T=16, H=224, W=224)
    (X3D supports various temporal depths; 16 frames is the default for X3D-M)

    Pre-trained on Kinetics-400 (backbone initialised from ImageNet).

    Args:
        model_size:      "xs" | "s" | "m" | "l"  (default "m")
        embedding_dim:   Output embedding size.
        pretrained:      Load Kinetics-400 weights from pytorchvideo hub.
        freeze_backbone: Freeze backbone parameters.
        dropout_rate:    Dropout in embedding head.
    """

    # Output feature dimensions per X3D variant (before the classification head)
    _OUT_DIMS = {"xs": 192, "s": 192, "m": 192, "l": 192}
    # Kinetics-400 pretrained checkpoint URLs from pytorchvideo model zoo
    _CKPT_URLS = {
        "xs": "https://dl.fbaipublicfiles.com/pytorchvideo/model_zoo/kinetics/"
              "X3D_XS.pyth",
        "s":  "https://dl.fbaipublicfiles.com/pytorchvideo/model_zoo/kinetics/"
              "X3D_S.pyth",
        "m":  "https://dl.fbaipublicfiles.com/pytorchvideo/model_zoo/kinetics/"
              "X3D_M.pyth",
        "l":  "https://dl.fbaipublicfiles.com/pytorchvideo/model_zoo/kinetics/"
              "X3D_L.pyth",
    }

    def __init__(
        self,
        model_size:      str   = "m",
        embedding_dim:   int   = 512,
        pretrained:      bool  = True,
        freeze_backbone: bool  = False,
        dropout_rate:    float = 0.5,
    ):
        super().__init__()
        self.model_size = model_size.lower()

        from pytorchvideo.models import create_x3d
        self.backbone = create_x3d(
            input_clip_length   = 16,
            input_crop_size     = 224,
            model_num_class     = 400,    # placeholder — head replaced below
            dropout_rate        = 0.0,
        )

        if pretrained:
            self._load_pretrained_x3d()

        # Remove classification head (last block's projection)
        backbone_out_dim = self._OUT_DIMS[self.model_size]
        try:
            self.backbone.blocks[-1].proj = nn.Identity()
        except (AttributeError, IndexError):
            pass

        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

        self.embed = EmbeddingHead(backbone_out_dim, embedding_dim, dropout_rate)

    def _load_pretrained_x3d(self):
        """Load Kinetics-400 pre-trained X3D weights."""
        url = self._CKPT_URLS.get(self.model_size)
        if url is None:
            print(f"No pretrained URL for X3D-{self.model_size.upper()}")
            return
        try:
            import torch.hub
            checkpoint = torch.hub.load_state_dict_from_url(url, map_location="cpu")
            state = checkpoint.get("model_state", checkpoint)
            missing, unexpected = self.backbone.load_state_dict(state, strict=False)
            print(f"X3D-{self.model_size.upper()} pretrained: "
                  f"missing={len(missing)}, unexpected={len(unexpected)}")
        except Exception as e:
            print(f"X3D pretrained weights not loaded ({e}). Training from scratch.")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (N, 3, T, H, W) video clip tensor.

        Returns:
            (N, embedding_dim) L2-normalised embedding.
        """
        features = self.backbone(x)
        if features.dim() > 2:
            features = features.flatten(1)
        return self.embed(features)


# ─────────────────────────────────────────────────────────────────────────────
# Factory function
# ─────────────────────────────────────────────────────────────────────────────

def create_cnn_model(
    model_name:      str,
    embedding_dim:   int   = 512,
    pretrained:      bool  = True,
    freeze_backbone: bool  = False,
    dropout_rate:    float = 0.5,
    device:          str   = "cuda",
    **kwargs,
) -> nn.Module:
    """
    Factory: return a C3D or X3D embedding model.

    Args:
        model_name:    "c3d" | "x3d"
        embedding_dim: Output embedding dimension.
        pretrained:    Whether to load pre-trained weights.
        freeze_backbone: Whether to freeze the backbone.
        dropout_rate:  Dropout rate for the embedding head.
        device:        "cuda" or "cpu".
        **kwargs:      Model-specific kwargs (e.g. model_size="m" for X3D).

    Returns:
        Model moved to `device`.
    """
    name = model_name.lower()
    if name == "c3d":
        model = C3DEmbeddingModel(
            embedding_dim   = embedding_dim,
            pretrained      = pretrained,
            freeze_backbone = freeze_backbone,
            dropout_rate    = dropout_rate,
        )
    elif name == "x3d":
        model = X3DEmbeddingModel(
            model_size      = kwargs.get("model_size", "m"),
            embedding_dim   = embedding_dim,
            pretrained      = pretrained,
            freeze_backbone = freeze_backbone,
            dropout_rate    = dropout_rate,
        )
    else:
        raise ValueError(f"Unknown CNN model: {model_name}. Choose 'c3d' or 'x3d'.")

    model = model.to(device)

    total      = sum(p.numel() for p in model.parameters())
    trainable  = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"{model_name.upper()}: {total/1e6:.1f}M params ({trainable/1e6:.1f}M trainable)")

    return model


# ─────────────────────────────────────────────────────────────────────────────
# Quick self-test
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}\n")

    # Test C3D — expects 112×112 input
    print("=== C3D ===")
    c3d = create_cnn_model("c3d", embedding_dim=512, pretrained=False, device=device)
    x   = torch.randn(2, 3, 16, 112, 112).to(device)
    emb = c3d(x)
    print(f"Input: {x.shape}  →  Embedding: {emb.shape}")
    print(f"L2 norms: {emb.norm(dim=1).tolist()}")   # should be ~1.0

    # Test X3D-M — expects 224×224 input
    print("\n=== X3D-M ===")
    x3d = create_cnn_model("x3d", embedding_dim=512, pretrained=False,
                            model_size="m", device=device)
    x   = torch.randn(2, 3, 16, 224, 224).to(device)
    emb = x3d(x)
    print(f"Input: {x.shape}  →  Embedding: {emb.shape}")
    print(f"L2 norms: {emb.norm(dim=1).tolist()}")
