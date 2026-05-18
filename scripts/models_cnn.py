"""
3D CNN Models for Video-Based Cow Re-Identification
====================================================

Implements two 3D-CNN backbones:
  - C3D   : Tran et al. 2015, "Learning Spatiotemporal Features with 3D CNNs"
  - X3D-M : Feichtenhofer 2020, "X3D: Expanding Architectures for Efficient
             Video Recognition" (Facebook Research)

Both models are wrapped in a unified VideoEmbeddingModel that:
  1. (Optional) Loads pre-trained weights (Kinetics-400 for X3D; C3D uses
     inflated-2D or random init — no public ImageNet C3D weights exist)
  2. Removes the original classification head
  3. Adds an embedding head: backbone → Linear(D) → BN → ReLU → Linear → L2-normalise

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
from typing import Optional


# ─────────────────────────────────────────────────────────────────────────────
# Shared embedding head
# ─────────────────────────────────────────────────────────────────────────────

class EmbeddingHead(nn.Module):
    """
    Projection head: backbone features → fixed-size L2-normalised embedding.

    Architecture:  Dropout → Linear → BatchNorm1d → ReLU → Linear → L2-norm

    L2 normalisation maps all embeddings onto the unit hypersphere, making
    cosine distance and Euclidean distance equivalent and keeping the
    triplet loss margin scale-invariant.
    """

    def __init__(
        self,
        in_features:   int,
        embedding_dim: int   = 512,
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
        return F.normalize(self.head(x), p=2, dim=1)


# ─────────────────────────────────────────────────────────────────────────────
# C3D backbone (implemented from scratch — Tran et al. 2015)
# ─────────────────────────────────────────────────────────────────────────────

class _C3DBackbone(nn.Module):
    """
    Faithful C3D implementation from Tran et al. (2015).

    Architecture:
      conv1(64) → pool1(1×2×2)
      conv2(128) → pool2(2×2×2)
      conv3a(256) → conv3b(256) → pool3(2×2×2)
      conv4a(512) → conv4b(512) → pool4(2×2×2)
      conv5a(512) → conv5b(512) → AdaptiveAvgPool3d(1,1,1)

    Output: (N, 512) feature vector.
    All kernels are 3×3×3 with padding=1 to preserve spatial size.

    NOTE: C3D was originally trained on Sports-1M, NOT ImageNet.
    There are no publicly available ImageNet-pretrained C3D weights.
    This model is trained from random initialisation, which works
    for small datasets with sufficient augmentation.
    """

    def __init__(self):
        super().__init__()

        def conv_bn_relu(in_c, out_c):
            return nn.Sequential(
                nn.Conv3d(in_c, out_c, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm3d(out_c),
                nn.ReLU(inplace=True),
            )

        self.layer1  = conv_bn_relu(3,   64)
        self.pool1   = nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2))

        self.layer2  = conv_bn_relu(64,  128)
        self.pool2   = nn.MaxPool3d(kernel_size=2, stride=2)

        self.layer3a = conv_bn_relu(128, 256)
        self.layer3b = conv_bn_relu(256, 256)
        self.pool3   = nn.MaxPool3d(kernel_size=2, stride=2)

        self.layer4a = conv_bn_relu(256, 512)
        self.layer4b = conv_bn_relu(512, 512)
        self.pool4   = nn.MaxPool3d(kernel_size=2, stride=2)

        self.layer5a = conv_bn_relu(512, 512)
        self.layer5b = conv_bn_relu(512, 512)
        self.pool5   = nn.AdaptiveAvgPool3d((1, 1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool1(self.layer1(x))
        x = self.pool2(self.layer2(x))
        x = self.pool3(self.layer3b(self.layer3a(x)))
        x = self.pool4(self.layer4b(self.layer4a(x)))
        x = self.pool5(self.layer5b(self.layer5a(x)))
        return x.flatten(1)                       # (N, 512)


class C3DEmbeddingModel(nn.Module):
    """
    C3D backbone + EmbeddingHead for re-identification.

    Input tensor shape: (N, 3, T, H, W)
    The AdaptiveAvgPool3d in the backbone handles arbitrary T, H, W.
    """

    def __init__(
        self,
        embedding_dim:   int   = 512,
        pretrained:      bool  = True,    # kept for API compat; no public weights
        freeze_backbone: bool  = False,
        dropout_rate:    float = 0.5,
    ):
        super().__init__()
        self.backbone = _C3DBackbone()

        if pretrained:
            # Try to load Sports-1M weights (may fail silently)
            self._try_load_sports1m()
        else:
            print("C3D: random initialisation (no ImageNet weights for C3D).")

        if freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False

        self.embed = EmbeddingHead(512, embedding_dim, dropout_rate)

    def _try_load_sports1m(self):
        """Attempt to load a public Sports-1M C3D checkpoint."""
        try:
            url = ("https://github.com/DavideA/c3d-pytorch/raw/master/"
                   "C3D_sports1M_pytorch.pkl")
            import urllib.request, pickle, io
            with urllib.request.urlopen(url, timeout=15) as resp:
                state = pickle.load(io.BytesIO(resp.read()), encoding="latin1")
            # best-effort load (key names may differ)
            missing, _ = self.backbone.load_state_dict(state, strict=False)
            print(f"C3D Sports-1M weights loaded (missing layers: {len(missing)}).")
        except Exception as e:
            print(f"C3D: could not load Sports-1M weights ({e}). "
                  "Training from random init.")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.embed(self.backbone(x))


# ─────────────────────────────────────────────────────────────────────────────
# X3D (Feichtenhofer 2020) — via pytorchvideo.models.x3d
# ─────────────────────────────────────────────────────────────────────────────

class X3DEmbeddingModel(nn.Module):
    """
    X3D backbone + EmbeddingHead for re-identification.

    X3D is a family of efficient 3D CNNs (xs / s / m / l).
    X3D-M balances accuracy and speed.  Pre-trained on Kinetics-400,
    which was itself initialised from ImageNet features.

    Input tensor shape: (N, 3, T=16, H=224, W=224)
    """

    _OUT_DIMS = {"xs": 2048, "s": 2048, "m": 2048, "l": 2048}  # after head expand conv
    _CKPT_URLS = {
        "xs": "https://dl.fbaipublicfiles.com/pytorchvideo/model_zoo/kinetics/X3D_XS.pyth",
        "s":  "https://dl.fbaipublicfiles.com/pytorchvideo/model_zoo/kinetics/X3D_S.pyth",
        "m":  "https://dl.fbaipublicfiles.com/pytorchvideo/model_zoo/kinetics/X3D_M.pyth",
        "l":  "https://dl.fbaipublicfiles.com/pytorchvideo/model_zoo/kinetics/X3D_L.pyth",
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

        # Correct import path for pytorchvideo 0.1.5
        from pytorchvideo.models.x3d import create_x3d
        self.backbone = create_x3d(
            input_clip_length = 16,
            input_crop_size   = 224,
            model_num_class   = 400,   # placeholder; head is replaced below
            dropout_rate      = 0.0,
        )

        if pretrained:
            self._load_pretrained()

        # Remove classification head (final proj layer)
        try:
            self.backbone.blocks[-1].proj = nn.Identity()
        except (AttributeError, IndexError):
            pass

        if freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False

        backbone_out_dim = self._OUT_DIMS[self.model_size]
        self.embed = EmbeddingHead(backbone_out_dim, embedding_dim, dropout_rate)

    def _load_pretrained(self):
        url = self._CKPT_URLS.get(self.model_size)
        if not url:
            return
        try:
            ckpt  = torch.hub.load_state_dict_from_url(url, map_location="cpu")
            state = ckpt.get("model_state", ckpt)
            miss, unex = self.backbone.load_state_dict(state, strict=False)
            print(f"X3D-{self.model_size.upper()} Kinetics-400 weights loaded "
                  f"(missing={len(miss)}, unexpected={len(unex)}).")
        except Exception as e:
            print(f"X3D pretrained weights not loaded ({e}). Random init.")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.backbone(x)
        if features.dim() > 2:
            features = features.flatten(1)
        return self.embed(features)


# ─────────────────────────────────────────────────────────────────────────────
# Factory
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
    """Return a C3D or X3D embedding model moved to device."""
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
        raise ValueError(f"Unknown CNN model: {model_name!r}. Choose c3d or x3d.")

    model = model.to(device)
    total     = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"{model_name.upper()}: {total/1e6:.1f}M params ({trainable/1e6:.1f}M trainable)")
    return model
