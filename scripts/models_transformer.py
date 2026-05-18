"""
Video Transformer Models for Video-Based Cow Re-Identification
==============================================================

Implements two transformer-based video backbones:
  - Video Swin Transformer (Liu et al. 2021)
  - ViViT — Video Vision Transformer (Arnab et al. 2021)

Both are wrapped in the same VideoEmbeddingModel interface as the CNN models:
  backbone → EmbeddingHead (Linear → BN → ReLU → Linear) → L2-normalise

Educational Note — Transformers vs. CNNs for Video
---------------------------------------------------
CNNs use LOCAL receptive fields: each neuron sees a small T×H×W patch.
Transformers use GLOBAL self-attention: every patch can attend to every other.

Advantages of transformers for re-ID:
  - Long-range temporal context (e.g. gait cycle across many frames).
  - No inductive bias toward locality — the model learns which patches matter.

Disadvantages:
  - Quadratic attention cost grows fast with number of patches.
  - Require large pre-training datasets (ImageNet-21K / Kinetics-400).

Video Swin: introduces SHIFTED WINDOW attention to reduce cost to linear.
ViViT:      applies standard ViT to spatio-temporal tube tokens.

References:
  Video Swin: https://arxiv.org/abs/2106.13230
  ViViT:      https://arxiv.org/abs/2103.15691
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

from .models_cnn import EmbeddingHead          # reuse the same embedding head


# ─────────────────────────────────────────────────────────────────────────────
# Video Swin Transformer
# ─────────────────────────────────────────────────────────────────────────────

class VideoSwinEmbeddingModel(nn.Module):
    """
    Video Swin Transformer backbone with embedding head.

    Uses torchvision's swin3d_t / swin3d_s / swin3d_b implementation
    (available since torchvision 0.14).  Pre-trained weights are loaded
    from the Kinetics-400 model zoo (backbone initialised from ImageNet-22K).

    Input tensor shape: (N, C=3, T=16, H=224, W=224)

    Architecture:
      PatchEmbed (tubelet) → 4 Swin stages with window attention →
      Adaptive avg pool → EmbeddingHead → L2-normalise

    Args:
        model_name:      "swin3d_t" | "swin3d_s" | "swin3d_b"
        embedding_dim:   Output embedding size.
        pretrained:      Load Kinetics-400 pre-trained weights.
        freeze_backbone: Freeze all backbone parameters.
        dropout_rate:    Dropout in embedding head.
    """

    _OUT_DIMS = {
        "swin3d_t": 768,
        "swin3d_s": 768,
        "swin3d_b": 1024,
    }

    def __init__(
        self,
        model_name:      str   = "swin3d_t",
        embedding_dim:   int   = 512,
        pretrained:      bool  = True,
        freeze_backbone: bool  = False,
        dropout_rate:    float = 0.1,
    ):
        super().__init__()
        self.model_name = model_name

        # Load from torchvision (v0.14+)
        try:
            import torchvision.models.video as tv_video
            weights_map = {
                "swin3d_t": "Swin3D_T_Weights.KINETICS400_V1" if pretrained else None,
                "swin3d_s": "Swin3D_S_Weights.KINETICS400_V1" if pretrained else None,
                "swin3d_b": "Swin3D_B_Weights.KINETICS400_IMAGENET22K_V1"
                            if pretrained else None,
            }
            weights = getattr(
                __import__("torchvision.models.video", fromlist=[""]),
                weights_map[model_name].split(".")[0],
                None,
            )
            if pretrained and weights is not None:
                w_enum = getattr(weights, weights_map[model_name].split(".")[1])
                self.backbone = getattr(tv_video, model_name)(weights=w_enum)
                print(f"Video Swin ({model_name}): loaded Kinetics-400 weights.")
            else:
                self.backbone = getattr(tv_video, model_name)(weights=None)
                print(f"Video Swin ({model_name}): no pretrained weights.")
        except Exception as e:
            print(f"Video Swin load failed ({e}), falling back to no weights.")
            import torchvision.models.video as tv_video
            self.backbone = getattr(tv_video, model_name)(weights=None)

        # torchvision Swin3D ends with: AdaptiveAvgPool + head (Linear)
        # Replace the head with Identity; keep the pool
        backbone_out_dim = self._OUT_DIMS[model_name]
        try:
            self.backbone.head = nn.Identity()
        except AttributeError:
            pass

        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

        self.embed = EmbeddingHead(backbone_out_dim, embedding_dim, dropout_rate)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (N, 3, T, H, W) video clip tensor.

        Returns:
            (N, embedding_dim) L2-normalised embedding.
        """
        features = self.backbone(x)                    # (N, backbone_out_dim)
        if features.dim() > 2:
            features = features.flatten(1)
        return self.embed(features)


# ─────────────────────────────────────────────────────────────────────────────
# ViViT — Video Vision Transformer
# ─────────────────────────────────────────────────────────────────────────────

class TubeletEmbedding(nn.Module):
    """
    Spatio-temporal tube token embedding (ViViT Model 2).

    Instead of embedding each frame independently and then concatenating,
    tubelet embedding extracts 3D patches of size (t × h × w) from the
    raw video volume and linearly projects each patch into a token.

    This gives the model temporal context within each token, unlike the
    frame-by-frame approach used in simpler ViT adaptations.

    Args:
        img_size:     Spatial size of each frame (H = W).
        patch_size:   Spatial patch size (h = w).
        tubelet_size: Temporal tube depth (t).
        in_channels:  Video channels (3 for RGB).
        embed_dim:    Token embedding dimension.
    """

    def __init__(
        self,
        img_size:     int = 224,
        patch_size:   int = 16,
        tubelet_size: int = 2,
        in_channels:  int = 3,
        embed_dim:    int = 768,
    ):
        super().__init__()
        self.patch_size   = patch_size
        self.tubelet_size = tubelet_size

        num_patches_h = img_size // patch_size
        num_patches_w = img_size // patch_size
        self.num_spatial_patches = num_patches_h * num_patches_w

        # 3D convolution implements the tubelet projection efficiently
        self.proj = nn.Conv3d(
            in_channels  = in_channels,
            out_channels = embed_dim,
            kernel_size  = (tubelet_size, patch_size, patch_size),
            stride       = (tubelet_size, patch_size, patch_size),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (N, C, T, H, W)

        Returns:
            (N, num_tokens, embed_dim) — sequence of tube tokens.
        """
        # After Conv3d: (N, embed_dim, T//tubelet, H//patch, W//patch)
        x = self.proj(x)
        N, D, t, h, w = x.shape
        # Flatten spatial + temporal into sequence dimension
        x = x.flatten(2).transpose(1, 2)    # (N, t*h*w, D)
        return x


class ViViTEmbeddingModel(nn.Module):
    """
    ViViT (Video Vision Transformer) with tubelet embeddings.

    Strategy: initialise from a ViT-Base (ImageNet-21K) checkpoint using
    timm.  The tubelet Conv3d weights are inflated from the 2D patch
    projection weights (average over time dimension), following the
    inflation strategy from Arnab et al. 2021.

    Input tensor shape: (N, C=3, T=16, H=224, W=224)

    Architecture:
      TubeletEmbed → [CLS] + pos_embed → L Transformer layers →
      CLS token → EmbeddingHead → L2-normalise

    Args:
        img_size:        Spatial size (H = W).
        num_frames:      Number of frames T in each clip.
        tubelet_size:    Temporal patch depth.
        embedding_dim:   Output embedding dimension.
        pretrained:      Load ViT-Base ImageNet-21K weights (timm).
        freeze_backbone: Freeze all transformer weights.
        dropout_rate:    Dropout in embedding head.
    """

    def __init__(
        self,
        img_size:        int   = 224,
        num_frames:      int   = 16,
        tubelet_size:    int   = 2,
        embedding_dim:   int   = 512,
        pretrained:      bool  = True,
        freeze_backbone: bool  = False,
        dropout_rate:    float = 0.1,
    ):
        super().__init__()
        import timm

        # Load ViT-Base pre-trained on ImageNet-21K
        self.vit = timm.create_model(
            "vit_base_patch16_224",
            pretrained  = pretrained,
            num_classes = 0,             # remove classification head
        )
        vit_embed_dim = self.vit.embed_dim   # 768 for ViT-Base

        # Replace 2D patch embedding with 3D tubelet embedding
        self.tubelet_embed = TubeletEmbedding(
            img_size     = img_size,
            patch_size   = 16,
            tubelet_size = tubelet_size,
            in_channels  = 3,
            embed_dim    = vit_embed_dim,
        )

        # Inflate 2D patch projection weights into 3D tubelet weights
        if pretrained:
            self._inflate_patch_embed(tubelet_size)

        # CLS token and positional embeddings
        num_time_tokens   = num_frames // tubelet_size
        num_spatial_patch = (img_size // 16) ** 2
        num_tokens        = num_time_tokens * num_spatial_patch

        # Resize position embeddings from 196+1 (ViT-B/16) to num_tokens+1
        self._resize_pos_embed(num_tokens, vit_embed_dim)

        if freeze_backbone:
            for param in self.vit.parameters():
                param.requires_grad = False
            for param in self.tubelet_embed.parameters():
                param.requires_grad = False

        self.embed = EmbeddingHead(vit_embed_dim, embedding_dim, dropout_rate)

    def _inflate_patch_embed(self, tubelet_size: int):
        """
        Copy ViT patch projection weights into tubelet Conv3d.

        The 2D kernel (out, in, H, W) is inflated to 3D (out, in, T, H, W)
        by averaging across T (uniform temporal sampling initialisation).
        """
        with torch.no_grad():
            w2d = self.vit.patch_embed.proj.weight  # (D, 3, 16, 16)
            # Repeat along time axis and normalise
            w3d = w2d.unsqueeze(2).repeat(1, 1, tubelet_size, 1, 1) / tubelet_size
            self.tubelet_embed.proj.weight.copy_(w3d)
            if self.vit.patch_embed.proj.bias is not None:
                self.tubelet_embed.proj.bias.copy_(
                    self.vit.patch_embed.proj.bias
                )

    def _resize_pos_embed(self, num_tokens: int, embed_dim: int):
        """Interpolate ViT's positional embedding to match num_tokens."""
        with torch.no_grad():
            old_pos_emb = self.vit.pos_embed                    # (1, 197, D)
            cls_emb     = old_pos_emb[:, :1, :]                 # (1, 1, D)
            spatial_emb = old_pos_emb[:, 1:, :]                 # (1, 196, D)

            # Bicubic interpolation in 2D, then expand for new token count
            spatial_emb = spatial_emb.reshape(1, 14, 14, embed_dim).permute(0, 3, 1, 2)
            new_size = int(num_tokens ** 0.5)
            spatial_emb = F.interpolate(
                spatial_emb, size=(new_size, new_size),
                mode="bicubic", align_corners=False,
            )
            spatial_emb = spatial_emb.permute(0, 2, 3, 1).reshape(1, -1, embed_dim)
            new_pos_emb = torch.cat([cls_emb, spatial_emb], dim=1)
            self.vit.pos_embed = nn.Parameter(new_pos_emb)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (N, 3, T, H, W) video clip tensor.

        Returns:
            (N, embedding_dim) L2-normalised embedding.
        """
        N = x.shape[0]

        # Tubelet tokenisation → (N, num_tokens, D)
        tokens = self.tubelet_embed(x)

        # Prepend CLS token
        cls_tokens = self.vit.cls_token.expand(N, -1, -1)     # (N, 1, D)
        tokens     = torch.cat([cls_tokens, tokens], dim=1)   # (N, 1+L, D)

        # Add positional embeddings (interpolated to match token count)
        if tokens.shape[1] != self.vit.pos_embed.shape[1]:
            # Dynamic resize if clip length differs from expected
            tokens = tokens + self.vit.pos_embed[:, :tokens.shape[1], :]
        else:
            tokens = tokens + self.vit.pos_embed

        # Pass through transformer blocks
        tokens = self.vit.blocks(tokens)
        tokens = self.vit.norm(tokens)

        # Use CLS token as video representation
        cls_out = tokens[:, 0]                                  # (N, D)

        return self.embed(cls_out)


# ─────────────────────────────────────────────────────────────────────────────
# Factory function
# ─────────────────────────────────────────────────────────────────────────────

def create_transformer_model(
    model_name:      str,
    embedding_dim:   int   = 512,
    pretrained:      bool  = True,
    freeze_backbone: bool  = False,
    dropout_rate:    float = 0.1,
    device:          str   = "cuda",
    **kwargs,
) -> nn.Module:
    """
    Factory: return a Video Swin or ViViT embedding model.

    Args:
        model_name:    "swin" | "vivit"
        embedding_dim: Output embedding dimension.
        pretrained:    Load pre-trained weights.
        freeze_backbone: Freeze backbone.
        dropout_rate:  Embedding head dropout.
        device:        "cuda" or "cpu".
        **kwargs:      Extra model-specific kwargs
                       (e.g. model_name="swin3d_s" for Swin,
                             num_frames=16 for ViViT).

    Returns:
        Model on `device`.
    """
    name = model_name.lower()
    if name in ("swin", "video_swin"):
        swin_name = kwargs.get("swin_variant", "swin3d_t")
        model = VideoSwinEmbeddingModel(
            model_name      = swin_name,
            embedding_dim   = embedding_dim,
            pretrained      = pretrained,
            freeze_backbone = freeze_backbone,
            dropout_rate    = dropout_rate,
        )
    elif name == "vivit":
        model = ViViTEmbeddingModel(
            img_size        = kwargs.get("img_size", 224),
            num_frames      = kwargs.get("num_frames", 16),
            tubelet_size    = kwargs.get("tubelet_size", 2),
            embedding_dim   = embedding_dim,
            pretrained      = pretrained,
            freeze_backbone = freeze_backbone,
            dropout_rate    = dropout_rate,
        )
    else:
        raise ValueError(f"Unknown transformer model: {model_name}. "
                         "Choose 'swin' or 'vivit'.")

    model = model.to(device)

    total     = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"{model_name.upper()}: {total/1e6:.1f}M params ({trainable/1e6:.1f}M trainable)")

    return model


# ─────────────────────────────────────────────────────────────────────────────
# Quick self-test
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}\n")

    # Video Swin-T
    print("=== Video Swin-T ===")
    swin = create_transformer_model("swin", embedding_dim=512, pretrained=False,
                                    device=device)
    x    = torch.randn(2, 3, 16, 224, 224).to(device)
    emb  = swin(x)
    print(f"Input: {x.shape}  →  Embedding: {emb.shape}")
    print(f"L2 norms: {emb.norm(dim=1).tolist()}")

    # ViViT
    print("\n=== ViViT ===")
    vivit = create_transformer_model("vivit", embedding_dim=512, pretrained=False,
                                     num_frames=16, device=device)
    x     = torch.randn(2, 3, 16, 224, 224).to(device)
    emb   = vivit(x)
    print(f"Input: {x.shape}  →  Embedding: {emb.shape}")
    print(f"L2 norms: {emb.norm(dim=1).tolist()}")
