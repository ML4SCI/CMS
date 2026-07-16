"""
ViT-Small implementation for HLS4ML LHC jets baseline (tier-4, 22M scale).

Standard ViT-Small: 12 layers, 384 hidden, 6 heads, MLP ratio 4.
Adapted for 2-channel (ECAL+HCAL) 100x100 jet images.
Patch size 10 matches DepthViT's 10x10 token grid for fair comparison
(same tokenization, different attention mechanism) — identical to the
ViT-Tiny baseline, just scaled to the Small configuration.

Cold start: positional embeddings and patch projection initialized
from scratch.

With (img=100, patch=10, in_channels=2, num_classes=5):
    standard 384/12/6 -> 21,412,613 params (verified by building the module)

Implemented from scratch with only torch + torch.nn to avoid pulling
in `transformers` or `timm` on Perlmutter. Structurally identical to
vit_tiny.py so the two baselines differ only in width/heads.
"""

import math
from typing import Optional

import torch
import torch.nn as nn


# --------------------------------------------------------------------- #
#  Building blocks                                                       #
# --------------------------------------------------------------------- #

class PatchEmbed(nn.Module):
    """Conv2d-based patch embedding. Input (B, C, H, W) -> (B, N, D)."""

    def __init__(self, img_size: int = 100, patch_size: int = 10,
                 in_channels: int = 2, embed_dim: int = 384):
        super().__init__()
        if img_size % patch_size != 0:
            raise ValueError(
                f"img_size ({img_size}) must be divisible by patch_size ({patch_size})"
            )
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid = img_size // patch_size
        self.n_patches = self.grid * self.grid
        self.proj = nn.Conv2d(
            in_channels, embed_dim,
            kernel_size=patch_size, stride=patch_size,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj(x)                  # (B, D, grid, grid)
        x = x.flatten(2).transpose(1, 2)  # (B, N, D)
        return x


class Attention(nn.Module):
    """Standard multi-head self-attention."""

    def __init__(self, dim: int, num_heads: int = 6,
                 qkv_bias: bool = True,
                 attn_drop: float = 0.0, proj_drop: float = 0.0):
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError(
                f"embed_dim ({dim}) must be divisible by num_heads ({num_heads})"
            )
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, D = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)        # (3, B, H, N, d)
        q, k, v = qkv.unbind(0)
        # Prefer fused SDPA when available (PyTorch >= 2.0)
        if hasattr(torch.nn.functional, "scaled_dot_product_attention"):
            x = torch.nn.functional.scaled_dot_product_attention(
                q, k, v,
                dropout_p=self.attn_drop.p if self.training else 0.0,
            )
        else:
            attn = (q @ k.transpose(-2, -1)) * self.scale
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)
            x = attn @ v
        x = x.transpose(1, 2).reshape(B, N, D)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class MLP(nn.Module):
    def __init__(self, dim: int, hidden_dim: int, drop: float = 0.0):
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_dim, dim)
        self.drop = nn.Dropout(drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Block(nn.Module):
    """Pre-norm transformer block."""

    def __init__(self, dim: int, num_heads: int, mlp_ratio: float = 4.0,
                 dropout: float = 0.0, attn_dropout: float = 0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = Attention(dim, num_heads,
                              attn_drop=attn_dropout, proj_drop=dropout)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = MLP(dim, int(dim * mlp_ratio), drop=dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


# --------------------------------------------------------------------- #
#  Model                                                                 #
# --------------------------------------------------------------------- #

class ViTSmall(nn.Module):
    """Vanilla ViT-Small classifier.

    Defaults reproduce the standard ViT-Small configuration
    (12 layers, 384 hidden, 6 heads, MLP ratio 4) but with patch
    size 10 to align tokenization with DepthViT on 100x100 jets.

    With (img=100, patch=10, in_channels=2, num_classes=5):
        param count = 21,412,613  (verified)
    """

    def __init__(
        self,
        img_size: int = 100,
        patch_size: int = 10,
        in_channels: int = 2,
        num_classes: int = 5,
        embed_dim: int = 384,
        num_layers: int = 12,
        num_heads: int = 6,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        attn_dropout: float = 0.0,
    ):
        super().__init__()
        self.patch_embed = PatchEmbed(img_size, patch_size, in_channels, embed_dim)
        n_patches = self.patch_embed.n_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, n_patches + 1, embed_dim))
        self.pos_drop = nn.Dropout(dropout)

        self.blocks = nn.ModuleList([
            Block(embed_dim, num_heads, mlp_ratio,
                  dropout=dropout, attn_dropout=attn_dropout)
            for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)

        self._init_weights()

    def _init_weights(self):
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        self.apply(self._init_module)

    @staticmethod
    def _init_module(m: nn.Module) -> None:
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.LayerNorm):
            nn.init.zeros_(m.bias)
            nn.init.ones_(m.weight)
        elif isinstance(m, nn.Conv2d):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        B = x.shape[0]
        x = self.patch_embed(x)                      # (B, N, D)
        cls = self.cls_token.expand(B, -1, -1)       # (B, 1, D)
        x = torch.cat([cls, x], dim=1)               # (B, N+1, D)
        x = x + self.pos_embed
        x = self.pos_drop(x)
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.forward_features(x)
        return self.head(x[:, 0])                    # CLS token

    @torch.no_grad()
    def param_count(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# --------------------------------------------------------------------- #
#  Quick self-test                                                       #
# --------------------------------------------------------------------- #

if __name__ == "__main__":
    m = ViTSmall()
    n_p = m.param_count()
    print(f"ViT-Small params: {n_p:,}  ({n_p/1e6:.2f}M)")
    x = torch.randn(2, 2, 100, 100)
    out = m(x)
    print(f"input  shape: {tuple(x.shape)}")
    print(f"output shape: {tuple(out.shape)}")
    assert out.shape == (2, 5), f"unexpected output shape: {out.shape}"
    print("OK")
