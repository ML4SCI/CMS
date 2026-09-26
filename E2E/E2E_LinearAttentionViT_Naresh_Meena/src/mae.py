import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.utils import CFG


class MAEPretrainer(nn.Module):
    """
    MAE pretrainer for XCiT.

    Masks 75% of patch tokens, replaces them with a learnable token,
    encodes the full sequence, and reconstructs masked patch pixels.
    We keep all tokens during encoding because XCiT's cross-covariance
    attention needs the full spatial structure.
    """
    def __init__(self, backbone, patch_size=16,
                 mask_ratio=0.75, img_size=128, in_chans=8):
        super().__init__()
        self.backbone   = backbone
        self.ps         = patch_size
        self.mr         = mask_ratio
        self.n_patches  = (img_size // patch_size) ** 2
        self.H = self.W = img_size // patch_size
        self.patch_dim  = in_chans * patch_size * patch_size
        feat_dim        = backbone.num_features

        self.mask_tok = nn.Parameter(torch.zeros(1, 1, feat_dim))
        nn.init.normal_(self.mask_tok, std=0.02)

        self.decoder = nn.Sequential(
            nn.LayerNorm(feat_dim),
            nn.Linear(feat_dim, 512),
            nn.GELU(),
            nn.Linear(512, self.patch_dim)
        )

        # probe patch_embed return type — varies across timm versions
        with torch.no_grad():
            dummy = torch.zeros(
                1, in_chans, img_size, img_size,
                device=next(backbone.parameters()).device
            )
            out = backbone.patch_embed(dummy)
            self._pe_tuple = isinstance(out, tuple)

    def _get_tokens(self, x):
        out = self.backbone.patch_embed(x)
        return out[0] if self._pe_tuple else out

    def _to_patches(self, imgs):
        B, C, H, W = imgs.shape
        p = self.ps
        x = imgs.reshape(B, C, H//p, p, W//p, p)
        x = x.permute(0, 2, 4, 3, 5, 1).reshape(B, -1, p*p*C)
        return x

    def forward(self, imgs):
        B      = imgs.shape[0]
        target = self._to_patches(imgs)

        tokens = self._get_tokens(imgs)
        N, D   = tokens.shape[1], tokens.shape[2]
        n_mask = int(N * self.mr)

        noise    = torch.rand(B, N, device=imgs.device)
        shuffled = torch.argsort(noise, dim=1)
        mask     = torch.zeros(B, N, dtype=torch.bool, device=imgs.device)
        mask.scatter_(1, shuffled[:, :n_mask], True)

        fill   = self.mask_tok.expand(B, N, D)
        tokens = torch.where(mask.unsqueeze(-1), fill, tokens)

        for blk in self.backbone.blocks:
            try:
                tokens = blk(tokens, self.H, self.W)
            except TypeError:
                tokens = blk(tokens)

        pred = self.decoder(tokens)
        return F.mse_loss(pred[mask], target[mask])


class L2ViTMAE(nn.Module):
    """
    MAE pretrainer for L2ViT.

    L2ViT's hierarchical stem makes token-level masking tricky so we
    work in pixel space instead — build a patch mask, expand it to image
    resolution, zero out those regions, fill with a learnable spatial
    token, then run the full encoder. Loss is MSE on masked patches only.
    """
    def __init__(self, model, patch_size=16, mask_ratio=0.75,
                 img_size=128, in_chans=8):
        super().__init__()
        self.model     = model
        self.ps        = patch_size
        self.mr        = mask_ratio
        self.patch_dim = in_chans * patch_size * patch_size
        feat_dim       = model.num_features

        self.mask_tok = nn.Parameter(torch.zeros(1, in_chans, 1, 1))
        nn.init.normal_(self.mask_tok, std=0.02)

        self.decoder = nn.Sequential(
            nn.LayerNorm(feat_dim),
            nn.Linear(feat_dim, 512),
            nn.GELU(),
            nn.Linear(512, self.patch_dim)
        )

    def _patchify(self, imgs):
        B, C, H, W = imgs.shape
        p      = self.ps
        nh, nw = H // p, W // p
        x = imgs.reshape(B, C, nh, p, nw, p)
        x = x.permute(0, 2, 4, 3, 5, 1).reshape(B, nh*nw, p*p*C)
        return x

    def forward(self, imgs):
        B, C, H, W = imgs.shape
        p          = self.ps
        nh, nw     = H // p, W // p
        N          = nh * nw
        n_mask     = int(N * self.mr)

        noise      = torch.rand(B, N, device=imgs.device)
        ids        = torch.argsort(noise, dim=1)
        patch_mask = torch.zeros(B, N, device=imgs.device)
        patch_mask.scatter_(1, ids[:, :n_mask], 1.0)

        px_mask = patch_mask.reshape(B, 1, nh, nw)
        px_mask = px_mask.repeat_interleave(p, dim=2) \
                         .repeat_interleave(p, dim=3)

        masked_imgs = imgs * (1 - px_mask) + self.mask_tok * px_mask

        feat    = self.model.forward_features(masked_imgs)
        feat_ex = feat.unsqueeze(1).expand(-1, N, -1)
        pred    = self.decoder(feat_ex)

        target    = self._patchify(imgs)
        bool_mask = patch_mask.bool()
        return F.mse_loss(pred[bool_mask], target[bool_mask])