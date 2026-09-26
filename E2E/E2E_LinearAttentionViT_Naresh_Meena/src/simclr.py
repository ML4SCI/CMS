"""
src/simclr.py

SimCLR contrastive pretraining for XCiT and L2ViT backbones.

Usage:
    from src.simclr import run_simclr

    model = build_model('xcit_mae').to(device)
    run_simclr(model, pretrain_loader,
               f"{CFG['save_dir']}/xcit_simclr_backbone.pth", CFG, 'xcit')
"""

import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast, GradScaler

from src.augmentations import JetAugmentation

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class SimCLRProjector(nn.Module):
    # SimCLR v2 head — discarded after pretraining, only backbone is saved.
    # BatchNorm here prevents collapse without a stop-gradient.
    def __init__(self, embed_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embed_dim, 256),
            nn.BatchNorm1d(256),
            nn.GELU(),
            nn.Linear(256, 128),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.net(x), dim=-1)


def nt_xent_loss(z1: torch.Tensor, z2: torch.Tensor,
                 temperature: float = 0.07) -> torch.Tensor:
    """NT-Xent loss. z1[i] and z2[i] are positives, all other pairs negatives."""
    B   = z1.shape[0]
    z   = torch.cat([z1, z2], dim=0)
    sim = torch.mm(z, z.T) / temperature

    mask = torch.eye(2 * B, dtype=torch.bool, device=z.device)
    sim  = sim.masked_fill(mask, float('-inf'))

    pos_idx      = torch.zeros(2 * B, dtype=torch.long, device=z.device)
    pos_idx[:B]  = torch.arange(B, 2 * B, device=z.device)
    pos_idx[B:]  = torch.arange(0, B,     device=z.device)

    return F.cross_entropy(sim, pos_idx)


class SimCLRPretrainer(nn.Module):
    def __init__(self, backbone, embed_dim: int, temperature: float = 0.07):
        super().__init__()
        self.backbone    = backbone
        self.projector   = SimCLRProjector(embed_dim)
        self.temperature = temperature
        self.augment     = JetAugmentation()

    def forward(self, imgs: torch.Tensor) -> torch.Tensor:
        view1, view2 = self.augment.simclr_views(imgs)
        z1 = self.projector(self.backbone.forward_features(view1))
        z2 = self.projector(self.backbone.forward_features(view2))
        return nt_xent_loss(z1, z2, self.temperature)


def run_simclr(model, loader, save_path: str, cfg: dict,
               model_name: str = 'model'):
    """
    SimCLR pretraining loop. Same interface as run_mae() in train.py.

    Saves backbone state dict at best loss → save_path
    Loss history → cfg['history_dir']/pretrain_simclr_{model_name}.npy

    Key differences from MAE:
        lr=1e-3, weight_decay=1e-4 (lower wd — contrastive is sensitive)
        10-epoch linear warmup (longer — SimCLR collapses without it)
    """
    embed_dim = model.backbone.num_features if hasattr(model, 'backbone') \
                else model.num_features
    backbone  = model.backbone if hasattr(model, 'backbone') else model

    simclr    = SimCLRPretrainer(model, embed_dim).to(device)
    optimizer = torch.optim.AdamW(simclr.parameters(), lr=1e-3, weight_decay=1e-4)

    n_epochs  = cfg['pretrain_epochs']
    warmup_ep = 10

    def lr_lambda(ep):
        if ep < warmup_ep:
            return (ep + 1) / warmup_ep
        progress = (ep - warmup_ep) / max(1, n_epochs - warmup_ep)
        return 0.5 * (1 + np.cos(np.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    scaler    = GradScaler(enabled=cfg['use_amp'])
    best_loss = float('inf')
    history   = []

    print(f"\nSimCLR Pretraining — {model_name}")
    print("=" * 50)

    for ep in range(n_epochs):
        simclr.train()
        ep_loss, n_batches = 0.0, 0
        t0 = time.time()

        for imgs in loader:
            imgs = imgs.to(device, non_blocking=True)
            optimizer.zero_grad()

            with autocast(enabled=cfg['use_amp']):
                loss = simclr(imgs)

            if torch.isnan(loss):
                continue

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(simclr.parameters(), cfg['grad_clip'])
            scaler.step(optimizer)
            scaler.update()

            ep_loss   += loss.item()
            n_batches += 1

        scheduler.step()
        avg = ep_loss / max(n_batches, 1)
        history.append(avg)

        if avg < best_loss:
            best_loss = avg
            torch.save(backbone.state_dict(), save_path)

        if (ep + 1) % 5 == 0 or ep == 0:
            print(f"  ep {ep+1:3d}/{n_epochs}  loss={avg:.5f}  "
                  f"best={best_loss:.5f}  lr={scheduler.get_last_lr()[0]:.2e}  "
                  f"t={time.time()-t0:.0f}s")

    np.save(
        os.path.join(cfg['history_dir'], f'pretrain_simclr_{model_name}.npy'),
        np.array(history)
    )

    backbone.load_state_dict(torch.load(save_path, map_location=device))
    print(f"\n  Best loss : {best_loss:.5f}\n  Saved     : {save_path}")
    return history