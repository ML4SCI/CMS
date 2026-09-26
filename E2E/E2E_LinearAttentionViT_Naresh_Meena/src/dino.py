"""
src/dino.py

DINO self-distillation pretraining for XCiT and L2ViT backbones.

Usage:
    from src.dino import run_dino

    model = build_model('xcit_mae').to(device)
    run_dino(model, pretrain_loader,
             f"{CFG['save_dir']}/xcit_dino_backbone.pth", CFG, 'xcit')
"""

import copy
import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast, GradScaler

from src.augmentations import JetAugmentation

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class DINOHead(nn.Module):
    # Same architecture for student and teacher.
    # Teacher weights are updated by EMA only — never backprop.
    def __init__(self, embed_dim: int, out_dim: int = 65536):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embed_dim, 256),
            nn.GELU(),
            nn.Linear(256, 256),
            nn.GELU(),
            nn.Linear(256, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.net(x), dim=-1)


class DINOLoss(nn.Module):
    """
    Cross-entropy between teacher and student softmax outputs.

    Centering (running mean subtracted from teacher logits) prevents
    collapse — without it the teacher converges to a single prototype.
    Teacher temperature is sharper than student by design.
    """
    def __init__(self, out_dim: int    = 65536,
                 tau_student: float    = 0.1,
                 tau_teacher: float    = 0.04,
                 center_momentum: float = 0.9):
        super().__init__()
        self.tau_s = tau_student
        self.tau_t = tau_teacher
        self.cm    = center_momentum
        self.register_buffer('center', torch.zeros(1, out_dim))

    def forward(self, student_views: list, teacher_views: list) -> torch.Tensor:
        with torch.no_grad():
            teacher_out = [
                F.softmax((t - self.center) / self.tau_t, dim=-1)
                for t in teacher_views
            ]
            all_teacher = torch.cat(teacher_views, dim=0)
            self.center = (self.cm * self.center
                           + (1 - self.cm) * all_teacher.mean(0, keepdim=True))

        student_log = [F.log_softmax(s / self.tau_s, dim=-1) for s in student_views]

        total, n = 0.0, 0
        for t_i, t_prob in enumerate(teacher_out):
            for s_i, s_log in enumerate(student_log):
                if s_i == t_i:
                    continue
                total += -(t_prob * s_log).sum(dim=-1).mean()
                n     += 1

        return total / max(n, 1)


class DINOPretrainer(nn.Module):
    def __init__(self, student_backbone,
                 embed_dim: int,
                 ema_momentum_start: float = 0.996,
                 ema_momentum_end: float   = 1.0,
                 out_dim: int              = 65536,
                 n_local_crops: int        = 4):
        super().__init__()
        self.student      = student_backbone
        self.teacher      = copy.deepcopy(student_backbone)
        self.student_head = DINOHead(embed_dim, out_dim)
        self.teacher_head = DINOHead(embed_dim, out_dim)
        self.loss_fn      = DINOLoss(out_dim)
        self.augment      = JetAugmentation(n_local_crops=n_local_crops)
        self.mom_start    = ema_momentum_start
        self.mom_end      = ema_momentum_end

        for p in self.teacher.parameters():
            p.requires_grad = False
        for p in self.teacher_head.parameters():
            p.requires_grad = False

        self.teacher_head.load_state_dict(self.student_head.state_dict())

    def forward(self, imgs: torch.Tensor) -> torch.Tensor:
        global1, global2, local_crops = self.augment.dino_views(imgs)
        all_views = [global1, global2] + local_crops

        student_outs = [
            self.student_head(self.student.forward_features(v)) for v in all_views
        ]
        with torch.no_grad():
            teacher_outs = [
                self.teacher_head(self.teacher.forward_features(v))
                for v in [global1, global2]
            ]

        return self.loss_fn(student_outs, teacher_outs)

    def update_teacher(self, momentum: float):
        with torch.no_grad():
            for s, t in zip(self.student.parameters(), self.teacher.parameters()):
                t.data = momentum * t.data + (1 - momentum) * s.data
            for s, t in zip(self.student_head.parameters(), self.teacher_head.parameters()):
                t.data = momentum * t.data + (1 - momentum) * s.data

    def get_momentum(self, ep: int, n_epochs: int) -> float:
        progress = ep / max(n_epochs - 1, 1)
        return self.mom_end - (self.mom_end - self.mom_start) * (
            0.5 * (1 + np.cos(np.pi * progress))
        )


def run_dino(model, loader, save_path: str, cfg: dict,
             model_name: str = 'model'):
    """
    DINO pretraining loop. Same interface as run_mae() in train.py.

    Saves student backbone state dict at best loss → save_path
    Loss history → cfg['history_dir']/pretrain_dino_{model_name}.npy

    Schedule details:
        lr=5e-4, 10-epoch linear warmup, cosine decay
        weight_decay: cosine 0.04 → 0.4  (DINO paper)
        EMA momentum: cosine 0.996 → 1.0
    """
    embed_dim = model.backbone.num_features if hasattr(model, 'backbone') \
                else model.num_features
    backbone  = model.backbone if hasattr(model, 'backbone') else model

    dino = DINOPretrainer(
        student_backbone   = model,
        embed_dim          = embed_dim,
        ema_momentum_start = 0.996,
        ema_momentum_end   = 1.0,
        out_dim            = 65536,
        n_local_crops      = 4,
    ).to(device)

    n_epochs       = cfg['pretrain_epochs']
    warmup_ep      = 10
    wd_start, wd_end = 0.04, 0.4

    optimizer = torch.optim.AdamW(
        list(dino.student.parameters()) + list(dino.student_head.parameters()),
        lr=5e-4, weight_decay=wd_start,
    )

    def lr_lambda(ep):
        if ep < warmup_ep:
            return (ep + 1) / warmup_ep
        progress = (ep - warmup_ep) / max(1, n_epochs - warmup_ep)
        return 0.5 * (1 + np.cos(np.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    scaler    = GradScaler(enabled=cfg['use_amp'])
    best_loss = float('inf')
    history   = []

    print(f"\nDINO Pretraining — {model_name}")
    print("=" * 50)

    for ep in range(n_epochs):
        dino.train()
        ep_loss, n_batches = 0.0, 0
        t0 = time.time()

        progress_wd = ep / max(n_epochs - 1, 1)
        wd_current  = wd_end - (wd_end - wd_start) * (
            0.5 * (1 + np.cos(np.pi * progress_wd))
        )
        for pg in optimizer.param_groups:
            pg['weight_decay'] = wd_current

        for imgs in loader:
            imgs = imgs.to(device, non_blocking=True)
            optimizer.zero_grad()

            with autocast(enabled=cfg['use_amp']):
                loss = dino(imgs)

            if torch.isnan(loss):
                continue

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(
                list(dino.student.parameters()) + list(dino.student_head.parameters()),
                cfg['grad_clip']
            )
            scaler.step(optimizer)
            scaler.update()

            momentum = dino.get_momentum(ep, n_epochs)
            dino.update_teacher(momentum)

            ep_loss   += loss.item()
            n_batches += 1

        scheduler.step()
        avg = ep_loss / max(n_batches, 1)
        history.append(avg)

        if avg < best_loss:
            best_loss = avg
            torch.save(backbone.state_dict(), save_path)

        if (ep + 1) % 5 == 0 or ep == 0:
            mom = dino.get_momentum(ep, n_epochs)
            print(f"  ep {ep+1:3d}/{n_epochs}  loss={avg:.5f}  "
                  f"best={best_loss:.5f}  lr={scheduler.get_last_lr()[0]:.2e}  "
                  f"wd={wd_current:.4f}  mom={mom:.4f}  "
                  f"t={time.time()-t0:.0f}s")

    np.save(
        os.path.join(cfg['history_dir'], f'pretrain_dino_{model_name}.npy'),
        np.array(history)
    )

    backbone.load_state_dict(torch.load(save_path, map_location=device))
    print(f"\n  Best loss : {best_loss:.5f}\n  Saved     : {save_path}")
    return history