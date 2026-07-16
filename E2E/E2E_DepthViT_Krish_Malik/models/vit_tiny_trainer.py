"""
ViT-Tiny trainer for HLS4ML LHC jets.

Reuses the existing DepthViT infrastructure:
  - UnifiedDataModule + lhc_jets format from data/
  - build_trainer, _build_callbacks, set_seed from imagenet_trainer.py

Only the LightningModule is new (ViTTinyModule). Single-phase
classification only — no iJEPA support (ViT-Tiny is the baseline,
not the subject of pretraining ablations for the workshop paper).

Usage:
    python3 -m torch.distributed.run --standalone --nproc_per_node=4 \
        vit_tiny_trainer.py --config configs/jets_vit_tiny_50p.json

Smoke test (caps to N batches per phase):
    python3 -m torch.distributed.run --standalone --nproc_per_node=4 \
        vit_tiny_trainer.py --config configs/jets_vit_tiny_smoke.json \
        --smoke-test 5
"""

import argparse
import json
import math
import os
from typing import Any, Dict, Optional

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR

import lightning as L

from vit_tiny import ViTTiny
from imagenet_trainer import (
    UnifiedDataModule,
    build_trainer,
    _build_callbacks,
    _pick_resume_ckpt,
    set_seed,
    WarmupLinearDecayLR,
)


# --------------------------------------------------------------------- #
#  LightningModule                                                       #
# --------------------------------------------------------------------- #

class ViTTinyModule(L.LightningModule):
    """Classification-only Lightning module for ViT-Tiny."""

    def __init__(
        self,
        model_cfg: Dict[str, Any],
        optim_cfg: Dict[str, Any],
        sched_cfg: Dict[str, Any],
        data_cfg: Dict[str, Any],
        max_epochs: int = 30,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["data_cfg"])
        self.optim_cfg = optim_cfg
        self.sched_cfg = sched_cfg
        self.data_cfg = data_cfg
        self.max_epochs = max_epochs

        self.model = ViTTiny(
            img_size=int(model_cfg.get("img_size", 100)),
            patch_size=int(model_cfg.get("patch_size", 10)),
            in_channels=int(data_cfg.get("n_channels", 2)),
            num_classes=int(data_cfg.get("num_classes", 5)),
            embed_dim=int(model_cfg.get("embed_dim", 192)),
            num_layers=int(model_cfg.get("num_layers", 12)),
            num_heads=int(model_cfg.get("num_heads", 3)),
            mlp_ratio=float(model_cfg.get("mlp_ratio", 4.0)),
            dropout=float(model_cfg.get("dropout", 0.0)),
            attn_dropout=float(model_cfg.get("attn_dropout", 0.0)),
        )

        if model_cfg.get("compile", False) and hasattr(torch, "compile"):
            try:
                self.model = torch.compile(
                    self.model,
                    mode=str(model_cfg.get("compile_mode", "default")),
                )
            except Exception as e:
                print(f"[WARN] torch.compile failed, falling back: {e}")

        self.criterion = nn.CrossEntropyLoss(
            label_smoothing=float(data_cfg.get("label_smoothing", 0.0))
        )

    # --------------------------------------------------------------- #
    #  Forward / metrics                                               #
    # --------------------------------------------------------------- #

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    @torch.no_grad()
    def _topk(self, logits: torch.Tensor, target: torch.Tensor, ks=(1, 5)):
        maxk = max(ks)
        batch_size = target.size(0)
        _, pred = logits.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))
        res = []
        for k in ks:
            correct_k = correct[:k].any(dim=0).float().sum()
            res.append(correct_k * (100.0 / float(batch_size)))
        return res

    def _extract_batch(self, batch):
        if isinstance(batch, (list, tuple)):
            if len(batch) == 2:
                x, y = batch
            elif len(batch) == 1:
                x, y = batch[0], None
            else:
                raise TypeError(f"Unsupported batch structure: len={len(batch)}")
        else:
            x, y = batch, None
        x = x.to(self.device, non_blocking=True)
        y = y.to(self.device, non_blocking=True) if y is not None else None
        return x, y

    def training_step(self, batch, batch_idx):
        x, y = self._extract_batch(batch)
        if y is None:
            raise ValueError("Expected labels in batch for classification")
        logits = self.forward(x)
        loss = self.criterion(logits, y)
        acc1, acc5 = self._topk(logits, y, ks=(1, 5))
        bs = int(y.size(0))
        self.log_dict(
            {"train_loss": loss, "train_acc1": acc1, "train_acc5": acc5},
            prog_bar=True, on_step=True, on_epoch=True,
            sync_dist=True, reduce_fx="mean", batch_size=bs,
        )
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = self._extract_batch(batch)
        if y is None:
            raise ValueError("Expected labels in batch for classification")
        logits = self.forward(x)
        loss = self.criterion(logits, y)
        acc1, acc5 = self._topk(logits, y, ks=(1, 5))
        bs = int(y.size(0))
        self.log_dict(
            {"val_loss": loss, "val_acc1": acc1, "val_acc5": acc5},
            prog_bar=True, on_epoch=True,
            sync_dist=True, reduce_fx="mean", batch_size=bs,
        )
        return loss

    # --------------------------------------------------------------- #
    #  Optimizer / scheduler                                           #
    # --------------------------------------------------------------- #

    def _steps_per_epoch(self) -> int:
        gbs = int(self.data_cfg["global_batch_size"])
        ds = int(self.data_cfg["dataset_size"])
        accum = 1
        try:
            accum = max(1, int(self.trainer.accumulate_grad_batches))
        except Exception:
            pass
        return max(1, ds // gbs // accum)

    def _total_training_steps(self) -> int:
        return self.max_epochs * self._steps_per_epoch()

    def configure_optimizers(self):
        ocfg = self.optim_cfg
        scfg = self.sched_cfg

        opt_name = str(ocfg.get("optimizer", "sgd")).lower()
        lr = float(ocfg.get("lr", 0.1))
        weight_decay = float(ocfg.get("weight_decay", 1e-4))
        total_steps = self._total_training_steps()

        trainable = [p for p in self.parameters() if p.requires_grad]
        if opt_name == "sgd":
            momentum = float(ocfg.get("momentum", 0.9))
            optimizer = optim.SGD(
                trainable, lr=lr, weight_decay=weight_decay, momentum=momentum
            )
        elif opt_name == "adam":
            optimizer = optim.Adam(trainable, lr=lr, weight_decay=weight_decay)
        else:
            optimizer = optim.AdamW(trainable, lr=lr, weight_decay=weight_decay)

        sched_name = str(scfg.get("sched", "cosine")).lower()
        min_lr = float(scfg.get("min_lr", 1e-6))
        warmup_steps = int(scfg.get("warmup_steps", 0))

        if sched_name == "linear":
            scheduler = WarmupLinearDecayLR(
                optimizer, warmup_steps=warmup_steps,
                total_steps=total_steps, min_lr=min_lr,
            )
            return {
                "optimizer": optimizer,
                "lr_scheduler": {"scheduler": scheduler, "interval": "step", "frequency": 1},
            }
        if sched_name == "cosine":
            min_lr_frac = min_lr / max(lr, 1e-12)

            def lr_lambda(step: int) -> float:
                if step < warmup_steps:
                    return float(step) / float(max(1, warmup_steps))
                t = (step - warmup_steps) / float(max(1, total_steps - warmup_steps))
                return max(min_lr_frac, 0.5 * (1.0 + math.cos(math.pi * t)))

            scheduler = LambdaLR(optimizer, lr_lambda=lr_lambda)
            return {
                "optimizer": optimizer,
                "lr_scheduler": {"scheduler": scheduler, "interval": "step", "frequency": 1},
            }
        return {"optimizer": optimizer}


# --------------------------------------------------------------------- #
#  Entry point                                                           #
# --------------------------------------------------------------------- #

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument(
        "--smoke-test", type=int, default=0,
        help="If >0, cap training at this many batches and exit "
             "(for pipeline validation on the debug queue).",
    )
    args = parser.parse_args()

    with open(args.config, "r") as f:
        cfg = json.load(f)

    seed = int(cfg.get("seed", 42))
    set_seed(seed)

    data_cfg = cfg["data"]
    model_cfg = cfg["model"]
    trainer_cfg = cfg["trainer"]

    # ViT-Tiny is single-phase; if a phases array is given, take the first.
    phases_cfg = cfg.get(
        "phases",
        [{
            "name": "cls_finetune",
            "type": "cls_finetune",
            "epochs": cfg.get("sched", {}).get("epochs", 30),
        }],
    )
    phase_cfg = phases_cfg[0]
    phase_name = phase_cfg.get("name", "cls_finetune")
    phase_type = phase_cfg.get("type", "cls_finetune")
    epochs = int(phase_cfg.get("epochs", 30))

    optim_cfg = phase_cfg.get("optim") or cfg.get("optim")
    sched_cfg = phase_cfg.get("sched") or cfg.get("sched")
    if optim_cfg is None or sched_cfg is None:
        raise ValueError("Config must provide either top-level optim/sched or per-phase.")

    module = ViTTinyModule(
        model_cfg=model_cfg,
        optim_cfg=optim_cfg,
        sched_cfg=sched_cfg,
        data_cfg=data_cfg,
        max_epochs=epochs,
    )

    n_p = sum(p.numel() for p in module.model.parameters() if p.requires_grad)
    print(f"[ViTTiny] trainable params: {n_p:,}  ({n_p/1e6:.2f}M)")

    cfg_phase = dict(cfg)
    cfg_phase["data"] = data_cfg
    dm = UnifiedDataModule(cfg=cfg_phase, phase_cfg=phase_cfg)

    callbacks, ckpt_cb, ckpt_dir = _build_callbacks(cfg, phase_name, phase_type)

    trainer_cfg_local = dict(trainer_cfg)
    if args.smoke_test > 0:
        trainer_cfg_local["limit_train_batches"] = args.smoke_test
        trainer_cfg_local["limit_val_batches"] = 1

    trainer = build_trainer(
        trainer_cfg_local,
        max_epochs=epochs,
        default_root_dir=os.path.join(
            trainer_cfg.get("default_root_dir", "./runs"), phase_name
        ),
        callbacks=callbacks,
    )

    resume_cfg = cfg.get("resume", {})
    ckpt_path = _pick_resume_ckpt(ckpt_dir, resume_cfg.get("path"))

    if ckpt_path:
        print(f"[ViTTiny] resuming from {ckpt_path}")
        trainer.fit(module, dm, ckpt_path=ckpt_path)
    else:
        trainer.fit(module, dm)


if __name__ == "__main__":
    main()
