"""
resnet9_trainer.py — train the matched-size ResNet9 CNN baseline.

Mirrors the structure of vit_tiny_trainer.py / DepthViTModule but builds a plain
CNN with a standard forward(x). Reuses the shared training infrastructure from
imagenet_trainer.py (data dispatch, scheduler, callbacks, trainer builder, top-k
metric convention) so that ResNet9 numbers are directly comparable to DepthViT
and ViT-Tiny under the same protocol.

Single-phase classification from scratch — no pretraining, no multi-phase logic.

Usage (4xA100 DDP):
    python3 -m torch.distributed.run --standalone --nproc_per_node=4 \
        resnet9_trainer.py --config configs/jets_150p_resnet9.json

Login-node smoke test (cap batches, CPU/1-GPU):
    python3 resnet9_trainer.py --config configs/jets_150p_resnet9.json --smoke-test 4
"""
import argparse
import json
import os
from typing import Any, Dict, Optional

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR

import lightning as L

from ResNet9 import ResNet9

# Reuse the exact same infrastructure DepthViT/ViT-Tiny run on, so the
# comparison is apples-to-apples (same loaders, same scheduler math, same
# checkpoint/metric plumbing). main() is guarded in imagenet_trainer, so the
# import has no side effects.
from imagenet_trainer import (
    set_seed,
    maybe_compile,
    WarmupLinearDecayLR,
    UnifiedDataModule,
    build_trainer,
    _build_callbacks,
    _pick_resume_ckpt,
)


class ResNet9Module(L.LightningModule):
    def __init__(
        self,
        model_cfg: Dict[str, Any],
        optim_cfg: Dict[str, Any],
        sched_cfg: Dict[str, Any],
        data_cfg: Dict[str, Any],
        max_epochs: int = 1,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["data_cfg"])
        num_classes = int(data_cfg.get("num_classes", 5))
        self.sched_cfg = sched_cfg
        self.optim_cfg = optim_cfg
        self.data_cfg = data_cfg
        self.max_epochs = max_epochs

        widths = tuple(model_cfg.get("widths", [14, 24, 44, 78]))
        self.model = ResNet9(
            in_channels=int(data_cfg["n_channels"]),
            num_classes=num_classes,
            widths=widths,
        )

        # Optional SyncBatchNorm for multi-GPU (off by default; 128/GPU is plenty
        # for stable BN stats at global_batch_size=512 on 4 GPUs).
        if bool(model_cfg.get("sync_bn", False)):
            self.model = nn.SyncBatchNorm.convert_sync_batchnorm(self.model)

        self.model = maybe_compile(
            self.model,
            bool(model_cfg.get("compile", False)),
            mode=str(model_cfg.get("compile_mode", "default")),
        )

        # Optional warm-start of trunk weights (kept for parity; unused for the
        # from-scratch baseline).
        ckpt = model_cfg.get("checkpoint_path", None)
        if ckpt:
            self._load_weights(ckpt)

        self.criterion = nn.CrossEntropyLoss(
            label_smoothing=float(data_cfg.get("label_smoothing", 0.0))
        )

    def _load_weights(self, ckpt_path: str):
        ckpt = torch.load(ckpt_path, map_location="cpu")
        sd = ckpt.get("state_dict", ckpt)
        sd = {
            k.replace("module.", "").replace("model.", "").replace("_orig_mod.", ""): v
            for k, v in sd.items()
        }
        model_to_load = getattr(self.model, "_orig_mod", self.model)
        model_sd = model_to_load.state_dict()
        filtered = {
            k: v for k, v in sd.items()
            if k in model_sd and model_sd[k].shape == v.shape and "fc" not in k
        }
        msg = model_to_load.load_state_dict(filtered, strict=False)
        print(f"Loaded {len(filtered)} tensors. "
              f"Missing: {len(msg.missing_keys)}  Unexpected: {len(msg.unexpected_keys)}")

    @torch.no_grad()
    def _topk(self, logits, target, ks=(1, 5)):
        # Same metric convention as imagenet_trainer._topk so values are
        # directly comparable to DepthViT / ViT-Tiny logs.
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

    def forward(self, x):
        return self.model(x)

    def _extract_batch(self, batch):
        if isinstance(batch, (list, tuple)):
            if len(batch) == 2:
                x, y = batch
            elif len(batch) == 1:
                x, y = batch[0], None
            else:
                raise TypeError("Unsupported batch structure")
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
        self.log_dict(
            {"train_loss": loss, "train_acc1": acc1, "train_acc5": acc5},
            prog_bar=True, on_step=True, on_epoch=True, sync_dist=True,
            reduce_fx="mean", batch_size=int(y.size(0)),
        )
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = self._extract_batch(batch)
        if y is None:
            raise ValueError("Expected labels in batch for classification")
        logits = self.forward(x)
        loss = self.criterion(logits, y)
        acc1, acc5 = self._topk(logits, y, ks=(1, 5))
        self.log_dict(
            {"val_loss": loss, "val_acc1": acc1, "val_acc5": acc5},
            prog_bar=True, on_epoch=True, sync_dist=True,
            reduce_fx="mean", batch_size=int(y.size(0)),
        )
        return loss

    def _steps_per_epoch(self):
        gbs = int(self.data_cfg["global_batch_size"])
        ds = int(self.data_cfg["dataset_size"])
        accum = 1
        try:
            accum = max(1, int(self.trainer.accumulate_grad_batches))
        except Exception:
            pass
        return max(1, ds // gbs // accum)

    def _total_training_steps(self):
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
            optimizer = optim.SGD(trainable, lr=lr, weight_decay=weight_decay, momentum=momentum)
        elif opt_name == "adam":
            optimizer = optim.Adam(trainable, lr=lr, weight_decay=weight_decay)
        else:
            optimizer = optim.AdamW(trainable, lr=lr, weight_decay=weight_decay)

        sched_name = str(scfg.get("sched", "cosine")).lower()
        min_lr = float(scfg.get("min_lr", 1e-6))
        warmup_steps = int(scfg.get("warmup_steps", 0))
        if sched_name == "linear":
            scheduler = WarmupLinearDecayLR(
                optimizer, warmup_steps=warmup_steps, total_steps=total_steps, min_lr=min_lr
            )
            return {"optimizer": optimizer,
                    "lr_scheduler": {"scheduler": scheduler, "interval": "step", "frequency": 1}}
        if sched_name == "cosine":
            import math
            min_lr_frac = min_lr / max(lr, 1e-12)
            def lr_lambda(step):
                if step < warmup_steps:
                    return float(step) / float(max(1, warmup_steps))
                t = (step - warmup_steps) / float(max(1, total_steps - warmup_steps))
                return max(min_lr_frac, 0.5 * (1.0 + math.cos(math.pi * t)))
            scheduler = LambdaLR(optimizer, lr_lambda=lr_lambda)
            return {"optimizer": optimizer,
                    "lr_scheduler": {"scheduler": scheduler, "interval": "step", "frequency": 1}}
        return {"optimizer": optimizer}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--smoke-test", type=int, default=0,
                        help="If >0, cap training at this many batches and exit.")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        cfg = json.load(f)

    # TF32 tensor cores on A100 for any fp32 math outside the bf16-autocast
    # region (BatchNorm running stats, optimizer updates, etc.) — free
    # speedup, negligible precision cost given bf16-mixed is already in use.
    torch.set_float32_matmul_precision("high")

    # Launch is via `torch.distributed.run --standalone`, which sets its own
    # RANK/LOCAL_RANK/WORLD_SIZE. Lightning's SLURMEnvironment.detect() always
    # calls _validate_srun_used(), which warns whenever `srun` is present on
    # PATH but not the actual launch command — true inside any sbatch job,
    # independent of SLURM_NTASKS. The previous fix (popping SLURM_NTASKS)
    # targeted the wrong check and did nothing. Lightning's own documented
    # escape hatch is SLURM_JOB_NAME in {"bash", "interactive"}, which makes
    # _is_slurm_interactive_mode() short-circuit the warning entirely and
    # lets the environment fall back correctly to LightningEnvironment.
    os.environ["SLURM_JOB_NAME"] = "interactive"

    set_seed(int(cfg.get("seed", 42)))

    data_cfg = cfg["data"]
    model_cfg = cfg["model"]
    trainer_cfg = cfg["trainer"]

    phases_cfg = cfg.get("phases", [{"name": "cls_finetune", "type": "cls_finetune",
                                     "epochs": cfg.get("sched", {}).get("epochs", 90)}])
    phase_cfg = phases_cfg[0]
    phase = phase_cfg.get("type", "cls_finetune")
    phase_name = phase_cfg.get("name", phase)
    epochs = int(phase_cfg.get("epochs", 90))

    optim_cfg = phase_cfg.get("optim") or cfg.get("optim")
    sched_cfg = phase_cfg.get("sched") or cfg.get("sched")

    module = ResNet9Module(
        model_cfg=model_cfg,
        optim_cfg=optim_cfg,
        sched_cfg=sched_cfg,
        data_cfg=data_cfg,
        max_epochs=epochs,
    )

    dm = UnifiedDataModule(cfg=cfg, phase_cfg=phase_cfg)

    callbacks, ckpt_cb, ckpt_dir = _build_callbacks(cfg, phase_name, phase)

    trainer_cfg_local = dict(trainer_cfg)
    if args.smoke_test > 0:
        trainer_cfg_local["limit_train_batches"] = args.smoke_test
        trainer_cfg_local["limit_val_batches"] = 1
        # Smoke tests run on the login node (no SLURM GPU allocation), so the
        # config's real devices=4/strategy=ddp would fail outside the job.
        # devices=1 + strategy="auto" works whether or not a GPU is visible:
        # Lightning's accelerator defaults to "auto", which picks CPU when
        # no CUDA device is present and a single GPU otherwise.
        trainer_cfg_local["devices"] = 1
        trainer_cfg_local["strategy"] = "auto"
        if not torch.cuda.is_available():
            trainer_cfg_local["precision"] = "32-true"

    trainer = build_trainer(
        trainer_cfg_local, max_epochs=epochs,
        default_root_dir=os.path.join(trainer_cfg.get("default_root_dir", "./runs"), phase_name),
        callbacks=callbacks,
    )

    resume_cfg = cfg.get("resume", {})
    ckpt_path = _pick_resume_ckpt(ckpt_dir, resume_cfg.get("path"))
    trainer.fit(module, dm, ckpt_path=ckpt_path)


if __name__ == "__main__":
    main()
