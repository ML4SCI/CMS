import argparse
import copy
import glob
import json
import math
import os
import random
from typing import Any, Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.distributed as dist
from torch.optim.lr_scheduler import LambdaLR

# Backward-compatible: keep the IMAGENET constants for any
# code that still uses them (e.g. _denorm). Loader dispatch
# now goes through the data package's format registry.
from wds_data import IMAGENET_MEAN, IMAGENET_STD
from data import make_loaders_dispatch
from DepthViT import *

import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor, TQDMProgressBar

# WebDataset is preferred for large tar shard datasets
try:
    import webdataset as wds
except Exception as e:
    wds = None

def set_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def _denorm(img):
    mean = torch.tensor(IMAGENET_MEAN, device=img.device)[None, :, None, None]
    std  = torch.tensor(IMAGENET_STD,  device=img.device)[None, :, None, None]
    return img * std + mean

def maybe_compile(model: nn.Module, enabled: bool, mode: str = "default"):
    if not enabled:
        return model
    if hasattr(torch, "compile"):
        try:
            return torch.compile(model, mode=mode)
        except Exception:
            return model
    return model

def _urls(path: str):
    return os.path.join(path, "*.tar") if os.path.isdir(path) else path


class ModelEMA(nn.Module):
    def __init__(self, model: nn.Module, decay: float = 0.9999, update_every: int = 1):
        super().__init__()
        if not (0.0 < float(decay) < 1.0):
            raise ValueError(f"EMA decay must be in (0,1); got {decay}")
        self.decay = float(decay)
        self.update_every = int(update_every) if int(update_every) > 0 else 1
        self.ema_model = copy.deepcopy(model)
        self.ema_model.requires_grad_(False)
        self._num_updates = 0

    @torch.no_grad()
    def update(self, model: nn.Module):
        self._num_updates += 1
        if (self._num_updates % self.update_every) != 0:
            return

        msd = model.state_dict()
        esd = self.ema_model.state_dict()
        for k, v_ema in esd.items():
            v_model = msd[k]
            if not torch.is_floating_point(v_ema):
                v_ema.copy_(v_model)
            else:
                v_ema.mul_(self.decay).add_(v_model, alpha=1.0 - self.decay)

    @torch.no_grad()
    def copy_to(self, model: nn.Module):
        msd = model.state_dict()
        esd = self.ema_model.state_dict()
        for k, v in msd.items():
            v.copy_(esd[k])

class WarmupLinearDecayLR(LambdaLR):
    def __init__(self, optimizer, warmup_steps: int, total_steps: int, min_lr: float = 0.0, last_epoch: int = -1):
        self.warmup_steps = max(1, int(warmup_steps))
        self.total_steps = max(self.warmup_steps + 1, int(total_steps))
        self.min_lr = float(min_lr)
        self._base_lr = float(optimizer.param_groups[0]["lr"])
        super().__init__(optimizer, lr_lambda=self.lr_lambda, last_epoch=last_epoch)

    def lr_lambda(self, step: int):
        if step < self.warmup_steps:
            return float(step) / float(max(1, self.warmup_steps))
        # decay to min_lr linearly
        decay_steps = max(1, self.total_steps - self.warmup_steps)
        progress = float(step - self.warmup_steps) / float(decay_steps)
        progress = min(max(progress, 0.0), 1.0)
        min_frac = self.min_lr / max(self._base_lr, 1e-12)
        return max(min_frac, 1.0 - progress)

class DepthViTModule(L.LightningModule):
    def __init__(
        self,
        model_cfg: Dict[str, Any],
        optim_cfg: Dict[str, Any],
        sched_cfg: Dict[str, Any],
        data_cfg: Dict[str, Any],
        phase: str = "cls_finetune",
        ema_cfg: Optional[Dict[str, Any]] = None,
        max_epochs: int = 1,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["data_cfg"])
        num_classes = int(data_cfg.get("num_classes", 1000))
        self.sched_cfg = sched_cfg
        self.optim_cfg = optim_cfg
        self.model = DepthViT(
            in_channels=data_cfg["n_channels"],
            k_factor=model_cfg["k_factor"],
            patch_size=model_cfg["patch_size"],
            num_layers=model_cfg["n_layers"],
            mlp_dim=model_cfg["mlp_dim"],
            linear_rank=model_cfg["linear_rank"],
            dropout=model_cfg["dropout"],
            num_classes=num_classes,
            max_image_height=model_cfg["max_image_height"],
            max_image_width=model_cfg["max_image_width"],
            num_hap_layers=int(model_cfg.get("num_hap_layers", 0)),
            hap_window_size=int(model_cfg.get("hap_window_size", 8)),
            hap_mlp_ratio=float(model_cfg.get("hap_mlp_ratio", 4.0)),
            hap_drop_path=float(model_cfg.get("hap_drop_path", 0.0)),
            grad_checkpointing=bool(model_cfg.get("grad_checkpointing", True)),
            k_chunk_size=int(model_cfg.get("k_chunk_size", 0)),
            compile_blocks=bool(model_cfg.get("compile_blocks", False)),
        )
        self.max_epochs = max_epochs
        self.data_cfg = data_cfg

        # Optionally compile
        self.model = maybe_compile(self.model, bool(model_cfg.get("compile", False)), mode=str(model_cfg.get("compile_mode", "default")))

        # Load checkpoint weights if provided
        ckpt = model_cfg.get("checkpoint_path", None)
        if ckpt:
            self._load_weights(ckpt)

        self.phase = phase
        self.criterion = nn.CrossEntropyLoss(label_smoothing=float(data_cfg.get("label_smoothing", 0.0)))

        self.ema_cfg = ema_cfg or {}
        self._use_ema_for_val = bool(self.ema_cfg.get("use_for_val", True))
        self._ema_backup: Optional[Dict[str, torch.Tensor]] = None
        self.ema: Optional[ModelEMA] = None
        if bool(self.ema_cfg.get("enabled", False)):
            base = getattr(self.model, "_orig_mod", self.model)
            self.ema = ModelEMA(
                base,
                decay=float(self.ema_cfg.get("decay", 0.9999)),
                update_every=int(self.ema_cfg.get("update_every", 1)),
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

        filtered = {}
        for k, v in sd.items():
            if "token_attn_head" in k:
                continue
            if k in model_sd and model_sd[k].shape == v.shape:
                filtered[k] = v

        msg = model_to_load.load_state_dict(filtered, strict=False)
        print(f"Loaded {len(filtered)} tensors. "
              f"Missing: {len(msg.missing_keys)}  Unexpected: {len(msg.unexpected_keys)}")

    def load_state_dict(self, state_dict, strict: bool = True):
        # Backward-compat: older checkpoints won't have EMA weights.
        ignore_prefixes = ("ema.",)
        has_ema_in_ckpt = any(k.startswith(ignore_prefixes) for k in state_dict.keys())

        incompatible = super().load_state_dict(state_dict, strict=False)

        # If the checkpoint didn't contain EMA weights, initialize EMA from current model weights.
        if getattr(self, "ema", None) is not None and not has_ema_in_ckpt:
            base = getattr(self.model, "_orig_mod", self.model)
            self.ema.ema_model.load_state_dict(base.state_dict(), strict=True)

        if strict:
            missing = [k for k in incompatible.missing_keys if not k.startswith(ignore_prefixes)]
            unexpected = [k for k in incompatible.unexpected_keys if not k.startswith(ignore_prefixes)]
            if missing or unexpected:
                lines = [f"Error(s) in loading state_dict for {self.__class__.__name__}:"]
                if missing:
                    lines.append('        Missing key(s) in state_dict: ' + ", ".join(f'"{k}"' for k in missing) + ".")
                if unexpected:
                    lines.append('        Unexpected key(s) in state_dict: ' + ", ".join(f'"{k}"' for k in unexpected) + ".")
                raise RuntimeError("\n".join(lines))

        return incompatible

    @torch.no_grad()
    def _topk(self, logits, target, ks=(1, 5)):
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
        return self.model.forward_cls(x)

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
            raise ValueError("Expected labels in batch for classification phases")
        logits = self.forward(x)
        loss = self.criterion(logits, y)
        acc1, acc5 = self._topk(logits, y, ks=(1, 5))
        self.log_dict({"train_loss": loss, "train_acc1": acc1, "train_acc5": acc5},
                      prog_bar=True, on_step=True, on_epoch=True, sync_dist=True, reduce_fx="mean", batch_size=int(y.size(0)))
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = self._extract_batch(batch)
        if y is None:
            raise ValueError("Expected labels in batch for classification phases")
        logits = self.forward(x)
        loss = self.criterion(logits, y)
        acc1, acc5 = self._topk(logits, y, ks=(1, 5))
        self.log_dict({"val_loss": loss, "val_acc1": acc1, "val_acc5": acc5},
                      prog_bar=True, on_epoch=True, sync_dist=True, reduce_fx="mean", batch_size=int(y.size(0)))
        return loss

    def optimizer_step(self, *args, **kwargs):
        out = super().optimizer_step(*args, **kwargs)
        if self.ema is not None:
            base = getattr(self.model, "_orig_mod", self.model)
            self.ema.update(base)
        return out

    def on_validation_epoch_start(self) -> None:
        if self.ema is None or not self._use_ema_for_val:
            return
        base = getattr(self.model, "_orig_mod", self.model)
        self._ema_backup = {k: v.detach().cpu().clone() for k, v in base.state_dict().items()}
        self.ema.copy_to(base)

    def on_validation_epoch_end(self) -> None:
        if self.ema is None or not self._use_ema_for_val:
            return
        if self._ema_backup is None:
            return
        base = getattr(self.model, "_orig_mod", self.model)
        sd = base.state_dict()
        for k, v in sd.items():
            v.copy_(self._ema_backup[k].to(v.device))
        self._ema_backup = None

    def configure_optimizers(self):
        ocfg = self.optim_cfg
        scfg = self.sched_cfg

        opt_name = str(ocfg.get("optimizer", "adamw")).lower()
        lr = float(ocfg.get("lr", 5e-4))
        weight_decay = float(ocfg.get("weight_decay", 0.05))
        total_steps = self._total_training_steps()

        trainable = [p for p in self.parameters() if p.requires_grad]
        if opt_name == "sgd":
            optimizer = optim.SGD(trainable, lr=lr, weight_decay=weight_decay, momentum=0.9)
        elif opt_name == "adam":
            optimizer = optim.Adam(trainable, lr=lr, weight_decay=weight_decay)
        else:
            optimizer = optim.AdamW(trainable, lr=lr, weight_decay=weight_decay)

        # Cosine scheduler
        sched_name = str(scfg.get("sched", "cosine")).lower()
        epochs = int(scfg.get("epochs", 0) or getattr(self.trainer, "max_epochs", 30))
        min_lr = float(scfg.get("min_lr", 1e-6))
        warmup_steps = int(scfg.get("warmup_steps", 0))
        if sched_name == "linear":
            scheduler = WarmupLinearDecayLR(optimizer, warmup_steps=warmup_steps, total_steps=total_steps, min_lr=min_lr)
            return {
                "optimizer": optimizer,
                "lr_scheduler": {"scheduler": scheduler, "interval": "step", "frequency": 1}
            }
        elif sched_name == "cosine":
            min_lr_frac = min_lr / max(lr, 1e-12)
            def lr_lambda(step):
                if step < warmup_steps:
                    return float(step) / float(max(1, warmup_steps))
                t = (step - warmup_steps) / float(max(1, total_steps - warmup_steps))
                return max(min_lr_frac, 0.5 * (1.0 + math.cos(math.pi * t)))
            scheduler = LambdaLR(optimizer, lr_lambda=lr_lambda)
            return {
                "optimizer": optimizer,
                "lr_scheduler": {"scheduler": scheduler, "interval": "step", "frequency": 1}
            }
        else:
            return {"optimizer": optimizer}

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


class IJEPA_DepthViTModule(L.LightningModule):
    def __init__(
        self,
        model_cfg: Dict[str, Any],
        ijepa_cfg: Dict[str, Any],
        optim_cfg: Dict[str, Any],
        sched_cfg: Dict[str, Any],
        data_cfg: Dict[str, Any],
        target_model_cfg: Optional[Dict[str, Any]] = None,
        ema_cfg: Optional[Dict[str, Any]] = None,
        max_epochs: int = 1,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["data_cfg"])
        self.sched_cfg = sched_cfg
        self.optim_cfg = optim_cfg
        self.data_cfg = data_cfg
        self.ijepa_cfg = ijepa_cfg or {}
        self.max_epochs = max_epochs

        # I-JEPA uses an online (context) encoder, a target encoder (EMA), and a predictor.
        # Keep num_classes tiny to avoid allocating a huge classification head on large label spaces.
        def _build_depthvit(mcfg):
            return DepthViT(
                in_channels=data_cfg["n_channels"],
                k_factor=mcfg["k_factor"],
                patch_size=mcfg["patch_size"],
                num_layers=mcfg["n_layers"],
                mlp_dim=mcfg["mlp_dim"],
                linear_rank=int(mcfg.get("linear_rank", 128)),
                dropout=float(mcfg.get("dropout", 0.0)),
                num_classes=int(data_cfg.get("pretrain_num_classes", 1)),
                max_image_height=mcfg["max_image_height"],
                max_image_width=mcfg["max_image_width"],
                num_hap_layers=int(mcfg.get("num_hap_layers", 0)),
                hap_window_size=int(mcfg.get("hap_window_size", 8)),
                hap_mlp_ratio=float(mcfg.get("hap_mlp_ratio", 4.0)),
                hap_drop_path=float(mcfg.get("hap_drop_path", 0.0)),
                hap_alpha=float(model_cfg.get("hap_alpha", 1.0)),
                hap_learnable_alpha=bool(model_cfg.get("hap_learnable_alpha", False)),
                grad_checkpointing=bool(mcfg.get("grad_checkpointing", True)),
                k_chunk_size=int(mcfg.get("k_chunk_size", 0)),
                compile_blocks=bool(mcfg.get("compile_blocks", False)),
            )

        self.model = _build_depthvit(model_cfg)

        base = getattr(self.model, "_orig_mod", self.model)
        # Token head isn't used for I-JEPA; freezing avoids DDP unused-parameter errors.
        head = getattr(base, "token_attn_head", None)
        if head is not None:
            for p in head.parameters():
                p.requires_grad = False
            head.eval()

        # Target encoder: use separate config if provided, else mirror online encoder.
        if target_model_cfg:
            self.target_model = _build_depthvit(target_model_cfg)
        else:
            self.target_model = copy.deepcopy(base)
        self.target_model.requires_grad_(False)
        self.target_model.eval()

        # If target encoder has different hidden_dim, add a projection so the
        # predictor output (online hidden_dim) can be compared with target output.
        online_dim = base.hidden_dim
        target_dim = self.target_model.hidden_dim
        if online_dim != target_dim:
            self.target_proj = nn.Linear(target_dim, online_dim, bias=False)
            nn.init.eye_(self.target_proj.weight[:min(online_dim, target_dim), :min(online_dim, target_dim)])
        else:
            self.target_proj = None

        pred_layers = int(self.ijepa_cfg.get("predictor_layers", max(2, int(model_cfg.get("n_layers", 12)) // 3)))
        pred_mlp_dim = int(self.ijepa_cfg.get("predictor_mlp_dim", int(model_cfg.get("mlp_dim", 256))))
        pred_dropout = float(self.ijepa_cfg.get("predictor_dropout", float(model_cfg.get("dropout", 0.0))))
        pred_hap_layers = int(self.ijepa_cfg.get("predictor_hap_layers", int(model_cfg.get("num_hap_layers", 0))))
        pred_hap_ws = int(self.ijepa_cfg.get("predictor_hap_window_size", int(model_cfg.get("hap_window_size", 7))))
        pred_hap_mlp = float(self.ijepa_cfg.get("predictor_hap_mlp_ratio", float(model_cfg.get("hap_mlp_ratio", 4.0))))
        self.predictor = IJEPAPredictor(
            num_layers=pred_layers,
            num_channels=data_cfg["n_channels"],
            k_factor=model_cfg["k_factor"],
            hidden_dim=online_dim,
            mlp_dim=pred_mlp_dim,
            dropout=pred_dropout,
            max_image_size=base.max_image_size,
            num_hap_layers=pred_hap_layers,
            hap_window_size=pred_hap_ws,
            hap_mlp_ratio=pred_hap_mlp,
        )

        # Optionally compile online encoder + predictor (target stays eager for EMA updates).
        self.model = maybe_compile(self.model, bool(model_cfg.get("compile", False)), mode=str(model_cfg.get("compile_mode", "default")))
        self.predictor = maybe_compile(self.predictor, bool(model_cfg.get("compile", False)), mode=str(model_cfg.get("compile_mode", "default")))

        ckpt = model_cfg.get("checkpoint_path", None)
        if ckpt:
            self._load_weights(ckpt)

        # Optional EMA over online weights for validation (separate from target encoder EMA).
        self.ema_cfg = ema_cfg or {}
        self._use_ema_for_val = bool(self.ema_cfg.get("use_for_val", False))
        self._ema_backup: Optional[Dict[str, torch.Tensor]] = None
        self.ema: Optional[ModelEMA] = None
        if bool(self.ema_cfg.get("enabled", False)):
            base_online = getattr(self.model, "_orig_mod", self.model)
            self.ema = ModelEMA(
                base_online,
                decay=float(self.ema_cfg.get("decay", 0.9999)),
                update_every=int(self.ema_cfg.get("update_every", 1)),
            )

        self.loss_type = str(self.ijepa_cfg.get("loss", "mse")).lower()
        self.normalize_targets = bool(self.ijepa_cfg.get("normalize_targets", True))

    def _load_weights(self, ckpt_path: str):
        ckpt = torch.load(ckpt_path, map_location="cpu")
        sd = ckpt.get("state_dict", ckpt)

        # normalize key prefixes from DDP / Lightning / torch.compile
        sd = {
            k.replace("module.", "").replace("_orig_mod.", ""): v
            for k, v in sd.items()
        }

        online = getattr(self.model, "_orig_mod", self.model)
        pred = getattr(self.predictor, "_orig_mod", self.predictor)

        online_sd = online.state_dict()
        pred_sd = pred.state_dict()

        online_filtered = {}
        pred_filtered = {}

        for k, v in sd.items():
            if k.startswith("model."):
                kk = k[len("model."):]
                if "token_attn_head" in kk:
                    continue
                if kk in online_sd and online_sd[kk].shape == v.shape:
                    online_filtered[kk] = v
            elif k.startswith("predictor."):
                kk = k[len("predictor."):]
                if kk in pred_sd and pred_sd[kk].shape == v.shape:
                    pred_filtered[kk] = v

        online.load_state_dict(online_filtered, strict=False)
        pred.load_state_dict(pred_filtered, strict=False)

        # Sync target encoder from online weights where shapes match.
        tgt_sd = self.target_model.state_dict()
        online_sd = online.state_dict()
        tgt_update = {}
        for k, v_t in tgt_sd.items():
            if k in online_sd and online_sd[k].shape == v_t.shape:
                tgt_update[k] = online_sd[k]
        self.target_model.load_state_dict({**tgt_sd, **tgt_update}, strict=True)

    def load_state_dict(self, state_dict, strict: bool = True):
        # Backward-compat: older checkpoints won't have EMA weights.
        ignore_prefixes = ("ema.",)
        has_ema_in_ckpt = any(k.startswith(ignore_prefixes) for k in state_dict.keys())

        incompatible = super().load_state_dict(state_dict, strict=False)

        if getattr(self, "ema", None) is not None and not has_ema_in_ckpt:
            base = getattr(self.model, "_orig_mod", self.model)
            self.ema.ema_model.load_state_dict(base.state_dict(), strict=True)

        if strict:
            missing = [k for k in incompatible.missing_keys if not k.startswith(ignore_prefixes)]
            unexpected = [k for k in incompatible.unexpected_keys if not k.startswith(ignore_prefixes)]
            if missing or unexpected:
                lines = [f"Error(s) in loading state_dict for {self.__class__.__name__}:"]
                if missing:
                    lines.append('        Missing key(s) in state_dict: ' + ", ".join(f'"{k}"' for k in missing) + ".")
                if unexpected:
                    lines.append('        Unexpected key(s) in state_dict: ' + ", ".join(f'"{k}"' for k in unexpected) + ".")
                raise RuntimeError("\n".join(lines))

        return incompatible

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

    def _target_momentum(self) -> float:
        base = float(self.ijepa_cfg.get("target_ema_decay", 0.996))
        final = float(self.ijepa_cfg.get("target_ema_decay_final", base))
        total = int(self._total_training_steps())
        if total <= 1 or base == final:
            return base
        t = float(self.global_step) / float(max(1, total - 1))
        t = min(max(t, 0.0), 1.0)
        return final - (final - base) * (math.cos(math.pi * t) + 1.0) / 2.0

    def _sample_mask(self, batch_size: int, n_h: int, n_w: int, device: torch.device) -> torch.Tensor:
        L = int(n_h * n_w)
        mask_ratio = float(self.ijepa_cfg.get("mask_ratio", 0.5))
        target = int(round(L * mask_ratio))
        target = max(1, min(target, L - 1))

        mask_type = str(self.ijepa_cfg.get("mask_type", "block")).lower()
        if mask_type == "random":
            noise = torch.rand(batch_size, L, device=device)
            ids = torch.argsort(noise, dim=1)
            ids_mask = ids[:, :target]  # mask these
            mask = torch.zeros(batch_size, L, device=device, dtype=torch.bool)
            mask.scatter_(1, ids_mask, True)
            return mask

        # block mask
        num_blocks = int(self.ijepa_cfg.get("num_blocks", 4))
        min_scale = float(self.ijepa_cfg.get("min_block_scale", 0.05))
        max_scale = float(self.ijepa_cfg.get("max_block_scale", 0.2))
        min_aspect = float(self.ijepa_cfg.get("min_aspect", 0.3))
        max_aspect = float(self.ijepa_cfg.get("max_aspect", 3.0))
        max_tries = int(self.ijepa_cfg.get("max_tries", 64))

        out = torch.zeros(batch_size, L, device=device, dtype=torch.bool)
        for i in range(batch_size):
            tries = 0
            while int(out[i].sum().item()) < target and tries < max_tries:
                block_area = int(round(random.uniform(min_scale, max_scale) * L))
                block_area = max(1, min(block_area, L))
                aspect = random.uniform(min_aspect, max_aspect)
                h = int(round(math.sqrt(block_area * aspect)))
                w = int(round(math.sqrt(block_area / max(aspect, 1e-6))))
                h = max(1, min(h, n_h))
                w = max(1, min(w, n_w))
                top = random.randint(0, max(0, n_h - h))
                left = random.randint(0, max(0, n_w - w))
                rows = torch.arange(top, top + h, device=device)
                cols = torch.arange(left, left + w, device=device)
                idx = (rows[:, None] * n_w + cols[None, :]).reshape(-1)
                out[i, idx] = True
                tries += 1

            masked = int(out[i].sum().item())
            if masked > target:
                idx = out[i].nonzero(as_tuple=False).squeeze(1)
                drop = idx[torch.randperm(idx.numel(), device=device)[: masked - target]]
                out[i, drop] = False
            elif masked < target:
                idx = (~out[i]).nonzero(as_tuple=False).squeeze(1)
                add = idx[torch.randperm(idx.numel(), device=device)[: target - masked]]
                out[i, add] = True
        return out

    # ------------------------------------------------------------------ #
    #  Energy-weighted masking (Phase: physics-aware iJEPA)               #
    #  Only patches with non-zero calorimeter energy are maskable.        #
    #  Falls back to random masking if a sample has no non-zero patches.  #
    # ------------------------------------------------------------------ #
    def _sample_mask_energy_weighted(
        self,
        x: torch.Tensor,       # raw image (B, C, H, W)
        n_h: int,
        n_w: int,
        device: torch.device,
    ) -> torch.Tensor:
        B, C, H, W = x.shape
        L = n_h * n_w
        mask_ratio = float(self.ijepa_cfg.get("mask_ratio", 0.5))
        energy_threshold = float(self.ijepa_cfg.get("energy_threshold", 1e-4))

        # Patchify: (B, C, H, W) -> (B, L, C*ph*pw)
        ph = H // n_h
        pw = W // n_w
        patches = x.reshape(B, C, n_h, ph, n_w, pw)
        patches = patches.permute(0, 2, 4, 1, 3, 5).reshape(B, L, -1)  # (B, L, C*ph*pw)
        patch_energy = patches.norm(dim=-1)  # (B, L)

        out = torch.zeros(B, L, device=device, dtype=torch.bool)

        for b in range(B):
            maskable = torch.where(patch_energy[b] > energy_threshold)[0]

            if len(maskable) == 0:
                # Fallback: standard random masking (shouldn't happen on real jets)
                target = max(1, int(round(L * mask_ratio)))
                ids = torch.randperm(L, device=device)[:target]
                out[b, ids] = True
                continue

            num_to_mask = max(1, int(round(len(maskable) * mask_ratio)))
            selected = maskable[torch.randperm(len(maskable), device=device)[:num_to_mask]]
            out[b, selected] = True

        return out

    @torch.no_grad()
    def _update_target_encoder(self):
        m = float(self._target_momentum())
        online = getattr(self.model, "_orig_mod", self.model)
        msd = online.state_dict()
        tsd = self.target_model.state_dict()
        for k, v_t in tsd.items():
            if k not in msd:
                continue
            v_o = msd[k]
            if v_t.shape != v_o.shape:
                continue
            if not torch.is_floating_point(v_t):
                v_t.copy_(v_o)
            else:
                v_t.mul_(m).add_(v_o, alpha=1.0 - m)
        self.log("target_m", m, prog_bar=False, on_step=True, on_epoch=False, sync_dist=True)

    def _loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if self.normalize_targets:
            pred = F.layer_norm(pred, (pred.shape[-1],))
            target = F.layer_norm(target, (target.shape[-1],))
        if self.loss_type == "smooth_l1":
            return F.smooth_l1_loss(pred, target)
        return F.mse_loss(pred, target)

    def _extract_x(self, batch):
        if isinstance(batch, (list, tuple)) and len(batch) >= 1:
            x = batch[0]
        else:
            x = batch
        return x.to(self.device, non_blocking=True)

    def training_step(self, batch, batch_idx):
        x = self._extract_x(batch)
        B, _, H, W = x.shape

        base_online = getattr(self.model, "_orig_mod", self.model)
        n_h = H // base_online.patch_height
        n_w = W // base_online.patch_width
        L = int(n_h * n_w)

        pos_idx = base_online.encoder._make_pos_idx(B, L, device=x.device)

        # --- Masking strategy dispatch ---
        if self.ijepa_cfg.get("masking_strategy") == "energy_weighted":
            mask = self._sample_mask_energy_weighted(x, n_h, n_w, device=x.device)
        else:
            mask = self._sample_mask(B, n_h, n_w, device=x.device)

        ctx_tokens, mask_out, ids_restore = base_online.forward_features(x, mask=mask, pos_idx=pos_idx)
        pred_tokens = self.predictor(ctx_tokens, ids_restore=ids_restore, pos_idx=pos_idx, n_h=n_h, n_w=n_w)

        keep_all = torch.zeros(B, L, device=x.device, dtype=torch.bool)
        with torch.no_grad():
            tgt_tokens, _, _ = self.target_model.forward_features(x, mask=keep_all, pos_idx=pos_idx)
            if self.target_proj is not None:
                tgt_tokens = self.target_proj(tgt_tokens)
            tgt_tokens = tgt_tokens.detach()

        mask_bool = mask_out > 0
        pred_m = pred_tokens[mask_bool]
        tgt_m = tgt_tokens[mask_bool]
        loss = self._loss(pred_m, tgt_m)
        self.log("train_loss", loss, prog_bar=True, on_step=True, on_epoch=True, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x = self._extract_x(batch)
        B, _, H, W = x.shape

        base_online = getattr(self.model, "_orig_mod", self.model)
        n_h = H // base_online.patch_height
        n_w = W // base_online.patch_width
        L = int(n_h * n_w)

        pos_idx = base_online.encoder._make_pos_idx(B, L, device=x.device)

        # --- Masking strategy dispatch ---
        if self.ijepa_cfg.get("masking_strategy") == "energy_weighted":
            mask = self._sample_mask_energy_weighted(x, n_h, n_w, device=x.device)
        else:
            mask = self._sample_mask(B, n_h, n_w, device=x.device)

        ctx_tokens, mask_out, ids_restore = base_online.forward_features(x, mask=mask, pos_idx=pos_idx)
        pred_tokens = self.predictor(ctx_tokens, ids_restore=ids_restore, pos_idx=pos_idx, n_h=n_h, n_w=n_w)

        keep_all = torch.zeros(B, L, device=x.device, dtype=torch.bool)
        with torch.no_grad():
            tgt_tokens, _, _ = self.target_model.forward_features(x, mask=keep_all, pos_idx=pos_idx)
            if self.target_proj is not None:
                tgt_tokens = self.target_proj(tgt_tokens)
            tgt_tokens = tgt_tokens.detach()

        mask_bool = mask_out > 0
        loss = self._loss(pred_tokens[mask_bool], tgt_tokens[mask_bool])
        self.log("val_loss", loss, prog_bar=True, on_epoch=True, sync_dist=True)
        return loss

    def optimizer_step(self, *args, **kwargs):
        out = super().optimizer_step(*args, **kwargs)
        self._update_target_encoder()
        if self.ema is not None:
            base_online = getattr(self.model, "_orig_mod", self.model)
            self.ema.update(base_online)
        return out

    def on_validation_epoch_start(self) -> None:
        if self.ema is None or not self._use_ema_for_val:
            return
        base_online = getattr(self.model, "_orig_mod", self.model)
        self._ema_backup = {k: v.detach().cpu().clone() for k, v in base_online.state_dict().items()}
        self.ema.copy_to(base_online)

    def on_validation_epoch_end(self) -> None:
        if self.ema is None or not self._use_ema_for_val:
            return
        if self._ema_backup is None:
            return
        base_online = getattr(self.model, "_orig_mod", self.model)
        sd = base_online.state_dict()
        for k, v in sd.items():
            v.copy_(self._ema_backup[k].to(v.device))
        self._ema_backup = None

    def configure_optimizers(self):
        ocfg = self.optim_cfg
        scfg = self.sched_cfg

        opt_name = str(ocfg.get("optimizer", "adamw")).lower()
        lr = float(ocfg.get("lr", 5e-4))
        weight_decay = float(ocfg.get("weight_decay", 0.05))
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
        epochs = int(scfg.get("epochs", 0) or getattr(self.trainer, "max_epochs", 30))
        min_lr = float(scfg.get("min_lr", 1e-6))
        warmup_steps = int(scfg.get("warmup_steps", 0))
        if sched_name == "linear":
            scheduler = WarmupLinearDecayLR(optimizer, warmup_steps=warmup_steps, total_steps=total_steps, min_lr=min_lr)
            return {"optimizer": optimizer, "lr_scheduler": {"scheduler": scheduler, "interval": "step", "frequency": 1}}
        if sched_name == "cosine":
            min_lr_frac = min_lr / max(lr, 1e-12)
            def lr_lambda(step):
                if step < warmup_steps:
                    return float(step) / float(max(1, warmup_steps))
                t = (step - warmup_steps) / float(max(1, total_steps - warmup_steps))
                return max(min_lr_frac, 0.5 * (1.0 + math.cos(math.pi * t)))
            scheduler = LambdaLR(optimizer, lr_lambda=lr_lambda)
            return {"optimizer": optimizer, "lr_scheduler": {"scheduler": scheduler, "interval": "step", "frequency": 1}}
        return {"optimizer": optimizer}

class UnifiedDataModule(L.LightningDataModule):
    """Format-agnostic DataModule.

    Reads cfg["data"]["format"] (defaults to "imagenet_wds" for
    back-compat) and dispatches to the registered loader.
    """
    def __init__(self, cfg: dict, phase_cfg: dict):
        super().__init__()
        self.cfg = cfg
        self.phase_type = phase_cfg.get("type", "cls_pretrain")
        self._train_loader = None
        self._val_loader = None

    def setup(self, stage=None):
        cfg_local = dict(self.cfg)
        if self.phase_type == "ijepa_pretrain":
            cfg_local["data"] = dict(cfg_local["data"])
            # WDS-specific knobs only matter for imagenet_wds
            cfg_local["data"]["rand_augment"] = None
            dyn = cfg_local["data"].get("dynamic_img_sizes")
            if dyn:
                cfg_local["data"]["img_size"] = int(max(dyn))
        self._train_loader, self._val_loader = make_loaders_dispatch(cfg_local)

    def train_dataloader(self):
        return self._train_loader

    def val_dataloader(self):
        return self._val_loader

# Back-compat alias: any old code that referenced the old name
# keeps working without modification.
ImageNetWDSDataModule = UnifiedDataModule


def build_trainer(trainer_cfg: Dict[str, Any], max_epochs: int,
                  default_root_dir: Optional[str] = None,
                  callbacks: Optional[list] = None):
    precision = trainer_cfg.get("precision", "16-mixed")
    accumulate_grad_batches = int(trainer_cfg.get("accumulate_grad_batches", 1))
    grad_clip_norm = float(trainer_cfg.get("grad_clip_norm", 0.0))
    devices = trainer_cfg.get("devices", "auto")
    strategy = trainer_cfg.get("strategy", "ddp")
    log_every_n_steps = int(trainer_cfg.get("log_every_n_steps", 50))
    val_check_interval = trainer_cfg.get("val_check_interval", 0.5)
    limit_train_batches = trainer_cfg.get("limit_train_batches", 1.0)
    limit_val_batches = trainer_cfg.get("limit_val_batches", 1.0)
    default_root_dir = default_root_dir or trainer_cfg.get("default_root_dir", "./runs")

    # Strategy alias handling
    try:
        if isinstance(strategy, str):
            if strategy == "ddp":
                strategy_obj = "ddp"
            else:
                strategy_obj = strategy
        else:
            strategy_obj = strategy
    except Exception:
        strategy_obj = "ddp"

    trainer = L.Trainer(
        max_epochs=max_epochs,
        precision=precision,
        accumulate_grad_batches=accumulate_grad_batches,
        gradient_clip_val=grad_clip_norm if grad_clip_norm > 0 else None,
        devices=devices,
        strategy=strategy_obj,
        log_every_n_steps=log_every_n_steps,
        val_check_interval=val_check_interval,
        limit_train_batches=limit_train_batches,
        limit_val_batches=limit_val_batches,
        default_root_dir=default_root_dir,
        callbacks=callbacks,
    )
    return trainer

def _build_callbacks(cfg: Dict[str, Any], phase_name: str, phase_type: str):
    ck = cfg.get("checkpointing", {})
    dir_tmpl = ck.get("dir", "./checkpoints/{phase}")
    dirpath = dir_tmpl.format(phase=phase_name, phase_name=phase_name)
    os.makedirs(dirpath, exist_ok=True)
    # Use val_loss for self-supervised pretrain phases, val_acc1 otherwise
    ssl = str(phase_type).lower() == "ijepa_pretrain"
    monitor = "val_loss" if ssl else ck.get("monitor", "val_acc1")
    mode = "min" if monitor == "val_loss" else ck.get("mode", "max")
    filename = "epoch={epoch:02d}-step={step}-" + monitor + "={" + monitor + ":.4f}"
    ckpt_cb = ModelCheckpoint(
        dirpath=dirpath, filename=filename,
        monitor=monitor, mode=mode,
        save_top_k=int(ck.get("save_top_k", 1)),
        save_last=True,
        every_n_epochs=int(ck.get("every_n_epochs", 1)),
    )
    lrm = LearningRateMonitor(logging_interval="epoch")
    pgr_bar = TQDMProgressBar(refresh_rate=ck.get("log_every_n_steps", 1))
    return [ckpt_cb, lrm, pgr_bar], ckpt_cb, dirpath

def _pick_resume_ckpt(dirpath: str, policy: Optional[str]):
    if not policy:
        return None
    if policy == "last":
        p = os.path.join(dirpath, "last.ckpt")
        return p if os.path.isfile(p) else None
    if policy == "auto":
        p = os.path.join(dirpath, "last.ckpt")
        if os.path.isfile(p):
            return p
        files = sorted(glob.glob(os.path.join(dirpath, "*.ckpt")),
                       key=os.path.getmtime, reverse=True)
        return files[0] if files else None
    # explicit path
    return policy if os.path.isfile(policy) else None

def fit_with_resume_or_warmstart(module, trainer, datamodule, ckpt_path):
    if not ckpt_path:
        return trainer.fit(module, datamodule)

    ckpt = torch.load(ckpt_path, map_location="cpu")
    sd = ckpt["state_dict"]

    # find classifier keys regardless of prefixes
    w_key = next((k for k in sd if k.endswith("token_attn_head.fc.weight")), None)
    b_key = next((k for k in sd if k.endswith("token_attn_head.fc.bias")), None)

    same_shape = (
        w_key is not None
        and sd[w_key].shape == module.model.token_attn_head.fc.weight.shape
        and (b_key is None or sd[b_key].shape == module.model.token_attn_head.fc.bias.shape)
    )

    if same_shape:
        # exact resume: restore optimizer and schedulers too
        return trainer.fit(module, datamodule, ckpt_path=ckpt_path)

    # warm start across phases or class-count change: keep trunk, reset head
    for k in (w_key, b_key):
        if k in sd:
            del sd[k]
    module.load_state_dict(sd, strict=False)  # head reinit comes from model ctor
    return trainer.fit(module, datamodule)    # start fresh optim/sched


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--smoke-test", type=int, default=0,
                        help="If >0, cap training at this many batches and exit (for pipeline validation).")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        cfg = json.load(f)

    seed = int(cfg.get("seed", 42))
    set_seed(seed)

    data_cfg = cfg["data"]
    model_cfg = cfg["model"]
    trainer_cfg = cfg["trainer"]
    save_cfg = cfg.get("save", {})
    phases_cfg = cfg.get("phases", [
        {"name": "ijepa_pretrain", "type": "ijepa_pretrain", "epochs": 10},
        {"name": "cls_finetune", "type": "cls_finetune", "epochs": cfg.get("sched", {}).get("epochs", 30)},
    ])
    resume_cfg = cfg.get("resume", {})
    start_phase = int(resume_cfg.get("start_phase", 0))

    # Run phases sequentially
    last_ckpt_path = None
    for i, phase_cfg in enumerate(phases_cfg):
        if i < start_phase:
            continue
        phase = phase_cfg.get("type", "cls_finetune")
        phase_name = phase_cfg.get("name", phase)
        epochs = int(phase_cfg.get("epochs", 10))

        # Merge per-phase optim/sched if provided, else fall back to global
        optim_cfg = phase_cfg.get("optim") or cfg.get("optim")
        sched_cfg = phase_cfg.get("sched") or cfg.get("sched")

        # Model checkpoint path handling
        model_cfg_local = dict(model_cfg)
        if last_ckpt_path is not None:
            model_cfg_local["checkpoint_path"] = last_ckpt_path

        data_cfg_local = dict(data_cfg)
        data_cfg_local.update(phase_cfg.get("data_override", {}))

        ema_cfg = phase_cfg.get("ema") or cfg.get("ema")
        if str(phase).lower() == "ijepa_pretrain":
            ijepa_cfg = phase_cfg.get("ijepa") or cfg.get("ijepa") or {}
            target_model_cfg = cfg.get("target_model", None)
            module = IJEPA_DepthViTModule(
                model_cfg=model_cfg_local,
                ijepa_cfg=ijepa_cfg,
                optim_cfg=optim_cfg,
                sched_cfg=sched_cfg,
                data_cfg=data_cfg_local,
                target_model_cfg=target_model_cfg,
                ema_cfg=ema_cfg,
                max_epochs=epochs,
            )
        else:
            module = DepthViTModule(
                model_cfg=model_cfg_local,
                optim_cfg=optim_cfg,
                sched_cfg=sched_cfg,
                data_cfg=data_cfg_local,
                phase=phase,
                ema_cfg=ema_cfg,
                max_epochs=epochs,
            )

        cfg_phase = dict(cfg)
        cfg_phase["data"] = data_cfg_local
        dm = ImageNetWDSDataModule(cfg=cfg_phase, phase_cfg=phase_cfg)

        callbacks, ckpt_cb, ckpt_dir = _build_callbacks(cfg, phase_name, phase)
        trainer_cfg_local = dict(trainer_cfg)
        if args.smoke_test > 0:
            trainer_cfg_local["limit_train_batches"] = args.smoke_test
            trainer_cfg_local["limit_val_batches"] = 1
        trainer = build_trainer(
            trainer_cfg_local, max_epochs=epochs,
            default_root_dir=os.path.join(trainer_cfg.get("default_root_dir", "./runs"), phase_name),
            callbacks=callbacks,
        )
        ckpt_path = _pick_resume_ckpt(ckpt_dir, resume_cfg.get("path"))
        fit_with_resume_or_warmstart(module, trainer, dm, ckpt_path=ckpt_path)
        # pick a path that actually exists (rank 0), then broadcast
        if getattr(trainer, "is_global_zero", False):
            chosen = ckpt_cb.best_model_path or ckpt_cb.last_model_path
            if not (chosen and os.path.isfile(chosen)):
                chosen = _pick_resume_ckpt(ckpt_dir, "auto")
        else:
            chosen = None
        if dist.is_available() and dist.is_initialized():
            obj = [chosen]
            dist.broadcast_object_list(obj, src=0)
            chosen = obj[0]
        last_ckpt_path = chosen

if __name__ == "__main__":
    main()
