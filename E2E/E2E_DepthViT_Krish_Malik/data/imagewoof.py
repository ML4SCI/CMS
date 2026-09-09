"""
Imagewoof resolution-study data formats, registered into the FORMAT_REGISTRY.

Two formats, both backed by bucket_data.py (repo root):

  "imagewoof_bucket"  DepthViT arm. Each image resized to its nearest-aspect
                      bucket in cfg["data"]["suite"] WITHOUT cropping. Uses a
                      DistributedBucketBatchSampler so every global batch is a
                      single bucket (uniform token count) and all ranks see the
                      same bucket per step. Native area retained ~100%.

  "imagewoof_fixed"   ViT arm. Standard short-side-resize -> center-crop to one
                      square (cfg["data"]["img_size"], default 384). The baseline
                      that discards ~44% of each image and upscales the survivor.

Both return (train_loader, val_loader), matching the interface the trainer's
UnifiedDataModule expects — same contract as imagenet_wds.make_loaders.

DDP note: world_size/rank are read from the torch.distributed env at call time,
inside make_loaders, so this stays correct under torchrun --nproc_per_node=4.
"""
import os
from typing import Any, Dict, Tuple

import torch
import torch.distributed as dist

from .base import register_format


def _world_and_rank() -> Tuple[int, int]:
    if dist.is_available() and dist.is_initialized():
        return dist.get_world_size(), dist.get_rank()
    return int(os.environ.get("WORLD_SIZE", "1")), int(os.environ.get("RANK", "0"))


def _split_name(cfg: Dict[str, Any], want: str) -> str:
                                                                  
    key = "train_split" if want == "train" else "val_split"
    default = "train" if want == "train" else "val"
    return str(cfg["data"].get(key, default))


@register_format("imagewoof_bucket")
def make_loaders_bucket(cfg: Dict[str, Any]):
    from bucket_data import build_bucket_loader, SUITE
    d = cfg["data"]
    root = d["root"]
    gbs = int(d["global_batch_size"])
    nw = int(d.get("num_workers", 8))
    seed = int(cfg.get("seed", 42))
    suite = [tuple(b) for b in d.get("suite", SUITE)]
    ws, rk = _world_and_rank()

    _, _, train_loader = build_bucket_loader(
        root, _split_name(cfg, "train"), train=True,
        global_batch_size=gbs, world_size=ws, rank=rk,
        num_workers=nw, seed=seed, suite=suite)
    _, _, val_loader = build_bucket_loader(
        root, _split_name(cfg, "val"), train=False,
        global_batch_size=gbs, world_size=ws, rank=rk,
        num_workers=nw, seed=seed, suite=suite)
    return train_loader, val_loader


@register_format("imagewoof_fixed")
def make_loaders_fixed(cfg: Dict[str, Any]):
    from bucket_data import build_fixed_loader
    d = cfg["data"]
    root = d["root"]
    gbs = int(d["global_batch_size"])
    nw = int(d.get("num_workers", 8))
    size = int(d.get("img_size", 384))
    seed = int(cfg.get("seed", 42))
    ws, rk = _world_and_rank()

    _, _, train_loader = build_fixed_loader(
        root, _split_name(cfg, "train"), train=True,
        global_batch_size=gbs, world_size=ws, rank=rk,
        size=size, num_workers=nw, seed=seed)
    _, _, val_loader = build_fixed_loader(
        root, _split_name(cfg, "val"), train=False,
        global_batch_size=gbs, world_size=ws, rank=rk,
        size=size, num_workers=nw, seed=seed)
    return train_loader, val_loader
