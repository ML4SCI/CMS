"""
data_cifar100.py  --  CIFAR-100 loaders for DepthViT generalization evidence.

Contract matches wds_data.make_loaders: takes the full config dict, returns
(train_loader, val_loader).

Deliberately NOT WebDataset.  CIFAR-100 is 60k 32x32 images (~170 MB); a plain
map-style torchvision dataset on $SCRATCH is faster, exactly reproducible, and
avoids sharding machinery that buys nothing at this size.

Samplers: none are constructed here.  Lightning's `use_distributed_sampler`
(default True) injects a DistributedSampler into a DataLoader returned from a
LightningDataModule.  Adding one here would double-shard the data.

Step accounting under DDP (this must line up or the cosine schedule is wrong):
    per-device batch = global_batch_size // (world_size * accumulate_grad_batches)
    per-rank batches  = 50000 // (world_size * per_device_batch)  [drop_last]
                      = dataset_size // global_batch_size
    which is exactly DepthViTModule._steps_per_epoch().
"""

import os
from typing import Optional

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader
from torchvision import transforms as T
from torchvision.datasets import CIFAR100

                                                                              
CIFAR100_MEAN = (0.5071, 0.4865, 0.4409)
CIFAR100_STD = (0.2673, 0.2564, 0.2762)

CIFAR100_TRAIN_SIZE = 50_000
CIFAR100_VAL_SIZE = 10_000


def _world_size() -> int:
    if dist.is_available() and dist.is_initialized():
        return dist.get_world_size()
    return int(os.environ.get("WORLD_SIZE", "1"))


def _per_device_bs(cfg: dict) -> int:
    d = cfg["data"]
    gbs = d.get("global_batch_size", d.get("batch_size"))
    accum = cfg.get("trainer", {}).get("accumulate_grad_batches", 1)
    ws = _world_size()
    if not gbs or int(gbs) < 1:
        raise ValueError("Set data.global_batch_size in the config")
    return max(1, int(gbs) // max(1, ws * int(accum)))


def _build_transforms(img_size: int, train: bool, ra: Optional[dict]):
    if train:
        ops = [
            T.RandomCrop(img_size, padding=4, padding_mode="reflect"),
            T.RandomHorizontalFlip(0.5),
        ]
        if ra:
            ops.append(T.RandAugment(num_ops=int(ra.get("num_layers", 2)),
                                     magnitude=int(ra.get("magnitude", 9))))
        ops += [T.ToTensor(), T.Normalize(CIFAR100_MEAN, CIFAR100_STD)]
    else:
        ops = []
        if int(img_size) != 32:
            ops.append(T.Resize(int(img_size), interpolation=T.InterpolationMode.BICUBIC))
        ops += [T.ToTensor(), T.Normalize(CIFAR100_MEAN, CIFAR100_STD)]
    return T.Compose(ops)


def make_loaders(cfg: dict):
    d = cfg["data"]
    root = d.get("root") or d.get("train_dir")
    if not root:
        raise ValueError("Set data.root to the directory holding cifar-100-python/")
    if not os.path.isdir(os.path.join(root, "cifar-100-python")):
        raise FileNotFoundError(
            f"{root}/cifar-100-python not found. Run scripts/prepare_cifar100.py "
            f"on a LOGIN node first (compute nodes have no outbound network)."
        )

    img_size = int(d.get("img_size", 32))
    num_workers = int(d.get("num_workers", 4))
    pin = bool(d.get("pin_memory", True))
    persist = bool(d.get("persistent_workers", True)) and num_workers > 0
    per_dev_bs = _per_device_bs(cfg)

    tfm_train = _build_transforms(img_size, True, d.get("rand_augment"))
    tfm_val = _build_transforms(img_size, False, None)

                                                                            
    train_ds = CIFAR100(root=root, train=True, transform=tfm_train, download=False)
    val_ds = CIFAR100(root=root, train=False, transform=tfm_val, download=False)

    train_ld = DataLoader(
        train_ds,
        batch_size=per_dev_bs,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin,
        persistent_workers=persist,
        drop_last=True,                                                                   
    )
    val_ld = DataLoader(
        val_ds,
        batch_size=per_dev_bs,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin,
        persistent_workers=persist,
        drop_last=False,
    )
    return train_ld, val_ld
