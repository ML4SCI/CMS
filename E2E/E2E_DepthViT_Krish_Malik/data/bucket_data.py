"""
Bucketed variable-resolution data pipeline for the Imagewoof resolution study.

Two arms share this file:
  - DepthViT arm ("bucket"): each image is routed to the nearest-aspect bucket
    in SUITE and resized to it WITHOUT cropping. Native content is preserved;
    only a mild isotropic-ish resize (mean ~0.906x) is applied.
  - ViT arm ("fixed"): standard short-side-resize -> center-crop to a single
    square (default 384). This is the baseline that pays the crop/upscale tax.

The DepthViT arm needs a sampler that guarantees every rank sees the SAME
bucket in the same step (variable token counts can't be mixed within a DDP
all-reduce step without padding), and that every global batch is single-bucket.

Imagewoof full_size layout (ImageFolder):
    <root>/train/<wnid>/<img>.JPEG
    <root>/val/<wnid>/<img>.JPEG
"""
import os, math, random
from typing import List, Tuple, Dict
from PIL import Image

import torch
from torch.utils.data import Dataset, Sampler
import torchvision.transforms as T
import torchvision.transforms.functional as TF
from torchvision.transforms import InterpolationMode

                                                                               
                                                                                   
SUITE: List[Tuple[int, int]] = [(480, 320), (448, 336), (384, 384), (336, 448), (320, 480)]

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def _route(w: int, h: int, suite=SUITE) -> Tuple[int, int]:
    """Nearest bucket by |log aspect ratio| — pure aspect match, no area term."""
    ar = math.log(w / h)
    return min(suite, key=lambda b: abs(ar - math.log(b[0] / b[1])))


class _ListImageFolder(Dataset):
    """Minimal ImageFolder that also exposes each sample's assigned bucket."""
    def __init__(self, root: str, split: str):
        self.dir = os.path.join(root, split)
        classes = sorted(d.name for d in os.scandir(self.dir) if d.is_dir())
        self.class_to_idx = {c: i for i, c in enumerate(classes)}
        self.samples: List[Tuple[str, int]] = []
        exts = (".jpeg", ".jpg", ".png")
        for c in classes:
            cd = os.path.join(self.dir, c)
            for fn in os.listdir(cd):
                if fn.lower().endswith(exts):
                    self.samples.append((os.path.join(cd, fn), self.class_to_idx[c]))
        self.samples.sort()

    def __len__(self):
        return len(self.samples)


class BucketDataset(_ListImageFolder):
    """DepthViT arm: resize-to-bucket, no crop. Precomputes each image's bucket."""
    def __init__(self, root: str, split: str, train: bool, suite=SUITE):
        super().__init__(root, split)
        self.train = train
        self.suite = suite
                                                                                           
        self.bucket_of: List[int] = []
        b2i = {b: i for i, b in enumerate(suite)}
        for path, _ in self.samples:
            with Image.open(path) as im:
                w, h = im.size
            self.bucket_of.append(b2i[_route(w, h, suite)])
        self.norm = T.Normalize(IMAGENET_MEAN, IMAGENET_STD)

    def bucket_indices(self) -> Dict[int, List[int]]:
        out: Dict[int, List[int]] = {i: [] for i in range(len(self.suite))}
        for idx, b in enumerate(self.bucket_of):
            out[b].append(idx)
        return out

    def __getitem__(self, idx: int):
        path, label = self.samples[idx]
        bw, bh = self.suite[self.bucket_of[idx]]
        img = Image.open(path).convert("RGB")
                                                                      
        img = TF.resize(img, [bh, bw], interpolation=InterpolationMode.BICUBIC, antialias=True)
        if self.train:
            if random.random() < 0.5:
                img = TF.hflip(img)
        x = TF.to_tensor(img)
        x = self.norm(x)
        return x, label


class FixedSquareDataset(_ListImageFolder):
    """ViT arm: short-side resize -> center crop to a single square size."""
    def __init__(self, root: str, split: str, train: bool, size: int = 384):
        super().__init__(root, split)
        self.size = size
        if train:
            self.tf = T.Compose([
                T.RandomResizedCrop(size, scale=(0.08, 1.0),
                                    interpolation=InterpolationMode.BICUBIC, antialias=True),
                T.RandomHorizontalFlip(),
                T.ToTensor(),
                T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
            ])
        else:
            self.tf = T.Compose([
                T.Resize(int(round(size * 256 / 224)),
                         interpolation=InterpolationMode.BICUBIC, antialias=True),
                T.CenterCrop(size),
                T.ToTensor(),
                T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
            ])

    def __getitem__(self, idx: int):
        path, label = self.samples[idx]
        img = Image.open(path).convert("RGB")
        return self.tf(img), label


class DistributedBucketBatchSampler(Sampler[List[int]]):
    """Yields per-rank index lists, one bucket per global batch.

    Guarantees:
      * every global batch contains a SINGLE bucket (uniform token count);
      * all `world_size` ranks receive the same bucket at the same step
        (so no rank stalls on a shape mismatch during all-reduce);
      * each rank gets global_batch_size // world_size indices per step.
    Drops the ragged tail of each bucket so all ranks stay in lockstep.
    """
    def __init__(self, dataset: BucketDataset, global_batch_size: int,
                 world_size: int, rank: int, shuffle: bool = True, seed: int = 42):
        self.ds = dataset
        self.gbs = global_batch_size
        assert global_batch_size % world_size == 0, "gbs must divide by world_size"
        self.per_rank = global_batch_size // world_size
        self.world_size = world_size
        self.rank = rank
        self.shuffle = shuffle
        self.seed = seed
        self.epoch = 0
        self._buckets = dataset.bucket_indices()
                                                     
        self._num_batches = sum(len(v) // self.gbs for v in self._buckets.values())

    def set_epoch(self, epoch: int):
        self.epoch = epoch

    def __len__(self):
        return self._num_batches

    def __iter__(self):
        g = random.Random(self.seed + self.epoch)
        batches: List[List[int]] = []
        for b, idxs in self._buckets.items():
            idxs = list(idxs)
            if self.shuffle:
                g.shuffle(idxs)
            nb = len(idxs) // self.gbs
            for i in range(nb):
                batches.append(idxs[i * self.gbs:(i + 1) * self.gbs])
        if self.shuffle:
            g.shuffle(batches)                                         
        for gb in batches:
                                                   
            yield gb[self.rank * self.per_rank:(self.rank + 1) * self.per_rank]


def build_bucket_loader(root, split, train, global_batch_size, world_size, rank,
                        num_workers=8, seed=42, suite=SUITE):
    ds = BucketDataset(root, split, train=train, suite=suite)
    sampler = DistributedBucketBatchSampler(ds, global_batch_size, world_size, rank,
                                            shuffle=train, seed=seed)
    loader = torch.utils.data.DataLoader(
        ds, batch_sampler=sampler, num_workers=num_workers,
        pin_memory=True, persistent_workers=num_workers > 0,
    )
    return ds, sampler, loader


def build_fixed_loader(root, split, train, global_batch_size, world_size, rank,
                       size=384, num_workers=8, seed=42):
                                                                               
                                                                                
                                                                                
    ds = FixedSquareDataset(root, split, train=train, size=size)
    per_rank = global_batch_size // world_size
    loader = torch.utils.data.DataLoader(
        ds, batch_size=per_rank, shuffle=train, num_workers=num_workers,
        pin_memory=True, persistent_workers=num_workers > 0, drop_last=train)
    return ds, None, loader


if __name__ == "__main__":
                                           
    import sys
    root = sys.argv[1] if len(sys.argv) > 1 else "/pscratch/sd/k/krish_m/imagewoof/imagewoof2"
    ds = BucketDataset(root, "train", train=False)
    occ = {SUITE[b]: len(v) for b, v in ds.bucket_indices().items()}
    tot = sum(occ.values())
    print(f"train N={tot}")
    for b, n in sorted(occ.items(), key=lambda x: -x[1]):
        print(f"  {b[0]}x{b[1]}  {(b[0]//16)*(b[1]//16):>3} tok   {n:>5} ({n/tot*100:.1f}%)")
    x, y = ds[0]
    print("sample tensor", tuple(x.shape), "label", y)
