"""DataLoader factory — single entry point for both dataset formats.

get_dataloader() returns (DataLoader, ChannelStats) regardless of whether the
underlying format is parquet or HDF5. Format is auto-detected from the directory
contents unless dataset_type is given explicitly.

Multi-scale: a collate closure downsamples HR to the requested `scale` and
attaches it as "lr". For scale == native_lr_scale on parquet, the detector-native
LR can be used instead (use_native_lr=True). Stats are computed on HR only, the
common reference scale for all models.
"""
from __future__ import annotations

from functools import partial
from pathlib import Path

import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, Subset

from ..utils.env import EnvConfig, prefetch_generator
from .cache import CachedJetSRDataset, ensure_hr_cache
from .hdf5_dataset import build_hdf5_dataset
from .multiscale import downscale_hr, pad_to_size
from .normalization import (
    ChannelStats,
    discover_parquet_files,
    held_out_row_split,
    split_files,
    stream_channel_stats,
    stream_channel_stats_parquet,
)
from .parquet_dataset import ParquetJetSRDataset


def detect_dataset_type(path: Path) -> str:
    """Infer 'parquet' or 'hdf5' from directory contents."""
    if sorted(path.glob("*.parquet")):
        return "parquet"
    if sorted(path.glob("dataset_2_*.hdf5")):
        return "hdf5"
    raise FileNotFoundError(
        f"No *.parquet or dataset_2_*.hdf5 files found in {path}. "
        "Pass --dataset-format explicitly if your files use a different name."
    )


def _multiscale_collate(
    samples: list[dict],
    scale: int,
    use_native_lr: bool,
    hr_size: int = 128,
) -> dict[str, torch.Tensor]:
    """Collate individual samples into a batch and build the scaled LR.

    HR is zero-padded (centered) from its native 125x125 up to hr_size before
    LR is derived, so LR and HR are always consistent with each other. Padding
    (not resizing) keeps every real decoded pixel value untouched — see
    ``pad_to_size``. LR is produced by area-downsampling the padded HR to
    (scale, scale) — the single place multi-scale inputs are created, so
    parquet and HDF5 are identical.
    """
    hr = torch.stack([s["hr"] for s in samples], dim=0)  # (B,3,H,W)
    hr = pad_to_size(hr, hr_size)
    if use_native_lr:
        lr = torch.stack([s["lr_native"] for s in samples], dim=0)
    else:
        lr = downscale_hr(hr, scale)
    return {
        "lr": lr,
        "hr": hr,
        "pt": torch.stack([s["pt"] for s in samples], dim=0),
        "m0": torch.stack([s["m0"] for s in samples], dim=0),
        "y": torch.stack([s["y"] for s in samples], dim=0),
    }


def _parquet_loader(
    path: Path,
    split: str,
    env: EnvConfig,
    batch_size: int,
    val_ratio: float,
    reservoir_size: int,
    collate,
) -> DataLoader:
    files = discover_parquet_files(path)
    train_files, held_out_files = split_files(files, val_ratio)
    row_filter = None
    if split == "train":
        chosen = train_files
    else:
        # "val" and "test" are two row-disjoint halves of the same held-out
        # file(s) — see held_out_row_split. Never a different file set, so
        # there is no way for "test" to leak into training.
        chosen = held_out_files
        row_filter = partial(held_out_row_split, split=split)

    dataset = ParquetJetSRDataset(
        files=chosen,
        batch_size=batch_size,
        shuffle_files=(split == "train"),
        reservoir_size=reservoir_size if split == "train" else 1,
        row_filter=row_filter,
    )
    nw = env.num_workers
    return DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=nw,
        pin_memory=env.pin_memory,
        prefetch_factor=env.prefetch_factor if nw > 0 else None,
        persistent_workers=env.persistent_workers and nw > 0,
        drop_last=(split == "train"),
        collate_fn=collate,
    )


def _cached_parquet_loader(
    path: Path,
    split: str,
    env: EnvConfig,
    batch_size: int,
    val_ratio: float,
    cache_dir: Path,
    collate,
) -> DataLoader:
    """Map-style loader over a one-time HR decode cache (opt-in).

    The cache is built from exactly the same file-level split as the streaming
    path (``split_files``), so there is no train/val(/test) leakage across
    files. "val" and "test" share one decoded cache (the held-out file group,
    decoded once into ``cache_dir/heldout``) and are separated by a row-parity
    filter applied at the dataset level — see ``held_out_row_split``.
    """
    files = discover_parquet_files(path)
    train_files, held_out_files = split_files(files, val_ratio)

    if split == "train":
        split_cache = cache_dir / "train"
        ensure_hr_cache(train_files, split_cache)
        dataset = CachedJetSRDataset(split_cache)
    else:
        if len(held_out_files) != 1:
            raise ValueError(
                f"Cached val/test row-split assumes a single held-out file, got "
                f"{len(held_out_files)}. Adjust val_ratio or add file-level test support."
            )
        split_cache = cache_dir / "heldout"
        ensure_hr_cache(held_out_files, split_cache)
        dataset = CachedJetSRDataset(split_cache, row_filter=partial(held_out_row_split, split=split))

    sampler = RandomSampler(dataset) if split == "train" else SequentialSampler(dataset)
    nw = env.num_workers
    return DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=nw,
        pin_memory=env.pin_memory,
        prefetch_factor=env.prefetch_factor if nw > 0 else None,
        persistent_workers=env.persistent_workers and nw > 0,
        drop_last=(split == "train"),
        collate_fn=collate,
    )


def _hdf5_loader(
    path: Path,
    split: str,
    env: EnvConfig,
    batch_size: int,
    val_ratio: float,
    hr_size: int,
    collate,
) -> DataLoader:
    dataset = build_hdf5_dataset(path, hr_size=hr_size)
    n = len(dataset)  # type: ignore[arg-type]
    n_val = max(1, int(round(n * val_ratio)))
    n_train = n - n_val
    if split == "train":
        indices = list(range(n_train))
        subset = Subset(dataset, indices)
        sampler = RandomSampler(subset)
    else:
        # "val"/"test" split the held-out index range by row parity, same
        # scheme as the parquet path (held_out_row_split) — no HDF5 index
        # is ever visible to training regardless of which half it lands in.
        held_out = range(n_train, n)
        indices = [n_train + i for i in range(len(held_out)) if held_out_row_split(i, split)]
        subset = Subset(dataset, indices)
        sampler = SequentialSampler(subset)

    nw = env.num_workers
    return DataLoader(
        subset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=nw,
        pin_memory=env.pin_memory,
        persistent_workers=env.persistent_workers and nw > 0,
        prefetch_factor=env.prefetch_factor if nw > 0 else None,
        drop_last=(split == "train"),
        collate_fn=collate,
    )


def get_dataloader(
    path: Path,
    split: str,
    env: EnvConfig,
    batch_size: int,
    scale: int,
    hr_size: int = 128,
    dataset_type: str | None = None,
    use_native_lr: bool = False,
    val_ratio: float = 0.33,
    reservoir_size: int = 512,
    stats_batch_size: int = 128,
    max_stats_batches: int | None = None,
    stats_cache_path: Path | None = None,
    skip_stats_cache: bool = False,
    use_cache: bool = False,
    cache_dir: Path | None = None,
) -> tuple[DataLoader, ChannelStats]:
    """Return (dataloader, channel_stats) for the requested split and scale.

    Stats are computed once on HR by streaming the training split, then cached
    to stats_cache_path. Subsequent calls reuse the cache unless
    skip_stats_cache=True. HR statistics are the reference scale for every model.

    ``split`` is one of "train", "val", "test". "val" and "test" are two
    row-disjoint halves of the same held-out file(s) (never the train files):
    "val" is for in-training checkpoint selection, "test" is the one held-out
    set that should be evaluated only once, for final reported numbers.
    """
    if split not in ("train", "val", "test"):
        raise ValueError(f"split must be 'train', 'val', or 'test', got {split!r}")
    path = Path(path)
    if dataset_type is None:
        dataset_type = detect_dataset_type(path)
    if dataset_type not in ("parquet", "hdf5"):
        raise ValueError(f"Unknown dataset_type {dataset_type!r}. Choose 'parquet' or 'hdf5'.")

    # --- normalization stats (cached, HR-only) ---
    stats: ChannelStats | None = None
    if stats_cache_path is not None and stats_cache_path.exists() and not skip_stats_cache:
        stats = ChannelStats.load(stats_cache_path)
        print(f"[data] loaded stats from cache: {stats_cache_path}")
    else:
        print("[data] computing normalization statistics (HR)...")
        if dataset_type == "parquet":
            files = discover_parquet_files(path)
            train_files, _ = split_files(files, val_ratio)
            stats = stream_channel_stats_parquet(
                train_files,
                batch_size=stats_batch_size,
                max_batches=max_stats_batches,
                hr_size=hr_size,
            )
        else:
            tmp_ds = build_hdf5_dataset(path, hr_size=hr_size)
            n = len(tmp_ds)  # type: ignore[arg-type]
            n_train = n - max(1, int(round(n * val_ratio)))
            tmp_subset = Subset(tmp_ds, list(range(n_train)))
            tmp_dl = DataLoader(tmp_subset, batch_size=stats_batch_size, num_workers=0)

            def _hr_iter():
                # Default collate stacks the dataset's "hr" dicts into (B,3,H,W).
                seen = 0
                for batch in tmp_dl:
                    yield batch["hr"]
                    seen += 1
                    if max_stats_batches is not None and seen >= max_stats_batches:
                        break

            stats = stream_channel_stats(_hr_iter())
        print(f"[data] stats: mean={stats.mean.tolist()} std={stats.std.tolist()}")

        if stats_cache_path is not None:
            stats_cache_path.parent.mkdir(parents=True, exist_ok=True)
            stats.save(stats_cache_path)
            print(f"[data] stats saved to {stats_cache_path}")

    collate = partial(_multiscale_collate, scale=scale, use_native_lr=use_native_lr, hr_size=hr_size)

    # --- dataloader ---
    if dataset_type == "parquet":
        if use_cache:
            if use_native_lr:
                raise ValueError(
                    "use_cache is incompatible with use_native_lr: the decode cache "
                    "stores HR only (LR is derived by area-downsampling). Drop one."
                )
            if cache_dir is None:
                cache_dir = path / ".sr_cache"
            loader = _cached_parquet_loader(
                path, split, env, batch_size, val_ratio, Path(cache_dir), collate
            )
        else:
            loader = _parquet_loader(
                path, split, env, batch_size, val_ratio, reservoir_size, collate
            )
            if split == "train":
                loader = _WrappedPrefetchLoader(loader, env)  # type: ignore[assignment]
    else:
        loader = _hdf5_loader(path, split, env, batch_size, val_ratio, hr_size, collate)

    return loader, stats


class _WrappedPrefetchLoader:
    """Thin wrapper that adds one-ahead prefetch to a DataLoader.

    Used for the parquet IterableDataset path where num_workers=0 (macOS/MPS).
    Exposes the same iteration interface as DataLoader.
    """

    def __init__(self, loader: DataLoader, env: EnvConfig) -> None:
        self._loader = loader
        self._env = env

    def __iter__(self):
        return iter(prefetch_generator(self._loader))

    def __len__(self):
        return len(self._loader)  # type: ignore[arg-type]
