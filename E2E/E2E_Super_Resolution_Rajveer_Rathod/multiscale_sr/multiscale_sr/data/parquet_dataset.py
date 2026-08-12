"""Streaming parquet dataset with worker-safe sharding and reservoir shuffle.

Design decisions (carried over from the SRGAN baseline, which is battle-tested):
- IterableDataset: streams one row-group at a time, never loads a full file.
- Reservoir shuffle: bounded memory, uniform per-sample distribution within a
  buffer window.
- Yields individual (C,H,W) samples — collate builds the multi-scale LR.

For the multi-scale study the native X_jets_LR is still yielded (key "lr_native")
so scale=64 can optionally use the detector-native LR instead of a downsampled
one; the collate decides. HR is always the ground truth.
"""
from __future__ import annotations

import random
from pathlib import Path
from typing import Callable, Iterator, Sequence

import torch
from torch import Tensor
from torch.utils.data import IterableDataset, get_worker_info

from .normalization import batch_to_tensor


class ParquetJetSRDataset(IterableDataset):
    """Lazy streaming dataset over a list of parquet files.

    Each item yielded is a dict with keys:
        hr        : Tensor (3, H_hr, W_hr)  — raw energy ground truth, float32
        lr_native : Tensor (3, H_lr, W_lr)  — detector-native LR (raw energy)
        pt        : Tensor scalar           — transverse momentum
        m0        : Tensor scalar           — invariant mass
        y         : Tensor scalar (int64)   — class label
    Normalization and multi-scale LR generation happen downstream.

    ``row_filter``, when given, is applied to each row's 0-based index *within
    its own file* (reset per file) and only rows where it returns True are
    yielded. This is how val/test are carved out of the same held-out file(s)
    — see ``normalization.held_out_row_split``. Because the filter is
    per-file, ``shuffle_files`` and worker-sharding are automatically disabled
    whenever a filter is set: file order and worker assignment don't affect
    per-file row indices, but forcing a fixed single-pass order keeps the
    split trivially reproducible and avoids any doubt.
    """

    _PHYS_COLUMNS = ("X_jets_LR", "X_jets", "pt", "m0", "y")

    def __init__(
        self,
        files: Sequence[Path],
        batch_size: int,
        shuffle_files: bool = True,
        reservoir_size: int = 512,
        row_filter: Callable[[int], bool] | None = None,
    ) -> None:
        super().__init__()
        self.files = list(files)
        self.batch_size = batch_size
        self.shuffle_files = shuffle_files
        self.reservoir_size = max(1, reservoir_size)
        self.row_filter = row_filter

    def _shard_files(self) -> list[Path]:
        files = self.files[:]
        if self.row_filter is not None:
            # Per-file row index must stay meaningful and reproducible — skip
            # file shuffling and worker sharding (the held-out group used with
            # row_filter is small, so single-worker streaming is cheap).
            return files
        if self.shuffle_files:
            random.shuffle(files)
        info = get_worker_info()
        if info is not None:
            files = files[info.id :: info.num_workers]
        return files

    def _iter_samples(self, files: list[Path]) -> Iterator[dict[str, Tensor]]:
        for path in files:
            import pyarrow.parquet as pq

            pf = pq.ParquetFile(path)
            row_idx = 0  # reset per file — see class docstring on row_filter
            for batch in pf.iter_batches(
                batch_size=self.batch_size,
                columns=list(self._PHYS_COLUMNS),
                use_threads=True,
            ):
                lr_batch = batch_to_tensor(batch.column(0))  # (B,3,H,W)
                hr_batch = batch_to_tensor(batch.column(1))
                pt_list = batch.column(2).to_pylist()
                m0_list = batch.column(3).to_pylist()
                y_list = batch.column(4).to_pylist()
                for i in range(hr_batch.shape[0]):
                    if self.row_filter is None or self.row_filter(row_idx):
                        yield {
                            "hr": hr_batch[i],
                            "lr_native": lr_batch[i],
                            "pt": torch.tensor(float(pt_list[i]), dtype=torch.float32),
                            "m0": torch.tensor(float(m0_list[i]), dtype=torch.float32),
                            "y": torch.tensor(int(y_list[i]), dtype=torch.int64),
                        }
                    row_idx += 1

    def __iter__(self) -> Iterator[dict[str, Tensor]]:
        files = self._shard_files()
        reservoir: list[dict[str, Tensor]] = []

        for sample in self._iter_samples(files):
            if len(reservoir) < self.reservoir_size:
                reservoir.append(sample)
            else:
                eject_idx = random.randrange(len(reservoir))
                yield reservoir[eject_idx]
                reservoir[eject_idx] = sample

        random.shuffle(reservoir)
        yield from reservoir
