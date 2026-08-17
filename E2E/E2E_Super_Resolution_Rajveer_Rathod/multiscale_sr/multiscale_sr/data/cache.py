"""One-time decode cache for the parquet dataset (opt-in, non-baseline).

The streaming ``ParquetJetSRDataset`` re-decodes nested-arrow parquet every
epoch, which is the dominant cost on macOS/MPS (num_workers=0). This module
decodes each split's parquet files **once** into a memory-mapped float32 array
of raw HR energy, plus small per-sample scalars (pt/m0/y). Subsequent epochs
read from mmap in seconds instead of re-decoding arrow.

Only HR is cached: the multi-scale LR is always derived from HR by
area-downsampling in the collate, so a single cache serves every scale
(16/32/64). The detector-native LR (``lr_native``) is *not* cached — the cached
path therefore does not support ``use_native_lr=True`` and the factory guards
against it.

The cache is keyed by a manifest of the source files (name + size + mtime +
row count); if any source file changes the cache is rebuilt automatically.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Callable, Sequence

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import Dataset

from .normalization import batch_to_tensor

_PHYS_COLUMNS = ("X_jets_LR", "X_jets", "pt", "m0", "y")  # HR = column index 1
_DECODE_BATCH = 512


def _manifest(files: Sequence[Path]) -> list[dict]:
    import pyarrow.parquet as pq

    entries = []
    for f in files:
        st = f.stat()
        entries.append(
            {
                "name": f.name,
                "size": st.st_size,
                "mtime": int(st.st_mtime),
                "rows": pq.ParquetFile(f).metadata.num_rows,
            }
        )
    return entries


def _cache_is_valid(cache_dir: Path, files: Sequence[Path]) -> bool:
    meta_path = cache_dir / "meta.json"
    hr_path = cache_dir / "hr.dat"
    scal_path = cache_dir / "scalars.npz"
    if not (meta_path.exists() and hr_path.exists() and scal_path.exists()):
        return False
    try:
        meta = json.loads(meta_path.read_text())
    except Exception:
        return False
    if meta.get("manifest") != _manifest(files):
        return False
    n, c, h, w = meta["n"], meta["c"], meta["h"], meta["w"]
    expected_bytes = n * c * h * w * np.dtype(np.float32).itemsize
    return hr_path.stat().st_size == expected_bytes


def build_hr_cache(files: Sequence[Path], cache_dir: Path) -> dict:
    """Decode ``files`` into ``cache_dir`` (HR memmap + scalars). Returns meta.

    Two passes: (1) sum parquet row counts from metadata (no decode) to size the
    memmap exactly; (2) decode HR + scalars and fill the memmap.
    """
    import pyarrow.parquet as pq

    files = list(files)
    cache_dir.mkdir(parents=True, exist_ok=True)

    # Pass 1 — exact sample count from metadata (cheap, no decode).
    n = sum(pq.ParquetFile(f).metadata.num_rows for f in files)

    # Probe shape from the first row.
    first = next(
        pq.ParquetFile(files[0]).iter_batches(batch_size=1, columns=list(_PHYS_COLUMNS))
    )
    hr0 = batch_to_tensor(first.column(1))  # (1,C,H,W)
    _, c, h, w = hr0.shape

    hr_path = cache_dir / "hr.dat"
    hr_mm = np.memmap(hr_path, dtype=np.float32, mode="w+", shape=(n, c, h, w))
    pt = np.empty(n, dtype=np.float32)
    m0 = np.empty(n, dtype=np.float32)
    y = np.empty(n, dtype=np.int64)

    # Pass 2 — decode and fill.
    off = 0
    for fi, f in enumerate(files, start=1):
        pf = pq.ParquetFile(f)
        for batch in pf.iter_batches(
            batch_size=_DECODE_BATCH, columns=list(_PHYS_COLUMNS), use_threads=True
        ):
            hr = batch_to_tensor(batch.column(1)).numpy()  # (b,C,H,W) float32
            b = hr.shape[0]
            hr_mm[off : off + b] = hr
            pt[off : off + b] = np.asarray(batch.column(2).to_pylist(), dtype=np.float32)
            m0[off : off + b] = np.asarray(batch.column(3).to_pylist(), dtype=np.float32)
            y[off : off + b] = np.asarray(batch.column(4).to_pylist(), dtype=np.int64)
            off += b
        print(f"[cache] decoded {off}/{n} samples ({fi}/{len(files)} files)", flush=True)

    if off != n:
        raise RuntimeError(f"cache build count mismatch: wrote {off}, expected {n}")

    hr_mm.flush()
    del hr_mm
    np.savez(cache_dir / "scalars.npz", pt=pt, m0=m0, y=y)
    meta = {"n": n, "c": c, "h": h, "w": w, "manifest": _manifest(files)}
    (cache_dir / "meta.json").write_text(json.dumps(meta))
    print(f"[cache] built HR cache: {n} samples ({c},{h},{w}) -> {cache_dir}", flush=True)
    return meta


def ensure_hr_cache(files: Sequence[Path], cache_dir: Path) -> dict:
    """Build the cache if missing/stale, else load its meta. Idempotent."""
    if _cache_is_valid(cache_dir, files):
        meta = json.loads((cache_dir / "meta.json").read_text())
        print(f"[cache] reusing HR cache: {meta['n']} samples -> {cache_dir}", flush=True)
        return meta
    return build_hr_cache(files, cache_dir)


class CachedJetSRDataset(Dataset):
    """Map-style dataset reading raw HR from a memmap cache.

    Yields the same dict schema as ``ParquetJetSRDataset`` minus ``lr_native``:
        hr : Tensor (C,H,W) raw energy float32
        pt, m0 : Tensor scalar float32
        y  : Tensor scalar int64
    The multi-scale LR is built downstream by the collate (area-downsample HR),
    so this dataset is scale-agnostic.

    ``row_filter``, when given, is applied once at construction time to the
    cache's row indices (0..n-1) and only the rows where it returns True are
    exposed — this is how a val/test row-parity split (see
    ``normalization.held_out_row_split``) is carved out of a cached held-out
    file group without decoding it twice. The index used here is the *global*
    concatenated offset across all files in the cache (matching the ``off``
    counter in ``build_hr_cache``), which only agrees with the streaming
    dataset's *per-file* row index when the held-out group is a single file
    — true for this dataset (3 parquet files total, one held out) but not
    guaranteed in general, so this is asserted in the factory rather than
    silently assumed.
    """

    def __init__(self, cache_dir: Path, row_filter: Callable[[int], bool] | None = None) -> None:
        self.cache_dir = Path(cache_dir)
        meta = json.loads((self.cache_dir / "meta.json").read_text())
        self.n, self.c, self.h, self.w = meta["n"], meta["c"], meta["h"], meta["w"]
        scal = np.load(self.cache_dir / "scalars.npz")
        self.pt = scal["pt"]
        self.m0 = scal["m0"]
        self.y = scal["y"]
        self._hr = None  # opened lazily so it is safe across worker processes
        if row_filter is None:
            self._indices = np.arange(self.n)
        else:
            self._indices = np.array([i for i in range(self.n) if row_filter(i)], dtype=np.int64)

    def _hr_mm(self) -> np.memmap:
        if self._hr is None:
            self._hr = np.memmap(
                self.cache_dir / "hr.dat",
                dtype=np.float32,
                mode="r",
                shape=(self.n, self.c, self.h, self.w),
            )
        return self._hr

    def __len__(self) -> int:
        return len(self._indices)

    def __getitem__(self, idx: int) -> dict[str, Tensor]:
        i = int(self._indices[idx])
        hr = np.array(self._hr_mm()[i])  # copy out of the read-only mmap
        return {
            "hr": torch.from_numpy(hr),
            "pt": torch.tensor(float(self.pt[i]), dtype=torch.float32),
            "m0": torch.tensor(float(self.m0[i]), dtype=torch.float32),
            "y": torch.tensor(int(self.y[i]), dtype=torch.int64),
        }
