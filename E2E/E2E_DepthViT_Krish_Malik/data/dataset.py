"""
HLS4ML LHC jet dataset adapter.

Format reference (per HDF5 shard):
    jetImageECAL: (N, 100, 100) float64 - electromagnetic calorimeter
    jetImageHCAL: (N, 100, 100) float64 - hadronic calorimeter
    jetImage:     (N, 100, 100) float64 - IDENTICAL to jetImageHCAL
                                          in this dataset version.
                                          We use only ECAL+HCAL.
    jets:         (N, 59) float64       - feature vector incl. labels
                  - cols 53..57: one-hot [j_g, j_q, j_w, j_z, j_t]
                  - col  58:     j_undef (=1 -> drop sample)

5 classes:
    0=g (gluon), 1=q (light quark), 2=W, 3=Z, 4=top

Channel asymmetry:
    ECAL is extremely sparse (~1 in 200K pixels nonzero) but carries
    strong signal when present (electromagnetic showers).
    HCAL fires for ~all jets and carries the dominant hadronic energy.
    These channels are physically distinct measurements with different
    detectors, scales, and noise characteristics — the canonical
    motivating case for DepthViT.

Normalization:
    Both channels are heavy-tailed (max up to ~1200 for HCAL).
    We apply log1p first to compress dynamic range, then per-channel
    mean/std on the log-transformed data.

Stats (computed via scripts/compute_jet_stats.py over the full 30p
training set, log1p applied first):
    ECAL: mean=0.0006, std=0.0282
    HCAL: mean=0.0451, std=0.3471
"""

from __future__ import annotations

import glob
import os
from typing import Any, Dict, Iterator, List, Tuple

import h5py
import numpy as np

from .base import DatasetAdapter, make_loaders_for_adapters, register_format


# Indices in the `jets` feature vector.
LABEL_COL_START = 53     # j_g
LABEL_COL_END = 58       # exclusive; cols 53..57 are the 5 classes
UNDEF_COL = 58           # j_undef

CLASS_NAMES = ("g", "q", "w", "z", "t")
NUM_CLASSES = 5
N_CHANNELS = 2          # ECAL, HCAL
SPATIAL = (100, 100)


# Default per-channel stats (after log1p). These are reasonable starting
# values; recompute via scripts/compute_jet_stats.py for production runs.
# Computed from a 5000-sample subset of 30p train; refine via the script.
DEFAULT_LOG_STATS = {
    "ecal_mean": 0.0006,
    "ecal_std":  0.0282,
    "hcal_mean": 0.0451,
    "hcal_std":  0.3471,
}


class LHCJetAdapter(DatasetAdapter):
    """One split (train or val) of the HLS4ML LHC jet dataset.

    Args:
        shard_glob: glob pattern matching .h5 shards for this split.
                    e.g. "/.../lhc_jets/30p/train/*.h5"
        stats:      optional dict overriding per-channel mean/std
                    (keys: ecal_mean, ecal_std, hcal_mean, hcal_std).
        samples_per_shard: hint used for steps_per_epoch (default 10000,
                    which is the actual count in the HLS4ML shards).
    """

    n_channels = N_CHANNELS
    spatial_hw = SPATIAL
    num_classes = NUM_CLASSES
    log_transform = True

    def __init__(
        self,
        shard_glob: str,
        stats: Dict[str, float] | None = None,
        samples_per_shard: int = 10000,
    ):
        s = dict(DEFAULT_LOG_STATS)
        if stats:
            s.update(stats)
        self.mean = [float(s["ecal_mean"]), float(s["hcal_mean"])]
        self.std = [float(s["ecal_std"]), float(s["hcal_std"])]

        paths = sorted(glob.glob(shard_glob))
        if not paths:
            raise FileNotFoundError(f"No shards matched: {shard_glob}")
        self.shard_paths = paths
        self.samples_per_shard = int(samples_per_shard)

    def iter_shard(self, shard_path: str) -> Iterator[Tuple[np.ndarray, int]]:
        with h5py.File(shard_path, "r") as hf:
            if "jetImageECAL" not in hf or "jetImageHCAL" not in hf or "jets" not in hf:
                import logging
                logging.warning(f"[LHCJets] Skipping bad shard (missing keys): {shard_path} -- found: {list(hf.keys())}")
                return
            ecal = hf["jetImageECAL"]
            hcal = hf["jetImageHCAL"]
            jets = hf["jets"]

            n = ecal.shape[0]
            if not (hcal.shape[0] == n and jets.shape[0] == n):
                raise ValueError(f"shape mismatch in {shard_path}")

            # Read in row chunks. h5py random access is slow; bulk
            # reads are much faster. 10000 rows of 100x100 float64 is
            # 800 MB — too big to slurp in one go on every worker.
            # Chunk through 1000 at a time (~80 MB/worker).
            CHUNK = 1000
            for start in range(0, n, CHUNK):
                end = min(start + CHUNK, n)
                ecal_chunk = ecal[start:end].astype(np.float32, copy=False)
                hcal_chunk = hcal[start:end].astype(np.float32, copy=False)
                jets_chunk = jets[start:end]

                onehot = jets_chunk[:, LABEL_COL_START:LABEL_COL_END]
                undef = jets_chunk[:, UNDEF_COL]
                labels = onehot.argmax(axis=1)
                # only keep clean single-class samples
                clean_mask = (
                    (undef == 0)
                    & (onehot.sum(axis=1) == 1)
                    & (onehot.max(axis=1) == 1)
                )

                for i in range(end - start):
                    if not clean_mask[i]:
                        continue
                    img = np.stack([ecal_chunk[i], hcal_chunk[i]], axis=0)
                    yield img, int(labels[i])


# ----------------------------------------------------------------------
# Format registration: cfg["data"]["format"] = "lhc_jets"
# ----------------------------------------------------------------------

@register_format("lhc_jets")
def make_loaders(cfg: Dict[str, Any]):
    d = cfg["data"]
    train_glob = d.get("train_glob") or os.path.join(d["train_dir"], "*.h5")
    val_glob = d.get("val_glob") or os.path.join(d["val_dir"], "*.h5")

    stats = d.get("channel_stats")  # optional override

    train_adapter = LHCJetAdapter(train_glob, stats=stats)
    val_adapter = LHCJetAdapter(val_glob, stats=stats)

    return make_loaders_for_adapters(cfg, train_adapter, val_adapter)
