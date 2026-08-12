"""Lazy map-style HDF5 dataset for CaloChallenge Dataset 2.

CaloChallenge Dataset 2 geometry (calochallenge.github.io):
    showers : (N, 6480) = (N, 45_r * 144_phi)  — single calorimeter layer
    incident_energies : (N, 1)                  — GeV, used as physics proxy

Since the parquet dataset has 3 channels (ECAL, HCAL, Tracks), we replicate the
single calorimeter layer to 3 channels so the model interface is unchanged.

For the multi-scale study, HR is optionally resized to a square hr_size so that
both datasets share one HR resolution (default 125 to match the parquet HR).
The LR at each scale is generated downstream in the collate, identical to the
parquet path — this dataset only produces HR + physics scalars.

Worker safety: HDF5 handles cannot be shared across forked processes, so we
store the path and re-open in __getitem__ if the PID changed.
"""
from __future__ import annotations

import os
from pathlib import Path

import h5py
import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import ConcatDataset, Dataset

_CALO_R = 45
_CALO_PHI = 144
_N_CHANNELS = 3  # replicate single layer to match parquet's (ECAL, HCAL, Tracks)


class HDF5JetDataset(Dataset):
    """Map-style dataset over one HDF5 file (CaloChallenge Dataset 2 format).

    Each item is a dict with keys matching ParquetJetSRDataset:
        hr        : Tensor (3, hr_size, hr_size)  — raw energy ground truth
        lr_native : Tensor (3, hr_size, hr_size)  — copy of HR (no native LR here)
        pt        : Tensor scalar  — incident_energy proxy (zero if absent)
        m0        : Tensor scalar  — zeros (not present in this dataset)
        y         : Tensor scalar  — zeros (no class label in this dataset)

    lr_native mirrors HR so the collate's optional "native LR for scale=64" path
    stays valid; for HDF5 there is no detector LR, so it falls back to HR.
    """

    def __init__(
        self,
        path: Path,
        hr_size: int = 128,
    ) -> None:
        super().__init__()
        self.path = Path(path)
        self.hr_size = hr_size

        with h5py.File(self.path, "r", swmr=True) as f:
            self._length = int(f["showers"].shape[0])
            self._has_energies = "incident_energies" in f

        self._file: h5py.File | None = None
        self._pid: int = -1

    def _get_file(self) -> h5py.File:
        pid = os.getpid()
        if self._file is None or self._pid != pid:
            if self._file is not None:
                try:
                    self._file.close()
                except Exception:
                    pass
            self._file = h5py.File(self.path, "r", swmr=True)
            self._pid = pid
        return self._file

    def __len__(self) -> int:
        return self._length

    def __getitem__(self, idx: int) -> dict[str, Tensor]:
        f = self._get_file()

        shower = f["showers"][idx]  # (6480,) float64
        hr_np = shower.reshape(_CALO_R, _CALO_PHI).astype(np.float32)
        hr = torch.from_numpy(hr_np).unsqueeze(0).expand(_N_CHANNELS, -1, -1).clone()
        # (3, 45, 144)

        # Resize HR to a square target so both datasets share one HR resolution.
        # area mode preserves the per-cell mean and avoids bicubic overshoot.
        hr = F.interpolate(
            hr.unsqueeze(0).float(),
            size=(self.hr_size, self.hr_size),
            mode="area",
        ).squeeze(0).clamp_min(0.0)

        if self._has_energies:
            pt_val = float(f["incident_energies"][idx].flat[0])
        else:
            pt_val = 0.0

        return {
            "hr": hr.float(),
            "lr_native": hr.float(),  # no native LR for CaloChallenge; fall back to HR
            "pt": torch.tensor(pt_val, dtype=torch.float32),
            "m0": torch.zeros((), dtype=torch.float32),
            "y": torch.zeros((), dtype=torch.int64),
        }

    def __del__(self) -> None:
        if self._file is not None:
            try:
                self._file.close()
            except Exception:
                pass


def build_hdf5_dataset(
    data_dir: Path,
    hr_size: int = 128,
) -> ConcatDataset:
    """Load all dataset_2_*.hdf5 files in data_dir as a single ConcatDataset."""
    files = sorted(data_dir.glob("dataset_2_*.hdf5"))
    if not files:
        raise FileNotFoundError(f"No dataset_2_*.hdf5 files found in {data_dir}")
    datasets = [HDF5JetDataset(f, hr_size=hr_size) for f in files]
    return ConcatDataset(datasets)
