import os
from typing import List, Tuple, Dict, Optional

import numpy as np

import torch
from torch import Tensor
from torch.utils.data import Dataset


class QuarkGluonDataset(Dataset):
    """
    PyTorch dataset for the QuarkGluon binary classification / masked pretraining task.

    Expects ``.npz`` files produced by the data-pipeline script, each containing:
    - ``X``: float32 array of shape ``(N, max_num_particles, 4)``
      with features ``[pT, eta, phi, energy]``.
    - ``y``: int64 array of shape ``(N,)`` with labels ``{0: gluon, 1: quark}``.

    The class stores particles in the same layout as
    :class:`JetClassDataset` — ``(N, 4, max_num_particles)`` — so that
    ``__getitem__`` can transpose to ``(max_num_particles, 4)`` and all
    downstream code (model, trainer) works unchanged.

    Parameters
    ----------
    data_path : str
        Path to a single ``.npz`` split file (e.g. ``train.npz``).
    normalize : list[bool]
        Per-feature flag ``[pT, eta, phi, energy]``.
    norm_dict : dict | None
        ``{'pT': (mean, std), 'eta': …, …}`` — same schema as JetClass.
    mask_mode : str | None
        ``'random'``, ``'biased'``, or ``'first'`` for masked pretraining.
        ``None`` → classification mode.
    """

    NUM_CLASSES = 2

    def __init__(
        self,
        data_path: str,
        normalize: List[bool] = [True, False, False, True],
        norm_dict: Optional[Dict[str, Tuple[float, float]]] = None,
        mask_mode: Optional[str] = None,
        # kept for call-site compat with JetClassDataset; ignored
        mask: bool = False,
    ):
        super().__init__()
        raw = np.load(data_path)
        X = raw["X"]  # (N, max_particles, 4)
        y_int = raw["y"]  # (N,)

        # Transpose to (N, 4, max_particles) — same layout as JetClassDataset
        self.X_particles = X.transpose(0, 2, 1).astype(np.float32)

        # One-hot encode: shape (N, 2)
        self.y = np.zeros((len(y_int), self.NUM_CLASSES), dtype=np.float32)
        self.y[np.arange(len(y_int)), y_int] = 1.0

        self.normalize = normalize
        self.norm_dict = norm_dict

        # If mask=True but mask_mode unset, default to 'random'
        if mask and mask_mode is None:
            mask_mode = "random"
        self.mask_mode = mask_mode

    def __len__(self) -> int:
        return len(self.X_particles)

    def __getitem__(self, idx: int) -> Tuple[Tensor, ...]:
        particles = self.X_particles[idx].T  # (max_num_particles, 4)

        if self.mask_mode is not None:
            particles = particles.copy()
            masked_particles, masked_targets, mask_idx = self._mask_particle(
                particles, self.mask_mode
            )

            if self.norm_dict is not None:
                self._apply_norm_inplace(masked_particles)
                self._apply_norm_inplace(masked_targets)

            return (
                torch.tensor(masked_particles, dtype=torch.float32),
                torch.tensor(masked_targets.squeeze(0), dtype=torch.float32),
                torch.tensor(mask_idx, dtype=torch.int64),
            )

        if self.norm_dict is not None:
            particles = particles.copy()
            self._apply_norm_inplace(particles)

        tensor = torch.from_numpy(particles).float()
        label = torch.from_numpy(self.y[idx]).float()
        return tensor, label

    # ------------------------------------------------------------------
    # Normalization — identical logic to JetClassDataset / LazyJetClassDataset
    # ------------------------------------------------------------------
    _FEAT_NAMES = ["pT", "eta", "phi", "energy"]

    def _apply_norm_inplace(self, arr: np.ndarray) -> None:
        for i, feat in enumerate(self._FEAT_NAMES):
            if not self.normalize[i]:
                continue
            mean, std = self.norm_dict[feat]
            if i in (0, 3):  # pT / energy — scale only
                arr[:, i] = arr[:, i] / mean
            else:
                arr[:, i] = (arr[:, i] - mean) / std

    # ------------------------------------------------------------------
    # Masking — reused verbatim from JetClassDataset
    # ------------------------------------------------------------------
    def _mask_particle(
        self, particles: np.ndarray, mode: str = "random"
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        valid_idx = np.where(np.any(particles != 0, axis=1))[0]

        if mode == "random":
            mask_idx = np.array([np.random.choice(valid_idx)])
        elif mode == "biased":
            total = np.sum(1.0 / (np.arange(particles.shape[0]) + 1))
            mask_idx = 127
            u, w = 0.0, 1.0
            while (u < w) or (mask_idx not in valid_idx):
                u = np.random.uniform(0, 1)
                mask_idx = np.random.randint(0, particles.shape[0])
                w = (1 / (mask_idx + 1)) / total
            mask_idx = np.array([mask_idx])
        elif mode == "first":
            mask_idx = valid_idx[:1]
        else:
            mask_idx = np.array([np.random.choice(valid_idx)])

        masked_particles = particles.copy()
        masked_targets = masked_particles[mask_idx, :].copy()
        masked_particles[mask_idx, :] = 0.0

        return masked_particles, masked_targets, mask_idx
