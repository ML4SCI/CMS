"""Distributed JetClass data loading for the ablation runs (ragged CSR).

Builds ``DataLoader``\\s via the shard-coherent ragged loader, so each rank
gets a disjoint partition of ``.pt`` shards per epoch.

The ``.pt`` shards are **never built here**.  Conversion is a single-writer
operation; use ``python -m preprocessing.convert_jetclass_ragged_pt`` once
before submitting.

Normalization is applied **on-the-fly** in the collate (never to the shards
on disk): when ``config.norm_stats_path`` points at a ``norm_stats.json``,
:class:`NormStats` z-scores the continuous particle features and
scale-normalizes pT/E, while the four-vector block ``v`` stays raw — the
pair-feature MLP is defined on raw four-momenta.
"""

from __future__ import annotations

import json
from functools import partial
from pathlib import Path
from typing import Optional, Tuple, Union

import numpy as np
import torch
from torch.utils.data import DataLoader

from dataloader.ca_augment import cambridge_aachen_augment_batch
from dataloader.ragged_loader import (
    ClassBalancedBatchSampler,
    ShardCoherentBatchSampler,
    coherent_collate,
    create_ragged_train_loader,
    create_ragged_val_loader,
)

__all__ = ["build_loaders", "InfiniteLoader", "NormStats"]


class NormStats:
    """Per-feature normalization constants from ``norm_stats.json``.

    The stats file carries per-feature mean and std over all 16 particle
    features.  Not every channel is z-scored: the two positive energy scales
    are divided by their mean only (so the normalized value cannot go
    negative), charge and the one-hot PID channels pass through untouched:

    =====  ==============================================================
    kind   ``x`` channels (matches ``ALL_PARTICLE_FEATURES`` ordering)
    =====  ==============================================================
    scale  0 (``part_pt``), 3 (``part_energy``)          ``x / mean``
    zscore 1, 2, 4, 5, 6, 7, 8, 9 (eta, phi, deta, dphi, ``d0*``, ``dz*``)
                                                        ``(x - mean) / std``
    raw    10-15 (charge, one-hot PID x5)                untouched
    =====  ==============================================================

    The four-vector block ``v`` is never touched.
    """

    #: ``x`` channels that are positive energy scales: divide by mean only,
    #: so the normalized feature stays positive.
    SCALE_ONLY_INDICES = (0, 3)  # part_pt, part_energy
    #: ``x`` channels that are continuous: z-score with (x - mean) / std.
    ZSCORE_INDICES = (1, 2, 4, 5, 6, 7, 8, 9)
    #: Charge (10) and the one-hot PID channels (11-15) are left raw.
    NUM_FEATURES = 16

    def __init__(self, mean, std):
        mean = np.asarray(mean, dtype=np.float64)
        std = np.asarray(std, dtype=np.float64)
        if mean.shape != (self.NUM_FEATURES,) or std.shape != (self.NUM_FEATURES,):
            raise ValueError(
                f"norm stats must have {self.NUM_FEATURES} entries (one per "
                f"particle feature); got mean={mean.shape}, std={std.shape}"
            )
        # Precompute x' = x * scale + offset per channel.
        scale = np.ones(self.NUM_FEATURES, dtype=np.float32)
        offset = np.zeros(self.NUM_FEATURES, dtype=np.float32)
        for i in self.SCALE_ONLY_INDICES:
            if mean[i] > 0:
                scale[i] = 1.0 / mean[i]
        for i in self.ZSCORE_INDICES:
            inv_std = 1.0 / std[i] if std[i] > 0 else 1.0
            scale[i] = inv_std
            offset[i] = -mean[i] * inv_std
        self._scale = scale
        self._offset = offset

    def lloca_kinematic_affine(self) -> tuple:
        """Scale/offset for LLoCa's six local-frame kinematic channels.

        LLoCa replaces lab-frame ``x[:, 0:6]`` with
        ``(log pT, eta, phi, log E, deta, dphi)`` in each particle's frame.
        Applying the lab-frame affine unchanged would divide *log* pT by
        ``mean(pT)``.  Eta/phi/deta/dphi keep the usual z-score; the two
        energy channels become ``log(value / mean)``.
        """
        scale = self._scale[:6].copy()
        offset = self._offset[:6].copy()
        for i in self.SCALE_ONLY_INDICES:
            if scale[i] > 0:
                offset[i] = float(np.log(scale[i]))
                scale[i] = 1.0
        return scale, offset

    @classmethod
    def load(cls, path: Path) -> "NormStats":
        """Load a ``norm_stats.json`` produced by ``compute_norm_stats``."""
        payload = json.loads(Path(path).read_text())
        return cls(payload["mean"], payload["std"])

    def apply(
        self, x: torch.Tensor, mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Normalize the feature block of one batch.

        Parameters
        ----------
        x : Tensor
            ``(B, NUM_FEATURES, P)`` float32, channel-first as the loader
            emits it.
        mask : Tensor, optional
            ``(B, 1, P)``, nonzero on real particles.  **Pass it.**  ``offset``
            is nonzero for every z-scored channel, so the affine alone turns
            padded slots from exact zeros into ``offset`` -- breaking the batch
            contract asserted in ``variants/tests/strategies.py`` ("padded
            slots carry exact zeros") and, worse, feeding those constants to
            weaver's ``Embed.input_bn``, a ``BatchNorm1d`` that reduces over
            ``(B, C, P)`` including pad columns.  A batch whose ``P_max`` is
            set by one long jet is mostly padding, so the real features get
            squashed relative to a batch of uniformly short jets: the same jet
            would embed differently depending on its batch-mates, and those
            polluted running stats carry into eval.

        Returns
        -------
        Tensor
            Normalized ``x``, with padded slots restored to exact zeros when
            *mask* is given; ``v`` and ``mask`` are the caller's business.
        """
        if x.dim() != 3 or x.shape[1] != self.NUM_FEATURES:
            raise ValueError(
                f"expected x (B, {self.NUM_FEATURES}, P), got {tuple(x.shape)}"
            )
        scale = torch.as_tensor(self._scale, device=x.device, dtype=x.dtype)
        offset = torch.as_tensor(self._offset, device=x.device, dtype=x.dtype)
        out = x * scale[None, :, None] + offset[None, :, None]
        if mask is not None:
            if mask.dim() != 3 or mask.shape[0] != x.shape[0] or mask.shape[-1] != x.shape[-1]:
                raise ValueError(
                    f"expected mask (B, 1, P) matching x {tuple(x.shape)}, "
                    f"got {tuple(mask.shape)}"
                )
            out = out * mask.to(dtype=out.dtype)
        return out


def _normalizing_collate(batch, stats: NormStats):
    """``coherent_collate`` followed by on-the-fly feature normalization.

    Module-level (not a closure) so the ``functools.partial`` below pickles
    into ``num_workers > 0`` DataLoader worker processes.
    """
    x, v, mask, y = coherent_collate(batch)
    return stats.apply(x, mask), v, mask, y


def _train_collate(
    batch,
    stats: Optional[NormStats] = None,
    ca_prob: float = 0.0,
    ca_rmax: float = 0.2,
    ca_min_particles: int = 2,
):
    """Train collate: C/A coarsening (optional) then feature normalization.

    Augmentation runs on raw features so merged kinematics are z-scored with
    the same ``NormStats`` as un-augmented particles.  Validation uses
    :func:`_normalizing_collate` / :func:`coherent_collate` and never coarsens.
    """
    x, v, mask, y = coherent_collate(batch)
    if ca_prob > 0.0:
        x, v, mask = cambridge_aachen_augment_batch(
            x,
            v,
            mask,
            prob=ca_prob,
            rmax=ca_rmax,
            min_particles=ca_min_particles,
        )
    if stats is not None:
        # After augmentation, so the mask reflects the merged particle counts.
        x = stats.apply(x, mask)
    return x, v, mask, y


TrainBatchSampler = Union[ClassBalancedBatchSampler, ShardCoherentBatchSampler]


def build_loaders(config, ctx) -> Tuple[DataLoader, DataLoader, TrainBatchSampler]:
    """Build the training and validation loaders for one run.

    Returns
    -------
    tuple
        ``(train_loader, val_loader, train_batch_sampler)``. The batch sampler
        is returned so the caller can call ``set_epoch`` — without it every
        epoch reuses the same shuffle and each rank keeps seeing the same
        shard order.
    """
    if not config.train_pt_dir:
        raise FileNotFoundError(
            "train_pt_dir is not set in the config. Point it at the directory "
            "holding ragged CSR .pt shards, e.g.:\n"
            "    --set train_pt_dir=$PSCRATCH/jetclass/pt_ragged/train_100M"
        )
    if not config.val_pt_dir:
        raise FileNotFoundError(
            "val_pt_dir is not set in the config. Point it at the directory "
            "holding ragged CSR .pt shards, e.g.:\n"
            "    --set val_pt_dir=$PSCRATCH/jetclass/pt_ragged/val_5M"
        )

    # Optional on-the-fly normalization, applied in the collate so both
    # loaders (and both train and eval paths) see normalized features and the
    # raw shards on disk are never touched.  Explicitly default to
    # coherent_collate — collate_fn=None would fall back to default_collate,
    # which cannot unwrap the sampler's single-element batches.
    stats = None
    if config.norm_stats_path:
        stats = NormStats.load(config.norm_stats_path)
        if ctx.is_main:
            print(
                f"applying on-the-fly feature normalization "
                f"from {config.norm_stats_path}",
                flush=True,
            )

    val_collate = coherent_collate
    if stats is not None:
        val_collate = partial(_normalizing_collate, stats=stats)

    train_collate = val_collate
    if config.ca_augment:
        train_collate = partial(
            _train_collate,
            stats=stats,
            ca_prob=config.ca_augment_prob,
            ca_rmax=config.ca_rmax,
            ca_min_particles=config.ca_min_particles,
        )
        if ctx.is_main:
            print(
                f"train-only C/A coarsening: prob={config.ca_augment_prob} "
                f"rmax={config.ca_rmax} min_particles={config.ca_min_particles}",
                flush=True,
            )

    train_loader = create_ragged_train_loader(
        pt_dir=config.train_pt_dir,
        batch_size=config.batch_size,
        rank=ctx.rank,
        world_size=ctx.world_size,
        num_workers=config.num_workers,
        prefetch_factor=config.prefetch_factor,
        pin_memory=ctx.device.type == "cuda",
        collate_fn=train_collate,
        class_balanced=config.class_balanced_train,
    )

    val_loader = create_ragged_val_loader(
        pt_dir=config.val_pt_dir,
        batch_size=config.batch_size,
        rank=ctx.rank,
        world_size=ctx.world_size,
        num_workers=config.num_workers,
        prefetch_factor=config.prefetch_factor,
        pin_memory=ctx.device.type == "cuda",
        # Global cap on the pooled evaluation payload: the shard-coherent
        # sampler splits it evenly across ranks.  Without it, val_5M means
        # ~1.25M jets per rank at 4 GPUs, and the metric all-gather pools
        # everything on one rank.
        max_jets=config.val_max_jets,
        collate_fn=val_collate,
    )

    train_batch_sampler = train_loader.batch_sampler
    return train_loader, val_loader, train_batch_sampler


class InfiniteLoader:
    """Iterate a ``DataLoader`` indefinitely, advancing the batch sampler epoch.

    ParT's recipe is specified in optimizer *steps*, not epochs, and one pass
    over JetClass is far longer than a Slurm allocation. This wraps the loader so
    the training loop is a flat step loop while ``set_epoch`` still gets called at
    each wrap-around, which is what keeps the shuffle (and each rank's shard)
    changing between passes.
    """

    def __init__(
        self,
        loader: DataLoader,
        batch_sampler: Optional[TrainBatchSampler] = None,
    ):
        self.loader = loader
        self.batch_sampler = batch_sampler
        self.epoch = 0
        self.batches_in_epoch = 0
        self._iterator = None

    def set_epoch(self, epoch: int, batches_in_epoch: int = 0) -> None:
        self.epoch = epoch
        self.batches_in_epoch = batches_in_epoch
        if self.batch_sampler is not None:
            self.batch_sampler.set_epoch(epoch)
            skip = getattr(self.batch_sampler, "set_resume_skip", None)
            if skip is not None:
                skip(batches_in_epoch)
        self._iterator = None

    def __iter__(self):
        return self

    def __next__(self):
        if self._iterator is None:
            if self.batch_sampler is not None:
                self.batch_sampler.set_epoch(self.epoch)
            self._iterator = iter(self.loader)
        try:
            batch = next(self._iterator)
        except StopIteration:
            self.epoch += 1
            self.batches_in_epoch = 0
            if self.batch_sampler is not None:
                self.batch_sampler.set_epoch(self.epoch)
                skip = getattr(self.batch_sampler, "set_resume_skip", None)
                if skip is not None:
                    skip(0)
            self._iterator = iter(self.loader)
            batch = next(self._iterator)
        self.batches_in_epoch += 1
        return batch
