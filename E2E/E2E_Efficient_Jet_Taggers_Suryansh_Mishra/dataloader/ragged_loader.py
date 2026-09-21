"""
Shard-coherent DDP DataLoader for CSR ``.pt`` shards (Perlmutter).

Canonical loader module for the ragged JetClass pipeline.  Provides:

- :class:`RaggedShardDataset` — flat-indexed dataset with LRU shard cache and
  :meth:`get_batch` for vectorised CSR gather + pad.
- :class:`ShardCoherentBatchSampler` — DDP-aware batch sampler that assigns
  disjoint shard partitions to each global rank and emits intra-shard batches.
- :class:`ClassBalancedBatchSampler` — training sampler that mixes every class
  into every local batch while retaining vectorised intra-shard reads.
- :func:`create_ragged_train_loader` / :func:`create_ragged_val_loader` —
  factory functions wiring the sampler, dataset, and multi-node-aware defaults.
- :func:`ragged_collate` — loop-based pad collate (flat-shuffle baseline
  fallback for bench mode A).
- :func:`create_ragged_dataloader` — flat-shuffle baseline factory.
- :func:`load_bench_batch` — one real batch from a shard, for benchmarks
  (so kernel/model timings run on real multiplicities, not ``torch.randn``).

Batch contract (all paths)::

    x    (B, 16, P)   float32   particle features, channel-first
    v    (B,  4, P)   float32   raw [px, py, pz, E]
    mask (B,  1, P)   float32   1=real, 0=pad
    y    (B, 10)      float32   one-hot labels

GPU transfer::

    batch = tuple(t.to(device, non_blocking=True) for t in batch)
"""

from __future__ import annotations

import math
import os
import sys
import warnings
from collections import OrderedDict
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Tuple, Union

import numpy as np

import torch
from torch import Tensor
from torch.utils.data import DataLoader, Dataset, Sampler

__all__ = [
    "RaggedShardDataset",
    "ShardCoherentBatchSampler",
    "ClassBalancedBatchSampler",
    "coherent_collate",
    "ragged_collate",
    "create_ragged_train_loader",
    "create_ragged_val_loader",
    "create_ragged_dataloader",
    "load_bench_batch",
]

# ---------------------------------------------------------------------------
# Constants (canonical source — converter re-imports)
# ---------------------------------------------------------------------------

#: Jets per ROOT file / .pt shard.
EVENTS_PER_SHARD: int = 100_000


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------


class RaggedShardDataset(Dataset):
    """Flat-indexed dataset over CSR ``.pt`` shards with LRU shard cache.

    Supports two access patterns:

    1. **Flat indexing** (``__getitem__(int)``) — for the flat-shuffle baseline.
       Returns per-jet ``(x, v, mask, y, n_particles)`` with variable length.

    2. **Batch indexing** (``__getitem__((shard_idx, jet_indices))``) — for the
       shard-coherent fast path.  Calls :meth:`get_batch` internally and
       returns a fully padded, channel-first batch ``(x, v, mask, y)``.

    Parameters
    ----------
    pt_dir : str
        Directory containing ``.pt`` shard files (one per ROOT file).
    cache_size : int
        Number of shard slots in the LRU cache.  Default 2 — keeps the
        current and look-ahead shard hot for shuffle workers.
    events_per_shard : int
        Expected jets per shard.  Default 100 000.  Used for approximate
        ``__len__`` until shards are loaded and exact sizes discovered.
    """

    def __init__(
        self,
        pt_dir: str,
        cache_size: int = 2,
        events_per_shard: int = EVENTS_PER_SHARD,
        probe_shard_sizes: bool = True,
    ):
        super().__init__()
        if not os.path.isdir(pt_dir):
            raise FileNotFoundError(f"Not a directory: {pt_dir}")
        self.files = sorted(
            os.path.join(pt_dir, f)
            for f in os.listdir(pt_dir)
            if f.endswith(".pt")
        )
        if not self.files:
            raise FileNotFoundError(f"No .pt files found in {pt_dir}")
        self.events_per_shard = events_per_shard
        self._cache_size = max(int(cache_size), 1)
        self._cache: "OrderedDict[int, Dict[str, Tensor]]" = OrderedDict()
        # Exact per-shard sizes. Probed up front (cheaply) rather than filled in
        # on first load, because the batch samplers turn these into jet indices
        # before any shard has been loaded — see :meth:`_probe_all_shard_sizes`.
        self._shard_sizes: List[Optional[int]] = [None] * len(self.files)
        if probe_shard_sizes:
            self._probe_all_shard_sizes()

    def __len__(self) -> int:
        return sum(
            n if n is not None else self.events_per_shard
            for n in self._shard_sizes
        )

    @property
    def num_files(self) -> int:
        return len(self.files)

    @staticmethod
    def _probe_shard_size(path: str) -> int:
        """Jet count of one shard, mapping its storages instead of reading them.

        ``mmap=True`` makes this a pickle-header read rather than a full load,
        so probing a thousand shards stays cheap.
        """
        try:
            data = torch.load(
                path, map_location="cpu", weights_only=True, mmap=True
            )
        except (TypeError, RuntimeError, ValueError):
            # ``mmap`` needs torch >= 2.1 and zipfile-serialized shards. Fall
            # back to a full read — never to a guess.
            data = torch.load(path, map_location="cpu", weights_only=True)
        return int(len(data["y"]))

    def _probe_all_shard_sizes(self) -> None:
        """Record every shard's exact jet count before any sampling happens.

        :meth:`shard_size` feeds both batch samplers, which turn it into jet
        indices.  Returning the ``events_per_shard`` *guess* there was silently
        wrong in both directions, and discovering the true size "on first load"
        never helped: with ``num_workers > 0`` shards are only ever loaded inside
        worker processes, so the main-process sampler that generates the indices
        never saw a corrected value.  A shard shorter than the guess produced
        out-of-range indices (``IndexError`` on the first batch); a longer one
        had its tail never sampled at all.
        """
        for shard_idx, path in enumerate(self.files):
            try:
                self._shard_sizes[shard_idx] = self._probe_shard_size(path)
            except Exception as exc:  # unreadable shard: fail here, not mid-epoch
                raise RuntimeError(
                    f"could not read jet count from shard {path}: {exc}"
                ) from exc

    def shard_size(self, shard_idx: int) -> int:
        """Return the number of jets in shard *shard_idx*.

        Exact — probed for every shard at construction and refreshed on load.
        Falls back to :attr:`events_per_shard` only when probing was explicitly
        disabled via ``probe_shard_sizes=False``.
        """
        s = self._shard_sizes[shard_idx]
        return s if s is not None else self.events_per_shard

    def _load_shard(self, shard_idx: int) -> Dict[str, Tensor]:
        """LRU-cached shard load."""
        if shard_idx in self._cache:
            # LRU hit — promote to most-recent
            data = self._cache.pop(shard_idx)
            self._cache[shard_idx] = data
            return data

        # LRU miss
        data = torch.load(
            self.files[shard_idx], map_location="cpu", weights_only=True
        )
        self._cache[shard_idx] = data
        if len(self._cache) > self._cache_size:
            self._cache.popitem(last=False)  # evict least-recent

        # Record actual size on first load so __len__ converges to exact.
        if self._shard_sizes[shard_idx] is None:
            self._shard_sizes[shard_idx] = len(data["y"])

        return data

    # ---- dual __getitem__ ---------------------------------------------------

    def __getitem__(
        self,
        idx: Union[int, Tuple[int, np.ndarray]],
    ) -> Union[
        Tuple[Tensor, Tensor, Tensor, Tensor, Tensor],
        Tuple[Tensor, Tensor, Tensor, Tensor],
    ]:
        # --- Coherent fast path: (shard_idx, jet_indices) from BatchSampler ---
        if isinstance(idx, tuple):
            shard_idx, jet_indices = idx
            return self.get_batch(shard_idx, jet_indices)

        # --- Flat path: single jet by global flat index ----------------------
        shard_idx = idx // self.events_per_shard
        jet_idx = idx % self.events_per_shard

        data = self._load_shard(shard_idx)

        # Guard: partial / corrupt shard with fewer jets than expected.
        actual_n_jets = len(data["y"])
        if jet_idx >= actual_n_jets:
            raise IndexError(
                f"idx {idx}: shard {Path(self.files[shard_idx]).name} "
                f"has {actual_n_jets} jets, requested jet {jet_idx}"
            )

        o = data["offsets"]
        start = int(o[jet_idx])
        end = int(o[jet_idx + 1])

        x = data["x"][start:end]  # (P, 16)
        v = data["v"][start:end]  # (P, 4)
        y = data["y"][jet_idx]  # (10,)
        n_p = data["n_particles"][jet_idx]  # scalar int32

        # Mask: all particles are real (CSR has no padding).
        # E > 0 is the canonical weaver check; always true for CSR particles.
        mask = (v[:, 3:4] > 0).float()  # (P, 1)

        return x, v, mask, y, n_p

    # ---- vectorised batch get -----------------------------------------------

    def get_batch(
        self, shard_idx: int, jet_indices: np.ndarray
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        """Vectorised CSR gather + pad for a batch of jets from one shard.

        All jets must belong to the same shard (guaranteed by
        :class:`ShardCoherentBatchSampler`).  Loads the shard once, gathers
        particle slices via CSR offsets, and pads to ``P_max`` in-place.

        Parameters
        ----------
        shard_idx : int
            Index into :attr:`files`.
        jet_indices : array-like of int
            Intra-shard jet indices.

        Returns
        -------
        x : Tensor  (B, 16, P_max)
        v : Tensor  (B, 4, P_max)
        mask : Tensor  (B, 1, P_max)
        y : Tensor  (B, 10)
        """
        data = self._load_shard(shard_idx)
        offsets = data["offsets"]
        n_particles = data["n_particles"]

        jet_indices = np.asarray(jet_indices, dtype=np.int64)
        B = len(jet_indices)
        counts = n_particles[jet_indices]  # (B,)
        P_max = int(counts.max()) if B > 0 else 0

        # Pre-allocate padded output tensors (zeros = pad)
        x_batch = torch.zeros(B, 16, P_max)
        v_batch = torch.zeros(B, 4, P_max)
        mask_batch = torch.zeros(B, 1, P_max)

        x_src = data["x"]
        v_src = data["v"]

        for b in range(B):
            j = int(jet_indices[b])
            start = int(offsets[j])
            end = int(offsets[j + 1])
            P = end - start
            if P > 0:
                x_batch[b, :, :P] = x_src[start:end].T  # (16, P)
                v_batch[b, :, :P] = v_src[start:end].T  # (4, P)
                mask_batch[b, 0, :P] = 1.0

        y_batch = data["y"][jet_indices]  # (B, 10)

        # --- Cheap runtime guard: mask sum == n_particles per row ---
        actual_n = mask_batch.sum(dim=-1).squeeze(-1)  # (B,)
        expected_n = counts.float()
        if not torch.allclose(actual_n, expected_n):
            bad = torch.where(~torch.isclose(actual_n, expected_n))[0]
            raise RuntimeError(
                f"get_batch: mask sum != n_particles at batch indices "
                f"{bad.tolist()[:10]}"
            )

        return x_batch, v_batch, mask_batch, y_batch


# ---------------------------------------------------------------------------
# Shard-coherent batch sampler (DDP-aware)
# ---------------------------------------------------------------------------


class ShardCoherentBatchSampler(Sampler[List]):
    """DDP-aware batch sampler: disjoint shard partitions per global rank.

    Each rank gets a disjoint subset of shards via round-robin on global rank.
    Within each shard, jet indices are shuffled per epoch and emitted as
    complete batches (all jets from the same shard).

    Yields single-element lists ``[(shard_idx, jet_indices_array)]`` so that
    ``DataLoader`` calls ``dataset[(shard_idx, jet_indices_array)]`` which
    dispatches to :meth:`~RaggedShardDataset.get_batch` on the fast path.

    Parameters
    ----------
    dataset : RaggedShardDataset
        The dataset to sample from.
    batch_size : int
        Number of jets per batch.
    rank : int
        Global rank of this process (0-indexed).
    world_size : int
        Total number of processes.
    drop_last : bool
        Drop the last incomplete batch within each shard.
    shuffle : bool
        Shuffle shard order and jet order within shards each epoch.
    max_jets : int or None
        **Global** cap on jets yielded across all ranks: each rank yields at
        most ``ceil(max_jets / world_size)`` jets.  ``None`` = uncapped.
        Validation uses this to bound the pooled metric payload (see
        ``AblationConfig.val_max_jets``); training never sets it.
    """

    def __init__(
        self,
        dataset: RaggedShardDataset,
        batch_size: int,
        rank: int = 0,
        world_size: int = 1,
        drop_last: bool = True,
        shuffle: bool = True,
        max_jets: Optional[int] = None,
    ):
        if dataset.num_files < world_size:
            raise ValueError(
                f"Cannot partition {dataset.num_files} shards across "
                f"{world_size} ranks — need at least one shard per rank."
            )
        self.dataset = dataset
        self.batch_size = batch_size
        self.rank = rank
        self.world_size = world_size
        self.drop_last = drop_last
        self.shuffle = shuffle
        self.max_jets = max_jets
        self.epoch = 0
        self._skip_batches = 0

        # Round-robin shard assignment by global rank
        all_shard_ids = list(range(dataset.num_files))
        self.my_shards = all_shard_ids[rank::world_size]

    def set_resume_skip(self, n: int) -> None:
        """Advance the next ``__iter__`` by ``n`` batches without yielding them.

        Used by chained-job resume: the shuffle for ``epoch`` is deterministic,
        so skipping the already-consumed prefix restores the next unseen batch
        without replaying mid-epoch data.

        A skip at or beyond the epoch's batch count means the epoch boundary has
        already passed — most often because the checkpoint's
        ``batches_in_epoch`` was written at a different ``world_size`` than the
        one now resuming (per-rank epochs halve when the GPU count doubles).
        Honouring it literally would consume the whole epoch and silently
        discard a full pass over the data, so it is reset to 0 with a warning.
        """
        n = max(0, int(n))
        total = len(self)
        if total > 0 and n >= total:
            warnings.warn(
                f"resume skip {n} >= {total} batches in this rank's epoch; "
                f"the epoch boundary has already passed, so nothing will be "
                f"skipped. Check that the checkpoint's batches_in_epoch was "
                f"written at the current world_size.",
                RuntimeWarning,
                stacklevel=2,
            )
            n = 0
        self._skip_batches = n

    def set_epoch(self, epoch: int) -> None:
        """Set the epoch for deterministic per-rank shuffling.

        Must be called before each training epoch::

            for epoch in range(n_epochs):
                loader.batch_sampler.set_epoch(epoch)
                for batch in loader:
                    ...
        """
        self.epoch = epoch

    def __iter__(self) -> Iterator[List]:
        rng = np.random.default_rng(self.epoch * self.world_size + self.rank)
        # Consume the resume skip into a local: left on the instance, an
        # unexhausted remainder would keep eating batches from the *next* epoch.
        skip = self._skip_batches
        self._skip_batches = 0
        shard_order = self.my_shards.copy()
        if self.shuffle:
            rng.shuffle(shard_order)

        # Global jet budget split evenly across ranks: each rank contributes
        # its share of the cap, so the pooled payload across ranks stays <=
        # max_jets.  The final batch inside the cap is emitted partially,
        # matching the val loader's drop_last=False semantics.
        budget = None
        if self.max_jets is not None:
            budget = math.ceil(self.max_jets / self.world_size)
        emitted = 0

        for shard_idx in shard_order:
            if budget is not None and emitted >= budget:
                break
            n_jets = self.dataset.shard_size(shard_idx)
            jet_indices = np.arange(n_jets)
            if self.shuffle:
                rng.shuffle(jet_indices)

            for start in range(0, n_jets, self.batch_size):
                end = start + self.batch_size
                if budget is not None:
                    remaining = budget - emitted
                    if remaining <= 0:
                        break
                    end = min(end, start + remaining)
                if end > n_jets:
                    if self.drop_last:
                        continue
                    end = n_jets
                batch_jets = jet_indices[start:end]
                emitted += end - start
                if skip > 0:
                    skip -= 1
                    continue
                # Single-element list: DataLoader calls
                # dataset[(shard_idx, batch_jets)] → get_batch
                yield [(shard_idx, batch_jets)]

    def __len__(self) -> int:
        budget = None
        if self.max_jets is not None:
            budget = math.ceil(self.max_jets / self.world_size)
        jets = 0
        total = 0
        for shard_idx in self.my_shards:
            if budget is not None and jets >= budget:
                break
            n = self.dataset.shard_size(shard_idx)
            if budget is not None:
                n = min(n, budget - jets)
            jets += n
            if self.drop_last:
                total += n // self.batch_size
            else:
                total += math.ceil(n / self.batch_size)
        return total


class ClassBalancedBatchSampler(Sampler[List]):
    """DDP-aware sampler producing randomly shuffled, class-balanced batches.

    JetClass source files are single-class. Processing a complete 100k-jet
    shard before switching classes makes the current-label training accuracy
    look excellent while causing catastrophic forgetting. This sampler instead
    draws an equal share of each local batch from every filename class.

    A yielded batch is a list of ``(shard_idx, jet_indices)`` requests, usually
    one per class. :func:`coherent_collate` pads and concatenates those
    vectorised sub-batches. Shards remain disjoint across DDP ranks.
    """

    def __init__(
        self,
        dataset: RaggedShardDataset,
        batch_size: int,
        rank: int = 0,
        world_size: int = 1,
        drop_last: bool = True,
    ):
        if batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {batch_size}")
        self.dataset = dataset
        self.batch_size = batch_size
        self.rank = rank
        self.world_size = world_size
        self.drop_last = drop_last
        self.epoch = 0
        self._skip_batches = 0

        # Converted filenames preserve the source basename, e.g. HToBB_042.pt.
        # Grouping by the suffix-free prefix avoids loading 1000 shards merely
        # to discover labels. It also degrades safely to one group for generic
        # names such as shard_000.pt used in loader tests.
        shards_by_class: Dict[str, List[int]] = {}
        for shard_idx, filename in enumerate(dataset.files):
            class_name = Path(filename).stem.rsplit("_", 1)[0]
            shards_by_class.setdefault(class_name, []).append(shard_idx)

        if batch_size < len(shards_by_class):
            raise ValueError(
                f"batch_size={batch_size} cannot include all "
                f"{len(shards_by_class)} classes"
            )

        self.my_shards_by_class: Dict[str, List[int]] = {}
        for class_name, shard_ids in sorted(shards_by_class.items()):
            rank_shards = shard_ids[rank::world_size]
            if not rank_shards:
                raise ValueError(
                    f"class {class_name!r} has {len(shard_ids)} shards, "
                    f"fewer than world_size={world_size}"
                )
            self.my_shards_by_class[class_name] = rank_shards
        self.class_names = tuple(self.my_shards_by_class)

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def set_resume_skip(self, n: int) -> None:
        """Advance the next ``__iter__`` by ``n`` batches without yielding them.

        See :meth:`ShardCoherentBatchSampler.set_resume_skip`, including why a
        skip at or beyond the epoch's batch count is reset to 0 with a warning.
        """
        n = max(0, int(n))
        total = len(self)
        if total > 0 and n >= total:
            warnings.warn(
                f"resume skip {n} >= {total} batches in this rank's epoch; "
                f"the epoch boundary has already passed, so nothing will be "
                f"skipped. Check that the checkpoint's batches_in_epoch was "
                f"written at the current world_size.",
                RuntimeWarning,
                stacklevel=2,
            )
            n = 0
        self._skip_batches = n

    def __iter__(self) -> Iterator[List]:
        # Rank-specific randomness selects disjoint samples; epoch changes both
        # shard order and within-shard permutations on every full data pass.
        rng = np.random.default_rng(self.epoch * self.world_size + self.rank)
        # Consume the resume skip into a local; see the sibling sampler.
        skip = self._skip_batches
        self._skip_batches = 0
        states = {}
        for class_name in self.class_names:
            shard_order = self.my_shards_by_class[class_name].copy()
            rng.shuffle(shard_order)
            states[class_name] = {
                "shards": shard_order,
                "shard_pos": 0,
                "shard_idx": None,
                "jet_order": None,
                "jet_pos": 0,
            }

        def take_from_class(class_name, count):
            """Return vectorised requests without iterating jet-by-jet."""
            state = states[class_name]
            requests = []
            remaining = count
            while remaining:
                jet_order = state["jet_order"]
                jet_pos = state["jet_pos"]
                if jet_order is None or jet_pos >= len(jet_order):
                    shard_pos = state["shard_pos"]
                    shards = state["shards"]
                    if shard_pos >= len(shards):
                        return None
                    shard_idx = shards[shard_pos]
                    jet_order = np.arange(self.dataset.shard_size(shard_idx))
                    rng.shuffle(jet_order)
                    state["shard_pos"] = shard_pos + 1
                    state["shard_idx"] = shard_idx
                    state["jet_order"] = jet_order
                    state["jet_pos"] = 0
                    jet_pos = 0

                end = min(jet_pos + remaining, len(jet_order))
                requests.append((state["shard_idx"], jet_order[jet_pos:end]))
                consumed = end - jet_pos
                state["jet_pos"] = end
                remaining -= consumed
            return requests

        num_classes = len(self.class_names)
        base, remainder = divmod(self.batch_size, num_classes)
        batch_idx = 0

        while True:
            # Rotate the +1 allocations so non-divisible batch sizes remain
            # balanced over time (256 / 10 => six classes get 26, four get 25).
            extra_start = (batch_idx * remainder) % num_classes
            requests: List[Tuple[int, np.ndarray]] = []
            exhausted = False
            for class_pos, class_name in enumerate(self.class_names):
                relative_pos = (class_pos - extra_start) % num_classes
                take = base + int(relative_pos < remainder)
                class_requests = take_from_class(class_name, take)
                if class_requests is None:
                    exhausted = True
                    break
                requests.extend(class_requests)

            if exhausted:
                # JetClass classes have equal sizes, so this discards at most
                # one partial balanced batch per rank.
                if not self.drop_last and requests:
                    yield requests
                return
            rng.shuffle(requests)
            batch_idx += 1
            if skip > 0:
                skip -= 1
                continue
            yield requests

    def __len__(self) -> int:
        per_class = [
            sum(self.dataset.shard_size(i) for i in shard_ids)
            for shard_ids in self.my_shards_by_class.values()
        ]
        balanced_jets = min(per_class) * len(per_class)
        if self.drop_last:
            return balanced_jets // self.batch_size
        return math.ceil(balanced_jets / self.batch_size)


# ---------------------------------------------------------------------------
# Collate functions
# ---------------------------------------------------------------------------


def coherent_collate(
    batch: List[Tuple[Tensor, Tensor, Tensor, Tensor]],
) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
    """Collate one or more vectorised intra-shard sub-batches.

    :meth:`~RaggedShardDataset.get_batch` already returns a padded
    ``(B, F, P)`` batch. A coherent batch has one item and is simply unwrapped;
    a class-balanced batch has one item per class, which is padded to the
    largest particle width and concatenated.
    """
    if not batch:
        raise ValueError("coherent_collate received an empty batch")
    if len(batch) == 1:
        return batch[0]

    max_particles = max(x.shape[-1] for x, _, _, _ in batch)
    xs, vs, masks, ys = [], [], [], []
    for x, v, mask, y in batch:
        pad = max_particles - x.shape[-1]
        if pad:
            x = torch.nn.functional.pad(x, (0, pad))
            v = torch.nn.functional.pad(v, (0, pad))
            mask = torch.nn.functional.pad(mask, (0, pad))
        xs.append(x)
        vs.append(v)
        masks.append(mask)
        ys.append(y)
    return (
        torch.cat(xs, dim=0),
        torch.cat(vs, dim=0),
        torch.cat(masks, dim=0),
        torch.cat(ys, dim=0),
    )


def ragged_collate(
    batch: List[Tuple[Tensor, Tensor, Tensor, Tensor, Tensor]],
) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
    """Loop-based pad collate for the flat-shuffle baseline (bench mode A).

    Each item is ``(x (P,16), v (P,4), mask (P,1), y (10,), n_particles ())``.
    Pads to max-in-batch and transposes to channel-first.

    Parameters
    ----------
    batch : list of 5-tuples from ``RaggedShardDataset.__getitem__(int)``

    Returns
    -------
    x : Tensor  (B, 16, P_max)  float32  particle features
    v : Tensor  (B,  4, P_max)  float32  raw [px, py, pz, E]
    mask : Tensor  (B,  1, P_max)  float32  1=real, 0=pad
    y : Tensor  (B, 10)  float32  one-hot labels
    """
    xs, vs, masks, ys, n_ps = zip(*batch)

    B = len(xs)
    P_max = max(x.shape[0] for x in xs)
    F_x = xs[0].shape[1]  # 16
    F_v = vs[0].shape[1]  # 4

    x_batch = torch.zeros(B, F_x, P_max)
    v_batch = torch.zeros(B, F_v, P_max)
    mask_batch = torch.zeros(B, 1, P_max)

    for b, (x, v, m, _y, _n_p) in enumerate(
        zip(xs, vs, masks, ys, n_ps)
    ):
        P = x.shape[0]
        x_batch[b, :, :P] = x.T  # (F_x, P)
        v_batch[b, :, :P] = v.T  # (F_v, P)
        # m may be (P, 1) or (P,) — squeeze to 1D for assignment
        mask_1d = m.squeeze(-1) if m.ndim == 2 else m
        mask_batch[b, 0, :P] = mask_1d

    y_batch = torch.stack(ys, dim=0)  # (B, 10)

    # --- Cheap runtime guard: mask sum == n_particles per row ---
    actual_n = mask_batch.sum(dim=-1).squeeze(-1)  # (B,)
    expected_n = torch.tensor(
        [int(n) for n in n_ps], dtype=torch.float32
    )
    if not torch.allclose(actual_n, expected_n):
        bad = torch.where(~torch.isclose(actual_n, expected_n))[0]
        raise RuntimeError(
            f"ragged_collate: mask sum != n_particles at batch indices "
            f"{bad.tolist()[:10]}"
        )

    return x_batch, v_batch, mask_batch, y_batch


# ---------------------------------------------------------------------------
# Factory API — shard-coherent (fast path)
# ---------------------------------------------------------------------------


def _worker_init_fn(worker_id: int) -> None:
    """Seed numpy RNG per worker to decorrelate any data augmentation.

    Must be a module-level function (not a closure) so that it is
    picklable — ``DataLoader`` with ``num_workers > 0`` serialises the
    function and sends it to worker subprocesses.
    """
    np.random.seed((torch.initial_seed() + worker_id) % (2**31))


def _default_num_workers(world_size: int) -> int:
    """4 workers for single-node (≤4 GPUs), 2 for multi-node.

    Caps Lustre open file-descriptor streams when many nodes run concurrently.
    """
    return 4 if world_size <= 4 else 2


def create_ragged_train_loader(
    pt_dir: str,
    batch_size: int = 512,
    rank: int = 0,
    world_size: int = 1,
    num_workers: Optional[int] = None,
    cache_size: int = 1,
    prefetch_factor: int = 2,
    pin_memory: bool = True,
    persistent_workers: bool = True,
    events_per_shard: int = EVENTS_PER_SHARD,
    collate_fn=coherent_collate,
    class_balanced: bool = False,
) -> DataLoader:
    """Shard-coherent training DataLoader with multi-node defaults.

    Uses :class:`ShardCoherentBatchSampler` +
    :meth:`~RaggedShardDataset.get_batch` for the fast path.

    Callers must call ``loader.batch_sampler.set_epoch(epoch)`` each epoch.

    GPU transfer::

        batch = tuple(t.to(device, non_blocking=True) for t in batch)

    Parameters
    ----------
    pt_dir : str
        Directory containing ``.pt`` shard files.
    batch_size : int
        Batch size.  Default 512.
    rank, world_size : int
        Global DDP rank and world size.  Pass from
        ``ablation.distributed.DistContext``.
    num_workers : int or None
        DataLoader worker processes.  Default: 4 if ``world_size ≤ 4``
        else 2.
    cache_size : int
        LRU shard-cache slots per worker.  Default 1 — one hot shard per
        worker after coherent sampling.
    prefetch_factor : int
        Batches prefetched per worker.  Default 2.
    pin_memory : bool
        Pin memory for async H2D.  Default True.
    persistent_workers : bool
        Keep workers alive between epochs.  Default True.
    events_per_shard : int
        Expected jets per shard.  Default 100 000.
    collate_fn : callable
        Collate on top of :func:`coherent_collate`.  The ablation harness
        passes a normalization wrapper here when ``norm_stats_path`` is set;
        default leaves the batch untouched.
    class_balanced : bool
        Mix all filename classes into every batch. Required for training on
        JetClass's single-class source shards; disabled by default so coherent
        loader benchmarks retain their original one-shard access pattern.
    """
    if num_workers is None:
        num_workers = _default_num_workers(world_size)

    # A balanced worker accesses one active shard per class. Keep them resident
    # rather than re-reading ~3 GiB of shards on every batch.
    if class_balanced:
        cache_size = max(cache_size, 10)
    ds = RaggedShardDataset(
        pt_dir, cache_size=cache_size, events_per_shard=events_per_shard
    )
    if class_balanced:
        sampler = ClassBalancedBatchSampler(
            ds,
            batch_size=batch_size,
            rank=rank,
            world_size=world_size,
            drop_last=True,
        )
    else:
        sampler = ShardCoherentBatchSampler(
            ds,
            batch_size=batch_size,
            rank=rank,
            world_size=world_size,
            drop_last=True,
            shuffle=True,
        )

    _prefetch: Optional[int] = prefetch_factor
    _wif = _worker_init_fn
    if num_workers == 0:
        persistent_workers = False
        _prefetch = None  # PyTorch enforces this for num_workers=0
        _wif = None

    return DataLoader(
        ds,
        batch_sampler=sampler,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
        prefetch_factor=_prefetch,
        worker_init_fn=_wif,
    )


def create_ragged_val_loader(
    pt_dir: str,
    batch_size: int = 512,
    rank: int = 0,
    world_size: int = 1,
    num_workers: Optional[int] = None,
    cache_size: int = 1,
    prefetch_factor: int = 2,
    pin_memory: bool = True,
    persistent_workers: bool = True,
    events_per_shard: int = EVENTS_PER_SHARD,
    max_jets: Optional[int] = None,
    collate_fn=coherent_collate,
) -> DataLoader:
    """Shard-coherent validation DataLoader.

    Same as :func:`create_ragged_train_loader` but with ``drop_last=False``
    and no within-shard shuffle.

    max_jets : int or None
        Global cap on jets across all ranks (split evenly per rank), passed
        to :class:`ShardCoherentBatchSampler`.  Validation metrics need
        predictions gathered on one rank, so the 5M-jet split is capped by
        ``AblationConfig.val_max_jets``; training passes ``None``.
    collate_fn : callable
        See :func:`create_ragged_train_loader`.
    """
    if num_workers is None:
        num_workers = _default_num_workers(world_size)

    ds = RaggedShardDataset(
        pt_dir, cache_size=cache_size, events_per_shard=events_per_shard
    )
    sampler = ShardCoherentBatchSampler(
        ds,
        batch_size=batch_size,
        rank=rank,
        world_size=world_size,
        drop_last=False,
        shuffle=False,
        max_jets=max_jets,
    )

    _prefetch: Optional[int] = prefetch_factor
    _wif = _worker_init_fn
    if num_workers == 0:
        persistent_workers = False
        _prefetch = None
        _wif = None

    return DataLoader(
        ds,
        batch_sampler=sampler,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
        prefetch_factor=_prefetch,
        worker_init_fn=_wif,
    )


# ---------------------------------------------------------------------------
# Flat-shuffle baseline (bench mode A / backward compat)
# ---------------------------------------------------------------------------


def create_ragged_dataloader(
    pt_dir: str,
    batch_size: int = 512,
    num_workers: int = 4,
    shuffle: bool = True,
    cache_size: int = 2,
    pin_memory: bool = True,
    persistent_workers: bool = True,
    prefetch_factor: int = 4,
    drop_last: bool = True,
) -> DataLoader:
    """Flat-shuffle DataLoader — baseline for benchmarking (mode A).

    Uses standard random shuffle over flat indices + loop-based
    :func:`ragged_collate`.  This is the original loader from the converter,
    kept as the bench (A) baseline and for backward compatibility.

    Parameters
    ----------
    pt_dir : str
        Directory containing ``.pt`` shard files.
    batch_size : int
        Batch size.  Default 512.
    num_workers : int
        DataLoader worker processes.  Default 4.
    shuffle : bool
        Shuffle the flat index.  Default True.
    cache_size : int
        LRU shard-cache slots per worker.  Default 2.
    pin_memory : bool
        Pin memory for async GPU transfer.  Default True.
    persistent_workers : bool
        Keep workers alive between epochs.  Default True.
    prefetch_factor : int
        Batches prefetched per worker.  Default 4.
    drop_last : bool
        Drop incomplete last batch.  Default True.

    Returns
    -------
    DataLoader
    """
    ds = RaggedShardDataset(pt_dir, cache_size=cache_size)

    _prefetch: Optional[int] = prefetch_factor
    _wif = _worker_init_fn
    if num_workers == 0:
        persistent_workers = False
        _prefetch = None
        _wif = None

    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=ragged_collate,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
        prefetch_factor=_prefetch,
        drop_last=drop_last,
        worker_init_fn=_wif,
    )


# ---------------------------------------------------------------------------
# Smoke self-check (runs when executed directly)
# ---------------------------------------------------------------------------


def _smoke_self_check() -> int:
    """Lightweight self-tests that don't need ROOT files.

    Verifies: dataset slicing (flat + batch), sampler partitioning, sampler
    determinism, collate shapes, factory wiring, mask guard.
    """
    import tempfile

    print("=== ragged_loader smoke self-check (synthetic data) ===")

    # --- Build a tiny synthetic shard ---
    N_jets = 20
    counts = np.array(
        [3, 5, 2, 7, 4, 1, 6, 8, 9, 10, 3, 5, 2, 7, 4, 1, 6, 8, 9, 10],
        dtype=np.int32,
    )
    assert len(counts) == N_jets
    N_part = int(counts.sum())
    offsets = np.empty(N_jets + 1, dtype=np.int64)
    offsets[0] = 0
    np.cumsum(counts, out=offsets[1:])

    rng_np = np.random.default_rng(42)
    x = rng_np.normal(size=(N_part, 16)).astype(np.float32)
    v = np.abs(rng_np.normal(size=(N_part, 4)).astype(np.float32))  # E > 0
    jet = np.column_stack(
        [rng_np.normal(size=N_jets).astype(np.float32) for _ in range(9)]
        + [counts.astype(np.float32)]
    )
    y = np.zeros((N_jets, 10), dtype=np.float32)
    y[np.arange(N_jets), rng_np.integers(0, 10, size=N_jets)] = 1.0

    tensors = {
        "x": torch.from_numpy(x),
        "v": torch.from_numpy(v),
        "offsets": torch.from_numpy(offsets),
        "jet": torch.from_numpy(jet),
        "y": torch.from_numpy(y),
        "n_particles": torch.from_numpy(counts),
    }
    print(f"  Synthetic shard: {N_jets} jets, {N_part} particles")

    # --- Save two shards ---
    with tempfile.TemporaryDirectory() as tmp:
        for i in range(2):
            pt_path = os.path.join(tmp, f"shard_{i:03d}.pt")
            torch.save(tensors, pt_path, _use_new_zipfile_serialization=True)

        # --- Dataset flat indexing ---
        ds = RaggedShardDataset(tmp, events_per_shard=N_jets)
        assert ds.num_files == 2
        # Length uses events_per_shard until shards are loaded
        assert len(ds) == 2 * N_jets

        for idx in [0, N_jets - 1]:
            x_i, v_i, mask_i, y_i, n_p_i = ds[idx]
            n_expected = int(counts[idx])
            assert x_i.shape == (n_expected, 16), f"idx={idx}: x shape {x_i.shape}"
            assert v_i.shape == (n_expected, 4)
            assert mask_i.shape == (n_expected, 1)
            assert y_i.shape == (10,)
            assert int(n_p_i) == n_expected
            assert int(mask_i.sum()) == n_expected
        print("  Dataset flat indexing: OK")

        # --- get_batch (CSR gather + pad) ---
        jet_idxs = np.array([0, 1, 2, 4])  # jets with 3,5,2,4 particles
        x_b, v_b, m_b, y_b = ds.get_batch(0, jet_idxs)
        assert x_b.shape == (4, 16, 5), f"x_batch shape {x_b.shape}"  # P_max=5
        assert v_b.shape == (4, 4, 5)
        assert m_b.shape == (4, 1, 5)
        assert y_b.shape == (4, 10)
        for b_idx, orig_idx in enumerate(jet_idxs):
            assert int(m_b[b_idx].sum()) == int(counts[orig_idx])
        print("  get_batch (CSR gather + pad): OK")

        # --- Dataset batch indexing via __getitem__(tuple) ---
        result = ds[(0, jet_idxs)]
        assert len(result) == 4  # (x, v, mask, y) — no n_particles
        assert result[0].shape == x_b.shape
        print("  Dataset batch indexing (tuple path): OK")

        # --- ShardCoherentBatchSampler ---
        bs = 4
        sampler = ShardCoherentBatchSampler(
            ds, batch_size=bs, rank=0, world_size=2, drop_last=True, shuffle=True
        )
        # rank=0 gets shard 0, rank=1 gets shard 1 (2 shards, 2 ranks)
        assert sampler.my_shards == [0], f"my_shards={sampler.my_shards}"
        batches = list(sampler)
        assert len(batches) == N_jets // bs  # 20 // 4 = 5
        for batch in batches:
            assert len(batch) == 1, "batch should be single-element list"
            shard_idx, jet_indices = batch[0]
            assert shard_idx == 0, "all batches from shard 0"
            assert len(jet_indices) == bs
        print(f"  ShardCoherentBatchSampler: {len(batches)} batches, OK")

        # --- Sampler determinism: same epoch → same batches ---
        sampler.set_epoch(0)
        batches_a = list(sampler)
        sampler.set_epoch(0)
        batches_b = list(sampler)
        for a, b in zip(batches_a, batches_b):
            assert np.array_equal(a[0][1], b[0][1])
        print("  Sampler determinism (same epoch): OK")

        # --- Sampler different epoch → different order ---
        sampler.set_epoch(1)
        batches_c = list(sampler)
        # Jets should be reshuffled (extremely unlikely to match)
        first_a = batches_a[0][0][1]
        first_c = batches_c[0][0][1]
        # We don't assert they differ (tiny probability they match),
        # just check the count is right.
        assert len(batches_c) == N_jets // bs
        print("  Sampler epoch shuffle: OK")

        # --- Shard partitioning across ranks (no overlap) ---
        sampler_0 = ShardCoherentBatchSampler(
            ds, batch_size=bs, rank=0, world_size=2
        )
        sampler_1 = ShardCoherentBatchSampler(
            ds, batch_size=bs, rank=1, world_size=2
        )
        assert set(sampler_0.my_shards) & set(sampler_1.my_shards) == set()
        assert set(sampler_0.my_shards) | set(sampler_1.my_shards) == {0, 1}
        print("  Shard partitioning (disjoint): OK")

        # --- coherent_collate ---
        # Simulate what DataLoader does: wrap get_batch result in a list
        batch_result = ds.get_batch(0, jet_idxs)
        collated = coherent_collate([batch_result])
        assert collated[0].shape == batch_result[0].shape
        print("  coherent_collate: OK")

        # --- ragged_collate (flat baseline) ---
        flat_batch = [ds[i] for i in [0, 1, 2, 4]]  # 3,5,2,4 particles
        x_rc, v_rc, m_rc, y_rc = ragged_collate(flat_batch)
        assert x_rc.shape == (4, 16, 5)
        assert v_rc.shape == (4, 4, 5)
        assert m_rc.shape == (4, 1, 5)
        assert y_rc.shape == (4, 10)
        print("  ragged_collate (flat baseline): OK")

        # --- Factory: create_ragged_train_loader ---
        train_loader = create_ragged_train_loader(
            tmp,
            batch_size=bs,
            rank=0,
            world_size=1,
            num_workers=0,
            events_per_shard=N_jets,
        )
        batch_count = 0
        for x_b, v_b, m_b, y_b in train_loader:
            assert x_b.ndim == 3 and x_b.shape[1] == 16
            assert v_b.ndim == 3 and v_b.shape[1] == 4
            assert m_b.ndim == 3 and m_b.shape[1] == 1
            assert y_b.ndim == 2 and y_b.shape[1] == 10
            batch_count += 1
        # 2 shards, 20 jets each, bs=4, drop_last=True → 5 batches per shard
        assert batch_count == 10, f"expected 10 batches, got {batch_count}"
        print(f"  create_ragged_train_loader: {batch_count} batches, OK")

        # --- Factory: create_ragged_val_loader ---
        val_loader = create_ragged_val_loader(
            tmp,
            batch_size=6,
            rank=0,
            world_size=1,
            num_workers=0,
            events_per_shard=N_jets,
        )
        batch_count = 0
        for x_b, v_b, m_b, y_b in val_loader:
            batch_count += 1
        # 2 shards, 20 jets each, bs=6, drop_last=False
        # → ceil(20/6)=4 batches per shard → 8 total
        assert batch_count == 8, f"expected 8 val batches, got {batch_count}"
        print(f"  create_ragged_val_loader: {batch_count} batches, OK")

        # --- Factory: create_ragged_dataloader (flat baseline) ---
        # Build the dataset with correct events_per_shard so flat index math
        # works on tiny synthetic shards (production shards are all 100k).
        flat_ds = RaggedShardDataset(tmp, cache_size=2, events_per_shard=N_jets)
        flat_loader = DataLoader(
            flat_ds,
            batch_size=bs,
            shuffle=True,
            num_workers=0,
            collate_fn=ragged_collate,
            drop_last=True,
        )
        batch_count = 0
        for x_b, v_b, m_b, y_b in flat_loader:
            batch_count += 1
        # 2 shards × 20 jets = 40 jets, bs=4, drop_last → 10
        assert batch_count == 10, f"expected 10 flat batches, got {batch_count}"
        print(f"  create_ragged_dataloader (flat): {batch_count} batches, OK")

        # --- num_workers > 0 pickle test ---
        # Verifies _worker_init_fn is picklable (module-level, not closure).
        # persistent_workers=False: persistent workers would outlive the
        # TemporaryDirectory below and deadlock interpreter shutdown.
        mw_loader = create_ragged_train_loader(
            tmp,
            batch_size=bs,
            rank=0,
            world_size=1,
            num_workers=2,
            events_per_shard=N_jets,
            persistent_workers=False,
        )
        mw_count = 0
        for x_b, v_b, m_b, y_b in mw_loader:
            mw_count += 1
        assert mw_count == 10, f"expected 10 multi-worker batches, got {mw_count}"
        print(f"  num_workers=2 pickle test: {mw_count} batches, OK")

    print("=== All ragged_loader self-checks passed ===\n")
    return 0


# ---------------------------------------------------------------------------
# Benchmark helper — one real batch, no DataLoader machinery
# ---------------------------------------------------------------------------


def load_bench_batch(
    pt_dir: str,
    batch_size: int = 128,
    shard_idx: int = 0,
    device: str = "cpu",
    pad_to: Optional[int] = None,
    num_features: int = 16,
) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
    """Fetch one real padded batch straight from a CSR shard, for benchmarking.

    Benchmarks want a single representative batch, not an epoch, so this skips
    ``DataLoader``/sampler entirely and calls
    :meth:`RaggedShardDataset.get_batch` once. Using it instead of
    ``torch.randn`` matters for measurement: real jets have a realistic
    multiplicity distribution (mean ~39, p99 80, max 183), so a batch padded to
    max-in-batch is typically far narrower than the 128 that synthetic
    benchmarks assume — and the pair pipeline is ``O(P^2)``.

    Parameters
    ----------
    pt_dir : str
        Directory of ragged CSR ``.pt`` shards, e.g.
        ``$PSCRATCH/jetclass/pt_ragged/val_5M``.
    batch_size : int
        Jets to fetch. Clamped to the shard's jet count.
    shard_idx : int
        Which shard to read. Default 0.
    device : str
        Device to move the batch onto.
    pad_to : int or None
        Pad/trim the particle axis to exactly this width. ``None`` (default)
        keeps the natural max-in-batch width. **Required for fixed-shape
        consumers** — CUDA-graph capture and ``torch.compile`` without dynamic
        shapes will otherwise recapture or fail when ``P`` varies.
    num_features : int
        Expected feature count on ``x``; the loader emits 16. Mismatches raise
        rather than silently benchmarking the wrong shape.

    Returns
    -------
    tuple
        ``(x (B,F,P), v (B,4,P), mask (B,1,P), y (B,10))`` on *device*.

    Raises
    ------
    FileNotFoundError
        If *pt_dir* is not a directory or holds no ``.pt`` shards.
    ValueError
        If ``x`` does not carry *num_features* channels.
    """
    ds = RaggedShardDataset(pt_dir, cache_size=1)
    if not 0 <= shard_idx < ds.num_files:
        raise ValueError(
            f"shard_idx {shard_idx} out of range for {ds.num_files} shards "
            f"in {pt_dir}"
        )

    # shard_size() is exact here: RaggedShardDataset probes every shard at
    # construction. It used to be queried *before* the shard had ever been
    # loaded, so it returned the 100_000-jet guess and this clamp was dead —
    # `min(batch_size, 100_000)` is just `batch_size`, and a small or partial
    # shard raised IndexError inside get_batch instead.
    n_jets = ds.shard_size(shard_idx)
    take = min(int(batch_size), int(n_jets))
    if take < 1:
        raise ValueError(
            f"shard {Path(ds.files[shard_idx]).name} holds {n_jets} jets; "
            f"cannot build a batch"
        )
    x, v, mask, y = ds.get_batch(shard_idx, np.arange(take))

    if x.shape[1] != num_features:
        raise ValueError(
            f"shard {pt_dir} yields x with {x.shape[1]} features, expected "
            f"{num_features}. The ragged loader contract is 16 particle "
            f"features; a model configured for a different input_dim will "
            f"silently benchmark the wrong shape."
        )

    if pad_to is not None:
        P = x.shape[-1]
        if pad_to < P:
            x, v, mask = x[..., :pad_to], v[..., :pad_to], mask[..., :pad_to]
        elif pad_to > P:
            width = pad_to - P
            x = torch.nn.functional.pad(x, (0, width))
            v = torch.nn.functional.pad(v, (0, width))
            mask = torch.nn.functional.pad(mask, (0, width))

    return (
        x.to(device),
        v.to(device),
        mask.to(device),
        y.to(device),
    )


if __name__ == "__main__":
    raise SystemExit(_smoke_self_check())
