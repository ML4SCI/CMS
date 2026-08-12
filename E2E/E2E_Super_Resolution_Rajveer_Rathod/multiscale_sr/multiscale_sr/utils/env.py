"""Platform detection and resource configuration.

Single entry point: call resolve_env() once at the top of train(). Every
downstream component receives the returned EnvConfig — no scattered
torch.cuda.is_available() calls anywhere else in the codebase.
"""
from __future__ import annotations

import os
import queue
import threading
from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class EnvConfig:
    device: torch.device
    num_workers: int
    pin_memory: bool
    persistent_workers: bool
    prefetch_factor: int | None
    use_amp: bool
    dtype: torch.dtype
    is_kaggle: bool
    is_colab: bool

    def __str__(self) -> str:
        return (
            f"device={self.device} num_workers={self.num_workers} "
            f"pin_memory={self.pin_memory} use_amp={self.use_amp} "
            f"kaggle={self.is_kaggle} colab={self.is_colab}"
        )


def resolve_env() -> EnvConfig:
    is_kaggle = os.path.exists("/kaggle/input")
    is_colab = "COLAB_GPU" in os.environ or "COLAB_RELEASE_TAG" in os.environ
    is_container = is_kaggle or is_colab

    cpu_count = os.cpu_count() or 1

    if torch.cuda.is_available():
        device = torch.device("cuda")
        # Containerised runtimes share /dev/shm; keep workers low to avoid OOM.
        num_workers = 2 if is_container else min(4, cpu_count)
        pin_memory = True
        use_amp = True
        dtype = torch.float16
    elif torch.backends.mps.is_available():
        # fork()+pyarrow deadlocks on macOS; use num_workers=0 and rely on
        # _WrappedPrefetchLoader's background thread for async I/O instead.
        device = torch.device("mps")
        num_workers = 0
        pin_memory = False   # MPS does not support pin_memory
        use_amp = False       # torch.amp not fully supported on MPS yet
        dtype = torch.float32
    else:
        device = torch.device("cpu")
        num_workers = min(2, cpu_count)
        pin_memory = False
        use_amp = False
        dtype = torch.float32

    # Escape hatch: fork()+pyarrow/memmap can deadlock the DataLoader on some
    # containerised CUDA runtimes (observed hanging on Colab after the decode
    # cache is built, GPU idle). Set MULTISCALE_SR_NUM_WORKERS=0 to force the
    # in-process prefetch path used on MPS. Overrides the auto value above.
    _nw_override = os.environ.get("MULTISCALE_SR_NUM_WORKERS")
    if _nw_override is not None and _nw_override.strip() != "":
        num_workers = max(0, int(_nw_override))

    persistent_workers = num_workers > 0 and device.type != "mps"
    prefetch_factor = 4 if num_workers > 0 else None

    return EnvConfig(
        device=device,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
        prefetch_factor=prefetch_factor,
        use_amp=use_amp,
        dtype=dtype,
        is_kaggle=is_kaggle,
        is_colab=is_colab,
    )


class _PrefetchIterator:
    """Reads one item ahead on a background thread using a bounded queue.

    Designed for IterableDatasets where num_workers=0 in DataLoader.
    Decouples CPU data-prep from GPU compute without forking processes,
    which would conflict with pyarrow's use_threads=True.
    """

    _SENTINEL = object()

    def __init__(self, iterable, maxsize: int = 2) -> None:
        self._iterable = iterable
        self._queue: queue.Queue = queue.Queue(maxsize=maxsize)
        self._thread = threading.Thread(target=self._producer, daemon=True)
        self._thread.start()

    def _producer(self) -> None:
        try:
            for item in self._iterable:
                self._queue.put(item)
        finally:
            self._queue.put(self._SENTINEL)

    def __iter__(self):
        return self

    def __next__(self):
        item = self._queue.get()
        if item is self._SENTINEL:
            raise StopIteration
        return item


def prefetch_generator(iterable, maxsize: int = 8) -> _PrefetchIterator:
    """Wrap any iterable with background prefetch (queue depth 8)."""
    return _PrefetchIterator(iterable, maxsize=maxsize)
