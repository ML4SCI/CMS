"""Distributed setup for Slurm-launched DDP runs (NERSC Perlmutter).

Perlmutter launches one task per GPU with ``srun``, so the process topology comes
from Slurm rather than ``torchrun``.  This module reads it from the environment
following NERSC's documented convention (``RANK=SLURM_PROCID``,
``LOCAL_RANK=SLURM_LOCALID``, ``WORLD_SIZE=SLURM_NTASKS``), falling back to
``torchrun``'s variables and finally to a single-process run, so the same script
works unchanged on a login node, a laptop, and a multi-node allocation.

See https://docs.nersc.gov/machinelearning/pytorch/ and
https://github.com/NERSC/nersc-dl-wandb.
"""

from __future__ import annotations

import os
import socket
from dataclasses import dataclass
from datetime import timedelta
from typing import Iterable, Optional

import torch
import torch.distributed as dist

__all__ = [
    "COLLECTIVE_TIMEOUT",
    "DistContext",
    "setup_distributed",
    "cleanup_distributed",
]

#: How long a rank waits at a collective before NCCL's watchdog aborts the job. PyTorch's
#: NCCL default is 10 minutes. After every evaluation, rank 0 alone scores 4.9M jets and
#: writes the ~200 MB fp32 prediction archive plus ``best.pt`` to scratch while ranks 1-3
#: are already parked at the next gradient all-reduce; tied_k1_1M (Slurm 58152351) died
#: at step 425k on 2026-09-13 when that post-eval work stalled past the 10-minute default
#: on a slow filesystem (16 earlier evals passed the same path). 30 minutes covers a 3x
#: worse stall; a rank that is still absent after that is a real hang, not I/O.
COLLECTIVE_TIMEOUT = timedelta(minutes=30)


@dataclass
class DistContext:
    """Resolved process topology."""

    rank: int
    local_rank: int
    world_size: int
    device: torch.device
    backend: str
    hostname: str

    @property
    def is_main(self) -> bool:
        """True on exactly one process — the one that logs and checkpoints."""
        return self.rank == 0

    @property
    def is_distributed(self) -> bool:
        return self.world_size > 1

    def barrier(self) -> None:
        if self.is_distributed:
            dist.barrier()

    def all_reduce_mean(self, value: torch.Tensor) -> torch.Tensor:
        """Average a scalar/tensor across ranks (returns a new tensor)."""
        if not self.is_distributed:
            return value
        out = value.clone()
        dist.all_reduce(out, op=dist.ReduceOp.SUM)
        return out / self.world_size

    def all_reduce_sum(self, value: torch.Tensor) -> torch.Tensor:
        if not self.is_distributed:
            return value
        out = value.clone()
        dist.all_reduce(out, op=dist.ReduceOp.SUM)
        return out

    def gather_objects(self, obj) -> Optional[list]:
        """Collect one Python object per rank onto every rank.

        Used to bring validation predictions together for global AUC and
        background-rejection numbers, which cannot be computed as an average of
        per-rank values. Callers must keep the payload small — see
        ``AblationConfig.val_max_jets``.
        """
        if not self.is_distributed:
            return [obj]
        bucket: list = [None] * self.world_size
        dist.all_gather_object(bucket, obj)
        return bucket


def _env_int(names: Iterable[str], default: int) -> int:
    for name in names:
        raw = os.environ.get(name)
        if raw is not None and raw != "":
            return int(raw)
    return default


def setup_distributed(seed: int = 0, allow_tf32: bool = True) -> DistContext:
    """Initialize the process group and bind this process to its GPU.

    Seeds are offset per rank so that dropout and any sampling decorrelate
    across processes, while the data order stays controlled by
    ``DistributedSampler``.
    """
    # Slurm first (Perlmutter), then torchrun, then single process.
    rank = _env_int(("SLURM_PROCID", "RANK"), 0)
    local_rank = _env_int(("SLURM_LOCALID", "LOCAL_RANK"), 0)
    world_size = _env_int(("SLURM_NTASKS", "WORLD_SIZE"), 1)

    use_cuda = torch.cuda.is_available()
    if use_cuda:
        # gpus-per-task=1 exposes a single device per task, in which case the
        # local rank is not a valid index into the visible devices.
        device_index = local_rank % torch.cuda.device_count()
        torch.cuda.set_device(device_index)
        device = torch.device("cuda", device_index)
        backend = "nccl"
    else:
        device = torch.device("cpu")
        backend = "gloo"

    if world_size > 1 and not dist.is_initialized():
        os.environ.setdefault("MASTER_PORT", "29500")
        if "MASTER_ADDR" not in os.environ:
            node_list = os.environ.get("SLURM_JOB_NODELIST", "")
            os.environ["MASTER_ADDR"] = (
                _first_hostname(node_list) if node_list else "127.0.0.1"
            )
        os.environ["RANK"] = str(rank)
        os.environ["LOCAL_RANK"] = str(local_rank)
        os.environ["WORLD_SIZE"] = str(world_size)
        dist.init_process_group(
            backend=backend,
            rank=rank,
            world_size=world_size,
            timeout=COLLECTIVE_TIMEOUT,
        )

    torch.manual_seed(seed + rank)
    if use_cuda:
        torch.cuda.manual_seed_all(seed + rank)
        torch.backends.cuda.matmul.allow_tf32 = allow_tf32
        torch.backends.cudnn.allow_tf32 = allow_tf32

    return DistContext(
        rank=rank,
        local_rank=local_rank,
        world_size=world_size,
        device=device,
        backend=backend,
        hostname=socket.gethostname(),
    )


def _first_hostname(node_list: str) -> str:
    """Best-effort first hostname from ``SLURM_JOB_NODELIST``.

    Prefers ``scontrol show hostnames``, which expands Slurm's bracket notation
    (``nid[001000-001003]``) correctly; falls back to a crude parse if
    ``scontrol`` is unavailable.
    """
    import subprocess

    try:
        out = subprocess.run(
            ["scontrol", "show", "hostnames", node_list],
            capture_output=True,
            text=True,
            check=True,
            timeout=10,
        )
        first = out.stdout.split()
        if first:
            return first[0]
    except Exception:
        pass

    head = node_list.split(",")[0]
    if "[" in head:
        prefix, _, rest = head.partition("[")
        return prefix + rest.split("-")[0].rstrip("]")
    return head


def cleanup_distributed() -> None:
    if dist.is_initialized():
        dist.destroy_process_group()
