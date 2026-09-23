"""
Feature: ParT ablation -- distributed setup.

Pins the collective timeout handed to ``torch.distributed.init_process_group``.
PyTorch's NCCL default is 10 minutes. tied_k1_1M (Slurm 58152351) died at step
425k on 2026-09-13 with a NCCL all-reduce timeout: rank 0 alone scores the 4.9M
validation jets and writes the ~200 MB fp32 prediction archive after every eval
while the other ranks are already parked at the next gradient all-reduce, and one
slow-scratch stall past 10 minutes took the whole job down. The trainer therefore
asks for a longer timeout explicitly; this test fails if that argument is ever
dropped or shortened again.

``init_process_group`` is monkeypatched so no process group is created: the test
only checks what ``setup_distributed`` asks for.
"""

from __future__ import annotations

from datetime import timedelta

import pytest
import torch

from ablation import distributed as dist_mod
from ablation.distributed import COLLECTIVE_TIMEOUT, setup_distributed

_SLURM_ENV = {
    "SLURM_PROCID": "1",
    "SLURM_LOCALID": "1",
    "SLURM_NTASKS": "4",
    "SLURM_JOB_NODELIST": "nid001000",
    "MASTER_ADDR": "127.0.0.1",
    "MASTER_PORT": "29999",
}


def test_collective_timeout_covers_the_observed_post_eval_stall():
    # 10 minutes is what killed 58152351; anything at or below it re-opens the failure.
    assert COLLECTIVE_TIMEOUT > timedelta(minutes=10)
    assert COLLECTIVE_TIMEOUT >= timedelta(minutes=30)


def test_setup_distributed_passes_the_timeout_to_init_process_group(monkeypatch):
    for key, value in _SLURM_ENV.items():
        monkeypatch.setenv(key, value)
    # Force the CPU/gloo path so the test runs on any machine.
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

    seen: dict = {}

    def fake_init(*args, **kwargs):
        seen.update(kwargs)

    monkeypatch.setattr(dist_mod.dist, "init_process_group", fake_init)
    monkeypatch.setattr(dist_mod.dist, "is_initialized", lambda: False)

    ctx = setup_distributed(seed=0)

    assert ctx.world_size == 4 and ctx.rank == 1 and ctx.backend == "gloo"
    assert seen["backend"] == "gloo"
    assert seen["world_size"] == 4 and seen["rank"] == 1
    assert isinstance(seen.get("timeout"), timedelta)
    assert seen["timeout"] == COLLECTIVE_TIMEOUT


def test_single_process_does_not_touch_the_process_group(monkeypatch):
    for key in _SLURM_ENV:
        monkeypatch.delenv(key, raising=False)
    for key in ("RANK", "LOCAL_RANK", "WORLD_SIZE"):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

    def fail(*args, **kwargs):  # pragma: no cover - only reached on regression
        pytest.fail("init_process_group must not be called for world_size == 1")

    monkeypatch.setattr(dist_mod.dist, "init_process_group", fail)

    ctx = setup_distributed(seed=0)
    assert ctx.world_size == 1 and not ctx.is_distributed
