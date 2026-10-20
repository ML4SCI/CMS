"""
Benchmark: flat shuffle vs shard-coherent ragged DataLoader.

Modes
-----
  A) ``flat``     — standard ``DataLoader(shuffle=True)`` + ``ragged_collate``
  B) ``coherent`` — ``ShardCoherentBatchSampler`` + ``get_batch``
  C) ``cuda``     — (B) + ``.to(device, non_blocking=True)``

Metrics: jets/s, batches/s, mean ``P_max``, peak RSS (MB).

Single process (laptop / login node)::

    python preprocessing/bench_ragged_loader.py \\
      --pt-dir /path/to/jetclass/pt_ragged/train_100M \\
      --modes flat coherent --num-batches 200

Multi-process (Slurm, via ``srun``)::

    srun ... python preprocessing/bench_ragged_loader.py \\
      --pt-dir ... --modes coherent cuda --num-batches 200

The script detects DDP topology from Slurm environment variables via
``ablation.distributed.setup_distributed``.
"""

from __future__ import annotations

import argparse
import os
import resource
import sys
import time
from typing import List, Optional, Tuple

import numpy as np

# Ensure parent package dirs are importable regardless of CWD.
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PACKAGE_DIR = os.path.dirname(_SCRIPT_DIR)
if _PACKAGE_DIR not in sys.path:
    sys.path.insert(0, _PACKAGE_DIR)

import torch

from dataloader.ragged_loader import (
    RaggedShardDataset,
    ShardCoherentBatchSampler,
    create_ragged_dataloader,
    create_ragged_train_loader,
    ragged_collate,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _peak_rss_mb() -> float:
    """Peak resident set size in MB (macOS/Linux)."""
    ru = resource.getrusage(resource.RUSAGE_SELF)
    # macOS: bytes; Linux: kilobytes
    if sys.platform == "darwin":
        return ru.ru_maxrss / (1024 * 1024)
    return ru.ru_maxrss / 1024


def _print_rank0(msg: str, rank: int = 0) -> None:
    if rank == 0:
        print(msg, flush=True)


# ---------------------------------------------------------------------------
# Bench modes
# ---------------------------------------------------------------------------


def _bench_flat(
    pt_dir: str,
    batch_size: int,
    num_batches: int,
    num_workers: int,
    rank: int,
) -> dict:
    """Mode A: flat-shuffle baseline."""
    loader = create_ragged_dataloader(
        pt_dir,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=True,
        cache_size=2,
        pin_memory=False,
        drop_last=True,
    )

    # Warm-up
    it = iter(loader)
    try:
        _ = next(it)
    except StopIteration:
        return {"error": "empty loader"}

    p_maxs: List[int] = []
    total_jets = 0
    t0 = time.perf_counter()

    for i, (x, v, mask, y) in enumerate(loader):
        if i >= num_batches:
            break
        total_jets += x.shape[0]
        p_maxs.append(x.shape[2])

    elapsed = time.perf_counter() - t0
    return {
        "mode": "flat",
        "batches": min(i + 1, num_batches),
        "jets": total_jets,
        "elapsed_s": elapsed,
        "jets_per_s": total_jets / elapsed if elapsed > 0 else 0,
        "batches_per_s": min(i + 1, num_batches) / elapsed if elapsed > 0 else 0,
        "mean_p_max": float(np.mean(p_maxs)) if p_maxs else 0,
        "peak_rss_mb": _peak_rss_mb(),
    }


def _bench_coherent(
    pt_dir: str,
    batch_size: int,
    num_batches: int,
    num_workers: int,
    rank: int,
    world_size: int,
    use_cuda: bool = False,
) -> dict:
    """Mode B (coherent) or C (coherent + CUDA)."""
    device = torch.device("cpu")
    if use_cuda:
        if not torch.cuda.is_available():
            return {"error": "CUDA not available"}
        local_rank = int(os.environ.get("SLURM_LOCALID", os.environ.get("LOCAL_RANK", "0")))
        device_idx = local_rank % torch.cuda.device_count()
        device = torch.device("cuda", device_idx)

    loader = create_ragged_train_loader(
        pt_dir,
        batch_size=batch_size,
        rank=rank,
        world_size=world_size,
        num_workers=num_workers,
        cache_size=1,
        pin_memory=use_cuda,
    )

    # Warm-up
    it = iter(loader)
    try:
        warmup = next(it)
        if use_cuda:
            _ = tuple(t.to(device, non_blocking=True) for t in warmup)
            torch.cuda.synchronize(device)
    except StopIteration:
        return {"error": "empty loader"}

    p_maxs: List[int] = []
    total_jets = 0
    t0 = time.perf_counter()

    for i, batch in enumerate(loader):
        if i >= num_batches:
            break
        if use_cuda:
            batch = tuple(t.to(device, non_blocking=True) for t in batch)
        x = batch[0]
        total_jets += x.shape[0]
        p_maxs.append(x.shape[2])

    if use_cuda:
        torch.cuda.synchronize(device)

    elapsed = time.perf_counter() - t0
    n_batches = min(i + 1, num_batches)
    return {
        "mode": "cuda" if use_cuda else "coherent",
        "batches": n_batches,
        "jets": total_jets,
        "elapsed_s": elapsed,
        "jets_per_s": total_jets / elapsed if elapsed > 0 else 0,
        "batches_per_s": n_batches / elapsed if elapsed > 0 else 0,
        "mean_p_max": float(np.mean(p_maxs)) if p_maxs else 0,
        "peak_rss_mb": _peak_rss_mb(),
    }


# ---------------------------------------------------------------------------
# Multi-process timing aggregation
# ---------------------------------------------------------------------------


def _aggregate_timing(result: dict, rank: int, world_size: int) -> dict:
    """All-reduce timing stats across ranks (requires dist init).

    Raises on failure so that a broken NCCL ring does not produce a
    false-pass on the 2-node smoke test.
    """
    if world_size <= 1:
        return result

    import torch.distributed as dist

    if not dist.is_initialized():
        raise RuntimeError(
            "_aggregate_timing: world_size > 1 but dist is not initialized"
        )

    jets_t = torch.tensor(result["jets"], dtype=torch.float64)
    elapsed_t = torch.tensor(result["elapsed_s"], dtype=torch.float64)

    dist.all_reduce(jets_t, op=dist.ReduceOp.SUM)
    # Use max elapsed for conservative throughput
    dist.all_reduce(elapsed_t, op=dist.ReduceOp.MAX)

    result["total_jets_all_ranks"] = int(jets_t.item())
    result["max_elapsed_s"] = elapsed_t.item()
    result["aggregate_jets_per_s"] = (
        int(jets_t.item()) / elapsed_t.item()
        if elapsed_t.item() > 0
        else 0
    )
    return result


# ---------------------------------------------------------------------------
# Shard partition check (multi-process)
# ---------------------------------------------------------------------------


def _check_shard_overlap(
    loader, rank: int, world_size: int
) -> Optional[str]:
    """Verify no two ranks share the same shard ID in their partition."""
    if world_size <= 1:
        return None

    try:
        import torch.distributed as dist

        if not dist.is_initialized():
            return None

        my_shards = set(loader.batch_sampler.my_shards)
        all_shards = [None] * world_size
        dist.all_gather_object(all_shards, my_shards)

        if rank == 0:
            for i in range(world_size):
                for j in range(i + 1, world_size):
                    overlap = all_shards[i] & all_shards[j]
                    if overlap:
                        return (
                            f"FAIL: rank {i} and rank {j} share shards: "
                            f"{sorted(overlap)[:10]}"
                        )
        return None
    except Exception as e:
        return f"shard overlap check failed: {e}"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Benchmark: flat shuffle vs shard-coherent ragged DataLoader",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--pt-dir",
        required=True,
        help="Directory containing .pt shard files",
    )
    parser.add_argument(
        "--modes",
        nargs="+",
        default=["flat", "coherent"],
        choices=["flat", "coherent", "cuda"],
        help="Benchmark modes to run (default: flat coherent)",
    )
    parser.add_argument(
        "--batch-size", type=int, default=512, help="Batch size (default: 512)"
    )
    parser.add_argument(
        "--num-batches",
        type=int,
        default=200,
        help="Number of batches to time (default: 200)",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=4,
        help="DataLoader workers (default: 4)",
    )
    parser.add_argument(
        "--distributed",
        action="store_true",
        help="Initialize DDP from Slurm/torchrun env (auto-detected from SLURM_NTASKS)",
    )
    args = parser.parse_args(argv)

    # --- DDP setup ---
    rank = 0
    world_size = 1

    # Auto-detect distributed from env
    use_dist = args.distributed or int(
        os.environ.get("SLURM_NTASKS", os.environ.get("WORLD_SIZE", "1"))
    ) > 1

    if use_dist:
        try:
            from ablation.distributed import setup_distributed, cleanup_distributed

            ctx = setup_distributed()
            rank = ctx.rank
            world_size = ctx.world_size
            _print_rank0(
                f"DDP: rank={rank}, world_size={world_size}, "
                f"device={ctx.device}, host={ctx.hostname}",
                rank,
            )
        except Exception as exc:
            # Fail hard: under srun every rank must initialise or the bench
            # silently runs world_size=1 on all tasks (false pass).
            print(
                f"FATAL: distributed setup failed "
                f"(SLURM_NTASKS={os.environ.get('SLURM_NTASKS', '?')}): {exc}",
                flush=True,
            )
            return 1

    # --- Header ---
    _print_rank0("=" * 60, rank)
    _print_rank0(f"Ragged DataLoader Benchmark", rank)
    _print_rank0(f"  pt_dir      : {args.pt_dir}", rank)
    _print_rank0(f"  modes       : {args.modes}", rank)
    _print_rank0(f"  batch_size  : {args.batch_size}", rank)
    _print_rank0(f"  num_batches : {args.num_batches}", rank)
    _print_rank0(f"  num_workers : {args.num_workers}", rank)
    _print_rank0(f"  rank/world  : {rank}/{world_size}", rank)
    _print_rank0("=" * 60, rank)

    results = []

    for mode in args.modes:
        _print_rank0(f"\n--- Mode: {mode} ---", rank)

        if mode == "flat":
            if world_size > 1:
                # Fair comparison: run flat single-process on rank 0 only.
                # Under DDP, all ranks shuffling the full shard set would
                # penalise flat with N-way Lustre contention that coherent
                # avoids by design — mixing sampler quality with I/O noise.
                import torch.distributed as dist

                if rank == 0:
                    _print_rank0(
                        "  (rank 0 only, world_size=1 for fair comparison)",
                        rank,
                    )
                    result = _bench_flat(
                        args.pt_dir,
                        args.batch_size,
                        args.num_batches,
                        args.num_workers,
                        rank,
                    )
                else:
                    result = None
                # Barrier so all ranks wait before the next mode
                if dist.is_initialized():
                    dist.barrier()
                if result is None:
                    continue
            else:
                result = _bench_flat(
                    args.pt_dir,
                    args.batch_size,
                    args.num_batches,
                    args.num_workers,
                    rank,
                )
        elif mode == "coherent":
            result = _bench_coherent(
                args.pt_dir,
                args.batch_size,
                args.num_batches,
                args.num_workers,
                rank,
                world_size,
                use_cuda=False,
            )
        elif mode == "cuda":
            result = _bench_coherent(
                args.pt_dir,
                args.batch_size,
                args.num_batches,
                args.num_workers,
                rank,
                world_size,
                use_cuda=True,
            )
        else:
            continue

        if "error" in result:
            _print_rank0(f"  ERROR: {result['error']}", rank)
            continue

        # Aggregate across ranks if distributed
        if world_size > 1:
            result = _aggregate_timing(result, rank, world_size)

        results.append(result)

        _print_rank0(
            f"  batches     : {result['batches']}\n"
            f"  jets        : {result['jets']}\n"
            f"  elapsed     : {result['elapsed_s']:.2f}s\n"
            f"  jets/s      : {result['jets_per_s']:,.0f}\n"
            f"  batches/s   : {result['batches_per_s']:.1f}\n"
            f"  mean P_max  : {result['mean_p_max']:.1f}\n"
            f"  peak RSS    : {result['peak_rss_mb']:.0f} MB",
            rank,
        )

        if "aggregate_jets_per_s" in result:
            _print_rank0(
                f"  [all ranks] total jets/s : {result['aggregate_jets_per_s']:,.0f}",
                rank,
            )

    # --- Shard partition check (coherent/cuda modes) ---
    if world_size > 1 and any(m in args.modes for m in ("coherent", "cuda")):
        _print_rank0("\n--- Shard partition check ---", rank)
        # Build a loader just for the check
        check_loader = create_ragged_train_loader(
            args.pt_dir,
            batch_size=args.batch_size,
            rank=rank,
            world_size=world_size,
            num_workers=0,
        )
        err = _check_shard_overlap(check_loader, rank, world_size)
        if err:
            _print_rank0(f"  {err}", rank)
        else:
            _print_rank0("  OK: no shard overlap between ranks", rank)

    # --- Summary table ---
    if len(results) >= 2 and rank == 0:
        _print_rank0("\n--- Summary ---", rank)
        _print_rank0(
            f"{'Mode':<12} {'jets/s':>10} {'batches/s':>10} {'P_max':>8} {'RSS MB':>8}",
            rank,
        )
        _print_rank0("-" * 52, rank)
        for r in results:
            _print_rank0(
                f"{r['mode']:<12} {r['jets_per_s']:>10,.0f} "
                f"{r['batches_per_s']:>10.1f} "
                f"{r['mean_p_max']:>8.1f} "
                f"{r['peak_rss_mb']:>8.0f}",
                rank,
            )

        # Speedup
        flat_r = [r for r in results if r["mode"] == "flat"]
        coherent_r = [r for r in results if r["mode"] in ("coherent", "cuda")]
        if flat_r and coherent_r:
            speedup = coherent_r[0]["jets_per_s"] / flat_r[0]["jets_per_s"]
            _print_rank0(
                f"\nSpeedup (coherent vs flat): {speedup:.1f}×",
                rank,
            )
            bar = "PASS ✓" if speedup >= 5.0 else "FAIL ✗ (target: ≥5×)"
            _print_rank0(f"Success bar (N=1): {bar}", rank)

    # --- Cleanup ---
    if use_dist:
        try:
            cleanup_distributed()
        except Exception:
            pass

    _print_rank0("\nDone.", rank)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
