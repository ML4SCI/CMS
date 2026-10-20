"""Compute per-feature normalization statistics from ragged CSR shards.

Samples N shards, computes mean and std for each of the 16 particle features,
and writes a ``norm_stats.json`` file.

Usage::

    python -m preprocessing.compute_norm_stats \\
        --pt-dir $PSCRATCH/jetclass/pt_ragged/train_100M \\
        --output norm_stats.json \\
        --num-shards 10

The output JSON has the shape::

    {
        "mean": [m0, m1, ..., m15],
        "std":  [s0, s1, ..., s15],
        "num_particles": 12345678,
        "num_shards": 10
    }

Use with the ablation config::

    norm_stats_path: /path/to/norm_stats.json

The trainer applies the normalization on-the-fly; the ``.pt`` shards remain raw.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import torch


def compute_norm_stats(
    pt_dir: str,
    num_shards: Optional[int] = None,
    seed: int = 42,
) -> dict:
    """Compute per-feature mean and std from ragged CSR shards.

    Parameters
    ----------
    pt_dir : str
        Directory containing ``.pt`` shard files.
    num_shards : int, optional
        Number of shards to sample.  ``None`` = all.
    seed : int
        Seed for shard sampling (for reproducibility when subsampling).

    Returns
    -------
    dict
        ``{"mean": list[float], "std": list[float],
          "num_particles": int, "num_shards": int}``
    """
    files = sorted(
        os.path.join(pt_dir, f)
        for f in os.listdir(pt_dir)
        if f.endswith(".pt")
    )
    if not files:
        raise FileNotFoundError(f"No .pt files found in {pt_dir}")

    if num_shards is not None and num_shards < len(files):
        rng = np.random.default_rng(seed)
        indices = rng.choice(len(files), size=num_shards, replace=False)
        files = [files[i] for i in sorted(indices)]

    # Welford's online algorithm for numerically stable mean/variance,
    # vectorised over particles: each chunk contributes (n, mean, M2) via
    # blockwise numpy ops, and chunks are merged with Chan's parallel formula
    # (algebraically identical to sequential Welford, but ~10^3x faster than
    # a per-particle Python loop — 10 shards x ~4M particles on Perlmutter).
    n_total = 0
    mean = np.zeros(16, dtype=np.float64)
    m2 = np.zeros(16, dtype=np.float64)

    chunk_rows = 1_000_000  # ~128 MB float64 at 16 features; bounds peak RSS
    for i, path in enumerate(files):
        print(f"  [{i + 1}/{len(files)}] {Path(path).name}", flush=True)
        data = torch.load(path, map_location="cpu", weights_only=True)
        x = data["x"].numpy()  # (N_particles, 16) float32

        for start in range(0, x.shape[0], chunk_rows):
            block = x[start : start + chunk_rows].astype(np.float64)
            n_b = block.shape[0]
            mean_b = block.mean(axis=0)
            m2_b = ((block - mean_b) ** 2).sum(axis=0)

            if n_total == 0:
                n_total, mean, m2 = n_b, mean_b, m2_b
            else:
                n_new = n_total + n_b
                delta = mean_b - mean
                mean = mean + delta * (n_b / n_new)
                m2 = m2 + m2_b + delta * delta * (n_total * n_b / n_new)
                n_total = n_new

    if n_total < 2:
        raise ValueError(
            f"Need at least 2 particles to compute stats; got {n_total}"
        )

    variance = m2 / (n_total - 1)
    std = np.sqrt(variance)
    # Prevent division by zero for constant features
    std = np.where(std > 0, std, 1.0)

    return {
        "mean": mean.tolist(),
        "std": std.tolist(),
        "num_particles": int(n_total),
        "num_shards": len(files),
    }


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Compute per-feature normalization statistics from ragged CSR shards."
    )
    parser.add_argument(
        "--pt-dir",
        required=True,
        help="Directory containing .pt shard files (training split).",
    )
    parser.add_argument(
        "--output",
        default="norm_stats.json",
        help="Output path for the JSON file (default: norm_stats.json).",
    )
    parser.add_argument(
        "--num-shards",
        type=int,
        default=None,
        help="Number of shards to sample (default: all).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for shard sampling.",
    )
    args = parser.parse_args(argv)

    print(f"Computing norm stats from {args.pt_dir}")
    stats = compute_norm_stats(args.pt_dir, args.num_shards, args.seed)

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(stats, indent=2))
    print(f"\nWrote {out}")
    print(f"  particles sampled: {stats['num_particles']:,}")
    print(f"  shards used:       {stats['num_shards']}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
