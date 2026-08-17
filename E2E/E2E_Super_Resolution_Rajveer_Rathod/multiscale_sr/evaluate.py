"""Standalone evaluation for a trained multi-scale SR checkpoint.

Loads a checkpoint, rebuilds the held-out **test** loader at the checkpoint's
scale, and reports L1 (normalized), PSNR (normalized), and energy response.
Optionally writes a fresh sample grid.

"test" is the row-half of the held-out file(s) never used for training or for
best.pt checkpoint selection (that's "val" — see train.py). --val-ratio must
match the value the checkpoint was trained with so "test" refers to the same
held-out file group.

Usage:
    python evaluate.py --checkpoint experiments/<run>/checkpoints/best.pt \
        --data-dir ../datasets
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from multiscale_sr.data import get_dataloader
from multiscale_sr.data.normalization import ChannelStats
from multiscale_sr.engine import evaluate, render_sample_grid
from multiscale_sr.models import Generator
from multiscale_sr.utils import resolve_env

PACKAGE_ROOT = Path(__file__).resolve().parent


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Evaluate a multi-scale SR checkpoint")
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--data-dir", type=str, required=True)
    p.add_argument("--dataset-format", type=str, default=None, choices=["parquet", "hdf5"])
    p.add_argument("--scale", type=int, default=None, help="Override scale (else read from checkpoint args)")
    p.add_argument("--hr-size", type=int, default=None, help="Override HR size (else from checkpoint args)")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--val-ratio", type=float, default=0.33,
                   help="Must match the checkpoint's training --val-ratio; determines "
                        "which held-out file group 'test' is drawn from")
    p.add_argument("--max-val-batches", type=int, default=None)
    p.add_argument("--save-grid", type=str, default=None, help="Path to write a sample PNG")
    return p


def main() -> None:
    args = build_parser().parse_args()
    env = resolve_env()
    print(f"[env] {env}")

    ckpt = torch.load(args.checkpoint, map_location=env.device)
    ckpt_args = ckpt.get("args", {})
    scale = args.scale or ckpt_args.get("scale")
    hr_size = args.hr_size or ckpt_args.get("hr_size", 125)
    if scale is None:
        raise SystemExit("scale not found in checkpoint; pass --scale explicitly.")

    stats = ChannelStats.from_dict(ckpt["stats"])
    gen = Generator(
        base_channels=ckpt_args.get("gen_channels", 64),
        num_blocks=ckpt_args.get("gen_blocks", 8),
    ).to(env.device)
    gen.load_state_dict(ckpt["generator"])

    # Stats come from the checkpoint, so skip recompute by seeding the cache.
    cache = Path(args.checkpoint).parent.parent / "normalization.json"
    if not cache.exists():
        stats.save(cache)

    test_loader, _ = get_dataloader(
        path=Path(args.data_dir), split="test", env=env, batch_size=args.batch_size,
        scale=scale, hr_size=hr_size, dataset_type=args.dataset_format,
        val_ratio=args.val_ratio, stats_cache_path=cache,
    )

    metrics = evaluate(gen, test_loader, stats, env.device, max_batches=args.max_val_batches)
    metrics["scale"] = scale
    print(json.dumps(metrics, indent=2))

    if args.save_grid:
        batch = next(iter(test_loader))
        render_sample_grid(gen, batch, stats, env.device, Path(args.save_grid))
        print(f"[grid] saved to {args.save_grid}")


if __name__ == "__main__":
    main()
