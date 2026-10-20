"""Profile one ablation training step on real JetClass batches.

Named ``profile_step`` and not ``profile``: a module called ``profile`` inside a
directory that can land on ``sys.path`` **shadows the stdlib ``profile``**, which
``cProfile`` imports, which ``torch._dynamo`` imports. That broke every direct script
invocation from this directory (e.g. ``python ablation/rank_audit.py``) with a spurious
"weaver-core not pinned" error. Renamed 2026-08-30. Do not rename it back.

Wraps a short forward+backward loop in ``torch.profiler`` and writes a Chrome
trace plus a top-op table.  Use this to see whether ``PairEmbed`` (the
``O(P^2)`` pair pipeline) or attention dominates before enabling
``part_kernels``.

Usage (single GPU on a compute node)::

    cd ml4sci_26
    module load pytorch && source ../.venv_train/bin/activate
    export PYTHONPATH=$PWD/..:$PWD
    export JETCLASS_PT_DIR=$PSCRATCH/jetclass/pt_ragged

    python -m ablation.profile_step --config ablation/configs/smoke.yaml \\
        --set arm=baseline \\
        --set train_pt_dir=$JETCLASS_PT_DIR/train_100M \\
        --set val_pt_dir=$JETCLASS_PT_DIR/val_5M \\
        --set use_part_kernels=true

    # open trace in chrome://tracing
    open profile_trace.json
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ablation.config import load_config  # noqa: E402
from ablation.data import build_loaders  # noqa: E402
from ablation.distributed import cleanup_distributed, setup_distributed  # noqa: E402
from ablation.train import _apply_part_kernels  # noqa: E402
from variants import build_variant_part, collect_moe_aux_loss  # noqa: E402


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Profile one training step.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--set", action="append", default=[], metavar="KEY=VALUE")
    parser.add_argument("--steps", type=int, default=20, help="warmup + profile steps")
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument(
        "--trace",
        default="profile_trace.json",
        help="Chrome trace output path (rank 0 only)",
    )
    args = parser.parse_args(argv)

    config = load_config(args.config, args.set)
    ctx = setup_distributed()
    device = ctx.device

    # build_loaders always constructs train + val; val_pt_dir must be set even
    # though this profiler only consumes one training batch.
    train_loader, _, _ = build_loaders(config, ctx)
    batch = next(iter(train_loader))
    x, v, mask, y = batch
    x = x.to(device, non_blocking=True)
    v = v.to(device, non_blocking=True)
    mask = mask.to(device, non_blocking=True)
    targets = y.argmax(dim=1).to(device, non_blocking=True)

    model = build_variant_part(config.arm, **config.model_kwargs()).to(device)
    model = _apply_part_kernels(model, config, ctx)
    model.train()

    amp_dtype = {"bf16": torch.bfloat16, "fp16": torch.float16}.get(config.precision)
    scaler = torch.cuda.amp.GradScaler(enabled=config.precision == "fp16")

    def step():
        with torch.autocast(
            device_type=device.type,
            dtype=amp_dtype,
            enabled=amp_dtype is not None,
        ):
            logits = model(x, v=v, mask=mask)
            loss = F.cross_entropy(logits, targets)
            if config.arm == "moe":
                loss = loss + collect_moe_aux_loss(model)
        scaler.scale(loss).backward()
        model.zero_grad(set_to_none=True)

    for _ in range(args.warmup):
        step()
    torch.cuda.synchronize()

    activities = [torch.profiler.ProfilerActivity.CPU]
    if device.type == "cuda":
        activities.append(torch.profiler.ProfilerActivity.CUDA)

    with torch.profiler.profile(
        activities=activities,
        record_shapes=True,
        profile_memory=True,
        with_stack=True,
    ) as prof:
        for _ in range(args.steps):
            step()
        torch.cuda.synchronize()

    if ctx.is_main:
        table = prof.key_averages().table(sort_by="cuda_time_total", row_limit=25)
        print(table)
        prof.export_chrome_trace(args.trace)
        print(f"wrote {args.trace}", flush=True)

    cleanup_distributed()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
