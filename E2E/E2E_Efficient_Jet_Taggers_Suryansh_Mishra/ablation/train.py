"""Training entrypoint for one ParT ablation arm.

Step-based DDP trainer. One process per GPU; the topology comes from Slurm on
Perlmutter (see :mod:`ablation.distributed`).

Usage
-----
Single process (smoke test)::

    python -m ablation.train --config configs/smoke.yaml --set arm=lloca

Under Slurm, one task per GPU::

    srun python -m ablation.train --config configs/baseline.yaml

Every run writes to ``<output_dir>/<run_name>/``:

===================  ========================================================
``config.json``      the fully resolved config, for reproducibility
``provenance.json``  git commit / branch / dirty flag at job start
``metrics.jsonl``    one JSON record per logged train step and per evaluation
``predictions/``     ``step_XXXXXXX.npz`` per eval (fp16 probs + labels)
``last.pt``          rolling checkpoint (resumable: model, optimizer, step)
``best.pt``          checkpoint at the best validation accuracy so far
===================  ========================================================
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from contextlib import nullcontext as _nullcontext
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel

# Allow `python -m ablation.train` from the package root without installation.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ablation.config import (  # noqa: E402
    AblationConfig,
    assert_checkpoint_architecture,
    load_config,
)
from ablation.data import InfiniteLoader, build_loaders  # noqa: E402
from ablation.distributed import (  # noqa: E402
    cleanup_distributed,
    setup_distributed,
)
from ablation.metrics import format_summary, summarize  # noqa: E402
from ablation.provenance import (  # noqa: E402
    git_provenance,
    provenance_json,
    save_eval_predictions,
)
from ablation.schedule import build_optimizer, build_scheduler  # noqa: E402
from variants import build_variant_part, collect_moe_aux_loss  # noqa: E402
from variants.tied import assert_shared_blocks_consistent  # noqa: E402

_AMP_DTYPES = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": None}


class Logger:
    """Rank-aware logger: prints to stdout and appends JSON lines on rank 0."""

    def __init__(self, path: Optional[Path], enabled: bool):
        self.enabled = enabled
        self.path = path
        if enabled and path is not None:
            path.parent.mkdir(parents=True, exist_ok=True)
            self.handle = path.open("a", buffering=1)
        else:
            self.handle = None

    def info(self, message: str) -> None:
        if self.enabled:
            print(message, flush=True)

    def record(self, payload: dict) -> None:
        if self.handle is not None:
            self.handle.write(json.dumps(payload, default=_jsonable) + "\n")

    def close(self) -> None:
        if self.handle is not None:
            self.handle.close()


def _jsonable(value):
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return str(value)
    return str(value)


def _part_kernel_target(model: torch.nn.Module) -> torch.nn.Module:
    """Inner weaver ``ParticleTransformer`` (LLoCa wraps it as ``.part``)."""
    return model.part if hasattr(model, "part") else model


def _apply_part_kernels(model: torch.nn.Module, config: AblationConfig, ctx) -> torch.nn.Module:
    if not config.use_part_kernels:
        return model
    from ml4sci_26.part_kernels import optimize_part_model

    target = _part_kernel_target(model)
    _, stats = optimize_part_model(
        target,
        use_attention_patch=config.part_kernels_attention,
    )
    if ctx.is_main:
        print(f"part_kernels active: {stats}", flush=True)
    return model


def build_model(config: AblationConfig, ctx) -> torch.nn.Module:
    """Instantiate the arm and move it onto this rank's device."""
    model = build_variant_part(config.arm, **config.model_kwargs())
    model = model.to(ctx.device)
    if config.arm == "lloca" and config.norm_stats_path:
        from ablation.data import NormStats

        stats = NormStats.load(Path(config.norm_stats_path))
        scale, offset = stats.lloca_kinematic_affine()
        model.set_kinematic_norm(scale, offset)
    model = _apply_part_kernels(model, config, ctx)
    if config.compile_model:
        model = torch.compile(model)
    return model


def _unpack(batch, device):
    """Move one loader 4-tuple ``(x, v, mask, y)`` to the device."""
    x, v, mask, y = batch
    return (
        x.to(device, non_blocking=True),
        v.to(device, non_blocking=True),
        mask.to(device, non_blocking=True),
        y.to(device, non_blocking=True),
    )


@torch.no_grad()
def evaluate(model, loader, ctx, config) -> tuple[dict, np.ndarray, np.ndarray]:
    """Run validation; return metrics plus pooled probs/labels for archival."""
    model.eval()
    amp_dtype = _AMP_DTYPES[config.precision]

    probs_chunks, label_chunks = [], []
    for batch in loader:
        x, v, mask, y = _unpack(batch, ctx.device)
        with torch.autocast(
            device_type=ctx.device.type,
            dtype=amp_dtype,
            enabled=amp_dtype is not None,
        ):
            logits = model(x, v=v, mask=mask)
        # float32 before softmax: bf16 probabilities are too coarse for a
        # rejection metric read off the far tail of the score distribution.
        probs_chunks.append(torch.softmax(logits.float(), dim=1).cpu())
        label_chunks.append(y.argmax(dim=1).cpu() if y.dim() > 1 else y.cpu())

    if probs_chunks:
        # Stay in float32 through the gather and the metrics. Until 2026-09-09 this cast to fp16
        # to shrink the all-gather; fp16 has ~3 significant digits, which rounds distinct scores
        # into tied groups exactly where Rej99/Rej99.5 are read (audit A11). The full 5M-jet
        # pool is 5M x 10 x 4 B = 200 MB, split across ranks -- affordable once per eval.
        local_probs = torch.cat(probs_chunks).numpy().astype(np.float32, copy=False)
        local_labels = torch.cat(label_chunks).to(torch.int16).numpy()
    else:
        local_probs = np.zeros((0, config.num_classes), dtype=np.float32)
        local_labels = np.zeros((0,), dtype=np.int16)

    gathered = ctx.gather_objects((local_probs, local_labels))
    model.train()

    probs = np.concatenate([p for p, _ in gathered]).astype(np.float32)
    labels = np.concatenate([lab for _, lab in gathered]).astype(np.int64)
    # No duplicated jets: ShardCoherentBatchSampler(drop_last=False) emits a
    # partial final batch per shard rather than padding it up to batch_size, so
    # unlike DistributedSampler it never repeats a jet to equalise rank counts.
    # Ranks therefore contribute unequal counts, which is fine — the metrics are
    # computed once on the pooled predictions, not averaged per rank.
    summary = summarize(probs, labels, config.rejection_efficiencies)
    return summary, probs, labels


def save_checkpoint(
    path: Path,
    model,
    optimizer,
    scheduler,
    step: int,
    best: float,
    epoch: int,
    config: AblationConfig,
    *,
    best_step: int = 0,
    best_summary: Optional[dict] = None,
    batches_in_epoch: int = 0,
    scaler=None,
) -> None:
    module = model.module if isinstance(model, DistributedDataParallel) else model
    tmp = path.with_suffix(path.suffix + ".tmp")
    payload = {
            "step": step,
            "best_accuracy": best,
            "best_step": int(best_step),
            "best_summary": best_summary or {},
            # Loader epoch, needed to resume the data order. The coherent
            # sampler seeds its shuffle on (epoch, rank), so resuming without
            # it replays epoch 0's shard order and permutation on every link
            # of a chained Slurm job.
            "epoch": epoch,
            # Batches already consumed in this epoch; the sampler skips them
            # on resume so mid-epoch data is not replayed.
            "batches_in_epoch": int(batches_in_epoch),
            "model": module.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "config": config.to_dict(),
        }
    if scaler is not None and scaler.is_enabled():
        payload["scaler"] = scaler.state_dict()
    torch.save(
        payload,
        tmp,
    )
    # Atomic replace: a job killed mid-write must not leave a corrupt checkpoint
    # that makes the next restart fail instead of resuming.
    os.replace(tmp, path)


def load_checkpoint(path: Path, model, optimizer, scheduler, ctx, scaler=None, config=None):
    module = model.module if isinstance(model, DistributedDataParallel) else model
    payload = torch.load(path, map_location=ctx.device, weights_only=False)
    if config is not None:
        # Before the weights are touched: a resume pointed at a checkpoint from a different
        # architecture must fail loudly. For tied arms `load_state_dict` cannot catch it -- the
        # key names and shapes are identical to an untied stack's, so the load "succeeds" and
        # keeps only the last duplicate of each shared block.
        assert_checkpoint_architecture(payload, config, path=path)
    module.load_state_dict(payload["model"])
    assert_shared_blocks_consistent(module, payload["model"], source=str(path))
    optimizer.load_state_dict(payload["optimizer"])
    scheduler.load_state_dict(payload["scheduler"])
    if scaler is not None and "scaler" in payload:
        scaler.load_state_dict(payload["scaler"])
    extra = {
        "best_step": int(payload.get("best_step", 0)),
        "best_summary": payload.get("best_summary") or {},
        "batches_in_epoch": int(payload.get("batches_in_epoch", 0)),
    }
    # ``epoch`` defaults to 0 for checkpoints written before it was recorded.
    return (
        int(payload["step"]),
        float(payload.get("best_accuracy", 0.0)),
        int(payload.get("epoch", 0)),
        extra,
    )


def train(config: AblationConfig) -> dict:
    ctx = setup_distributed(seed=config.seed, allow_tf32=config.allow_tf32)
    run_dir = config.run_dir
    logger = Logger(run_dir / "metrics.jsonl", ctx.is_main)
    provenance = git_provenance()

    if ctx.is_main:
        run_dir.mkdir(parents=True, exist_ok=True)
        config.save(run_dir / "config.json")
        (run_dir / "provenance.json").write_text(provenance_json(provenance))
        git_line = provenance.get("git_commit_short") or "unknown"
        if provenance.get("git_dirty"):
            git_line += "-dirty"
        logger.info(
            f"experiment={config.experiment}  arm={config.arm}  "
            f"git={git_line}  world_size={ctx.world_size}  "
            f"host={ctx.hostname}  device={ctx.device}  "
            f"precision={config.precision}"
        )

    model = build_model(config, ctx)
    num_params = sum(p.numel() for p in model.parameters())
    if ctx.is_main:
        global_batch = config.batch_size * ctx.world_size * config.grad_accum_steps
        logger.info(f"params={num_params:,}  global_batch={global_batch}")
        logger.record(
            {
                "event": "start",
                "experiment": config.experiment,
                "arm": config.arm,
                "params": num_params,
                "world_size": ctx.world_size,
                "global_batch": global_batch,
                **provenance,
            }
        )

    if ctx.is_distributed:
        model = DistributedDataParallel(
            model,
            device_ids=[ctx.device.index] if ctx.device.type == "cuda" else None,
            # The MoE arm leaves un-selected experts without gradient on any
            # given step, which DDP's default gradient-readiness check rejects.
            find_unused_parameters=config.arm == "moe",
        )

    train_loader, val_loader, train_batch_sampler = build_loaders(config, ctx)
    steps = InfiniteLoader(train_loader, train_batch_sampler)

    optimizer = build_optimizer(model.parameters(), config)
    scheduler = build_scheduler(optimizer, config)

    amp_dtype = _AMP_DTYPES[config.precision]
    # GradScaler is only needed for fp16; bf16 has fp32's exponent range.
    scaler = torch.amp.GradScaler(
        ctx.device.type, enabled=config.precision == "fp16"
    )

    start_step, best_accuracy, start_epoch = 0, 0.0, 0
    best_step = 0
    best_summary: dict = {}
    resume_path = Path(config.resume) if config.resume else run_dir / "last.pt"
    if resume_path.exists():
        start_step, best_accuracy, start_epoch, extra = load_checkpoint(
            resume_path, model, optimizer, scheduler, ctx, scaler=scaler, config=config
        )
        best_step = extra["best_step"]
        best_summary = extra["best_summary"]
        # Restore the loader epoch and the already-consumed prefix, otherwise
        # every chained job replays this epoch from batch 0.
        steps.set_epoch(start_epoch, batches_in_epoch=extra["batches_in_epoch"])
        if ctx.is_main:
            logger.info(
                f"resumed from {resume_path} at step {start_step} "
                f"(loader epoch {start_epoch}, "
                f"skip {extra['batches_in_epoch']} batches)"
            )

    stop_step = config.total_steps
    if config.max_steps is not None:
        stop_step = min(stop_step, start_step + config.max_steps)

    model.train()
    running_loss, running_correct, running_count = 0.0, 0, 0
    running_class_counts = torch.zeros(
        config.num_classes, dtype=torch.long, device=ctx.device
    )
    tick = time.time()
    final_summary: dict = {}

    for step in range(start_step, stop_step):
        optimizer.zero_grad(set_to_none=True)
        accum_loss = 0.0

        for micro in range(config.grad_accum_steps):
            x, v, mask, y = _unpack(next(steps), ctx.device)
            targets = y.argmax(dim=1) if y.dim() > 1 else y
            batch_class_counts = torch.bincount(
                targets, minlength=config.num_classes
            )
            if (
                config.class_balanced_train
                and step == start_step
                and micro == 0
            ):
                spread = int(
                    batch_class_counts.max() - batch_class_counts.min()
                )
                if batch_class_counts.min() == 0 or spread > 1:
                    raise RuntimeError(
                        "class-balanced loader emitted invalid first batch: "
                        f"{batch_class_counts.cpu().tolist()}"
                    )
                if ctx.is_main:
                    logger.info(
                        "first train batch class counts: "
                        f"{batch_class_counts.cpu().tolist()}"
                    )
            running_class_counts += batch_class_counts

            # Suppress the gradient all-reduce on every micro-batch but the
            # last; otherwise accumulation costs one full sync per micro-step.
            is_last_micro = micro == config.grad_accum_steps - 1
            sync_context = (
                model.no_sync()
                if (ctx.is_distributed and not is_last_micro)
                else _nullcontext()
            )

            with sync_context:
                with torch.autocast(
                    device_type=ctx.device.type,
                    dtype=amp_dtype,
                    enabled=amp_dtype is not None,
                ):
                    logits = model(x, v=v, mask=mask)
                    # Cross-entropy in fp32: bf16 logits lose accuracy in the
                    # log-sum-exp, and the loss is cheap relative to the model.
                    loss = F.cross_entropy(logits.float(), targets)
                    if config.arm == "moe" and config.moe_aux_alpha > 0:
                        aux = collect_moe_aux_loss(
                            model.module
                            if isinstance(model, DistributedDataParallel)
                            else model
                        )
                        loss = loss + config.moe_aux_alpha * aux.to(loss.dtype)

                scaled = loss / config.grad_accum_steps
                if scaler.is_enabled():
                    scaler.scale(scaled).backward()
                else:
                    scaled.backward()

            accum_loss += loss.detach()
            running_correct += (logits.detach().argmax(dim=1) == targets).sum().item()
            running_count += targets.numel()

        grad_norm = None
        if config.grad_clip > 0:
            if scaler.is_enabled():
                scaler.unscale_(optimizer)
            # KEEP the return value. `clip_grad_norm_` returns the PRE-clip total norm, and
            # discarding it was leaving a 14 A100-hour run with one scalar and no way to attribute a
            # shortfall. This is the single trace that separates "the architecture underperformed"
            # from "the optimisation went wrong" -- e.g. a tied arm whose residual stream grows too
            # fast shows up here as a rising pre-clip norm long before accuracy moves.
            grad_norm = float(
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
            )

        if scaler.is_enabled():
            scaler.step(optimizer)
            scaler.update()
        else:
            optimizer.step()
        scheduler.step()

        running_loss += float(accum_loss) / config.grad_accum_steps

        if (step + 1) % config.log_every == 0:
            window = config.log_every
            elapsed = time.time() - tick
            jets = (
                window
                * config.batch_size
                * config.grad_accum_steps
                * ctx.world_size
            )
            payload = {
                "event": "train",
                "step": step + 1,
                "loss": running_loss / window,
                "accuracy": running_correct / max(1, running_count),
                "num_jets": running_count,
                "class_counts": running_class_counts.cpu().tolist(),
                "log_every_steps": window,
                "lr": optimizer.param_groups[0]["lr"],
                "jets_per_sec": jets / max(elapsed, 1e-9),
                # Pre-clip global gradient norm. With grad_clip=1.0 the clip factor is
                # 1.0 / grad_norm, so this also records how hard clipping is biting.
                "grad_norm": grad_norm,
                "epoch": steps.epoch,
            }
            if ctx.is_main:
                logger.info(
                    f"step {step + 1:>8}/{config.total_steps}  "
                    f"loss={payload['loss']:.4f}  "
                    f"acc={payload['accuracy']:.4f}  "
                    f"class_range={min(payload['class_counts'])}-"
                    f"{max(payload['class_counts'])}  "
                    f"lr={payload['lr']:.3e}  "
                    f"{payload['jets_per_sec']:.0f} jets/s"
                )
                logger.record(payload)
            running_loss, running_correct, running_count = 0.0, 0, 0
            running_class_counts.zero_()
            tick = time.time()

        is_last = step + 1 == stop_step
        if (step + 1) % config.eval_every == 0 or is_last:
            summary, probs, labels = evaluate(model, val_loader, ctx, config)
            summary.update(event="eval", step=step + 1)
            final_summary = summary
            if ctx.is_main:
                logger.info(f"[eval @ {step + 1}] {format_summary(summary)}")
                logger.record(summary)
                if config.save_eval_predictions:
                    save_eval_predictions(run_dir, step + 1, probs, labels)
                if summary["accuracy"] > best_accuracy:
                    best_accuracy = summary["accuracy"]
                    best_step = step + 1
                    best_summary = dict(summary)
                    save_checkpoint(
                        run_dir / "best.pt", model, optimizer, scheduler,
                        step + 1, best_accuracy, steps.epoch, config,
                        best_step=best_step, best_summary=best_summary,
                        batches_in_epoch=steps.batches_in_epoch, scaler=scaler,
                    )
            tick = time.time()

        if ((step + 1) % config.checkpoint_every == 0 or is_last) and ctx.is_main:
            save_checkpoint(
                run_dir / "last.pt", model, optimizer, scheduler,
                step + 1, best_accuracy, steps.epoch, config,
                best_step=best_step, best_summary=best_summary,
                batches_in_epoch=steps.batches_in_epoch, scaler=scaler,
            )

    ctx.barrier()
    if ctx.is_main:
        logger.info(
            f"done: best val accuracy {best_accuracy:.4f} at step {best_step}"
        )
        logger.record(
            {
                "event": "finish",
                "experiment": config.experiment,
                "step": stop_step,
                "best_step": best_step,
                "best_accuracy": best_accuracy,
                "best_eval": best_summary,
                "final_eval": final_summary,
                **provenance,
            }
        )
    logger.close()
    cleanup_distributed()
    return final_summary


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="Train one arm of the ParT ablation."
    )
    parser.add_argument("--config", help="YAML config file")
    parser.add_argument(
        "--set",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="override a config field; repeatable",
    )
    args = parser.parse_args(argv)

    config = load_config(args.config, args.set)
    train(config)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
