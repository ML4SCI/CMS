"""Train one multi-scale super-resolution GAN (Option A).

A separate model is trained per LR scale: HR is the fixed ground truth and the
LR input is HR area-downsampled to (scale, scale). Compare runs across scales to
quantify how reconstruction quality and energy fidelity degrade as input
resolution drops.

Usage:
    python train.py --data-dir ../datasets --scale 32 --run-name baseline
    python train.py --config configs/scale_16.yaml --data-dir ../datasets
    python train.py --data-dir ../datasets/calochallenge_dataset2 --scale 32

CLI flags override values loaded from --config.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from dataclasses import replace
from itertools import islice
from pathlib import Path

import torch

from multiscale_sr.data import detect_dataset_type, get_dataloader
from multiscale_sr.data.normalization import ChannelStats, denormalize, normalize
from multiscale_sr.engine import (
    collect_tagging_tensors,
    discriminator_loss,
    energy_response,
    evaluate,
    generator_adv_loss,
    physics_loss,
    render_metrics_plot,
    render_sample_grid,
    semd_loss,
    weighted_l1_loss,
)
from multiscale_sr.experiment import load_config, make_experiment_dir, save_config
from multiscale_sr.models import Discriminator, Generator
from multiscale_sr.tagger import JetTagger, eval_tagger_auc
from multiscale_sr.utils import resolve_env, seed_everything
from multiscale_sr.wandb_logger import WandbLogger, load_env
from multiscale_sr.recovery import Diagnostics, save_checkpoint
from multiscale_sr.training_state import capture_rng, restore_rng, validate_resume, adversarial_weight, noise_std as scheduled_noise

PACKAGE_ROOT = Path(__file__).resolve().parent


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Train a multi-scale SR GAN (Option A)")
    p.add_argument("--config", type=str, default=None, help="YAML config; CLI flags override it")
    p.add_argument("--data-dir", type=str, default=None, help="Directory with parquet or HDF5 files")
    p.add_argument("--dataset-format", type=str, default=None, choices=["parquet", "hdf5"],
                   help="Force dataset format (auto-detected if omitted)")
    p.add_argument("--scale", type=int, default=None, help="LR resolution: e.g. 64, 32, 16")
    p.add_argument("--hr-size", type=int, default=128, help="Target HR square size")
    p.add_argument("--use-native-lr", action="store_true",
                   help="Use detector-native LR (parquet only) instead of downsampling HR")
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--d-lr-ratio", type=float, default=0.5, help="D LR = lr * ratio (TTUR)")
    p.add_argument("--gen-channels", type=int, default=64)
    p.add_argument("--gen-blocks", type=int, default=8)
    p.add_argument("--no-lr-skip", action="store_true",
                   help="Disable the generator's bicubic LR global-residual skip")
    p.add_argument("--lambda-l1", type=float, default=10.0)
    p.add_argument("--lambda-physics", type=float, default=10.0)
    p.add_argument("--physics-loss-type", type=str, default="ratio", choices=["ratio", "l2"],
                   help="Energy-conservation physics loss formulation: 'ratio' "
                        "(|sum(E_pred)/sum(E_true) - 1|, equal weight per fractional "
                        "error) or 'l2' ((sum(E_pred)/sum(E_true) - 1)^2, penalizes "
                        "large misses more heavily — see multiscale_sr/engine.py "
                        "physics_loss_ratio/physics_loss_l2)")
    p.add_argument("--lambda-adv", type=float, default=1.0, help="Peak adversarial loss weight")
    p.add_argument("--lambda-semd", type=float, default=0.0,
                   help="Weight on the SEMD geometric loss. Default 0.0 = OFF, which leaves "
                        "the loss identical to before this term existed. val_semd is logged "
                        "either way. Only enable after the metric is shown to correlate with "
                        "tagging efficiency (see semd_correlation.py)")
    p.add_argument("--semd-topk", type=int, default=128,
                   help="Brightest pixels kept per image for SEMD (paper benchmarks use N=125)")
    p.add_argument("--semd-omega-R", type=float, default=1.0,
                   help="Angular scale at which SR/HR energy imbalance is deposited")
    p.add_argument("--semd-beta", type=float, default=1.0,
                   help="SEMD ground-metric exponent: omega_ij = dist_ij ** beta")
    p.add_argument("--semd-chunk", type=int, default=1,
                   help="Samples per SEMD call; cumulative-integral storage scales as chunk * (topk+1)**2")
    p.add_argument("--semd-loss-raw", action="store_true",
                   help="Use UNNORMALIZED SEMD in the loss. Off by default because raw SEMD "
                        "is ~1e8 on this data vs ~2 for L1, so a plausible-looking lambda "
                        "would silently make SEMD the entire objective. Normalized by "
                        "E_tot_HR^2, lambda behaves like the other lambdas")
    p.add_argument("--l1-weighting", type=str, default="energy", choices=["uniform", "energy"],
                   help="Up-weight bright HR pixels in L1 to prevent the 'predict small' collapse")
    p.add_argument("--l1-weight-alpha", type=float, default=5.0,
                   help="Energy-weighting strength for --l1-weighting energy")
    p.add_argument("--real-label", type=float, default=0.9, help="One-sided label smoothing target")
    p.add_argument("--n-critic", type=int, default=1, help="(legacy) Update D once every N G steps")
    # Adversarial stabilization: warm up G on L1+physics, then ramp adv weight,
    # throttle D so it cannot win outright, and add annealed instance noise.
    p.add_argument("--adv-warmup-epochs", type=int, default=3,
                   help="Epochs with L1+physics only (no D updates, adv weight 0)")
    p.add_argument("--adv-ramp-epochs", type=int, default=5,
                   help="Epochs to linearly ramp adv weight 0 -> lambda_adv after warmup")
    p.add_argument("--g-steps-per-d", type=int, default=2,
                   help="Update D once per N generator steps (D throttle)")
    p.add_argument("--d-loss-floor", type=float, default=0.1,
                   help="Skip the D update on a batch if its d_loss is below this floor")
    p.add_argument("--d-input-noise", type=float, default=0.05,
                   help="Std of Gaussian instance noise added to D's HR inputs (annealed to 0)")
    p.add_argument("--val-ratio", type=float, default=0.33)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--run-name", type=str, default="run")
    p.add_argument("--resume", type=str, default=None, help="Checkpoint path to resume from")
    p.add_argument("--run-dir", default=None, help="Explicit unique local run directory")
    p.add_argument("--recovery-dir", default=None, help="Durable per-run directory (e.g. Drive)")
    p.add_argument("--run-fingerprint", default=None)
    p.add_argument("--checkpoint-seconds", type=float, default=300,
                   help="Save and verify recovery checkpoint at a batch boundary every N seconds")
    p.add_argument("--experiments-root", type=str, default=str(PACKAGE_ROOT / "experiments"))
    p.add_argument("--sample-every", type=int, default=5, help="Save a sample grid every N epochs")
    p.add_argument("--log-every", type=int, default=20, help="Log per-step train losses to W&B every N steps")
    # Debug throughput caps.
    p.add_argument("--max-train-batches", type=int, default=None)
    p.add_argument("--max-val-batches", type=int, default=None)
    p.add_argument("--max-stats-batches", type=int, default=None)
    p.add_argument("--cache", action="store_true",
                   help="Use a one-time HR decode cache (memmap) instead of streaming "
                        "parquet each epoch. First run builds it; later runs read from mmap. "
                        "Enables full-dataset epochs at ~minutes/epoch on MPS.")
    p.add_argument("--cache-dir", type=str, default=None,
                   help="Where to store the decode cache (default: <data-dir>/.sr_cache).")
    p.add_argument("--stats-batch-size", type=int, default=128)
    # Tagging efficiency (optional, per-epoch). Requires a pre-trained frozen
    # tagger checkpoint; without it the per-epoch tag-AUC is skipped (zero cost).
    # For the full HR/LR/SR analysis use the standalone tag_efficiency.py.
    p.add_argument("--tagger-ckpt", type=str, default=None,
                   help="Frozen JetTagger checkpoint; logs val_tag_auc_{hr,sr} each eval epoch")
    p.add_argument("--tag-max-samples", type=int, default=2000,
                   help="Max val samples used for per-epoch tagging AUC")
    # W&B.
    p.add_argument("--wandb-project", type=str, default="multiscale-sr")
    p.add_argument("--wandb-entity", type=str, default=None)
    p.add_argument("--no-wandb", action="store_true")
    return p


def resolve_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse CLI, then layer config file values *under* explicit CLI flags."""
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.config:
        cfg = load_config(Path(args.config))
        tokens = sys.argv[1:] if argv is None else argv
        explicit = {a.dest for a in parser._actions
                    if any(t.split("=", 1)[0] in a.option_strings for t in tokens)}
        for key, value in cfg.items():
            dest = key.replace("-", "_")
            if not hasattr(args, dest):
                continue
            # Explicit CLI flags win even if equal to the parser's default.
            if dest not in explicit:
                setattr(args, dest, value)

    if args.scale is None:
        parser.error("--scale is required (via CLI or --config)")
    if args.data_dir is None:
        parser.error("--data-dir is required (via CLI or --config)")
    for name in ("epochs", "batch_size", "log_every", "sample_every", "semd_chunk", "semd_topk", "g_steps_per_d"):
        if getattr(args, name) < 1:
            parser.error(f"{name} must be positive")
    for name in ("max_train_batches", "max_val_batches", "max_stats_batches"):
        if getattr(args, name) is not None and getattr(args, name) < 1:
            parser.error(f"{name} must be positive when set")
    if args.checkpoint_seconds <= 0:
        parser.error("checkpoint-seconds must be positive")
    return args


def train(args: argparse.Namespace) -> None:
    seed_everything(args.seed)
    load_env(PACKAGE_ROOT)

    env = resolve_env()
    # Timed mid-epoch recovery replays a deterministic loader, then restores RNG.
    # Worker/prefetch RNG cannot be reconstructed by this protocol.
    if args.recovery_dir or args.resume:
        if not args.cache:
            raise ValueError("Recoverable training currently requires --cache (parquet map-style loader)")
        env = replace(env, num_workers=0, persistent_workers=False, prefetch_factor=None)
    print(f"[env] {env}")

    data_dir = Path(args.data_dir)
    dataset_type = args.dataset_format or detect_dataset_type(data_dir)
    if (args.recovery_dir or args.resume) and dataset_type != "parquet":
        raise ValueError("Timed recovery currently supports cached parquet only; HDF5 needs its own sampler recovery validation")
    dataset_name = data_dir.name if data_dir.name else dataset_type
    print(f"[data] format={dataset_type} dir={data_dir} scale={args.scale} hr={args.hr_size}")

    paths = make_experiment_dir(
        Path(args.experiments_root), dataset_name, args.scale, args.run_name,
        run_dir=Path(args.run_dir) if args.run_dir else None, resume=bool(args.resume),
    )
    print(f"[exp] {paths.root}")
    recovery_dir = Path(args.recovery_dir) if args.recovery_dir else None
    diagnostics = Diagnostics(paths.root, env.device, recovery_dir).start()
    _ACTIVE_RESOURCES.append(diagnostics)

    config = vars(args).copy()
    config.update({"dataset_type": dataset_type, "device": str(env.device)})
    save_config(paths.config_yaml, config)

    stats_cache = paths.root / "normalization.json"
    resume_checkpoint = None
    if args.resume:
        resume_checkpoint = torch.load(args.resume, map_location="cpu", weights_only=True)
        validate_resume(resume_checkpoint, args)
        # Resume using the original normalization, not another expensive pass.
        ChannelStats.from_dict(resume_checkpoint["stats"]).save(stats_cache)
    use_native = args.use_native_lr and dataset_type == "parquet"
    if args.use_native_lr and not use_native:
        print("[data] --use-native-lr ignored (only valid for parquet); downsampling HR instead.")

    diagnostics.phase("data_and_normalization", seed=args.seed)
    train_loader, stats = get_dataloader(
        path=data_dir, split="train", env=env, batch_size=args.batch_size,
        scale=args.scale, hr_size=args.hr_size, dataset_type=dataset_type,
        use_native_lr=use_native, val_ratio=args.val_ratio,
        max_stats_batches=args.max_stats_batches, stats_batch_size=args.stats_batch_size,
        stats_cache_path=stats_cache,
        use_cache=args.cache, cache_dir=Path(args.cache_dir) if args.cache_dir else None,
    )
    val_loader, _ = get_dataloader(
        path=data_dir, split="val", env=env, batch_size=args.batch_size,
        scale=args.scale, hr_size=args.hr_size, dataset_type=dataset_type,
        use_native_lr=use_native, val_ratio=args.val_ratio,
        stats_cache_path=stats_cache,
        use_cache=args.cache, cache_dir=Path(args.cache_dir) if args.cache_dir else None,
    )

    generator = Generator(
        base_channels=args.gen_channels, num_blocks=args.gen_blocks, lr_skip=not args.no_lr_skip
    ).to(env.device)
    discriminator = Discriminator().to(env.device)
    opt_g = torch.optim.Adam(generator.parameters(), lr=args.lr, betas=(0.5, 0.999))
    opt_d = torch.optim.Adam(discriminator.parameters(), lr=args.lr * args.d_lr_ratio, betas=(0.5, 0.999))

    start_epoch = 1
    best_val = math.inf
    resume_state = None
    if args.resume:
        ckpt = resume_checkpoint
        generator.load_state_dict(ckpt["generator"], strict=True)
        discriminator.load_state_dict(ckpt["discriminator"])
        opt_g.load_state_dict(ckpt["optimizer_g"])
        opt_d.load_state_dict(ckpt["optimizer_d"])
        start_epoch = ckpt["epoch"] + int(ckpt["epoch_complete"])
        best_val = ckpt.get("best_val", math.inf)
        resume_state = ckpt
        # A new local session may resume after the final epoch just to finish
        # evaluation/persistence; in that case the loop will write no checkpoint.
        save_checkpoint(ckpt, paths.latest_ckpt)
        print(f"[resume] from {args.resume} at epoch {start_epoch}")

    # Optional frozen tagger for per-epoch tagging efficiency.
    frozen_tagger: JetTagger | None = None
    if args.tagger_ckpt:
        tck = torch.load(args.tagger_ckpt, map_location=env.device)
        state = tck["tagger"] if isinstance(tck, dict) and "tagger" in tck else tck
        width = tck.get("width", 32) if isinstance(tck, dict) else 32
        frozen_tagger = JetTagger(width=width).to(env.device)
        frozen_tagger.load_state_dict(state)
        frozen_tagger.eval()
        print(f"[tag] loaded frozen tagger from {args.tagger_ckpt}; "
              f"logging val_tag_auc_{{hr,sr}} each eval epoch")

    wb = WandbLogger(
        enabled=not args.no_wandb, project=args.wandb_project, entity=args.wandb_entity,
        run_name=f"{dataset_name}_{args.scale}x_{args.run_name}", config=config, dir=paths.root,
    )
    _ACTIVE_RESOURCES.append(wb)

    def _adv_weight(epoch: int) -> float:
        """0 during warmup, linear ramp to lambda_adv over adv_ramp_epochs, then flat."""
        return adversarial_weight(args, epoch)

    def _noise_std(epoch: int) -> float:
        """Instance-noise std annealed linearly to 0 over the whole run."""
        return scheduled_noise(args, epoch)

    history = resume_state["history"] if resume_state else []
    global_step = resume_state["global_step"] if resume_state else 0
    if resume_state:
        restore_rng(resume_state["rng"])
        # Discard uncommitted metric lines left after the last verified snapshot.
        paths.metrics_jsonl.write_text("".join(json.dumps(r) + "\n" for r in history))
    last_checkpoint = time.monotonic()

    def checkpoint(epoch, complete, progress, epoch_rng, best=False):
        nonlocal last_checkpoint
        payload = {
            "recovery_version": 1, "epoch": epoch, "epoch_complete": complete,
            "progress": progress, "epoch_rng": epoch_rng, "rng": capture_rng(),
            "history": history, "global_step": global_step,
            "generator": generator.state_dict(), "discriminator": discriminator.state_dict(),
            "optimizer_g": opt_g.state_dict(), "optimizer_d": opt_d.state_dict(),
            "stats": stats.to_dict(), "args": vars(args), "best_val": best_val,
        }
        diagnostics.phase("checkpoint", epoch=epoch, step=global_step)
        if best:
            save_checkpoint(payload, paths.best_ckpt, recovery_dir)
        save_checkpoint(payload, paths.latest_ckpt, recovery_dir)
        last_checkpoint = time.monotonic()
        return payload

    for epoch in range(start_epoch, args.epochs + 1):
        generator.train()
        discriminator.train()
        g_run = d_run = l1_run = phys_run = resp_run = semd_run = 0.0
        last_d = 0.0
        seen = 0
        d_updates = 0
        d_eligible = 0
        epoch_rng = capture_rng()
        skipped_steps = 0
        if resume_state and not resume_state["epoch_complete"]:
            state = resume_state["progress"]
            (g_run, d_run, l1_run, phys_run, resp_run, semd_run, last_d,
             seen, d_updates, d_eligible, skipped_steps) = state
            epoch_rng = resume_state["epoch_rng"]
            restore_rng(epoch_rng)

        adv_w = _adv_weight(epoch)
        noise_std = _noise_std(epoch)
        adv_active = adv_w > 0.0

        train_iter = train_loader
        if args.max_train_batches is not None:
            train_iter = islice(train_loader, args.max_train_batches)

        train_iter = iter(train_iter)
        for _ in range(skipped_steps):
            next(train_iter)
        if resume_state:
            if not resume_state["epoch_complete"]:
                restore_rng(resume_state["rng"])
            resume_state = None
        step = skipped_steps
        diagnostics.phase("training", epoch=epoch, step=global_step)
        for step, batch in enumerate(train_iter, start=skipped_steps + 1):
            opt_g.zero_grad(set_to_none=True)
            opt_d.zero_grad(set_to_none=True)
            discriminator.requires_grad_(True)
            lr = normalize(batch["lr"].to(env.device), stats)
            hr = normalize(batch["hr"].to(env.device), stats)

            fake = generator(lr, hr.shape[-2:])
            fake_raw = denormalize(fake, stats)
            hr_raw = denormalize(hr, stats)

            # --- Discriminator update (throttled + floored + instance noise) ---
            # Skipped entirely during adversarial warmup so G first learns
            # coarse structure from L1+physics alone.
            if adv_active and (step % args.g_steps_per_d == 0):
                d_eligible += 1
                if noise_std > 0:
                    real_in = hr + torch.randn_like(hr) * noise_std
                    fake_in = fake.detach() + torch.randn_like(fake) * noise_std
                else:
                    real_in = hr
                    fake_in = fake.detach()
                real_logits = discriminator(lr, real_in)
                fake_logits_d = discriminator(lr, fake_in)
                d_loss = discriminator_loss(real_logits, fake_logits_d, args.real_label)
                if not torch.isfinite(d_loss).item():
                    raise FloatingPointError(f"Non-finite discriminator loss at epoch {epoch}, batch {step}")
                # Don't let an already-winning D keep sharpening — that is what
                # collapses the adversarial gradient to G.
                if d_loss.item() >= args.d_loss_floor:
                    opt_d.zero_grad(set_to_none=True)
                    d_loss.backward()
                    opt_d.step()
                    d_updates += 1
                last_d = d_loss.item()

            # --- Generator update ---
            l1 = weighted_l1_loss(fake, hr, weighting=args.l1_weighting, alpha=args.l1_weight_alpha)
            phys = physics_loss(fake_raw, hr_raw, formulation=args.physics_loss_type)
            g_loss = args.lambda_l1 * l1 + args.lambda_physics * phys
            # Geometric term, OFF by default (lambda_semd=0.0). Guarded rather
            # than multiplied by zero so a disabled run pays none of its cost
            # and the loss is bit-for-bit what it was before this term existed.
            semd_val = 0.0
            if args.lambda_semd > 0.0:
                semd = semd_loss(
                    fake_raw, hr_raw, topk=args.semd_topk,
                    omega_R=args.semd_omega_R, beta=args.semd_beta,
                    normalize_by_energy=not args.semd_loss_raw,
                    chunk=args.semd_chunk,
                )
                g_loss = g_loss + args.lambda_semd * semd
                semd_val = semd.item()
            if adv_active:
                discriminator.requires_grad_(False)
                fake_logits_g = discriminator(lr, fake)
                adv = generator_adv_loss(fake_logits_g)
                g_loss = g_loss + adv_w * adv

            if not torch.isfinite(g_loss).item():
                raise FloatingPointError(f"Non-finite generator loss at epoch {epoch}, batch {step}")
            opt_g.zero_grad(set_to_none=True)
            g_loss.backward()
            opt_g.step()

            bs = lr.shape[0]
            seen += bs
            global_step += 1
            g_val, d_val, l1_val, phys_val = g_loss.item(), last_d, l1.item(), phys.item()
            g_run += g_val * bs
            d_run += d_val * bs
            l1_run += l1_val * bs
            phys_run += phys_val * bs
            resp_run += energy_response(fake_raw, hr_raw).mean().item() * bs
            semd_run += semd_val * bs
            diagnostics.state.update(epoch=epoch, batch=step, step=global_step)
            if not math.isfinite(g_val):
                raise FloatingPointError(f"Non-finite generator loss at epoch {epoch}, batch {step}")

            if time.monotonic() - last_checkpoint >= args.checkpoint_seconds:
                checkpoint(epoch, False, [g_run, d_run, l1_run, phys_run, resp_run,
                    semd_run, last_d, seen, d_updates, d_eligible, step], epoch_rng)
                diagnostics.phase("training", epoch=epoch, step=global_step)

            # Per-step logging so W&B curves populate within seconds instead of
            # waiting a full epoch (one epoch over the full stream is very long).
            if global_step % args.log_every == 0:
                wb.log(
                    {
                        "step/g_loss": g_val,
                        "step/d_loss": d_val,
                        "step/l1": l1_val,
                        "step/phys": phys_val,
                        "step/adv_weight": adv_w,
                        "epoch": epoch,
                    },
                    step=global_step,
                )
            # In particular, floored/skipped D losses have not run backward;
            # release their graphs rather than retaining them through validation.
            fake = fake_raw = hr = hr_raw = lr = g_loss = phys = l1 = None
            real_in = fake_in = real_logits = fake_logits_d = d_loss = None
            fake_logits_g = adv = semd = None

        if not seen:
            raise ValueError("Training loader produced no samples")
        # Preserve all training updates before validation/SEMD can fail.
        checkpoint(epoch, False, [g_run, d_run, l1_run, phys_run, resp_run,
            semd_run, last_d, seen, d_updates, d_eligible, step], epoch_rng)
        denom = seen
        d_skip_frac = 1.0 - (d_updates / d_eligible) if d_eligible else 0.0
        train_metrics = {
            "epoch": epoch,
            "train_g_loss": g_run / denom,
            "train_d_loss": d_run / denom,
            "train_l1": l1_run / denom,
            "train_phys": phys_run / denom,
            "train_response": resp_run / denom,
            "train_semd": semd_run / denom,
            "adv_weight": adv_w,
            "d_skip_frac": d_skip_frac,
            "d_input_noise": noise_std,
            "lr_skip_alpha": generator.lr_skip_alpha.item(),
        }
        diagnostics.phase("validation_semd", epoch=epoch)
        val_metrics = evaluate(
            generator, val_loader, stats, env.device, max_batches=args.max_val_batches,
            semd_topk=args.semd_topk, semd_omega_R=args.semd_omega_R, semd_beta=args.semd_beta,
            semd_chunk=args.semd_chunk,
        )
        record = {**train_metrics, **{f"val_{k}": v for k, v in val_metrics.items()}}

        # Per-epoch tagging efficiency with the frozen tagger (parquet only).
        if frozen_tagger is not None:
            try:
                tag_data = collect_tagging_tensors(
                    generator, val_loader, stats, env.device, max_samples=args.tag_max_samples,
                )
                auc_hr = eval_tagger_auc(frozen_tagger, tag_data["hr"], tag_data["y"], env.device)
                auc_sr = eval_tagger_auc(frozen_tagger, tag_data["sr"], tag_data["y"], env.device)
                auc_lr = eval_tagger_auc(frozen_tagger, tag_data["lr"], tag_data["y"], env.device)
                record["val_tag_auc_hr"] = auc_hr
                record["val_tag_auc_sr"] = auc_sr
                record["val_tag_auc_lr"] = auc_lr
                record["val_tag_efficiency"] = (auc_sr / auc_hr) if auc_hr else float("nan")
            except ValueError:
                frozen_tagger = None  # no labels -> stop trying (HDF5)

        history.append(record)

        print(
            f"epoch {epoch:03d} g={record['train_g_loss']:.4f} d={record['train_d_loss']:.4f} "
            f"l1={record['train_l1']:.4f} phys[{args.physics_loss_type}]={record['train_phys']:.4f} "
            f"val_l1={record['val_l1']:.4f} val_psnr={record['val_psnr_norm']:.2f} "
            f"resp={record['val_energy_response']:.3f} "
            f"phys_ratio={record['val_phys_loss_ratio']:.4f} phys_l2={record['val_phys_loss_l2']:.4f} "
            f"peak={record['val_peak_ratio']:.3f} nz={record['val_nonzero_ratio']:.3f} "
            f"semd={record['val_semd']:.3g} "
            f"advw={adv_w:.2f} dskip={d_skip_frac:.2f} alpha={record['lr_skip_alpha']:.3f}"
        )
        with paths.metrics_jsonl.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")
        # Log at global_step (not epoch) so this shares a monotonically increasing
        # x-axis with the per-step logs above — W&B silently drops out-of-order steps.
        wb.log(record, step=global_step)

        is_best = val_metrics["l1"] < best_val
        if is_best:
            best_val = val_metrics["l1"]
        diagnostics.phase("rendering", epoch=epoch)
        if epoch % args.sample_every == 0 or epoch == args.epochs or is_best:
            sample_batch = next(iter(val_loader))
            grid = render_sample_grid(
                generator, sample_batch, stats, env.device,
                paths.figures / f"sample_epoch_{epoch:03d}.png",
            )
            wb.log_image("samples", grid, step=global_step, caption=f"epoch {epoch}")

        ckpt = checkpoint(epoch, True, None, epoch_rng, best=is_best)
        if is_best:
            wb.log_artifact(paths.best_ckpt, name=f"model-{args.run_name}", aliases=["best"])

    render_metrics_plot(history, paths.figures / "metrics.png")

    diagnostics.phase("final_validation_semd")
    final_eval = evaluate(generator, val_loader, stats, env.device, max_batches=args.max_val_batches,
        semd_topk=args.semd_topk, semd_omega_R=args.semd_omega_R,
        semd_beta=args.semd_beta, semd_chunk=args.semd_chunk)
    final_eval.update({"best_val_l1": best_val, "scale": args.scale, "dataset": dataset_name})
    paths.eval_json.write_text(json.dumps(final_eval, indent=2), encoding="utf-8")
    wb.set_summary({
        "best_val_l1": best_val,
        "final_energy_response": final_eval["energy_response"],
        "final_psnr_norm": final_eval["psnr_norm"],
        "scale": args.scale,
        "dataset": dataset_name,
    })
    wb.finish()
    print(f"[done] best val L1 = {best_val:.5f} | eval -> {paths.eval_json}")


_ACTIVE_RESOURCES = []


def main() -> None:
    try:
        train(resolve_args())
    finally:
        for resource in reversed(_ACTIVE_RESOURCES):
            try:
                resource.close() if hasattr(resource, "close") else resource.finish()
            except Exception:
                pass
        _ACTIVE_RESOURCES.clear()


if __name__ == "__main__":
    main()
