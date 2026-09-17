"""Training state shared by timed checkpoints and resume validation."""
import random

import numpy as np
import torch


def capture_rng():
    ns = np.random.get_state()
    return {"python": random.getstate(), "numpy": [ns[0], ns[1].tolist(), *ns[2:]],
            "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []}


def restore_rng(state):
    random.setstate(state["python"])
    ns = state["numpy"]
    np.random.set_state((ns[0], np.asarray(ns[1], dtype=np.uint32), *ns[2:]))
    torch.set_rng_state(state["torch"].cpu())
    if torch.cuda.is_available() and state["cuda"]:
        torch.cuda.set_rng_state_all([s.cpu() for s in state["cuda"]])


def validate_resume(checkpoint, args):
    if checkpoint.get("recovery_version") != 1:
        raise ValueError("Resume requires a recovery_version=1 checkpoint; legacy weights require a separate warm-start experiment.")
    runtime = {"resume", "run_name", "run_dir", "recovery_dir", "experiments_root", "config",
               "cache_dir", "checkpoint_seconds", "log_every", "sample_every",
               "no_wandb", "wandb_project", "wandb_entity"}
    changed = [k for k, v in vars(args).items() if k not in runtime
               and checkpoint["args"].get(k) != v]
    if changed:
        raise ValueError(f"Incompatible resume configuration: {changed}")


def adversarial_weight(args, epoch):
    e = epoch - 1
    if e < args.adv_warmup_epochs:
        return 0.0
    if args.adv_ramp_epochs <= 0:
        return args.lambda_adv
    return args.lambda_adv * min(1.0, (e - args.adv_warmup_epochs + 1) / args.adv_ramp_epochs)


def noise_std(args, epoch):
    return max(0.0, args.d_input_noise * (1 - (epoch - 1) / max(1, args.epochs - 1)))
