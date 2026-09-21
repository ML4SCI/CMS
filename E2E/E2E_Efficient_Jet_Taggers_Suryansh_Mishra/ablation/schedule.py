"""Optimizer and learning-rate schedule matching the official ParT recipe.

ParT's published JetClass runs use Lookahead wrapped around RAdam ("Ranger") with
a peak LR of 1e-3 held constant for the first 70% of iterations and then decayed
exponentially, with no weight decay.  Reproducing that schedule matters here:
every arm in the ablation must be trained under the *same* optimizer, or an
apparent architecture effect could just be a tuning difference.

A short linear warmup is added on top. The published recipe starts at the peak LR,
but several arms (differential attention's subtracted maps, MoE's untrained
router) are noticeably less stable in the first few hundred steps, and warming up
is a smaller intervention than per-arm LR tuning.
"""

from __future__ import annotations

import math
from typing import Iterable

import torch
from torch.optim import Optimizer

from variants.optim import Lookahead

__all__ = ["build_optimizer", "build_scheduler", "lr_multiplier"]


def build_optimizer(parameters: Iterable[torch.nn.Parameter], config) -> Optimizer:
    """Lookahead(RAdam) as in the official ParT / weaver "ranger" optimizer."""
    trainable = [p for p in parameters if p.requires_grad]
    inner = torch.optim.RAdam(
        trainable,
        lr=config.lr,
        betas=config.radam_betas,
        eps=config.radam_eps,
        weight_decay=config.weight_decay,
    )
    return Lookahead(
        inner,
        la_steps=config.lookahead_steps,
        la_alpha=config.lookahead_alpha,
    )


def lr_multiplier(step: int, config) -> float:
    """LR multiplier at ``step``: warmup, then constant, then exponential decay.

    The decay is chosen so the multiplier reaches ``lr_final_factor`` exactly at
    ``total_steps``.  Note the schedule always spans ``total_steps`` even when a
    run is cut short by ``max_steps``, so a smoke test does not silently train
    under a compressed schedule.
    """
    warmup = max(0, int(config.warmup_steps))
    if warmup > 0 and step < warmup:
        return (step + 1) / warmup

    constant_until = int(config.total_steps * config.lr_constant_fraction)
    if step <= constant_until:
        return 1.0

    decay_steps = max(1, config.total_steps - constant_until)
    progress = min(1.0, (step - constant_until) / decay_steps)
    # exp interpolation from 1.0 down to lr_final_factor
    return float(math.exp(progress * math.log(max(config.lr_final_factor, 1e-12))))


def build_scheduler(optimizer: Optimizer, config) -> torch.optim.lr_scheduler.LambdaLR:
    """LambdaLR implementing :func:`lr_multiplier`.

    Applied to the *inner* optimizer's param groups; ``Lookahead`` proxies
    ``param_groups`` through, so scheduling it works unchanged.
    """
    return torch.optim.lr_scheduler.LambdaLR(
        optimizer, lr_lambda=lambda step: lr_multiplier(step, config)
    )
