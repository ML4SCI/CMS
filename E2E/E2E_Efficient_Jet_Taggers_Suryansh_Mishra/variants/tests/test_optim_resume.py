"""Regression tests for Lookahead optimizer checkpoint resume.

Feature: ParT ablation — Lookahead(RAdam) optimizer.

The trainer chains Slurm jobs that resume from ``last.pt``.  ``Lookahead``
holds per-parameter *slow* weights (``cached_params``, ``cached_mom``) and a
sync counter ``_la_step`` in addition to the inner optimizer's state.  If
``state_dict`` discards them, a resumed run interpolates trained parameters
toward the fresh random initialization the optimizer was constructed on —
silently corrupting training after the first Lookahead sync (every
``la_steps`` iterations).

These tests pin the resume contract:

* the slow-weight cache and sync counter survive a save/load round-trip;
* a run interrupted at step ``M`` and resumed from checkpoint is exactly
  equal to a run that never stopped (this fails on the old implementation).
"""

from __future__ import annotations

import torch
from torch import nn
from torch.optim import RAdam, SGD

from variants.optim import Lookahead

TOTAL_STEPS = 12
INTERRUPT_AT = 5
LA_STEPS = 3
LA_ALPHA = 0.5


def _step(model: nn.Module, opt: Lookahead) -> None:
    """One deterministic optimizer step on a fixed pseudo-loss."""
    opt.zero_grad(set_to_none=True)
    x = torch.tensor([[1.0, -2.0, 0.5, 3.0]])
    model(x).square().mean().backward()
    opt.step()


def _make(seed: int):
    torch.manual_seed(seed)
    model = nn.Linear(4, 2)
    opt = Lookahead(
        RAdam(model.parameters(), lr=0.01), la_steps=LA_STEPS, la_alpha=LA_ALPHA
    )
    return model, opt


def test_lookahead_slow_weights_survive_roundtrip() -> None:
    """cached_params, cached_mom and _la_step must be restored, not re-primed."""
    model, opt = _make(0)
    for _ in range(INTERRUPT_AT):
        _step(model, opt)

    saved = opt.state_dict()

    # Simulate a resume: fresh optimizer primed on *different* weights, then
    # the model checkpoint is loaded, then the optimizer state.
    model2, opt2 = _make(99)
    model2.load_state_dict(model.state_dict())
    opt2.load_state_dict(saved)

    for p, p2 in zip(model.parameters(), model2.parameters()):
        assert torch.equal(
            opt.state[p]["cached_params"], opt2.state[p2]["cached_params"]
        ), "slow weights were lost on resume"
    assert opt2._la_step == opt._la_step, "lookahead sync counter was lost"

    # A stale cache (different from the resumed model weights) would make the
    # first sync blend toward random init — the very bug under test.
    for p in model2.parameters():
        cached = opt2.state[p]["cached_params"]
        assert not torch.equal(cached, p.data), "cache should reflect trained state"


def test_lookahead_resume_matches_continuous_run() -> None:
    """Interrupt + resume produces identical weights to an uninterrupted run.

    With ``la_steps=3`` the first post-resume sync happens 3 steps in, so any
    corruption of ``cached_params`` changes the weights by step 8 and the test
    catches it.
    """
    # Continuous reference run.
    model_ref, opt_ref = _make(0)
    for _ in range(TOTAL_STEPS):
        _step(model_ref, opt_ref)

    # Interrupted run: step to INTERRUPT_AT, checkpoint, then resume in a
    # "fresh process" (new model + optimizer, weights loaded from checkpoint).
    model, opt = _make(0)
    for _ in range(INTERRUPT_AT):
        _step(model, opt)
    sd_model, sd_opt = model.state_dict(), opt.state_dict()

    model2, opt2 = _make(0)
    model2.load_state_dict(sd_model)
    opt2.load_state_dict(sd_opt)
    for _ in range(INTERRUPT_AT, TOTAL_STEPS):
        _step(model2, opt2)

    for a, b in zip(model_ref.parameters(), model2.parameters()):
        assert torch.allclose(a.data, b.data, atol=1e-12), (
            "resumed run diverged from continuous run — Lookahead cache or "
            "sync counter not restored"
        )


def test_legacy_state_dict_is_supported() -> None:
    """Old checkpoints (bare inner state) must load without corrupting weights.

    The slow-weight cache is genuinely lost in a legacy checkpoint, so the
    documented safe default is to re-prime it from the checkpoint model weights
    (``train.py`` loads the model before the optimizer).  The first Lookahead
    sync then interpolates between the checkpoint weights and the newly
    trained steps — never toward the optimizer's fresh random init.
    """
    model, opt = _make(0)
    for _ in range(INTERRUPT_AT):
        _step(model, opt)

    legacy = opt.optimizer.state_dict()  # pre-fix format

    # Legacy resume: fresh process, model weights from checkpoint, then the
    # legacy (inner-only) optimizer state.
    model2, opt2 = _make(99)
    model2.load_state_dict(model.state_dict())
    opt2.load_state_dict(legacy)

    # 1) The fallback primes the slow cache from the checkpoint weights...
    for p in model2.parameters():
        assert torch.equal(opt2.state[p]["cached_params"], p.data)
    assert opt2._la_step == 0

    # 2) ...which differs from the pre-fix behavior (cache left at the fresh
    # random init the optimizer was constructed on).  Run both paths forward;
    # the old behavior visibly drags the weights toward init.
    # Buggy path: identical resume but the cache is never re-primed.
    model_bug, opt_bug = _make(99)
    model_bug.load_state_dict(model.state_dict())
    for _ in range(LA_STEPS + 1):
        _step(model2, opt2)
        _step(model_bug, opt_bug)
    for p_fixed, p_bug in zip(model2.parameters(), model_bug.parameters()):
        assert not torch.allclose(p_fixed.data, p_bug.data, atol=1e-6), (
            "legacy resume is indistinguishable from the pre-fix cache corruption"
        )

