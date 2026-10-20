# Normalizing Flows — Exploration and Recommendation

**Status:** exploratory writeup, not implemented. See project issue #6.

## Question

The current physics-fidelity mechanism is a single auxiliary loss term (`physics_loss`, in [`multiscale_sr/engine.py`](../multiscale_sr/engine.py)) added to an LSGAN objective — it constrains only the *first moment* of the energy distribution (`sum(E_pred)/sum(E_true) → 1`, either as `|·-1|` or `(·-1)²`, see `physics_loss_ratio`/`physics_loss_l2`). Would a normalizing-flow component, which gives an exact tractable likelihood and an invertible transform, enforce physical consistency more rigorously than this scalar auxiliary loss — either as a full alternative to the GAN generator, or as a module layered on top of it?

## 1. Prior art in HEP

Normalizing flows are an established alternative to GANs for HEP fast-simulation and generative calorimeter modeling, motivated by exactly the property this project's physics loss is a workaround for:

- **CaloFlow** and related calorimeter-shower flow models (built on the same CaloChallenge-style datasets this project's `hdf5_dataset.py` already supports) use conditional normalizing flows to generate full shower images with an exact likelihood objective, reporting better distributional fidelity on per-event energy and shape statistics than GAN baselines, at the cost of slower sampling and larger models.
- **Conditional RealNVP / Glow-style coupling flows** are the standard architecture choice when the target is "generate `X` conditioned on `Y`" with tractable density — directly analogous to this project's `(LR → HR)` setup.
- **Flow-matching / continuous normalizing flows** are a more recent alternative that trades exact discrete invertibility for a continuous-time formulation, often cheaper to train than classical coupling-layer flows, and have started appearing in fast-sim literature as a successor to both GANs and classical flows.

The throughline: flows are attractive in this literature specifically because an **exact likelihood directly constrains the full output distribution**, not just a single summary statistic — which is precisely the gap between this project's current `physics_loss` (one number: total energy ratio) and "physically consistent" in the fuller sense (correct per-pixel energy *distribution*, not just correct total).

## 2. Compatibility with the current conditional (LR→HR) setup

**As a full alternative generator:** technically compatible in principle (`p(HR | LR)` is exactly what a conditional flow models) but architecturally a from-scratch replacement, not an adaptation:

- The current `Generator` ([`multiscale_sr/models/generator.py`](../multiscale_sr/models/generator.py)) is a **non-invertible** stack of `Conv → PixelShuffle(2) → ReLU → ResidualBlock` stages — ReLU and PixelShuffle-based upsampling are not bijective, and the architecture changes shape between input and output (`16×16 → 128×128`). A flow's transform must be invertible and (in the classical coupling-layer formulation) dimension-preserving, so "replace the generator with a flow" means designing a new conditional coupling architecture that handles the resolution change itself (e.g. via a learned or fixed dimension-augmenting step, as CaloFlow and similar models do), not swapping in a drop-in replacement.
- The current `Discriminator` ([`multiscale_sr/models/discriminator.py`](../multiscale_sr/models/discriminator.py)) has no role in a flow-based setup — training is by exact log-likelihood, not adversarial signal, so the LSGAN loss (`discriminator_loss`/`generator_adv_loss` in `engine.py`) and the whole adversarial-training apparatus (`d_loss_floor` throttling, instance noise, the warmup/ramp schedule — see [`TRAINING_STABILITY.md`](TRAINING_STABILITY.md)) becomes irrelevant. That apparatus was hard-won (it's what fixed the original mode-collapse failure), so replacing the generator with a flow discards that investment rather than building on it.
- The energy-weighted L1 reconstruction loss (`weighted_l1_loss`) also has no direct analogue — a flow is trained by maximizing likelihood of the true HR under the model, not by minimizing a reconstruction distance to a single HR sample.

**As a physics-consistency module layered on the current GAN:** more compatible, but narrower in scope than "normalizing flow" usually implies:

- A small conditional flow (or even a simpler 1D flow) fit to the **marginal per-image energy distribution** conditioned on LR could replace the single-point constraint in `physics_loss` with a full likelihood term — e.g. an auxiliary loss `-log p_flow(sum(E_pred) | lr)` instead of `|sum(E_pred)/sum(E_true) - 1|`. This slots into the existing `L_G = L_adv + lambda_l1*L_l1 + lambda_physics*L_phys` composition (`engine.py` module docstring) as a drop-in alternative to the current dispatcher's `"ratio"`/`"l2"` formulations, without touching the generator/discriminator at all.
- This is a much smaller, much more defensible piece of work than "a flow-based generator" — it tests the actual hypothesis (does a distributional constraint beat a scalar one) without requiring an invertible image-to-image architecture.

## 3. Minimal feasibility experiment

Before any architecture change, the cheapest diagnostic is **not training anything**: fit a small unconditional (or LR-conditioned) 1D flow to the *existing* per-image total-energy values already computed by `energy_response()` in `engine.py`, using data already produced by a completed run (`evaluations/*/classification/classification_eval.json` and the `pt_correlation`/`energy_correlation` figures already contain per-sample energy pairs for HR/LR/SR). Concretely:

1. Take the HR total-energy distribution from an existing evaluation run (no new training).
2. Fit a small 1D normalizing flow (e.g. a handful of RealNVP-style coupling layers on the scalar, or even a simpler flow like an autoregressive quantile transform) to `p(E_HR)`, and separately conditioned on `E_LR` or `scale`.
3. Compare the flow's log-likelihood of held-out HR energies against a Gaussian fit to the same data (the implicit assumption behind the current L2 physics-loss variant) and against the empirical distribution.
4. If the flow captures structure (e.g. skew, multi-modality) the Gaussian assumption misses — and that structure correlates with the known 16× taggability failure — that's a concrete, cheap signal that a distributional physics loss is worth building. If the energy distribution is close to unimodal/Gaussian already, the current scalar loss is probably already capturing most of the available signal and a flow adds complexity without benefit.

This experiment is hours, not days, and requires no changes to the training pipeline — it operates on already-computed evaluation outputs.

## Recommendation

**Not worth pursuing as a full generator replacement before the Nov 3 / Nov 10 deadline; the layered physics-consistency module is a reasonable post-physics-loss-rework follow-up, not a priority now.**

Tradeoffs vs. the current GAN + physics-loss approach:

| | Current (GAN + scalar physics loss) | Flow-based generator | Flow as layered physics module |
|---|---|---|---|
| Constrains | Total energy ratio only | Full output distribution | Full energy-distribution shape (if extended beyond the scalar) |
| Architecture cost | Done — measured, stabilized (`TRAINING_STABILITY.md`) | Full rebuild (generator, loss, training loop); discards the adversarial-stability fixes already invested | Small — one new loss term, existing generator/discriminator untouched |
| Sampling cost | One forward pass | Depends on flow type; coupling flows are one pass, continuous-flow variants can be slower | N/A (used only as a training-time loss, not for sampling) |
| Risk given timeline | None (already working) | High — new failure modes, no stabilization experience with this pipeline's data | Low — additive, falls back to current behavior if it underperforms |

Given issue #2 (physics loss formulation) and #1 (learnable LR-skip) are explicitly prioritized first and this project's timeline is compressed, the recommended path is:

1. Finish and validate the scalar physics-loss work (#2) first — a flow-based distributional loss is a strict superset of that idea, so it should build on a settled baseline, not compete with it mid-flight.
2. **If time remains after the priority items**, run the minimal feasibility experiment in §3 (hours of effort, no training pipeline changes) as evidence for whether to invest further.
3. Do **not** attempt a full flow-based generator replacement in this project cycle — the architectural cost (§2) and the loss of the hard-won adversarial-training stability work outweigh the likely benefit given the remaining time budget.
