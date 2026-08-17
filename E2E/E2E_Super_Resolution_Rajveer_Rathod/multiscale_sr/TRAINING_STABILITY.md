# Diagnosing and Fixing GAN Mode Collapse on Sparse Calorimeter Images

**Author:** Rajveer Rathod · **Project:** GSoC 2026 — ML4Sci, CMS Calorimeter Super-Resolution

This is the engineering story behind the multi-scale results: what broke, how it was diagnosed, what fixed it, and what the fix cost. The quantitative isolation of one knob is in [`reports/multiscale_2026-07/REPORT.md`](reports/multiscale_2026-07/REPORT.md).

---

## 1. Symptom

The super-resolution GAN collapsed. SR output was a **dim, smeared blob**: the bright energy core lost its peak amplitude, fine structure disappeared, and the whole image drifted toward a low-amplitude average. The effect was worst at the coarse scales (16× and 32×), where the generator has the most missing information to invent.

Two training signals moved the wrong way:

- `val_psnr` **decreasing** across epochs rather than improving.
- `peak_ratio` — the ratio of SR peak brightness to HR peak brightness — stuck well below 1.0, confirming the core was being dimmed rather than reconstructed.

## 2. Root cause

Two mechanisms compounded each other.

**The discriminator won outright.** `d_loss` fell to roughly **0.004**. At that point the discriminator classifies real-vs-fake essentially perfectly, its output saturates, and the gradient flowing back to the generator vanishes. The adversarial term stopped contributing any useful learning signal.

**The L1 term dominated, and pointed the wrong way.** With `λ_l1 = 50` on data where the overwhelming majority of pixels are zero, the loss-minimising strategy is *"predict small everywhere"*. Uniform L1 treats a missed 100-GeV core deposit and a missed empty pixel as equally important, so the optimiser rationally chose the blob. On natural images this failure mode is mild; on sparse, high-dynamic-range calorimeter data it is severe.

The generator was therefore optimising an objective whose adversarial half was dead and whose reconstruction half rewarded exactly the wrong behaviour.

## 3. The fix

Five coordinated changes. None is sufficient alone.

| # | Change | What it addresses |
|---|---|---|
| 1 | **Adversarial warmup** (`adv_warmup_epochs: 3`) | Train the generator on reconstruction alone first, so the discriminator does not face a random generator and win instantly. |
| 2 | **Adversarial ramp** (`adv_ramp_epochs: 5`) | Introduce the adversarial weight gradually (0 → 1) rather than as a step change. |
| 3 | **Discriminator throttling** (`d_loss_floor`, `g_steps_per_d: 2`) | Skip the discriminator update when its loss drops below a floor; give the generator 2 steps per discriminator step. Keeps the critic beatable. |
| 4 | **Instance noise** (`d_input_noise: 0.05`, annealed to 0) | Blur the real/fake boundary early so the discriminator cannot separate them trivially. |
| 5 | **Energy-weighted L1 at λ=10** (`l1_weighting: energy`, `alpha: 5.0`) | Weight the L1 term by local energy so high-deposit pixels dominate. Removes the "predict small everywhere" incentive. |

A sixth change sits in the generator itself: **nearest-neighbour rather than bicubic upsampling** in the residual path, which avoids the interpolation smoothing that was flattening peaks.

## 4. Outcome

| | Before | After |
|---|---|---|
| Discriminator | dead (`d_loss` ≈ 0.004) | active & stable (~0.04–0.09) |
| `val_psnr` | decreasing | increasing / stable |
| `peak_ratio` | dim blob | climbing toward ~1.0 (0.89 at 64×) |
| Energy response | — | ≈ 1.0 (conserved to <1%) |
| SR image | dim central smear | sharp, correctly-placed energy core |

Measured end state under the corrected settings, on the physics metric:

| Scale | Epochs | peak_ratio | Tagging efficiency | LR→HR gap recovered |
|---:|---:|---:|---:|---:|
| 16× | 30 | 0.688 | 72.9% | −153.9% |
| 32× | 40 | 0.822 | 75.0% | +13.8% |
| 64× | 40 | 0.887 | 93.9% | +78.5% |

All three rows share one tagger instance (AUC_HR = 0.6686). A longer 32× run at 60 epochs reaches peak_ratio 0.873 / efficiency 80.2% / recovery +25.3% against a *different* tagger instance (AUC_HR = 0.6671), so it is kept out of this table.

## 5. The trap: a fix that can silently un-fix itself

`d_loss_floor` is the delicate knob. Set too **low**, the discriminator overpowers the generator and the original collapse returns. Set too **high**, the discriminator is skipped so often that it freezes — and the generator quietly reverts to L1-only behaviour, which is the failure the whole package was built to prevent.

The logs make this visible. At `d_loss_floor = 0.10` on 32×, the discriminator was skipped on **98.5% of steps on average**, and on **100% of steps for a third of all epochs**. Training looked healthy — smooth, monotonic val_L1 descent, no divergence — while the adversarial term contributed essentially nothing.

**Diagnostic:** watch `d_skip_frac` in `metrics.jsonl`. Sustained values near 1.0 mean the floor is too high for that scale.

**The floor is scale-dependent** because the discriminator settles at different loss levels depending on task difficulty. At 16× it settles near 0.04, so the floor had to be lowered 0.05 → **0.02** to keep it alive and unfreeze peak recovery.

| Scale | Upscale | `d_loss_floor` |
|---:|---:|---:|
| 16× | 8× | 0.02 |
| 32× | 4× | 0.05 |
| 64× | 2× | 0.05 |

## 6. Honest limitation

The fix restores the **dominant energy core** reliably. It does not recover HR's sparse single-pixel deposits at 8× upscale (16×) — that information is largely destroyed by the downsampling and is not recoverable by this architecture. The model gives a physically sensible concentrated response, with residual error confined to that sub-core "dust".

More importantly: **fixing the collapse did not fix the physics at 16×.** That model now produces clean, sharp images with the **best energy correlation of any scale** (r = 0.998 vs HR) and a pt–energy correlation matching HR to three decimals — and a tagger trained on real HR still cannot use them. Recovery is −153.9%, background rejection is below the bicubic floor, and calibration is the worst of the three sources.

Pixel-level stability, energy conservation, and physics-level *usefulness* are three different problems, and solving the first two does not solve the third. That is the main reason this project treats tagging efficiency, not PSNR or energy response, as the headline metric — see [`reports/multiscale_2026-07/REPORT.md §7`](reports/multiscale_2026-07/REPORT.md).

## 7. Documentation note

An earlier version of the README documented the objective as `L_G = L_adv + 50·L_l1 + 10·L_phys`. That **λ=50 was the pre-fix value** — the very setting that caused the collapse described here. Every committed config and the `train.py` default now use `λ_l1 = 10.0` with energy weighting. The README has been corrected; this note records why the two ever disagreed.
