# Multi-Scale SRGAN — Architecture

**Project:** GSoC 2026 · ML4Sci · Physics-Aware Super-Resolution for CMS Calorimeters
**Component:** `multiscale_sr/` — Option A (independent GAN per downsampling scale)
**Source of truth:** `multiscale_sr/models/generator.py`, `models/discriminator.py`, `engine.py`

> **Purpose.** A faithful, line-referenced description of the GAN used in the
> multi-scale study. This is **not textbook SRGAN** — almost every deviation from the
> SRGAN paper is justified by either *calorimeter sparsity* or *GAN training stability*.
> Each design choice is tagged with its reason so the document doubles as a design
> rationale for the mentor and as input to a diagram generator.

---

## 0. One-paragraph summary

The high-resolution (HR) calorimeter image is the fixed ground truth. A low-resolution
(LR) input is produced by **area-downsampling** HR to a target scale (64, 32, 16). A
**bicubic-residual generator** upsamples the LR back to HR size with plain bicubic
interpolation and then adds a *learned residual correction* — so the network only learns
the hard part (sharpening), never the easy part (upscaling). A **conditional
spectral-norm PatchGAN discriminator** judges whether an HR image is a plausible
super-resolution *of its specific LR* by consuming the `(LR, HR)` pair. Training combines
an LSGAN adversarial loss, a heavy L1 reconstruction loss, and a direct
energy-conservation physics loss. The architecture is **resolution-agnostic**: the output
size is a forward-time argument, so one checkpoint serves every scale.

---

## 1. The central idea — bicubic-residual generation

The single most important design choice (`generator.py:55-57`):

```python
def forward(self, lr, target_size):
    lr_up = F.interpolate(lr, size=target_size, mode="bicubic", align_corners=False)
    return lr_up + self.net(lr_up)          # bicubic baseline + learned residual
```

The generator **does not learn to upsample**. It bicubic-upsamples first, then the CNN
learns only a **residual correction**:

```
    SR = bicubic(LR) + CNN(bicubic(LR))
```

**Why this design:**
- **Capacity goes to the hard part.** Bicubic handles basic upscaling for free; the CNN
  spends all its parameters sharpening edges and fixing the deposits bicubic blurs.
- **It can never be much worse than the baseline.** If the residual → 0, the output is
  exactly bicubic. The GAN's job is *literally* "improve on bicubic" — and that is the
  same bicubic that serves as the evaluation floor (see `README.md` tagging section). The
  architecture bakes the baseline in as a skip connection.
- **Resolution-agnostic.** `target_size` is passed at forward time, not baked into the
  weights. The same checkpoint upsamples 64→125, 32→125, or 16→125 — exactly what the
  multi-scale study needs (identical architecture across LR scales, only input resolution
  changes).

This is closer to **EDSR / VDSR** (residual SR) than to the original SRGAN, which used a
learned PixelShuffle upsampler.

**Diagram instruction:** draw LR entering a "bicubic upsample" box (dashed, = not
learned), the result branching into (a) a long skip arrow and (b) the CNN body; the CNN
output and the skip meet at a `+` node producing SR. Label the skip "bicubic baseline
(the floor)" and the CNN "learned residual correction."

---

## 2. Generator structure (`Generator`, `generator.py:29-57`)

Config used in runs: `gen_channels=64` (`base_channels`), `gen_blocks=8` (`num_blocks`).

```
LR (3, s, s)
   │  bicubic upsample → (3, 125, 125)              [lr_up — the residual skip]
   ▼
Conv 7×7  (3 → 64)  + ReLU                           [head: large receptive field]
   ▼
8 × ResidualBlock(64)                                [body]
   ▼
Conv 3×3  (64 → 64) + InstanceNorm + ReLU            [neck]
   ▼
Conv 7×7  (64 → 3)                                   [tail → residual map]
   ▼
+ lr_up                                              [add bicubic skip]
   ▼
SR (3, 125, 125)
```

Approx. 1.5M parameters — a compact generator. The **7×7 convs** at head and tail (vs
SRGAN's 9×9 / 3×3) give a large receptive field at input and output to capture the spatial
extent of jet deposits.

### 2.1 Residual block (`ResidualBlock`, `generator.py:7-26`)

```python
def forward(self, x):
    return x + 0.1 * self.block(x)          # EDSR-style ×0.1 damping
# block = Conv3×3 → InstanceNorm → ReLU → Conv3×3 → InstanceNorm
```

Two deliberate deviations from SRGAN, both physics/stability driven:

| Choice | Textbook SRGAN | This code | Reason |
|---|---|---|---|
| **Normalization** | BatchNorm | **InstanceNorm (affine)** | Calorimeter images are ~2–3% nonzero and batches are small → BatchNorm statistics collapse on sparse data. InstanceNorm normalizes per-sample, immune to batch composition. |
| **Skip scaling** | `x + block(x)` | `x + 0.1·block(x)` | EDSR-style damping keeps the residual small early in training → stable gradients, model starts near bicubic and learns corrections gradually. |

**Diagram instruction:** standard residual block, but annotate the skip-add with "×0.1"
and the norm layers as "InstanceNorm (sparse-safe)".

---

## 3. Discriminator (`Discriminator`, `discriminator.py:9-38`)

A **conditional PatchGAN with spectral normalization** — also not textbook SRGAN. Three
defining features:

### 3.1 Conditional (pix2pix-style) — judges the (LR, HR) pair

```python
def forward(self, lr, hr):
    lr_up = F.interpolate(lr, size=hr.shape[-2:], mode="bicubic", ...)
    return self.net(torch.cat([lr_up, hr], dim=1))   # 6-channel input (in_channels*2)
```

It does not ask "is this a real jet?" — it asks **"is this HR a plausible
super-resolution *of this specific LR*?"** Concatenating the upsampled LR with the HR
(channel dim) means the generator cannot satisfy it by producing *any* realistic jet; the
output must be consistent with the input. Much stronger training signal.

### 3.2 PatchGAN — local, not global

The final conv outputs a **1-channel map** (`discriminator.py:33`), not a single scalar.
Each output element judges a *patch* as real/fake, focusing the discriminator on local
texture and sharpness — precisely the SR failure mode (local detail), not global structure.

### 3.3 Spectral norm on every conv — the anti-collapse mechanism

Every `Conv2d` is wrapped in `spectral_norm(...)` (`discriminator.py:22-33`). This
constrains the discriminator's Lipschitz constant, preventing it from becoming too strong
too fast and collapsing the adversarial signal (the classic GAN failure: D wins → G gets
no gradient → blurry output). This is a structural protection against discriminator
collapse observed in earlier baseline runs.

```
(LR↑ ‖ HR) (6, 125, 125)
   ▼  Conv4×4 s2 (6→64)            + LeakyReLU(0.2)                 [spectral norm]
   ▼  Conv4×4 s2 (64→128)  + IN    + LeakyReLU(0.2)                 [spectral norm]
   ▼  Conv4×4 s2 (128→256) + IN    + LeakyReLU(0.2)                 [spectral norm]
   ▼  Conv4×4 s1 (256→512) + IN    + LeakyReLU(0.2)                 [spectral norm]
   ▼  Conv3×3 s1 (512→1)                                            [spectral norm]
   ▼
patch real/fake map (1, h', w')
```

**Diagram instruction:** show LR and HR merging into a concat block, then the conv stack
shrinking spatially, ending in a small grid (the patch map). Tag the whole stack "spectral
norm (Lipschitz-constrained → collapse-resistant)" and the input "conditional: sees LR+HR".

---

## 4. Training objective (`engine.py:7-48`)

```
L_G   = L_adv + λ_l1 · L_l1 + λ_phys · L_phys
L_adv = 0.5 · mean[(D(fake) − 1)²]                      (LSGAN generator,  line 34-36)
L_D   = 0.5 · mean[(D(real) − real_label)² + D(fake)²]  (LSGAN disc,      line 29-31)
L_l1  = mean|G(lr) − hr|                                 (normalized,      line 10)
L_phys= mean|Σ(E_pred)/Σ(E_true) − 1|                    (denormalized,    line 46-48)
```

Run config: `λ_l1 = 50`, `λ_phys = 10`, `real_label = 0.9`, `d_lr_ratio = 0.5`,
`n_critic = 1`.

Four deviations from vanilla SRGAN, all stabilizing:

| Term / setting | Vanilla SRGAN | This code | Reason |
|---|---|---|---|
| **Adversarial loss** | BCE | **LSGAN (squared)** | Smoother gradients, less mode collapse. |
| **L1 weight** | small / perceptual-dominated | **50 (heavy)** | Model is L1-dominated; adversarial is a sharpening *seasoning*, not the driver. Keeps energy roughly correct even if the GAN misbehaves. |
| **Physics loss** | none | **`\|Σpred/Σtrue − 1\|`** | Direct energy-conservation constraint on raw (denormalized) energies. Drives energy response → 1.0. |
| **D vs G learning rate** | equal | **TTUR (`d_lr = 0.5·lr`)** | Slows the discriminator so it doesn't overpower the generator. |
| **Real label** | 1.0 | **0.9 (one-sided smoothing)** | Label smoothing prevents the discriminator from becoming overconfident. |

**The through-line:** spectral norm + TTUR + label smoothing + LSGAN is a *quadruple*
defense against discriminator collapse; heavy L1 + physics loss is a *double* defense
against energy bias.

**Diagram instruction:** a loss-flow diagram — G produces fake, D scores real & fake;
arrows for L_adv (G←D), L_l1 (G←HR), L_phys (G←HR raw energies); annotate λ weights and
mark the four stability mechanisms.

---

## 5. Data path — downsampling vs the bicubic baseline

A subtlety that is easy to get wrong (and which this code gets right):

| Operation | Direction | Interpolation mode | Where | Why |
|---|---|---|---|---|
| **Make LR** (HR → LR input) | shrink | **`area`** | `data/multiscale.py:34` | Area (average) pooling preserves the per-cell mean and **cannot go negative** — bicubic's negative lobes would create unphysical negative energy around sparse deposits. |
| **Baseline / display / D input** (LR → HR) | grow | **`bicubic`** | `engine.py:140`, `generator.py:56`, `discriminator.py:37` | Bicubic-up is the field-standard "do-nothing" floor; there is no `area` upsampling mode. |

So **"LR (bicubic)" in the tagging eval = area-downsampled HR, then bicubic-upsampled to
HR size.** It is the floor against which both the GAN and (later) the INR are measured.

---

## 6. Evaluation metrics (`engine.py:39-104`)

| Metric | Definition | Notes |
|---|---|---|
| `val_l1` | L1 on **normalized** tensors | primary model-selection metric |
| `val_psnr_norm` | PSNR with fixed `max_val=1.0` (`psnr_normalized`, line 54) | comparable across scales and across models (avoids the raw-vs-normalized PSNR trap) |
| `val_energy_response` | `Σ(E_pred)/Σ(E_true)` on **denormalized** energy (`energy_response`, line 39) | 1.0 = unbiased |
| **tagging efficiency** | `AUC_SR / AUC_HR`; recovery `= (AUC_SR − AUC_LR)/(AUC_HR − AUC_LR)` | physics-facing headline; see `classification_eval.py` / `tag_efficiency.py` |

---

## 7. The multi-scale study (Option A)

One independent GAN is trained per LR scale to quantify how reconstruction quality and
physics fidelity degrade as input resolution drops:

| Scale | LR input | Config | Trained run (example) |
|---|---|---|---|
| 64× | 64×64 → 125×125 | `configs/scale_64.yaml` | `experiments/2026-06-23_datasets_64x_baseline` |
| 32× | 32×32 → 125×125 | `configs/scale_32.yaml` | `experiments/2026-06-24_datasets_32x_baseline` |
| 16× | 16×16 → 125×125 | `configs/scale_16.yaml` | `experiments/2026-06-23_datasets_16x_baseline` |

The architecture is **identical** across scales — only the input resolution differs — which
is the whole point of Option A: a controlled study of "quality vs input granularity."

---

## 8. Component → code map

| Component | Class / function | File:line |
|---|---|---|
| Bicubic-residual generator | `Generator.forward` | `generator.py:55` |
| Residual block (InstanceNorm, ×0.1) | `ResidualBlock` | `generator.py:7` |
| Conditional PatchGAN + spectral norm | `Discriminator` | `discriminator.py:9` |
| LSGAN discriminator loss | `discriminator_loss` | `engine.py:29` |
| LSGAN generator adversarial loss | `generator_adv_loss` | `engine.py:34` |
| Energy response | `energy_response` | `engine.py:39` |
| Physics (energy-conservation) loss | `physics_loss` | `engine.py:46` |
| Normalized PSNR | `psnr_normalized` | `engine.py:54` |
| Area downsampling (make LR) | `downscale_hr` | `data/multiscale.py:21` |
| Tagging-tensor collection | `collect_tagging_tensors` | `engine.py:112` |

---

## 9. Design rationale in one table (for the mentor)

Every deviation from textbook SRGAN, and why:

| Deviation | Motivation | Category |
|---|---|---|
| Bicubic-residual (don't learn upscaling) | spend capacity on sharpening; never worse than baseline; resolution-agnostic | architecture |
| InstanceNorm (not BatchNorm) | sparse images + small batches break BatchNorm | sparsity |
| ×0.1 residual scaling | stable early gradients (EDSR) | stability |
| Conditional discriminator (LR+HR) | SR must match *its* LR, not just look real | architecture |
| PatchGAN (local) | SR failure is local detail | architecture |
| Spectral norm | prevent discriminator collapse | stability |
| LSGAN loss | smoother gradients than BCE | stability |
| Heavy L1 (λ=50) | L1-dominated; energy stays right if GAN wobbles | physics |
| Physics loss `\|Σpred/Σtrue−1\|` | enforce energy conservation directly | physics |
| TTUR + label smoothing | balance D vs G | stability |
| Area downsample / bicubic upsample split | no negative energy on sparse deposits | physics |

---

### Figure checklist for the diagram generator
1. **Generator** — LR → bicubic skip + residual CNN → SR (emphasize the `+`).
2. **Residual block** — Conv/IN/ReLU ×2 with ×0.1 skip.
3. **Discriminator** — conditional (LR‖HR) concat → spectral-norm conv stack → patch map.
4. **Loss flow** — G/D adversarial + L1 + physics, with λ weights and stability tags.
5. **Multi-scale study** — same architecture, three input scales (64/32/16) → one HR.
6. **Data path** — area-down (make LR) vs bicubic-up (baseline), with the negative-energy note.
