# Point-Cloud Representation — Exploration and Recommendation

**Status:** exploratory writeup, not implemented. See project issue #5.

## Question

The current pipeline represents every jet as a dense `(3, 128, 128)` image and runs it through a CNN generator/discriminator, even though calorimeter deposits are extremely sparse (**~97% zero pixels** — see the `weighted_l1_loss` docstring in [`multiscale_sr/engine.py`](../multiscale_sr/engine.py)). Would representing a jet as a point cloud — a variable-length list of `(x, y, energy)` hits instead of a dense grid — be a better fit, and is it worth pursuing given the timeline (project completion ~Nov 3, evaluation ~Nov 10)?

## 1. Prior art in HEP

Point-cloud generative models are an established line of work in fast-simulation / generative HEP, distinct from image-based calorimeter GANs:

- **PointNet / PointNet++**-style encoders (permutation-invariant, operate directly on unordered hit sets) are the standard architecture whenever HEP data is treated as a point cloud rather than an image — used broadly for jet tagging and increasingly for generation.
- **Graph-based generative models** (message-passing GNNs over hits, edges built by k-NN or learned) are the more common choice specifically for *generating* calorimeter showers as point sets, since they can output a variable number of hits and don't need a fixed grid resolution.
- **Energy Flow Networks / Deep Sets**-style permutation-invariant architectures are used where the object *is* naturally a set (particles in a jet) rather than a raster.

The throughline: point-cloud approaches in this space are used because detector output is fundamentally an unordered, variable-cardinality set of hits, and forcing it onto a fixed grid is itself a modeling choice — one this project currently makes for good practical reasons (see §3), not because it's the only option.

## 2. Conversion cost against this pipeline

The current data pipeline (`multiscale_sr/data/`) is grid-based at every layer, so adopting point clouds is not a localized change:

- **`parquet_dataset.py`** yields dense `(3, H, W)` tensors directly from the parquet `X_jets`/`X_jets_LR` columns — no hit list exists; the data is already rasterized upstream of this repo. A point-cloud loader would need to convert each dense image back into a hit list (`nonzero()` + coordinates + energy), which is lossless but adds a per-sample variable-length step that the current fixed-batch tensor pipeline doesn't have to handle.
- **`multiscale.py` (`downscale_hr`)** builds LR via `F.interpolate(mode="area")` — a purely grid operation. The point-cloud analogue (spatially binning/merging hits into a coarser hit set, or sub-sampling) is a different operation with different energy-conservation properties and would need its own validation against the `E-corr r` metric this project already tracks.
- **`factory.py` (`_multiscale_collate`)** stacks fixed-size tensors into batches (`torch.stack`). Point clouds have variable hit counts per sample, so batching needs padding + masks or a specialized collate (as in most GNN/point-cloud training code) — a real but well-trodden engineering cost, not a research problem.
- **Generator/Discriminator** (`multiscale_sr/models/`) are fully convolutional (`PixelShuffleUpBlock`, spectral-norm PatchGAN) — neither transfers to point clouds. A point-cloud generator/discriminator pair is a from-scratch model, not an adaptation.
- **`engine.py`** losses (`weighted_l1_loss`, `physics_loss_ratio/l2`) operate on dense tensors (`.sum()`, `.abs()` over a grid). The physics/energy-conservation losses port over conceptually (sum of hit energies still needs to match), but the pixel-reconstruction loss has no direct point-cloud equivalent — point-cloud generation losses are usually Chamfer distance, Earth Mover's Distance, or a learned discriminator score instead of per-pixel L1.
- **`classification_eval.py` / `tag_efficiency.py` / `tagger.py`** — the downstream tagger is also CNN-based on the dense image. A point-cloud SR output would need either a re-rasterization step before tagging (defeating part of the purpose) or a parallel point-cloud tagger, doubling the evaluation surface.

Net: this is a **second, parallel pipeline** (data loader, model, loss, and likely tagger), not an incremental change to the existing one. Realistic estimate is on the order of the effort already spent building the current grid pipeline end-to-end.

## 3. Would it avoid the current failure modes?

The two concrete failure modes this project has already hit:

- **Near-zero collapse under L1 on sparse images** (documented in [`TRAINING_STABILITY.md`](TRAINING_STABILITY.md) and the `weighted_l1_loss` docstring) — plain mean-L1 on ~97%-zero targets is minimized by predicting near-zero everywhere. **A point-cloud representation plausibly sidesteps this by construction**: there is no "predict zero" degenerate solution when the output is a list of hits rather than a dense grid to be mostly-empty — the loss is defined over the (small) set of actual hits, not diluted by empty pixels. This is a genuine structural advantage, not just a re-weighting workaround like the current `alpha=5.0` energy weighting.
- **Bicubic pre-blurring at small input sizes** (documented in the generator docstring / README "Method" section — the earlier bicubic-then-residual design lost peaks before any learned layer ran, at 16×/32×) — this is specifically an artifact of *interpolation-based* upsampling on a grid. A point-cloud model has no analogous blurring step; "upsampling" becomes predicting new hits or refining hit positions/energies directly, which is a different problem with its own failure modes (e.g. how many hits to generate, where) rather than a fix for this one.

So: plausibly yes on both counts, but by replacing the representation and the entire failure-mode profile, not by patching the current one. The comparison this project would need to make is CNN-image-GAN-with-workarounds vs. point-cloud-GAN-with-different-workarounds, not "point cloud is strictly better."

## Recommendation

**Not worth pursuing before the Nov 3 / Nov 10 deadline.** Reasoning:

1. It requires a parallel data pipeline, model architecture, loss formulation, and likely a parallel evaluation path (§2) — this is a second research project, not an extension of the current one.
2. The current pipeline's failure modes are already mitigated with measured, working fixes (energy-weighted L1, staged learned upsampling, discriminator throttling — see `TRAINING_STABILITY.md`), and the 64× result is complete and defensible per the README's Status section. There's a stronger near-term payoff in the already-identified open threads (32× convergence, tagger domain-adaptation for the 16× distribution-mismatch finding) than in a from-scratch representation change.
3. The strongest argument *for* point clouds — avoiding near-zero collapse by construction — is a genuine advantage, but the current pipeline already has a working (if less elegant) fix for the same problem, so the marginal benefit is smaller than it would be if the sparse-image problem were still open.

**If picked up post-deadline**, the highest-value first step is not a full reimplementation: prototype only the loader (`parquet → hit list`) and a Chamfer/EMD reconstruction-quality check against the *existing* HR/LR pairs, without training a full GAN, to get a cheap read on how much of a jet's energy/structure survives a naive point-cloud conversion before committing to the larger build.
