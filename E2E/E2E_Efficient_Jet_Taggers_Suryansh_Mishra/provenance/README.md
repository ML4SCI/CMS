# Provenance: QuarkGluon training scripts behind the kernel numbers

Copied verbatim from `omasho-codes/ml4sci_cms_e2e26` (midterm repo). These are
the scripts that produced `logs/kernel_benchmarks/*_train_e2e.out`:

- `quarkgluon_dataset.py` — the QuarkGluon loader the benchmarks ran on.
- `train_lgatr_comparison.py` — stock-vs-fused L-GATr training comparison.
- `train_lorentz_part_comparison.py` — stock-vs-fused Hybrid LorentzParT comparison.

Frozen reference copies: the live, maintained versions of the kernels they
benchmark live in `part_kernels/` and `lgatr_kernels/` at this folder's top level.