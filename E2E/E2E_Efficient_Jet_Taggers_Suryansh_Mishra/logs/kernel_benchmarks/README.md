# Kernel benchmark logs (QuarkGluon, single A100-SXM4-80GB)

Raw stdout from the benchmark runs quoted in the Fused kernels section of the
main README. Source repo: `omasho-codes/ml4sci_cms_e2e26` (`nohup/`).

| file | what produced it |
|------|------------------|
| `part_bench.out` | `part_kernels/benchmarks/bench_e2e.py` — ParT kernel microbenchmarks (pairwise features, attention w/ bias, pair-embedding MLP) |
| `lgatr_bench.out` | `lgatr_kernels/benchmarks/bench_e2e.py` — L-GATr kernel microbenchmarks (equivariant linear, geometric product) |
| `lgatr_train_e2e.out` | `provenance/train_lgatr_comparison.py` — full 10-epoch QuarkGluon training, stock vs fused (2.72x) |
| `hybrid_train_e2e.out` | `provenance/train_lorentz_part_comparison.py` — Hybrid LorentzParT training, stock vs fused (2.00x) |

The training scripts and the QuarkGluon loader they ran on live in `provenance/`.
`parT_bench.out` was renamed to `part_bench.out` on import (same content).