"""End-to-end LGATr benchmark on JetClass ragged CSR shards.

The model under test is ``variants.lgatr_model.LGATrJetClassifier`` — the
self-contained jet tagger built on the official ``lgatr`` package (no
borrowed-tree code).

Part 1 -- Kernel-level microbenchmarks:
  Each primitive (EquiLinear, geometric product, LayerNorm, gated GELU)
  is benchmarked baseline vs optimized at shapes matching the default
  LGATrJetClassifier config (B=128, items=128, mv=8, s=16).

Part 2 -- Model-level tiers:
  1. baseline              -- stock lgatr from pip, zero patches
  2. +kernels              -- FusedEquiLinear + Triton GP (proven faster only)
  3. +compile (with patches)      -- above + compile-friendly patches + torch.compile

Each model tier runs in a **fresh subprocess** so global monkey-patches
from one tier never contaminate another.

Reports latency, speedup, and peak GPU memory throughout.

Data comes from the canonical ``dataloader.ragged_loader`` when ``--pt-dir``
points at ragged CSR ``.pt`` shards; otherwise synthetic four-vectors are used
and every report line says so. The previous QuarkGluon ``.npz`` path was dropped
— that ``data/`` tree no longer exists, and JetClass is the dataset the rest of
the pipeline targets (so ``num_classes`` is now 10, not 2).

Usage:
    python lgatr_kernels/benchmarks/bench_e2e.py \\
        --pt-dir $PSCRATCH/jetclass/pt_ragged/val_5M
    python lgatr_kernels/benchmarks/bench_e2e.py     # synthetic fallback
"""

from __future__ import annotations

import json
import os
import subprocess
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
#: ``variants`` and ``dataloader`` live inside the training package, not at the
#: repo root. ROOT alone makes ``lgatr_kernels`` importable but not ``variants``
#: — that broke when the kernel packages were lifted out of the training
#: package, so workers need both entries.
PKG_ROOT = os.path.join(ROOT, "ml4sci_26")
PYTHON = sys.executable
BS = 128
#: JetClass has 10 classes; the ragged loader emits ``y`` as ``(B, 10)``.
#: This benchmark previously ran on QuarkGluon (binary) via a ``data/`` tree
#: that no longer exists.
NUM_CLASSES = 10
#: Particle-axis width the batch is padded to, so CUDA-graph/compile tiers see
#: a fixed shape.
P = 128


# -----------------------------------------------------------------------
# Worker script executed in a subprocess for each tier
# -----------------------------------------------------------------------

_WORKER = r'''
import gc, json, os, sys, time
ROOT = {root!r}
PKG_ROOT = {pkg_root!r}
# ROOT makes `lgatr_kernels` importable; PKG_ROOT makes `variants` and
# `dataloader` importable. Both are needed when this worker runs as
# `python -c` from any cwd. No borrowed-tree paths.
sys.path.insert(0, ROOT)
sys.path.insert(0, PKG_ROOT)

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

torch.backends.cuda.enable_flash_sdp(True)
torch.backends.cuda.enable_mem_efficient_sdp(True)
torch.backends.cuda.enable_math_sdp(True)
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

DEVICE = "cuda"
BS = {bs}
PT_DIR = {pt_dir!r}
NUM_CLASSES = {num_classes}
P = {particles}
TIER = {tier}

def mark_step():
    fn = getattr(torch.compiler, "cudagraph_mark_step_begin", None)
    if fn is not None:
        fn()

def clear():
    gc.collect(); torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
    if hasattr(torch, "_dynamo"):
        torch._dynamo.reset()

def load_batch():
    """One batch in the canonical ragged_loader contract.

    Real path: ``dataloader.ragged_loader.load_bench_batch`` reads a CSR shard
    and returns ``x (B,16,P)``, ``v (B,4,P)`` raw ``[px,py,pz,E]``,
    ``mask (B,1,P)``, ``y (B,10)`` — no conversion needed, which is the point of
    routing through the loader. ``pad_to=P`` fixes the particle axis so the
    compile tier does not recapture.

    Fallback: synthetic physical four-vectors when PT_DIR is unset.
    """
    if PT_DIR:
        from dataloader.ragged_loader import load_bench_batch
        _x, v, mask, y = load_bench_batch(
            PT_DIR, batch_size=BS, device=DEVICE, pad_to=P,
        )
        return v, mask, y.argmax(dim=1)

    g = torch.Generator(device="cpu").manual_seed(0)
    p3 = torch.randn(BS, 3, P, generator=g) * 10
    mass = torch.rand(BS, 1, P, generator=g) * 0.5 + 0.1
    E = (p3.square().sum(dim=1, keepdim=True) + mass.square()).sqrt()
    v = torch.cat([p3, E], dim=1).to(DEVICE)
    mask = torch.ones(BS, 1, P, device=DEVICE)
    y = torch.randint(0, NUM_CLASSES, (BS,), generator=g).to(DEVICE)
    return v, mask, y

def build_model():
    from variants.lgatr_model import LGATrJetClassifier
    # Config mapping from the retired borrowed-tree LGATrConfig (same
    # LGATr-backbone scale, so benchmark numbers stay comparable):
    #   num_classes=2         -> num_classes=2
    #   num_layers=8          -> num_blocks=8
    #   hidden_mv_channels=8  -> mv_channels=8
    #   hidden_s_channels=16  -> s_channels=16
    #   num_heads=8           -> num_heads=8
    # The old wrapper's class-attention decoder head (embed_dim=128,
    # num_cls_layers=2, hidden_dim=256, dropout, expansion_factor) is
    # replaced by LGATrJetClassifier's masked mean pool + linear head.
    model = LGATrJetClassifier(
        num_classes=NUM_CLASSES, mv_channels=8, s_channels=16,
        num_blocks=8, num_heads=8,
    )
    return model.to(DEVICE)

# Called with the harness convention model(x, v=v, mask=mask). x is None here:
# this arm is four-vector-only and ignores it (see LGATrJetClassifier.forward).
def bench_inference(model, v, mask, warmup=15, timed=40):
    model.eval()
    with torch.no_grad():
        for _ in range(warmup):
            mark_step(); model(None, v=v, mask=mask)
    torch.cuda.synchronize()
    times = []
    with torch.no_grad():
        for _ in range(timed):
            mark_step(); torch.cuda.synchronize()
            t0 = time.perf_counter(); model(None, v=v, mask=mask); torch.cuda.synchronize()
            times.append(time.perf_counter() - t0)
    times.sort()
    return times[len(times) // 2] * 1000

def bench_training(model, v, mask, y, warmup=8, timed=20):
    model.train()
    for _ in range(warmup):
        mark_step(); model.zero_grad(set_to_none=True)
        loss = F.cross_entropy(model(None, v=v, mask=mask), y); loss.backward()
    torch.cuda.synchronize()
    times = []
    for _ in range(timed):
        mark_step(); torch.cuda.synchronize()
        t0 = time.perf_counter()
        model.zero_grad(set_to_none=True)
        loss = F.cross_entropy(model(None, v=v, mask=mask), y); loss.backward()
        torch.cuda.synchronize()
        times.append(time.perf_counter() - t0)
    times.sort()
    return times[len(times) // 2] * 1000

def measure_mem(fn):
    clear()
    fn()
    torch.cuda.synchronize()
    return torch.cuda.max_memory_allocated() / 1e6

v, mask, y = load_batch()
torch.manual_seed(42)

if TIER == 1:
    model = build_model()
elif TIER == 2:
    from lgatr_kernels.primitives import patch_lgatr
    from lgatr_kernels.layers import fuse_equi_linear_layers
    patch_lgatr()
    model = build_model()
    fuse_equi_linear_layers(model)
elif TIER == 3:
    from lgatr_kernels.primitives import patch_lgatr
    from lgatr_kernels.layers import fuse_equi_linear_layers
    from lgatr_kernels.compile_patches import patch_lgatr_compile
    patch_lgatr()
    patch_lgatr_compile()
    model = build_model()
    fuse_equi_linear_layers(model)
    model.eval()
    with torch.no_grad():
        model(None, v=v, mask=mask)
    model = torch.compile(model, mode="reduce-overhead")

params = sum(p.numel() for p in model.parameters())

infer_ms = bench_inference(model, v, mask)

def _infer_mem_fn():
    model.eval()
    with torch.no_grad():
        model(None, v=v, mask=mask)
infer_mem = measure_mem(_infer_mem_fn)

if TIER == 3:
    del model; clear()
    torch.manual_seed(42)
    from lgatr_kernels.primitives import patch_lgatr
    from lgatr_kernels.layers import fuse_equi_linear_layers
    from lgatr_kernels.compile_patches import patch_lgatr_compile
    patch_lgatr(); patch_lgatr_compile()
    model = build_model(); fuse_equi_linear_layers(model)
    model.eval()
    with torch.no_grad():
        model(None, v=v, mask=mask)
    model = torch.compile(model, mode="reduce-overhead")

train_ms = bench_training(model, v, mask, y)

def _train_mem_fn():
    model.train()
    model.zero_grad(set_to_none=True)
    loss = F.cross_entropy(model(None, v=v, mask=mask), y)
    loss.backward()
train_mem = measure_mem(_train_mem_fn)

print(json.dumps({{
    "tier": TIER,
    "params": params,
    "infer_ms": round(infer_ms, 2),
    "train_ms": round(train_ms, 2),
    "infer_mem": round(infer_mem, 1),
    "train_mem": round(train_mem, 1),
}}))
'''


# -----------------------------------------------------------------------
# Kernel-level microbenchmark worker (also runs in subprocess)
# -----------------------------------------------------------------------

_KERNEL_WORKER = r'''
import gc, json, os, sys, time
ROOT = {root!r}
PKG_ROOT = {pkg_root!r}
sys.path.insert(0, ROOT)
sys.path.insert(0, PKG_ROOT)

import torch
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

DEVICE = "cuda"
B, N, MV, S = 128, 128, 8, 16

def clear():
    gc.collect(); torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()

def bench(fn, warmup=20, timed=80):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    times = []
    for _ in range(timed):
        torch.cuda.synchronize()
        t0 = time.perf_counter(); fn(); torch.cuda.synchronize()
        times.append(time.perf_counter() - t0)
    times.sort()
    return times[len(times)//2] * 1e6  # microseconds

def peak_mem(fn):
    clear(); fn(); torch.cuda.synchronize()
    return torch.cuda.max_memory_allocated() / 1e6

results = []

# --- EquiLinear ---
from lgatr.layers.linear import EquiLinear
from lgatr_kernels.layers.fused_linear import FusedEquiLinear

el_base = EquiLinear(MV, MV, in_s_channels=S, out_s_channels=S).cuda().eval()
el_fused = FusedEquiLinear(el_base).cuda().eval()

mv_in = torch.randn(B, N, MV, 16, device=DEVICE)
sc_in = torch.randn(B, N, S, device=DEVICE)

with torch.no_grad():
    t_base = bench(lambda: el_base(mv_in, scalars=sc_in))
    t_fused = bench(lambda: el_fused(mv_in, scalars=sc_in))
    m_base = peak_mem(lambda: el_base(mv_in, scalars=sc_in))
    m_fused = peak_mem(lambda: el_fused(mv_in, scalars=sc_in))
results.append({{"name": "EquiLinear (fwd)", "base_us": round(t_base,1), "opt_us": round(t_fused,1), "base_mb": round(m_base,1), "opt_mb": round(m_fused,1)}})
del el_base, el_fused; clear()

# --- Geometric Product ---
from lgatr.primitives.bilinear import geometric_product as ref_gp
from lgatr_kernels.autograd.geometric_product import triton_geometric_product

gp_x = torch.randn(B, N, MV, 16, device=DEVICE)
gp_y = torch.randn(B, N, MV, 16, device=DEVICE)

with torch.no_grad():
    t_base = bench(lambda: ref_gp(gp_x, gp_y))
    t_opt = bench(lambda: triton_geometric_product(gp_x, gp_y))
    m_base = peak_mem(lambda: ref_gp(gp_x, gp_y))
    m_opt = peak_mem(lambda: triton_geometric_product(gp_x, gp_y))
results.append({{"name": "Geometric Product (fwd)", "base_us": round(t_base,1), "opt_us": round(t_opt,1), "base_mb": round(m_base,1), "opt_mb": round(m_opt,1)}})
del gp_x, gp_y; clear()

# --- Equivariant LayerNorm ---
from lgatr.primitives.normalization import equi_layer_norm as ref_ln
from lgatr_kernels.autograd.equi_layernorm import triton_equi_layer_norm

ln_x = torch.randn(B, N, MV, 16, device=DEVICE)

with torch.no_grad():
    t_base = bench(lambda: ref_ln(ln_x))
    t_opt = bench(lambda: triton_equi_layer_norm(ln_x))
    m_base = peak_mem(lambda: ref_ln(ln_x))
    m_opt = peak_mem(lambda: triton_equi_layer_norm(ln_x))
results.append({{"name": "Equi LayerNorm (fwd)", "base_us": round(t_base,1), "opt_us": round(t_opt,1), "base_mb": round(m_base,1), "opt_mb": round(m_opt,1)}})
del ln_x; clear()

# --- Gated GELU ---
from lgatr.primitives.nonlinearities import gated_gelu as ref_gelu
from lgatr_kernels.autograd.gated_gelu import triton_gated_gelu

gg_x = torch.randn(B, N, MV, 16, device=DEVICE)
gg_gates = gg_x[..., 0:1]

with torch.no_grad():
    t_base = bench(lambda: ref_gelu(gg_x, gg_gates))
    t_opt = bench(lambda: triton_gated_gelu(gg_x))
    m_base = peak_mem(lambda: ref_gelu(gg_x, gg_gates))
    m_opt = peak_mem(lambda: triton_gated_gelu(gg_x))
results.append({{"name": "Gated GELU (fwd)", "base_us": round(t_base,1), "opt_us": round(t_opt,1), "base_mb": round(m_base,1), "opt_mb": round(m_opt,1)}})

print(json.dumps(results))
'''


def run_kernel_microbench() -> list[dict]:
    code = _KERNEL_WORKER.format(root=ROOT, pkg_root=PKG_ROOT)
    result = subprocess.run(
        [PYTHON, "-c", code],
        capture_output=True,
        text=True,
        timeout=300,
    )
    if result.returncode != 0:
        print("  Kernel microbench FAILED")
        for line in result.stderr.strip().split("\n")[-8:]:
            print(f"    {line}")
        return []
    for line in result.stdout.strip().split("\n"):
        line = line.strip()
        if line.startswith("["):
            return json.loads(line)
    return []


def run_tier(tier: int, pt_dir: str = "") -> dict:
    code = _WORKER.format(
        root=ROOT, pkg_root=PKG_ROOT, pt_dir=pt_dir,
        bs=BS, num_classes=NUM_CLASSES, particles=P, tier=tier,
    )
    result = subprocess.run(
        [PYTHON, "-c", code],
        capture_output=True,
        text=True,
        timeout=600,
    )
    if result.returncode != 0:
        print(f"  TIER {tier} FAILED (exit {result.returncode})")
        for line in result.stderr.strip().split("\n")[-10:]:
            print(f"    {line}")
        return {"tier": tier, "error": True}

    for line in result.stdout.strip().split("\n"):
        line = line.strip()
        if line.startswith("{"):
            return json.loads(line)

    print(f"  TIER {tier}: no JSON output found")
    print(f"  stdout: {result.stdout[-500:]}")
    return {"tier": tier, "error": True}


def main():
    import argparse

    import torch

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--pt-dir",
        default=os.environ.get("JETCLASS_BENCH_PT_DIR", ""),
        help="Directory of ragged CSR .pt shards to benchmark on, e.g. "
             "$PSCRATCH/jetclass/pt_ragged/val_5M. Defaults to "
             "$JETCLASS_BENCH_PT_DIR; if neither is set, synthetic jets are "
             "used and the output says so.",
    )
    args = parser.parse_args()

    pt_dir = args.pt_dir
    if pt_dir and not os.path.isdir(pt_dir):
        parser.error(f"--pt-dir is not a directory: {pt_dir}")
    data_desc = (
        f"ragged CSR shards {pt_dir} (padded to P={P})"
        if pt_dir
        else f"SYNTHETIC (no --pt-dir); P={P} fully dense"
    )

    print("=" * 78)
    print(f"  LorentzGATr Benchmark on JetClass (bs={BS})")
    print(f"  GPU: {torch.cuda.get_device_name(0)}")
    print(f"  PyTorch: {torch.__version__}")
    print(f"  Data: {data_desc}")
    print(f"  Each tier runs in a FRESH subprocess (no cross-contamination)")
    print("=" * 78)

    # === Part 1: kernel-level microbenchmarks ===
    print("\n  --- Part 1: Kernel-Level Microbenchmarks ---")
    print(f"  Shape: B={BS}, items=128, mv_channels=8, s_channels=16\n")

    kernels = run_kernel_microbench()
    if kernels:
        print(f"  {'Kernel':<28s} {'Baseline':>10s} {'Optimized':>10s} {'Speedup':>8s} {'BaseMem':>8s} {'OptMem':>8s}")
        print(f"  {'-'*28} {'-'*10} {'-'*10} {'-'*8} {'-'*8} {'-'*8}")
        for k in kernels:
            sp = f"{k['base_us']/k['opt_us']:.2f}x" if k['opt_us'] > 0 else "N/A"
            print(f"  {k['name']:<28s} {k['base_us']:>9.0f}us {k['opt_us']:>9.0f}us {sp:>8s} {k['base_mb']:>7.0f}MB {k['opt_mb']:>7.0f}MB")
    else:
        print("  (kernel microbench failed or returned no data)")

    # === Part 2: model-level tiers ===
    print("\n  --- Part 2: Model-Level Tiers ---\n")

    tier_names = {
        1: "1. baseline (stock lgatr)",
        2: "2. +kernels (fuse+GP)",
        3: "3. +compile (with patches)",
    }

    results = []
    for tier in [1, 2, 3]:
        print(f"\n  Running {tier_names[tier]} ...")
        r = run_tier(tier, pt_dir)
        r["name"] = tier_names[tier]
        results.append(r)
        if "error" not in r:
            print(f"    infer={r['infer_ms']:.2f}ms  train={r['train_ms']:.2f}ms  "
                  f"infer_mem={r['infer_mem']:.0f}MB  train_mem={r['train_mem']:.0f}MB")

    valid = [r for r in results if "error" not in r]
    if not valid:
        print("\n  All tiers failed.")
        return

    base = valid[0]

    print("\n" + "=" * 78)
    print("  RESULTS")
    print("=" * 78)
    print(f"\n  {'Tier':<32s} {'Infer':>8s} {'Spd':>6s} {'IMem':>7s} "
          f"{'Train':>8s} {'Spd':>6s} {'TMem':>7s}")
    print(f"  {'-'*32} {'-'*8} {'-'*6} {'-'*7} {'-'*8} {'-'*6} {'-'*7}")
    for r in results:
        if "error" in r:
            print(f"  {r['name']:<32s} {'FAIL':>8s}")
            continue
        i_sp = f"{base['infer_ms']/r['infer_ms']:.1f}x"
        t_sp = f"{base['train_ms']/r['train_ms']:.1f}x"
        print(
            f"  {r['name']:<32s} {r['infer_ms']:>7.1f}ms {i_sp:>6s} {r['infer_mem']:>6.0f}MB "
            f"{r['train_ms']:>7.1f}ms {t_sp:>6s} {r['train_mem']:>6.0f}MB"
        )

    if valid:
        print(f"\n  Model: LGATrJetClassifier (variants.lgatr_model), {valid[0]['params']:,} params")
    print(f"  Config: 8 LGATr blocks (mv=8, s=16), num_classes={NUM_CLASSES}")
    print(f"  Data: {data_desc}, {BS} jets, raw [px, py, pz, E] four-vectors")
    print(f"  Note: this arm ignores the loader's 16 x-features (four-vector "
          f"only), unlike the ParT arms")
    print(f"  Isolation: each tier ran in a fresh Python subprocess")
    print()


if __name__ == "__main__":
    main()
