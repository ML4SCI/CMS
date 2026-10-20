"""End-to-end Particle Transformer benchmark.

Part 1 -- Kernel-level microbenchmarks:
  Each primitive (pairwise features, attention, fused pair MLP) is
  benchmarked baseline vs optimized at shapes matching the default
  ParT config (N=128, P=128, H=8, D=16).

Part 2 -- Model-level tiers:
  1. baseline              -- stock weaver ParticleTransformer
  2. +kernels              -- OptimizedParticleTransformer (fused attn + pair)
  3. +kernels + graph      -- above wrapped in CUDAGraphInferenceWrapper
  4. +compile              -- torch.compile(mode='reduce-overhead')

Each tier runs in a **fresh subprocess** so global state (torch.compile
caches, Triton JIT caches, monkey-patches) never contaminates another.

Reports latency, speedup, and peak GPU memory.

Data comes from the canonical ``dataloader.ragged_loader`` when ``--pt-dir``
points at ragged CSR ``.pt`` shards; otherwise synthetic jets are used and every
report line says so. Prefer real shards: synthetic batches fill all ``P=128``
slots, whereas real JetClass jets average ~39 particles, so synthetic runs
inflate the ``O(P^2)`` pair pipeline that these kernels exist to accelerate.

Usage:
    python part_kernels/benchmarks/bench_e2e.py \\
        --pt-dir $PSCRATCH/jetclass/pt_ragged/val_5M
    python part_kernels/benchmarks/bench_e2e.py        # synthetic fallback
"""

from __future__ import annotations

import json
import os
import subprocess
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
#: In this repo layout the training package IS this directory (variants/,
#: dataloader/, kernels all top-level), so workers just need ROOT on sys.path.
PKG_ROOT = ROOT
PYTHON = sys.executable

N_DEFAULT = 128
P = 128
#: 16, matching the ``ragged_loader`` contract (``x`` is ``(B, 16, P)``) and
#: ``AblationConfig.input_dim``. Was 17 — the padded loader's old width — which
#: benchmarked a shape the pipeline never produces.
INPUT_DIM = 16
NUM_CLASSES = 10


# -----------------------------------------------------------------------
# Kernel-level microbenchmark worker (runs in subprocess)
# -----------------------------------------------------------------------

_KERNEL_WORKER = r'''
import gc, json, math, os, sys, time
ROOT = {root!r}
PKG_ROOT = {pkg_root!r}
PT_DIR = {pt_dir!r}
INPUT_DIM = {input_dim}
sys.path.insert(0, ROOT)
sys.path.insert(0, PKG_ROOT)

import torch
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

DEVICE = "cuda"
N, P, H, D = {batch}, 128, 8, 16

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
    return times[len(times)//2] * 1e6

def peak_mem(fn):
    clear(); fn(); torch.cuda.synchronize()
    return torch.cuda.max_memory_allocated() / 1e6

def make_physical_4vec(n, p):
    """Real four-vectors from ragged CSR shards when PT_DIR is set.

    pad_to=p keeps the pair kernels at the fixed (n, p) shape the reference
    implementations are timed against, so baseline and optimized see identical
    work.
    """
    if PT_DIR:
        from dataloader.ragged_loader import load_bench_batch
        _x, v, _mask, _y = load_bench_batch(
            PT_DIR, batch_size=n, device=DEVICE, pad_to=p,
            num_features=INPUT_DIM,
        )
        return v
    p3 = torch.randn(n, 3, p, device=DEVICE) * 10
    mass = torch.rand(n, 1, p, device=DEVICE) * 0.5 + 0.1
    E = (p3.square().sum(dim=1, keepdim=True) + mass.square()).sqrt()
    return torch.cat([p3, E], dim=1)

results = []

# --- Pairwise features ---
from weaver.nn.model.ParticleTransformer import pairwise_lv_fts
from part_kernels.triton.pairwise_kernel import fused_pairwise_lv_fts

v = make_physical_4vec(N, P)

def ref_pairwise():
    xi = v.unsqueeze(-1)
    xj = v.unsqueeze(-2)
    return pairwise_lv_fts(xi, xj, num_outputs=4, eps=1e-8)

with torch.no_grad():
    t_base = bench(ref_pairwise)
    t_opt = bench(lambda: fused_pairwise_lv_fts(v, eps=1e-8))
    m_base = peak_mem(ref_pairwise)
    m_opt = peak_mem(lambda: fused_pairwise_lv_fts(v, eps=1e-8))
results.append({{"name": "Pairwise LV (fwd)", "base_us": round(t_base,1), "opt_us": round(t_opt,1), "base_mb": round(m_base,1), "opt_mb": round(m_opt,1)}})
del v; clear()

# --- Attention ---
from part_kernels.autograd.attention import fused_attention_with_bias

NH = N * H
Q = torch.randn(NH, P, D, device=DEVICE)
K = torch.randn(NH, P, D, device=DEVICE)
V = torch.randn(NH, P, D, device=DEVICE)
bias = torch.randn(NH, P, P, device=DEVICE) * 0.1
scale = 1.0 / math.sqrt(D)

def ref_attn():
    S = torch.matmul(Q, K.transpose(-2, -1)) * scale
    S = S + bias
    return torch.matmul(torch.softmax(S, dim=-1), V)

with torch.no_grad():
    t_base = bench(ref_attn)
    t_opt = bench(lambda: fused_attention_with_bias(Q, K, V, bias, None, scale, H))
    m_base = peak_mem(ref_attn)
    m_opt = peak_mem(lambda: fused_attention_with_bias(Q, K, V, bias, None, scale, H))
results.append({{"name": "Attention+Bias (fwd)", "base_us": round(t_base,1), "opt_us": round(t_opt,1), "base_mb": round(m_base,1), "opt_mb": round(m_opt,1)}})
del Q, K, V, bias; clear()

# --- Fused pair MLP (eval) ---
from weaver.nn.model.ParticleTransformer import ParticleTransformer
from part_kernels.layers.fused_pair_mlp import FusedPairMLP

tmp = ParticleTransformer(
    input_dim=INPUT_DIM, num_classes=10, pair_input_dim=4, pair_extra_dim=0,
    embed_dims=[128, 512, 128], pair_embed_dims=[64, 64, 64],
    num_heads=8, num_layers=2, num_cls_layers=1,
    trim=False, for_inference=True, use_amp=False,
).to(DEVICE).eval()

embed_seq = tmp.pair_embed.embed
fused_mlp = FusedPairMLP(embed_seq, eps=1e-8).to(DEVICE).eval()
v = make_physical_4vec(N, P)

def ref_pair_embed():
    xi = v.unsqueeze(-1); xj = v.unsqueeze(-2)
    pair_fts = pairwise_lv_fts(xi, xj, num_outputs=4, eps=1e-8)
    return embed_seq(pair_fts.view(N, 4, P*P)).view(N, -1, P, P)

with torch.no_grad():
    t_base = bench(ref_pair_embed, warmup=10, timed=40)
    t_opt = bench(lambda: fused_mlp(v), warmup=10, timed=40)
    m_base = peak_mem(ref_pair_embed)
    m_opt = peak_mem(lambda: fused_mlp(v))
results.append({{"name": "PairEmbed+MLP (fwd)", "base_us": round(t_base,1), "opt_us": round(t_opt,1), "base_mb": round(m_base,1), "opt_mb": round(m_opt,1)}})

print(json.dumps(results))
'''


# -----------------------------------------------------------------------
# Model-level tier worker (runs in subprocess)
# -----------------------------------------------------------------------

_WORKER = r'''
import copy, gc, json, os, sys, time
ROOT = {root!r}
PKG_ROOT = {pkg_root!r}
PT_DIR = {pt_dir!r}
sys.path.insert(0, ROOT)
sys.path.insert(0, PKG_ROOT)

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
P = {particles}
INPUT_DIM = {input_dim}
NUM_CLASSES = {num_classes}
TIER = {tier}

def mark_step():
    fn = getattr(torch.compiler, "cudagraph_mark_step_begin", None)
    if fn is not None:
        fn()

def clear():
    gc.collect(); torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
    if hasattr(torch, "_dynamo"):
        torch._dynamo.reset()

def make_inputs(for_train=False):
    """One batch, from real ragged CSR shards when PT_DIR is set.

    pad_to=P is mandatory here, not cosmetic: tiers 3 and 4 capture a CUDA
    graph / compile for a fixed shape, so the particle axis must not vary
    between the warmup batch and the timed batch.
    """
    if PT_DIR:
        from dataloader.ragged_loader import load_bench_batch
        x, v, mask, y = load_bench_batch(
            PT_DIR, batch_size=BS, device=DEVICE, pad_to=P,
            num_features=INPUT_DIM,
        )
        if for_train:
            return x, v, mask, y.argmax(dim=1)
        return x, v, mask

    # Synthetic fallback: physical four-vectors, every slot occupied. Note this
    # is P=128 fully dense, which is far wider than a real JetClass batch
    # (mean multiplicity ~39), so it overstates the O(P^2) pair cost.
    p3 = torch.randn(BS, 3, P, device=DEVICE) * 10
    mass = torch.rand(BS, 1, P, device=DEVICE) * 0.5 + 0.1
    energy = (p3.square().sum(dim=1, keepdim=True) + mass.square()).sqrt()
    v = torch.cat([p3, energy], dim=1)
    x = torch.randn(BS, INPUT_DIM, P, device=DEVICE)
    mask = torch.ones(BS, 1, P, device=DEVICE, dtype=torch.float32)
    if for_train:
        labels = torch.randint(0, NUM_CLASSES, (BS,), device=DEVICE)
        return x, v, mask, labels
    return x, v, mask

def build_baseline(for_inference):
    from weaver.nn.model.ParticleTransformer import ParticleTransformer
    return ParticleTransformer(
        input_dim=INPUT_DIM, num_classes=NUM_CLASSES,
        pair_input_dim=4, pair_extra_dim=0,
        embed_dims=[128, 512, 128], pair_embed_dims=[64, 64, 64],
        num_heads=8, num_layers=8, num_cls_layers=2,
        trim=False, for_inference=for_inference, use_amp=False,
    ).to(DEVICE)

def bench_inference(model, x, v, mask, warmup=15, timed=40):
    model.eval()
    with torch.no_grad():
        for _ in range(warmup):
            mark_step(); model(x, v=v, mask=mask)
    torch.cuda.synchronize()
    times = []
    with torch.no_grad():
        for _ in range(timed):
            mark_step(); torch.cuda.synchronize()
            t0 = time.perf_counter(); model(x, v=v, mask=mask); torch.cuda.synchronize()
            times.append(time.perf_counter() - t0)
    times.sort()
    return times[len(times) // 2] * 1000

def bench_training(model, x, v, mask, labels, warmup=8, timed=20):
    model.train()
    for _ in range(warmup):
        mark_step(); model.zero_grad(set_to_none=True)
        loss = F.cross_entropy(model(x, v=v, mask=mask), labels); loss.backward()
    torch.cuda.synchronize()
    times = []
    for _ in range(timed):
        mark_step(); torch.cuda.synchronize()
        t0 = time.perf_counter()
        model.zero_grad(set_to_none=True)
        loss = F.cross_entropy(model(x, v=v, mask=mask), labels); loss.backward()
        torch.cuda.synchronize()
        times.append(time.perf_counter() - t0)
    times.sort()
    return times[len(times) // 2] * 1000

def measure_mem(fn):
    clear()
    fn()
    torch.cuda.synchronize()
    return torch.cuda.max_memory_allocated() / 1e6

torch.manual_seed(42)

if TIER == 1:
    model = build_baseline(for_inference=True)
elif TIER == 2:
    from part_kernels import OptimizedParticleTransformer
    model = OptimizedParticleTransformer.from_pretrained(build_baseline(for_inference=True)).to(DEVICE)
elif TIER == 3:
    from part_kernels import OptimizedParticleTransformer, CUDAGraphInferenceWrapper
    base = build_baseline(for_inference=True)
    opt = OptimizedParticleTransformer.from_pretrained(base).to(DEVICE).eval()
    x_w, v_w, mask_w = make_inputs()
    model = CUDAGraphInferenceWrapper(opt, batch_size=BS, max_particles=P,
                                       input_dim=INPUT_DIM, num_classes=NUM_CLASSES)
    model.capture()
elif TIER == 4:
    from part_kernels import OptimizedParticleTransformer
    base = build_baseline(for_inference=False)
    model = OptimizedParticleTransformer.from_pretrained(base).to(DEVICE)
    model.eval()
    x_w, v_w, mask_w = make_inputs()
    with torch.no_grad():
        model(x_w, v=v_w, mask=mask_w)
    model = torch.compile(model, mode="reduce-overhead")

params = sum(p.numel() for p in model.parameters())

x_i, v_i, mask_i = make_inputs()

if TIER == 3:
    def infer_fn():
        with torch.no_grad():
            return model(x_i, v_i, mask_i)
else:
    def infer_fn():
        model.eval()
        with torch.no_grad():
            return model(x_i, v=v_i, mask=mask_i)

infer_ms = bench_inference(model, x_i, v_i, mask_i) if TIER != 3 else -1.0
if TIER == 3:
    with torch.no_grad():
        for _ in range(15):
            model(x_i, v_i, mask_i)
    torch.cuda.synchronize()
    times = []
    with torch.no_grad():
        for _ in range(40):
            torch.cuda.synchronize()
            t0 = time.perf_counter(); model(x_i, v_i, mask_i); torch.cuda.synchronize()
            times.append(time.perf_counter() - t0)
    times.sort()
    infer_ms = times[len(times) // 2] * 1000

infer_mem = measure_mem(infer_fn)

train_ms = -1.0
train_mem = -1.0
if TIER in (1, 2, 4):
    if TIER == 4:
        del model; clear()
        torch.manual_seed(42)
        from part_kernels import OptimizedParticleTransformer
        base = build_baseline(for_inference=False)
        model = OptimizedParticleTransformer.from_pretrained(base).to(DEVICE)
        model.eval()
        x_w, v_w, mask_w = make_inputs()
        with torch.no_grad():
            model(x_w, v=v_w, mask=mask_w)
        model = torch.compile(model, mode="reduce-overhead")

    x_t, v_t, mask_t, labels = make_inputs(for_train=True)
    train_ms = bench_training(model, x_t, v_t, mask_t, labels)
    def _train_fn():
        model.train()
        model.zero_grad(set_to_none=True)
        loss = F.cross_entropy(model(x_t, v=v_t, mask=mask_t), labels)
        loss.backward()
    train_mem = measure_mem(_train_fn)

print(json.dumps({{
    "tier": TIER,
    "params": params,
    "infer_ms": round(infer_ms, 2),
    "train_ms": round(train_ms, 2),
    "infer_mem": round(infer_mem, 1),
    "train_mem": round(train_mem, 1),
}}))
'''


def run_kernel_microbench(batch: int, pt_dir: str = "") -> list[dict]:
    code = _KERNEL_WORKER.format(
        root=ROOT, pkg_root=PKG_ROOT, pt_dir=pt_dir,
        batch=batch, input_dim=INPUT_DIM,
    )
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


def run_tier(tier: int, batch: int, pt_dir: str = "") -> dict:
    code = _WORKER.format(
        root=ROOT, pkg_root=PKG_ROOT, pt_dir=pt_dir,
        bs=batch, particles=P,
        input_dim=INPUT_DIM, num_classes=NUM_CLASSES, tier=tier,
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
    parser.add_argument(
        "--batch", type=int, default=N_DEFAULT, help="Batch size (jets).",
    )
    args = parser.parse_args()

    batch = args.batch
    pt_dir = args.pt_dir
    if pt_dir and not os.path.isdir(pt_dir):
        parser.error(f"--pt-dir is not a directory: {pt_dir}")
    data_desc = (
        f"ragged CSR shards {pt_dir} (padded to P={P})"
        if pt_dir
        else f"SYNTHETIC (no --pt-dir); P={P} fully dense, overstates O(P^2)"
    )

    print("=" * 78)
    print(f"  Particle Transformer Benchmark (N={batch}, P={P})")
    print(f"  GPU: {torch.cuda.get_device_name(0)}")
    print(f"  PyTorch: {torch.__version__}")
    print(f"  Data: {data_desc}")
    print(f"  Each tier runs in a FRESH subprocess (no cross-contamination)")
    print("=" * 78)

    # === Part 1: kernel-level microbenchmarks ===
    print("\n  --- Part 1: Kernel-Level Microbenchmarks ---")
    print(f"  Shape: N={batch}, P={P}, H=8, D=16\n")

    kernels = run_kernel_microbench(batch, pt_dir)
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
        1: "1. baseline (stock ParT)",
        2: "2. +kernels (fused attn+pair)",
        3: "3. +kernels + CUDA graph",
        4: "4. +compile (reduce-overhead)",
    }

    results = []
    for tier in [1, 2, 3, 4]:
        print(f"\n  Running {tier_names[tier]} ...")
        r = run_tier(tier, batch, pt_dir)
        r["name"] = tier_names[tier]
        results.append(r)
        if "error" not in r:
            train_str = f"train={r['train_ms']:.2f}ms" if r['train_ms'] > 0 else "train=N/A"
            print(f"    infer={r['infer_ms']:.2f}ms  {train_str}  "
                  f"infer_mem={r['infer_mem']:.0f}MB")

    valid = [r for r in results if "error" not in r]
    if not valid:
        print("\n  All tiers failed.")
        return

    base = valid[0]

    print("\n" + "=" * 78)
    print("  RESULTS")
    print("=" * 78)
    print(f"\n  {'Tier':<34s} {'Infer':>8s} {'Spd':>6s} {'IMem':>7s} "
          f"{'Train':>8s} {'Spd':>6s} {'TMem':>7s}")
    print(f"  {'-'*34} {'-'*8} {'-'*6} {'-'*7} {'-'*8} {'-'*6} {'-'*7}")
    for r in results:
        if "error" in r:
            print(f"  {r['name']:<34s} {'FAIL':>8s}")
            continue
        i_sp = f"{base['infer_ms']/r['infer_ms']:.1f}x" if r['infer_ms'] > 0 else "N/A"
        if r['train_ms'] > 0 and base['train_ms'] > 0:
            t_sp = f"{base['train_ms']/r['train_ms']:.1f}x"
            t_str = f"{r['train_ms']:>7.1f}ms {t_sp:>6s} {r['train_mem']:>6.0f}MB"
        else:
            t_str = f"{'N/A':>7s} {'--':>6s} {'--':>7s}"
        print(
            f"  {r['name']:<34s} {r['infer_ms']:>7.1f}ms {i_sp:>6s} {r['infer_mem']:>6.0f}MB "
            f"{t_str}"
        )

    if valid:
        print(f"\n  Model: ParticleTransformer, {valid[0]['params']:,} params")
    print(f"  Config: 8 blocks, 2 cls blocks, embed=[128,512,128], pair=[64,64,64]")
    print(f"  Data: {data_desc} (N={batch}, {INPUT_DIM} features)")
    print(f"  Isolation: each tier ran in a fresh Python subprocess")
    print()


if __name__ == "__main__":
    main()
