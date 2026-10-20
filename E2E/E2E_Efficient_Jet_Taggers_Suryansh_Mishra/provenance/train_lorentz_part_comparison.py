"""Train LorentzParT (Hybrid) on QuarkGluon: baseline vs optimized.

Runs each mode in a clean subprocess so monkey-patches and torch.compile
caches never contaminate the other run, then prints a comparison table.

Optimized mode applies:
  - Fused pairwise Lorentz kernel  (part_kernels)
  - Fused attention with bias      (part_kernels)
  - Fused EquiLinear               (lgatr_kernels)
  - Compile-friendly lgatr patches (lgatr_kernels)
  - torch.compile(mode="reduce-overhead")

Usage:
    /root/.venv/bin/python train_lorentz_part_comparison.py
"""

import json
import os
import subprocess
import sys

ROOT = os.path.abspath(os.path.dirname(__file__))
PYTHON = sys.executable

# -----------------------------------------------------------------------
# Shared config
# -----------------------------------------------------------------------
CFG = dict(
    root=ROOT,
    train_path=os.path.join(ROOT, "data/QuarkGluon/train.npz"),
    val_path=os.path.join(ROOT, "data/QuarkGluon/val.npz"),
    batch_size=512,
    num_workers=4,
    n_epochs=10,
    lr=1e-3,
    gamma=0.736,
    seed=42,
)

# -----------------------------------------------------------------------
# Worker code (runs in a fresh subprocess for each mode)
# -----------------------------------------------------------------------
_WORKER = r'''
import sys, os, time, gc, math, types, warnings, json
import numpy as np

ROOT = {root!r}
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "hybrid_transformer/MAEs/Hybrid_Transformer_Thanh_Nguyen"))
warnings.filterwarnings("ignore")

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
DEVICE = torch.device("cuda")

TRAIN_PATH = {train_path!r}
VAL_PATH = {val_path!r}
BATCH_SIZE = {batch_size}
NUM_WORKERS = {num_workers}
N_EPOCHS = {n_epochs}
LR = {lr}
GAMMA = {gamma}
SEED = {seed}
MODE = {mode!r}

from quarkgluon_dataset import QuarkGluonDataset
from src.models import LorentzParT
from src.configs import LorentzParTConfig

def compute_norm_dict(path):
    raw = np.load(path)
    X = raw["X"]
    mask = np.any(X != 0, axis=-1)
    nd = {{}}
    for i, name in enumerate(["pT", "eta", "phi", "energy"]):
        vals = X[..., i][mask]
        nd[name] = (float(vals.mean()), float(vals.std()))
    return nd

norm_dict = compute_norm_dict(TRAIN_PATH)
train_ds = QuarkGluonDataset(TRAIN_PATH, [True, False, False, True], norm_dict)
val_ds = QuarkGluonDataset(VAL_PATH, [True, False, False, True], norm_dict)
train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                          num_workers=NUM_WORKERS, pin_memory=True)
val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False,
                        num_workers=NUM_WORKERS, pin_memory=True)

model_cfg = LorentzParTConfig(
    num_classes=2, embed_dim=128, num_heads=8, num_layers=8,
    num_cls_layers=2, num_mlp_layers=0, hidden_dim=256,
    hidden_mv_channels=8, hidden_s_channels=16,
    attention={{}}, mlp={{}}, dropout=0.1, expansion_factor=4,
    max_num_particles=128, num_particle_features=4,
    pair_embed_dims=[64, 64, 64],
    mask=False, weights=None, inference=False,
)

torch.manual_seed(SEED)
model = LorentzParT(config=model_cfg).to(DEVICE)
n_params = sum(p.numel() for p in model.parameters())

use_mark_step = False

if MODE == "optimized":
    from part_kernels.triton.pairwise_kernel import fused_pairwise_lv_fts
    from part_kernels.autograd.attention import fused_attention_with_bias
    from lgatr_kernels.layers import fuse_equi_linear_layers
    from lgatr_kernels.compile_patches import patch_lgatr_compile
    from lgatr_kernels.primitives import patch_lgatr

    # --- 1. Fused pairwise kernel on ParticleProcessor._get_interaction ---
    from src.models.processor import ParticleProcessor

    def _fused_get_interaction(self, x):
        mask = x[..., 3] > 0
        pT, eta, phi, energy = x[..., 0], x[..., 1], x[..., 2], x[..., 3]
        px = pT * torch.cos(phi)
        py = pT * torch.sin(phi)
        pz = pT * torch.sinh(eta)
        v = torch.stack([px, py, pz, energy], dim=1)
        with torch.no_grad():
            pair_fts = fused_pairwise_lv_fts(v, eps=1e-8)
        # kernel [lnkt,lnz,lndelta,lnm2] -> hybrid [lndelta,lnkt,lnz,lnm2]
        pair_fts = pair_fts.permute(0, 2, 3, 1)
        pair_fts = pair_fts[..., [2, 0, 1, 3]]
        U = torch.full_like(pair_fts, fill_value=-1e9)
        valid_pairs = mask.unsqueeze(2) & mask.unsqueeze(1)
        U[valid_pairs] = pair_fts[valid_pairs]
        idx = torch.arange(U.size(1), device=U.device)
        U[:, idx, idx, :] = 0
        return U

    for module in model.modules():
        if isinstance(module, ParticleProcessor):
            module._get_interaction = types.MethodType(_fused_get_interaction, module)

    # --- 2. Fused attention on ParticleAttentionBlock ---
    from src.models.particle_transformer import ParticleAttentionBlock

    def _make_fused_attn_forward(block):
        pmha = block.pmha
        embed_dim = block.embed_dim
        num_heads = block.num_heads
        head_dim = embed_dim // num_heads
        in_proj_weight = pmha.in_proj_weight
        in_proj_bias = pmha.in_proj_bias
        out_proj = pmha.out_proj

        def fused_forward(self, x, padding_mask, U=None):
            residual = x
            x = self.layernorm1(x)
            B, N, C = x.shape
            qkv = F.linear(x, in_proj_weight, in_proj_bias)
            q, k, v = qkv.chunk(3, dim=-1)
            q = q.view(B, N, num_heads, head_dim).permute(0, 2, 1, 3).reshape(B * num_heads, N, head_dim)
            k = k.view(B, N, num_heads, head_dim).permute(0, 2, 1, 3).reshape(B * num_heads, N, head_dim)
            v = v.view(B, N, num_heads, head_dim).permute(0, 2, 1, 3).reshape(B * num_heads, N, head_dim)
            scale = 1.0 / math.sqrt(head_dim)
            pad_mask = padding_mask.bool() if padding_mask is not None else None
            out = fused_attention_with_bias(q, k, v, U, pad_mask, scale, num_heads)
            out = out.view(B, num_heads, N, head_dim).permute(0, 2, 1, 3).reshape(B, N, C)
            out = out_proj(out)
            x = self.layernorm2(out)
            x = self.dropout(x)
            x += residual
            x = self.feedforward(x)
            return x
        return fused_forward

    for module in model.modules():
        if isinstance(module, ParticleAttentionBlock):
            fused_fwd = _make_fused_attn_forward(module)
            module.forward = types.MethodType(fused_fwd, module)

    # --- 3. Fused EquiLinear ---
    patch_lgatr()
    fuse_equi_linear_layers(model)

    # --- 4. Compile-friendly patches + torch.compile ---
    patch_lgatr_compile()
    with torch.no_grad():
        torch.compiler.cudagraph_mark_step_begin()
        _ = model(torch.randn(BATCH_SIZE, 128, 4, device=DEVICE))
    model = torch.compile(model, mode="reduce-overhead")
    use_mark_step = True

    print(f"  [optimized] Patches applied: fused pairwise, fused attention, "
          f"fused EquiLinear, compile patches, torch.compile", flush=True)

def accuracy(logits, y):
    return (logits.argmax(1) == y.argmax(1)).float().mean().item()

optimizer = torch.optim.Adam(model.parameters(), lr=LR)
scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=GAMMA)

gc.collect(); torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
total_start = time.time()
epoch_times = []
best_val_acc = 0.0
best_val_loss = float("inf")

for epoch in range(N_EPOCHS):
    model.train()
    tl, ta, tc = 0.0, 0.0, 0
    torch.cuda.synchronize(); t0 = time.time()
    for X, y in train_loader:
        X = X.to(DEVICE, non_blocking=True)
        y = y.to(DEVICE, non_blocking=True)
        if use_mark_step:
            torch.compiler.cudagraph_mark_step_begin()
        optimizer.zero_grad()
        logits = model(X)
        loss = F.cross_entropy(logits, y.argmax(1))
        loss.backward()
        optimizer.step()
        bsz = y.size(0); tl += loss.item()*bsz; ta += accuracy(logits,y)*bsz; tc += bsz
    model.eval()
    vl, va, vc = 0.0, 0.0, 0
    with torch.no_grad():
        for X, y in val_loader:
            X = X.to(DEVICE, non_blocking=True)
            y = y.to(DEVICE, non_blocking=True)
            if use_mark_step:
                torch.compiler.cudagraph_mark_step_begin()
            logits = model(X)
            loss = F.cross_entropy(logits, y.argmax(1))
            bsz = y.size(0); vl += loss.item()*bsz; va += accuracy(logits,y)*bsz; vc += bsz
    torch.cuda.synchronize()
    elapsed = time.time() - t0
    scheduler.step()
    tl/=tc; ta/=tc; vl/=vc; va/=vc
    epoch_times.append(elapsed)
    if va > best_val_acc: best_val_acc = va
    if vl < best_val_loss: best_val_loss = vl
    lr = optimizer.param_groups[0]["lr"]
    print(f"  Epoch {{epoch+1:>2}}/{{N_EPOCHS}}  "
          f"train_loss={{tl:.4f}}  val_loss={{vl:.4f}}  "
          f"train_acc={{ta:.4f}}  val_acc={{va:.4f}}  "
          f"lr={{lr:.2e}}  time={{elapsed:.1f}}s", flush=True)

torch.cuda.synchronize()
total_time = time.time() - total_start
peak_mem = torch.cuda.max_memory_allocated() / (1024**2)

print(json.dumps({{
    "mode": MODE,
    "params": n_params,
    "total_time": round(total_time, 1),
    "avg_epoch": round(np.mean(epoch_times), 1),
    "best_val_acc": round(best_val_acc, 4),
    "best_val_loss": round(best_val_loss, 4),
    "peak_mem_mb": round(peak_mem, 0),
    "epoch_times": [round(t, 1) for t in epoch_times],
}}))
'''


def run_mode(mode: str) -> dict:
    code = _WORKER.format(**CFG, mode=mode)
    print(f"\n{'='*70}")
    print(f"  Running: {mode.upper()}")
    print(f"{'='*70}", flush=True)

    proc = subprocess.Popen(
        [PYTHON, "-u", "-c", code],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    json_line = None
    for line in proc.stdout:
        line_s = line.rstrip()
        print(line_s, flush=True)
        if line_s.strip().startswith("{"):
            json_line = line_s.strip()
    proc.wait()

    if json_line:
        return json.loads(json_line)

    print(f"  WARNING: no JSON output found for {mode}")
    return {"mode": mode, "error": True}


def main():
    import torch
    print(f"GPU    : {torch.cuda.get_device_name(0)}")
    print(f"PyTorch: {torch.__version__}")
    print(f"Model  : LorentzParT (Hybrid)")
    print(f"Data   : {CFG['train_path']}")
    print(f"Epochs : {CFG['n_epochs']}  |  Batch: {CFG['batch_size']}  |  Seed: {CFG['seed']}")

    baseline = run_mode("baseline")
    optimized = run_mode("optimized")

    if "error" in baseline or "error" in optimized:
        print("\n  One or both runs failed. Check output above.")
        return

    print(f"\n{'='*70}")
    print(f"  COMPARISON: LorentzParT (Hybrid) on QuarkGluon")
    print(f"{'='*70}\n")

    rows = [
        ("Params", f"{baseline['params']:,}", f"{optimized['params']:,}"),
        ("Best val accuracy", f"{baseline['best_val_acc']:.4f}", f"{optimized['best_val_acc']:.4f}"),
        ("Best val loss", f"{baseline['best_val_loss']:.4f}", f"{optimized['best_val_loss']:.4f}"),
        ("Total time", f"{baseline['total_time']:.1f}s", f"{optimized['total_time']:.1f}s"),
        ("Avg epoch time", f"{baseline['avg_epoch']:.1f}s", f"{optimized['avg_epoch']:.1f}s"),
        ("Training speedup", "1.00x", f"{baseline['avg_epoch'] / optimized['avg_epoch']:.2f}x"),
        ("Peak GPU memory", f"{baseline['peak_mem_mb']:.0f} MB", f"{optimized['peak_mem_mb']:.0f} MB"),
    ]

    print(f"  {'Metric':<24s} {'Baseline':>16s} {'Optimized':>16s}")
    print(f"  {'-'*24} {'-'*16} {'-'*16}")
    for label, base, opt in rows:
        print(f"  {label:<24s} {base:>16s} {opt:>16s}")

    print(f"\n  Optimized = fused pairwise + fused attention (part_kernels)")
    print(f"           + fused EquiLinear + compile patches (lgatr_kernels)")
    print(f"           + torch.compile(mode='reduce-overhead')")
    print()


if __name__ == "__main__":
    main()
