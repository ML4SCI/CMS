"""Train LorentzGATr on QuarkGluon: baseline vs optimized.

Runs the baseline in a subprocess (clean Python, no patches) and the
optimized model in the main process, then prints a comparison table.

Usage:
    /root/.venv/bin/python train_lgatr_comparison.py
"""

import json
import os
import subprocess
import sys
import time

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
    gamma=0.95,
    seed=42,
)

# -----------------------------------------------------------------------
# Worker code (runs in subprocess for baseline, exec'd in-process for opt)
# -----------------------------------------------------------------------
_WORKER = r'''
import sys, os, time, gc, warnings, json
import numpy as np

ROOT = {root!r}
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "hybrid_transformer/MAEs/Hybrid_Transformer_Thanh_Nguyen"))
warnings.filterwarnings("ignore")

import torch
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
from src.models import LorentzGATr
from src.configs import LGATrConfig

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

model_cfg = LGATrConfig(
    num_classes=2, embed_dim=128, num_heads=8, num_layers=8,
    num_cls_layers=2, num_mlp_layers=0, hidden_dim=256,
    hidden_mv_channels=8, hidden_s_channels=16,
    attention={{}}, mlp={{}}, dropout=0.1, expansion_factor=4,
    max_num_particles=128, num_particle_features=4,
    mask=False, weights=None, inference=False,
)

torch.manual_seed(SEED)
model = LorentzGATr(model_cfg).to(DEVICE)
n_params = sum(p.numel() for p in model.parameters())

use_mark_step = False

if MODE == "optimized":
    from lgatr_kernels import optimize_lgatr_model
    model, stats = optimize_lgatr_model(
        model, use_compile_patches=True, compile_mode="reduce-overhead")
    with torch.no_grad():
        torch.compiler.cudagraph_mark_step_begin()
        _ = model(torch.randn(BATCH_SIZE, 128, 4, device=DEVICE))
    use_mark_step = True

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
    print(f"Data   : {CFG['train_path']}")
    print(f"Epochs : {CFG['n_epochs']}  |  Batch: {CFG['batch_size']}  |  Seed: {CFG['seed']}")

    # Run baseline first (clean subprocess, no patches)
    baseline = run_mode("baseline")

    # Run optimized (clean subprocess, with patches + compile)
    optimized = run_mode("optimized")

    if "error" in baseline or "error" in optimized:
        print("\n  One or both runs failed. Check output above.")
        return

    # Comparison
    print(f"\n{'='*70}")
    print(f"  COMPARISON")
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
    print()


if __name__ == "__main__":
    main()
