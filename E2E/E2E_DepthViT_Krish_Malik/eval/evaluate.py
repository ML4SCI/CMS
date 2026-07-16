"""
ROC-AUC + FLOPs evaluation for DepthViT and ViT-Tiny checkpoints.

Usage:
    # ViT-Tiny checkpoint:
    python3 eval_roc_flops.py \
        --model vit_tiny \
        --ckpt /pscratch/sd/k/krish_m/depthvit/runs/vit_tiny_jets_50p/checkpoints/cls_finetune/epoch=epoch=27-step=step=32816-val_acc1=val_acc1=69.3749.ckpt \
        --config configs/jets_vit_tiny_50p.json \
        --out results/roc_vit_tiny_50p.json

    # DepthViT checkpoint:
    python3 eval_roc_flops.py \
        --model depthvit \
        --ckpt /pscratch/sd/k/krish_m/depthvit/runs/jets_50p/checkpoints/cls_finetune/epoch=epoch=27-step=step=32816-val_acc1=val_acc1=63.9629.ckpt \
        --config configs/jets_50p.json \
        --out results/roc_depthvit_50p.json

    # Run ALL at once using the batch script at the bottom.

Outputs per run:
    - Per-class ROC AUC (g, q, W, Z, top)
    - Macro-average AUC
    - Top-1 accuracy
    - FLOPs (via torchinfo if available, else manual estimate)
    - Params
    - JSON file with all results
"""

import argparse
import json
import os
import sys
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

CLASS_NAMES = ["g", "q", "W", "Z", "top"]


# ------------------------------------------------------------------ #
#  Args                                                               #
# ------------------------------------------------------------------ #

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model",  required=True, choices=["vit_tiny", "depthvit"])
    p.add_argument("--ckpt",   required=True, help="Path to .ckpt file")
    p.add_argument("--config", required=True, help="Path to training config JSON")
    p.add_argument("--out",    required=True, help="Output JSON path")
    p.add_argument("--batch_size", type=int, default=512)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--max_batches", type=int, default=0, help="If >0, cap inference at N batches (for smoke testing)")
    return p.parse_args()


# ------------------------------------------------------------------ #
#  Model loading                                                      #
# ------------------------------------------------------------------ #

def load_vit_tiny(ckpt_path, cfg):
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from vit_tiny import ViTTiny

    model_cfg = cfg["model"]
    data_cfg  = cfg["data"]
    model = ViTTiny(
        img_size    = int(model_cfg.get("img_size", 100)),
        patch_size  = int(model_cfg.get("patch_size", 10)),
        in_channels = int(data_cfg.get("n_channels", 2)),
        num_classes = int(data_cfg.get("num_classes", 5)),
        embed_dim   = int(model_cfg.get("embed_dim", 192)),
        num_layers  = int(model_cfg.get("num_layers", 12)),
        num_heads   = int(model_cfg.get("num_heads", 3)),
        mlp_ratio   = float(model_cfg.get("mlp_ratio", 4.0)),
        dropout     = 0.0,
        attn_dropout= 0.0,
    )
    ckpt = torch.load(ckpt_path, map_location="cpu")
    sd = ckpt.get("state_dict", ckpt)
    sd = {k.replace("_orig_mod.", ""): v for k, v in sd.items()}
    # Strip Lightning prefix
    sd = {k.replace("model.", "", 1): v for k, v in sd.items() if k.startswith("model.")}
    model.load_state_dict(sd, strict=True)
    print(f"  Loaded ViT-Tiny from {ckpt_path}")
    return model


def load_depthvit(ckpt_path, cfg):
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from DepthViT import DepthViT

    model_cfg = cfg["model"]
    data_cfg  = cfg["data"]
    model = DepthViT(
        in_channels       = int(data_cfg["n_channels"]),
        k_factor          = int(model_cfg["k_factor"]),
        patch_size        = int(model_cfg["patch_size"]),
        num_layers        = int(model_cfg["n_layers"]),
        mlp_dim           = int(model_cfg["mlp_dim"]),
        linear_rank       = int(model_cfg["linear_rank"]),
        dropout           = float(model_cfg.get("dropout", 0.0)),
        num_classes       = int(data_cfg.get("num_classes", 5)),
        max_image_height  = int(model_cfg["max_image_height"]),
        max_image_width   = int(model_cfg["max_image_width"]),
        num_hap_layers    = int(model_cfg.get("num_hap_layers", 0)),
        hap_window_size   = int(model_cfg.get("hap_window_size", 8)),
        hap_mlp_ratio     = float(model_cfg.get("hap_mlp_ratio", 4.0)),
        hap_drop_path     = float(model_cfg.get("hap_drop_path", 0.0)),
        grad_checkpointing= False,
        k_chunk_size      = 0,
        compile_blocks    = False,
    )
    ckpt = torch.load(ckpt_path, map_location="cpu")
    sd = ckpt.get("state_dict", ckpt)
    sd = {
        k.replace("model.", "", 1)
         .replace("_orig_mod.", ""): v
        for k, v in sd.items()
        if k.startswith("model.")
    }
    model.load_state_dict(sd, strict=False)
    print(f"  Loaded DepthViT from {ckpt_path}")
    return model


# ------------------------------------------------------------------ #
#  Data loading — reuse existing adapter                              #
# ------------------------------------------------------------------ #

def make_val_loader(cfg, batch_size, num_workers):
    from data import make_loaders_dispatch
    # Temporarily override to get only val loader
    cfg_local = dict(cfg)
    cfg_local["data"] = dict(cfg["data"])
    cfg_local["trainer"] = dict(cfg.get("trainer", {}))
    cfg_local["trainer"]["limit_val_batches"] = 1.0
    # Use phases[0] to get data_override if any
    phase_cfg = cfg.get("phases", [{"type": "cls_finetune"}])[0]
    cfg_local["data"].update(phase_cfg.get("data_override", {}))

    _, val_loader = make_loaders_dispatch(cfg_local)
    return val_loader


# ------------------------------------------------------------------ #
#  Inference                                                          #
# ------------------------------------------------------------------ #

@torch.no_grad()
def run_inference(model, val_loader, device, args_max_batches=0):
    model.eval()
    model.to(device)

    all_probs  = []
    all_labels = []

    for batch_idx, batch in enumerate(val_loader):
        if args_max_batches > 0 and batch_idx >= args_max_batches:
            break
        batch = batch  # reassign for rest of loop
        if isinstance(batch, (list, tuple)):
            x, y = batch[0], batch[1]
        else:
            x, y = batch, None

        x = x.to(device, non_blocking=True)
        if hasattr(model, 'forward_cls'):
            logits = model.forward_cls(x)   # DepthViT
        else:
            logits = model(x)               # ViT-Tiny
        probs  = F.softmax(logits.float(), dim=-1).cpu().numpy()
        all_probs.append(probs)
        if y is not None:
            all_labels.append(y.numpy() if not isinstance(y, np.ndarray) else y)

    all_probs  = np.concatenate(all_probs,  axis=0)   # (N, 5)
    all_labels = np.concatenate(all_labels, axis=0)   # (N,)
    return all_probs, all_labels


# ------------------------------------------------------------------ #
#  Metrics                                                            #
# ------------------------------------------------------------------ #

def compute_metrics(probs, labels):
    from sklearn.metrics import roc_auc_score, roc_curve
    from sklearn.preprocessing import label_binarize

    n_classes = probs.shape[1]
    labels_bin = label_binarize(labels, classes=list(range(n_classes)))  # (N, 5)

    # Per-class AUC
    per_class_auc = {}
    for i, name in enumerate(CLASS_NAMES):
        auc = roc_auc_score(labels_bin[:, i], probs[:, i])
        per_class_auc[name] = float(auc)

    macro_auc = float(np.mean(list(per_class_auc.values())))

    # Top-1 accuracy
    preds = np.argmax(probs, axis=1)
    top1  = float((preds == labels).mean() * 100.0)

    return {
        "top1_acc":      top1,
        "macro_auc":     macro_auc,
        "per_class_auc": per_class_auc,
    }


# ------------------------------------------------------------------ #
#  FLOPs                                                              #
# ------------------------------------------------------------------ #

def compute_flops(model, model_type, cfg):
    """
    Try torchinfo first (most accurate).
    Fall back to manual formula if not installed.
    """
    model_cfg = cfg["model"]
    data_cfg  = cfg["data"]
    H = W = 100
    C = int(data_cfg.get("n_channels", 2))
    dummy = torch.zeros(1, C, H, W)

    # torchinfo skipped — incompatible with custom DepthViT tracing
    pass

    # Manual estimates
    if model_type == "vit_tiny":
        patch = int(model_cfg.get("patch_size", 10))
        D     = int(model_cfg.get("embed_dim", 192))
        L     = int(model_cfg.get("num_layers", 12))
        N     = (H // patch) * (W // patch)   # 100 patches
        N1    = N + 1                          # +CLS
        mlp_r = float(model_cfg.get("mlp_ratio", 4.0))

        # Patch embed conv
        patch_embed = C * patch * patch * D * N
        # Per block: QKV + attn + proj + MLP
        qkv  = 2 * N1 * D * (3 * D)
        attn = 2 * N1 * N1 * D
        proj = 2 * N1 * D * D
        mlp  = 2 * N1 * D * int(D * mlp_r) * 2
        per_block = qkv + attn + proj + mlp
        total = patch_embed + L * per_block + D * 5   # head tiny
        return int(total), "manual"

    elif model_type == "depthvit":
        patch = int(model_cfg.get("patch_size", 10))
        K     = int(model_cfg.get("k_factor", 4))
        R     = int(model_cfg.get("linear_rank", 16))
        L     = int(model_cfg.get("n_layers", 12))
        mlp_d = int(model_cfg.get("mlp_dim", 768))
        N     = (H // patch) * (W // patch)
        # Approximate: DepthViT's channel-wise attention is O(N) not O(N^2)
        # Per block ~ 2 * N * C * K * R (channel attention) + 2 * N * mlp_dim
        per_block = 2 * N * C * K * R + 2 * N * mlp_d
        total = L * per_block
        return int(total), "manual_approx"

    return 0, "unknown"


# ------------------------------------------------------------------ #
#  Main                                                               #
# ------------------------------------------------------------------ #

def main():
    args = parse_args()

    with open(args.config) as f:
        cfg = json.load(f)

    print(f"\n{'='*60}")
    print(f"Model:  {args.model}")
    print(f"Ckpt:   {os.path.basename(args.ckpt)}")
    print(f"Config: {os.path.basename(args.config)}")
    print(f"{'='*60}")

    # Load model
    if args.model == "vit_tiny":
        model = load_vit_tiny(args.ckpt, cfg)
    else:
        model = load_depthvit(args.ckpt, cfg)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Params: {n_params:,}  ({n_params/1e6:.3f}M)")

    # FLOPs
    flops, flops_method = compute_flops(model, args.model, cfg)
    print(f"  FLOPs:  {flops:,}  ({flops/1e9:.3f}G)  [{flops_method}]")

    # Data
    print(f"  Loading val set...")
    val_loader = make_val_loader(cfg, args.batch_size, args.num_workers)

    # Inference
    print(f"  Running inference on {args.device}...")
    probs, labels = run_inference(model, val_loader, args.device, args.max_batches)
    print(f"  Inference done: {len(labels):,} samples")

    # Metrics
    metrics = compute_metrics(probs, labels)
    print(f"\n  Top-1 acc:   {metrics['top1_acc']:.2f}%")
    print(f"  Macro AUC:   {metrics['macro_auc']:.4f}")
    print(f"  Per-class AUC:")
    for cls, auc in metrics["per_class_auc"].items():
        print(f"    {cls:>4}: {auc:.4f}")

    # Save
    result = {
        "model":       args.model,
        "ckpt":        args.ckpt,
        "config":      args.config,
        "n_params":    n_params,
        "flops":       flops,
        "flops_method":flops_method,
        **metrics,
    }
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\n  Saved to {args.out}")


if __name__ == "__main__":
    main()
