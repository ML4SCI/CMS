#!/usr/bin/env python3
"""
count_params.py — validate DepthViT parameter count and find a scaling config.

Run on the login node, CPU-only, no GPU needed:
    python3 count_params.py --config configs/config_<...>.json
    python3 count_params.py --config configs/config_<...>.json --sweep-k 4 32
    python3 count_params.py --config configs/config_<...>.json --k 27          # try a specific k_factor
    python3 count_params.py --config configs/config_<...>.json --k 20 --layers 16

This builds the *classification* model (DepthViT) directly from config values,
mirroring the trainer's config->kwargs mapping. If the trainer uses different
keys, fix the MAP below to match it exactly — that is the piping check.
"""
import argparse, json
import torch
from DepthViT import DepthViT

# ---- config-key -> DepthViT constructor-arg mapping --------------------------
# This MUST match how imagenet_trainer.py builds the model. The one to watch:
# config uses "n_layers" but the constructor arg is "num_layers".
def build_model(cfg, k_override=None, layers_override=None):
    m = cfg["model"]
    d = cfg["data"]
    k_factor = k_override if k_override is not None else m["k_factor"]
    num_layers = layers_override if layers_override is not None else m["n_layers"]
    return DepthViT(
        patch_size       = m["patch_size"],
        in_channels      = d["n_channels"],
        k_factor         = k_factor,
        num_layers       = num_layers,           # <-- config key is "n_layers"
        mlp_dim          = m["mlp_dim"],
        linear_rank      = m["linear_rank"],
        max_image_height = m["max_image_height"],
        max_image_width  = m["max_image_width"],
        dropout          = m.get("dropout", 0.0),
        num_classes      = d["num_classes"],
        num_hap_layers   = m.get("num_hap_layers", 0),
        hap_window_size  = m.get("hap_window_size", 8),
        hap_mlp_ratio    = m.get("hap_mlp_ratio", 4.0),
        hap_drop_path    = m.get("hap_drop_path", 0.0),
        grad_checkpointing = False,               # irrelevant to param count
        k_chunk_size     = m.get("k_chunk_size", 0),
        compile_blocks   = False,                 # skip torch.compile during counting
        hap_alpha        = m.get("hap_alpha", 1.0),
        hap_learnable_alpha = m.get("hap_learnable_alpha", False),
    )

def count(model):
    total = sum(p.numel() for p in model.parameters())
    train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, train

def breakdown(model):
    groups = {}
    for name, p in model.named_parameters():
        top = name.split(".")[0]
        # split encoder into sub-parts for readability
        if top == "encoder":
            parts = name.split(".")
            top = "encoder." + (parts[1] if len(parts) > 1 else "")
            if "blocks" in name:
                top = "encoder.blocks(enc+hap)"
            elif "enc_pos_embedding" in name:
                top = "encoder.pos_embedding"
            elif name.endswith("encoder.ln.weight") or name.endswith("encoder.ln.bias"):
                top = "encoder.final_ln"
        groups[top] = groups.get(top, 0) + p.numel()
    return groups

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--k", type=int, default=None, help="override k_factor")
    ap.add_argument("--layers", type=int, default=None, help="override n_layers")
    ap.add_argument("--sweep-k", nargs=2, type=int, metavar=("LO", "HI"),
                    help="sweep k_factor over [LO, HI] and print param counts")
    ap.add_argument("--target", type=float, default=1_000_000, help="target param count for sweep")
    args = ap.parse_args()

    cfg = json.load(open(args.config))

    if args.sweep_k:
        lo, hi = args.sweep_k
        print(f"{'k_factor':>9} | {'hidden_dim':>10} | {'params':>12} | {'M':>7} | vs target")
        print("-" * 60)
        best = None
        for k in range(lo, hi + 1):
            mdl = build_model(cfg, k_override=k, layers_override=args.layers)
            tot, _ = count(mdl)
            mark = ""
            if best is None or abs(tot - args.target) < abs(best[1] - args.target):
                best = (k, tot)
            print(f"{k:>9} | {mdl.hidden_dim:>10} | {tot:>12,} | {tot/1e6:>7.3f} | {tot/args.target:>5.2f}x")
        print("-" * 60)
        bk, bt = best
        print(f"closest to {args.target/1e6:.2f}M -> k_factor={bk}  ({bt:,} params, {bt/1e6:.3f}M)")
        return

    model = build_model(cfg, k_override=args.k, layers_override=args.layers)
    tot, train = count(model)
    print(f"config:      {args.config}")
    print(f"k_factor:    {model.k_factor}   (hidden_dim = in_channels*k_factor = {model.hidden_dim})")
    print(f"num_layers:  {args.layers if args.layers else cfg['model']['n_layers']}"
          f"   num_hap_layers: {cfg['model'].get('num_hap_layers', 0)}")
    print(f"total params:     {tot:,}  ({tot/1e6:.4f}M)")
    print(f"trainable params: {train:,}")
    print("\nbreakdown by module:")
    for name, n in sorted(breakdown(model).items(), key=lambda x: -x[1]):
        print(f"  {name:<28} {n:>12,}  ({100*n/tot:5.1f}%)")

if __name__ == "__main__":
    main()
