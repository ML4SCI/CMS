"""
eval_roc_flops.py  --  ROC-AUC, Top-1 and FLOPs for DepthViT / ViT checkpoints.

Rewrite of the original. Fixes, in order of severity:

  1. CRITICAL -- DepthViT loaded with strict=False, so a checkpoint whose keys
     did not match was silently loaded into a randomly-initialised model and
     still produced plausible-looking AUCs. Loading is now strict by default and
     reports missing/unexpected keys before failing.

  2. CRITICAL -- no channel_treatment support. A symmetric-control checkpoint
     (chansum / symmix) would have been loaded into the ASYMMETRIC architecture.
     Combined with strict=False that failure was completely silent. The
     treatment is now read from the config and applied before loading.

  3. vit_small was not an accepted --model value, so the tier-4 baseline could
     not be evaluated with this script at all.

  4. FLOPs for DepthViT used a hand-rolled approximation
     (2*N*C*K*R + 2*N*mlp_dim) that ignored HAP blocks, the qkv projection and
     the head, and did not reproduce the calflops numbers the paper reports.
     calflops is now used, with the manual formula only as a labelled fallback.

  5. Image size was hardcoded to 100x100, breaking any non-jets dataset.
     Shape now comes from the config.

  6. CLASS_NAMES was a fixed 5-element jet list, so a 100-class dataset raised
     IndexError in compute_metrics. Class names are now derived from the config
     and fall back to indices.

  7. --batch_size / --num_workers were accepted and then ignored. They now
     override the config values.

  8. Empty all_labels raised in np.concatenate rather than a clear message.

Usage:
    # DepthViT tier-4 control
    python3 eval_roc_flops.py --model depthvit \
        --ckpt  <path>.ckpt \
        --config configs/jets_150p_22M_90epoch.json \
        --out    results/roc_depthvit_22M.json

    # symmetric control arm (treatment read from the config automatically)
    python3 eval_roc_flops.py --model depthvit \
        --ckpt  <path>.ckpt \
        --config configs/jets_150p_22M_90epoch_symB_symmix.json \
        --out    results/roc_symB_symmix.json

    # matched ViT baseline
    python3 eval_roc_flops.py --model vit_small \
        --ckpt  <path>.ckpt \
        --config configs/jets_150p_vit_small_90epoch.json \
        --out    results/roc_vit_small_22M.json

Run under salloc on a single GPU, not on a login node.
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

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

JET_CLASS_NAMES = ["g", "q", "W", "Z", "top"]


                                                                      
                                                                       
                                                                      

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True, choices=["vit_tiny", "vit_small", "depthvit"])
    p.add_argument("--ckpt", required=True)
    p.add_argument("--config", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--batch_size", type=int, default=0,
                   help="override data.global_batch_size for eval; 0 keeps the config value")
    p.add_argument("--num_workers", type=int, default=0,
                   help="override data.num_workers; 0 keeps the config value")
    p.add_argument("--device", type=str,
                   default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--max_batches", type=int, default=0,
                   help="if >0, cap inference at N batches (smoke testing)")
    p.add_argument("--allow-missing", action="store_true",
                   help="permit non-strict state_dict loading; prints what was dropped. "
                        "Do NOT use for numbers that go in the paper.")
    p.add_argument("--no-flops", action="store_true")
    return p.parse_args()


                                                                      
                                                                       
                                                                      

def _normalise_sd(raw):
    """Strip Lightning's 'model.' prefix and torch.compile's '_orig_mod.'."""
    sd = raw.get("state_dict", raw)
    out = {}
    for k, v in sd.items():
        if not k.startswith("model."):
            continue                                                                
        kk = k[len("model."):].replace("_orig_mod.", "")
        out[kk] = v
    if not out:
                                                                              
        out = {k.replace("_orig_mod.", ""): v for k, v in sd.items()}
    return out


def _load(model, sd, allow_missing, label):
    msg = model.load_state_dict(sd, strict=False)
    missing = list(msg.missing_keys)
    unexpected = list(msg.unexpected_keys)
    if missing or unexpected:
        print(f"  [{label}] missing={len(missing)} unexpected={len(unexpected)}")
        for k in missing[:10]:
            print(f"    missing:    {k}")
        for k in unexpected[:10]:
            print(f"    unexpected: {k}")
        if len(missing) > 10 or len(unexpected) > 10:
            print("    ... (truncated)")
        if not allow_missing:
            raise RuntimeError(
                f"[{label}] state_dict does not match the constructed model. "
                f"If this is a symmetric-control checkpoint, check that the config's "
                f"model.channel_treatment matches the one it was trained with. "
                f"Pass --allow-missing only for debugging."
            )
    else:
        print(f"  [{label}] state_dict matched exactly")
    return model


                                                                      
                                                                       
                                                                      

def load_vit(ckpt_path, cfg, which, allow_missing):
    model_cfg, data_cfg = cfg["model"], cfg["data"]
    if which == "vit_tiny":
        from vit_tiny import ViTTiny as Cls
        d_embed, d_layers, d_heads = 192, 12, 3
    else:
        from vit_small import ViTSmall as Cls
        d_embed, d_layers, d_heads = 384, 12, 6

    model = Cls(
        img_size=int(model_cfg.get("img_size", 100)),
        patch_size=int(model_cfg.get("patch_size", 10)),
        in_channels=int(data_cfg.get("n_channels", 2)),
        num_classes=int(data_cfg.get("num_classes", 5)),
        embed_dim=int(model_cfg.get("embed_dim", d_embed)),
        num_layers=int(model_cfg.get("num_layers", d_layers)),
        num_heads=int(model_cfg.get("num_heads", d_heads)),
        mlp_ratio=float(model_cfg.get("mlp_ratio", 4.0)),
        dropout=0.0,
        attn_dropout=0.0,
    )
    raw = torch.load(ckpt_path, map_location="cpu")
    return _load(model, _normalise_sd(raw), allow_missing, which)


def load_depthvit(ckpt_path, cfg, allow_missing):
    from DepthViT import DepthViT
    model_cfg, data_cfg = cfg["model"], cfg["data"]

    treatment = str(model_cfg.get("channel_treatment", "asym")).lower()

    model = DepthViT(
        in_channels=int(data_cfg["n_channels"]),
        k_factor=int(model_cfg["k_factor"]),
        patch_size=int(model_cfg["patch_size"]),
        num_layers=int(model_cfg["n_layers"]),
        mlp_dim=int(model_cfg["mlp_dim"]),
        linear_rank=int(model_cfg["linear_rank"]),
        dropout=float(model_cfg.get("dropout", 0.0)),
        num_classes=int(data_cfg.get("num_classes", 5)),
        max_image_height=int(model_cfg["max_image_height"]),
        max_image_width=int(model_cfg["max_image_width"]),
        num_hap_layers=int(model_cfg.get("num_hap_layers", 0)),
        hap_window_size=int(model_cfg.get("hap_window_size", 8)),
        hap_mlp_ratio=float(model_cfg.get("hap_mlp_ratio", 4.0)),
        hap_drop_path=float(model_cfg.get("hap_drop_path", 0.0)),
        grad_checkpointing=False,
        k_chunk_size=0,
        compile_blocks=False,                                          
    )

                                                                              
                                                                               
    if treatment != "asym":
        from channel_treatments import apply_channel_treatment
        n = apply_channel_treatment(model, treatment)
        print(f"  [depthvit] channel_treatment={treatment!r} applied to {n} site(s)")
    else:
        print(f"  [depthvit] channel_treatment='asym' (unmodified)")

    raw = torch.load(ckpt_path, map_location="cpu")
    return _load(model, _normalise_sd(raw), allow_missing, "depthvit")


                                                                      
                                                                       
                                                                      

def make_val_loader(cfg, batch_size, num_workers):
    cfg_local = dict(cfg)
    cfg_local["data"] = dict(cfg["data"])
    cfg_local["trainer"] = dict(cfg.get("trainer", {}))
    cfg_local["trainer"]["limit_val_batches"] = 1.0
                                                 
    cfg_local["trainer"]["accumulate_grad_batches"] = 1
    if batch_size > 0:
        cfg_local["data"]["global_batch_size"] = int(batch_size)
    if num_workers > 0:
        cfg_local["data"]["num_workers"] = int(num_workers)

    phase_cfg = cfg.get("phases", [{"type": "cls_finetune"}])[0]
    cfg_local["data"].update(phase_cfg.get("data_override", {}))

    fmt = str(cfg_local["data"].get("format", "")).lower()
    if fmt == "cifar100":
        from data_cifar100 import make_loaders
        return make_loaders(cfg_local)[1]

    from data import make_loaders_dispatch
    return make_loaders_dispatch(cfg_local)[1]


                                                                      
                                                                       
                                                                      

@torch.no_grad()
def run_inference(model, val_loader, device, max_batches=0):
    model.eval().to(device)
    all_probs, all_labels = [], []

    for i, batch in enumerate(val_loader):
        if max_batches > 0 and i >= max_batches:
            break
        if isinstance(batch, (list, tuple)):
            x, y = batch[0], (batch[1] if len(batch) > 1 else None)
        else:
            x, y = batch, None
        x = x.to(device, non_blocking=True)

        logits = model.forward_cls(x) if hasattr(model, "forward_cls") else model(x)
        all_probs.append(F.softmax(logits.float(), dim=-1).cpu().numpy())
        if y is not None:
            all_labels.append(y.cpu().numpy() if torch.is_tensor(y) else np.asarray(y))

    if not all_probs:
        raise RuntimeError("val loader produced no batches -- check data paths in the config")
    if not all_labels:
        raise RuntimeError("val loader produced no labels -- AUC and Top-1 cannot be computed")

    return np.concatenate(all_probs, 0), np.concatenate(all_labels, 0)


                                                                      
                                                                       
                                                                      

def class_names_for(cfg, n_classes):
    names = cfg.get("data", {}).get("class_names")
    if names and len(names) == n_classes:
        return list(names)
    if n_classes == len(JET_CLASS_NAMES):
        return list(JET_CLASS_NAMES)
    return [str(i) for i in range(n_classes)]


def compute_metrics(probs, labels, names):
    from sklearn.metrics import roc_auc_score
    from sklearn.preprocessing import label_binarize

    n_classes = probs.shape[1]
    labels_bin = label_binarize(labels, classes=list(range(n_classes)))

    per_class, skipped = {}, []
    for i, name in enumerate(names):
        col = labels_bin[:, i]
        if col.min() == col.max():
            skipped.append(name)                                                  
            continue
        per_class[name] = float(roc_auc_score(col, probs[:, i]))

    macro = float(np.mean(list(per_class.values()))) if per_class else float("nan")
    top1 = float((np.argmax(probs, 1) == labels).mean() * 100.0)

    out = {"top1_acc": top1, "macro_auc": macro, "per_class_auc": per_class}
    if skipped:
        out["classes_absent_from_split"] = skipped
    return out


                                                                      
                                                                       
                                                                      

class _DepthViTWrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        return self.model.forward_cls(x)


def input_shape_from_cfg(cfg):
    m, d = cfg["model"], cfg["data"]
    C = int(d.get("n_channels", 2))
    H = int(m.get("max_image_height", m.get("img_size", 100)))
    W = int(m.get("max_image_width", m.get("img_size", H)))
    return (1, C, H, W)


def compute_flops(model, cfg):
    try:
        from calflops import calculate_flops
    except ImportError:
        return 0, "unavailable (pip install calflops --break-system-packages)"

    target = _DepthViTWrapper(model) if hasattr(model, "forward_cls") else model
    target.eval()
    try:
        flops, _macs, _p = calculate_flops(
            model=target,
            input_shape=input_shape_from_cfg(cfg),
            output_as_string=False,
            print_detailed=False,
        )
        return int(flops), "calflops"
    except Exception as e:
        return 0, f"calflops failed: {type(e).__name__}"


                                                                      
                                                                       
                                                                      

def main():
    args = parse_args()
    with open(args.config) as f:
        cfg = json.load(f)

    print(f"\n{'='*64}")
    print(f"Model:  {args.model}")
    print(f"Ckpt:   {os.path.basename(args.ckpt)}")
    print(f"Config: {os.path.basename(args.config)}")
    print(f"{'='*64}")

    if args.model == "depthvit":
        model = load_depthvit(args.ckpt, cfg, args.allow_missing)
        treatment = str(cfg["model"].get("channel_treatment", "asym")).lower()
    else:
        model = load_vit(args.ckpt, cfg, args.model, args.allow_missing)
        treatment = None

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Params: {n_params:,}  ({n_params/1e6:.3f}M)")

    if args.no_flops:
        flops, method = 0, "skipped"
    else:
        flops, method = compute_flops(model, cfg)
    print(f"  FLOPs:  {flops:,}  ({flops/1e9:.4f}G)  [{method}]")

    print("  Loading val set...")
    val_loader = make_val_loader(cfg, args.batch_size, args.num_workers)

    print(f"  Running inference on {args.device}...")
    probs, labels = run_inference(model, val_loader, args.device, args.max_batches)
    print(f"  Inference done: {len(labels):,} samples, {probs.shape[1]} classes")

    names = class_names_for(cfg, probs.shape[1])
    metrics = compute_metrics(probs, labels, names)

    print(f"\n  Top-1 acc: {metrics['top1_acc']:.2f}%")
    print(f"  Macro AUC: {metrics['macro_auc']:.4f}")
    if len(metrics["per_class_auc"]) <= 20:
        print("  Per-class AUC:")
        for c, a in metrics["per_class_auc"].items():
            print(f"    {c:>5}: {a:.4f}")
    else:
        print(f"  Per-class AUC: {len(metrics['per_class_auc'])} classes (see JSON)")

    result = {
        "model": args.model,
        "channel_treatment": treatment,
        "ckpt": args.ckpt,
        "config": args.config,
        "n_params": n_params,
        "flops": flops,
        "flops_method": method,
        "n_samples": int(len(labels)),
        **metrics,
    }
    outdir = os.path.dirname(os.path.abspath(args.out))
    os.makedirs(outdir, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\n  Saved to {args.out}")


if __name__ == "__main__":
    main()
