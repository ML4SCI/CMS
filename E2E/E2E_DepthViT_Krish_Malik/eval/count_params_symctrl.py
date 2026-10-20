"""
count_params_symctrl.py  --  verify the symmetric control is budget-matched.

Builds all three arms (asym / chansum / symmix) from ONE config, reconstructing
the actual nn.Module in each case, and prints exact parameter counts, measured
FLOPs, and an analytic MAC estimate for the channel module.

Run this on a login node BEFORE spending queue time.  If any arm is outside the
tolerance you decide with Eric, adjust k_factor (and re-verify) rather than
editing anything else -- every other knob is what the control is holding fixed.

    module load pytorch/2.6.0
    source /pscratch/sd/k/krish_m/venvs/depthvit/bin/activate
    cd /pscratch/sd/k/krish_m/depthvit/repo
    pip install calflops --break-system-packages --quiet
    python3 count_params_symctrl.py --config configs/jets_150p_22M_90epoch.json

Note on FLOPs: both the asymmetric and symmetric channel modules use torch.einsum
for their projections. calflops does not always account for einsum, so the
measured totals may undercount BOTH arms by a similar amount. The analytic
channel-module MAC column is printed alongside so the comparison is not resting
on the tracer alone.
"""

import argparse
import json
import os
import sys

import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from DepthViT import DepthViT
from channel_treatments import apply_channel_treatment, VALID_TREATMENTS


class _Wrapper(nn.Module):
    """calflops traces forward(); DepthViT classifies through forward_cls()."""

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        return self.model.forward_cls(x)


def build(cfg, treatment):
    mcfg, dcfg = cfg["model"], cfg["data"]
    model = DepthViT(
        in_channels=int(dcfg["n_channels"]),
        k_factor=int(mcfg["k_factor"]),
        patch_size=int(mcfg["patch_size"]),
        num_layers=int(mcfg["n_layers"]),
        mlp_dim=int(mcfg["mlp_dim"]),
        linear_rank=int(mcfg["linear_rank"]),
        dropout=float(mcfg.get("dropout", 0.0)),
        num_classes=int(dcfg.get("num_classes", 5)),
        max_image_height=int(mcfg["max_image_height"]),
        max_image_width=int(mcfg["max_image_width"]),
        num_hap_layers=int(mcfg.get("num_hap_layers", 0)),
        hap_window_size=int(mcfg.get("hap_window_size", 8)),
        hap_mlp_ratio=float(mcfg.get("hap_mlp_ratio", 4.0)),
        hap_drop_path=float(mcfg.get("hap_drop_path", 0.0)),
        hap_alpha=float(mcfg.get("hap_alpha", 1.0)),
        hap_learnable_alpha=bool(mcfg.get("hap_learnable_alpha", False)),
        grad_checkpointing=False,
        k_chunk_size=0,
        compile_blocks=False,                                       
    )
    apply_channel_treatment(model, treatment)
    model.eval()
    return model


def analytic_channel_macs(cfg, treatment):
    """Per-token MACs of the channel module, summed over encoder blocks."""
    mcfg, dcfg = cfg["model"], cfg["data"]
    k = int(mcfg["k_factor"])
    C = int(dcfg["n_channels"])
    L_enc = int(mcfg["n_layers"])
    if treatment == "symmix":
                                                                         
                                       
        per_token = C * k * 3 * k + k * 3 * k + C * k + C * k * k
    else:
                                                                              
        per_token = C * k * 3 * k + 2 * C * C * k + C * k * k
    n_tokens = (int(mcfg["max_image_height"]) // int(mcfg["patch_size"])) *\
               (int(mcfg["max_image_width"]) // int(mcfg["patch_size"]))
    return per_token * n_tokens * L_enc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, required=True)
    ap.add_argument("--arms", type=str, default="asym,chansum,symmix")
    ap.add_argument("--tolerance", type=float, default=1.0,
                    help="max %% deviation from the asym arm before flagging")
    ap.add_argument("--no-flops", action="store_true")
    args = ap.parse_args()

    with open(args.config) as f:
        cfg = json.load(f)

    arms = [a.strip() for a in args.arms.split(",") if a.strip()]
    for a in arms:
        if a not in VALID_TREATMENTS:
            raise SystemExit(f"unknown arm {a!r}; expected {VALID_TREATMENTS}")

    calculate_flops = None
    if not args.no_flops:
        try:
            from calflops import calculate_flops
        except ImportError:
            print("calflops not installed; reporting params only "
                  "(pip install calflops --break-system-packages)\n")

    C = int(cfg["data"]["n_channels"])
    H = int(cfg["model"]["max_image_height"])
    W = int(cfg["model"]["max_image_width"])

    rows = []
    for arm in arms:
        model = build(cfg, arm)
        params = sum(p.numel() for p in model.parameters())
        flops = None
        if calculate_flops is not None:
            wrapper = _Wrapper(model).eval()
            flops, _macs, _p = calculate_flops(
                model=wrapper,
                input_shape=(1, C, H, W),
                output_as_string=False,
                print_detailed=False,
            )
        rows.append((arm, params, flops, analytic_channel_macs(cfg, arm)))
        del model

    base = next((r for r in rows if r[0] == "asym"), rows[0])

    print()
    print("=" * 88)
    print(f"SYMMETRIC CONTROL -- budget match   ({os.path.basename(args.config)})")
    print("=" * 88)
    print(f"{'arm':<10} {'params':>13} {'d%':>8} {'GFLOPs':>11} {'d%':>8} "
          f"{'chan MMACs':>12} {'d%':>8}")
    print("-" * 88)
    flagged = []
    for arm, params, flops, cmacs in rows:
        dp = 100.0 * (params - base[1]) / base[1]
        df = 100.0 * (flops - base[2]) / base[2] if (flops and base[2]) else float("nan")
        dc = 100.0 * (cmacs - base[3]) / base[3]
        gf = f"{flops/1e9:11.4f}" if flops else f"{'--':>11}"
        dfs = f"{df:+8.2f}" if flops else f"{'--':>8}"
        print(f"{arm:<10} {params:>13,} {dp:+8.3f} {gf} {dfs} "
              f"{cmacs/1e6:>12.2f} {dc:+8.2f}")
        if abs(dp) > args.tolerance:
            flagged.append((arm, "params", dp))
        if flops and abs(df) > args.tolerance:
            flagged.append((arm, "flops", df))
    print("-" * 88)
    print(f"tolerance: +/-{args.tolerance:.2f}%   (parameter match is the primary "
          f"criterion; FLOPs approximate)")
    print()

    if flagged:
        print("OUT OF TOLERANCE:")
        for arm, what, v in flagged:
            print(f"  {arm:<10} {what:<7} {v:+.3f}%")
        print()
        print("Adjust k_factor for that arm and re-run. Do not touch n_layers,")
        print("mlp_dim, num_hap_layers, hap_window_size, linear_rank or the head --")
        print("those are what the control holds fixed.")
    else:
        print("ALL ARMS WITHIN TOLERANCE.")


if __name__ == "__main__":
    main()
