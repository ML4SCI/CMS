"""
Exact FLOPs computation for DepthViT and ViT-Tiny using calflops.
Run from /pscratch/sd/k/krish_m/depthvit/repo

Usage:
    pip install calflops --break-system-packages --quiet
    python3 compute_flops.py
"""

import torch
import torch.nn as nn
import sys, os, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from calflops import calculate_flops

                                                                      
                                                                       
                                                                      
print("=" * 60)
print("ViT-Tiny")
print("=" * 60)
from vit_tiny import ViTTiny

vit = ViTTiny(
    img_size=100, patch_size=10, in_channels=2,
    num_classes=5, embed_dim=192, num_layers=12,
    num_heads=3, mlp_ratio=4.0,
)
vit.eval()

flops_v, macs_v, params_v = calculate_flops(
    model=vit,
    input_shape=(1, 2, 100, 100),
    output_as_string=False,
    print_detailed=False,
)
print(f"FLOPs:  {flops_v:,}  ({flops_v/1e9:.4f} GFLOPs)")
print(f"MACs:   {macs_v:,}  ({macs_v/1e9:.4f} GMACs)")
print(f"Params: {sum(p.numel() for p in vit.parameters()):,}")

                                                                      
                                                                       
                                                                      
print()
print("=" * 60)
print("DepthViT")
print("=" * 60)
from DepthViT import DepthViT

cfg  = json.load(open("configs/jets_150p_5M_90epoch.json"))
mcfg = cfg["model"]
dcfg = cfg["data"]

depthvit = DepthViT(
    in_channels       = int(dcfg["n_channels"]),
    k_factor          = int(mcfg["k_factor"]),
    patch_size        = int(mcfg["patch_size"]),
    num_layers        = int(mcfg["n_layers"]),
    mlp_dim           = int(mcfg["mlp_dim"]),
    linear_rank       = int(mcfg["linear_rank"]),
    dropout           = float(mcfg.get("dropout", 0.0)),
    num_classes       = int(dcfg.get("num_classes", 5)),
    max_image_height  = int(mcfg["max_image_height"]),
    max_image_width   = int(mcfg["max_image_width"]),
    num_hap_layers    = int(mcfg.get("num_hap_layers", 0)),
    hap_window_size   = int(mcfg.get("hap_window_size", 8)),
    hap_mlp_ratio     = float(mcfg.get("hap_mlp_ratio", 4.0)),
    hap_drop_path     = float(mcfg.get("hap_drop_path", 0.0)),
    grad_checkpointing= False,
    k_chunk_size      = 0,
    compile_blocks    = False,
)
depthvit.eval()

                                                      
class DepthViTWrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
    def forward(self, x):
        return self.model.forward_cls(x)

wrapper = DepthViTWrapper(depthvit)
wrapper.eval()

flops_d, macs_d, params_d = calculate_flops(
    model=wrapper,
    input_shape=(1, 2, 100, 100),
    output_as_string=False,
    print_detailed=False,
)
print(f"FLOPs:  {flops_d:,}  ({flops_d/1e9:.4f} GFLOPs)")
print(f"MACs:   {macs_d:,}  ({macs_d/1e9:.4f} GMACs)")
print(f"Params: {sum(p.numel() for p in depthvit.parameters()):,}")

                                                                      
                                                                       
                                                                      
print()
print("=" * 60)
print("FINAL SUMMARY")
print("=" * 60)
vit_p   = sum(p.numel() for p in vit.parameters())
dvit_p  = sum(p.numel() for p in depthvit.parameters())
print(f"{'Model':<12} {'Params':>12} {'FLOPs':>18} {'GFLOPs':>10}")
print("-" * 56)
print(f"{'ViT-Tiny':<12} {vit_p:>12,} {flops_v:>18,} {flops_v/1e9:>10.4f}")
print(f"{'DepthViT':<12} {dvit_p:>12,} {flops_d:>18,} {flops_d/1e9:>10.4f}")
print("-" * 56)
print(f"{'Ratio':<12} {vit_p/dvit_p:>11.1f}x {flops_v/flops_d:>17.1f}x")
