#!/usr/bin/env python3
"""
Tier 1 — wall-clock vs FLOPs breakdown for DepthViT-22M (L=18, k=196).

Produces the paper artifact: a table showing that channel-attention consumes a
share of wall-clock far in excess of its share of FLOPs, which is the measured
version of the "unfused small-C attention" claim.

Two independent measurements:
  (A) Isolated component microbenchmark (CUDA events) -- the paper table.
      Times CrossDepthMultiheadSelfAttention, the FFN/MLPBlock, and HAPBlock
      separately at real tier-4 shapes, forward and forward+backward.
  (B) Full-model torch.profiler trace -- supporting evidence, exports a
      chrome trace you can open in perfetto if a reviewer wants detail.

Synthetic inputs only: no dataloader, no dataset, no checkpoint. Runs on one
GPU inside a debug-queue allocation in a couple of minutes.

Usage:
    python3 profile_tier4.py                    # eager, bf16 autocast
    python3 profile_tier4.py --compile          # matches compile_blocks=true
    python3 profile_tier4.py --batch-size 32    # default; real micro-batch
"""

import argparse
import json
import os
import sys
import time

import torch
import torch.nn as nn

              
from DepthViT import (
    CrossDepthMultiheadSelfAttention,
    MLPBlock,
    HAPBlock,
    EncoderBlock,
    DepthViT,
)


                                                                              
                
                                                                              

def cuda_time(fn, n_warmup=10, n_iter=50):
    """Median-of-iters wall-clock in ms for a callable, using CUDA events."""
    for _ in range(n_warmup):
        fn()
    torch.cuda.synchronize()

    times = []
    for _ in range(n_iter):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        fn()
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end))
    times.sort()
    return times[len(times) // 2]


def make_fwd(module, *inputs, autocast_dtype=None):
    def fn():
        with torch.no_grad():
            if autocast_dtype is not None:
                with torch.autocast("cuda", dtype=autocast_dtype):
                    module(*inputs)
            else:
                module(*inputs)
    return fn


def make_fwd_bwd(module, *inputs, autocast_dtype=None):
    def fn():
        for p in module.parameters():
            p.grad = None
        if autocast_dtype is not None:
            with torch.autocast("cuda", dtype=autocast_dtype):
                out = module(*inputs)
                loss = out.float().pow(2).mean()
        else:
            out = module(*inputs)
            loss = out.pow(2).mean()
        loss.backward()
    return fn


                                                                              
                                                          
                                                                              

def chan_attn_flops(L, C, K):
    """CrossDepthMultiheadSelfAttention forward FLOPs (multiply-add = 2 FLOPs)."""
    qkv = 2 * L * C * K * (3 * K)                             
    scores = 2 * L * K * C * C                               
    ctx = 2 * L * K * C * C                                   
    fc_out = 2 * L * C * K * K                         
    return qkv + scores + ctx + fc_out


def mlp_flops(L, D, M):
    """MLPBlock (D -> M -> D)."""
    return 2 * L * D * M + 2 * L * M * D


def hap_flops(L, D, ws, mlp_ratio, n_micro_shifts, n_macro_shifts):
    """Rough HAPBlock forward FLOPs. Dominated by cross1/cross2 + MLP."""
    n_win = L // (ws * ws) if L >= ws * ws else 1
    micro = 2 * L * D * n_micro_shifts                                           
    macro = 2 * n_win * D * n_macro_shifts
    cross = 2 * (2 * n_win * D * D)                                                   
    mlp = 2 * L * D * int(D * mlp_ratio) * 2
    return micro + macro + cross + mlp


                                                                              
      
                                                                              

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/jets_150p_22M_90epoch.json")
    ap.add_argument("--batch-size", type=int, default=32,
                    help="Real per-GPU micro-batch: global 512 / 4 devices / 4 accum = 32")
    ap.add_argument("--compile", action="store_true",
                    help="Wrap blocks in torch.compile, matching compile_blocks=true")
    ap.add_argument("--iters", type=int, default=50)
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--skip-full-model", action="store_true",
                    help="Run only the component microbenchmark (part A)")
    ap.add_argument("--trace-out", default="tier4_trace.json")
    ap.add_argument("--json-out", default="tier4_profile_results.json")
    args = ap.parse_args()

    if not torch.cuda.is_available():
        sys.exit("ERROR: no CUDA device visible. Run inside a GPU allocation.")

    cfg = json.load(open(args.config))
    m = cfg["model"]

    C = cfg["data"]["n_channels"]
    K = m["k_factor"]
    D = C * K
    L = (m["max_image_height"] // m["patch_size"]) * (m["max_image_width"] // m["patch_size"])
    n_layers = m["n_layers"]
    n_hap = m["num_hap_layers"]
    mlp_dim = m["mlp_dim"]
    ws = m["hap_window_size"]
    B = args.batch_size
    dev = torch.device("cuda")
    autocast_dtype = torch.bfloat16                                 

    n_h = m["max_image_height"] // m["patch_size"]
    n_w = m["max_image_width"] // m["patch_size"]

    print("=" * 78)
    print("DepthViT tier-4 profile")
    print("=" * 78)
    print(f"  GPU              : {torch.cuda.get_device_name(0)}")
    print(f"  torch            : {torch.__version__}")
    print(f"  config           : {args.config}")
    print(f"  C x K -> D       : {C} x {K} -> {D}")
    print(f"  tokens L         : {L}  ({n_h} x {n_w} grid, patch={m['patch_size']})")
    print(f"  blocks           : {n_layers} encoder + {n_hap} HAP")
    print(f"  micro-batch B    : {B}")
    print(f"  precision        : bf16 autocast")
    print(f"  torch.compile    : {args.compile}")
    print()

    results = {"meta": {
        "gpu": torch.cuda.get_device_name(0), "torch": torch.__version__,
        "C": C, "K": K, "D": D, "L": L, "B": B,
        "n_layers": n_layers, "n_hap": n_hap, "compile": args.compile,
    }}

                                                                        
                                           
                                                                        
    print("-" * 78)
    print("(A) COMPONENT MICROBENCHMARK  (median of %d iters)" % args.iters)
    print("-" * 78)

    attn = CrossDepthMultiheadSelfAttention(K, C, m["k_chunk_size"]).to(dev)
    ffn = MLPBlock(D, mlp_dim, m["dropout"]).to(dev)
    hap = HAPBlock(dim=D, window_size=ws, mlp_ratio=m["hap_mlp_ratio"],
                   drop=m["dropout"], drop_path=m["hap_drop_path"]).to(dev)

    if args.compile:
        attn = torch.compile(attn)
        ffn = torch.compile(ffn)
        hap = torch.compile(hap)

    x_seq = torch.randn(B, L, D, device=dev, requires_grad=True)
    x_2d = torch.randn(B, n_h, n_w, D, device=dev, requires_grad=True)

    comps = [
        ("chan-attn", attn, (x_seq,), n_layers,
         chan_attn_flops(L, C, K) * B),
        ("FFN/MLP", ffn, (x_seq,), n_layers,
         mlp_flops(L, D, mlp_dim) * B),
        ("HAP", hap, (x_2d,), n_hap,
         hap_flops(L, D, ws, m["hap_mlp_ratio"],
                   len(getattr(hap, "micro_mix", hap).shifts) if hasattr(hap, "micro_mix") else 8,
                   len(hap.macro_mix.shifts) if hasattr(hap, "macro_mix") else 8) * B),
    ]

    rows = []
    for name, mod, inp, count, flops_per_block in comps:
        t_f = cuda_time(make_fwd(mod, *inp, autocast_dtype=autocast_dtype),
                        args.warmup, args.iters)
        t_fb = cuda_time(make_fwd_bwd(mod, *inp, autocast_dtype=autocast_dtype),
                         args.warmup, args.iters)
        rows.append({
            "component": name,
            "blocks": count,
            "fwd_ms_per_block": t_f,
            "fwd_bwd_ms_per_block": t_fb,
            "fwd_ms_total": t_f * count,
            "fwd_bwd_ms_total": t_fb * count,
            "fwd_gflops_total": flops_per_block * count / 1e9,
        })

    tot_time = sum(r["fwd_bwd_ms_total"] for r in rows)
    tot_time_f = sum(r["fwd_ms_total"] for r in rows)
    tot_flops = sum(r["fwd_gflops_total"] for r in rows)

    hdr = f"{'component':<12} {'blks':>5} {'fwd ms':>9} {'f+b ms':>9} "\
          f"{'time %':>8} {'GFLOPs':>10} {'FLOP %':>8} {'ratio':>7}"
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        tpct = 100 * r["fwd_bwd_ms_total"] / tot_time
        fpct = 100 * r["fwd_gflops_total"] / tot_flops
        r["time_pct"] = tpct
        r["flop_pct"] = fpct
        r["ratio"] = tpct / fpct if fpct > 0 else float("inf")
        print(f"{r['component']:<12} {r['blocks']:>5} "
              f"{r['fwd_ms_total']:>9.2f} {r['fwd_bwd_ms_total']:>9.2f} "
              f"{tpct:>7.1f}% {r['fwd_gflops_total']:>10.3f} {fpct:>7.1f}% "
              f"{r['ratio']:>7.2f}")
    print("-" * len(hdr))
    print(f"{'TOTAL':<12} {'':>5} {tot_time_f:>9.2f} {tot_time:>9.2f} "
          f"{100.0:>7.1f}% {tot_flops:>10.3f} {100.0:>7.1f}%")
    print()
    print("  'ratio' = (share of wall-clock) / (share of FLOPs).")
    print("  ratio >> 1 means the component is memory-bandwidth-bound relative")
    print("  to its arithmetic -- this is the number the paper section needs.")
    print()

    results["components"] = rows

                                                                        
                                   
                                                                        
    if not args.skip_full_model:
        print("-" * 78)
        print("(B) FULL-MODEL torch.profiler TRACE")
        print("-" * 78)

        model = DepthViT(
            patch_size=m["patch_size"],
            in_channels=C,
            k_factor=K,
            num_layers=n_layers,
            mlp_dim=mlp_dim,
            linear_rank=m["linear_rank"],
            max_image_height=m["max_image_height"],
            max_image_width=m["max_image_width"],
            dropout=m["dropout"],
            num_classes=cfg["data"]["num_classes"],
            num_hap_layers=n_hap,
            hap_window_size=ws,
            hap_mlp_ratio=m["hap_mlp_ratio"],
            hap_drop_path=m["hap_drop_path"],
            grad_checkpointing=m["grad_checkpointing"],
            k_chunk_size=m["k_chunk_size"],
            compile_blocks=args.compile,
            hap_alpha=m["hap_alpha"],
            hap_learnable_alpha=m["hap_learnable_alpha"],
        ).to(dev)

                                                                             
                              
        imgs = torch.randn(B, C, m["max_image_height"], m["max_image_width"], device=dev)

        def step():
            model.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=autocast_dtype):
                logits = model.forward_cls(imgs)
                loss = logits.float().pow(2).mean()
            loss.backward()

        for _ in range(5):
            step()
        torch.cuda.synchronize()

        t_step = cuda_time(step, n_warmup=3, n_iter=20)
        print(f"  full fwd+bwd step: {t_step:.2f} ms  (B={B})")
        results["full_step_ms"] = t_step

        from torch.profiler import profile, ProfilerActivity
        with profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
            record_shapes=True,
            with_stack=False,
        ) as prof:
            for _ in range(3):
                step()
            torch.cuda.synchronize()

        print()
        print(prof.key_averages().table(
            sort_by="self_cuda_time_total", row_limit=25))

        prof.export_chrome_trace(args.trace_out)
        print(f"\n  chrome trace written to: {args.trace_out}")

                                                                            
        evts = prof.key_averages()
        n_launches = sum(e.count for e in evts if e.self_device_time_total > 0)\
            if hasattr(evts[0], "self_device_time_total") else\
            sum(e.count for e in evts if getattr(e, "self_cuda_time_total", 0) > 0)
        print(f"  CUDA op invocations across 3 steps: {n_launches:,} "
              f"(~{n_launches // 3:,} per step)")
        results["cuda_ops_per_step"] = n_launches // 3

    with open(args.json_out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults JSON: {args.json_out}")


if __name__ == "__main__":
    main()
