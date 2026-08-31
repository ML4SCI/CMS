#!/usr/bin/env python3
"""
Tier 2 -- isolated microbenchmark: fused Triton kernel for small-C channel
attention vs. the current unfused matmul -> softmax -> einsum path.

SCOPE (deliberately narrow, and this is what the paper should say):
  * Forward pass only. No backward, no autograd.Function, no integration into
    DepthViT.py, no training run.
  * The point is a feasibility data point for the future-work paragraph:
    "a naive fused kernel achieves Nx on the isolated operation", not a
    shipped optimization.

It reproduces _attend_chunk EXACTLY, including the einsum contraction over the
query-side channel index (see NOTE below), so parity numbers are meaningful.

Usage:
    python3 bench_chanattn_triton.py
    python3 bench_chanattn_triton.py --dtype fp32
    python3 bench_chanattn_triton.py --batch-size 32 --k 196
"""

import argparse
import json
import sys

import torch
import torch.nn.functional as F

try:
    import triton
    import triton.language as tl
    HAVE_TRITON = True
except ImportError:
    HAVE_TRITON = False


                                                                             
                                                    
                                                                             
                                                                             
                                                                             
                                                                             
                                                                            
                                                             

def attend_eager(q, k, v, k_factor):
    queries = q.transpose(-1, -2).unsqueeze(-1)                      
    keys = k.transpose(-1, -2).unsqueeze(-2)                         
    scores = torch.matmul(queries, keys) / (k_factor ** 0.5)                
    attention = F.softmax(scores, -1)
    context = torch.einsum('ijklm,ijlk->ijkm', attention, v)              
    return context.transpose(-2, -1)                              


                                                                             
                                         
                                                                             

if HAVE_TRITON:

    @triton.jit
    def _chan_attn_c2_fwd(q_ptr, k_ptr, v_ptr, o_ptr,
                          K, scale,
                          BLOCK: tl.constexpr):
        pid_n = tl.program_id(0)                         
        pid_k = tl.program_id(1)                            

        offs = pid_k * BLOCK + tl.arange(0, BLOCK)
        mask = offs < K
        base = pid_n * 2 * K                

        q0 = tl.load(q_ptr + base + offs, mask=mask, other=0.0).to(tl.float32)
        q1 = tl.load(q_ptr + base + K + offs, mask=mask, other=0.0).to(tl.float32)
        k0 = tl.load(k_ptr + base + offs, mask=mask, other=0.0).to(tl.float32)
        k1 = tl.load(k_ptr + base + K + offs, mask=mask, other=0.0).to(tl.float32)
        v0 = tl.load(v_ptr + base + offs, mask=mask, other=0.0).to(tl.float32)
        v1 = tl.load(v_ptr + base + K + offs, mask=mask, other=0.0).to(tl.float32)

                                                                        
        s00 = q0 * k0 * scale
        s01 = q0 * k1 * scale
        s10 = q1 * k0 * scale
        s11 = q1 * k1 * scale

                                                    
        m0 = tl.maximum(s00, s01)
        e00 = tl.exp(s00 - m0)
        e01 = tl.exp(s01 - m0)
        d0 = e00 + e01
        a00 = e00 / d0
        a01 = e01 / d0

        m1 = tl.maximum(s10, s11)
        e10 = tl.exp(s10 - m1)
        e11 = tl.exp(s11 - m1)
        d1 = e10 + e11
        a10 = e10 / d1
        a11 = e11 / d1

                                                                                     
        c0 = a00 * v0 + a10 * v1
        c1 = a01 * v0 + a11 * v1

        tl.store(o_ptr + base + offs, c0.to(o_ptr.dtype.element_ty), mask=mask)
        tl.store(o_ptr + base + K + offs, c1.to(o_ptr.dtype.element_ty), mask=mask)

    def attend_triton(q, k, v, k_factor, block=128):
        assert q.shape[-2] == 2, "this kernel is specialised for C=2"
        q, k, v = q.contiguous(), k.contiguous(), v.contiguous()
        B, L, C, K = q.shape
        out = torch.empty_like(q)
        grid = (B * L, triton.cdiv(K, block))
        _chan_attn_c2_fwd[grid](
            q, k, v, out, K, 1.0 / (k_factor ** 0.5), BLOCK=block,
        )
        return out


                                                                             

def cuda_time(fn, n_warmup=20, n_iter=100):
    for _ in range(n_warmup):
        fn()
    torch.cuda.synchronize()
    times = []
    for _ in range(n_iter):
        s = torch.cuda.Event(enable_timing=True)
        e = torch.cuda.Event(enable_timing=True)
        s.record()
        fn()
        e.record()
        torch.cuda.synchronize()
        times.append(s.elapsed_time(e))
    times.sort()
    return times[len(times) // 2]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--tokens", type=int, default=100)
    ap.add_argument("--channels", type=int, default=2)
    ap.add_argument("--k", type=int, default=196)
    ap.add_argument("--blocks", type=int, default=18,
                    help="encoder blocks, for the per-step extrapolation")
    ap.add_argument("--dtype", choices=["bf16", "fp32"], default="bf16")
    ap.add_argument("--block", type=int, default=128, help="Triton BLOCK size")
    ap.add_argument("--json-out", default="tier2_triton_results.json")
    args = ap.parse_args()

    if not torch.cuda.is_available():
        sys.exit("ERROR: no CUDA device visible. Run inside a GPU allocation.")
    if not HAVE_TRITON:
        sys.exit("ERROR: triton not importable. Try: pip install triton "
                 "--break-system-packages   (or check the pytorch/2.6.0 module)")

    dev = torch.device("cuda")
    dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float32
    B, L, C, K = args.batch_size, args.tokens, args.channels, args.k

    print("=" * 74)
    print("Tier 2 -- fused small-C channel attention microbenchmark (forward only)")
    print("=" * 74)
    print(f"  GPU     : {torch.cuda.get_device_name(0)}")
    print(f"  triton  : {triton.__version__}")
    print(f"  shape   : q/k/v = ({B}, {L}, {C}, {K})   dtype={args.dtype}")
    print(f"  tiny attn ops per call: {B * L * K:,}  ({C}x{C} each)")
    print()

    q = torch.randn(B, L, C, K, device=dev, dtype=dtype)
    k = torch.randn(B, L, C, K, device=dev, dtype=dtype)
    v = torch.randn(B, L, C, K, device=dev, dtype=dtype)

                                                                            
    print("-" * 74)
    print("NUMERICAL PARITY")
    print("-" * 74)
    ref = attend_eager(q.float(), k.float(), v.float(), K)
    got = attend_triton(q.float(), k.float(), v.float(), K, args.block)
    err_f32 = (ref - got).abs().max().item()
    rel_f32 = err_f32 / ref.abs().max().item()
    print(f"  fp32   max abs err : {err_f32:.3e}   (rel {rel_f32:.3e})")

    ref_d = attend_eager(q, k, v, K)
    got_d = attend_triton(q, k, v, K, args.block)
    err_d = (ref_d.float() - got_d.float()).abs().max().item()
    rel_d = err_d / ref_d.float().abs().max().item()
    print(f"  {args.dtype:<6} max abs err : {err_d:.3e}   (rel {rel_d:.3e})")
    ok = rel_f32 < 1e-5
    print(f"  fp32 parity: {'PASS' if ok else 'FAIL -- do not report speedup'}")
    print()
    if not ok:
        sys.exit(1)

                                                                            
    print("-" * 74)
    print("WALL-CLOCK (median of 100 iters, single attention op)")
    print("-" * 74)

    t_eager = cuda_time(lambda: attend_eager(q, k, v, K))
    t_triton = cuda_time(lambda: attend_triton(q, k, v, K, args.block))

    rows = [("eager (unfused)", t_eager), ("triton (fused)", t_triton)]

    try:
        compiled = torch.compile(attend_eager)
        compiled(q, k, v, K)
        t_comp = cuda_time(lambda: compiled(q, k, v, K))
        rows.append(("torch.compile", t_comp))
    except Exception as exc:
        t_comp = None
        print(f"  (torch.compile comparison skipped: {type(exc).__name__})")

    for name, t in rows:
        print(f"  {name:<20} {t:8.4f} ms      speedup vs eager: "
              f"{t_eager / t:5.2f}x")
    print()

                                                                            
    print("-" * 74)
    print("EXTRAPOLATION")
    print("-" * 74)
    per_step_eager = t_eager * args.blocks
    per_step_triton = t_triton * args.blocks
    print(f"  x{args.blocks} encoder blocks, forward only:")
    print(f"    eager  : {per_step_eager:7.2f} ms")
    print(f"    triton : {per_step_triton:7.2f} ms   "
          f"(saves {per_step_eager - per_step_triton:.2f} ms/fwd)")
    print()

                                                                             
                                                                                
    elem = q.element_size()
    bytes_moved = 4 * B * L * C * K * elem                              
    bw_eager = bytes_moved / (t_eager * 1e-3) / 1e9
    bw_triton = bytes_moved / (t_triton * 1e-3) / 1e9
    print(f"  minimum traffic (3 reads + 1 write): {bytes_moved / 1e6:.2f} MB")
    print(f"    eager  achieved : {bw_eager:7.1f} GB/s")
    print(f"    triton achieved : {bw_triton:7.1f} GB/s")
    print(f"    A100 HBM2e peak : ~1555 GB/s (40GB SXM4)")
    print()
    print("  The eager path moves far more than the minimum because every")
    print("  intermediate (scores, softmax output) is materialised to HBM.")
    print()

    out = {
        "shape": {"B": B, "L": L, "C": C, "K": K, "dtype": args.dtype},
        "parity": {"fp32_max_abs_err": err_f32, "fp32_rel_err": rel_f32,
                   f"{args.dtype}_max_abs_err": err_d},
        "ms": {"eager": t_eager, "triton": t_triton,
               "torch_compile": t_comp},
        "speedup_vs_eager": {"triton": t_eager / t_triton,
                             "torch_compile": (t_eager / t_comp) if t_comp else None},
        "bandwidth_gbs": {"eager": bw_eager, "triton": bw_triton},
        "per_step_fwd_ms": {"eager": per_step_eager, "triton": per_step_triton},
    }
    with open(args.json_out, "w") as f:
        json.dump(out, f, indent=2)
    print(f"Results JSON: {args.json_out}")


if __name__ == "__main__":
    main()
