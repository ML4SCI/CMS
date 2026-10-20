"""
channel_treatments.py  --  symmetric control for DepthViT.

Purpose
-------
Isolate whether DepthViT's gains come from CHANNEL-ASYMMETRIC attention
specifically, by holding everything else fixed (HAP blocks + placement,
slot-aware TokenAttnHead, depth, training protocol, parameter budget) and
changing ONLY the channel treatment.

Three arms, selected by `model.channel_treatment` in the config JSON:

  "asym"     (default) -- unchanged DepthViT.  This is the control arm and
                          reproduces the locked tier-4 configuration exactly.

  "chansum"  -- channel-SUMMING patch embedding.  conv_proj goes from
                groups=in_channels (each detector channel gets its own private
                stem) to groups=1 (every output feature sees BOTH ECAL and
                HCAL).  The 2-way axis that CrossDepthMultiheadSelfAttention
                attends over is still there, but it no longer carries physical
                per-detector identity.
                Answers: "is the gain about *physical* channel asymmetry?"
                Cost: +39,200 params, +7.8 MFLOPs at the 22M tier (~0.18% on
                both) -- see count_params_symctrl.py for exact measured values.

  "symmix"   -- channel-SYMMETRIC mixing.  CrossDepthMultiheadSelfAttention is
                replaced by ChannelSymmetricMixing, a permutation-equivariant
                map over the channel axis with an EXACTLY matched parameter
                count (for in_channels=2).  No channel has private weights.
                Answers: "is the gain about asymmetric (private-per-channel)
                treatment, versus any symmetric mixing of the same budget?"

Both variants preserve: block count, block order, HAP placement, hidden_dim,
mlp_dim, linear_rank, TokenAttnHead, position embeddings, mask token.

Usage (handled for you by apply_patches.py):
    from channel_treatments import maybe_apply_channel_treatment, channel_treatment_of
    build DepthViT with compile_blocks=False if treatment != "asym"
    maybe_apply_channel_treatment(model, model_cfg)
"""

import math
from typing import Any, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

from DepthViT import CrossDepthMultiheadSelfAttention

VALID_TREATMENTS = ("asym", "chansum", "symmix")


                                                                      
                                                                       
                                                                      
class ChannelSymmetricMixing(nn.Module):
    """
    Permutation-equivariant replacement for CrossDepthMultiheadSelfAttention.

    Parameter accounting (per block, C = in_channels = 2, k = k_factor):

        asymmetric  qkv_weight (C, k, 3k) = 3Ck^2 = 6k^2
                    qkv_bias   (C, 3k)    = 3Ck   = 6k
                    fc_out     k -> k     = k^2 + k
                    TOTAL                 = 7k^2 + 7k

        symmetric   qkv_weight[0] reused as W_self  (k, 3k) = 3k^2
                    qkv_weight[1] reused as W_cross (k, 3k) = 3k^2
                    qkv_bias[0]   reused as bias    (3k)
                    qkv_bias[1]   reused as gain    (3k)
                    fc_out        k -> k            = k^2 + k
                    TOTAL                           = 7k^2 + 7k     <-- exact

    The tensors keep their original SHAPES so the parameter count is matched to
    the integer; only their INTERPRETATION changes.  W_self is applied to every
    channel identically and W_cross is applied to the channel-pooled mean, so
    the map is equivariant under permuting the channel axis -- no channel owns
    private weights.

    Restricted to in_channels == 2.  A permutation-equivariant linear map has
    exactly two free matrices for any C, so exact budget matching against C
    private matrices only holds at C = 2.  That is the calorimeter setting
    (ECAL/HCAL), which is where this control lives.
    """

    def __init__(self, k_factor: int, in_channels: int, k_chunk_size: int = 0):
        super().__init__()
        if int(in_channels) != 2:
            raise ValueError(
                f"ChannelSymmetricMixing requires in_channels=2 for exact parameter "
                f"matching; got {in_channels}. See the docstring."
            )
        self.k_factor = int(k_factor)
        self.in_channels = int(in_channels)
        self.k_chunk_size = int(k_chunk_size)                                         

                                                                                   
        self.qkv_weight = nn.Parameter(torch.randn(in_channels, k_factor, k_factor * 3))
        self.qkv_bias = nn.Parameter(torch.randn(in_channels, k_factor * 3))
        self.fc_out = nn.Linear(k_factor, k_factor)

                                                                              
                                                                          
                                                                           
                                                                    
                                                                            
                                                                                
                                                                                
                                                                         
        self.register_buffer("_symmetric_marker", torch.ones(1), persistent=True)

        self.reset_parameters()

    def reset_parameters(self):
                                                                               
                                                
        for i in range(self.in_channels):
            nn.init.kaiming_uniform_(self.qkv_weight[i], a=math.sqrt(5))

        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.qkv_weight[0])
        bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
        with torch.no_grad():
                                                                                 
            nn.init.uniform_(self.qkv_bias[0], -bound, bound)
                                                                              
                                                                              
                                                                              
            self.qkv_bias[1].fill_(1.0)

        self.fc_out.reset_parameters()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, L, D = x.shape
        C, K = self.in_channels, self.k_factor
        x_ch = x.view(B, L, C, K)

        w_self = self.qkv_weight[0]                    
        w_cross = self.qkv_weight[1]                   
        bias = self.qkv_bias[0]                      
        gain = self.qkv_bias[1]                      

                                                                         
                                                                                  
        self_part = torch.einsum('blck,km->blcm', x_ch, w_self)                      
        pooled = x_ch.mean(dim=2)                                                 
        cross_part = torch.einsum('blk,km->blm', pooled, w_cross) * gain           

        qkv = self_part + cross_part.unsqueeze(2) + bias
        q, k, v = qkv.chunk(3, dim=-1)                                                   

                                                                                 
                                                                                    
                                                                      
        score = (q * k).sum(dim=-1) / (K ** 0.5)                                  
        w = torch.sigmoid(score).unsqueeze(-1)                                      
        v_pool = v.mean(dim=2, keepdim=True)                                        
        context = v + w * (v_pool - v)                                              

        out = self.fc_out(context)
        return out.flatten(start_dim=2)


                                                                      
                                                                       
                                                                      
def channel_treatment_of(model_cfg: Dict[str, Any]) -> str:
    ct = str(model_cfg.get("channel_treatment", "asym")).lower().strip()
    if ct in ("", "none", "baseline", "default"):
        ct = "asym"
    if ct not in VALID_TREATMENTS:
        raise ValueError(f"Unknown channel_treatment={ct!r}; expected one of {VALID_TREATMENTS}")
    return ct


def _swap_patch_embed_to_channel_summing(model: nn.Module) -> int:
    old = model.conv_proj
    if not isinstance(old, nn.Conv2d):
        raise TypeError(f"Expected conv_proj to be nn.Conv2d, got {type(old)}")
    if old.groups == 1:
        return 0
    new = nn.Conv2d(
        in_channels=old.in_channels,
        out_channels=old.out_channels,
        kernel_size=old.kernel_size,
        stride=old.stride,
        padding=old.padding,
        groups=1,
        bias=(old.bias is not None),
    )
                                                   
    fan_in = new.in_channels * new.kernel_size[0] * new.kernel_size[1]
    nn.init.trunc_normal_(new.weight, std=math.sqrt(1 / fan_in))
    if new.bias is not None:
        nn.init.zeros_(new.bias)
    new = new.to(old.weight.device, old.weight.dtype)
    model.conv_proj = new
    return 1


def _swap_channel_attention_to_symmetric(model: nn.Module) -> int:
    n = 0
    blocks = model.encoder.blocks
    for i in range(len(blocks)):
        blk = blocks[i]
        attn = getattr(blk, "self_attention", None)
        if isinstance(attn, CrossDepthMultiheadSelfAttention):
            new = ChannelSymmetricMixing(attn.k_factor, attn.in_channels)
            new = new.to(attn.qkv_weight.device, attn.qkv_weight.dtype)
            blk.self_attention = new
            n += 1
    if n == 0:
        raise RuntimeError(
            "symmix: found no CrossDepthMultiheadSelfAttention to replace. "
            "Build the model with compile_blocks=False before applying the treatment."
        )
    return n


def compile_blocks_after_swap(model: nn.Module, grad_checkpointing: bool) -> int:
    """
    Reproduce Encoder.__init__'s compile_blocks behaviour AFTER module surgery.
    Encoder compiles blocks at construction time, which would leave the swapped
    modules buried inside an OptimizedModule, so symctrl runs build with
    compile_blocks=False and compile here instead.
    """
    if not hasattr(torch, "compile"):
        return 0
    enc = model.encoder
    is_hap = enc._is_hap
    n = 0
    for i in range(len(enc.blocks)):
        if grad_checkpointing and is_hap[i]:
            continue                                                                
        enc.blocks[i] = torch.compile(enc.blocks[i])
        n += 1
    return n


def apply_channel_treatment(model: nn.Module, treatment: str) -> int:
    treatment = str(treatment).lower().strip()
    if treatment == "asym":
        return 0
    if treatment == "chansum":
        return _swap_patch_embed_to_channel_summing(model)
    if treatment == "symmix":
        return _swap_channel_attention_to_symmetric(model)
    raise ValueError(f"Unknown channel_treatment={treatment!r}")


def maybe_apply_channel_treatment(model: nn.Module, model_cfg: Dict[str, Any]) -> str:
    """
    Entry point called from the trainer.  Applies the treatment, then restores
    the config's compile_blocks setting so every arm is compiled identically.
    """
    ct = channel_treatment_of(model_cfg)
    if ct == "asym":
        return ct
    base = getattr(model, "_orig_mod", model)
    n = apply_channel_treatment(base, ct)
    msg = f"[channel_treatment] applied {ct!r} to {n} site(s)"
    if bool(model_cfg.get("compile_blocks", False)):
        gc = bool(model_cfg.get("grad_checkpointing", True))
        c = compile_blocks_after_swap(base, gc)
        msg += f"; torch.compile applied to {c} block(s) post-swap"
    print(msg, flush=True)
    return ct
