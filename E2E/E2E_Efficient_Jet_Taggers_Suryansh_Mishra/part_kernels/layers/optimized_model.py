"""
Optimized Particle Transformer using custom Triton kernels.

Drop-in replacement for weaver's ParticleTransformer that:
  1. Uses fused Triton pairwise kernel (no intermediate tensors)
  2. Computes full (N,4,P,P) pair features instead of triangle gather/expand
  3. Uses fused Triton attention with additive bias (no full attn matrix alloc)
  4. Preserves exact weight compatibility with the original model

Note (DD11): these classes are user-authored GPU benchmark assets. They are
NOT used by ``optimize_part_model`` (which patches weaver instances in place)
and are NOT part of the CPU verification suite. Weight extraction from the
pinned weaver 0.5.3 ``Attention`` goes through ``_compat.get_in_proj_params``.
"""
import copy
import math

import torch
import torch.nn as nn

from .._compat import get_in_proj_params
from ..autograd.attention import fused_attention_with_bias
from ..triton.pairwise_kernel import fused_pairwise_lv_fts
from .fused_pair_mlp import FusedPairMLP


# ---------- optimised PairEmbed -----------------------------------------

class OptimizedPairEmbed(nn.Module):
    """PairEmbed that uses the fused Triton pairwise kernel and skips
    triangle indexing / scatter-expand."""

    def __init__(self, pairwise_lv_dim, pairwise_input_dim, dims,
                 remove_self_pair=False, use_pre_activation_pair=True,
                 mode='sum', normalize_input=True, activation='gelu',
                 eps=1e-8):
        super().__init__()
        assert pairwise_lv_dim == 4, "Fused kernel supports 4 Lorentz features"
        assert pairwise_input_dim == 0, "Extra pair inputs not yet supported"
        assert mode == 'sum', "Only 'sum' mode supported"

        self.eps = eps
        self.remove_self_pair = remove_self_pair
        self.out_dim = dims[-1]

        input_dim = pairwise_lv_dim
        module_list = [nn.BatchNorm1d(input_dim)] if normalize_input else []
        for dim in dims:
            module_list.extend([
                nn.Conv1d(input_dim, dim, 1),
                nn.BatchNorm1d(dim),
                nn.GELU() if activation == 'gelu' else nn.ReLU(),
            ])
            input_dim = dim
        if use_pre_activation_pair:
            module_list = module_list[:-1]
        self.embed = nn.Sequential(*module_list)

    def forward(self, x, uu=None):
        """
        Args:
            x:  (N, 4, P) particle 4-vectors
            uu: ignored (not supported in fused path)
        Returns:
            (N, num_heads, P, P) attention bias
        """
        N, _, P = x.shape

        pair_fts = fused_pairwise_lv_fts(x, eps=self.eps)

        if self.remove_self_pair:
            idx = torch.arange(P, device=x.device)
            pair_fts[:, :, idx, idx] = 0.0

        pair_fts = pair_fts.view(N, 4, P * P)
        elements = self.embed(pair_fts)
        return elements.view(N, self.out_dim, P, P)

    @classmethod
    def from_standard(cls, orig):
        """Copy weights from an original PairEmbed module."""
        obj = cls.__new__(cls)
        nn.Module.__init__(obj)
        obj.eps = orig.pairwise_lv_fts.keywords.get('eps', 1e-8)
        obj.remove_self_pair = orig.remove_self_pair
        obj.out_dim = orig.out_dim
        obj.embed = copy.deepcopy(orig.embed)
        return obj


# ---------- optimised attention Block -----------------------------------

class OptimizedBlock(nn.Module):
    """Transformer block with fused Triton attention (self-attn path)
    and standard MHA for the class-attention path."""

    def __init__(self, embed_dim=128, num_heads=8, ffn_ratio=4,
                 dropout=0.1, attn_dropout=0.1, activation_dropout=0.1,
                 add_bias_kv=False, activation='gelu',
                 scale_fc=True, scale_attn=True, scale_heads=True,
                 scale_resids=True):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        self.pre_attn_norm = nn.LayerNorm(embed_dim)

        self.qkv_proj = nn.Linear(embed_dim, 3 * embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.attn_drop_p = attn_dropout

        self.cls_attn = nn.MultiheadAttention(
            embed_dim, num_heads, dropout=attn_dropout,
            add_bias_kv=add_bias_kv)

        self.post_attn_norm = nn.LayerNorm(embed_dim) if scale_attn else None
        self.dropout = nn.Dropout(dropout)

        self.pre_fc_norm = nn.LayerNorm(embed_dim)
        self.fc1 = nn.Linear(embed_dim, embed_dim * ffn_ratio)
        self.act = nn.GELU() if activation == 'gelu' else nn.ReLU()
        self.act_dropout = nn.Dropout(activation_dropout)
        self.post_fc_norm = nn.LayerNorm(embed_dim * ffn_ratio) if scale_fc else None
        self.fc2 = nn.Linear(embed_dim * ffn_ratio, embed_dim)

        self.c_attn = nn.Parameter(torch.ones(num_heads), requires_grad=True) if scale_heads else None
        self.w_resid = nn.Parameter(torch.ones(embed_dim), requires_grad=True) if scale_resids else None

    def _self_attn(self, x, padding_mask, attn_mask):
        P, N, C = x.shape
        qkv = self.qkv_proj(x)
        q, k, v = qkv.chunk(3, dim=-1)

        q = q.view(P, N, self.num_heads, self.head_dim).permute(1, 2, 0, 3).reshape(N * self.num_heads, P, self.head_dim)
        k = k.view(P, N, self.num_heads, self.head_dim).permute(1, 2, 0, 3).reshape(N * self.num_heads, P, self.head_dim)
        v = v.view(P, N, self.num_heads, self.head_dim).permute(1, 2, 0, 3).reshape(N * self.num_heads, P, self.head_dim)

        scale = 1.0 / math.sqrt(self.head_dim)
        out = fused_attention_with_bias(q, k, v, attn_mask, padding_mask, scale, self.num_heads)

        out = out.view(N, self.num_heads, P, self.head_dim).permute(2, 0, 1, 3).reshape(P, N, C)
        return self.out_proj(out)

    def forward(self, x, x_cls=None, padding_mask=None, attn_mask=None):
        if x_cls is not None:
            with torch.no_grad():
                padding_mask = torch.cat((torch.zeros_like(padding_mask[:, :1]), padding_mask), dim=1)
            residual = x_cls
            u = torch.cat((x_cls, x), dim=0)
            u = self.pre_attn_norm(u)
            x = self.cls_attn(x_cls, u, u, key_padding_mask=padding_mask)[0]
        else:
            residual = x
            x = self.pre_attn_norm(x)
            x = self._self_attn(x, padding_mask, attn_mask)

        if self.c_attn is not None:
            tgt_len = x.size(0)
            x = x.view(tgt_len, -1, self.num_heads, self.head_dim)
            x = torch.einsum('tbhd,h->tbdh', x, self.c_attn)
            x = x.reshape(tgt_len, -1, self.embed_dim)
        if self.post_attn_norm is not None:
            x = self.post_attn_norm(x)
        x = self.dropout(x)
        x += residual

        residual = x
        x = self.pre_fc_norm(x)
        x = self.act(self.fc1(x))
        x = self.act_dropout(x)
        if self.post_fc_norm is not None:
            x = self.post_fc_norm(x)
        x = self.fc2(x)
        x = self.dropout(x)
        if self.w_resid is not None:
            residual = torch.mul(self.w_resid, residual)
        x += residual
        return x

    @classmethod
    def from_standard(cls, orig):
        """Convert a standard Block to an OptimizedBlock, copying all weights."""
        obj = cls.__new__(cls)
        nn.Module.__init__(obj)
        obj.embed_dim = orig.embed_dim
        obj.num_heads = orig.num_heads
        obj.head_dim = orig.head_dim

        obj.pre_attn_norm = copy.deepcopy(orig.pre_attn_norm)

        embed_dim = orig.embed_dim
        in_proj_weight, in_proj_bias = get_in_proj_params(orig.attn)
        device = in_proj_weight.device
        obj.qkv_proj = nn.Linear(embed_dim, 3 * embed_dim,
                                 bias=in_proj_bias is not None, device=device)
        with torch.no_grad():
            obj.qkv_proj.weight.copy_(in_proj_weight)
            if in_proj_bias is not None:
                obj.qkv_proj.bias.copy_(in_proj_bias)
        obj.out_proj = copy.deepcopy(orig.attn.out_proj)
        obj.attn_drop_p = orig.attn.dropout

        obj.cls_attn = copy.deepcopy(orig.attn)

        obj.post_attn_norm = copy.deepcopy(orig.post_attn_norm) if orig.post_attn_norm else None
        obj.dropout = copy.deepcopy(orig.dropout)

        obj.pre_fc_norm = copy.deepcopy(orig.pre_fc_norm)
        obj.fc1 = copy.deepcopy(orig.fc1)
        obj.act = copy.deepcopy(orig.act)
        obj.act_dropout = copy.deepcopy(orig.act_dropout)
        obj.post_fc_norm = copy.deepcopy(orig.post_fc_norm) if orig.post_fc_norm else None
        obj.fc2 = copy.deepcopy(orig.fc2)

        obj.c_attn = copy.deepcopy(orig.c_attn) if orig.c_attn is not None else None
        obj.w_resid = copy.deepcopy(orig.w_resid) if orig.w_resid is not None else None
        return obj


# ---------- optimised full model ----------------------------------------

class OptimizedParticleTransformer(nn.Module):
    """Drop-in replacement for weaver's ParticleTransformer that uses
    custom Triton kernels for pairwise features and attention."""

    def __init__(self, orig_model):
        super().__init__()
        self.for_inference = orig_model.for_inference
        self.use_amp = orig_model.use_amp

        from weaver.nn.model.ParticleTransformer import SequenceTrimmer
        self.trimmer = copy.deepcopy(orig_model.trimmer)

        self.embed = copy.deepcopy(orig_model.embed)

        if orig_model.pair_embed is not None:
            self.pair_embed = OptimizedPairEmbed.from_standard(orig_model.pair_embed)
            self._fused_pair_mlp = None
        else:
            self.pair_embed = None
            self._fused_pair_mlp = None

        self.blocks = nn.ModuleList([OptimizedBlock.from_standard(b) for b in orig_model.blocks])
        self.cls_blocks = nn.ModuleList([copy.deepcopy(b) for b in orig_model.cls_blocks])
        self.norm = copy.deepcopy(orig_model.norm)
        self.fc = copy.deepcopy(orig_model.fc) if orig_model.fc is not None else None
        self.cls_token = copy.deepcopy(orig_model.cls_token)

    def _build_fused_pair_mlp(self):
        if self.pair_embed is not None and self._fused_pair_mlp is None:
            self._fused_pair_mlp = FusedPairMLP(
                self.pair_embed.embed,
                eps=self.pair_embed.eps,
                remove_self_pair=self.pair_embed.remove_self_pair,
            ).to(next(self.parameters()).device)

    def forward(self, x, v=None, mask=None, uu=None, uu_idx=None):
        with torch.no_grad():
            if not self.for_inference:
                if uu_idx is not None:
                    from weaver.nn.model.ParticleTransformer import build_sparse_tensor
                    uu = build_sparse_tensor(uu, uu_idx, x.size(-1))
            x, v, mask, uu = self.trimmer(x, v, mask, uu)
            padding_mask = ~mask.squeeze(1)

        with torch.cuda.amp.autocast(enabled=self.use_amp):
            x = self.embed(x).masked_fill(~mask.permute(2, 0, 1), 0)
            attn_mask = None
            if v is not None and self.pair_embed is not None:
                if not self.training and self._fused_pair_mlp is not None:
                    attn_mask = self._fused_pair_mlp(v).view(-1, v.size(-1), v.size(-1))
                else:
                    attn_mask = self.pair_embed(v, uu).view(-1, v.size(-1), v.size(-1))

            for block in self.blocks:
                x = block(x, x_cls=None, padding_mask=padding_mask, attn_mask=attn_mask)

            cls_tokens = self.cls_token.expand(1, x.size(1), -1)
            for block in self.cls_blocks:
                cls_tokens = block(x, x_cls=cls_tokens, padding_mask=padding_mask)

            x_cls = self.norm(cls_tokens).squeeze(0)

            if self.fc is None:
                return x_cls
            output = self.fc(x_cls)
            if self.for_inference:
                output = torch.softmax(output, dim=1)
            return output

    def eval(self):
        """Override eval() to also build the fused pair MLP."""
        super().eval()
        self._build_fused_pair_mlp()
        return self

    @classmethod
    def from_pretrained(cls, orig_model):
        """Build from a standard weaver ParticleTransformer."""
        return cls(orig_model)
