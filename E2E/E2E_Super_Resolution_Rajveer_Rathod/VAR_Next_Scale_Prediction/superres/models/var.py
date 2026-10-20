"""VAR transformer adapted to super-resolution.

Same next-scale prediction as the paper (eq. 6): the token maps r_1..r_K of the
HR image are predicted coarse-to-fine with a block-wise causal mask, a GPT-2
style decoder, AdaLN and L2-normalised queries/keys. The class-label start token
is replaced by the low-resolution measurement:

  * an LR encoder maps the (nearest-upsampled) LR image to the latent grid;
  * its pooled vector is the start token [s] and the AdaLN condition;
  * its feature map, resized to each scale, is added to that scale's tokens,
    so every predicted token sees the coarse energy deposited "under" it.
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .nn_utils import ConvNd, ResBlock, norm
from .vqvae import down


class SelfAttention(nn.Module):
    def __init__(self, D, H, attn_drop=0.0, proj_drop=0.0):
        super().__init__()
        self.H, self.hd = H, D // H
        self.qkv = nn.Linear(D, 3 * D)
        self.proj = nn.Linear(D, D)
        self.drop = nn.Dropout(proj_drop)
        self.attn_drop = attn_drop
        self.log_scale = nn.Parameter(torch.full((1, H, 1, 1), math.log(4.0)))

    def forward(self, x, mask):
        B, L, D = x.shape
        q, k, v = self.qkv(x).view(B, L, 3, self.H, self.hd).permute(2, 0, 3, 1, 4)
        scale = self.log_scale.clamp(max=math.log(100.0)).exp()
        q = F.normalize(q, dim=-1) * scale.to(q.dtype)
        k = F.normalize(k, dim=-1)
        o = F.scaled_dot_product_attention(q, k, v, attn_mask=mask,
                                           dropout_p=self.attn_drop if self.training else 0.0, scale=1.0)
        return self.drop(self.proj(o.transpose(1, 2).reshape(B, L, D)))


class AdaLNBlock(nn.Module):
    def __init__(self, D, H, mlp_ratio=4.0, drop=0.0):
        super().__init__()
        self.ln1 = nn.LayerNorm(D, elementwise_affine=False, eps=1e-6)
        self.ln2 = nn.LayerNorm(D, elementwise_affine=False, eps=1e-6)
        self.attn = SelfAttention(D, H, attn_drop=drop, proj_drop=drop)
        hidden = int(D * mlp_ratio)
        self.mlp = nn.Sequential(nn.Linear(D, hidden), nn.GELU(approximate="tanh"),
                                 nn.Linear(hidden, D), nn.Dropout(drop))
        self.ada = nn.Sequential(nn.SiLU(), nn.Linear(D, 6 * D))
        nn.init.zeros_(self.ada[1].weight)
        nn.init.zeros_(self.ada[1].bias)

    def forward(self, x, c, mask):
        g1, g2, s1, s2, b1, b2 = self.ada(c).view(-1, 1, 6, x.shape[-1]).unbind(2)
        x = x + g1 * self.attn(self.ln1(x) * (1 + s1) + b1, mask)
        return x + g2 * self.mlp(self.ln2(x) * (1 + s2) + b2)


class LREncoder(nn.Module):
    """LR image -> feature map on the VQ latent grid (same strides as the VQ encoder)."""

    def __init__(self, ndim, in_ch, widths, strides, D, circ=()):
        super().__init__()
        assert len(widths) == len(strides) + 1
        layers, c = [ConvNd(ndim, in_ch, widths[0], circular_dims=circ)], widths[0]
        for i, w in enumerate(widths):
            layers.append(ResBlock(ndim, c, w, circ))
            c = w
            if i < len(strides):
                layers.append(ConvNd(ndim, c, c, 3, stride=tuple(strides[i]), circular_dims=circ))
        layers += [norm(c), nn.SiLU(), ConvNd(ndim, c, D, 1)]
        self.net = nn.Sequential(*layers)

    def forward(self, lr, hr_shape):
        return self.net(F.interpolate(lr, size=tuple(hr_shape), mode="nearest"))


def sample_logits(logits, temperature=1.0, top_k=0, greedy=False, generator=None):
    if greedy:
        return logits.argmax(-1)
    logits = logits / max(temperature, 1e-5)
    if top_k and top_k > 0:
        kth = logits.topk(top_k, dim=-1).values[..., -1:]
        logits = logits.masked_fill(logits < kth, float("-inf"))
    probs = logits.softmax(-1)
    flat = torch.multinomial(probs.reshape(-1, probs.shape[-1]), 1, generator=generator)
    return flat.view(logits.shape[:-1])


class ConditionalVAR(nn.Module):
    def __init__(self, ndim, in_ch, hr_shape, vq_cfg, var_cfg, circ=()):
        super().__init__()
        self.hr_shape = tuple(hr_shape)
        self.scales = [tuple(s) for s in vq_cfg["scales"]]
        self.V, self.Cvae = vq_cfg["codebook_size"], vq_cfg["z_ch"]
        D, depth, H = var_cfg["width"], var_cfg["depth"], var_cfg["heads"]
        self.D = D
        lens = [math.prod(s) for s in self.scales]
        self.L = sum(lens)
        ends = torch.tensor(lens).cumsum(0).tolist()
        self.begin_ends = [(e - n, e) for n, e in zip(lens, ends)]

        self.lr_enc = LREncoder(ndim, in_ch, var_cfg["lr_enc_widths"], vq_cfg["strides"], D, circ)
        self.cond_mlp = nn.Sequential(nn.Linear(D, D), nn.SiLU(), nn.Linear(D, D))
        self.spatial_proj = nn.Linear(D, D)
        self.word_embed = nn.Linear(self.Cvae, D)
        self.pos_start = nn.Parameter(torch.zeros(1, 1, D))
        self.pos = nn.Parameter(torch.zeros(1, self.L, D))
        self.lvl_embed = nn.Embedding(len(self.scales), D)
        for p in (self.pos_start, self.pos, self.lvl_embed.weight):
            nn.init.trunc_normal_(p, std=0.02)
        self.blocks = nn.ModuleList(AdaLNBlock(D, H, var_cfg["mlp_ratio"], var_cfg["drop"]) for _ in range(depth))
        self.head_ln = nn.LayerNorm(D, elementwise_affine=False, eps=1e-6)
        self.head_ada = nn.Sequential(nn.SiLU(), nn.Linear(D, 2 * D))
        nn.init.zeros_(self.head_ada[1].weight)
        nn.init.zeros_(self.head_ada[1].bias)
        self.head = nn.Linear(D, self.V)

        lvl = torch.cat([torch.full((n,), k) for k, n in enumerate(lens)])
        self.register_buffer("lvl_ids", lvl, persistent=False)
        # block-wise causal mask: scale k attends to scales <= k (True = attend)
        self.register_buffer("mask", lvl[:, None] >= lvl[None, :], persistent=False)

    def condition(self, lr):
        feat = self.lr_enc(lr, self.hr_shape)
        c = self.cond_mlp(feat.flatten(2).mean(2))
        sp = torch.cat([down(feat, s).flatten(2).transpose(1, 2) for s in self.scales], 1)
        return c, self.spatial_proj(sp)

    def _sequence(self, c, sp, x_in):
        x = c[:, None] + self.pos_start
        if x_in is not None and x_in.shape[1] > 0:
            x = torch.cat([x, self.word_embed(x_in.to(x.dtype))], 1)
        n = x.shape[1]
        return x + self.pos[:, :n] + self.lvl_embed(self.lvl_ids[:n]) + sp[:, :n]

    def _run(self, x, c):
        n = x.shape[1]
        mask = self.mask[:n, :n]
        for blk in self.blocks:
            x = blk(x, c, mask)
        shift, scale = self.head_ada(c).view(-1, 1, 2, self.D).unbind(2)
        return self.head(self.head_ln(x) * (1 + scale) + shift)

    def forward(self, lr, x_in):
        """Teacher forcing: x_in = vq.idx_to_var_input(gt_idx) -> logits (B, L, V)."""
        c, sp = self.condition(lr)
        return self._run(self._sequence(c, sp, x_in), c)

    @torch.no_grad()
    def generate(self, lr, vqvae, temperature=1.0, top_k=0, greedy=False, generator=None):
        """Coarse-to-fine token prediction; the tokenizer decodes them together with the LR image.
        Returns the hard output in normalised log-energy space (exact zeros where no hit)."""
        c, sp = self.condition(lr)
        B = lr.shape[0]
        f_hat = torch.zeros(B, self.Cvae, *self.scales[-1], device=lr.device)
        inputs = []
        for k, s in enumerate(self.scales):
            x_in = torch.cat(inputs, 1) if inputs else None
            b, e = self.begin_ends[k]
            logits = self._run(self._sequence(c, sp, x_in), c)[:, b:e].float()
            idx = sample_logits(logits, temperature, top_k, greedy, generator).view(B, *s)
            f_hat, nxt = vqvae.vq.next_input(k, f_hat, idx)
            if nxt is not None:
                inputs.append(nxt)
        return vqvae.decode_fhat(f_hat, lr)
