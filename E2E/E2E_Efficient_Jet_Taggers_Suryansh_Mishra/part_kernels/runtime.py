"""High-level runtime helpers for optimized ParticleTransformer workflows.

``optimize_part_model`` patches the internals of a weaver-core
``ParticleTransformer`` instance **in place** (per-instance
``types.MethodType`` patches, never class-level) so that:

- ``PairEmbed.forward`` mirrors weaver's ``_forward_sparse`` structure
  (valid-pair gather -> embed -> scatter; BatchNorm statistics stay
  restricted to valid pairs) with the full pair-feature grid computed by
  the fused Triton kernel on CUDA and by weaver's own
  ``pairwise_lv_fts_pp`` reference math on CPU (Req 8.3).
- ``Attention.forward`` (for every ``block.attn`` in ``model.blocks``;
  ``cls_blocks`` untouched) dispatches to the fused Triton attention
  kernel when the call is eligible, and delegates to the saved original
  weaver forward otherwise.

``unpatch_part_model`` restores the saved originals exactly.

Import safety (Req 2.6, 8.3): this module never imports triton at module
scope; the fused kernels are imported lazily inside the CUDA dispatch
branches only.
"""

from __future__ import annotations

import math
import types
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

from . import _compat

#: Attribute set on the patched model to hold unpatch bookkeeping.
_PATCH_STATE_ATTR = "_part_kernels_patch"


# ---------------------------------------------------------------------------
# weaver patcher (Req 2.1, 2.2, 8.3)
# ---------------------------------------------------------------------------

def optimize_part_model(
    model: nn.Module,
    *,
    compile_mode: str | None = None,
    use_pairwise_patch: bool = True,
    use_attention_patch: bool = True,
) -> tuple[nn.Module, dict[str, Any]]:
    """Patch weaver ``ParticleTransformer`` internals in-place with fused kernels.

    On CUDA (with triton available) the patched forwards run the fused Triton
    kernels; on CPU every dispatch site falls back to weaver's own reference
    math, so the patched model is behaviorally equivalent to the pristine
    model (Req 8.3). Asserts all patch targets exist first (Req 2.2) via
    :func:`_compat.assert_weaver_compat`.

    Parameters
    ----------
    model:
        A weaver ``ParticleTransformer`` instance. Anything else raises
        ``TypeError``.
    compile_mode:
        If not None, the patched model is wrapped in
        ``torch.compile(model, mode=compile_mode)``.
    use_pairwise_patch:
        Patch every ``PairEmbed`` instance (Patch A). Unsupported
        configurations (extra pairwise inputs ``uu`` /
        ``pairwise_input_dim > 0``, ``pairwise_lv_dim != 4``, non-``'pp'``
        pairwise type) transparently fall back to the original forward at
        call time; occurrences are counted in ``stats["pairwise_fallbacks"]``.
    use_attention_patch:
        Patch ``Attention.forward`` for every ``block.attn`` in
        ``model.blocks`` (Patch B). The fused path is taken at call time only
        for self-attention with identity q/k norms, no output gates, no
        train-mode attention dropout (the fused kernel implements no
        attention dropout -- such calls preserve training semantics by
        delegating to the original weaver path), and triton+CUDA available;
        otherwise the saved original forward runs. ``cls_blocks`` are left
        unpatched (class-attention shape does not fit the square-P kernel).

    Returns
    -------
    (model, stats):
        The model patched in-place (wrapped by ``torch.compile`` if
        ``compile_mode`` is set) and a metadata dict with keys:
        ``"patches"`` (list of applied patch names),
        ``"pair_embed_instances"`` / ``"attention_instances"`` (counts),
        ``"pairwise_fallbacks"`` (call-time fallback counter, live),
        ``"device_dispatch"`` (``"triton"`` or ``"cpu-stub"`` at patch time),
        ``"compile_mode"``, and ``"params"`` (parameter count).

    Use :func:`unpatch_part_model` to restore the saved originals.
    """
    # Req 2.2: every patch target must exist before any patching is attempted.
    _compat.assert_weaver_compat()

    from weaver.nn.model.ParticleTransformer import (
        Attention,
        PairEmbed,
        ParticleTransformer,
        pairwise_lv_fts_pp,
    )

    if not isinstance(model, ParticleTransformer):
        raise TypeError(
            "optimize_part_model expects a "
            "'weaver.nn.model.ParticleTransformer.ParticleTransformer' "
            f"instance; got {type(model).__module__}.{type(model).__qualname__}"
        )

    if getattr(model, _PATCH_STATE_ATTR, None) is not None:
        raise RuntimeError(
            "model is already patched by optimize_part_model; call "
            "unpatch_part_model(model) before re-patching"
        )

    stats: dict[str, Any] = {
        "patches": [],
        "pair_embed_instances": 0,
        "attention_instances": 0,
        "pairwise_fallbacks": 0,
        "device_dispatch": "triton" if _compat.has_triton() else "cpu-stub",
        "compile_mode": compile_mode,
        "params": sum(p.numel() for p in model.parameters()),
    }

    patched_pair_embeds: list[nn.Module] = []
    patched_attentions: list[nn.Module] = []

    # ---- Patch A: PairEmbed.forward (per instance) -------------------------
    if use_pairwise_patch:
        for module in model.modules():
            if isinstance(module, PairEmbed):
                _patch_pair_embed(module, stats, pairwise_lv_fts_pp)
                patched_pair_embeds.append(module)
        stats["pair_embed_instances"] = len(patched_pair_embeds)
        if patched_pair_embeds:
            stats["patches"].append("pair_embed")

    # ---- Patch B: Attention.forward for every block.attn in model.blocks ---
    # cls_blocks are intentionally left unpatched.
    if use_attention_patch:
        for block in model.blocks:
            attn = getattr(block, "attn", None)
            if isinstance(attn, Attention):
                _patch_attention(attn)
                patched_attentions.append(attn)
        stats["attention_instances"] = len(patched_attentions)
        if patched_attentions:
            stats["patches"].append("attention")

    setattr(model, _PATCH_STATE_ATTR, {
        "pair_embeds": patched_pair_embeds,
        "attentions": patched_attentions,
        "stats": stats,
    })

    if compile_mode is not None:
        model = torch.compile(model, mode=compile_mode)

    return model, stats


def unpatch_part_model(model: nn.Module) -> nn.Module:
    """Restore the original forwards saved by :func:`optimize_part_model`.

    Removes every per-instance ``forward`` override (restoring class-level
    method lookup, i.e. the exact pristine weaver behavior) and the patch
    bookkeeping. Accepts either the patched model or its ``torch.compile``
    wrapper. A model that was never patched is returned unchanged.
    """
    model = getattr(model, "_orig_mod", model)  # unwrap torch.compile
    state = model.__dict__.pop(_PATCH_STATE_ATTR, None)
    if state is None:
        return model
    for instance in (*state["pair_embeds"], *state["attentions"]):
        instance.__dict__.pop("forward", None)
    return model


# ---------------------------------------------------------------------------
# Patch A -- PairEmbed.forward
# ---------------------------------------------------------------------------

def _patch_pair_embed(pair_embed: nn.Module, stats: dict[str, Any],
                      pairwise_lv_fts_pp) -> None:
    """Install the per-instance ``PairEmbed.forward`` patch.

    The patched forward mirrors weaver ``_forward_sparse`` exactly --
    valid-pair gather -> ``self.embed`` -> scatter -- so BatchNorm statistics
    remain restricted to valid pairs in training mode. Only the full-grid
    pair-feature computation is replaced: the fused Triton kernel on CUDA,
    weaver's own ``pairwise_lv_fts_pp`` reference math on CPU.
    """
    original_forward = pair_embed.forward  # bound class method, pre-patch

    # Static config support: exactly weaver's 'pp' pairwise features with the
    # 4 outputs [lnkt, lnz, lndelta, lnm2] the Triton kernel produces, and no
    # extra pairwise-input embedding path.
    supported = (
        getattr(pair_embed, "pairwise_lv_dim", None) == 4
        and getattr(pair_embed, "pairwise_input_dim", 0) == 0
        and getattr(getattr(pair_embed, "pairwise_lv_fts", None), "func", None)
        is pairwise_lv_fts_pp
    )
    keywords = getattr(getattr(pair_embed, "pairwise_lv_fts", None), "keywords", None) or {}
    eps = keywords.get("eps", 1e-8)

    def patched_forward(self, x, uu=None, mask=None):
        # Unsupported configuration: transparent fallback to the original
        # weaver forward, counted so silent non-acceleration is observable.
        if not supported or uu is not None or x is None:
            stats["pairwise_fallbacks"] += 1
            return original_forward(x, uu=uu, mask=mask)

        # Mirror the pristine dispatch: only the sparse path is
        # re-implemented; the dense path delegates unchanged.
        sparse_eval = self.sparse_eval[0 if self.training else 1]
        if not (sparse_eval and mask is not None):
            return original_forward(x, uu=uu, mask=mask)

        # --- weaver _forward_sparse structure ------------------------------
        batch_size, _, seq_len = x.size()
        pair_mask = mask.unsqueeze(-1) * mask.unsqueeze(-2)  # (B, 1, P, P)
        if self.is_symmetric:
            offset = -1 if self.remove_self_pair else 0
            i0, _, i2, i3 = pair_mask.float().tril(offset).nonzero(as_tuple=True)
        else:
            i0, _, i2, i3 = pair_mask.nonzero(as_tuple=True)

        # Full (B, 4, P, P) pair-feature grid: fused Triton kernel on CUDA,
        # weaver reference math as the CPU stub (Req 8.3). Channel order
        # matches pairwise_lv_fts_pp(num_outputs=4): [lnkt, lnz, lndelta, lnm2].
        if x.is_cuda and _compat.has_triton():
            from .triton.pairwise_kernel import fused_pairwise_lv_fts

            grid = fused_pairwise_lv_fts(x, eps=eps)
        else:
            grid = self.pairwise_lv_fts(x.unsqueeze(-1), x.unsqueeze(-2))

        # Valid-pair gather -> embed -> scatter. Running the embed nets on
        # gathered valid pairs only keeps BatchNorm statistics restricted to
        # valid pairs (identical index logic to weaver _forward_sparse).
        fts = grid.permute(0, 2, 3, 1)[i0, i2, i3, :]  # (num_elements, 4)
        fts = fts.T.unsqueeze(0)                       # (1, 4, num_elements)
        elements = self.embed(fts)                     # (1, out_dim, num_elements)
        elements = elements.squeeze(0).T               # (num_elements, out_dim)

        y = torch.zeros(
            batch_size, seq_len, seq_len, self.out_dim,
            dtype=elements.dtype, device=elements.device,
        )
        y[i0, i2, i3, :] = elements
        if self.is_symmetric:
            y[i0, i3, i2, :] = elements
        return y.permute(0, 3, 1, 2).contiguous()

    pair_embed.forward = types.MethodType(patched_forward, pair_embed)


# ---------------------------------------------------------------------------
# Patch B -- Attention.forward
# ---------------------------------------------------------------------------

def _patch_attention(attn: nn.Module) -> None:
    """Install the per-instance ``Attention.forward`` patch.

    Call-time dispatch: the fused Triton path runs only when the call is
    plain self-attention with identity q/k norms, no output gates, no
    train-mode attention dropout, and triton+CUDA are available. Every other
    call delegates to the saved original weaver forward (the CPU stub --
    Req 8.3).
    """
    original_forward = attn.forward  # bound class method, pre-patch

    def patched_forward(self, query, key, value,
                        key_padding_mask=None, attn_mask=None):
        bsz, tgt_len, _ = query.shape
        use_fused = (
            # self-attention only: the class-attention path has tgt_len != src_len
            query is key and key is value
            # identity q/k norms; no output gates
            and isinstance(self.q_norm, nn.Identity)
            and isinstance(self.k_norm, nn.Identity)
            and not getattr(self, "headwise_attn_output_gate", False)
            and not getattr(self, "elementwise_attn_output_gate", False)
            # fused kernel implements no attention dropout
            and (not self.training or self.dropout == 0.0)
            # mask layouts the fused kernel understands
            and (key_padding_mask is None or key_padding_mask.dtype == torch.bool)
            and (attn_mask is None
                 or (torch.is_floating_point(attn_mask)
                     and attn_mask.shape == (bsz, self.num_heads, tgt_len, tgt_len)))
            # triton + CUDA available
            and query.is_cuda
            and _compat.has_triton()
        )
        if not use_fused:
            return original_forward(
                query, key, value,
                key_padding_mask=key_padding_mask, attn_mask=attn_mask,
            )

        from .autograd.attention import fused_attention_with_bias

        num_heads, head_dim = self.num_heads, self.head_dim
        weight, bias_param = _compat.get_in_proj_params(self)
        q, k, v = F._in_projection_packed(query, key, value, weight, bias_param)
        q = (q.view(bsz, tgt_len, num_heads, head_dim)
              .transpose(1, 2).reshape(bsz * num_heads, tgt_len, head_dim))
        k = (k.view(bsz, tgt_len, num_heads, head_dim)
              .transpose(1, 2).reshape(bsz * num_heads, tgt_len, head_dim))
        v = (v.view(bsz, tgt_len, num_heads, head_dim)
              .transpose(1, 2).reshape(bsz * num_heads, tgt_len, head_dim))

        # weaver attn_mask (B, H, P, P) float bias -> (B*H, P, P);
        # bool key_padding_mask (B, P) passes straight through.
        bias = None
        if attn_mask is not None:
            bias = attn_mask.reshape(bsz * num_heads, tgt_len, tgt_len)

        out = fused_attention_with_bias(
            q, k, v, bias, key_padding_mask,
            1.0 / math.sqrt(head_dim), num_heads,
        )
        out = (out.view(bsz, num_heads, tgt_len, head_dim)
                  .transpose(1, 2).reshape(bsz, tgt_len, self.embed_dim))
        return self.out_proj(out), None

    attn.forward = types.MethodType(patched_forward, attn)
