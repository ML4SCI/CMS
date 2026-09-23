"""Weaver-core interface adaptation for the variant blocks (design DD6).

Provides the three pieces that let the self-contained variant blocks occupy
encoder slots inside the official weaver-core ``ParticleTransformer``
(pinned to git commit ``154db69``) without modifying any installed weaver
sources:

- :class:`WeaverBlockAdapter` — translates weaver's
  ``Block.forward(x, x_cls, padding_mask, attn_mask)`` call signature to the
  variants' ``forward(x, padding_mask, U)``.
- :func:`build_variant_part` — factory constructing a weaver
  ``ParticleTransformer`` with its encoder ``blocks`` replaced by adapted
  variant blocks (``"baseline"`` returns stock weaver).
- :func:`collect_moe_aux_loss` — sums the MoE load-balancing auxiliary loss
  over all :class:`~variants.blocks.moe_feedforward.MoEFeedforward` modules.

Interface-compatibility facts (verified against the installed weaver-core
source at pin ``154db69``, version string 0.5.3):

- weaver ``Block`` is **batch-first**: ``x`` is ``(B, N, C)`` with padded
  positions zeroed by the encoder (``embed(x).masked_fill``) — this matches
  the variant blocks' expectation directly, so no seq-first/batch-first
  translation is needed.
- weaver calls encoder blocks only with ``x_cls=None``; class attention lives
  in separate ``cls_blocks``, which all arms keep as stock weaver ``Block``s.
  The adapter therefore raises ``RuntimeError`` if ``x_cls`` is provided.
- ``padding_mask`` is ``(B, N)`` bool with ``True`` = padded (weaver derives
  it as ``~mask.squeeze(1)``) — identical semantics to the variants'
  ``padding_mask.bool()...masked_fill``.
- ``attn_mask`` is the pair-embed bias ``(B, num_heads, N, N)``, contiguous
  float — the variants' ``U.view(B, num_heads, N, N)`` accepts it unchanged
  (they also accept the legacy ``(B*num_heads, N, N)`` layout).
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Optional, Sequence, Union

import torch
from torch import Tensor, nn

from weaver.nn.model.ParticleTransformer import ParticleTransformer

from .blocks.attention_block import SoftmaxAttentionBlock
from .blocks.differential_attention import DifferentialAttentionBlock
from .blocks.differential_attention_v2 import DifferentialAttentionV2Block
from .blocks.moe_feedforward import MoEFeedforward
from .blocks.sparsemax_attention import SparsemaxAttentionBlock
from .blocks.u_rotary_attention import URotaryAttentionBlock
from .lloca.part import LLoCaParT
from .n8.part import N8ParT

__all__ = [
    "WeaverBlockAdapter",
    "VARIANTS",
    "MOE_PRESETS",
    "DEFAULT_MOE_PRESET",
    "build_variant_part",
    "collect_moe_aux_loss",
]


class WeaverBlockAdapter(nn.Module):
    """Adapts a variant block to weaver's ``Block`` call signature.

    Wraps a block exposing ``forward(x, padding_mask, U)`` so it can occupy a
    slot in ``ParticleTransformer.blocks`` — weaver calls its encoder blocks
    as ``block(x, x_cls=None, padding_mask=..., attn_mask=...)``.

    Variant blocks are encoder-stack only: class attention must stay in
    weaver's stock ``cls_blocks``, so a non-``None`` ``x_cls`` raises
    ``RuntimeError`` (fail fast instead of silently mis-handling class
    attention).
    """

    def __init__(self, block: nn.Module):
        super().__init__()
        self.block = block

    def forward(
        self,
        x: Tensor,
        x_cls: Optional[Tensor] = None,
        padding_mask: Optional[Tensor] = None,
        attn_mask: Optional[Tensor] = None,
    ) -> Tensor:
        if x_cls is not None:
            raise RuntimeError(
                "variant blocks are encoder-stack only; "
                "cls_blocks must remain standard weaver Blocks"
            )
        return self.block(x, padding_mask, attn_mask)


#: ParT-family ablation arms accepted by :func:`build_variant_part`.
VARIANTS = {
    "baseline",
    "tied",
    "lowrank",
    "lloca",
    "n8",
    "sparsemax",
    "moe",
    "diff_v1",
    "diff_v2",
    "urot",
}

#: Arms whose blocks replace entries of ``ParticleTransformer.blocks`` in place.
#: ``"lloca"`` and ``"n8"`` are excluded: they intercept inputs before
#: embedding (frames / post-trim momenta) so they wrap a transformer rather
#: than swapping blocks — see :class:`~variants.lloca.part.LLoCaParT` and
#: :class:`~variants.n8.part.N8ParT`.
_BLOCK_SWAP_VARIANTS = {"sparsemax", "moe", "diff_v1", "diff_v2", "urot"}

#: Arms whose encoder blocks are STOCK weaver ``Block``s and therefore carry the ``ls1``/``ls2``
#: LayerScale hooks that ``residual_scale_lambda`` is realised through. Every other arm replaces
#: or wraps the blocks and cannot honour the field; :func:`build_variant_part` refuses it there
#: rather than store a control that never existed (2026-09-09 audit, A1).
_LAYER_SCALE_ARMS = {"baseline", "tied", "lowrank"}

#: Named MoE configurations for the ``"moe"`` arm.  ``H`` below is the dense
#: baseline FFN hidden width (``embed_dim * expansion_factor``, i.e. 512 for
#: ParT).  Pick one with ``build_variant_part("moe", moe_config="<name>")`` or
#: pass an explicit dict of :class:`~variants.blocks.moe_feedforward.MoEFeedforward`
#: keyword arguments.
#:
#: ==========================  ==========  ============  ==================
#: preset                      FLOPs       FFN params    per-expert hidden
#: ==========================  ==========  ============  ==================
#: ``flop_matched_full``       ``1.00x``   ``4.00x``     ``H``
#: ``flop_matched_top2``       ``1.00x``   ``2.00x``     ``H / 2``
#: ``e8_top2_subflop``         ``0.75x``   ``3.00x``     ``3H / 8`` (192)
#: ``shared_flop_matched``     ``1.00x``   ``2.00x``     ``H / 2``
#: ``param_matched``           ``0.50x``   ``1.00x``     ``H / 4``
#: ==========================  ==========  ============  ==================
MOE_PRESETS = {
    # Each expert is exactly the dense ParT FFN; one fires per token, so the
    # arm costs the baseline's FLOPs and holds 4x its FFN capacity.
    "flop_matched_full": dict(
        num_experts=4, top_k=1, hidden_mode="full", gate_mode="full_softmax"
    ),
    # Two half-width experts per token: same FLOPs, 2x params, but the token is
    # a mixture of two specialists rather than assigned to one.
    "flop_matched_top2": dict(num_experts=4, top_k=2, hidden_mode="flop_matched"),
    # Eight narrower experts, two active: Mixtral-style routing with activated
    # FFN under the dense baseline.  h=192 → 2×192/512 = 0.75× FLOPs, 8×192/512
    # = 3× stored.  Addresses the top-1 sparse-update gap without matching
    # (or exceeding) baseline compute.
    "e8_top2_subflop": dict(num_experts=8, top_k=2, expert_hidden=192),
    # One always-on shared expert plus one routed expert, both half width
    # (DeepSeekMoE-style): keeps a dense path while the routed half specializes.
    "shared_flop_matched": dict(
        num_experts=3,
        top_k=1,
        num_shared_experts=1,
        hidden_mode="flop_matched",
        gate_mode="full_softmax",
    ),
    # Identical parameter count to the dense baseline, at half its FLOPs.
    "param_matched": dict(num_experts=4, top_k=2, hidden_mode="param_matched"),
}

#: Default preset for the ``"moe"`` arm — FLOP-matched to the dense baseline.
DEFAULT_MOE_PRESET = "flop_matched_full"


def _resolve_moe_config(moe_config) -> dict:
    """Normalize ``moe_config`` (``None`` / preset name / dict) to kwargs.

    Raises
    ------
    ValueError
        If a preset name is not in :data:`MOE_PRESETS`.
    TypeError
        If ``moe_config`` is neither ``None``, a string, nor a mapping.
    """
    if moe_config is None:
        return dict(MOE_PRESETS[DEFAULT_MOE_PRESET])
    if isinstance(moe_config, str):
        if moe_config not in MOE_PRESETS:
            raise ValueError(
                f"unknown MoE preset {moe_config!r}; expected one of "
                f"{sorted(MOE_PRESETS)}"
            )
        return dict(MOE_PRESETS[moe_config])
    if isinstance(moe_config, Mapping):
        return dict(moe_config)
    raise TypeError(
        f"moe_config must be None, a preset name, or a mapping; "
        f"got {type(moe_config).__name__}"
    )


def _make_variant_block(
    variant: str,
    *,
    embed_dim: int,
    num_heads: int,
    dropout: float,
    expansion_factor: int,
    moe_config: Optional[dict] = None,
    per_plane: bool = False,
    rope_apply: bool = False,
) -> nn.Module:
    """Construct one freshly-initialized variant block for the given arm."""
    if variant == "sparsemax":
        return SparsemaxAttentionBlock(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            expansion_factor=expansion_factor,
        )
    if variant == "moe":
        # DD5: the MoE arm is hosted by SoftmaxAttentionBlock so all four
        # variant arms share identical block wiring and differ in exactly
        # one component (here: the FFN).
        return SoftmaxAttentionBlock(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            expansion_factor=expansion_factor,
            ffn=MoEFeedforward(
                embed_dim,
                expansion_factor,
                dropout,
                **(moe_config or {}),
            ),
        )
    if variant == "diff_v1":
        return DifferentialAttentionBlock(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            expansion_factor=expansion_factor,
        )
    if variant == "diff_v2":
        return DifferentialAttentionV2Block(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            expansion_factor=expansion_factor,
        )
    if variant == "urot":
        return URotaryAttentionBlock(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            expansion_factor=expansion_factor,
            per_plane=per_plane,
            rope_apply=rope_apply,
        )
    raise ValueError(
        f"variant {variant!r} is not a block-swap arm; expected one of "
        f"{sorted(_BLOCK_SWAP_VARIANTS)}"
    )


def _ffn_second_linears(block: nn.Module) -> list[nn.Linear]:
    """Return the FFN second-linear layer(s) of a variant block.

    For the standard ``Feedforward`` this is ``linear2``; for the MoE FFN it
    is every expert's ``linear2`` — routed *and* shared (each expert is a small
    two-layer FFN, so its ``linear2`` is the analogue of weaver's ``fc2``).
    """
    ffn = block.feedforward
    if isinstance(ffn, MoEFeedforward):
        return [expert.linear2 for expert in ffn.all_experts]
    return [ffn.linear2]


def _apply_weaver_rescale(blocks: nn.ModuleList) -> None:
    """Replicate weaver's ``ParticleTransformer.fix_init_weight`` on adapted blocks.

    Weaver's rescale (source at pin ``154db69``)::

        def rescale(param, _layer_id):
            param.div_(math.sqrt(2.0 * _layer_id))

        for layer_id, layer in enumerate(self.blocks):
            rescale(layer.attn.out_proj.weight.data, layer_id + 1)
            rescale(layer.fc2.weight.data, layer_id + 1)

    i.e. layer ``i`` (0-based) has its attention output projection weight and
    FFN second-linear weight divided by ``sqrt(2 * (i + 1))``.

    ``fix_init_weight`` indexes ``layer.attn.out_proj`` / ``layer.fc2``, which
    do not exist on :class:`WeaverBlockAdapter` — so variant models are built
    with ``fix_init=False`` and this function applies the identical per-layer
    factor to the analogous sublayers of each variant block: the attention
    output projection (``block.out_proj``) and the FFN second linear
    (``Feedforward.linear2``, or every expert's ``linear2`` for the MoE FFN),
    keeping initialization comparable to stock weaver.
    """
    with torch.no_grad():
        for layer_id, adapter in enumerate(blocks):
            factor = math.sqrt(2.0 * (layer_id + 1))
            block = adapter.block
            block.out_proj.weight.data.div_(factor)
            for linear in _ffn_second_linears(block):
                linear.weight.data.div_(factor)


def _assert_realized_ffn_width(model, embed_dims, expansion_factor, variant: str) -> None:
    """Fail loudly when the FFN width the model actually built differs from the one requested.

    This exists because a silently-ignored config key is worse than one that raises: the
    ``ffn2x`` arm recorded ``expansion_factor: 8`` in its checkpoint while training a width-512
    FFN, so its config, its provenance JSON and its own filename all asserted an experiment that
    never ran. Nothing could catch it except comparing a tensor shape against the request --
    which is exactly what this does, at build time, before any GPU-hours are spent.

    Deliberately permissive about *finding* the layer: weaver block internals are not a stable
    contract, so if ``fc1`` is absent or has no ``out_features`` the check is skipped rather than
    made brittle. It only fires on a genuine, detectable mismatch.
    """
    if not embed_dims:
        return
    blocks = getattr(model, "blocks", None)
    if not blocks:
        return
    fc1 = getattr(blocks[0], "fc1", None)
    realized = getattr(fc1, "out_features", None)
    if realized is None:
        return
    expected = int(embed_dims[-1] * expansion_factor)
    if realized != expected:
        raise RuntimeError(
            f"variant {variant!r}: requested expansion_factor={expansion_factor} with "
            f"embed_dim={embed_dims[-1]} implies an FFN hidden width of {expected}, but the "
            f"model built {realized}. The config would have been stored as if it applied. "
            "Refusing to train a model whose architecture does not match its own provenance."
        )


def _assert_realized_layer_scale(model, layer_scale_init_values, variant: str) -> None:
    """Fail loudly unless every encoder block carries a real LayerScale when one was requested.

    Companion to :func:`_assert_realized_ffn_width` for the residual-branch scale. weaver only
    installs ``LayerScale`` when ``layer_scale_init_values`` is truthy and otherwise leaves an
    ``nn.Identity`` at ``ls1``/``ls2`` -- so a value that did not survive the ``block_params``
    merge would build a model that trains without its control and still records the field in its
    provenance. Asserting on the module type of the built model is what closes that gap.
    """
    if layer_scale_init_values is None:
        return
    blocks = getattr(model, "blocks", None)
    if not blocks:
        return
    missing = [
        i
        for i, block in enumerate(blocks)
        for name in ("ls1", "ls2")
        # Unwrap MoRBlock / ModulatedBlock, which hold the stock Block at `.block`.
        if not hasattr(getattr(getattr(block, "block", block), name, None), "gamma")
    ]
    if missing:
        raise RuntimeError(
            f"variant {variant!r}: residual_scale_lambda was set, but encoder block(s) "
            f"{sorted(set(missing))} have no LayerScale on ls1/ls2. The config would have been "
            "stored as if it applied. Refusing to build a model without its own control."
        )


def build_variant_part(
    variant: str,
    *,
    input_dim: int,
    num_classes: int,
    embed_dims=(128, 512, 128),
    pair_embed_dims=(64, 64, 64),
    num_heads=8,
    num_layers=8,
    num_cls_layers=2,
    dropout=0.1,
    expansion_factor=4,
    moe_config: Optional[Union[str, Mapping]] = None,
    layer_scale_init_values: Optional[float] = None,
    residual_scale_lambda: Optional[float] = None,
    lowrank_rank: int = 32,
    lowrank_hidden: Optional[Sequence[int]] = None,
    lowrank_self_pair: bool = True,
    tie_num_unique: Optional[int] = None,
    tie_scope: str = "block",
    tie_strategy: str = "cycle",
    tie_depth_modulation: bool = False,
    tie_mor_capacity: Optional[float] = None,
    **weaver_kwargs,
) -> nn.Module:
    """Build a weaver ``ParticleTransformer`` for one ablation arm.

    Constructs a weaver ``ParticleTransformer``, then (for non-baseline
    variants) replaces every entry of ``model.blocks`` with
    ``WeaverBlockAdapter(variant_block)``. The variant blocks' ``embed_dim``
    is taken from ``embed_dims[-1]``; ``cls_blocks`` / ``pair_embed`` /
    trimmer remain stock weaver.

    ``"baseline"`` returns stock weaver unchanged (default ``fix_init``).
    Variant arms are constructed with ``fix_init=False`` (weaver's
    ``fix_init_weight`` indexes ``layer.attn.out_proj`` / ``layer.fc2``,
    which do not exist on adapters) and the identical rescale is applied
    manually to each variant block's ``out_proj`` / FFN second linear —
    see :func:`_apply_weaver_rescale`. A ``fix_init`` entry in
    ``weaver_kwargs`` is ignored for variant arms.

    Parameters
    ----------
    variant : str
        One of :data:`VARIANTS`.  ``"lloca"`` and ``"n8"`` return wrappers
        (:class:`~variants.lloca.part.LLoCaParT`,
        :class:`~variants.n8.part.N8ParT`) rather than a bare
        ``ParticleTransformer``; every arm is still called the same way,
        ``model(x, v=v, mask=mask)``.
    input_dim : int
        Number of per-particle input features (weaver ``input_dim``).
    num_classes : int
        Number of output classes.
    embed_dims, pair_embed_dims, num_heads, num_layers, num_cls_layers
        Forwarded to weaver ``ParticleTransformer``.
    dropout, expansion_factor
        Variant-block hyperparameters (unused for ``"baseline"``, whose
        stock blocks take their configuration from weaver defaults /
        ``block_params``).
    moe_config : str or Mapping or None
        Only used by the ``"moe"`` arm.  A key of :data:`MOE_PRESETS`, an
        explicit dict of
        :class:`~variants.blocks.moe_feedforward.MoEFeedforward` keyword arguments, or
        ``None`` for :data:`DEFAULT_MOE_PRESET` (FLOP-matched to the dense
        baseline).
    **weaver_kwargs
        Any additional weaver ``ParticleTransformer`` keyword arguments.

    Raises
    ------
    ValueError
        If ``variant`` is not a known arm.
    """
    if variant not in VARIANTS:
        raise ValueError(
            f"unknown variant {variant!r}; expected one of {sorted(VARIANTS)}"
        )

    # The residual-branch scale (eps = lambda / (N * sqrt(L))) is realised through weaver's
    # `LayerScale` (`Block.ls1` / `Block.ls2`), which only exists on STOCK weaver blocks. The swap
    # arms replace those blocks with custom ones that have no such hook, so for them the field
    # would be accepted, serialised into the checkpoint and silently ignored -- the exact `ffn2x`
    # failure (a config key with no effect on the built model, found in the 2026-09-09 audit,
    # A1). Refuse at construction rather than let a "control" arm train without its control.
    if (layer_scale_init_values is not None or residual_scale_lambda is not None) and (
        variant not in _LAYER_SCALE_ARMS
    ):
        raise ValueError(
            f"residual_scale_lambda / layer_scale_init_values cannot be realised on arm "
            f"{variant!r}: its blocks are not stock weaver Blocks and have no LayerScale "
            f"(ls1/ls2). Supported arms: {sorted(_LAYER_SCALE_ARMS)}. Remove the field from this "
            "config, or wire LayerScale into the custom block before comparing it at a fixed eps."
        )

    common = dict(
        input_dim=input_dim,
        num_classes=num_classes,
        embed_dims=embed_dims,
        pair_embed_dims=pair_embed_dims,
        num_heads=num_heads,
        num_layers=num_layers,
        num_cls_layers=num_cls_layers,
    )

    if variant == "baseline":
        # Stock weaver, default fix_init (unless the caller overrides it).
        #
        # `common` deliberately omits expansion_factor/dropout because weaver's
        # ParticleTransformer does not accept them as kwargs at all: its ``Block`` reads
        # ``ffn_ratio`` and ``dropout`` out of ``block_params`` / ``cls_block_params``
        # (``cfg_block.update(block_params)``), both defaulting to None.
        #
        # Before 2026-09-04 they were simply dropped on this path, so a config carrying
        # ``expansion_factor: 8`` was accepted, serialised into the checkpoint, and silently
        # ignored. ``baseline_ffn2x`` therefore trained an FFN of width 512 with a stored
        # provenance of 1024 and a parameter count byte-identical to plain baseline -- its
        # registered hypothesis was never tested. See logs/cluster-audit-2026-08-30.md
        # section 4.1. An explicit caller-supplied block_params still wins.
        # `layer_scale_init_values` installs weaver's LayerScale on each residual BRANCH
        # (ls1 after attention, ls2 after the FFN). It is the looped-transformer eps law and it
        # is applied to EVERY arm from one formula in AblationConfig.residual_branch_scale(),
        # which is what makes it a control rather than a favour to the tied arm.
        block_overrides = {"ffn_ratio": expansion_factor, "dropout": dropout}
        if layer_scale_init_values is not None:
            block_overrides["layer_scale_init_values"] = float(layer_scale_init_values)
        for key in ("block_params", "cls_block_params"):
            merged = dict(block_overrides)
            merged.update(weaver_kwargs.get(key) or {})
            weaver_kwargs[key] = merged
        model = ParticleTransformer(**common, **weaver_kwargs)
        _assert_realized_ffn_width(model, embed_dims, expansion_factor, variant)
        _assert_realized_layer_scale(model, layer_scale_init_values, variant)
        return model

    if variant == "tied":
        # Weight tying is NOT a block swap: the blocks stay stock weaver Blocks and only their
        # *arrangement* changes, so this reuses the baseline construction and then rebuilds
        # `model.blocks` as a schedule over fewer unique blocks. weaver iterates `self.blocks`
        # itself, so handing it a ModuleList of the right length is enough and no weaver source
        # is touched (DD6).
        from .tied import TiedSchedule, build_tied_blocks

        # `layer_scale_init_values` installs weaver's LayerScale on each residual BRANCH
        # (ls1 after attention, ls2 after the FFN). It is the looped-transformer eps law and it
        # is applied to EVERY arm from one formula in AblationConfig.residual_branch_scale(),
        # which is what makes it a control rather than a favour to the tied arm.
        block_overrides = {"ffn_ratio": expansion_factor, "dropout": dropout}
        if layer_scale_init_values is not None:
            block_overrides["layer_scale_init_values"] = float(layer_scale_init_values)
        for key in ("block_params", "cls_block_params"):
            merged = dict(block_overrides)
            merged.update(weaver_kwargs.get(key) or {})
            weaver_kwargs[key] = merged
        model = ParticleTransformer(**common, **weaver_kwargs)
        _assert_realized_ffn_width(model, embed_dims, expansion_factor, variant)

        schedule = TiedSchedule(
            depth=len(model.blocks),
            num_unique=int(tie_num_unique if tie_num_unique is not None else 1),
            strategy=tie_strategy,
        )
        embed_dim = embed_dims[-1] if len(embed_dims) else input_dim
        model.blocks = build_tied_blocks(
            list(model.blocks),
            schedule,
            embed_dim=embed_dim,
            depth_modulation=tie_depth_modulation,
            mor_capacity=tie_mor_capacity,
            scope=tie_scope,
        )
        # Recorded so a checkpoint carries how it was tied. The ffn2x episode -- a config key
        # accepted, stored and silently ignored -- is the reason provenance is attached to the
        # object rather than left to the YAML.
        model.tie_schedule = schedule
        model.tie_scope = tie_scope
        if layer_scale_init_values is not None:
            # Rewrite the LayerScale gammas with each block's OWN loop count and only on the
            # branch that is actually shared. The single value passed through block_params applies
            # one averaged N to both branches, which is wrong for non-uniform schedules
            # (middle-cycle k=3 runs block 1 six times and blocks 0/2 once) and wrong for sub-block
            # scopes (scope="attn" leaves eight distinct FFNs that are not looped at all).
            from .tied import apply_residual_scale

            model.residual_scale = apply_residual_scale(
                model, schedule, residual_scale_lambda, tie_scope
            )
        _assert_realized_layer_scale(model, layer_scale_init_values, variant)
        return model

    if variant == "lowrank":
        # Also not a block swap. Weaver builds the pair bias in ONE place --
        # `attn_mask = self.pair_embed(v, uu=uu, mask=mask)` -- and hands the identical tensor to
        # every block, so replacing `model.pair_embed` with a module of the same signature and
        # output shape is the entire integration. No weaver source is touched (DD6), and this is
        # the same seam `ablation/bias_truncation_probe.py` already validated by truncating through
        # it (its `full` control reproduced the unmodified accuracy exactly).
        from .lowrank import LowRankPairEmbed

        # `layer_scale_init_values` installs weaver's LayerScale on each residual BRANCH
        # (ls1 after attention, ls2 after the FFN). It is the looped-transformer eps law and it
        # is applied to EVERY arm from one formula in AblationConfig.residual_branch_scale(),
        # which is what makes it a control rather than a favour to the tied arm.
        block_overrides = {"ffn_ratio": expansion_factor, "dropout": dropout}
        if layer_scale_init_values is not None:
            block_overrides["layer_scale_init_values"] = float(layer_scale_init_values)
        for key in ("block_params", "cls_block_params"):
            merged = dict(block_overrides)
            merged.update(weaver_kwargs.get(key) or {})
            weaver_kwargs[key] = merged
        model = ParticleTransformer(**common, **weaver_kwargs)
        _assert_realized_ffn_width(model, embed_dims, expansion_factor, variant)
        _assert_realized_layer_scale(model, layer_scale_init_values, variant)

        if getattr(model, "pair_embed", None) is None:
            raise ValueError(
                "the `lowrank` arm replaces the pair bias, so the base model must have one; "
                "pair_embed_dims resolved to None"
            )
        model.pair_embed = LowRankPairEmbed(
            rank=int(lowrank_rank),
            num_heads=num_heads,
            hidden=tuple(lowrank_hidden) if lowrank_hidden else (64, 64),
            self_pair_scalar=bool(lowrank_self_pair),
        )
        # Provenance on the object, not the YAML. `ffn2x` was a config key accepted, stored and
        # silently ignored, so what the model actually IS gets recorded where it cannot drift.
        model.lowrank_config = {
            "rank": int(lowrank_rank),
            "hidden": list(lowrank_hidden) if lowrank_hidden else [64, 64],
            "self_pair_scalar": bool(lowrank_self_pair),
        }
        return model

    if variant == "lloca":
        # Not a block swap: LLoCa must canonicalize the inputs before they are
        # embedded, so it wraps a transformer instead of replacing its blocks.
        return LLoCaParT(
            **common,
            dropout=dropout,
            expansion_factor=expansion_factor,
            **weaver_kwargs,
        )

    if variant == "n8":
        n8_kwargs = dict(
            minkowski=weaver_kwargs.pop("minkowski", True),
            rotary=weaver_kwargs.pop("rotary", True),
            rotary_pairs=weaver_kwargs.pop("rotary_pairs", 4),
            degree=weaver_kwargs.pop("degree", 1),
        )
        # PairEmbed is the thing N8 replaces; keep it off even if a shared
        # config still lists pair_embed_dims.
        common["pair_embed_dims"] = None
        return N8ParT(
            **common,
            dropout=dropout,
            expansion_factor=expansion_factor,
            **n8_kwargs,
            **weaver_kwargs,
        )

    # Variant arms: weaver's fix_init_weight cannot run on adapter-wrapped
    # blocks, so it is forced off and replicated manually below.
    weaver_kwargs.pop("fix_init", None)
    per_plane = bool(weaver_kwargs.pop("per_plane", False))
    rope_apply = bool(weaver_kwargs.pop("rope_apply", False))
    model = ParticleTransformer(**common, fix_init=False, **weaver_kwargs)

    embed_dim = embed_dims[-1] if len(embed_dims) > 0 else input_dim
    resolved_moe = _resolve_moe_config(moe_config) if variant == "moe" else None
    for i in range(len(model.blocks)):
        model.blocks[i] = WeaverBlockAdapter(
            _make_variant_block(
                variant,
                embed_dim=embed_dim,
                num_heads=num_heads,
                dropout=dropout,
                expansion_factor=expansion_factor,
                moe_config=resolved_moe,
                per_plane=per_plane,
                rope_apply=rope_apply,
            )
        )
    _apply_weaver_rescale(model.blocks)
    return model


def collect_moe_aux_loss(model: nn.Module) -> Tensor:
    """Sum the MoE load-balancing auxiliary loss over a model.

    Iterates ``model.modules()`` and sums ``aux_loss`` over every
    :class:`~variants.blocks.moe_feedforward.MoEFeedforward` instance. Each
    ``aux_loss`` is set during the module's forward pass and is
    graph-connected through the router probabilities, so the returned tensor
    can be added to the training loss
    (``loss = task_loss + moe_aux_alpha * collect_moe_aux_loss(model)``).

    Returns a scalar ``tensor(0.0)`` when the model contains no
    ``MoEFeedforward`` modules (e.g. the baseline arm).
    """
    total: Optional[Tensor] = None
    for module in model.modules():
        if isinstance(module, MoEFeedforward):
            total = module.aux_loss if total is None else total + module.aux_loss
    if total is None:
        return torch.tensor(0.0)
    return total
