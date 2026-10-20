"""Property-based tests for the variant blocks (design "Correctness Properties").

Feature: self-contained-kernels-migration.

This module holds the block-level correctness properties:

- Property 1 (below): variant blocks are padding-invariant.
- Property 2 (below): variant blocks produce finite gradients on padded
  batches.

The four ablation variant arms under test (design Property 1):

- ``sparsemax``: ``SparsemaxAttentionBlock``
- ``moe``: ``SoftmaxAttentionBlock`` hosting ``MoEFeedforward`` (DD5)
- ``diff_v1``: ``DifferentialAttentionBlock``
- ``diff_v2``: ``DifferentialAttentionV2Block``

All arms share the invariant interface ``forward(x, padding_mask, U)`` with
``x (B, N, C)`` batch-first, ``padding_mask (B, N)`` bool (True = padded),
and ``U`` an optional ``(B, H, N, N)`` additive attention bias.

Inputs are drawn from the shared strategies in
``variants.tests.strategies`` (task 3.7).
"""

from __future__ import annotations

import pytest
import torch
from hypothesis import given, settings
from hypothesis import strategies as st

from variants import (
    DifferentialAttentionBlock,
    DifferentialAttentionV2Block,
    MoEFeedforward,
    SoftmaxAttentionBlock,
    SparsemaxAttentionBlock,
    URotaryAttentionBlock,
)
from variants.tests.strategies import BlockBatch, block_inputs

# ---------------------------------------------------------------------------
# Variant arm builders (shared by Properties 1 and 2)
# ---------------------------------------------------------------------------

#: Common constructor kwargs. dropout=0.1 exercises the real configuration;
#: Property 1 runs the blocks in eval mode so dropout is inactive.
_DROPOUT = 0.1
_EXPANSION_FACTOR = 4


def _build_sparsemax(embed_dim: int, num_heads: int) -> torch.nn.Module:
    return SparsemaxAttentionBlock(
        embed_dim=embed_dim,
        num_heads=num_heads,
        dropout=_DROPOUT,
        expansion_factor=_EXPANSION_FACTOR,
    )


def _build_moe(embed_dim: int, num_heads: int) -> torch.nn.Module:
    # The MoE ablation arm: SoftmaxAttentionBlock hosting MoEFeedforward (DD5).
    return SoftmaxAttentionBlock(
        embed_dim=embed_dim,
        num_heads=num_heads,
        dropout=_DROPOUT,
        expansion_factor=_EXPANSION_FACTOR,
        ffn=MoEFeedforward(
            embed_dim=embed_dim,
            expansion_factor=_EXPANSION_FACTOR,
            dropout=_DROPOUT,
            num_experts=4,
            top_k=2,
        ),
    )


def _build_diff_v1(embed_dim: int, num_heads: int) -> torch.nn.Module:
    return DifferentialAttentionBlock(
        embed_dim=embed_dim,
        num_heads=num_heads,
        dropout=_DROPOUT,
        expansion_factor=_EXPANSION_FACTOR,
    )


def _build_diff_v2(embed_dim: int, num_heads: int) -> torch.nn.Module:
    return DifferentialAttentionV2Block(
        embed_dim=embed_dim,
        num_heads=num_heads,
        dropout=_DROPOUT,
        expansion_factor=_EXPANSION_FACTOR,
    )


def _build_urot(embed_dim: int, num_heads: int) -> torch.nn.Module:
    return URotaryAttentionBlock(
        embed_dim=embed_dim,
        num_heads=num_heads,
        dropout=_DROPOUT,
        expansion_factor=_EXPANSION_FACTOR,
    )


def _build_urot_rope(embed_dim: int, num_heads: int) -> torch.nn.Module:
    return URotaryAttentionBlock(
        embed_dim=embed_dim,
        num_heads=num_heads,
        dropout=_DROPOUT,
        expansion_factor=_EXPANSION_FACTOR,
        rope_apply=True,
    )


#: variant name -> builder(embed_dim, num_heads). Module-level so Property 2
#: (task 3.9) parameterizes over the identical four arms.
BLOCK_BUILDERS = {
    "sparsemax": _build_sparsemax,
    "moe": _build_moe,
    "diff_v1": _build_diff_v1,
    "diff_v2": _build_diff_v2,
    "urot": _build_urot,
    "urot_rope": _build_urot_rope,
}

VARIANT_NAMES = sorted(BLOCK_BUILDERS)


def build_block(variant: str, embed_dim: int, num_heads: int, seed: int) -> torch.nn.Module:
    """Deterministically construct one variant block (params seeded)."""
    torch.manual_seed(seed)
    return BLOCK_BUILDERS[variant](embed_dim, num_heads)


# ---------------------------------------------------------------------------
# Perturbation helpers
# ---------------------------------------------------------------------------

#: Std-dev of the perturbation noise — large relative to the standard-normal
#: inputs, so any leakage from padded positions is far above float tolerance.
_NOISE_SCALE = 4.0


def _perturb_padded(batch: BlockBatch, seed: int):
    """Return (x_pert, U_pert) with ONLY padded positions perturbed.

    - ``x``: rows at padded positions get additive random noise.
    - ``U``: entries whose query row i OR key column j is padded get
      additive random noise; entries with both endpoints valid are untouched.
    """
    gen = torch.Generator().manual_seed(seed)
    padded = batch.padding_mask  # (B, N) bool, True = padded

    x_noise = torch.randn(batch.x.shape, generator=gen) * _NOISE_SCALE
    x_pert = torch.where(padded.unsqueeze(-1), batch.x + x_noise, batch.x)

    U_pert = batch.U
    if batch.U is not None:
        # (B, N, N): True where query i or key j is padded.
        pad_pair = padded.unsqueeze(-1) | padded.unsqueeze(-2)
        u_noise = torch.randn(batch.U.shape, generator=gen) * _NOISE_SCALE
        U_pert = torch.where(pad_pair.unsqueeze(1), batch.U + u_noise, batch.U)

    return x_pert, U_pert


# ---------------------------------------------------------------------------
# Property 1
# ---------------------------------------------------------------------------

# Feature: self-contained-kernels-migration, Property 1: Variant blocks are padding-invariant
@pytest.mark.parametrize("variant", VARIANT_NAMES)
@settings(max_examples=100, deadline=None)
@given(
    batch=block_inputs(min_padded_per_jet=1),
    seed=st.integers(min_value=0, max_value=2**32 - 1),
)
def test_variant_blocks_are_padding_invariant(variant, batch, seed):
    """Perturbing padded positions of ``x`` and ``U`` must not change the
    block's outputs at valid positions (eval mode), and the output shape
    must equal the input shape ``(B, N, C)``.

    **Validates: Requirements 1.5**
    """
    block = build_block(variant, batch.embed_dim, batch.num_heads, seed)
    block.eval()

    B, N, C = batch.x.shape

    with torch.no_grad():
        out_base = block(batch.x, batch.padding_mask, batch.U)

    x_pert, U_pert = _perturb_padded(batch, seed)
    with torch.no_grad():
        out_pert = block(x_pert, batch.padding_mask, U_pert)

    # Output shape equals input shape (B, N, C) for both runs.
    assert out_base.shape == (B, N, C)
    assert out_pert.shape == (B, N, C)

    # Valid-position outputs unchanged within float32 tolerance. The mask
    # neutralizes padded keys exactly, so differences beyond tiny BLAS
    # reduction-order effects (MoE gather batching) indicate padding leakage.
    valid = ~batch.padding_mask  # (B, N) bool, True = valid
    base_valid = out_base[valid]
    pert_valid = out_pert[valid]
    max_abs = (base_valid - pert_valid).abs().max().item() if valid.any() else 0.0
    assert torch.allclose(base_valid, pert_valid, atol=1e-6, rtol=1e-5), (
        f"padding perturbation leaked into valid outputs for variant "
        f"'{variant}': max |delta| = {max_abs:.3e} "
        f"(B={B}, N={N}, C={C}, lengths={batch.lengths}, "
        f"U={'present' if batch.U is not None else 'None'})"
    )


# ---------------------------------------------------------------------------
# Property 2
# ---------------------------------------------------------------------------

def _iter_named_params(block: torch.nn.Module):
    return [(name, p) for name, p in block.named_parameters() if p.requires_grad]


def _moe_expert_prefix(name: str) -> str | None:
    """Return ``"feedforward.experts.<i>"`` if ``name`` is an expert param."""
    parts = name.split(".")
    if len(parts) >= 3 and parts[0] == "feedforward" and parts[1] == "experts":
        return ".".join(parts[:3])
    return None


# Feature: self-contained-kernels-migration, Property 2: Variant blocks produce finite gradients on padded batches
@pytest.mark.parametrize("variant", VARIANT_NAMES)
@settings(max_examples=100, deadline=None)
@given(
    batch=block_inputs(),  # default full [1, N] length range: includes jets
    #                        padded down to a single valid particle
    seed=st.integers(min_value=0, max_value=2**32 - 1),
)
def test_variant_blocks_produce_finite_gradients_on_padded_batches(variant, batch, seed):
    """Train-mode backward on padded batches yields finite gradients.

    The loss is computed from VALID-position outputs only
    (``out[~padding_mask].sum()``): padded outputs are not part of the
    training signal — weaver zeroes/ignores them — so this matches the
    design's Property 2 statement ("a loss computed from the valid-position
    outputs"). Dropout is active in train mode; that only zeroes some
    gradient paths and never affects finiteness.

    Gradient-coverage decision (documented per task 3.9):

    - Non-MoE arms: every forward touches every parameter, so EVERY
      trainable parameter must receive a finite, non-None gradient — the
      design's strict "every trainable parameter" reading applies as-is.
    - MoE arm: the loss additionally includes the router's auxiliary
      load-balancing loss (``block.feedforward.aux_loss``), mirroring the
      module's documented training usage, so the router receives gradients
      through both the top-k softmax weights and the aux loss. Top-k
      routing, however, structurally excludes never-selected experts from
      the autograd graph (their sub-modules are skipped entirely), so those
      expert parameters legitimately have ``grad is None`` — this is
      inherent to sparse MoE routing, not a padding defect, and the aux
      loss touches only the router, never unselected experts. The check is
      therefore: all NON-expert parameters (router, attention projections,
      layernorms) must receive gradients; expert parameters are covered
      per-expert (an expert either participated — all of its params got
      gradients — or was never selected — none did); and every gradient
      that exists must be finite.

    **Validates: Requirements 1.5, 1.6**
    """
    # build_block seeds the global RNG; the train-mode dropout draws that
    # follow are therefore deterministic per (variant, batch, seed) example.
    block = build_block(variant, batch.embed_dim, batch.num_heads, seed)
    block.train()

    out = block(batch.x, batch.padding_mask, batch.U)

    valid = ~batch.padding_mask  # (B, N) bool, True = valid
    loss = out[valid].sum()
    if variant == "moe":
        # Router training signal per MoEFeedforward's documented usage.
        loss = loss + block.feedforward.aux_loss
    assert torch.isfinite(loss), (
        f"non-finite loss for variant '{variant}' "
        f"(lengths={batch.lengths}, U={'present' if batch.U is not None else 'None'})"
    )

    loss.backward()

    missing: list[str] = []
    non_finite: list[str] = []
    expert_grad_state: dict[str, dict[str, bool]] = {}

    for name, param in _iter_named_params(block):
        expert = _moe_expert_prefix(name) if variant == "moe" else None
        if param.grad is None:
            if expert is not None:
                expert_grad_state.setdefault(expert, {})[name] = False
            else:
                # α only enters θ = π tanh(α U).  With U=None the content
                # path is used and angle_scale is unused — not a padding leak.
                if (
                    variant in ("urot", "urot_rope")
                    and name == "angle_scale"
                    and batch.U is None
                ):
                    continue
                missing.append(name)
            continue
        if expert is not None:
            expert_grad_state.setdefault(expert, {})[name] = True
        if not torch.isfinite(param.grad).all():
            non_finite.append(name)

    ctx = (
        f"variant '{variant}' (B={batch.x.shape[0]}, N={batch.x.shape[1]}, "
        f"C={batch.embed_dim}, lengths={batch.lengths}, "
        f"U={'present' if batch.U is not None else 'None'})"
    )

    # Every gradient that exists is finite (Req 1.6).
    assert not non_finite, f"non-finite gradients for {ctx}: {non_finite}"

    # Core parameters (attention projections, layernorms, FFN/router, lambdas)
    # all received gradients (Req 1.5/1.6).
    assert not missing, f"parameters received no gradient for {ctx}: {missing}"

    # MoE experts: gradient coverage is all-or-none per expert — a selected
    # expert's params all got gradients; a never-selected expert's got none.
    for expert, states in expert_grad_state.items():
        got = [n for n, has in states.items() if has]
        without = [n for n, has in states.items() if not has]
        assert not (got and without), (
            f"inconsistent gradient coverage inside {expert} for {ctx}: "
            f"with grad {got}, without grad {without}"
        )
