"""Tests for the MoE arm's compute/capacity accounting and router health.

Feature: ParT ablation — MoE arm.

An MoE arm is only a meaningful ablation point if you know what it is being held
equal to. The ``hidden_mode`` presets claim specific ratios against the dense
``Feedforward`` baseline, and those claims are easy to get subtly wrong (an
off-by-one in the expert width silently halves the arm's compute). These tests
check the claims against **measured** FLOPs and parameter counts rather than
re-deriving the same arithmetic the implementation uses.

They also pin two failure modes that do not show up as a crash:

* a ``top_k=1`` router gated by a softmax over the single selected logit gets
  **zero gradient** and never learns to route, so the arm degenerates into a
  fixed random partition of tokens;
* load-balancing statistics computed over padded positions are dominated by
  padding on ragged jets, pushing the router to balance noise.
"""

from __future__ import annotations

import pytest
import torch
from torch.utils.flop_counter import FlopCounterMode

from variants import MOE_PRESETS, build_variant_part
from variants.blocks.feedforward import Feedforward
from variants.tests.strategies import (
    _make_four_vectors,
    _make_lengths_masks,
    assert_loader_contract,
)
from variants.blocks.moe_feedforward import MoEFeedforward

EMBED_DIM = 128
EXPANSION = 4
BATCH, TOKENS = 8, 50

PRESET_NAMES = sorted(MOE_PRESETS)


def _measure_flops(module: torch.nn.Module, *args) -> int:
    counter = FlopCounterMode(display=False)
    with counter:
        module(*args)
    return counter.get_total_flops()


@pytest.fixture(scope="module")
def dense_cost():
    """Measured cost of the dense baseline FFN the MoE presets are matched to."""
    torch.manual_seed(0)
    dense = Feedforward(EMBED_DIM, EXPANSION, 0.0).eval()
    x = torch.randn(BATCH, TOKENS, EMBED_DIM)
    return _measure_flops(dense, x), sum(p.numel() for p in dense.parameters())


@pytest.mark.parametrize("preset", PRESET_NAMES)
def test_preset_flops_match_claim(preset, dense_cost) -> None:
    """Measured active FLOPs must match ``flop_ratio``.

    Only an upper bound is asserted: with a finite token batch a routed expert
    can end up with no tokens at all, which makes the measurement *cheaper*
    than the nominal ratio. Overshooting, by contrast, always means the experts
    are wider than advertised.
    """
    dense_flops, _ = dense_cost
    torch.manual_seed(0)
    moe = MoEFeedforward(EMBED_DIM, EXPANSION, 0.0, **MOE_PRESETS[preset]).eval()

    x = torch.randn(BATCH, TOKENS, EMBED_DIM)
    measured = _measure_flops(moe, x) / dense_flops

    # The router itself is a small extra matmul, hence the tolerance.
    assert measured <= moe.flop_ratio + 0.02, (
        f"{preset}: measured {measured:.3f}x > claimed {moe.flop_ratio:.3f}x"
    )


@pytest.mark.parametrize("preset", PRESET_NAMES)
def test_preset_params_match_claim(preset, dense_cost) -> None:
    _, dense_params = dense_cost
    torch.manual_seed(0)
    moe = MoEFeedforward(EMBED_DIM, EXPANSION, 0.0, **MOE_PRESETS[preset])

    measured = sum(p.numel() for p in moe.parameters()) / dense_params
    # Router + per-expert LayerNorms sit outside the hidden-width ratio.
    assert measured == pytest.approx(moe.param_ratio, abs=0.03), (
        f"{preset}: measured {measured:.3f}x vs claimed {moe.param_ratio:.3f}x"
    )


def test_flop_matched_presets_are_actually_flop_matched() -> None:
    """Every preset advertising a FLOP match must claim exactly ``1.0x``."""
    for preset in PRESET_NAMES:
        moe = MoEFeedforward(EMBED_DIM, EXPANSION, **MOE_PRESETS[preset])
        if preset.startswith(("flop_matched", "shared_flop_matched")):
            assert moe.flop_ratio == pytest.approx(1.0), preset
        if preset == "param_matched":
            assert moe.param_ratio == pytest.approx(1.0), preset


def test_e8_top2_subflop_activates_two_experts_under_baseline() -> None:
    """More experts, Mixtral-style top-2, activated FFN strictly below dense."""
    moe = MoEFeedforward(EMBED_DIM, EXPANSION, **MOE_PRESETS["e8_top2_subflop"])
    assert moe.num_experts == 8
    assert moe.top_k == 2
    assert moe.expert_hidden == 192
    assert moe.flop_ratio == pytest.approx(0.75)
    assert moe.param_ratio == pytest.approx(3.0)
    assert moe.flop_ratio < 1.0


@pytest.mark.parametrize("preset", PRESET_NAMES)
def test_router_receives_gradient(preset) -> None:
    """The router must be trainable under every preset's gating scheme."""
    torch.manual_seed(0)
    moe = MoEFeedforward(EMBED_DIM, EXPANSION, 0.0, **MOE_PRESETS[preset])

    out = moe(torch.randn(BATCH, TOKENS, EMBED_DIM))
    (out.square().mean() + 0.01 * moe.aux_loss).backward()

    assert moe.router.weight.grad is not None
    assert moe.router.weight.grad.norm() > 0, (
        f"{preset}: router got no gradient — it cannot learn to route"
    )


def test_top1_with_topk_softmax_is_rejected() -> None:
    """Softmax over one selected logit is identically 1: reject, don't train blind."""
    with pytest.raises(ValueError, match="without gradient"):
        MoEFeedforward(EMBED_DIM, EXPANSION, top_k=1, gate_mode="topk_softmax")


def test_top_k_above_num_experts_is_rejected() -> None:
    with pytest.raises(ValueError, match="must be <= num_experts"):
        MoEFeedforward(EMBED_DIM, EXPANSION, num_experts=2, top_k=3)


def test_degenerate_expert_width_is_rejected() -> None:
    """A config whose experts round down to zero width must fail at build time."""
    with pytest.raises(ValueError, match="expert_hidden"):
        MoEFeedforward(
            embed_dim=8, expansion_factor=1, num_experts=64, top_k=2,
            hidden_mode="param_matched",
        )


def test_padding_mask_changes_aux_loss() -> None:
    """Padded tokens must be excluded from the load-balancing statistics."""
    torch.manual_seed(0)
    moe = MoEFeedforward(
        EMBED_DIM, EXPANSION, 0.0, **MOE_PRESETS["flop_matched_top2"]
    ).eval()
    x = torch.randn(BATCH, TOKENS, EMBED_DIM)

    padding_mask = torch.zeros(BATCH, TOKENS, dtype=torch.bool)
    padding_mask[:, TOKENS // 2 :] = True

    moe(x)
    unmasked = moe.aux_loss.clone()
    moe(x, padding_mask)
    masked = moe.aux_loss.clone()

    assert not torch.allclose(unmasked, masked)


def test_aux_loss_is_finite_when_every_token_is_padding() -> None:
    """A fully-padded batch must not produce NaN in the aux loss."""
    torch.manual_seed(0)
    moe = MoEFeedforward(EMBED_DIM, EXPANSION, 0.0, **MOE_PRESETS["param_matched"])
    x = torch.randn(2, 4, EMBED_DIM)

    moe(x, torch.ones(2, 4, dtype=torch.bool))
    assert torch.isfinite(moe.aux_loss)


def test_moe_arm_forwards_padding_mask_to_the_router() -> None:
    """The host block must hand the mask down, or the exclusion never happens."""
    torch.manual_seed(0)
    model = build_variant_part("moe", input_dim=16, num_classes=10, num_layers=1)
    ffn = model.blocks[0].block.feedforward
    assert isinstance(ffn, MoEFeedforward)
    assert model.blocks[0].block._ffn_accepts_mask

    seen = {}
    original = ffn.forward

    def spy(x, padding_mask=None):
        seen["padding_mask"] = padding_mask
        return original(x, padding_mask)

    ffn.forward = spy

    # Build the batch with the shared loader-contract builders rather than by
    # hand. The previous inline `v = torch.rand(2, 4, 8) + 1.0` was neither
    # zero-padded nor physically valid (every component in [1, 2], so |p| could
    # exceed E, giving a negative m^2 and a NaN ln(m^2) in the pair features).
    # It only survived because this test asserts nothing about the output.
    lengths = [3, 7]
    generator = torch.Generator().manual_seed(0)
    mask, _ = _make_lengths_masks(lengths, 8)
    v = _make_four_vectors(lengths, 8, generator)
    x = torch.randn(2, 16, 8, generator=generator) * mask
    assert_loader_contract(x, v, mask, lengths=lengths)

    model(x, v=v, mask=mask)

    assert seen["padding_mask"] is not None, "block did not pass the padding mask"
    assert seen["padding_mask"].dtype == torch.bool


def test_unknown_preset_is_rejected() -> None:
    with pytest.raises(ValueError, match="unknown MoE preset"):
        build_variant_part(
            "moe", input_dim=16, num_classes=10, num_layers=1,
            moe_config="not_a_preset",
        )


def test_explicit_moe_config_dict_is_honoured() -> None:
    model = build_variant_part(
        "moe", input_dim=16, num_classes=10, num_layers=1,
        moe_config={"num_experts": 6, "top_k": 3, "hidden_mode": "flop_matched"},
    )
    ffn = model.blocks[0].block.feedforward
    assert (ffn.num_experts, ffn.top_k) == (6, 3)
    assert ffn.expert_hidden == (EMBED_DIM * EXPANSION) // 3
