"""Properties of the weight-tied (looped) encoder stack — arms R1-R4.

The tests that matter here are the ones that would catch a *silent* failure, because that is the
failure mode this repository has actually suffered: `baseline_ffn2x` trained a model identical to
baseline while its config recorded otherwise, and a green test named "doubles expansion" asserted
only that the YAML contained the number 8. So every assertion below is on a **built model** — its
parameter count, its outputs, its schedule — never on a config field.

Three properties carry most of the weight:

* ``num_unique == depth`` must be **exactly** stock ParT, parameter-for-parameter and
  output-for-output. If the tied path perturbs anything at that setting, every tied-vs-untied
  comparison is confounded by the perturbation rather than by tying.
* tying must actually **share** parameters, not merely construct fewer and re-init them.
* ``block_applications`` must stay equal to ``depth`` at every ``num_unique``, because that is the
  FLOP-relevant count and the comparison isolates parameter sharing only if it is held fixed
  (the standard SMELT, arXiv:2609.01343, sets).
"""

from __future__ import annotations

import math

import pytest
import torch

from ablation.config import AblationConfig
from variants import build_variant_part
from variants.tied import (
    DepthModulation,
    MoRRouter,
    TiedSchedule,
    build_tied_blocks,
    tied_cost_report,
)

_BASELINE_PARAMS = 2_143_354
_DEPTH = 8


def _kwargs() -> dict:
    return AblationConfig(arm="baseline").model_kwargs()


def _build(**tie) -> torch.nn.Module:
    torch.manual_seed(0)
    return build_variant_part("tied", **_kwargs(), **tie)


def _params(model) -> int:
    return sum(p.numel() for p in model.parameters())


def _batch(batch: int = 2, particles: int = 12, seed: int = 3):
    g = torch.Generator().manual_seed(seed)
    p3 = torch.randn(batch, 3, particles, generator=g) * 5.0
    energy = (p3 ** 2).sum(1, keepdim=True).sqrt() * 1.05  # timelike: E > |p|
    v = torch.cat([p3, energy], dim=1)
    x = torch.randn(batch, 16, particles, generator=g)
    mask = torch.ones(batch, 1, particles)
    return x, v, mask


# ---------------------------------------------------------------------------
# schedules
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "num_unique,strategy,expected",
    [
        (1, "cycle", (0,) * 8),
        (1, "sequence", (0,) * 8),
        (2, "cycle", (0, 1, 0, 1, 0, 1, 0, 1)),
        (2, "sequence", (0, 0, 0, 0, 1, 1, 1, 1)),
        (4, "cycle", (0, 1, 2, 3, 0, 1, 2, 3)),
        (4, "sequence", (0, 0, 1, 1, 2, 2, 3, 3)),
        (8, "cycle", tuple(range(8))),
        (8, "sequence", tuple(range(8))),
    ],
)
def test_schedule_indices(num_unique: int, strategy: str, expected: tuple) -> None:
    assert TiedSchedule(_DEPTH, num_unique, strategy).indices == expected


def test_schedule_uneven_split_spreads_remainder_early() -> None:
    """depth 8 over 3 unique blocks: the remainder goes to the earliest blocks, not the last.

    Pinned because the alternative (dumping the remainder on the final block) makes the deepest
    block do disproportionate work, which would confound depth with capacity.
    """
    assert TiedSchedule(8, 3, "sequence").indices == (0, 0, 0, 1, 1, 1, 2, 2)


@pytest.mark.parametrize("bad", [dict(depth=0, num_unique=1), dict(depth=8, num_unique=0),
                                dict(depth=8, num_unique=9)])
def test_schedule_rejects_impossible_shapes(bad: dict) -> None:
    with pytest.raises(ValueError):
        TiedSchedule(**bad)


def test_schedule_rejects_unknown_strategy() -> None:
    with pytest.raises(ValueError):
        TiedSchedule(8, 2, "spiral")


def test_is_stock_only_when_fully_untied() -> None:
    assert TiedSchedule(8, 8).is_stock
    assert not TiedSchedule(8, 7).is_stock
    assert not TiedSchedule(8, 1).is_stock


# ---------------------------------------------------------------------------
# the identity property
# ---------------------------------------------------------------------------

def test_fully_untied_is_exactly_stock_baseline() -> None:
    """num_unique == depth must reproduce stock ParT exactly, in params AND in outputs.

    This is the single most important test in the file: it is what makes a tied-vs-untied
    comparison a measurement of tying rather than of an incidental difference in the tied path.
    """
    torch.manual_seed(0)
    base = build_variant_part("baseline", **_kwargs())
    tied = _build(tie_num_unique=8)

    assert _params(tied) == _params(base) == _BASELINE_PARAMS

    x, v, mask = _batch()
    base.eval(); tied.eval()
    with torch.no_grad():
        torch.testing.assert_close(
            tied(x, v=v, mask=mask), base(x, v=v, mask=mask), rtol=0, atol=0
        )


# ---------------------------------------------------------------------------
# sharing is real
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "num_unique,expected_params",
    [(1, 745_538), (2, 945_226), (4, 1_344_602), (8, _BASELINE_PARAMS)],
)
def test_parameter_count_matches_closed_form(num_unique: int, expected_params: int) -> None:
    """199,688 params per block, so tying k of 8 saves 199,688 * (8 - k) exactly."""
    model = _build(tie_num_unique=num_unique)
    assert _params(model) == expected_params
    assert expected_params == _BASELINE_PARAMS - 199_688 * (8 - num_unique)


def test_tied_blocks_are_the_same_object_not_copies() -> None:
    """Sharing must be by identity: a gradient into depth 0 must reach depth 7's weights.

    Constructing k blocks and re-initialising them would give the same parameter count while
    being a completely different (and much weaker) model, so the count alone is not sufficient.
    """
    model = _build(tie_num_unique=1)
    assert len({id(b) for b in model.blocks}) == 1
    ids = {id(p) for p in model.blocks[0].parameters()}
    assert {id(p) for p in model.blocks[7].parameters()} == ids


def test_gradient_reaches_shared_weights_from_every_depth() -> None:
    model = _build(tie_num_unique=1)
    x, v, mask = _batch()
    model(x, v=v, mask=mask).sum().backward()
    grads = [p.grad for p in model.blocks[0].parameters() if p.requires_grad]
    assert grads and all(g is not None for g in grads)
    assert any(float(g.abs().sum()) > 0 for g in grads)


# ---------------------------------------------------------------------------
# FLOP matching
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("num_unique", [1, 2, 4, 8])
@pytest.mark.parametrize("strategy", ["cycle", "sequence"])
def test_depth_and_applications_are_invariant_to_tying(num_unique: int, strategy: str) -> None:
    """Tying reduces parameters and leaves FLOPs alone. It is not a speedup.

    Asserted because a tied-vs-untied comparison isolates parameter sharing only when the number
    of block applications is held fixed, and because it is the claim most likely to be
    accidentally overstated in a write-up.
    """
    model = _build(tie_num_unique=num_unique, tie_strategy=strategy)
    report = tied_cost_report(model, model.tie_schedule)
    assert len(model.blocks) == _DEPTH
    assert report["block_applications"] == _DEPTH
    assert report["unique_blocks_instantiated"] == num_unique
    assert report["unique_block_params"] == 199_688 * num_unique


@pytest.mark.parametrize("num_unique", [1, 2, 3, 4, 8])
@pytest.mark.parametrize("strategy", ["cycle", "sequence"])
def test_forward_shape_for_every_schedule(num_unique: int, strategy: str) -> None:
    model = _build(tie_num_unique=num_unique, tie_strategy=strategy).eval()
    x, v, mask = _batch()
    with torch.no_grad():
        out = model(x, v=v, mask=mask)
    assert out.shape == (x.shape[0], 10)
    assert torch.isfinite(out).all()


def test_cycle_and_sequence_differ_at_intermediate_k() -> None:
    """The two strategies must not be silently the same model, or one column is wasted compute."""
    a = _build(tie_num_unique=2, tie_strategy="cycle").eval()
    b = _build(tie_num_unique=2, tie_strategy="sequence").eval()
    x, v, mask = _batch()
    with torch.no_grad():
        assert not torch.allclose(a(x, v=v, mask=mask), b(x, v=v, mask=mask))


# ---------------------------------------------------------------------------
# R3: per-depth modulation
# ---------------------------------------------------------------------------

def test_depth_modulation_is_identity_at_init() -> None:
    """A capacity-recovery mechanism that perturbs the model at init would confound its own effect."""
    plain = _build(tie_num_unique=1).eval()
    modulated = _build(tie_num_unique=1, tie_depth_modulation=True).eval()
    x, v, mask = _batch()
    with torch.no_grad():
        torch.testing.assert_close(
            modulated(x, v=v, mask=mask), plain(x, v=v, mask=mask), rtol=0, atol=0
        )


def test_depth_modulation_costs_two_vectors_per_depth() -> None:
    plain = _params(_build(tie_num_unique=1))
    modulated = _params(_build(tie_num_unique=1, tie_depth_modulation=True))
    assert modulated - plain == 2 * 128 * _DEPTH == 2048


def test_depth_modulation_is_actually_per_depth() -> None:
    """Each depth must own its own affine, otherwise it is a global scale wearing a disguise."""
    mod = DepthModulation(embed_dim=4, depth=3)
    with torch.no_grad():
        mod.beta[1] = 1.0
    x = torch.zeros(2, 5, 4)
    with torch.no_grad():
        assert float(mod(x, 0).abs().sum()) == 0.0
        assert float(mod(x, 1).abs().sum()) == pytest.approx(2 * 5 * 4)
        assert float(mod(x, 2).abs().sum()) == 0.0


def test_modulation_and_mor_cannot_be_combined() -> None:
    """R3 and R4 are separate arms; combining them confounds two mechanisms in one number."""
    with pytest.raises(ValueError, match="separate arms"):
        build_tied_blocks(
            [torch.nn.Identity()], TiedSchedule(2, 1),
            embed_dim=8, depth_modulation=True, mor_capacity=0.5,
        )


# ---------------------------------------------------------------------------
# R4: MoR routing
# ---------------------------------------------------------------------------

def test_router_selects_fixed_capacity() -> None:
    """Fixed capacity is the point: deterministic FLOPs and worst-case latency."""
    router = MoRRouter(embed_dim=8, capacity=0.5)
    x = torch.randn(3, 10, 8)
    _, keep = router(x)
    assert keep.shape == (3, 10)
    assert keep.sum(dim=1).tolist() == [5, 5, 5]


def test_router_never_spends_capacity_on_padding() -> None:
    """Padded slots are not particles. Selecting them would silently shrink the real capacity."""
    router = MoRRouter(embed_dim=8, capacity=1.0)
    x = torch.randn(2, 10, 8)
    padding = torch.zeros(2, 10, dtype=torch.bool)
    padding[:, 6:] = True                     # 6 valid, 4 padded
    gate, keep = router(x, padding)
    assert not bool((keep & padding).any())
    assert keep.sum(dim=1).tolist() == [6, 6]
    assert float(gate[padding].abs().sum()) == 0.0


def test_router_capacity_bounds_are_enforced() -> None:
    for bad in (0.0, -0.1, 1.5):
        with pytest.raises(ValueError):
            MoRRouter(embed_dim=8, capacity=bad)


def test_mor_forward_is_finite_and_right_shape() -> None:
    model = _build(tie_num_unique=1, tie_mor_capacity=0.5).eval()
    x, v, mask = _batch()
    with torch.no_grad():
        out = model(x, v=v, mask=mask)
    assert out.shape == (x.shape[0], 10)
    assert torch.isfinite(out).all()


def test_mor_block_is_identity_when_gate_is_zero() -> None:
    """The residual form must make a fully-skipped token provably unchanged.

    Zero-initialised router weights give sigmoid(0) = 0.5, not 0, so this is asserted against an
    explicitly zeroed gate rather than assumed from the init.
    """
    from variants.tied.stack import MoRBlock

    class _AddOne(torch.nn.Module):
        def forward(self, x, x_cls=None, padding_mask=None, attn_mask=None):
            return x + 1.0

    block = MoRBlock(_AddOne(), MoRRouter(embed_dim=4, capacity=1.0))
    with torch.no_grad():
        block.router.score.bias.fill_(-40.0)   # sigmoid(-40) underflows to 0
    x = torch.randn(2, 6, 4)
    with torch.no_grad():
        torch.testing.assert_close(block(x), x, rtol=0, atol=0)


def test_mor_adds_only_a_router_worth_of_parameters() -> None:
    plain = _params(_build(tie_num_unique=1))
    mor = _params(_build(tie_num_unique=1, tie_mor_capacity=0.5))
    assert mor - plain == 128 + 1          # one Linear(embed_dim -> 1)


@pytest.mark.parametrize("capacity", [0.25, 0.5, 1.0])
def test_mor_routes_particles_within_each_jet_in_the_built_model(capacity: float) -> None:
    """Integrated axis check: the router must rank PARTICLES within a jet, not jets within a slot.

    The direct router tests above feed synthetic ``(B, N, C)`` tensors, so they cannot tell
    whether the layout weaver actually hands the block agrees with the router's assumption. This
    test builds the real tied+MoR ParT, runs a RAGGED batch through weaver's own encoder, and
    inspects what the router selected on every recursion step. Three things must hold at once:
    the selection tensor is ``(B, P)``; each jet gets exactly ``ceil(capacity * n_valid)``
    particles (never a fraction of the padded width); and no padded slot is ever chosen.
    A batch-first/sequence-first mix-up fails the first assertion outright, and a router that
    silently ranked across the batch would fail the second.
    """
    from variants.tied.stack import MoRBlock

    batch, particles = 3, 12
    model = _build(tie_num_unique=1, tie_mor_capacity=capacity).eval()
    x, v, mask = _batch(batch=batch, particles=particles)
    n_valid = [12, 7, 4]                              # ragged: distinct multiplicities per jet
    for jet, n in enumerate(n_valid):
        mask[jet, :, n:] = 0.0
        x[jet, :, n:] = 0.0
        v[jet, :, n:] = 0.0

    seen: list[torch.Tensor] = []
    routers = {id(m.router): m.router for m in model.blocks if isinstance(m, MoRBlock)}
    assert len(routers) == 1, "whole-block MoR tying shares ONE router across all depths"
    router = next(iter(routers.values()))
    original = router.forward

    def spy(x_in, padding_mask=None):
        gate, keep = original(x_in, padding_mask)
        seen.append(keep.detach().clone())
        return gate, keep

    router.forward = spy
    try:
        with torch.no_grad():
            out = model(x, v=v, mask=mask)
    finally:
        router.forward = original

    assert out.shape == (batch, 10) and torch.isfinite(out).all()
    assert len(seen) == _DEPTH, "the router must run once per recursion step"
    padded = ~mask.squeeze(1).bool()                   # (B, P), True = padded
    expected_k = [max(1, math.ceil(capacity * n - 1e-6)) for n in n_valid]
    for keep in seen:
        # Axis check: (B, P), not (P, B). With B=3 and P=12 the transpose cannot pass by luck.
        assert keep.shape == (batch, particles), keep.shape
        assert not bool((keep & padded).any()), "capacity spent on a padded slot"
        assert keep.sum(dim=1).tolist() == expected_k, (keep.sum(dim=1).tolist(), expected_k)


def test_tied_arm_is_registered() -> None:
    from variants import VARIANTS

    assert "tied" in VARIANTS


# ---------------------------------------------------------------------------
# YAML -> config -> BUILT MODEL
# ---------------------------------------------------------------------------
# Tested as a chain, deliberately. `ffn2x` failed because its config was parsed correctly and
# then dropped between config and model, and its test only checked the parsing half. Asserting
# the built model's parameter count against a closed form is what closes that gap.

_CONFIG_DIR = __import__("pathlib").Path(__file__).resolve().parents[2] / "ablation" / "configs"


@pytest.mark.parametrize(
    "name,num_unique,strategy,expected_params",
    [
        ("tied_k1", 1, "cycle", 745_538),
        ("tied_k2_cycle", 2, "cycle", 945_226),
        ("tied_k4_cycle", 4, "cycle", 1_344_602),
        ("tied_k4_sequence", 4, "sequence", 1_344_602),
    ],
)
def test_tied_configs_reach_the_built_model(
    name: str, num_unique: int, strategy: str, expected_params: int
) -> None:
    from ablation.config import load_config

    config = load_config(str(_CONFIG_DIR / f"{name}.yaml"))
    assert config.arm == "tied"
    assert config.run_name == name
    assert config.tie_num_unique == num_unique
    assert config.tie_strategy == strategy
    assert config.max_steps == 200_000
    assert config.total_steps == 1_000_000

    torch.manual_seed(0)
    model = build_variant_part(config.arm, **config.model_kwargs())
    report = tied_cost_report(model, model.tie_schedule)
    assert _params(model) == expected_params
    assert report["num_unique"] == num_unique
    assert report["strategy"] == strategy
    assert report["block_applications"] == _DEPTH


_BLOCK_LS_PARAMS = 2 * 128                                    # ls1 + ls2 gammas, one Block
_UNTIED_EPS = 1.0 / math.sqrt(_DEPTH)                          # lambda=1, N=1, L=8


def _gammas(block) -> tuple[float, float]:
    inner = getattr(block, "block", block)                     # unwrap MoR/Modulated wrappers
    return float(inner.ls1.gamma.detach()[0]), float(inner.ls2.gamma.detach()[0])


@pytest.mark.parametrize(
    "name,arm,expected_params,expected_encoder_gammas,num_layers",
    [
        # baseline + LayerScale on 8 encoder + 2 class blocks: 2,143,354 + 10 * 256.
        ("baseline_wave2", "baseline", _BASELINE_PARAMS + 10 * _BLOCK_LS_PARAMS,
         [(_UNTIED_EPS, _UNTIED_EPS)] * 8, 8),
        # k=1: one unique block applied 8x -> eps = 1/8 on both branches. 745,538 + 3 * 256.
        ("tied_k1_wave2", "tied", 745_538 + 3 * _BLOCK_LS_PARAMS,
         [(1 / 8, 1 / 8)] * 8, 8),
        # k=1 + MoR router (capacity 0.5): tied_k1_wave2 + one Linear(128 -> 1) = +129. The gammas
        # must be read THROUGH the MoRBlock wrapper and be identical to tied_k1_wave2's.
        ("tied_k1_mor_wave2", "tied", 745_538 + 3 * _BLOCK_LS_PARAMS + 129,
         [(1 / 8, 1 / 8)] * 8, 8),
        # tied_k1_wave2 to the full 1M-step schedule: byte-identical model, only run_name/max_steps.
        ("tied_k1_1M", "tied", 745_538 + 3 * _BLOCK_LS_PARAMS,
         [(1 / 8, 1 / 8)] * 8, 8),
        # tied_k1_mor_wave2 to the full 1M-step schedule: same model as tied_k1_mor_wave2 (router
        # included, +129), only run_name/max_steps. Its router-off twin is tied_k1_1M.
        ("tied_k1_mor_1M", "tied", 745_538 + 3 * _BLOCK_LS_PARAMS + 129,
         [(1 / 8, 1 / 8)] * 8, 8),
        # scope=attn: attention shared (eps 1/8), eight distinct FFNs keep the untied 1/sqrt(8).
        # Params: 8 blocks minus 7 duplicate attentions (66,048 each) + 10 * 256.
        ("tied_attn_wave2", "tied", _BASELINE_PARAMS - 7 * 66_048 + 10 * _BLOCK_LS_PARAMS,
         [(1 / 8, _UNTIED_EPS)] * 8, 8),
        # scope=ffn: mirror image. 8 blocks minus 7 duplicate FFNs (131,712 each).
        ("tied_ffn_wave2", "tied", _BASELINE_PARAMS - 7 * 131_712 + 10 * _BLOCK_LS_PARAMS,
         [(_UNTIED_EPS, 1 / 8)] * 8, 8),
        # k=2 at depth 16: two unique blocks, 8 applications each -> 1/(8 sqrt 2). 945,226 + 4*256.
        ("tied_k2d16_wave2", "tied", 945_226 + 4 * _BLOCK_LS_PARAMS,
         [(1 / (8 * math.sqrt(2)),) * 2] * 16, 16),
        # middle-cycle k=3: indices (0,1,1,1,1,1,1,2). Blocks 0/2 run once -> 1/sqrt(3); block 1
        # runs six times -> 1/(6 sqrt 3). One averaged eps (RUN-PLAN's 0.217) describes none of them.
        ("tied_mc3_wave2", "tied", _BASELINE_PARAMS - 5 * 199_688 + 5 * _BLOCK_LS_PARAMS,
         [(1 / math.sqrt(3),) * 2] + [(1 / (6 * math.sqrt(3)),) * 2] * 6 + [(1 / math.sqrt(3),) * 2],
         8),
    ],
)
def test_wave2_configs_reach_the_built_model(
    name: str, arm: str, expected_params: int, expected_encoder_gammas: list, num_layers: int
) -> None:
    """The five wave-2 tied YAMLs and their baseline control, loaded -> built -> asserted.

    These are the files about to be submitted. The 2026-09-09 audit (A7) found none of them was
    loaded and built by any test, and that ``tied_mc3_wave2.yaml``'s own header understated its
    parameter count by exactly the LayerScale gammas -- a sign nothing was checking. This asserts,
    on the BUILT model: total parameter count; the residual-branch gamma actually written into
    every encoder block (per-block, using each block's own loop count -- not one averaged eps);
    that the untied class-attention blocks carry the untied value in every arm (so the control
    and the arm differ only where tying happens); the schedule; and a finite forward.
    """
    from ablation.config import load_config

    config = load_config(str(_CONFIG_DIR / f"{name}.yaml"))
    assert config.arm == arm and config.run_name == name
    assert config.residual_scale_lambda == 1.0
    assert config.num_layers == num_layers
    assert config.total_steps == 1_000_000
    assert config.max_steps == (1_000_000 if name.endswith("_1M") else 200_000)

    torch.manual_seed(0)
    model = build_variant_part(config.arm, **config.model_kwargs())
    assert _params(model) == expected_params

    realised = [_gammas(b) for b in model.blocks]
    assert len(realised) == len(expected_encoder_gammas) == num_layers
    for depth, (got, want) in enumerate(zip(realised, expected_encoder_gammas)):
        assert got == pytest.approx(want, abs=1e-6), (name, depth, got, want)

    # Class-attention blocks are never tied, so their scale must be the untied one, lam/sqrt(depth),
    # in EVERY arm -- what an untied baseline of the same depth gets. If a tied arm's cls blocks
    # inherited the arm's uniform initial eps instead (they did, before 2026-09-09), the wave-2
    # comparison would differ from its control in a place that has nothing to do with tying.
    untied = 1.0 / math.sqrt(num_layers)
    for i, cls_block in enumerate(model.cls_blocks):
        assert _gammas(cls_block) == pytest.approx((untied, untied), abs=1e-6), (
            name, "cls_block", i, _gammas(cls_block)
        )

    if arm == "tied":
        assert model.tie_schedule.depth == num_layers
        assert model.tie_schedule.num_unique == config.tie_num_unique
        assert model.tie_scope == config.tie_scope
        assert model.residual_scale["lambda"] == 1.0

    x, v, mask = _batch()
    with torch.no_grad():
        out = model.eval()(x, v=v, mask=mask)
    assert out.shape == (x.shape[0], 10) and torch.isfinite(out).all()


# ---------------------------------------------------------------------------
# residual_scale_lambda must not be a phantom on arms that cannot realise it
# ---------------------------------------------------------------------------

_NO_LAYER_SCALE_ARMS = ("moe", "sparsemax", "diff_v1", "diff_v2", "urot", "lloca", "n8")


@pytest.mark.parametrize("arm", _NO_LAYER_SCALE_ARMS)
def test_residual_scale_lambda_is_refused_at_config_load_for_custom_block_arms(arm: str) -> None:
    """A config key that cannot reach the model must fail loudly, not be stored as if it applied."""
    kwargs = {"arm": arm, "residual_scale_lambda": 1.0}
    if arm == "lloca":
        kwargs["precision"] = "fp32"
    with pytest.raises(ValueError, match="no LayerScale"):
        AblationConfig(**kwargs)


@pytest.mark.parametrize("arm", _NO_LAYER_SCALE_ARMS)
def test_residual_scale_lambda_is_refused_by_the_builder_for_custom_block_arms(arm: str) -> None:
    """Belt and braces: bypass the config and call the builder directly. It must still refuse."""
    with pytest.raises(ValueError, match="cannot be realised"):
        build_variant_part(arm, **_kwargs(), layer_scale_init_values=0.125)


@pytest.mark.parametrize("arm", ["baseline", "lowrank", "tied"])
def test_residual_scale_lambda_is_realised_on_stock_block_arms(arm: str) -> None:
    """The positive case: on stock-block arms every encoder block must carry the requested gamma."""
    extra = {"tie_num_unique": 2} if arm == "tied" else {}
    config = AblationConfig(arm=arm, residual_scale_lambda=1.0, **extra)
    torch.manual_seed(0)
    model = build_variant_part(config.arm, **config.model_kwargs())
    for block in model.blocks:
        inner = getattr(block, "block", block)
        assert hasattr(inner.ls1, "gamma") and hasattr(inner.ls2, "gamma")
    assert _params(model) > _params(build_variant_part(config.arm, **AblationConfig(arm=arm, **extra).model_kwargs()))


def test_tied_config_rejects_impossible_schedules() -> None:
    """Fail at config load, not mid-run: a bad schedule must not reach an sbatch."""
    from ablation.config import AblationConfig

    for bad in (
        dict(tie_num_unique=0),
        dict(tie_num_unique=9),
        dict(tie_strategy="spiral"),
        dict(tie_mor_capacity=0.0),
        dict(tie_mor_capacity=2.0),
        dict(tie_depth_modulation=True, tie_mor_capacity=0.5),
    ):
        with pytest.raises(ValueError):
            AblationConfig(arm="tied", **bad)


def test_tied_settings_are_ignored_for_other_arms() -> None:
    """They must not leak into baseline's kwargs, or every arm's provenance becomes ambiguous."""
    from ablation.config import AblationConfig

    kwargs = AblationConfig(arm="baseline", tie_num_unique=1).model_kwargs()
    assert "tie_num_unique" not in kwargs


# ---------------------------------------------------------------------------
# Regressions for defects found in external review (2026-09-05)
# ---------------------------------------------------------------------------
# Each test below corresponds to a bug that shipped green: the suite passed while the code was
# wrong. They are grouped so that the property, not the fix, is what is asserted.

def test_router_capacity_is_a_fraction_of_each_jets_own_multiplicity() -> None:
    """capacity=0.5 must mean half of *this jet's* particles, not half the padded width.

    The padded width is a property of the batch, not of the jet. Deriving `k` from it made the
    effective capacity depend on how much padding a jet sat beside: at capacity=0.5 in a tensor
    padded to 128, a 20-particle jet got 64 >= 20 slots and therefore ran DENSE, as did a
    40-particle jet, while only jets above 128 particles saw a real 0.5. Since JetClass
    multiplicity averages ~39, that made the majority of an arm labelled "capacity 0.5" silently
    identical to the dense baseline.
    """
    router = MoRRouter(embed_dim=8, capacity=0.5)
    valid = [20, 40, 100]
    x = torch.randn(len(valid), 128, 8)
    padding = torch.zeros(len(valid), 128, dtype=torch.bool)
    for row, count in enumerate(valid):
        padding[row, count:] = True

    _, keep = router(x, padding)
    assert keep.sum(dim=1).tolist() == [10, 20, 50]
    assert not bool((keep & padding).any())


@pytest.mark.parametrize("capacity,valid,expected", [
    (0.5, 39, 20),    # ceil(19.5) -- rounds up, so a step is never empty
    (0.25, 4, 1),
    (1.0, 7, 7),      # full capacity means every valid particle, never a padded slot
    (0.5, 1, 1),      # a single-particle jet still gets one
])
def test_router_capacity_rounds_up_per_jet(capacity: float, valid: int, expected: int) -> None:
    router = MoRRouter(embed_dim=8, capacity=capacity)
    width = 64                      # comfortably wider than any `valid` below, so it is padding
    x = torch.randn(1, width, 8)
    padding = torch.zeros(1, width, dtype=torch.bool)
    padding[0, valid:] = True
    _, keep = router(x, padding)
    assert int(keep.sum()) == expected


def test_router_worst_case_selection_is_still_bounded() -> None:
    """The trigger budget is worst-case, so per-jet capacity must not raise the upper bound."""
    router = MoRRouter(embed_dim=8, capacity=0.5)
    x = torch.randn(4, 64, 8)
    _, keep = router(x)                       # no padding: every slot valid, the worst case
    assert int(keep.sum(dim=1).max()) == 32   # == ceil(capacity * N)


def test_router_selection_is_deterministic_at_init() -> None:
    """Zero-init makes every logit 0, so the selected set is decided entirely by tie-breaking.

    `topk` does not document a stable tie order and its choice varies by backend, which would make
    an untrained MoR arm irreproducible across devices at a fixed seed. A stable sort does document
    it: lower index first, i.e. toward the leading (highest-pT) constituents.
    """
    router = MoRRouter(embed_dim=8, capacity=0.5)
    x = torch.randn(1, 10, 8)
    _, first = router(x)
    _, again = router(torch.randn(1, 10, 8))   # different input, still all-zero logits
    assert torch.equal(first, again)
    assert first[0, :5].all() and not first[0, 5:].any()


def test_shared_block_is_invoked_once_per_depth() -> None:
    """Eight depths must mean eight applications of the shared block, forward and backward.

    The prior gradient test only checked that *some* aggregate gradient reached the shared weights,
    which a stack that applied the block once would also pass.
    """
    model = _build(tie_num_unique=1)
    shared = model.blocks[0]
    counts = {"forward": 0, "backward": 0}
    shared.register_forward_hook(
        lambda *_: counts.__setitem__("forward", counts["forward"] + 1)
    )
    shared.register_full_backward_hook(
        lambda *_: counts.__setitem__("backward", counts["backward"] + 1)
    )

    x, v, mask = _batch()
    model(x, v=v, mask=mask).sum().backward()
    assert counts["forward"] == _DEPTH
    assert counts["backward"] == _DEPTH


def test_router_receives_gradient() -> None:
    """If the router cannot learn, R4 is a fixed random subset wearing MoR's name."""
    model = _build(tie_num_unique=1, tie_mor_capacity=0.5)
    x, v, mask = _batch()
    model(x, v=v, mask=mask).sum().backward()
    router = model.blocks[0].router
    assert router.score.weight.grad is not None
    assert float(router.score.weight.grad.abs().sum()) > 0


# ---- checkpoint compatibility ---------------------------------------------------------------
# `load_state_dict` is a name-and-shape check, and tied and untied stacks agree on both. So a
# mismatched load reports "all keys matched successfully" and keeps only the last duplicate of
# each shared block: training then continues from a model equal to no part of the checkpoint,
# with finite weights, a reasonable loss, and a config that says otherwise.

def test_untied_checkpoint_into_tied_model_is_refused() -> None:
    from variants.tied import assert_shared_blocks_consistent

    torch.manual_seed(0)
    untied = build_variant_part("baseline", **_kwargs())
    tied = _build(tie_num_unique=1)

    # The load itself "succeeds" -- that is the whole problem.
    assert tied.load_state_dict(untied.state_dict()) is not None
    with pytest.raises(ValueError, match="disagrees across depths"):
        assert_shared_blocks_consistent(tied, untied.state_dict())


def test_same_schedule_round_trip_is_accepted() -> None:
    """The guard must not fire on a legitimate resume, or it just blocks the normal path."""
    from variants.tied import assert_shared_blocks_consistent

    source = _build(tie_num_unique=2, tie_strategy="cycle")
    target = _build(tie_num_unique=2, tie_strategy="cycle")
    state = source.state_dict()
    target.load_state_dict(state)
    assert_shared_blocks_consistent(target, state)      # must not raise

    for a, b in zip(source.parameters(), target.parameters()):
        torch.testing.assert_close(a, b, rtol=0, atol=0)


def test_untied_model_load_is_never_blocked() -> None:
    """No shared groups means nothing to check; the guard must be a no-op for baseline."""
    from variants.tied import assert_shared_blocks_consistent

    torch.manual_seed(0)
    model = build_variant_part("baseline", **_kwargs())
    assert_shared_blocks_consistent(model, model.state_dict())


def test_architecture_fingerprint_separates_tying_schedules() -> None:
    from ablation.config import AblationConfig

    base = AblationConfig(arm="baseline").architecture_fingerprint()
    k1 = AblationConfig(arm="tied", tie_num_unique=1).architecture_fingerprint()
    k4 = AblationConfig(arm="tied", tie_num_unique=4).architecture_fingerprint()
    cyc = AblationConfig(arm="tied", tie_num_unique=4, tie_strategy="cycle")
    seq = AblationConfig(arm="tied", tie_num_unique=4, tie_strategy="sequence")

    assert base != k1 and k1 != k4
    assert cyc.architecture_fingerprint() != seq.architecture_fingerprint()
    assert (AblationConfig(arm="tied", tie_num_unique=4).architecture_fingerprint() == k4)


def test_architecture_fingerprint_tracks_model_kwargs() -> None:
    """The fingerprint is derived from `model_kwargs`, not from a hand-written field list.

    A hand-written list is a second source of truth, and the `ffn2x` failure is exactly what
    happens when one of them stops matching the builder.
    """
    from ablation.config import AblationConfig

    config = AblationConfig(arm="tied", tie_num_unique=2)
    fingerprint = config.architecture_fingerprint()
    for key in config.model_kwargs():
        assert key in fingerprint


def test_checkpoint_architecture_mismatch_is_refused() -> None:
    from ablation.config import AblationConfig, assert_checkpoint_architecture

    tied = AblationConfig(arm="tied", tie_num_unique=1)
    payload = {"config": AblationConfig(arm="baseline").to_dict()}

    with pytest.raises(ValueError, match="different architecture"):
        assert_checkpoint_architecture(payload, tied)

    # Matching config: must pass.
    assert_checkpoint_architecture({"config": tied.to_dict()}, tied)


def test_checkpoint_without_config_is_refused_not_assumed_compatible() -> None:
    from ablation.config import AblationConfig, assert_checkpoint_architecture

    with pytest.raises(ValueError, match="no `config`"):
        assert_checkpoint_architecture({}, AblationConfig(arm="tied", tie_num_unique=1))


def test_expansion_factor_reaches_the_fingerprint() -> None:
    """The original silent failure, asserted at the layer that now guards against it."""
    from ablation.config import AblationConfig

    four = AblationConfig(arm="baseline", expansion_factor=4).architecture_fingerprint()
    eight = AblationConfig(arm="baseline", expansion_factor=8).architecture_fingerprint()
    assert four != eight


# ---------------------------------------------------------------------------
# The residual-scaling law (eps = lambda / (N sqrt(L))) — added 2026-09-06
# ---------------------------------------------------------------------------
# `build_tied_blocks` slices `blocks[:num_unique]` AFTER weaver's per-depth `fix_init`, so a tied arm
# keeps the shallowest, LEAST-rescaled blocks and applies them N times — the opposite of what a
# looped network needs. Measured residual growth over depth: baseline 3.72x, tied k=1 9.23x. These
# tests pin the correction and, critically, that it is applied to BOTH arms so it is a control
# rather than a favour to the tied arm.

def _residual_growth(model, batch) -> float:
    """Ratio of the token-representation RMS entering the last block to the first."""
    x, v, mask = batch
    rms: list[float] = []

    def hook(_module, inputs):
        t = inputs[0] if isinstance(inputs, tuple) else inputs
        rms.append(float(t.detach().float().pow(2).mean().sqrt()))

    handles = [b.register_forward_pre_hook(hook) for b in model.blocks]
    try:
        with torch.no_grad():
            model(x, v=v, mask=mask)
    finally:
        for h in handles:
            h.remove()
    return rms[-1] / rms[0]


@pytest.mark.parametrize(
    "arm,extra,expected",
    [
        ("baseline", {}, 1.0 / math.sqrt(8)),      # L = num_layers, N = 1
        ("tied", dict(tie_num_unique=1), 0.125),   # L = 1, N = 8
        ("tied", dict(tie_num_unique=2), 1.0 / (4.0 * math.sqrt(2))),
        ("tied", dict(tie_num_unique=4), 0.25),
        ("tied", dict(tie_num_unique=8), 1.0 / math.sqrt(8)),   # == baseline, as it must
    ],
)
def test_residual_branch_scale_follows_the_epsilon_law(arm, extra, expected) -> None:
    """One formula covers every arm — an untied stack is just L = num_layers, N = 1.

    That is what makes applying it a control: if the law only existed for the tied arm, the
    comparison would be confounded in the tied arm's favour, which is the mirror of the confound it
    exists to remove.
    """
    config = AblationConfig(arm=arm, residual_scale_lambda=1.0, **extra)
    assert config.residual_branch_scale() == pytest.approx(expected, rel=1e-12)


def test_epsilon_law_is_off_by_default() -> None:
    """Existing runs must stay reproducible, so the stock path is untouched."""
    assert AblationConfig(arm="baseline").residual_branch_scale() is None
    model = build_variant_part("baseline", **AblationConfig(arm="baseline").model_kwargs())
    assert isinstance(model.blocks[0].ls1, torch.nn.Identity)


def test_epsilon_law_installs_layerscale_on_both_arms() -> None:
    """Asserted on the BUILT model: weaver gates LayerScale on a truthy value, so a silently
    dropped config key would leave `nn.Identity` and the run would be the uncorrected one."""
    for arm, extra in (("baseline", {}), ("tied", dict(tie_num_unique=1))):
        config = AblationConfig(arm=arm, residual_scale_lambda=1.0, **extra)
        torch.manual_seed(0)
        model = build_variant_part(arm, **config.model_kwargs())
        assert not isinstance(model.blocks[0].ls1, torch.nn.Identity), arm
        assert not isinstance(model.blocks[0].ls2, torch.nn.Identity), arm


def test_epsilon_law_makes_residual_growth_arm_independent() -> None:
    """The property the law exists for, measured rather than assumed.

    Without it, growth over depth spans 3.72x (baseline) to 9.23x (tied k=1) — a 2.5x spread that
    would make any tied-vs-baseline delta partly a comparison of forward statistics. With it, every
    arm lands in a narrow band.
    """
    batch = _batch(batch=8, particles=32)
    arms = [("baseline", {}), ("tied", dict(tie_num_unique=1)),
            ("tied", dict(tie_num_unique=2)), ("tied", dict(tie_num_unique=4))]

    uncorrected, corrected = [], []
    for arm, extra in arms:
        for lam, sink in ((None, uncorrected), (1.0, corrected)):
            config = AblationConfig(arm=arm, residual_scale_lambda=lam, **extra)
            torch.manual_seed(0)
            model = build_variant_part(arm, **config.model_kwargs()).eval()
            sink.append(_residual_growth(model, batch))

    # Uncorrected: the tied arms grow much faster than baseline.
    assert max(uncorrected) / min(uncorrected) > 2.0
    # Corrected: all arms within 25% of each other, and every one below the worst uncorrected.
    assert max(corrected) / min(corrected) < 1.25
    assert max(corrected) < min(uncorrected)


# ---------------------------------------------------------------------------
# Schedules the prior art and this repo's own data point at
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "num_unique,strategy,expected",
    [
        (3, "middle-cycle", (0, 1, 1, 1, 1, 1, 1, 2)),
        (4, "middle-cycle", (0, 1, 2, 1, 2, 1, 2, 3)),
        (2, "head-unique", (0, 1, 1, 1, 1, 1, 1, 1)),
        (3, "head-unique", (0, 1, 2, 1, 2, 1, 2, 1)),
    ],
)
def test_entry_exit_preserving_schedules(num_unique, strategy, expected) -> None:
    """`middle-cycle` keeps the first AND last block unique; `head-unique` keeps only the first.

    middle-cycle is the configuration four independent papers converge on (MoR arXiv:2507.10524
    finds it "consistently achieves the lowest validation loss"). head-unique is what this repo's
    OWN R0 data points at: the trained ffn similarity matrix is monotone in depth with block 0 the
    outlier, so "block 0 unique, deep blocks tied" is the highest-prior partial schedule.
    """
    assert TiedSchedule(_DEPTH, num_unique, strategy).indices == expected


@pytest.mark.parametrize("bad", [
    dict(depth=8, num_unique=2, strategy="middle-cycle"),   # needs >= 3
    dict(depth=8, num_unique=1, strategy="head-unique"),    # needs >= 2
    dict(depth=8, num_unique=8, strategy="head-unique"),    # no middle left to share
])
def test_entry_exit_schedules_reject_degenerate_shapes(bad) -> None:
    with pytest.raises(ValueError):
        TiedSchedule(**bad)


def test_schedule_supports_more_loops_than_the_untied_baseline() -> None:
    """Depth > 8 is the configuration Takase & Kiyono found beats untied at MATCHED parameters
    (+0.83 BLEU at 61M, also beating untied 149M and 210M). Without it the arm can only ever be
    "cheaper at fixed depth", never "deeper at fixed cost"."""
    schedule = TiedSchedule(16, 2, "cycle")
    assert len(schedule.indices) == 16
    assert schedule.indices == (0, 1) * 8


# ---------------------------------------------------------------------------
# Sub-block tying scopes (ALBERT's ablation says attn and ffn are not equivalent)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "scope,expected_params,shares",
    [
        ("attn", _BASELINE_PARAMS - 66_048 * 7, ("attn",)),
        ("ffn", _BASELINE_PARAMS - (66_048 + 65_664) * 7, ("fc1", "fc2")),
    ],
)
def test_sub_block_scopes_share_only_their_submodule(scope, expected_params, shares) -> None:
    """ALBERT Table 4: sharing attention is free (+0.1) and the damage is in the FFN (-1.4).

    In ParT attention is only 33.1% of a block (66,048 of 199,688), so the ALBERT-safe scope gives
    1.28x where whole-block tying gives 2.87x — i.e. this repo's headline reduction comes precisely
    from sharing the sublayer ALBERT identifies as costly. Running both turns "does tying work?"
    into "which sublayer carries depthwise specialisation?".
    """
    model = _build(tie_num_unique=1, tie_scope=scope)
    assert _params(model) == expected_params

    for attr in shares:
        assert getattr(model.blocks[0], attr) is getattr(model.blocks[7], attr)
    # Norms and gain controls stay per-depth: that is the conservative reading of "share the
    # attention parameters", and it keeps the per-depth gain control the literature finds matters.
    for attr in ("pre_attn_norm", "post_attn_norm", "pre_fc_norm", "w_resid", "c_attn"):
        assert getattr(model.blocks[0], attr) is not getattr(model.blocks[7], attr)
    not_shared = ("fc1",) if scope == "attn" else ("attn",)
    for attr in not_shared:
        assert getattr(model.blocks[0], attr) is not getattr(model.blocks[7], attr)


def test_sub_block_scope_keeps_all_eight_block_objects() -> None:
    """Sub-block sharing must NOT collapse the ModuleList — each depth keeps its own block so its
    own norms survive. Whole-block tying is the one that repeats objects."""
    model = _build(tie_num_unique=1, tie_scope="attn")
    assert len({id(b) for b in model.blocks}) == _DEPTH
    assert len({id(b) for b in _build(tie_num_unique=1, tie_scope="block").blocks}) == 1


def test_sub_block_scope_refuses_to_combine_with_r3_or_r4() -> None:
    """Sub-block sharing plus a second mechanism would confound two things in one number."""
    for extra in (dict(depth_modulation=True), dict(mor_capacity=0.5)):
        with pytest.raises(ValueError, match="scope"):
            build_tied_blocks(
                [torch.nn.Identity() for _ in range(8)],
                TiedSchedule(8, 1), embed_dim=128, scope="attn", **extra,
            )


def test_config_rejects_unknown_tie_scope() -> None:
    with pytest.raises(ValueError, match="tie_scope"):
        AblationConfig(arm="tied", tie_scope="everything")
