"""Properties of the rank-r factorized pair bias — the arm the rank audit licensed.

Every assertion is on a **built model** or on the bias tensor it emits, never on a config field.
That is not a style preference: `baseline_ffn2x` trained a model byte-identical to baseline while a
green test asserted only that its YAML contained the number 8, and that arm's +0.00296 was reported
as a result for weeks.

The properties that carry the weight here:

* the bias really is **rank r** — off the ``c*I`` diagonal, which is deliberately rank ``N`` on one
  parameter (result P4). This is the ``const_diag`` form P5 measured, so the arm matches the
  measurement that justified it;
* it is **symmetric**, matching weaver's ``PairEmbed`` (the audit asserts asymmetry < 1e-5);
* it is **indefinite at init**, because ``F^T diag(s) F`` with all-positive ``s`` is PSD and would
  start the arm on the boundary of the family it is supposed to explore;
* padded pairs are **exactly zero**, which the audit's ``slice_valid`` and attention's key masking
  both depend on;
* the init bias scale **matches the PairEmbed it replaces**, so a lowrank-vs-baseline comparison is
  not partly a comparison of effective initialisations.
"""

from __future__ import annotations

import math

import pytest
import torch

from ablation.config import AblationConfig
from variants import build_variant_part
from variants.lowrank import (
    PARTICLE_FEATURE_NAMES,
    LowRankPairEmbed,
    lowrank_cost_report,
    particle_features,
)

_HEADS = 8


def _kwargs(**over) -> dict:
    return AblationConfig(arm="lowrank", **over).model_kwargs()


def _build(rank: int = 16, **over):
    torch.manual_seed(0)
    return build_variant_part("lowrank", **_kwargs(lowrank_rank=rank, **over))


def _batch(sizes=(32, 20, 8), width=32, seed=3):
    g = torch.Generator().manual_seed(seed)
    b = len(sizes)
    p3 = torch.randn(b, 3, width, generator=g) * 5.0
    energy = (p3 ** 2).sum(1, keepdim=True).sqrt() * 1.05      # timelike
    v = torch.cat([p3, energy], dim=1)
    x = torch.randn(b, 16, width, generator=g)
    mask = torch.zeros(b, 1, width)
    for row, n in enumerate(sizes):
        mask[row, 0, :n] = 1.0
        v[row, :, n:] = 0.0                                    # padded slots are exact zeros
        x[row, :, n:] = 0.0
    return x, v, mask


# ---------------------------------------------------------------------------
# the rank property
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("rank", [4, 8, 16, 32])
def test_bias_is_rank_r_off_the_diagonal_term(rank: int) -> None:
    """rank(U - c*I) <= r on every jet's valid block.

    Stated as ``U - c*I`` rather than ``U`` deliberately. The ``c[h] * delta_ij`` term is rank ``N``
    on a *single* parameter, which is exactly why P4 found it worth a factor of 2 in effective rank
    — so ``U`` itself is full rank and the budget applies to everything else. This is the
    ``const_diag`` configuration P5 measured, and conflating the two would make the arm look like it
    violated its own rank claim.
    """
    model = _build(rank).eval()
    with torch.no_grad():
        model.pair_embed.self_pair.fill_(0.7)                  # non-trivial diagonal
        _, v, mask = _batch()
        bias = model.pair_embed(v, uu=None, mask=mask).double()

    for row, n in enumerate((32, 20, 8)):
        block = bias[row, 0, :n, :n] - 0.7 * torch.eye(n, dtype=torch.float64)
        sv = torch.linalg.svdvals(block)
        assert int((sv > sv[0] * 1e-6).sum()) <= min(rank, n)


def test_diagonal_term_is_one_number_per_head() -> None:
    """P4/P5's finding is that ONE scalar per head buys a factor of 2 in rank. Assert the shape."""
    model = _build(16)
    assert model.pair_embed.self_pair.shape == (_HEADS,)
    assert model.pair_embed.self_pair.numel() == _HEADS


def test_self_pair_term_can_be_disabled_for_the_ablation() -> None:
    with_term = _build(16, lowrank_self_pair=True)
    without = _build(16, lowrank_self_pair=False)
    assert with_term.pair_embed.self_pair is not None
    assert without.pair_embed.self_pair is None
    delta = sum(p.numel() for p in with_term.parameters()) - sum(
        p.numel() for p in without.parameters()
    )
    assert delta == _HEADS


# ---------------------------------------------------------------------------
# matching what it replaces
# ---------------------------------------------------------------------------

def test_bias_shape_and_symmetry_match_weaver_pair_embed() -> None:
    """Weaver hands this tensor to every block, so shape and symmetry are interface contracts."""
    model = _build(16).eval()
    _, v, mask = _batch()
    with torch.no_grad():
        bias = model.pair_embed(v, uu=None, mask=mask)
    assert bias.shape == (3, _HEADS, 32, 32)
    assert torch.isfinite(bias).all()
    assert float((bias - bias.transpose(-1, -2)).abs().max()) < 1e-6


def test_padded_pairs_are_exactly_zero() -> None:
    """The audit documents "padded pairs in U are exactly 0.0" and `slice_valid` relies on it.

    Not approximately: a nonzero padded entry would add spurious singular values to every rank
    measurement taken downstream, and attention's key masking assumes it too.
    """
    model = _build(16).eval()
    with torch.no_grad():
        model.pair_embed.self_pair.fill_(0.5)     # the diagonal term must respect padding too
        _, v, mask = _batch()
        bias = model.pair_embed(v, uu=None, mask=mask)
    checked = 0
    for row, n in enumerate((32, 20, 8)):
        if n >= bias.shape[-1]:
            continue                    # this jet fills the width; nothing is padded
        assert float(bias[row, :, n:, :].abs().max()) == 0.0
        assert float(bias[row, :, :, n:].abs().max()) == 0.0
        checked += 1
    assert checked >= 2, "the batch must actually contain padded jets for this to test anything"


def test_init_bias_scale_matches_the_pair_embed_it_replaces() -> None:
    """Within a factor of ~3 of stock `PairEmbed`, so the arm is not confounded by its own init.

    `U` is *quadratic* in the factors, so this is easy to get catastrophically wrong: a 0.1 shrink on
    the final layer shrinks the bias 100x. A first version did that and emitted |U| std ~ 5e-5,
    i.e. numerically no pair bias — which starts the arm at P5's r=0 point (0.30 accuracy) with a
    vanishing gradient, and would have looked like "rank-r biases do not train".
    """
    lowrank = _build(16).eval()
    torch.manual_seed(0)
    baseline = build_variant_part("baseline", **AblationConfig(arm="baseline").model_kwargs()).eval()

    _, v, mask = _batch()
    with torch.no_grad():
        ours = float(lowrank.pair_embed(v, uu=None, mask=mask).std())
        theirs = float(baseline.pair_embed(v, uu=None, mask=mask).std())
    assert theirs > 0
    assert 1 / 3 < ours / theirs < 3


def test_bias_is_indefinite_at_init() -> None:
    """All-positive component scales would make `F^T diag(s) F` PSD.

    The measured trained bias is indefinite, so a PSD init starts the arm on the boundary of the
    family it is meant to explore and forces it to learn its way off. Signs alternate at init.
    """
    model = _build(16).eval()
    _, v, mask = _batch(sizes=(32,), width=32)
    with torch.no_grad():
        bias = model.pair_embed(v, uu=None, mask=mask).double()
    eigenvalues = torch.linalg.eigvalsh(bias[0, 0])
    assert float(eigenvalues.min()) < -1e-3
    assert float(eigenvalues.max()) > 1e-3


# ---------------------------------------------------------------------------
# features
# ---------------------------------------------------------------------------

def test_particle_features_are_finite_on_padded_zeros() -> None:
    """Padded slots are exact zeros, so ln(pT) and ln(E) would be -inf without the clamps.

    `-inf * 0` is `nan`, so masking afterwards does not save it — the clamp is load-bearing.
    """
    v = torch.zeros(2, 4, 6)
    v[:, :, :3] = torch.randn(2, 4, 3)
    v[:, 3, :3] = v[:, :3, :3].pow(2).sum(1).sqrt() * 1.05
    feats = particle_features(v)
    assert feats.shape == (2, len(PARTICLE_FEATURE_NAMES), 6)
    assert torch.isfinite(feats).all()


def test_phi_is_encoded_without_a_wrap_discontinuity() -> None:
    """cos/sin phi, not phi: `cos(phi_i - phi_j)` is then exactly rank 2 in these features.

    Verified by rotating a jet in phi and checking the (cos, sin) pair rotates continuously rather
    than jumping — a raw periodic phi would put a seam in the input space.
    """
    names = list(PARTICLE_FEATURE_NAMES)
    ci, si = names.index("cos_phi"), names.index("sin_phi")
    previous = None
    for angle in torch.linspace(-math.pi, math.pi, 64):
        v = torch.tensor([[[math.cos(float(angle))], [math.sin(float(angle))],
                           [0.0], [1.05]]])
        feats = particle_features(v)[0, :, 0]
        point = torch.tensor([feats[ci], feats[si]])
        if previous is not None:
            assert float((point - previous).norm()) < 0.3      # no jump anywhere, including the wrap
        previous = point


# ---------------------------------------------------------------------------
# gradients and cost
# ---------------------------------------------------------------------------

def test_every_parameter_group_receives_gradient() -> None:
    """If the component scales or the self-pair term were starved, the arm would be a fixed random
    projection wearing the name of a learned factorization."""
    model = _build(16)
    x, v, mask = _batch()
    model(x, v=v, mask=mask).sum().backward()
    pair = model.pair_embed
    for name, param in (("net.0", pair.net[0].weight),
                        ("net[-1]", pair.net[-1].weight),
                        ("component_scale", pair.component_scale),
                        ("self_pair", pair.self_pair)):
        assert param.grad is not None, name
        assert float(param.grad.abs().sum()) > 0, name


def test_forward_runs_and_is_finite_for_every_rank() -> None:
    x, v, mask = _batch()
    for rank in (1, 4, 16, 32):
        model = _build(rank).eval()
        with torch.no_grad():
            out = model(x, v=v, mask=mask)
        assert out.shape == (3, 10)
        assert torch.isfinite(out).all()


def test_cost_report_separates_flops_from_a_speedup_it_does_not_deliver() -> None:
    """The FLOP win is real; the wall-clock win is not, because weaver needs a dense attn_mask.

    Asserted because conflating the two is the exact error this repository already published once,
    and a cost report is precisely where it would happen again.
    """
    report = lowrank_cost_report(rank=16, num_particles=128)
    assert report["mlp_evaluations_pair"] == 128 * 128
    assert report["mlp_evaluations_lowrank"] == 128
    assert report["mlp_evaluation_reduction"] == 128
    assert report["flop_reduction"] > 1.0
    assert report["materializes_dense_bias"] is True
    assert report["speedup_delivered"] is None
    assert "NOT be wall-clock faster" in report["speedup_note"]


def test_arm_is_registered_and_config_reaches_the_built_model() -> None:
    from variants import VARIANTS

    assert "lowrank" in VARIANTS
    for rank in (8, 16, 32):
        model = _build(rank)
        assert model.lowrank_config["rank"] == rank
        assert model.pair_embed.rank == rank
        assert model.pair_embed.component_scale.shape == (_HEADS, rank)


def test_rank_changes_the_architecture_fingerprint() -> None:
    """So a resume cannot silently load a rank-8 checkpoint into a rank-32 model."""
    a = AblationConfig(arm="lowrank", lowrank_rank=8).architecture_fingerprint()
    b = AblationConfig(arm="lowrank", lowrank_rank=32).architecture_fingerprint()
    c = AblationConfig(arm="lowrank", lowrank_self_pair=False).architecture_fingerprint()
    assert a != b and b != c


def test_lowrank_settings_do_not_leak_into_other_arms() -> None:
    kwargs = AblationConfig(arm="baseline", lowrank_rank=8).model_kwargs()
    assert "lowrank_rank" not in kwargs


def test_config_rejects_impossible_settings() -> None:
    with pytest.raises(ValueError):
        AblationConfig(arm="lowrank", lowrank_rank=0)
    with pytest.raises(ValueError):
        AblationConfig(arm="lowrank", pair_embed_dims=None)


def test_module_rejects_impossible_settings() -> None:
    with pytest.raises(ValueError):
        LowRankPairEmbed(rank=0, num_heads=8)
    with pytest.raises(ValueError):
        LowRankPairEmbed(rank=4, num_heads=0)


def test_part_kernels_skips_our_pair_embed_rather_than_clobbering_it() -> None:
    """part_kernels must leave `LowRankPairEmbed` alone, and we must know that it does.

    Patch A fires on ``isinstance(module, PairEmbed)``. This module is not a weaver ``PairEmbed``
    subclass, so it is skipped — but "skipped" and "clobbered" look identical from the outside until
    you check, and the runs use ``USE_PART_KERNELS=true``. If the patch ever started matching by
    duck-typing instead, the arm's entire point would silently vanish into a fused dense kernel.

    The asymmetry this pins has a real consequence recorded in `lowrank_cost_report`: baseline gets
    its dominant kernel fused 4.54x and this arm does not, so wall-clock is biased AGAINST this arm
    even though it does ~40x fewer FLOPs. Accuracy stays comparable; throughput does not.
    """
    part_kernels = pytest.importorskip("part_kernels")

    cfg = AblationConfig(arm="lowrank", lowrank_rank=16, use_part_kernels=True)
    torch.manual_seed(0)
    model = build_variant_part("lowrank", **cfg.model_kwargs()).eval()
    x, v, mask = _batch(sizes=(24, 16), width=24)
    with torch.no_grad():
        before = model(x, v=v, mask=mask).clone()

    result = part_kernels.optimize_part_model(model, use_attention_patch=cfg.part_kernels_attention)
    patched = result[0] if isinstance(result, tuple) else model
    stats = result[1] if isinstance(result, tuple) else {}

    assert isinstance(patched.pair_embed, LowRankPairEmbed), "our module was replaced"
    assert stats.get("pair_embed_instances") == 0, "Patch A matched our module; it must not"
    with torch.no_grad():
        assert torch.equal(before, patched(x, v=v, mask=mask))


def test_baseline_does_get_the_fused_pair_kernel() -> None:
    """The other half of the asymmetry, pinned so the comparison caveat stays true.

    Asserted separately from the test above because the caveat is only meaningful as a *pair*: it is
    not "kernels are off for everyone", it is "baseline gets one and lowrank does not".
    """
    part_kernels = pytest.importorskip("part_kernels")

    cfg = AblationConfig(arm="baseline", use_part_kernels=True)
    torch.manual_seed(0)
    model = build_variant_part("baseline", **cfg.model_kwargs()).eval()
    result = part_kernels.optimize_part_model(model, use_attention_patch=cfg.part_kernels_attention)
    stats = result[1] if isinstance(result, tuple) else {}
    assert stats.get("pair_embed_instances") == 1
    assert "pair_embed" in (stats.get("patches") or [])
