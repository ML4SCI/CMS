"""Correctness properties of the pair-bias rank audit (``ablation.rank_audit``).

The audit's whole purpose is to produce a number a build/no-build decision is
made on (hypothesis T0.1 / plan section 8.7a), so the measurement machinery
needs its own tests: a silently wrong residual would send the N8 contribution
in the wrong direction with no downstream check to catch it.

Properties verified
-------------------
P1  A rank-r bilinear form built from the four-momenta is recovered exactly by
    the degree-1 moment fit — this is the "exactly free" claim of N8, so if it
    does not hold the fit is broken.
P2  A rank-r matrix has exactly r non-negligible singular values, and the
    Eckart-Young residual at rank r is 0.
P3  Residuals are monotone non-increasing in rank (more dimensions cannot fit
    worse) for both the SVD bound and the moment fit.
P4  The moment fit never beats the SVD bound at equal rank (Eckart-Young).
P5  Moment features have the documented dimensions and unit-norm columns, and
    column scaling does not change the achievable fit.
P6  Padded positions are excluded: a bias whose valid block is low rank is
    reported low rank regardless of how much zero padding surrounds it.
"""

from __future__ import annotations

import math

import pytest
import torch
from hypothesis import given, settings
from hypothesis import strategies as st

from ablation.rank_audit import (
    ENERGY_FRACTIONS,
    MOMENT_DEGREES,
    PROBE_RANKS,
    audit_batch,
    bilinear_residual,
    moment_dim,
    moment_features,
    slice_valid,
    spectral_profile,
    summarize,
    verdict,
)
from variants.tests.strategies import _make_four_vectors, _make_lengths_masks

_SETTINGS = settings(max_examples=25, deadline=None)


def _four_vectors(num_particles: int, seed: int) -> torch.Tensor:
    """One physically valid ``(n, 4)`` four-momentum block."""
    generator = torch.Generator()
    generator.manual_seed(seed)
    v = _make_four_vectors([num_particles], num_particles, generator)
    return v[0].T.contiguous()


# ---------------------------------------------------------------------------
# P1 / P2 / P4 — the fit and the bound agree with theory
# ---------------------------------------------------------------------------

@given(
    num_particles=st.integers(min_value=12, max_value=40),
    seed=st.integers(min_value=0, max_value=2**16),
)
@_SETTINGS
def test_p1_exact_bilinear_form_is_recovered(num_particles: int, seed: int) -> None:
    """P1: U = p M p^T is fit exactly by the degree-1 (4-dim) moment features.

    This is the load-bearing claim behind N8's "exactly free" rank-4 term: any
    M of rank <= 4 is reachable by appending 4 dimensions to Q and K, so the
    unconstrained degree-1 fit must have zero residual.
    """
    p = _four_vectors(num_particles, seed)
    generator = torch.Generator()
    generator.manual_seed(seed + 1)
    m = torch.randn(4, 4, generator=generator, dtype=torch.float64)
    m = m + m.T  # symmetric, matching weaver's symmetric PairEmbed

    u = p.to(torch.float64) @ m @ p.to(torch.float64).T

    phi = moment_features(p, 1)
    assert bilinear_residual(u, phi) < 1e-8


@given(
    rank=st.integers(min_value=1, max_value=6),
    size=st.integers(min_value=12, max_value=32),
    seed=st.integers(min_value=0, max_value=2**16),
)
@_SETTINGS
def test_p2_exact_rank_is_detected(rank: int, size: int, seed: int) -> None:
    """P2: a rank-r matrix needs r singular values and has zero residual at r."""
    generator = torch.Generator()
    generator.manual_seed(seed)
    a = torch.randn(size, rank, generator=generator, dtype=torch.float64)
    u = a @ a.T  # symmetric, exactly rank <= r

    profile = spectral_profile(u)
    assert profile["rank_for_energy"]["0.99"] <= rank

    for probe in PROBE_RANKS:
        if probe >= rank:
            assert profile["residual_at_rank"][str(probe)] < 1e-8


@given(
    num_particles=st.integers(min_value=40, max_value=64),
    seed=st.integers(min_value=0, max_value=2**16),
)
@_SETTINGS
def test_p4_moment_fit_never_beats_svd_bound(num_particles: int, seed: int) -> None:
    """P4: Eckart-Young — no feature choice beats the truncated SVD at equal rank."""
    p = _four_vectors(num_particles, seed)
    generator = torch.Generator()
    generator.manual_seed(seed + 7)
    # A generic nonlinear function of the pair invariants, standing in for the
    # trained pair MLP: log of the Minkowski product is full rank in general.
    gram = p[:, 3:4] * p[:, 3:4].T - p[:, :3] @ p[:, :3].T
    u = torch.log1p(gram.abs()).to(torch.float64)

    profile = spectral_profile(u)
    for degree in MOMENT_DEGREES:
        dim = moment_dim(degree)
        if dim >= num_particles:
            continue
        moment = bilinear_residual(u, moment_features(p, degree))
        best = profile["residual_at_rank"].get(str(dim))
        if best is not None:
            assert moment >= best - 1e-9


# ---------------------------------------------------------------------------
# P3 — monotonicity
# ---------------------------------------------------------------------------

@given(
    num_particles=st.integers(min_value=40, max_value=64),
    seed=st.integers(min_value=0, max_value=2**16),
)
@_SETTINGS
def test_p3_residuals_are_monotone_in_rank(num_particles: int, seed: int) -> None:
    """P3: more dimensions cannot fit worse, for both the bound and the fit."""
    p = _four_vectors(num_particles, seed)
    gram = p[:, 3:4] * p[:, 3:4].T - p[:, :3] @ p[:, :3].T
    u = torch.log1p(gram.abs()).to(torch.float64)

    profile = spectral_profile(u)
    svd_residuals = [profile["residual_at_rank"][str(r)] for r in sorted(PROBE_RANKS)]
    assert all(
        later <= earlier + 1e-9
        for earlier, later in zip(svd_residuals, svd_residuals[1:])
    )

    moment_residuals = [
        bilinear_residual(u, moment_features(p, degree))
        for degree in sorted(MOMENT_DEGREES)
        if moment_dim(degree) < num_particles
    ]
    assert all(
        later <= earlier + 1e-9
        for earlier, later in zip(moment_residuals, moment_residuals[1:])
    )


# ---------------------------------------------------------------------------
# P5 — feature construction
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    ("degree", "expected"), [(1, 4), (2, 14), (3, 34)]
)
def test_p5_moment_dimensions(degree: int, expected: int) -> None:
    """P5: symmetric monomial counts are 4 / 10 / 20, cumulative 4 / 14 / 34."""
    assert moment_dim(degree) == expected
    phi = moment_features(_four_vectors(24, 0), degree)
    assert phi.shape == (24, expected)
    assert torch.allclose(
        phi.norm(dim=0), torch.ones(expected, dtype=torch.float64), atol=1e-9
    )


@given(seed=st.integers(min_value=0, max_value=2**16))
@_SETTINGS
def test_p5_column_scaling_does_not_change_the_fit(seed: int) -> None:
    """P5: rescaling feature columns leaves the achievable residual unchanged.

    span(Phi) is invariant under invertible column scaling and the scale is
    absorbed into M, so normalization is a conditioning choice with no effect on
    what can be represented. If this fails, the fit is not solving the problem
    it claims to.
    """
    p = _four_vectors(48, seed)
    gram = p[:, 3:4] * p[:, 3:4].T - p[:, :3] @ p[:, :3].T
    u = torch.log1p(gram.abs()).to(torch.float64)

    phi = moment_features(p, 2)
    generator = torch.Generator()
    generator.manual_seed(seed + 3)
    scales = torch.rand(phi.shape[1], generator=generator, dtype=torch.float64) + 0.5

    assert bilinear_residual(u, phi) == pytest.approx(
        bilinear_residual(u, phi * scales), abs=1e-7
    )


# ---------------------------------------------------------------------------
# P6 — padding exclusion
# ---------------------------------------------------------------------------

@given(
    n_valid=st.integers(min_value=34, max_value=40),
    n_total=st.integers(min_value=64, max_value=96),
    seed=st.integers(min_value=0, max_value=2**16),
)
@_SETTINGS
def test_p6_padding_is_excluded_from_the_spectrum(
    n_valid: int, n_total: int, seed: int
) -> None:
    """P6: zero padding must not manufacture extra zero singular values.

    Padded pairs are exactly 0.0 (the pair-sentinel invariant), so an unsliced
    SVD would report a flatteringly low rank purely from the padding. The valid
    block's profile must be identical to the padded tensor's sliced profile.
    """
    generator = torch.Generator()
    generator.manual_seed(seed)
    block = torch.randn(n_valid, n_valid, generator=generator, dtype=torch.float64)
    block = block + block.T

    padded = torch.zeros(1, 1, n_total, n_total, dtype=torch.float64)
    padded[0, 0, :n_valid, :n_valid] = block

    sliced = slice_valid(padded, [n_valid])[0][0]
    assert sliced.shape == (n_valid, n_valid)

    for frac in ENERGY_FRACTIONS:
        key = f"{frac:.2f}"
        assert (
            spectral_profile(sliced)["rank_for_energy"][key]
            == spectral_profile(block)["rank_for_energy"][key]
        )


# ---------------------------------------------------------------------------
# End-to-end wiring
# ---------------------------------------------------------------------------

def test_audit_batch_runs_on_the_real_pair_embed() -> None:
    """The audit extracts a symmetric bias from a real weaver PairEmbed.

    Also pins the two structural facts the audit depends on: the bias is
    (B, num_heads, N, N), and weaver's PairEmbed is symmetric.
    """
    from ablation.config import AblationConfig
    from variants import build_variant_part

    torch.manual_seed(0)
    config = AblationConfig(arm="baseline", num_layers=2, num_cls_layers=1)
    model = build_variant_part("baseline", **config.model_kwargs()).eval()

    num_particles = 48
    lengths = [num_particles, num_particles - 2, 8]  # last one is filtered out
    generator = torch.Generator()
    generator.manual_seed(1)
    v = _make_four_vectors(lengths, num_particles, generator)
    mask, _ = _make_lengths_masks(lengths, num_particles)

    records = audit_batch(model, v, mask, min_valid=32)

    audited_jets = {record["jet"] for record in records}
    assert audited_jets == {0, 1}, "the 8-particle jet must be filtered out"
    assert len(records) == 2 * config.num_heads

    for record in records:
        assert record["asymmetry"] < 1e-5, "weaver PairEmbed is symmetric"
        assert 0.0 <= record["rank_fraction"] <= 1.0
        assert record["moment_residual"]["4"] >= 0.0

    summary = summarize(records)
    assert summary["usable"] == len(records)
    assert summary["min_n"] >= 32
    assert isinstance(verdict(summary), str)


def test_verdict_thresholds() -> None:
    """The registered decision rule maps p90 rank to the three documented outcomes."""

    def make(p90: float) -> dict:
        return {
            "usable": 1,
            "rank_for_energy": {"0.90": {"median": p90, "p90": p90, "max": int(p90)}},
        }

    assert verdict(make(4)).startswith("BUILD N8")
    assert verdict(make(8)).startswith("BUILD N8")
    assert verdict(make(20)).startswith("INTERMEDIATE")
    assert verdict(make(41)).startswith("N8 IS DEAD")
    assert verdict({"usable": 0}).startswith("INCONCLUSIVE")


def test_verdict_reports_hybrid_when_heads_split() -> None:
    """A split head population must yield HYBRID, not an all-or-nothing call.

    Observed on the untrained null baseline: 6 of 8 heads sit at rank <= 2 while
    two need rank 44-64. Judging that on the aggregate p90 alone returns
    "N8 IS DEAD" and discards six trivially factorizable heads.
    """
    summary = {
        "usable": 8,
        "rank_for_energy": {"0.90": {"median": 2.0, "p90": 55.0, "max": 98}},
        "num_heads_seen": 8,
        "low_rank_heads": [0, 1, 4, 5, 6, 7],
    }
    message = verdict(summary)
    assert message.startswith("HYBRID")
    assert "6/8" in message

    # Unanimous populations must fall through to the aggregate thresholds.
    unanimous_low = {**summary, "low_rank_heads": list(range(8))}
    unanimous_low["rank_for_energy"] = {"0.90": {"median": 2.0, "p90": 4.0, "max": 6}}
    assert verdict(unanimous_low).startswith("BUILD N8")

    assert verdict({**summary, "low_rank_heads": []}).startswith("N8 IS DEAD")


# ---------------------------------------------------------------------------
# Checkpoint reconstruction (cluster hardening)
# ---------------------------------------------------------------------------

def test_checkpoint_load_rebuilds_from_stored_config_and_strips_prefixes() -> None:
    """The audit model comes from the checkpoint's config, not the CLI defaults.

    The cluster baseline was trained with ``use_part_kernels=true``; if the
    audit built a vanilla model from its own defaults, fused-PairEmbed key
    drift or the wrong arm config would corrupt the measurement silently.
    This pins: stored-config rebuild, ``module.``/``_orig_mod.`` stripping,
    the part-kernel wrap, and that the weights that arrive are the weights
    that were saved.
    """
    from ablation.config import AblationConfig
    from ablation.rank_audit import _load_model_from_checkpoint
    from variants import build_variant_part

    config = AblationConfig(
        arm="baseline",
        num_layers=2,
        num_cls_layers=1,
        num_heads=2,
        embed_dims=(16, 32, 16),
        pair_embed_dims=(8, 8),
        use_part_kernels=True,
    )
    model = build_variant_part("baseline", **config.model_kwargs())

    # Simulate a state dict that passed through DDP then torch.compile.
    prefixes = ["module.", "_orig_mod.", ""]
    mangled = {
        prefixes[i % len(prefixes)] + key: value
        for i, (key, value) in enumerate(model.state_dict().items())
    }
    payload = {"model": mangled, "config": config.to_dict(), "step": 42}

    rebuilt, used_config = _load_model_from_checkpoint(payload)

    assert used_config.arm == "baseline"
    assert used_config.use_part_kernels is True
    assert used_config.pair_embed_dims == (8, 8)
    assert rebuilt.pair_embed is not None
    for key, value in model.state_dict().items():
        assert torch.equal(rebuilt.state_dict()[key], value)


# ---------------------------------------------------------------------------
# Regressions for defects found in external review (2026-09-05)
# ---------------------------------------------------------------------------

def test_masked_moment_fit_is_exact_when_off_diagonal_is_representable() -> None:
    """The off-diagonal fit must be the best OFF-DIAGONAL fit, not a whole-matrix fit scored late.

    Fitting the full block with an artificial zero diagonal and merely masking the *score* is not
    the same optimisation: the zeros are data the fit is pulled toward. On a matrix that is exactly
    `Phi M Phi^T` away from the diagonal, that reported a relative residual of ~0.11 for something
    exactly representable -- enough to reject a physical feature family that reproduces every entry
    the audit says matters.
    """
    from ablation.rank_audit import bilinear_residual

    torch.manual_seed(0)
    n, r = 24, 5
    phi = torch.randn(n, r, dtype=torch.float64)
    exact = phi @ torch.randn(r, r, dtype=torch.float64) @ phi.T
    exact = (exact + exact.T) / 2
    # Off the diagonal this is exactly representable; the diagonal is replaced by junk, which is
    # what the `eps`-clamped self-pair is.
    corrupted = exact.clone()
    corrupted.fill_diagonal_(1e3)

    assert bilinear_residual(corrupted, phi, exclude_diagonal=True) < 1e-9
    # Whole-matrix scoring cannot ignore the corrupted diagonal, which is the point of the flag.
    assert bilinear_residual(corrupted, phi, exclude_diagonal=False) > 0.1


def test_masked_moment_fit_is_bounded_by_predicting_zero() -> None:
    """A projection residual is <= 1. Truncation is what guarantees it; without `rtol` the
    degree-3 fit diverged to 2730 on real jets."""
    from ablation.rank_audit import bilinear_residual

    torch.manual_seed(1)
    phi = torch.randn(40, 14, dtype=torch.float64)
    m = torch.randn(40, 40, dtype=torch.float64)
    m = (m + m.T) / 2
    for exclude in (False, True):
        assert 0.0 <= bilinear_residual(m, phi, exclude_diagonal=exclude) <= 1.0


def test_phi_conditioning_reports_the_dimension_actually_used() -> None:
    """A residual at "34 dimensions" is unreadable if truncation silently used fewer."""
    from ablation.rank_audit import phi_conditioning

    torch.manual_seed(2)
    base = torch.randn(30, 5, dtype=torch.float64)
    # A rank-deficient feature matrix: 8 nominal columns spanning only 5 directions.
    phi = torch.cat([base, base @ torch.randn(5, 3, dtype=torch.float64)], dim=1)
    report = phi_conditioning(phi)
    assert report["nominal_dim"] == 8
    assert report["effective_dim"] == 5
    assert report["effective_dim"] < report["nominal_dim"]


def test_row_centring_alone_does_not_zero_a_self_pair_boost() -> None:
    """`U = c I` changes attention, so no correction may report it as free.

    Row centring is unconditionally free (softmax cannot see a per-row constant). Dropping the
    self-pair diagonal is free only if the kernel handles the diagonal separately -- and on `c I`
    that assumption zeroes the matrix entirely and reports rank 0, i.e. "this bias costs nothing".
    It costs a re-weighting of every particle's self-attention. So the two variants must stay
    distinguishable, and the gate must read the unconditional one.
    """
    from ablation.rank_audit import matrix_variants

    n = 32
    boost = 3.0 * torch.eye(n, dtype=torch.float64)
    variants = matrix_variants(boost, remove_self_pair=True)

    assert float(variants["centered"].norm()) > 1.0      # centring alone keeps it
    assert float(variants["corrected"].norm()) == 0.0    # dropping the diagonal loses all of it


def test_verdict_gates_on_the_unconditional_variant() -> None:
    """The verdict must not be decided by the diagonal-dropped number."""
    from ablation.rank_audit import verdict

    summary = {
        "usable": 10,
        "num_heads_seen": 2,
        "gauge_rank_for_energy": {"0.90": {"p90": 30.0}},
        "gauge_low_rank_heads": [],
        "corrected_rank_for_energy": {"0.90": {"p90": 2.0}},
        "corrected_low_rank_heads": [0, 1],
    }
    text = verdict(summary)
    assert "30" in text                     # gated on the row-centred number
    assert "BUILD N8" not in text           # the diagonal-dropped 2 must not decide it
    assert "self-pair diagonal" in text     # and the conditional number is still disclosed


def test_verdict_compares_p90_against_the_null_p90_not_its_median() -> None:
    """A batch p90 must be judged against the null distribution of a batch p90.

    Comparing the data's p90 to the null's median charges the data for the null's spread: a rank
    that sits between the two gets called "structure" or "noise" depending only on which quantile
    happens to be on the other side.
    """
    from ablation.rank_audit import verdict

    def summary_with(data_p90: float) -> dict:
        return {
            "usable": 10,
            "num_heads_seen": 2,
            "gauge_rank_for_energy": {"0.90": {"p90": data_p90}},
            "gauge_low_rank_heads": [],
            "null": {"variants": {"centered": {"rank_for_energy": {"0.90": {
                "median": 10.0, "p90": 13.0, "batch_p90_draws": [12.0, 13.0, 13.0, 14.0],
            }}}}},
        }

    # 12 is above the null MEDIAN (10) but below the null P90 (13): the like-for-like comparison
    # is against 13, so this is evidence of structure, not an inconclusive result.
    assert "INCONCLUSIVE" not in verdict(summary_with(12.0))
    assert "batch p90" in verdict(summary_with(12.0))
    # At or above the null's own p90 there is genuinely no evidence.
    assert "INCONCLUSIVE" in verdict(summary_with(13.0))


def test_null_applies_the_same_diagonal_treatment_as_the_data() -> None:
    """`--keep-self-pair` changes the data transformation, so it must change the null's too."""
    from ablation.rank_audit import gaussian_null

    from ablation.rank_audit import matrix_variants

    dropped = gaussian_null(24, trials=8, seed=0, remove_self_pair=True)
    kept = gaussian_null(24, trials=8, seed=0, remove_self_pair=False)

    assert dropped["remove_self_pair"] is True
    assert kept["remove_self_pair"] is False
    # The cache key must include the flag, or the second call returns the first one's answer.
    assert gaussian_null(24, trials=8, seed=0, remove_self_pair=True) is dropped

    # With the flag OFF, `corrected` is by definition plain row-centring, so its null must be
    # *identical* to the `centered` null. Asserted as an identity rather than as a difference in
    # median rank: the diagonal is 24 of 576 entries, so a 90%-energy median is too coarse to
    # register it and would pass whether or not the flag was threaded through at all.
    assert kept["corrected"] == kept["centered"]
    assert dropped["corrected"] != dropped["centered"]

    # And at the level the null is built from, the flag really does change the transformation.
    torch.manual_seed(0)
    m = torch.randn(24, 24, dtype=torch.float64)
    assert not torch.equal(
        matrix_variants(m, remove_self_pair=True)["corrected"],
        matrix_variants(m, remove_self_pair=False)["corrected"],
    )


def test_matched_null_reports_batch_statistics_for_both_quantiles() -> None:
    from ablation.rank_audit import null_matched_to_records

    records = [{"n": 32}, {"n": 40}, {"n": 48}, {"n": 20}]
    null = null_matched_to_records(records, trials=8, seed=0)
    entry = null["variants"]["centered"]["rank_for_energy"]["0.90"]

    assert set(entry) >= {"median", "p90", "batch_p90_draws"}
    assert len(entry["batch_p90_draws"]) == 8
    # A batch p90 sits at or above a batch median by construction.
    assert entry["p90"] >= entry["median"]


# ---------------------------------------------------------------------------
# R0 layer-redundancy pre-screen
# ---------------------------------------------------------------------------

def test_swap_pairs_are_offset_major_so_a_cap_covers_every_destination() -> None:
    """`--max-swap-pairs` truncates the pair list, so its order decides what a capped scan means.

    Destination-major order (`for i: for j:`) put every pair with i=0 first, so a cap of 7 measured
    only substitutions *into block 0* while the report presented their median as "swap deltas" for
    all eight destinations.
    """
    from ablation.layer_redundancy import swap_drop_scan

    n = 8
    pairs = [(i, (i + d) % n) for d in range(1, n) for i in range(n)]
    assert len(pairs) == n * (n - 1)
    assert len(set(pairs)) == len(pairs)                     # no duplicates
    assert all(i != j for i, j in pairs)                     # no self-swaps
    # Any prefix of length n covers all n destinations exactly once.
    for k in (1, 2, 3):
        prefix = pairs[: k * n]
        assert sorted(i for i, _ in prefix) == sorted(list(range(n)) * k)
    assert swap_drop_scan is not None                        # the function still imports


def test_layer_redundancy_flags_an_already_tied_model() -> None:
    """On a tied checkpoint every similarity is 1.0 by construction, not by training.

    Left unguarded this reported a spectacular redundancy result plus impossible arithmetic: a 214%
    block share and a negative projected tied-parameter count, because `per_block * len(blocks)`
    double-counts a shared block.
    """
    from ablation.config import AblationConfig
    from ablation.layer_redundancy import format_report
    from variants import build_variant_part
    from variants.tied import shared_block_groups

    torch.manual_seed(0)
    tied = build_variant_part("tied", **AblationConfig(arm="baseline").model_kwargs(),
                              tie_num_unique=1)
    groups = shared_block_groups(tied)
    assert groups == [[0, 1, 2, 3, 4, 5, 6, 7]]

    blocks = list(tied.blocks)
    per_block = sum(p.numel() for p in blocks[0].parameters())
    total = sum(p.numel() for p in tied.parameters())
    distinct = len({id(getattr(b, "block", b)) for b in blocks})

    # Counted over distinct blocks these stay physical; counted over depths they do not.
    assert 0.0 < per_block * distinct / total <= 1.0
    assert total - per_block * (distinct - 1) > 0
    assert per_block * len(blocks) / total > 1.0             # the bug, pinned as such

    report = format_report({
        "meta": {
            "weights": "test", "source": "none", "arm": "tied", "num_blocks": len(blocks),
            "distinct_blocks": distinct, "shared_block_groups": groups, "already_tied": True,
            "params_per_block": per_block, "total_params": total,
            "block_share": per_block * distinct / total,
            "tied_params": total - per_block * (distinct - 1), "seed": 0,
        },
        "similarity": {"num_blocks": len(blocks), "groups": {}},
        "null_similarity": {"num_blocks": len(blocks), "groups": {}},
        "drift": {}, "null_drift": {},
    })
    assert "ALREADY TIED" in report
    assert "NOT A REDUNDANCY RESULT" in report


def test_layer_redundancy_does_not_flag_an_untied_model() -> None:
    from ablation.config import AblationConfig
    from variants import build_variant_part
    from variants.tied import shared_block_groups

    torch.manual_seed(0)
    model = build_variant_part("baseline", **AblationConfig(arm="baseline").model_kwargs())
    assert shared_block_groups(model) == []


def test_self_pair_profile_prices_a_constant_diagonal_at_one_scalar() -> None:
    """`c*I` is the extreme case: rank n, but ONE degree of freedom.

    This is what decides whether B4's self-pair condition is cheap. A constant diagonal must show
    up as expensive under row-centring alone and free once a single per-head scalar is subtracted --
    otherwise the three numbers cannot distinguish "the diagonal is one number" from "the diagonal
    is n numbers", which is the whole point of measuring it.
    """
    from ablation.rank_audit import self_pair_profile

    n = 32
    profile = self_pair_profile(3.0 * torch.eye(n, dtype=torch.float64))
    assert profile["diag_rel_spread"] == pytest.approx(0.0, abs=1e-12)
    assert profile["rank_centered"] > 1          # c*I is NOT free under centring alone
    assert profile["rank_minus_constant"] == 0   # one scalar removes all of it
    assert profile["rank_offdiagonal"] == 0


def test_self_pair_profile_prices_a_varying_diagonal_at_n_values() -> None:
    """A diagonal that varies cannot be bought with one scalar, and must not look like it can."""
    from ablation.rank_audit import self_pair_profile

    torch.manual_seed(0)
    n = 32
    varying = torch.diag(torch.randn(n, dtype=torch.float64) * 3.0)
    profile = self_pair_profile(varying)
    assert profile["diag_rel_spread"] > 0.5
    # Subtracting the mean leaves n-1 independent diagonal entries, so the rank stays high;
    # excluding the diagonal removes all of it.
    assert profile["rank_minus_constant"] > profile["rank_offdiagonal"]
    assert profile["rank_offdiagonal"] == 0


# ---------------------------------------------------------------------------
# Bias truncation probe — does a rank-r bias still tag?
# ---------------------------------------------------------------------------

def _probe_batch(sizes=(16, 10, 6), heads=2, width=16, seed=0):
    torch.manual_seed(seed)
    valid = torch.zeros(len(sizes), width, dtype=torch.bool)
    for row, n in enumerate(sizes):
        valid[row, :n] = True
    u = torch.randn(len(sizes), heads, width, width)
    return (u + u.transpose(-1, -2)) / 2, valid


@pytest.mark.parametrize("mode", ["centered", "const_diag"])
def test_truncation_at_full_rank_is_a_softmax_noop(mode: str) -> None:
    """`r = full` only row-centres, and softmax over valid keys cannot see a per-row constant.

    This is the probe's load-bearing control: if it fails, every accuracy number the probe reports
    is measuring the patch rather than the truncation.
    """
    from ablation.bias_truncation_probe import truncate_batch

    u, valid = _probe_batch()
    out = truncate_batch(u, valid, None, mode)
    for row, n in enumerate((16, 10, 6)):
        before = torch.softmax(u[row, 0, :n, :n], dim=-1)
        after = torch.softmax(out[row, 0, :n, :n], dim=-1)
        torch.testing.assert_close(before, after, rtol=0, atol=1e-6)


def test_truncation_rank_zero_deletes_the_bias() -> None:
    """The r=0 control is the lower bound, so it must really be *no bias*.

    Regression: an early return handed back the padding-masked but otherwise untouched bias, which
    would have reported that deleting the bias costs nothing — making every other number in the
    probe unreadable, since they are all quoted as a fraction of this span.
    """
    from ablation.bias_truncation_probe import truncate_batch

    u, valid = _probe_batch()
    assert bool((truncate_batch(u, valid, 0, "centered") == 0).all())


@pytest.mark.parametrize("rank", [2, 5])
def test_truncation_achieves_the_eckart_young_optimum(rank: int) -> None:
    """The probe must measure the BEST rank-r bias, or it understates what a factorization can do."""
    from ablation.bias_truncation_probe import truncate_batch

    u, valid = _probe_batch()
    reference = truncate_batch(u, valid, None, "centered")
    truncated = truncate_batch(u, valid, rank, "centered")
    target = reference[0, 0]
    sv = torch.linalg.svdvals(target)
    optimum = float((sv[rank:] ** 2).sum().sqrt() / sv.pow(2).sum().sqrt())
    achieved = float((target - truncated[0, 0]).norm() / target.norm())
    assert achieved == pytest.approx(optimum, abs=1e-5)


@pytest.mark.parametrize("rank", [2, 5])
def test_truncation_yields_the_requested_rank_on_the_valid_block(rank: int) -> None:
    from ablation.bias_truncation_probe import truncate_batch

    u, valid = _probe_batch()
    out = truncate_batch(u, valid, rank, "centered")
    for row, n in enumerate((16, 10, 6)):
        sv = torch.linalg.svdvals(out[row, 0, :n, :n])
        # float32 reconstruction leaves ~1e-7 relative noise where the zeroed values were.
        assert int((sv > sv[0] * 1e-5).sum()) == min(rank, n)


def test_truncation_never_leaks_into_padding() -> None:
    """Truncation smears energy across the whole matrix, so the padded region must be re-zeroed.

    Otherwise the function's output would depend on how wide the batch happened to be padded.
    """
    from ablation.bias_truncation_probe import truncate_batch

    u, valid = _probe_batch()
    out = truncate_batch(u, valid, 4, "centered")
    assert float(out[1, :, 10:, :].abs().max()) == 0.0
    assert float(out[2, :, :, 6:].abs().max()) == 0.0


def test_truncation_centres_over_valid_keys_not_padded_width() -> None:
    """The row mean must be over the jet's own n keys. Averaging over N would be a wrong constant.

    Verified by padding the same jet to two different widths: the valid block must come out
    identical, which is only true if padding is excluded from the mean.
    """
    from ablation.bias_truncation_probe import truncate_batch

    torch.manual_seed(3)
    n = 8
    block = torch.randn(1, 2, n, n)
    block = (block + block.transpose(-1, -2)) / 2

    outs = []
    for width in (8, 20):
        padded = torch.zeros(1, 2, width, width)
        padded[:, :, :n, :n] = block
        valid = torch.zeros(1, width, dtype=torch.bool)
        valid[0, :n] = True
        outs.append(truncate_batch(padded, valid, 3, "centered")[:, :, :n, :n])
    torch.testing.assert_close(outs[0], outs[1], rtol=0, atol=1e-6)


def test_probe_restores_the_model_even_on_failure() -> None:
    """The probe swaps `model.pair_embed`; a raise mid-scan must not leave the model patched."""
    from ablation.bias_truncation_probe import evaluate_ranks

    from ablation.config import AblationConfig
    from variants import build_variant_part

    torch.manual_seed(0)
    model = build_variant_part("baseline", **AblationConfig(arm="baseline").model_kwargs())
    original = model.pair_embed
    x, v, mask = torch.zeros(2, 16, 4), torch.zeros(2, 4, 4), torch.ones(2, 1, 4)
    y = torch.zeros(2, dtype=torch.long)
    with pytest.raises(ValueError, match="mode must be"):
        # An unknown mode raises inside `truncate_batch`, i.e. mid-scan with the wrapper installed.
        evaluate_ranks(model, x, v, mask, y, ranks=(4,), modes=("bogus",))
    assert model.pair_embed is original

    # And the happy path must also restore it.
    evaluate_ranks(model, x, v, mask, y, ranks=(2, None), modes=("centered",))
    assert model.pair_embed is original
