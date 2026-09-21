"""Property tests for the N8 K6 factorized pair bias.

Pins the identities the K6 screen is betting on.  A wiring bug still produces
finite logits and trains; these tests are what would catch it before a 12h job.

1. K1 identity — zero content Q/K → logits equal ``λ d_ij / (E_i E_j)``.
2. Minkowski ``d_ij`` is invariant under a Lorentz boost of ``v``.
3. Integer-harmonic rotary ``R(k φ)`` is ``2π``-periodic; wrap jets must not
   blow up extra-logit scale vs collimated jets.
4. Tensor product is not additive in ``Δy`` and ``Δφ``.
5. Wrapper builds no PairEmbed and a ragged forward is finite.
6. ``n8_k6.yaml`` is the 200k rotary+Minkowski screen.
"""

from __future__ import annotations

import math
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

from ablation.config import load_config
from variants import N8ParT, build_variant_part
from variants.n8.attention import FactorizedAttentionBlock
from variants.n8.features import (
    azimuth,
    minkowski_angle_pair_matrix,
    minkowski_pair_matrix,
    minkowski_qk_extras,
    relative_rapidity,
    rotation_2d,
    rotary_qk_extras,
    tensor_product_basis,
    tensor_product_matrix,
)
from variants.tests.strategies import _make_four_vectors, _make_lengths_masks

_REPO_ROOT = Path(__file__).resolve().parents[2]
_K6_CONFIG = _REPO_ROOT / "ablation" / "configs" / "n8_k6.yaml"

_TINY = dict(
    input_dim=16,
    num_classes=10,
    embed_dims=(16, 32, 16),
    num_heads=2,
    num_layers=2,
    num_cls_layers=1,
)

ATOL = 1e-5


def _synthetic_batch(num_particles: int, lengths: list[int], seed: int = 0):
    generator = torch.Generator()
    generator.manual_seed(seed)
    batch = len(lengths)
    x = torch.randn(batch, _TINY["input_dim"], num_particles, generator=generator)
    v = _make_four_vectors(lengths, num_particles, generator)
    mask, _ = _make_lengths_masks(lengths, num_particles)
    return x, v, mask


def _zero_content(block: FactorizedAttentionBlock) -> None:
    with torch.no_grad():
        block.q_proj.weight.zero_()
        block.q_proj.bias.zero_()
        block.k_proj.weight.zero_()
        block.k_proj.bias.zero_()


def _synthetic_500gev_jet(
    num_particles: int = 40,
    *,
    seed: int = 0,
    phi_center: float = 0.2,
    phi_spread: float = 0.15,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Collimated 500 GeV jet in weaver ``(1, 4, P)`` layout."""
    torch.manual_seed(seed)
    frac = torch.distributions.Dirichlet(torch.ones(num_particles) * 0.5).sample()
    pt = frac * 500.0
    eta = 0.3 + 0.15 * torch.randn(num_particles)
    phi = phi_center + phi_spread * torch.randn(num_particles)
    px, py = pt * torch.cos(phi), pt * torch.sin(phi)
    pz = pt * torch.sinh(eta)
    energy = torch.sqrt(px**2 + py**2 + pz**2 + 0.14**2)
    v = torch.stack([px, py, pz, energy]).unsqueeze(0)
    mask = torch.ones(1, 1, num_particles)
    padding_mask = torch.zeros(1, num_particles, dtype=torch.bool)
    return v, mask, padding_mask


def _wrap_straddle_jet(
    num_particles: int = 40,
    *,
    seed: int = 0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Jet with ``φ ≈ wrap(3.0 + 0.4 randn)`` — particles straddle ``±π``."""
    torch.manual_seed(seed)
    frac = torch.distributions.Dirichlet(torch.ones(num_particles) * 0.5).sample()
    pt = frac * 500.0
    eta = 0.3 + 0.15 * torch.randn(num_particles)
    phi = 3.0 + 0.4 * torch.randn(num_particles)
    phi = (phi + math.pi) % (2.0 * math.pi) - math.pi
    px, py = pt * torch.cos(phi), pt * torch.sin(phi)
    pz = pt * torch.sinh(eta)
    energy = torch.sqrt(px**2 + py**2 + pz**2 + 0.14**2)
    v = torch.stack([px, py, pz, energy]).unsqueeze(0)
    mask = torch.ones(1, 1, num_particles)
    padding_mask = torch.zeros(1, num_particles, dtype=torch.bool)
    return v, mask, padding_mask


def _k6_block() -> FactorizedAttentionBlock:
    return FactorizedAttentionBlock(
        embed_dim=128,
        num_heads=8,
        minkowski=True,
        rotary=True,
        rotary_pairs=4,
        degree=1,
    )


# ---------------------------------------------------------------------------
# 1. K1 identity
# ---------------------------------------------------------------------------

def test_k1_identity_zero_content_logits_are_lambda_dij() -> None:
    """With content Q/K zeroed and rotary off, scores are ``λ_h d_ij / (E_i E_j)``."""
    torch.manual_seed(0)
    block = FactorizedAttentionBlock(
        embed_dim=32,
        num_heads=4,
        minkowski=True,
        rotary=False,
        degree=1,
    )
    _zero_content(block)

    generator = torch.Generator().manual_seed(1)
    lengths = [12, 9]
    v = _make_four_vectors(lengths, 12, generator)
    mask, padding_mask = _make_lengths_masks(lengths, 12)
    x = torch.randn(2, 12, 32, generator=generator)

    block.set_momenta(v, mask)
    logits = block.attention_logits(x, padding_mask)

    angle = minkowski_angle_pair_matrix(v)
    lam = block.minkowski_scale().detach()
    expected = lam.view(1, -1, 1, 1) * angle.unsqueeze(1)

    pad_keys = padding_mask.bool()[:, None, None, :]
    assert torch.allclose(
        logits.masked_fill(pad_keys, 0.0),
        expected.masked_fill(pad_keys, 0.0),
        atol=ATOL,
        rtol=1e-4,
    )
    assert torch.all(logits[pad_keys.expand_as(logits)] == -1e9)


def test_minkowski_qk_dot_is_dij() -> None:
    """The fold ``Q=(p,E), K=(-p,E)`` is exactly ``d_ij`` before ``λ`` and ``m_J``."""
    generator = torch.Generator().manual_seed(2)
    v = _make_four_vectors([16], 16, generator).double()
    q, k = minkowski_qk_extras(v, normalize=False)
    got = torch.matmul(q, k.transpose(-2, -1))
    assert torch.allclose(got, minkowski_pair_matrix(v), atol=1e-10, rtol=1e-8)


def test_minkowski_qk_dot_is_angle_kernel() -> None:
    """Default extras are the opening-angle kernel ``d_ij / (E_i E_j)``."""
    generator = torch.Generator().manual_seed(2)
    lengths = [16, 11]
    v = _make_four_vectors(lengths, 16, generator).double()
    mask, _ = _make_lengths_masks(lengths, 16)
    q, k = minkowski_qk_extras(v, mask)
    got = torch.matmul(q, k.transpose(-2, -1))
    assert torch.allclose(
        got, minkowski_angle_pair_matrix(v), atol=1e-10, rtol=1e-8
    )


# ---------------------------------------------------------------------------
# 2. Lorentz invariance of d_ij
# ---------------------------------------------------------------------------

def _boost_z(v: torch.Tensor, beta: float) -> torch.Tensor:
    """Boost weaver ``(B, 4, P)`` along z.  ``|beta| < 1``."""
    gamma = 1.0 / math.sqrt(1.0 - beta * beta)
    px, py, pz, energy = v.unbind(dim=1)
    energy2 = gamma * energy - gamma * beta * pz
    pz2 = -gamma * beta * energy + gamma * pz
    return torch.stack([px, py, pz2, energy2], dim=1)


@pytest.mark.parametrize("beta", [-0.6, -0.2, 0.3, 0.8])
def test_minkowski_dij_invariant_under_boost(beta: float) -> None:
    generator = torch.Generator().manual_seed(3)
    v = _make_four_vectors([20, 11], 20, generator).double()
    d0 = minkowski_pair_matrix(v)
    d1 = minkowski_pair_matrix(_boost_z(v, beta))
    assert torch.allclose(d0, d1, atol=1e-10, rtol=1e-8)


def test_angle_kernel_extras_are_o_one_on_collimated_jet() -> None:
    """``d_ij / (E_i E_j)`` stays O(10⁻²–10⁻¹) on a 500 GeV collimated jet."""
    v, mask, _ = _synthetic_500gev_jet()
    q, k = minkowski_qk_extras(v, mask)
    scores = torch.matmul(q, k.transpose(-2, -1))
    assert scores.abs().max().item() < 1.0
    assert scores.abs().median().item() < 0.2


def test_jetclass_scale_minkowski_logits_are_not_saturating() -> None:
    """Raw GeV ``d_ij`` would drown content; the angle-kernel extras must not.

    A collimated 500 GeV jet is the production scale.  Extra Minkowski logits
    (content Q/K zeroed, rotary off) have to stay O(1), not O(10³).
    """
    v, mask, padding_mask = _synthetic_500gev_jet()

    block = FactorizedAttentionBlock(
        embed_dim=128, num_heads=8, minkowski=True, rotary=False, degree=1
    )
    _zero_content(block)
    block.set_momenta(v, mask)
    logits = block.attention_logits(torch.zeros(1, v.shape[-1], 128), padding_mask)
    extra = logits.masked_fill(padding_mask[:, None, None, :], 0.0)
    assert extra.abs().max().item() < 5.0
    assert extra.abs().median().item() < 1.0


# ---------------------------------------------------------------------------
# 3 / 4. Rotary identities
# ---------------------------------------------------------------------------

def test_rotation_product_depends_only_on_delta_phi() -> None:
    """``R(φ_i)^T R(φ_j) = R(φ_j - φ_i)``, independent of rapidity."""
    phi = torch.tensor([0.3, -1.1, 2.4, 0.0])
    u = torch.tensor([0.9, -0.4, 1.7, 0.2])  # unused; must not enter R
    del u
    rotation = rotation_2d(phi)
    product = torch.matmul(
        rotation.unsqueeze(1).transpose(-1, -2), rotation.unsqueeze(0)
    )
    expected = rotation_2d(phi.unsqueeze(0) - phi.unsqueeze(1))
    assert torch.allclose(product, expected, atol=ATOL)

    phi_shift = phi + 0.7
    product_shift = torch.matmul(
        rotation_2d(phi_shift).unsqueeze(1).transpose(-1, -2),
        rotation_2d(phi_shift).unsqueeze(0),
    )
    assert torch.allclose(product, product_shift, atol=ATOL)


def test_rapidity_rotation_depends_only_on_delta_y() -> None:
    """``M(u_i)^T M(u_j) = M(u_j - u_i)``, independent of azimuth."""
    u = torch.tensor([0.2, -0.8, 1.1, 0.0])
    rapidity = rotation_2d(u)
    product = torch.matmul(
        rapidity.unsqueeze(1).transpose(-1, -2), rapidity.unsqueeze(0)
    )
    expected = rotation_2d(u.unsqueeze(0) - u.unsqueeze(1))
    assert torch.allclose(product, expected, atol=ATOL)


def test_tensor_product_is_not_additive_in_delta_y_and_delta_phi() -> None:
    """``(R⊗M)_i · (R⊗M)_j = cos(Δφ) cos(Δy)``, not ``cos(Δφ) + cos(Δy)``.

    Additive 2D RoPEs would give the sum; ``ln ΔR`` is nonlinear in the two
    angles, which is why K6 uses the Kronecker product.
    """
    alpha_i, alpha_j = 0.5, 1.2
    beta_i, beta_j = 0.3, -0.8
    score = (
        tensor_product_basis(
            torch.tensor(alpha_i), torch.tensor(beta_i)
        )
        * tensor_product_basis(
            torch.tensor(alpha_j), torch.tensor(beta_j)
        )
    ).sum()
    d_alpha = alpha_i - alpha_j
    d_beta = beta_i - beta_j
    product = math.cos(d_alpha) * math.cos(d_beta)
    additive = math.cos(d_alpha) + math.cos(d_beta)

    assert score.item() == pytest.approx(product, abs=ATOL)
    assert abs(product - additive) > 0.2

    kron_i = tensor_product_matrix(torch.tensor(alpha_i), torch.tensor(beta_i))
    kron_j = tensor_product_matrix(torch.tensor(alpha_j), torch.tensor(beta_j))
    kron_prod = kron_i.T @ kron_j
    expected_kron = tensor_product_matrix(
        torch.tensor(alpha_j - alpha_i), torch.tensor(beta_j - beta_i)
    )
    assert torch.allclose(kron_prod, expected_kron, atol=ATOL)


def test_rotary_integer_harmonic_is_2pi_periodic() -> None:
    """``φ`` and ``φ + 2π`` must yield identical rotary extra scores."""
    num_heads = 4
    num_pairs = 4
    rotary_a = torch.full((num_heads, num_pairs), 0.1)

    phi = torch.tensor([[0.3, -2.1, 1.4]])
    u = torch.tensor([[0.2, -0.5, 0.8]])

    extras0 = rotary_qk_extras(phi, u, rotary_a, num_pairs=num_pairs)
    extras_shift = rotary_qk_extras(phi + 2.0 * math.pi, u, rotary_a, num_pairs=num_pairs)
    assert torch.allclose(extras0, extras_shift, atol=ATOL, rtol=1e-5)

    scores0 = torch.matmul(extras0, extras0.transpose(-2, -1))
    scores_shift = torch.matmul(extras_shift, extras_shift.transpose(-2, -1))
    assert torch.allclose(scores0, scores_shift, atol=ATOL, rtol=1e-5)


def test_rotary_periodic_under_global_phi_shift() -> None:
    """A collimated jet shifted by a constant ``Δφ`` keeps the same rotary logits."""
    v_col, mask_col, _ = _synthetic_500gev_jet(seed=1)
    shift = 1.7
    px, py = v_col[:, 0], v_col[:, 1]
    phi = torch.atan2(py, px) + shift
    v_shift = v_col.clone()
    v_shift[:, 0], v_shift[:, 1] = torch.cos(phi) * torch.hypot(px, py), torch.sin(phi) * torch.hypot(px, py)

    rotary_a = torch.full((2, 4), 0.1)
    u_col = relative_rapidity(v_col, mask_col)
    u_shift = relative_rapidity(v_shift, mask_col)
    rot_col = rotary_qk_extras(
        azimuth(v_col), u_col, rotary_a, num_pairs=4
    )
    rot_shift = rotary_qk_extras(
        azimuth(v_shift), u_shift, rotary_a, num_pairs=4
    )
    score_col = torch.matmul(rot_col, rot_col.transpose(-2, -1))
    score_shift = torch.matmul(rot_shift, rot_shift.transpose(-2, -1))
    assert torch.allclose(score_col, score_shift, atol=1e-4, rtol=1e-4)


def test_wrap_vs_collimated_extra_logit_scale() -> None:
    """K6 extras on a wrap-straddle jet must not dwarf a collimated jet at init."""
    block = _k6_block()
    _zero_content(block)

    v_col, mask_col, pad_col = _synthetic_500gev_jet(seed=0)
    v_wrap, mask_wrap, pad_wrap = _wrap_straddle_jet(seed=0)

    block.set_momenta(v_col, mask_col)
    logits_col = block.attention_logits(
        torch.zeros(1, v_col.shape[-1], 128), pad_col
    )
    block.set_momenta(v_wrap, mask_wrap)
    logits_wrap = block.attention_logits(
        torch.zeros(1, v_wrap.shape[-1], 128), pad_wrap
    )

    extra_col = logits_col.flatten()
    extra_wrap = logits_wrap.flatten()
    std_col = extra_col.std(unbiased=False).item()
    std_wrap = extra_wrap.std(unbiased=False).item()

    assert extra_col.abs().max().item() < 5.0
    assert extra_wrap.abs().max().item() < 5.0
    assert std_wrap <= max(std_col * 5.0, 1e-6)
    assert std_wrap < 1.0


def test_rotary_amplitude_positive_and_small_at_init() -> None:
    """``λ_rotary`` is softplus-positive; rotary self-scores are O(0.1), not 4."""
    block = FactorizedAttentionBlock(
        embed_dim=128,
        num_heads=8,
        minkowski=False,
        rotary=True,
        rotary_pairs=4,
        degree=1,
    )
    assert torch.all(block.rotary_scale() > 0)
    assert torch.allclose(block.rotary_scale(), F.softplus(block.rotary_lambda_raw))
    assert block.rotary_scale().max().item() <= 0.11

    v, mask, padding_mask = _synthetic_500gev_jet()
    _zero_content(block)
    block.set_momenta(v, mask)
    logits = block.attention_logits(torch.zeros(1, v.shape[-1], 128), padding_mask)
    diag = logits.diagonal(dim1=-2, dim2=-1)
    assert diag.abs().max().item() < 0.6
    assert diag.abs().median().item() < 0.45

    with torch.no_grad():
        block.rotary_lambda_raw.fill_(-8.0)
    assert torch.all(block.rotary_scale() > 0)


def test_n8_lambda_stays_positive() -> None:
    block = FactorizedAttentionBlock(embed_dim=16, num_heads=2, rotary=False)
    with torch.no_grad():
        block.lambda_raw.fill_(-8.0)
    assert torch.all(block.minkowski_scale() > 0)
    assert torch.allclose(
        block.minkowski_scale(), F.softplus(block.lambda_raw)
    )

def test_n8_wrapper_has_no_pair_embed_and_forwards_ragged() -> None:
    torch.manual_seed(0)
    model = build_variant_part(
        "n8",
        pair_embed_dims=(8, 8),
        minkowski=True,
        rotary=True,
        rotary_pairs=4,
        degree=1,
        **_TINY,
    ).eval()

    assert isinstance(model, N8ParT)
    assert model.pair_embed is None
    assert model.part.pair_embed is None
    assert model.n8_blocks[0].extra_dim() == 4 + 16  # Minkowski + 4 rotary pairs

    x, v, mask = _synthetic_batch(24, [24, 17, 8])
    with torch.no_grad():
        logits = model(x, v=v, mask=mask)

    assert logits.shape == (3, _TINY["num_classes"])
    assert torch.isfinite(logits).all()


def test_n8_rejects_missing_four_vectors() -> None:
    torch.manual_seed(0)
    model = build_variant_part("n8", **_TINY).eval()
    x = torch.randn(2, 16, 8)
    mask = torch.ones(2, 1, 8)
    with pytest.raises(ValueError, match="four-vectors"):
        model(x, v=None, mask=mask)


def test_n8_block_rejects_missing_momenta() -> None:
    block = FactorizedAttentionBlock(embed_dim=32, num_heads=4)
    with pytest.raises(RuntimeError, match="without momenta"):
        block(torch.randn(2, 5, 32), torch.zeros(2, 5, dtype=torch.bool))


def test_n8_block_rejects_momenta_token_mismatch() -> None:
    block = FactorizedAttentionBlock(embed_dim=32, num_heads=4)
    block.set_momenta(torch.randn(2, 4, 9), torch.ones(2, 1, 9))
    with pytest.raises(RuntimeError, match="momenta/token length mismatch"):
        block(torch.randn(2, 5, 32), torch.zeros(2, 5, dtype=torch.bool))


def test_n8_cls_blocks_stay_stock_weaver() -> None:
    from weaver.nn.model.ParticleTransformer import Block as WeaverBlock

    model = build_variant_part("n8", **_TINY)
    for cls_block in model.part.cls_blocks:
        assert isinstance(cls_block, WeaverBlock)


def test_degree_two_forwards() -> None:
    """Degree-2 (r=14) is wired behind the flag even though K6 does not submit it."""
    torch.manual_seed(0)
    model = build_variant_part("n8", minkowski=True, rotary=False, degree=2, **_TINY).eval()
    assert model.n8_blocks[0].extra_dim() == 4 + 10
    x, v, mask = _synthetic_batch(16, [16, 7])
    with torch.no_grad():
        logits = model(x, v=v, mask=mask)
    assert torch.isfinite(logits).all()


# ---------------------------------------------------------------------------
# 6. YAML
# ---------------------------------------------------------------------------

def test_n8_k6_yaml_is_the_k6_screen() -> None:
    config = load_config(str(_K6_CONFIG))

    assert config.arm == "n8"
    assert config.run_name == "n8_k6_v2"
    assert config.experiment == "n8_k6_v2"
    assert config.pair_embed_dims is None
    assert config.use_part_kernels is False
    assert config.max_steps == 200_000
    assert config.total_steps == 1_000_000
    assert config.n8_minkowski is True
    assert config.n8_rotary is True
    assert config.n8_rotary_pairs == 4
    assert config.n8_degree == 1
    assert config.num_layers == 8
    assert config.num_heads == 8

    kwargs = config.model_kwargs()
    assert kwargs["minkowski"] is True
    assert kwargs["rotary"] is True
    assert kwargs["rotary_pairs"] == 4
    assert kwargs["degree"] == 1
    assert kwargs["pair_embed_dims"] is None
