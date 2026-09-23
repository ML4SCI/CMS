"""Property tests for U-as-rotation attention (the inverse of K6).

K6 replaced PairEmbed with extras-from-momenta and added those extras as a
scalar logit.  ``urot`` keeps PairEmbed and consumes ``U`` as a pairwise
rotation of content Q/K.  These tests pin the identities a 12h job cannot.

1. ``θ = 0`` (U = 0 or U is None) matches content ``QKᵀ / √d``.
2. A π/2 rotation on a 2-plane mixes the symplectic product, not the dot.
3. ``U`` is not added to the logits (the K6 failure mode in reverse).
4. PairEmbed stays on; YAML is a 200k screen, not K2, not a resume of n8_k6.
"""

from __future__ import annotations

import math
from pathlib import Path

import pytest
import torch

from ablation.config import AblationConfig, load_config
from variants import URotaryAttentionBlock, build_variant_part
from variants.blocks.u_rotary_attention import (
    ANGLE_SCALE_INIT,
    ROPE_BASE,
    ROPE_PHASE_SCALE_INIT,
    apply_rope,
    pairwise_rotary_logits,
    pairwise_rotary_logits_per_plane,
    rotate_half,
    u_to_angles,
    u_to_token_phases,
)
from variants.weaver_adapter import WeaverBlockAdapter

_REPO_ROOT = Path(__file__).resolve().parents[2]
_UROT_CONFIG = _REPO_ROOT / "ablation" / "configs" / "urot.yaml"
_UROT_MF_CONFIG = _REPO_ROOT / "ablation" / "configs" / "urot_mf.yaml"
_UROT_ROPE_CONFIG = _REPO_ROOT / "ablation" / "configs" / "urot_rope.yaml"

_TINY = dict(
    input_dim=16,
    num_classes=10,
    embed_dims=(16, 32, 16),
    pair_embed_dims=(8, 8),
    num_heads=2,
    num_layers=2,
    num_cls_layers=1,
)

ATOL = 1e-5


def test_theta_zero_matches_content_qk() -> None:
    """U = 0 must recover content attention (the nopair kernel)."""
    torch.manual_seed(0)
    B, H, N, D = 2, 2, 5, 8
    query = torch.randn(B, H, N, D)
    key = torch.randn(B, H, N, D)
    theta = torch.zeros(B, H, N, N)
    scale = 1.0 / math.sqrt(D)
    got = pairwise_rotary_logits(query, key, theta, scale)
    want = torch.matmul(query, key.transpose(-2, -1)) * scale
    assert torch.allclose(got, want, atol=ATOL, rtol=1e-5)


def test_pi_over_two_uses_symplectic_cross_not_dot() -> None:
    """On a 2-plane, θ = π/2 replaces Q·K with Q⋆K."""
    query = torch.tensor([[[[1.0, 0.0], [0.0, 1.0]]]])  # (1,1,2,2)
    key = torch.tensor([[[[1.0, 0.0], [0.0, 1.0]]]])
    theta = torch.full((1, 1, 2, 2), math.pi / 2)
    scale = 1.0
    got = pairwise_rotary_logits(query, key, theta, scale)
    # (0,1): q=(1,0), k=(0,1) → cross = 1*1 - 0*0 = 1, dot = 0
    assert got[0, 0, 0, 1].item() == pytest.approx(1.0, abs=ATOL)
    # (0,0): q=k=(1,0) → cross = 0, so π/2 kills the self-dot
    assert got[0, 0, 0, 0].item() == pytest.approx(0.0, abs=ATOL)


def test_u_is_not_added_as_a_scalar() -> None:
    """A huge U must not appear as logits + U (the additive baseline)."""
    torch.manual_seed(1)
    block = URotaryAttentionBlock(
        embed_dim=16, num_heads=2, dropout=0.0, expansion_factor=2
    ).eval()
    x = torch.randn(1, 4, 16)
    U = torch.full((1, 2, 4, 4), 50.0)
    with torch.no_grad():
        rotary = block.attention_logits(x, padding_mask=None, U=U)
        content = block.attention_logits(x, padding_mask=None, U=None)
    # Additive U would shift every entry by +50.  Rotation saturates at ±π
    # and cannot reproduce that shift.
    assert not torch.allclose(rotary, content + 50.0, atol=1.0)
    assert torch.isfinite(rotary).all()
    theta = u_to_angles(U, block.angle_scale)
    assert theta.abs().max().item() < math.pi + 1e-5
    assert theta.abs().min().item() > 1.0


def test_none_u_matches_softmax_content_path() -> None:
    """With U=None the scores match a content-only softmax block."""
    torch.manual_seed(2)
    urot = URotaryAttentionBlock(
        embed_dim=16, num_heads=2, dropout=0.0, expansion_factor=2
    ).eval()
    x = torch.randn(2, 6, 16)
    pad = torch.zeros(2, 6, dtype=torch.bool)
    pad[:, -1] = True
    with torch.no_grad():
        a = urot.attention_logits(x, pad, U=None)
        x_norm = urot.layernorm1(x)
        B, N, _ = x_norm.shape
        Q = urot.q_proj(x_norm).view(B, N, 2, 8).permute(0, 2, 1, 3)
        K = urot.k_proj(x_norm).view(B, N, 2, 8).permute(0, 2, 1, 3)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(8)
        scores = scores.masked_fill(pad.unsqueeze(1).unsqueeze(2), -1e9)
    assert torch.allclose(a, scores, atol=ATOL, rtol=1e-5)


def test_pair_embed_stays_on_and_wrapper_forwards() -> None:
    model = build_variant_part("urot", **_TINY).eval()
    assert model.pair_embed is not None
    assert isinstance(model.blocks[0], WeaverBlockAdapter)
    assert isinstance(model.blocks[0].block, URotaryAttentionBlock)
    x = torch.randn(2, 16, 8)
    v = torch.rand(2, 4, 8)
    v[:, 3] = v[:, :3].norm(dim=1) + 1.0
    mask = torch.ones(2, 1, 8)
    mask[0, 0, 6:] = 0
    with torch.no_grad():
        logits = model(x, v=v, mask=mask)
    assert logits.shape == (2, 10)
    assert torch.isfinite(logits).all()


def test_urot_yaml_is_a_200k_pair_on_screen_not_k2() -> None:
    config = load_config(str(_UROT_CONFIG))
    assert config.arm == "urot"
    assert config.run_name == "urot"
    assert config.pair_embed_dims == (64, 64, 64)
    assert config.max_steps == 200_000
    assert config.total_steps == 1_000_000
    assert config.n8_degree == 1
    assert config.part_kernels_attention is False
    kwargs = config.model_kwargs()
    assert kwargs["pair_embed_dims"] == (64, 64, 64)
    assert kwargs["per_plane"] is False
    assert kwargs["rope_apply"] is False
    assert config.urot_per_plane is False
    assert config.urot_rope_apply is False
    assert config.embed_dims == (128, 512, 128)


def test_urot_rejects_pair_embed_off() -> None:
    with pytest.raises(ValueError, match="pairwise rotation"):
        AblationConfig(arm="urot", pair_embed_dims=None)


def test_angle_scale_init_is_conservative() -> None:
    block = URotaryAttentionBlock(embed_dim=16, num_heads=2)
    assert torch.allclose(
        block.angle_scale.detach(),
        torch.full((2,), ANGLE_SCALE_INIT),
    )


def test_per_plane_matches_shared_when_alphas_equal() -> None:
    """Independent planes with identical α recover the shared-θ kernel."""
    torch.manual_seed(4)
    B, H, N, D = 2, 2, 5, 8
    query = torch.randn(B, H, N, D)
    key = torch.randn(B, H, N, D)
    U = torch.randn(B, H, N, N)
    alpha = 0.25
    shared = u_to_angles(U, torch.full((H,), alpha))
    scale = 1.0 / math.sqrt(D)
    want = pairwise_rotary_logits(query, key, shared, scale)
    got = pairwise_rotary_logits_per_plane(
        query, key, U, torch.full((H, D // 2), alpha), scale
    )
    assert torch.allclose(got, want, atol=ATOL, rtol=1e-5)


def test_per_plane_two_frequencies_are_not_one_angle() -> None:
    """Different α_p must not collapse to a single shared θ."""
    query = torch.tensor([[[[1.0, 0.0, 0.0, 1.0]]]])  # two planes
    key = torch.tensor([[[[0.0, 1.0, 1.0, 0.0]]]])
    U = torch.ones(1, 1, 1, 1)
    scale = 1.0
    shared = pairwise_rotary_logits(
        query, key, u_to_angles(U, torch.tensor([1.0])), scale
    )
    split = pairwise_rotary_logits_per_plane(
        query, key, U, torch.tensor([[0.1, 2.0]]), scale
    )
    assert not torch.allclose(shared, split, atol=1e-4)


def test_urot_mf_yaml_is_wider_per_plane_not_k2() -> None:
    config = load_config(str(_UROT_MF_CONFIG))
    assert config.arm == "urot"
    assert config.run_name == "urot_mf"
    assert config.urot_per_plane is True
    assert config.embed_dims == (256, 1024, 256)
    assert config.num_heads == 8
    assert config.embed_dims[-1] % config.num_heads == 0
    assert (config.embed_dims[-1] // config.num_heads) % 2 == 0
    assert config.max_steps == 200_000
    assert config.n8_degree == 1
    kwargs = config.model_kwargs()
    assert kwargs["per_plane"] is True
    assert kwargs["rope_apply"] is False
    assert kwargs["embed_dims"] == (256, 1024, 256)


def test_per_plane_block_inits_geometric_freqs_and_forwards() -> None:
    block = URotaryAttentionBlock(
        embed_dim=16, num_heads=2, dropout=0.0, per_plane=True
    )
    assert block.angle_scale.shape == (2, 4)
    freqs = 1.0 / (2.0 ** torch.arange(4))
    assert torch.allclose(block.angle_scale[0].detach(), ANGLE_SCALE_INIT * freqs)
    model = build_variant_part("urot", per_plane=True, **_TINY).eval()
    assert model.blocks[0].block.per_plane is True
    x = torch.randn(2, 16, 8)
    v = torch.rand(2, 4, 8)
    v[:, 3] = v[:, :3].norm(dim=1) + 1.0
    mask = torch.ones(2, 1, 8)
    with torch.no_grad():
        logits = model(x, v=v, mask=mask)
    assert logits.shape == (2, 10)
    assert torch.isfinite(logits).all()


def test_rope_equal_phases_match_content_qk() -> None:
    """Identical ψ for every token is a global rotation and cancels in QKᵀ."""
    torch.manual_seed(5)
    B, H, N, D = 2, 2, 5, 8
    query = torch.randn(B, H, N, D)
    key = torch.randn(B, H, N, D)
    psi = torch.full((B, H, N), 0.7)
    inv_freq = 1.0 / (ROPE_BASE ** (torch.arange(0, D, 2).float() / D))
    q_rot, k_rot = apply_rope(query, key, psi, inv_freq)
    scale = 1.0 / math.sqrt(D)
    got = torch.matmul(q_rot, k_rot.transpose(-2, -1)) * scale
    want = torch.matmul(query, key.transpose(-2, -1)) * scale
    assert torch.allclose(got, want, atol=ATOL, rtol=1e-5)


def test_rope_d2_matches_fused_relative_angle() -> None:
    """On one 2-plane, apply-then-matmul is Q · R(ψ_j − ψ_i) K.

    ``pairwise_rotary_logits`` implements Q · R(−θ) K, so θ = ψ_i − ψ_j.
    """
    torch.manual_seed(6)
    B, H, N, D = 1, 1, 4, 2
    query = torch.randn(B, H, N, D)
    key = torch.randn(B, H, N, D)
    psi = torch.randn(B, H, N)
    inv_freq = torch.ones(1)  # ω_0 = 1 for d=2, base^{0}
    q_rot, k_rot = apply_rope(query, key, psi, inv_freq)
    scale = 1.0
    got = torch.matmul(q_rot, k_rot.transpose(-2, -1))
    theta = psi.unsqueeze(-1) - psi.unsqueeze(-2)
    want = pairwise_rotary_logits(query, key, theta, scale)
    assert torch.allclose(got, want, atol=1e-5, rtol=1e-5)


def test_rope_does_not_add_u_as_a_scalar() -> None:
    torch.manual_seed(7)
    block = URotaryAttentionBlock(
        embed_dim=16, num_heads=2, dropout=0.0, rope_apply=True
    ).eval()
    x = torch.randn(1, 4, 16)
    U = torch.full((1, 2, 4, 4), 50.0)
    with torch.no_grad():
        rotary = block.attention_logits(x, padding_mask=None, U=U)
        content = block.attention_logits(x, padding_mask=None, U=None)
    assert not torch.allclose(rotary, content + 50.0, atol=1.0)
    assert torch.isfinite(rotary).all()


def test_rope_padding_excluded_from_phase() -> None:
    """A padded key with huge U must not move the valid tokens' phases."""
    U = torch.zeros(1, 1, 3, 3)
    U[0, 0, :, 2] = 100.0
    pad = torch.tensor([[False, False, True]])
    alpha = torch.ones(1)
    with_pad = u_to_token_phases(U, alpha, pad)
    clean = u_to_token_phases(U[:, :, :2, :2], alpha, None)
    assert torch.allclose(with_pad[0, 0, :2], clean[0, 0], atol=ATOL)
    unmasked = u_to_token_phases(U, alpha, None)
    assert not torch.allclose(with_pad[0, 0, :2], unmasked[0, 0, :2], atol=1e-4)


def test_urot_rope_yaml_is_apply_then_matmul_not_k2() -> None:
    config = load_config(str(_UROT_ROPE_CONFIG))
    assert config.arm == "urot"
    assert config.run_name == "urot_rope"
    assert config.urot_rope_apply is True
    assert config.urot_per_plane is False
    assert config.embed_dims == (128, 512, 128)
    assert config.max_steps == 200_000
    assert config.n8_degree == 1
    kwargs = config.model_kwargs()
    assert kwargs["rope_apply"] is True
    assert kwargs["per_plane"] is False


def test_config_rejects_per_plane_and_rope_apply() -> None:
    with pytest.raises(ValueError, match="cannot both be True"):
        AblationConfig(arm="urot", urot_per_plane=True, urot_rope_apply=True)


def test_block_rejects_per_plane_and_rope_apply() -> None:
    with pytest.raises(ValueError, match="cannot both be True"):
        URotaryAttentionBlock(embed_dim=16, num_heads=2, per_plane=True, rope_apply=True)


def test_rope_block_inits_llama_freqs_and_forwards() -> None:
    block = URotaryAttentionBlock(
        embed_dim=16, num_heads=2, dropout=0.0, rope_apply=True
    )
    assert block.rope_apply is True
    assert block.angle_scale.shape == (2,)
    assert torch.allclose(
        block.angle_scale.detach(), torch.full((2,), ROPE_PHASE_SCALE_INIT)
    )
    d = 8
    want = 1.0 / (ROPE_BASE ** (torch.arange(0, d, 2).float() / d))
    assert torch.allclose(block.rope_inv_freq.cpu(), want)
    model = build_variant_part("urot", rope_apply=True, **_TINY).eval()
    assert model.blocks[0].block.rope_apply is True
    x = torch.randn(2, 16, 8)
    v = torch.rand(2, 4, 8)
    v[:, 3] = v[:, :3].norm(dim=1) + 1.0
    mask = torch.ones(2, 1, 8)
    with torch.no_grad():
        logits = model(x, v=v, mask=mask)
    assert logits.shape == (2, 10)
    assert torch.isfinite(logits).all()


def test_rotate_half_is_llama_layout() -> None:
    x = torch.tensor([1.0, 2.0, 3.0, 4.0])
    got = rotate_half(x)
    assert torch.equal(got, torch.tensor([-3.0, -4.0, 1.0, 2.0]))
