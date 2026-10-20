"""Property-based tests for the LLoCa arm's Lorentz-geometric invariants.

Feature: ParT ablation — LLoCa (Lorentz Local Canonicalization) arm.

LLoCa's correctness rests on exact algebraic identities that ordinary
forward-pass or gradient tests cannot see: a subtly wrong frame still produces
finite logits and trains, it just silently stops being equivariant. This module
pins the identities down.

Properties
----------
1. ``boost_matrix(v0)`` carries ``v0`` to its rest frame.
2. Predicted frames are Lorentz transformations: ``L g L^T = g``.
3. ``invert_frames`` is an exact inverse: ``L^-1 L = I``.
4. Frames are *equivariant*: ``L(Lambda v) = L(v) Lambda^-1`` (Eq. 5), which is
   precisely what makes local features ``L x`` invariant (Eq. 6).
5. The assembled :class:`~variants.lloca.part.LLoCaParT` is Lorentz-**invariant**
   end-to-end when configured with every symmetry-breaking channel disabled.
6. The paper's symmetry-breaking channels do measurably break that invariance —
   the ablation's whole point, so a config that silently kept exact invariance
   would be the bug.

All frame algebra runs in float64, matching the module (and App. D.1 of
arXiv:2505.20280, which uses double precision for precision-critical steps).

Reference frames are compared only at real-particle positions: padded slots are
assigned the identity frame by construction and carry no meaning.
"""

from __future__ import annotations

import pytest
import torch
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from variants import build_variant_part
from variants.lloca.frames import (
    MINKOWSKI_SIGNATURE,
    FramesNet,
    boost_matrix,
    frames_from_vectors,
    invert_frames,
    minkowski_dot,
    minkowski_norm,
    to_time_first,
    to_weaver_order,
)

DTYPE = torch.float64
METRIC = torch.diag(torch.tensor(MINKOWSKI_SIGNATURE, dtype=DTYPE))

#: Frame algebra is exact up to float64 round-off amplified by the boost.
ATOL = 1e-9

#: Config with every symmetry-breaking channel off, i.e. exactly Lorentz-invariant.
#: ``frames_min_mass=0`` matters: the regulator ``E -> sqrt(E^2 + m^2)`` raises the
#: energy component alone, so it is frame-dependent by construction.
EXACT_EQUIVARIANT = dict(
    symmetry_breaking=False,
    frames_use_nis=False,
    frames_min_mass=0.0,
    pair_embed_dims=None,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _generator(seed: int) -> torch.Generator:
    g = torch.Generator()
    g.manual_seed(seed)
    return g


def _forward_timelike(count: int, seed: int) -> torch.Tensor:
    """Random four-vectors with ``<p, p> > 0`` and ``p^0 > 0``."""
    g = _generator(seed)
    spatial = torch.randn(count, 3, generator=g, dtype=DTYPE)
    energy = spatial.norm(dim=-1, keepdim=True) + 0.5 + torch.rand(
        count, 1, generator=g, dtype=DTYPE
    )
    return torch.cat([energy, spatial], dim=-1)


def _random_lorentz(seed: int, boost_scale: float = 0.4) -> torch.Tensor:
    """A random element of ``SO+(1, 3)`` as ``rotation @ boost``."""
    g = _generator(seed)
    beta = torch.randn(1, 3, generator=g, dtype=DTYPE) * boost_scale
    beta = beta / (1.0 + beta.norm(dim=-1, keepdim=True))  # keep |beta| < 1
    boost = boost_matrix(torch.cat([torch.ones(1, 1, dtype=DTYPE), -beta], dim=-1))

    matrix = torch.randn(1, 3, 3, generator=g, dtype=DTYPE)
    q, r = torch.linalg.qr(matrix)
    q = q * torch.sign(torch.diagonal(r, dim1=-2, dim2=-1)).unsqueeze(-2)
    q = torch.where(torch.linalg.det(q)[:, None, None] < 0, q.flip(-1), q)

    rotation = torch.zeros(1, 4, 4, dtype=DTYPE)
    rotation[:, 0, 0] = 1.0
    rotation[:, 1:, 1:] = q
    return (rotation @ boost)[0]


def _jet_batch(batch: int, particles: int, seed: int):
    """Padded weaver-order batch plus its float64 time-first four-momenta."""
    g = _generator(seed)
    lengths = torch.randint(3, particles + 1, (batch,), generator=g)
    mask = (torch.arange(particles)[None, :] < lengths[:, None]).to(DTYPE)

    momenta = _forward_timelike(batch * particles, seed + 1).reshape(
        batch, particles, 4
    )
    momenta = momenta * mask.unsqueeze(-1)
    v = to_weaver_order(momenta).transpose(1, 2)  # (B, 4, P)
    x = torch.randn(batch, 16, particles, generator=g, dtype=DTYPE)
    x = x * mask.unsqueeze(1)
    return x, v, mask.unsqueeze(1), momenta, mask.bool()


def _with_global_kinematics(
    x: torch.Tensor, momenta: torch.Tensor, mask: torch.Tensor
) -> torch.Tensor:
    """Overwrite ``x``'s kinematic channels with values derived from ``momenta``.

    The loader derives channels 0-5 (``pt, eta, phi, energy, deta, dphi``) from
    the four-momenta, so boosting an event changes them. Tests that hold ``x``
    fixed while rotating ``v`` therefore feed *stale* kinematics, which behave
    like genuine invariant scalars and hide the "non-invariant scalars"
    symmetry breaking entirely. This rebuilds them consistently.
    """
    px, py, pz = momenta[..., 1], momenta[..., 2], momenta[..., 3]
    pt = torch.sqrt(px * px + py * py + 1e-12)
    eta = torch.asinh(pz / pt)
    phi = torch.atan2(py, px)

    jet = (momenta * mask.unsqueeze(-1).to(momenta.dtype)).sum(dim=1, keepdim=True)
    jet_pt = torch.sqrt(jet[..., 1] ** 2 + jet[..., 2] ** 2 + 1e-12)
    jet_eta = torch.asinh(jet[..., 3] / jet_pt)
    jet_phi = torch.atan2(jet[..., 2], jet[..., 1])

    kinematics = torch.stack(
        [
            pt,
            eta,
            phi,
            momenta[..., 0],
            eta - jet_eta,
            torch.remainder(phi - jet_phi + torch.pi, 2 * torch.pi) - torch.pi,
        ],
        dim=-1,
    ).transpose(1, 2)  # (B, 6, P)

    out = x.clone()
    out[:, :6] = kinematics * mask.unsqueeze(1).to(x.dtype)
    return out


# ---------------------------------------------------------------------------
# Property 1 — the boost reaches the rest frame
# ---------------------------------------------------------------------------

@given(seed=st.integers(0, 2**16), count=st.integers(1, 64))
@settings(max_examples=25, deadline=None)
def test_boost_carries_vector_to_rest_frame(seed: int, count: int) -> None:
    v0 = _forward_timelike(count, seed)
    rest = (boost_matrix(v0) @ v0.unsqueeze(-1)).squeeze(-1)

    expected = torch.zeros_like(rest)
    expected[:, 0] = minkowski_norm(v0)
    assert torch.allclose(rest, expected, atol=ATOL)


def test_boost_is_stable_at_zero_velocity() -> None:
    """``(gamma - 1) / beta^2`` is a 0/0 limit at rest; it must stay finite.

    The implementation evaluates it as ``gamma^2 / (gamma + 1)``, which is 1/2
    at ``beta = 0`` rather than NaN.
    """
    at_rest = torch.tensor([[2.0, 0.0, 0.0, 0.0]], dtype=DTYPE)
    boost = boost_matrix(at_rest)
    assert torch.isfinite(boost).all()
    assert torch.allclose(boost, torch.eye(4, dtype=DTYPE).unsqueeze(0), atol=ATOL)


# ---------------------------------------------------------------------------
# Properties 2 and 3 — frames are Lorentz transformations, exactly invertible
# ---------------------------------------------------------------------------

@given(seed=st.integers(0, 2**16), count=st.integers(1, 48))
@settings(max_examples=25, deadline=None)
def test_frames_are_lorentz_transformations(seed: int, count: int) -> None:
    vectors = _forward_timelike(3 * count, seed).reshape(count, 3, 4)
    frames = frames_from_vectors(vectors[:, 0], vectors[:, 1], vectors[:, 2])

    gram = frames @ METRIC @ frames.transpose(-1, -2)
    assert torch.allclose(gram, METRIC.expand_as(gram), atol=ATOL)


@given(seed=st.integers(0, 2**16), count=st.integers(1, 48))
@settings(max_examples=25, deadline=None)
def test_invert_frames_is_exact_inverse(seed: int, count: int) -> None:
    vectors = _forward_timelike(3 * count, seed).reshape(count, 3, 4)
    frames = frames_from_vectors(vectors[:, 0], vectors[:, 1], vectors[:, 2])

    identity = torch.eye(4, dtype=DTYPE).expand_as(frames)
    assert torch.allclose(invert_frames(frames) @ frames, identity, atol=ATOL)


# ---------------------------------------------------------------------------
# Property 4 — frame equivariance and the invariance it implies
# ---------------------------------------------------------------------------

@given(seed=st.integers(0, 2**16), count=st.integers(1, 32))
@settings(max_examples=25, deadline=None)
def test_frames_transform_as_inverse_on_the_right(seed: int, count: int) -> None:
    """``L(Lambda v) = L(v) Lambda^-1`` — Eq. (5)."""
    vectors = _forward_timelike(3 * count, seed).reshape(count, 3, 4)
    transform = _random_lorentz(seed)

    frames = frames_from_vectors(vectors[:, 0], vectors[:, 1], vectors[:, 2])
    rotated = torch.einsum("ab,nkb->nka", transform, vectors)
    frames_rotated = frames_from_vectors(rotated[:, 0], rotated[:, 1], rotated[:, 2])

    assert torch.allclose(
        frames_rotated, frames @ invert_frames(transform), atol=ATOL
    )


@given(seed=st.integers(0, 2**16))
@settings(max_examples=15, deadline=None)
def test_local_four_vectors_are_invariant(seed: int) -> None:
    """``L x`` is unchanged by a global Lorentz transformation — Eq. (6)."""
    vectors = _forward_timelike(3 * 16, seed).reshape(16, 3, 4)
    probe = _forward_timelike(16, seed + 7)
    transform = _random_lorentz(seed)

    frames = frames_from_vectors(vectors[:, 0], vectors[:, 1], vectors[:, 2])
    rotated = torch.einsum("ab,nkb->nka", transform, vectors)
    frames_rotated = frames_from_vectors(rotated[:, 0], rotated[:, 1], rotated[:, 2])

    local = (frames @ probe.unsqueeze(-1)).squeeze(-1)
    probe_rotated = (transform @ probe.unsqueeze(-1)).squeeze(-1)
    local_rotated = (frames_rotated @ probe_rotated.unsqueeze(-1)).squeeze(-1)

    assert torch.allclose(local, local_rotated, atol=ATOL)


@given(seed=st.integers(0, 2**16), particles=st.integers(3, 24))
@settings(max_examples=12, deadline=None, suppress_health_check=[HealthCheck.too_slow])
def test_frames_net_is_equivariant(seed: int, particles: int) -> None:
    """The learned frame predictor inherits Eq. (5) from Eq. (13).

    ``min_mass=0`` because the mass regulator is deliberately frame-dependent
    preprocessing.
    """
    net = FramesNet(
        scalar_dim=5, hidden_dim=16, symmetry_breaking=False, min_mass=0.0
    ).to(DTYPE)

    g = _generator(seed)
    batch = 3
    lengths = torch.randint(3, particles + 1, (batch,), generator=g)
    mask = torch.arange(particles)[None, :] < lengths[:, None]
    momenta = _forward_timelike(batch * particles, seed).reshape(batch, particles, 4)
    momenta = momenta * mask.unsqueeze(-1)
    scalars = torch.randn(batch, particles, 5, generator=g, dtype=DTYPE)

    transform = _random_lorentz(seed)
    rotated = torch.einsum("ab,ijb->ija", transform, momenta) * mask.unsqueeze(-1)

    frames = net(momenta, scalars, mask)
    frames_rotated = net(rotated, scalars, mask)

    expected = frames @ invert_frames(transform)
    assert torch.allclose(frames_rotated[mask], expected[mask], atol=ATOL)


# ---------------------------------------------------------------------------
# Properties 5 and 6 — end-to-end invariance, and its deliberate breaking
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("num_vector_channels", [None, 0, 1])
def test_lloca_part_is_lorentz_invariant(num_vector_channels) -> None:
    """Class logits are unchanged by a global Lorentz transformation.

    ``num_vector_channels=0`` covers the scalar-only message path (the Table 2
    ablation), which must also be invariant — just less expressive.
    """
    torch.manual_seed(0)
    model = (
        build_variant_part(
            "lloca",
            input_dim=16,
            num_classes=10,
            num_layers=2,
            num_vector_channels=num_vector_channels,
            **EXACT_EQUIVARIANT,
        )
        .eval()
        .to(DTYPE)
    )

    x, v, mask, momenta, _ = _jet_batch(batch=3, particles=16, seed=11)
    transform = _random_lorentz(3)
    rotated = torch.einsum("ab,ijb->ija", transform, momenta)
    v_rotated = to_weaver_order(rotated).transpose(1, 2) * mask

    with torch.no_grad():
        logits = model(x, v=v, mask=mask)
        logits_rotated = model(x, v=v_rotated, mask=mask)

    assert torch.allclose(logits, logits_rotated, atol=1e-9)


@pytest.mark.parametrize(
    "overrides, derive_kinematics",
    [
        pytest.param({"symmetry_breaking": True}, False, id="reference-vectors"),
        pytest.param({"frames_use_nis": True}, True, id="non-invariant-scalars"),
        pytest.param({"pair_embed_dims": (16, 16)}, False, id="global-pair-bias"),
    ],
)
def test_symmetry_breaking_channels_break_invariance(
    overrides, derive_kinematics
) -> None:
    """Each symmetry-breaking channel must actually break exact invariance.

    ParT's task is only ``SO(2)``-equivariant about the beam axis, so this
    breaking is intentional (App. E, Tab. 6 shows it is worth a large amount of
    background rejection). A channel that left invariance intact would mean it
    is not wired in.
    """
    config = {**EXACT_EQUIVARIANT, **overrides}
    torch.manual_seed(0)
    model = (
        build_variant_part(
            "lloca", input_dim=16, num_classes=10, num_layers=2, **config
        )
        .eval()
        .to(DTYPE)
    )

    x, v, mask, momenta, mask_bool = _jet_batch(batch=3, particles=16, seed=11)
    transform = _random_lorentz(3)
    rotated = torch.einsum("ab,ijb->ija", transform, momenta)
    v_rotated = to_weaver_order(rotated).transpose(1, 2) * mask

    x_rotated = x
    if derive_kinematics:
        # The NIS channels only break the symmetry when they actually track the
        # four-momenta, as the loader's do.
        x = _with_global_kinematics(x, momenta, mask_bool)
        x_rotated = _with_global_kinematics(x, rotated, mask_bool)

    with torch.no_grad():
        deviation = (
            model(x, v=v, mask=mask) - model(x_rotated, v=v_rotated, mask=mask)
        ).abs().max()

    assert deviation > 1e-6, f"{overrides} did not break invariance"


def test_lloca_block_rejects_missing_frames() -> None:
    """A block used without frames must fail loudly, not reuse stale ones."""
    from variants.lloca.attention import LLoCaAttentionBlock

    block = LLoCaAttentionBlock(embed_dim=32, num_heads=4)
    x = torch.randn(2, 5, 32)
    padding_mask = torch.zeros(2, 5, dtype=torch.bool)

    with pytest.raises(RuntimeError, match="without frames"):
        block(x, padding_mask)

    frames = torch.eye(4).expand(2, 5, 4, 4).contiguous()
    block.set_frames(frames, frames)
    block(x, padding_mask)  # consumes the frames

    with pytest.raises(RuntimeError, match="without frames"):
        block(x, padding_mask)


def test_lloca_block_rejects_frame_token_mismatch() -> None:
    """Frames built before trimming would silently mis-pair with tokens."""
    from variants.lloca.attention import LLoCaAttentionBlock

    block = LLoCaAttentionBlock(embed_dim=32, num_heads=4)
    frames = torch.eye(4).expand(2, 9, 4, 4).contiguous()
    block.set_frames(frames, frames)

    with pytest.raises(RuntimeError, match="frame/token length mismatch"):
        block(torch.randn(2, 5, 32), torch.zeros(2, 5, dtype=torch.bool))


def test_time_first_roundtrip() -> None:
    """The weaver <-> time-first reordering is its own inverse pair."""
    v = torch.randn(3, 7, 4, dtype=DTYPE)
    assert torch.equal(to_weaver_order(to_time_first(v)), v)
    # And the Minkowski product picks up energy from the right slot.
    p = to_time_first(v)
    expected = p[..., 0] ** 2 - p[..., 1:].pow(2).sum(-1)
    assert torch.allclose(minkowski_dot(p, p), expected, atol=ATOL)
