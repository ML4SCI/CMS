"""Shared Hypothesis strategies for the self-contained-kernels-migration suite.

Feature: self-contained-kernels-migration (design "Testing Strategy" section).

Provides the generator strategies shared by the property-based tests in
``variants/tests`` (Properties 1-3) and ``part_kernels/tests`` (Property 4):

- physically valid four-vector batches (``E > |p|`` with a safety margin so
  downstream ``log``/mass math stays finite in float32);
- padded batches with per-jet valid lengths in ``[1, P]`` — Hypothesis biases
  integer draws toward their lower bound, so the 1-valid-particle edge case
  is exercised routinely;
- small weaver ``ParticleTransformer`` configs (2 layers, ``embed_dim <= 64``
  divisible by ``num_heads``, ``P <= 32``) sized so 100 CPU examples run fast.

Determinism: strategies draw plain Python ints/floats from Hypothesis (sizes,
lengths, and one RNG seed per batch) and derive all tensor content through a
seeded ``torch.Generator`` — no global-RNG nondeterminism inside strategies,
and every example shrinks reproducibly.

Public API
----------
``jet_kinematics(...)``
    -> ``JetBatch(v, mask, padding_mask, lengths)`` — raw four-vectors +
    masks (weaver's ``v``/``mask`` contract).
``block_inputs(...)``
    -> ``BlockBatch(x, padding_mask, U, embed_dim, num_heads, lengths)`` —
    inputs for the variant blocks' ``forward(x, padding_mask, U)``.
``weaver_configs(...)``
    -> dict of ``build_variant_part`` / weaver ``ParticleTransformer`` kwargs.
``weaver_batches(...)``
    -> ``WeaverBatch(config, x, v, mask, y, lengths)`` — a small config plus a
    matched ``(x, v, mask, y)`` input batch.

Relationship to the real loader
-------------------------------
These strategies deliberately reproduce the batch contract of
``dataloader.ragged_loader`` (see :data:`LOADER_CONTRACT`), so that a property
proved here transfers to production batches:

===============  =====================================================
``x``            ``(B, 16, P)`` float32, channel-first, padded slots 0
``v``            ``(B, 4, P)``  float32 raw ``[px, py, pz, E]``
``mask``         ``(B, 1, P)``  float32, ``1.0`` real / ``0.0`` pad
``y``            ``(B, 10)``    float32 one-hot
padding          trailing-only (the loader gathers CSR rows in order)
``P``            dynamic — max multiplicity *in that batch*
valid per jet    ``>= 1`` (the loader never emits an empty jet)
===============  =====================================================

That correspondence is **asserted, not assumed**: ``test_loader_contract.py``
builds a real CSR shard, runs it through ``RaggedShardDataset.get_batch``, and
checks the loader's own output against the same invariants these strategies
guarantee. If the loader contract changes, that test fails rather than the
property suite silently drifting onto stale shapes.

Coverage caveat: ``max_particles`` defaults to 32 to keep 100 CPU examples fast,
which is *below* JetClass's mean multiplicity (~39). Use
``max_particles=JETCLASS_P99`` for tests that need realistic widths — see
:data:`JETCLASS_MEAN_PARTICLES` and friends.
"""

from __future__ import annotations

import math
from typing import NamedTuple, Optional

import torch
from hypothesis import strategies as st

__all__ = [
    "JetBatch",
    "BlockBatch",
    "WeaverBatch",
    "jet_kinematics",
    "block_inputs",
    "weaver_configs",
    "weaver_batches",
    "LOADER_CONTRACT",
    "LOADER_NUM_FEATURES",
    "LOADER_NUM_CLASSES",
    "JETCLASS_MEAN_PARTICLES",
    "JETCLASS_P99_PARTICLES",
    "JETCLASS_MAX_PARTICLES",
    "assert_loader_contract",
]

# ---------------------------------------------------------------------------
# The loader contract these strategies mirror
# ---------------------------------------------------------------------------

#: Particle features on ``x``, per ``ALL_PARTICLE_FEATURES`` in
#: ``preprocessing.convert_jetclass_ragged_pt``.
LOADER_NUM_FEATURES: int = 16
#: JetClass classes (one-hot width of ``y``).
LOADER_NUM_CLASSES: int = 10

#: Measured JetClass multiplicity statistics over all 125M jets (see
#: ``JetClass_Readme.mdc``). Used to size realistic-width tests rather than
#: hardcoding a number that drifted in from the retired padded loader.
JETCLASS_MEAN_PARTICLES: int = 39
JETCLASS_P99_PARTICLES: int = 80
JETCLASS_MAX_PARTICLES: int = 183

#: Human-readable summary of the contract, kept next to the code that mirrors
#: it so a reader does not have to open the loader to check.
LOADER_CONTRACT: str = (
    "x (B,16,P) float32 | v (B,4,P) float32 [px,py,pz,E] | "
    "mask (B,1,P) float32 1=real | y (B,10) float32 one-hot | "
    "channel-first, trailing-only padding, padded slots exactly 0.0, "
    ">=1 valid particle per jet, P = max multiplicity in batch"
)


def assert_loader_contract(x, v, mask, y=None, *, lengths=None) -> None:
    """Assert a batch obeys the ``ragged_loader`` contract.

    Shared by the strategy self-checks and by ``test_loader_contract.py``, which
    runs it against the **real** loader's output — so both the synthetic
    fixtures and the production loader are held to one definition.

    Parameters
    ----------
    x, v, mask : Tensor
        Batch tensors to check.
    y : Tensor or None
        One-hot labels, checked when provided.
    lengths : sequence of int or None
        Expected per-jet valid counts. When given, the mask must agree exactly
        and padding must be trailing-only.

    Raises
    ------
    AssertionError
        On any contract violation, naming the field.
    """
    B, F, P = x.shape
    assert F == LOADER_NUM_FEATURES, f"x has {F} features, expected 16"
    assert v.shape == (B, 4, P), f"v is {tuple(v.shape)}, expected {(B, 4, P)}"
    assert mask.shape == (B, 1, P), f"mask is {tuple(mask.shape)}, expected {(B, 1, P)}"
    for name, t in (("x", x), ("v", v), ("mask", mask)):
        assert t.dtype == torch.float32, f"{name} is {t.dtype}, expected float32"
        assert torch.isfinite(t).all(), f"{name} has non-finite values"

    counts = mask.sum(dim=-1).squeeze(-1)
    assert (counts >= 1).all(), "a jet has no valid particles"
    # mask must be strictly binary
    assert torch.logical_or(mask == 0.0, mask == 1.0).all(), "mask is not 0/1"

    for i in range(B):
        n = int(counts[i])
        # Trailing-only padding: leading n are real, the rest are pad.
        assert mask[i, 0, :n].eq(1.0).all(), f"jet {i}: padding is not trailing-only"
        assert mask[i, 0, n:].eq(0.0).all(), f"jet {i}: padding is not trailing-only"
        # Padded slots carry exact zeros, never a sentinel like -1e9.
        assert v[i, :, n:].eq(0.0).all(), f"jet {i}: v padding is not exactly 0"
        assert x[i, :, n:].eq(0.0).all(), f"jet {i}: x padding is not exactly 0"

    if lengths is not None:
        assert [int(c) for c in counts] == [int(n) for n in lengths], (
            f"mask counts {[int(c) for c in counts]} != lengths {list(lengths)}"
        )

    if y is not None:
        assert y.shape == (B, LOADER_NUM_CLASSES), (
            f"y is {tuple(y.shape)}, expected {(B, LOADER_NUM_CLASSES)}"
        )
        assert y.dtype == torch.float32, f"y is {y.dtype}, expected float32"
        assert y.sum(dim=1).eq(1.0).all(), "y rows are not one-hot"

# ---------------------------------------------------------------------------
# Physical bounds (jet-physics-ish scales, chosen for float32 stability)
# ---------------------------------------------------------------------------

#: Transverse momentum range [GeV]. Lower bound keeps ln(pt), ln(kt) finite.
PT_MIN, PT_MAX = 0.5, 50.0
#: Longitudinal momentum range [GeV].
PZ_MAX = 50.0
#: Particle mass range [GeV]. E = sqrt(|p|^2 + m^2) with m >= MASS_MIN
#: guarantees E > |p| with a margin of at least m^2 / (2E) ~ 2e-4 at the
#: largest momenta here — far above float32 resolution (~1e-5 at |p| ~ 86),
#: so E - |p| never rounds to zero and ln(m^2) stays well away from -inf.
MASS_MIN, MASS_MAX = 0.2, 10.0


# ---------------------------------------------------------------------------
# Result containers
# ---------------------------------------------------------------------------

class JetBatch(NamedTuple):
    """Raw four-vector batch in weaver's ``(v, mask)`` contract."""

    v: torch.Tensor             #: (B, 4, P) float32, [px, py, pz, E], padded slots zeroed
    mask: torch.Tensor          #: (B, 1, P) float32, 1.0 = real particle, 0.0 = padding
    padding_mask: torch.Tensor  #: (B, P) bool, True = padded (weaver Block convention)
    lengths: list[int]          #: per-jet valid particle counts, each in [1, P]


class BlockBatch(NamedTuple):
    """Inputs for a variant block's ``forward(x, padding_mask, U)``."""

    x: torch.Tensor             #: (B, N, C) float32, padded positions zeroed
    padding_mask: torch.Tensor  #: (B, N) bool, True = padded
    U: Optional[torch.Tensor]   #: (B, H, N, N) float32 attention bias, or None
    embed_dim: int              #: C (divisible by num_heads)
    num_heads: int              #: H
    lengths: list[int]          #: per-jet valid counts, each in [1, N]


class WeaverBatch(NamedTuple):
    """A small weaver config plus a matched ``(x, v, mask, y)`` input batch.

    Mirrors one ``ragged_loader`` batch, ``y`` included, so a test can exercise
    the loss path (``y.argmax(dim=1)``) the trainer actually uses instead of
    inventing labels locally.
    """

    config: dict                #: kwargs for build_variant_part / ParticleTransformer
    x: torch.Tensor             #: (B, C_in, P) float32 token features
    v: torch.Tensor             #: (B, 4, P) float32 raw [px, py, pz, E]
    mask: torch.Tensor          #: (B, 1, P) float32, 1.0 = real particle
    y: torch.Tensor             #: (B, num_classes) float32 one-hot labels
    lengths: list[int]          #: per-jet valid counts, each in [1, P]


# ---------------------------------------------------------------------------
# Deterministic tensor builders (all randomness flows through one seed)
# ---------------------------------------------------------------------------

def _make_lengths_masks(lengths: list[int], num_particles: int):
    """Build (mask (B,1,P) float32, padding_mask (B,P) bool) from lengths."""
    batch = len(lengths)
    mask = torch.zeros(batch, 1, num_particles, dtype=torch.float32)
    for i, n_valid in enumerate(lengths):
        mask[i, 0, :n_valid] = 1.0
    padding_mask = mask.squeeze(1) < 0.5  # True = padded
    return mask, padding_mask


def _make_four_vectors(
    lengths: list[int], num_particles: int, generator: torch.Generator
) -> torch.Tensor:
    """Deterministically build a physically valid (B, 4, P) four-vector batch.

    Construction guarantees, at every valid position:
    - ``pt`` in [PT_MIN, PT_MAX] (so ln(pt)-style features are finite),
    - ``E = sqrt(px^2 + py^2 + pz^2 + m^2)`` with ``m >= MASS_MIN``, hence
      strictly ``E > |p|`` with a float32-safe margin.
    Padded slots are zeroed (matching the loader contract: E == 0 at padding).
    """
    batch = len(lengths)
    shape = (batch, num_particles)

    def uniform(lo: float, hi: float) -> torch.Tensor:
        u = torch.rand(shape, generator=generator, dtype=torch.float64)
        return lo + (hi - lo) * u

    pt = uniform(PT_MIN, PT_MAX)
    phi = uniform(-math.pi, math.pi)
    pz = uniform(-PZ_MAX, PZ_MAX)
    m = uniform(MASS_MIN, MASS_MAX)

    px = pt * torch.cos(phi)
    py = pt * torch.sin(phi)
    energy = torch.sqrt(px**2 + py**2 + pz**2 + m**2)

    v = torch.stack([px, py, pz, energy], dim=1).to(torch.float32)  # (B, 4, P)

    valid = torch.zeros(batch, 1, num_particles, dtype=torch.float32)
    for i, n_valid in enumerate(lengths):
        valid[i, 0, :n_valid] = 1.0
    return v * valid


def _make_one_hot_labels(
    batch: int, num_classes: int, generator: torch.Generator
) -> torch.Tensor:
    """Build a ``(B, num_classes)`` float32 one-hot label block.

    Matches the loader, which stores ``y`` as one-hot float32 rather than an
    integer class index; the trainer reduces it with ``y.argmax(dim=1)``.
    """
    idx = torch.randint(0, num_classes, (batch,), generator=generator)
    return torch.nn.functional.one_hot(idx, num_classes).to(torch.float32)


# ---------------------------------------------------------------------------
# Shared draw helpers
# ---------------------------------------------------------------------------

def _draw_lengths(draw, batch_size: int, max_valid: int) -> list[int]:
    """Per-jet valid lengths in [1, max_valid].

    Hypothesis shrinks/biases integers toward the lower bound, so the
    1-valid-particle edge case is generated and shrunk to routinely.
    """
    return draw(
        st.lists(
            st.integers(min_value=1, max_value=max_valid),
            min_size=batch_size,
            max_size=batch_size,
        )
    )


def _draw_generator(draw) -> torch.Generator:
    seed = draw(st.integers(min_value=0, max_value=2**32 - 1))
    generator = torch.Generator()
    generator.manual_seed(seed)
    return generator


# ---------------------------------------------------------------------------
# Public strategies
# ---------------------------------------------------------------------------

@st.composite
def jet_kinematics(
    draw,
    max_batch: int = 4,
    max_particles: int = 32,
    min_particles: int = 2,
) -> JetBatch:
    """Padded batches of physically valid four-vectors (weaver ``v``/``mask``).

    Per-jet valid lengths are drawn from ``[1, P]`` inclusive — jets padded
    down to a single valid particle are part of the input space.
    """
    batch = draw(st.integers(min_value=1, max_value=max_batch))
    num_particles = draw(
        st.integers(min_value=min_particles, max_value=max_particles)
    )
    lengths = _draw_lengths(draw, batch, num_particles)
    generator = _draw_generator(draw)

    v = _make_four_vectors(lengths, num_particles, generator)
    mask, padding_mask = _make_lengths_masks(lengths, num_particles)
    return JetBatch(v=v, mask=mask, padding_mask=padding_mask, lengths=lengths)


@st.composite
def block_inputs(
    draw,
    max_batch: int = 4,
    max_particles: int = 32,
    with_bias: bool = True,
    min_padded_per_jet: int = 0,
) -> BlockBatch:
    """Inputs for the variant blocks' ``forward(x, padding_mask, U)``.

    - ``x`` is ``(B, N, C)`` with padded positions zeroed (weaver zeroes them
      before its encoder blocks run);
    - ``padding_mask`` is ``(B, N)`` bool, True = padded;
    - ``U`` is a ``(B, H, N, N)`` float attention bias (pair-embed shaped),
      or ``None`` when ``with_bias`` draws it away;
    - ``embed_dim`` <= 64 and divisible by ``num_heads``.

    ``min_padded_per_jet=1`` restricts valid lengths to ``[1, N-1]`` so every
    jet has at least one padded position (Property 1's precondition); the
    default keeps the full ``[1, N]`` range (Property 2's input space).
    """
    batch = draw(st.integers(min_value=1, max_value=max_batch))
    # Ensure at least one valid particle can coexist with the padding demand.
    min_n = max(2, 1 + min_padded_per_jet)
    num_particles = draw(st.integers(min_value=min_n, max_value=max_particles))

    num_heads = draw(st.sampled_from([1, 2, 4]))
    head_dim = draw(st.sampled_from([4, 8, 16]))
    embed_dim = num_heads * head_dim  # 4..64, divisible by num_heads

    max_valid = num_particles - min_padded_per_jet
    lengths = _draw_lengths(draw, batch, max_valid)
    generator = _draw_generator(draw)

    mask, padding_mask = _make_lengths_masks(lengths, num_particles)

    x = torch.randn(
        batch, num_particles, embed_dim, generator=generator, dtype=torch.float32
    )
    x = x * mask.squeeze(1).unsqueeze(-1)  # zero the padded positions

    U: Optional[torch.Tensor] = None
    if with_bias and draw(st.booleans()):
        U = torch.randn(
            batch,
            num_heads,
            num_particles,
            num_particles,
            generator=generator,
            dtype=torch.float32,
        )

    return BlockBatch(
        x=x,
        padding_mask=padding_mask,
        U=U,
        embed_dim=embed_dim,
        num_heads=num_heads,
        lengths=lengths,
    )


@st.composite
def weaver_configs(
    draw,
    input_dim: int = 16,
    num_classes: int = 10,
) -> dict:
    """Small weaver / ``build_variant_part`` configs (fast on CPU).

    Always 2 encoder layers, 1 class-attention layer, ``embed_dim <= 64``
    divisible by ``num_heads``, tiny pair-embed stack. The returned dict is
    directly splattable into ``build_variant_part(variant, **config)`` and
    (minus ``dropout``/``expansion_factor``) into weaver's
    ``ParticleTransformer``.
    """
    num_heads = draw(st.sampled_from([1, 2, 4]))
    head_dim = draw(st.sampled_from([8, 16]))
    embed_dim = num_heads * head_dim  # 8..64, divisible by num_heads

    pair_hidden = draw(st.sampled_from([8, 16]))

    return {
        "input_dim": input_dim,
        "num_classes": num_classes,
        "embed_dims": (embed_dim, embed_dim * 2, embed_dim),
        "pair_embed_dims": (pair_hidden, pair_hidden),
        "num_heads": num_heads,
        "num_layers": 2,
        "num_cls_layers": 1,
        "dropout": 0.1,
        "expansion_factor": 2,
    }


@st.composite
def weaver_batches(
    draw,
    max_batch: int = 4,
    max_particles: int = 32,
    min_particles: int = 2,
    input_dim: int = 16,
    num_classes: int = 10,
) -> WeaverBatch:
    """A small weaver config plus a matched ``(x, v, mask)`` input batch.

    ``x`` is ``(B, input_dim, P)`` standard-normal token features (weaver's
    ``Embed`` consumes channel-first), ``v`` is a physically valid raw
    four-vector batch, ``mask`` is ``(B, 1, P)`` float, and ``y`` is a
    ``(B, num_classes)`` float32 one-hot label block. Feature values at padded
    slots are zeroed.

    The result satisfies :func:`assert_loader_contract`, i.e. it is shaped
    exactly like a ``dataloader.ragged_loader`` batch, so it can be fed to a
    model as ``model(x, v=v, mask=mask)`` and to a loss as ``y.argmax(dim=1)``.
    """
    config = draw(weaver_configs(input_dim=input_dim, num_classes=num_classes))

    batch = draw(st.integers(min_value=1, max_value=max_batch))
    num_particles = draw(
        st.integers(min_value=min_particles, max_value=max_particles)
    )
    lengths = _draw_lengths(draw, batch, num_particles)
    generator = _draw_generator(draw)

    v = _make_four_vectors(lengths, num_particles, generator)
    mask, _ = _make_lengths_masks(lengths, num_particles)

    x = torch.randn(
        batch, input_dim, num_particles, generator=generator, dtype=torch.float32
    )
    x = x * mask  # (B, 1, P) broadcasts over the feature dim

    y = _make_one_hot_labels(batch, num_classes, generator)

    return WeaverBatch(
        config=config, x=x, v=v, mask=mask, y=y, lengths=lengths
    )
