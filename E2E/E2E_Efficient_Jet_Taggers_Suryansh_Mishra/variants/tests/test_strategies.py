"""Self-checks for the shared Hypothesis strategies (task 3.7).

Feature: self-contained-kernels-migration.

These are lightweight guards on the generators themselves — every property
test in this suite (Properties 1-4) draws its inputs from
``variants.tests.strategies``, so a broken generator invariant would
silently weaken all of them. They intentionally use a small ``max_examples``
budget; the heavy 100-example runs belong to the numbered property tests.

Checked invariants:
- four-vectors are physically valid at valid positions: ``E > |p|`` strictly,
  ``pt >= PT_MIN``; padded slots are exactly zero;
- every jet has at least 1 valid particle; lengths lie in ``[1, P]`` and
  match the masks; ``min_padded_per_jet=1`` leaves >= 1 padded slot per jet;
- ``embed_dim`` is divisible by ``num_heads`` and <= 64; ``x``/``U`` shapes
  are consistent; padded positions of ``x`` are zeroed;
- weaver configs stay in the fast-on-CPU envelope (2 layers, embed <= 64).

Validates: Requirements 1.5, 1.6 (generator preconditions for the block
property tests).
"""

import torch
from hypothesis import given, settings

from variants.tests.strategies import (
    PT_MIN,
    assert_loader_contract,
    block_inputs,
    jet_kinematics,
    weaver_batches,
    weaver_configs,
)

_SETTINGS = settings(max_examples=25, deadline=None)


@given(jet_kinematics())
@_SETTINGS
def test_jet_kinematics_invariants(batch):
    v, mask, padding_mask, lengths = batch
    B, four, P = v.shape
    assert four == 4
    assert v.dtype == torch.float32
    assert mask.shape == (B, 1, P) and mask.dtype == torch.float32
    assert padding_mask.shape == (B, P) and padding_mask.dtype == torch.bool
    assert len(lengths) == B

    for i, n_valid in enumerate(lengths):
        # Per-jet valid lengths in [1, P]; masks agree with lengths.
        assert 1 <= n_valid <= P
        assert mask[i, 0, :n_valid].eq(1.0).all()
        assert mask[i, 0, n_valid:].eq(0.0).all()
        assert padding_mask[i].eq(mask[i, 0] < 0.5).all()

        px, py, pz, energy = v[i, :, :n_valid]
        p_mag = torch.sqrt(px**2 + py**2 + pz**2)
        # Physical validity with a float32-safe margin (strict inequality).
        assert (energy > p_mag).all()
        # pt bounded below so ln(pt)/ln(kt) math downstream stays finite.
        pt = torch.sqrt(px**2 + py**2)
        assert (pt >= PT_MIN * 0.999).all()
        # Padded slots zeroed.
        assert v[i, :, n_valid:].eq(0.0).all()


@given(jet_kinematics())
@_SETTINGS
def test_jet_kinematics_survives_weaver_pairwise_math(batch):
    """The weaver reference pair-feature math must be finite on our jets."""
    from weaver.nn.model.ParticleTransformer import pairwise_lv_fts_pp

    v = batch.v
    fts = pairwise_lv_fts_pp(v.unsqueeze(-1), v.unsqueeze(-2), num_outputs=4)
    # Only valid-x-valid pairs are consumed downstream (sparse gather).
    valid = batch.mask.squeeze(1).bool()
    pair_valid = valid.unsqueeze(-1) & valid.unsqueeze(-2)  # (B, P, P)
    assert torch.isfinite(fts.permute(0, 2, 3, 1)[pair_valid]).all()


@given(block_inputs(min_padded_per_jet=1))
@_SETTINGS
def test_block_inputs_with_forced_padding(batch):
    x, padding_mask, U, embed_dim, num_heads, lengths = batch
    B, N, C = x.shape
    assert C == embed_dim
    assert embed_dim % num_heads == 0
    assert embed_dim <= 64
    assert padding_mask.shape == (B, N) and padding_mask.dtype == torch.bool
    if U is not None:
        assert U.shape == (B, num_heads, N, N)
    for i, n_valid in enumerate(lengths):
        assert 1 <= n_valid <= N - 1  # at least one padded slot per jet
        assert not padding_mask[i, :n_valid].any()
        assert padding_mask[i, n_valid:].all()
        assert x[i, n_valid:].eq(0.0).all()  # padded positions zeroed


@given(block_inputs())
@_SETTINGS
def test_block_inputs_full_length_range(batch):
    # Default range is [1, N]: fully valid jets are allowed, empty jets never.
    N = batch.x.shape[1]
    for n_valid in batch.lengths:
        assert 1 <= n_valid <= N


@given(weaver_configs())
@_SETTINGS
def test_weaver_configs_envelope(config):
    embed_dim = config["embed_dims"][-1]
    assert config["num_layers"] == 2
    assert embed_dim <= 64
    assert embed_dim % config["num_heads"] == 0


@given(weaver_batches(max_particles=16))
@_SETTINGS
def test_weaver_batches_shapes_match_config(batch):
    config, x, v, mask, y, lengths = batch
    B, C_in, P = x.shape
    assert C_in == config["input_dim"]
    assert v.shape == (B, 4, P)
    assert mask.shape == (B, 1, P)
    assert y.shape == (B, config["num_classes"])
    assert P <= 32
    for n_valid in lengths:
        assert 1 <= n_valid <= P


@given(weaver_batches(max_particles=16))
@_SETTINGS
def test_weaver_batches_satisfy_the_loader_contract(batch):
    """The strategy must produce batches shaped exactly like loader output.

    This is the same assertion ``test_loader_contract.py`` runs against the real
    ``RaggedShardDataset``, so the synthetic fixtures and the production loader
    are held to one definition instead of drifting apart.
    """
    assert_loader_contract(
        batch.x, batch.v, batch.mask, batch.y, lengths=batch.lengths
    )


@given(weaver_batches(max_particles=16))
@_SETTINGS
def test_weaver_batches_labels_drive_the_training_loss_path(batch):
    """``y`` must survive the exact reduction the trainer applies.

    ``ablation.train`` does ``targets = y.argmax(dim=1)`` and feeds that to
    ``F.cross_entropy``; a label block that is not one-hot float32 would slip
    through a shape check but silently train on the wrong targets.
    """
    targets = batch.y.argmax(dim=1)
    B = batch.x.shape[0]
    num_classes = batch.config["num_classes"]

    assert targets.shape == (B,)
    assert targets.dtype == torch.int64
    assert (targets >= 0).all() and (targets < num_classes).all()
    # One-hot round-trip: argmax then re-encode must recover y exactly.
    reencoded = torch.nn.functional.one_hot(targets, num_classes).to(batch.y.dtype)
    assert torch.equal(reencoded, batch.y)
