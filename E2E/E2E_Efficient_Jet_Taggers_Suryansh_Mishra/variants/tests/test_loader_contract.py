"""The real ``ragged_loader`` and the test strategies obey one contract.

``variants/tests/strategies.py`` reproduces the batch contract of
``dataloader.ragged_loader`` so that properties proved on synthetic batches
transfer to production ones. Until now that correspondence was only asserted of
the *synthetic* side — the loader itself was never checked, so a change to the
loader (a bool mask, a sentinel pad value, channel-last output) would leave the
property suite passing on shapes production no longer emits.

This module closes that loop: it writes a real CSR ``.pt`` shard, reads it back
through ``RaggedShardDataset``, and runs the **same**
:func:`~variants.tests.strategies.assert_loader_contract` used by the strategy
self-checks. It also feeds genuine loader output through the ablation arms, so
the models are exercised on loader tensors rather than only on fixtures.

No dataset is required: the shards are synthesised in a ``tmp_path``, in exactly
the on-disk schema ``preprocessing/convert_jetclass_ragged_pt.py`` writes.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from variants import build_variant_part
from variants.tests.strategies import (
    JETCLASS_MAX_PARTICLES,
    LOADER_NUM_CLASSES,
    LOADER_NUM_FEATURES,
    assert_loader_contract,
)

pytestmark = pytest.mark.filterwarnings("ignore::FutureWarning")


# ---------------------------------------------------------------------------
# Build a real CSR shard on disk
# ---------------------------------------------------------------------------

#: Multiplicities spanning the cases the loader must handle: the 1-particle
#: floor, small jets, and a jet above JetClass's p99 (80).
_COUNTS = (1, 2, 5, 17, 39, 80, 3, 12, 64, 7)


def _write_shard(directory, counts=_COUNTS, seed=0) -> int:
    """Write one CSR ``.pt`` shard in the converter's on-disk schema.

    Schema (see ``convert_jetclass_ragged_pt``): ``x`` (N_part, 16),
    ``v`` (N_part, 4), ``offsets`` (N_jets+1,) int64, ``jet`` (N_jets, 10),
    ``y`` (N_jets, 10), ``n_particles`` (N_jets,) int32.
    """
    rng = np.random.default_rng(seed)
    counts_arr = np.asarray(counts, dtype=np.int32)
    n_jets = len(counts_arr)
    n_part = int(counts_arr.sum())

    offsets = np.empty(n_jets + 1, dtype=np.int64)
    offsets[0] = 0
    np.cumsum(counts_arr, out=offsets[1:])

    x = rng.normal(size=(n_part, LOADER_NUM_FEATURES)).astype(np.float32)

    # Physically valid four-vectors: E = sqrt(|p|^2 + m^2) with m > 0, so
    # E > |p| strictly and E > 0 everywhere (the loader's validity signal).
    p3 = rng.normal(size=(n_part, 3)).astype(np.float32) * 10.0
    mass = rng.uniform(0.2, 10.0, size=(n_part, 1)).astype(np.float32)
    energy = np.sqrt((p3**2).sum(axis=1, keepdims=True) + mass**2)
    v = np.concatenate([p3, energy], axis=1).astype(np.float32)

    labels = rng.integers(0, LOADER_NUM_CLASSES, size=n_jets)
    y = np.zeros((n_jets, LOADER_NUM_CLASSES), dtype=np.float32)
    y[np.arange(n_jets), labels] = 1.0

    jet = rng.normal(size=(n_jets, LOADER_NUM_CLASSES)).astype(np.float32)
    jet[:, -1] = counts_arr  # jet_nparticles is the last jet feature

    torch.save(
        {
            "x": torch.from_numpy(x),
            "v": torch.from_numpy(v),
            "offsets": torch.from_numpy(offsets),
            "jet": torch.from_numpy(jet),
            "y": torch.from_numpy(y),
            "n_particles": torch.from_numpy(counts_arr),
        },
        str(directory / "HToBB_000.pt"),
    )
    return n_jets


@pytest.fixture
def shard_dir(tmp_path):
    """A directory holding two real CSR shards."""
    d = tmp_path / "pt_ragged"
    d.mkdir()
    _write_shard(d, seed=0)
    # A second shard so multi-shard sampling and rank partitioning are legal.
    rng_counts = (4, 9, 1, 22, 55, 31, 2, 18, 76, 6)
    d2 = d / "second"
    d2.mkdir()
    _write_shard(d2, counts=rng_counts, seed=1)
    # Flatten: RaggedShardDataset globs *.pt in one directory.
    (d2 / "HToBB_000.pt").rename(d / "HToCC_000.pt")
    d2.rmdir()
    return d


# ---------------------------------------------------------------------------
# The loader's own output must satisfy the shared contract
# ---------------------------------------------------------------------------

def test_get_batch_output_satisfies_the_loader_contract(shard_dir):
    """``RaggedShardDataset.get_batch`` output passes the same assertion the
    synthetic strategies are held to."""
    from dataloader.ragged_loader import RaggedShardDataset

    ds = RaggedShardDataset(str(shard_dir), cache_size=1, events_per_shard=len(_COUNTS))
    x, v, mask, y = ds.get_batch(0, np.arange(len(_COUNTS)))

    assert_loader_contract(x, v, mask, y)
    # P is the max multiplicity in this batch, not a fixed pad width.
    assert x.shape[-1] == max(_COUNTS)
    assert x.shape[-1] <= JETCLASS_MAX_PARTICLES


def test_get_batch_mask_counts_match_n_particles(shard_dir):
    """The mask must reproduce the shard's own ``n_particles``, in order."""
    from dataloader.ragged_loader import RaggedShardDataset

    ds = RaggedShardDataset(str(shard_dir), cache_size=1, events_per_shard=len(_COUNTS))
    idx = np.arange(len(_COUNTS))
    _x, _v, mask, _y = ds.get_batch(0, idx)

    counts = [int(c) for c in mask.sum(dim=-1).squeeze(-1)]
    assert counts == list(_COUNTS)


def test_single_particle_jet_survives_the_loader(shard_dir):
    """The 1-particle floor is a real loader output, not just a fixture edge.

    ``_COUNTS[0] == 1``. A jet with one valid particle has a 1x1 pair grid,
    which is where weaver's BatchNorm and the pair pipeline degenerate — so the
    property suite must be able to reach it from real data.
    """
    from dataloader.ragged_loader import RaggedShardDataset

    ds = RaggedShardDataset(str(shard_dir), cache_size=1, events_per_shard=len(_COUNTS))
    x, v, mask, y = ds.get_batch(0, np.array([0]))

    assert_loader_contract(x, v, mask, y, lengths=[1])
    assert x.shape[-1] == 1


def test_flat_and_batch_paths_agree(shard_dir):
    """The flat ``__getitem__`` path and the vectorised ``get_batch`` path must
    return the same particles for the same jet.

    They are separate implementations — a CSR slice versus a gather-and-pad — and
    the flat path feeds ``ragged_collate`` (bench mode A) while ``get_batch``
    feeds training. Divergence would make the benchmark measure different data
    from the trainer.
    """
    from dataloader.ragged_loader import RaggedShardDataset

    ds = RaggedShardDataset(str(shard_dir), cache_size=2, events_per_shard=len(_COUNTS))
    jet = 3
    x_flat, v_flat, mask_flat, y_flat, n_flat = ds[jet]
    x_b, v_b, mask_b, y_b = ds.get_batch(0, np.array([jet]))

    n = int(n_flat)
    assert n == _COUNTS[jet]
    # Flat is (P, F) channel-last; get_batch is (1, F, P) channel-first.
    torch.testing.assert_close(x_b[0, :, :n], x_flat.T)
    torch.testing.assert_close(v_b[0, :, :n], v_flat.T)
    torch.testing.assert_close(y_b[0], y_flat)
    assert int(mask_b[0, 0].sum()) == n
    assert int(mask_flat.sum()) == n


def test_class_balanced_sampler_mixes_shards_in_every_batch(shard_dir):
    """Single-class source shards must be mixed before an optimizer step."""
    from dataloader.ragged_loader import (
        ClassBalancedBatchSampler,
        RaggedShardDataset,
    )

    ds = RaggedShardDataset(
        str(shard_dir), cache_size=2, events_per_shard=len(_COUNTS)
    )
    sampler = ClassBalancedBatchSampler(ds, batch_size=8)
    requests = next(iter(sampler))

    counts_by_shard = {
        shard_idx: len(jet_indices) for shard_idx, jet_indices in requests
    }
    assert counts_by_shard == {0: 4, 1: 4}


def test_class_balanced_loader_collates_mixed_subbatches(shard_dir):
    """Vectorised class portions are padded and concatenated into one batch."""
    from dataloader.ragged_loader import create_ragged_train_loader

    loader = create_ragged_train_loader(
        str(shard_dir),
        batch_size=8,
        num_workers=0,
        events_per_shard=len(_COUNTS),
        class_balanced=True,
    )
    x, v, mask, y = next(iter(loader))

    assert x.shape[0] == v.shape[0] == mask.shape[0] == y.shape[0] == 8
    assert_loader_contract(x, v, mask, y)


# ---------------------------------------------------------------------------
# The arms must consume real loader batches
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("variant", ["baseline", "sparsemax", "moe", "diff_v1", "diff_v2", "urot"])
def test_arms_forward_on_real_loader_batches(shard_dir, variant):
    """Every ParT-family arm forwards on genuine loader output.

    The property suite otherwise only ever sees synthetic fixtures; this runs the
    arms on tensors that came off disk through the loader, using the same call
    convention the trainer uses.
    """
    from dataloader.ragged_loader import RaggedShardDataset

    ds = RaggedShardDataset(str(shard_dir), cache_size=1, events_per_shard=len(_COUNTS))
    # Skip the 1-particle jet: a single valid pair makes weaver's own BatchNorm
    # raise in train mode, and it is covered separately above.
    x, v, mask, y = ds.get_batch(0, np.array([3, 4, 5]))

    torch.manual_seed(0)
    model = build_variant_part(
        variant,
        input_dim=LOADER_NUM_FEATURES,
        num_classes=LOADER_NUM_CLASSES,
        embed_dims=(16, 32, 16),
        pair_embed_dims=(8, 8),
        num_heads=2,
        num_layers=1,
        num_cls_layers=1,
    ).eval()

    with torch.no_grad():
        logits = model(x, v=v, mask=mask)

    assert logits.shape == (3, LOADER_NUM_CLASSES)
    assert torch.isfinite(logits).all()

    # The loss path the trainer uses.
    loss = torch.nn.functional.cross_entropy(logits.float(), y.argmax(dim=1))
    assert torch.isfinite(loss)


def test_lloca_arm_forwards_on_real_loader_batches(shard_dir):
    """LLoCa is a wrapper rather than a block swap, and requires ``v``.

    It runs in float64 for equivariance, so the loader's float32 batch has to be
    cast — this pins that the cast is all that is needed.
    """
    from dataloader.ragged_loader import RaggedShardDataset

    ds = RaggedShardDataset(str(shard_dir), cache_size=1, events_per_shard=len(_COUNTS))
    x, v, mask, y = ds.get_batch(0, np.array([3, 4, 5]))

    torch.manual_seed(0)
    model = build_variant_part(
        "lloca",
        input_dim=LOADER_NUM_FEATURES,
        num_classes=LOADER_NUM_CLASSES,
        embed_dims=(16, 32, 16),
        pair_embed_dims=(8, 8),
        num_heads=2,
        num_layers=1,
        num_cls_layers=1,
    ).eval()

    with torch.no_grad():
        logits = model(x, v=v, mask=mask)

    assert logits.shape == (3, LOADER_NUM_CLASSES)
    assert torch.isfinite(logits).all()


def test_n8_arm_forwards_on_real_loader_batches(shard_dir):
    """N8 is a wrapper rather than a block swap, and requires ``v``."""
    from dataloader.ragged_loader import RaggedShardDataset

    ds = RaggedShardDataset(str(shard_dir), cache_size=1, events_per_shard=len(_COUNTS))
    x, v, mask, y = ds.get_batch(0, np.array([3, 4, 5]))

    torch.manual_seed(0)
    model = build_variant_part(
        "n8",
        input_dim=LOADER_NUM_FEATURES,
        num_classes=LOADER_NUM_CLASSES,
        embed_dims=(16, 32, 16),
        pair_embed_dims=(8, 8),
        num_heads=2,
        num_layers=1,
        num_cls_layers=1,
        minkowski=True,
        rotary=True,
        rotary_pairs=4,
        degree=1,
    ).eval()

    with torch.no_grad():
        logits = model(x, v=v, mask=mask)

    assert logits.shape == (3, LOADER_NUM_CLASSES)
    assert torch.isfinite(logits).all()
    assert model.pair_embed is None


def test_lloca_rejects_missing_four_vectors(shard_dir):
    """LLoCa must fail loudly when ``v`` is absent, not silently degrade.

    The loader always supplies ``v``; a caller that drops it would otherwise get
    a non-equivariant model with no signal that the frames were never built.
    """
    torch.manual_seed(0)
    model = build_variant_part(
        "lloca",
        input_dim=LOADER_NUM_FEATURES,
        num_classes=LOADER_NUM_CLASSES,
        embed_dims=(16, 32, 16),
        pair_embed_dims=(8, 8),
        num_heads=2,
        num_layers=1,
        num_cls_layers=1,
    ).eval()

    x = torch.randn(2, LOADER_NUM_FEATURES, 8)
    mask = torch.ones(2, 1, 8)
    with pytest.raises(ValueError):
        model(x, v=None, mask=mask)


# ---------------------------------------------------------------------------
# The L-GATr arm shares the call convention
# ---------------------------------------------------------------------------

def test_lgatr_arm_accepts_the_harness_call_convention(shard_dir):
    """``LGATrJetClassifier`` must be callable as ``model(x, v=v, mask=mask)``.

    It ignores ``x`` (four-vector-only arm) but has to accept the same signature
    as every other arm, or it cannot be driven by the training harness.
    """
    lgatr = pytest.importorskip(
        "lgatr", reason="the official lgatr package is not installed"
    )
    del lgatr
    from dataloader.ragged_loader import RaggedShardDataset
    from variants.lgatr_model import LGATrJetClassifier

    ds = RaggedShardDataset(str(shard_dir), cache_size=1, events_per_shard=len(_COUNTS))
    x, v, mask, y = ds.get_batch(0, np.array([3, 4, 5]))

    torch.manual_seed(0)
    model = LGATrJetClassifier(
        num_classes=LOADER_NUM_CLASSES, mv_channels=4, s_channels=8,
        num_blocks=1, num_heads=2,
    ).eval()

    with torch.no_grad():
        logits = model(x, v=v, mask=mask)

    assert logits.shape == (3, LOADER_NUM_CLASSES)
    assert torch.isfinite(logits).all()

    with pytest.raises(ValueError, match="requires both v"):
        model(x, v=None, mask=mask)
