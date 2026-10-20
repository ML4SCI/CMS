"""Gauge-invariant block similarity — the repair for R0's retracted metric.

R0 (`ablation/layer_redundancy.py::block_similarity`) compared encoder blocks by cosine similarity of
their raw flattened weights, found them at the √2 random-init null, and concluded "do not tie the
blocks". That conclusion was **retracted on 2026-09-06**: the metric is blind to the symmetries that
leave a block's function unchanged, so it reports "as dissimilar as independent random inits" for
blocks that compute the *identical* function — and that sentence is therefore also true of a model
that IS tied, which makes it uninformative about tying.

These tests pin both halves of the repair, because either alone would be misleading:

* the **defect** — a symmetry-moved block is functionally identical and the raw cosine says otherwise;
* the **fix** — :func:`gauge_invariant_composites` says 1.0 for that same pair, *and* still
  discriminates genuinely independent initialisations (a metric that returned 1.0 for everything
  would "pass" the first test while being useless).

The FFN case is the sharpest: the attention symmetry is *continuous*, so how badly it corrupts the
raw cosine depends on how far the symmetry element sits from the identity, but permuting the 512 FFN
hidden units is *discrete* and destroys the cosine unconditionally. The FFN is 66 % of a block, so
R0's most heavily weighted group is the one where its metric fails hardest.
"""

from __future__ import annotations

import copy

import pytest
import torch

from ablation.config import AblationConfig
from ablation.layer_redundancy import (
    _flat_groups,
    gauge_invariant_composites,
    gauge_invariant_similarity,
)
from variants import build_variant_part

_HEADS = 8
_EMBED = 128


def _block():
    torch.manual_seed(0)
    model = build_variant_part("baseline", **AblationConfig(arm="baseline").model_kwargs())
    return copy.deepcopy(model.blocks[0]).eval()


def _apply_symmetry(block, scale: float = 0.05, seed: int = 0):
    """Move a block by an element of its exact symmetry group, in place.

    Three independent families, all of which provably leave the block's function alone:

    * ``Wq -> M Wq``, ``Wk -> M^-T Wk`` per head. The attention logit is ``x^T Wq^T Wk y`` and
      ``M^T M^-T = I``, so the bilinear form is untouched. Biases transform with their weights.
    * ``Wv -> N Wv``, ``O -> O N^-1`` per head. The output is ``O (Wv y)``.
    * a permutation of the 512 FFN hidden units, applied to ``fc1`` rows, ``fc2`` columns and the
      ``post_fc_norm`` affine. This one is exact for any pointwise activation, which a general
      ``GL(512)`` would not be.
    """
    gen = torch.Generator().manual_seed(seed)
    head_dim = _EMBED // _HEADS
    with torch.no_grad():
        weight = block.attn.in_proj.weight
        bias = block.attn.in_proj.bias
        out = block.attn.out_proj.weight
        for head in range(_HEADS):
            lo, hi = head * head_dim, (head + 1) * head_dim
            m = torch.eye(head_dim) + scale * torch.randn(head_dim, head_dim, generator=gen)
            m_inv_t = torch.linalg.inv(m).T
            weight[lo:hi] = m @ weight[lo:hi].clone()
            bias[lo:hi] = m @ bias[lo:hi].clone()
            weight[_EMBED + lo:_EMBED + hi] = m_inv_t @ weight[_EMBED + lo:_EMBED + hi].clone()
            bias[_EMBED + lo:_EMBED + hi] = m_inv_t @ bias[_EMBED + lo:_EMBED + hi].clone()

            n = torch.eye(head_dim) + scale * torch.randn(head_dim, head_dim, generator=gen)
            n_inv = torch.linalg.inv(n)
            weight[2 * _EMBED + lo:2 * _EMBED + hi] = (
                n @ weight[2 * _EMBED + lo:2 * _EMBED + hi].clone()
            )
            bias[2 * _EMBED + lo:2 * _EMBED + hi] = (
                n @ bias[2 * _EMBED + lo:2 * _EMBED + hi].clone()
            )
            out[:, lo:hi] = out[:, lo:hi].clone() @ n_inv

        perm = torch.randperm(block.fc1.weight.shape[0], generator=gen)
        block.fc1.weight.copy_(block.fc1.weight[perm].clone())
        block.fc1.bias.copy_(block.fc1.bias[perm].clone())
        block.fc2.weight.copy_(block.fc2.weight[:, perm].clone())
        norm = getattr(block, "post_fc_norm", None)
        if norm is not None and getattr(norm, "weight", None) is not None:
            norm.weight.copy_(norm.weight[perm].clone())
            if norm.bias is not None:
                norm.bias.copy_(norm.bias[perm].clone())
    return block


def _probe():
    gen = torch.Generator().manual_seed(5)
    x = torch.randn(4, 40, _EMBED, generator=gen)
    padding = torch.zeros(4, 40, dtype=torch.bool)
    padding[:, 30:] = True
    bias = torch.randn(4, _HEADS, 40, 40, generator=gen)
    return x, padding, (bias + bias.transpose(-1, -2)) / 2


def _cosine(u, v) -> float:
    return float(u @ v) / (float(u.norm()) * float(v.norm()))


def test_the_symmetry_really_preserves_the_function() -> None:
    """Everything else here is meaningless if this fails — it is what makes the pair a fair test."""
    original, moved = _block(), _apply_symmetry(_block())
    x, padding, bias = _probe()
    with torch.no_grad():
        a = original(x, x_cls=None, padding_mask=padding, attn_mask=bias)
        b = moved(x, x_cls=None, padding_mask=padding, attn_mask=bias)
    relative = float((a - b).abs().max()) / float(a.abs().max())
    assert relative < 1e-5, f"symmetry changed the function (relative {relative:.2e})"


def test_raw_weight_cosine_fails_on_a_functionally_identical_block() -> None:
    """R0's defect, pinned. The FFN group is the unconditional case: permuting hidden units is a
    DISCRETE symmetry, so it destroys the cosine no matter how small the rest of the move is."""
    original, moved = _block(), _apply_symmetry(_block())
    flat_a, flat_b = _flat_groups(original), _flat_groups(moved)
    assert _cosine(flat_a["ffn"], flat_b["ffn"]) < 0.1, (
        "if this ever passes, the FFN permutation stopped being applied and the test is vacuous"
    )


def test_gauge_invariants_are_one_on_a_functionally_identical_block() -> None:
    """The fix. Tolerance is tight on purpose: these are exact identities up to float64 rounding."""
    original, moved = _block(), _apply_symmetry(_block())
    comp_a = gauge_invariant_composites(original, _HEADS)
    comp_b = gauge_invariant_composites(moved, _HEADS)
    for group in ("qk", "vo", "ffn"):
        assert _cosine(comp_a[group], comp_b[group]) == pytest.approx(1.0, abs=1e-6), group


def test_gauge_invariants_still_discriminate_independent_inits() -> None:
    """Without this, a metric that returned 1.0 for everything would pass the test above.

    This is the half R0 got right and the repair must not lose.
    """
    original = _block()
    torch.manual_seed(99)
    other = build_variant_part(
        "baseline", **AblationConfig(arm="baseline").model_kwargs()
    ).eval().blocks[0]
    comp_a = gauge_invariant_composites(original, _HEADS)
    comp_b = gauge_invariant_composites(other, _HEADS)
    for group in ("qk", "vo", "ffn"):
        assert abs(_cosine(comp_a[group], comp_b[group])) < 0.05, group


def test_gauge_invariant_similarity_matches_the_report_shape() -> None:
    """It has to drop into the same report `block_similarity` feeds, or it will not get used."""
    torch.manual_seed(0)
    model = build_variant_part("baseline", **AblationConfig(arm="baseline").model_kwargs())
    result = gauge_invariant_similarity(list(model.blocks), _HEADS)
    assert result["num_blocks"] == 8
    assert set(result["groups"]) >= {"qk", "vo", "ffn"}
    for entry in result["groups"].values():
        assert len(entry["cosine"]) == 8 and len(entry["cosine"][0]) == 8
        for key in ("mean_offdiag_cosine", "max_offdiag_cosine",
                    "mean_offdiag_relative_distance", "numel_per_block"):
            assert key in entry


def test_composites_use_the_shapes_attention_actually_computes() -> None:
    """`qk` is the bilinear form in the logit and `vo` the output map, one (128,128) per head."""
    comp = gauge_invariant_composites(_block(), _HEADS)
    assert comp["qk"].numel() == _HEADS * _EMBED * _EMBED
    assert comp["vo"].numel() == _HEADS * _EMBED * _EMBED
    assert comp["ffn"].numel() == _EMBED * _EMBED


def test_composites_reject_a_block_without_attention() -> None:
    with pytest.raises(AttributeError, match="in_proj"):
        gauge_invariant_composites(torch.nn.Linear(4, 4), _HEADS)
