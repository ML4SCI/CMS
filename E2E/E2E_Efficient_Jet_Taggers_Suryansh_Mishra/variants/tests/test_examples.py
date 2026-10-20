"""Example-based unit tests for the self-contained ``variants`` package.

Feature: self-contained-kernels-migration (task 3.11).

Deliberately few, fast structural checks that complement the Hypothesis
properties in ``test_block_properties.py`` / ``test_weaver_integration.py``
(design "Example-based unit tests" section):

- class-existence + ``__all__``-export checks for the variant blocks
  (Requirement 1.2);
- attribution/docstring checks for differential attention v1/v2
  (Requirement 1.3);
- ``WeaverBlockAdapter`` rejects ``x_cls`` — variant blocks are
  encoder-stack only (Requirement 1.4);
- ``build_variant_part("baseline")`` returns stock weaver blocks, and a
  variant arm substitutes the encoder stack only, leaving ``cls_blocks``
  as stock weaver (Requirement 1.4).
"""

from __future__ import annotations

import pytest
import torch
from torch import nn

from weaver.nn.model.ParticleTransformer import Block as WeaverBlock

import variants
from variants import (
    DifferentialAttentionBlock,
    DifferentialAttentionV2Block,
    MoEFeedforward,
    SoftmaxAttentionBlock,
    SparsemaxAttentionBlock,
    URotaryAttentionBlock,
    WeaverBlockAdapter,
    build_variant_part,
    sparsemax,
)
from variants.blocks import differential_attention, differential_attention_v2

# ---------------------------------------------------------------------------
# Class existence and package exports (Requirement 1.2)
# ---------------------------------------------------------------------------

BLOCK_CLASSES = [
    SparsemaxAttentionBlock,
    SoftmaxAttentionBlock,
    MoEFeedforward,
    DifferentialAttentionBlock,
    DifferentialAttentionV2Block,
    URotaryAttentionBlock,
]


@pytest.mark.parametrize("cls", BLOCK_CLASSES, ids=lambda c: c.__name__)
def test_variant_block_is_exported_nn_module(cls):
    """Each variant block is an ``nn.Module`` subclass exported by name
    from the ``variants`` package (present in ``__all__`` and resolvable
    as a top-level attribute).

    **Validates: Requirements 1.2**
    """
    assert issubclass(cls, nn.Module), f"{cls.__name__} is not an nn.Module"
    assert cls.__name__ in variants.__all__, (
        f"{cls.__name__} missing from variants.__all__"
    )
    assert getattr(variants, cls.__name__) is cls


def test_sparsemax_function_is_exported():
    """The ``sparsemax`` activation is a callable exported alongside its
    block (part of the sparsemax-attention lift).

    **Validates: Requirements 1.2**
    """
    assert callable(sparsemax)
    assert "sparsemax" in variants.__all__
    assert variants.sparsemax is sparsemax


# ---------------------------------------------------------------------------
# Attribution / docstring checks (Requirement 1.3)
# ---------------------------------------------------------------------------

def test_differential_attention_v1_docstring_attribution():
    """The v1 module docstring retains the Differential Transformer
    attribution (arXiv:2410.05258).

    **Validates: Requirements 1.3**
    """
    doc = differential_attention.__doc__
    assert doc is not None, "differential_attention has no module docstring"
    assert "arXiv:2410.05258" in doc


def test_differential_attention_v2_docstring_attribution():
    """The v2 module docstring retains both the Differential Transformer
    paper attribution (arXiv:2410.05258) and the upstream microsoft/unilm
    repository link.

    **Validates: Requirements 1.3**
    """
    doc = differential_attention_v2.__doc__
    assert doc is not None, "differential_attention_v2 has no module docstring"
    assert "arXiv:2410.05258" in doc
    assert "github.com/microsoft/unilm" in doc


# ---------------------------------------------------------------------------
# WeaverBlockAdapter rejects x_cls (Requirement 1.4)
# ---------------------------------------------------------------------------

_EMBED_DIM = 16
_NUM_HEADS = 2


def _small_adapted_block() -> WeaverBlockAdapter:
    torch.manual_seed(0)
    return WeaverBlockAdapter(
        SparsemaxAttentionBlock(
            embed_dim=_EMBED_DIM,
            num_heads=_NUM_HEADS,
            dropout=0.0,
            expansion_factor=2,
        )
    )


def test_weaver_block_adapter_rejects_x_cls():
    """A non-``None`` ``x_cls`` raises ``RuntimeError``: variant blocks are
    encoder-stack only, class attention must stay in weaver's stock
    ``cls_blocks``.

    **Validates: Requirements 1.4**
    """
    adapter = _small_adapted_block()
    x = torch.randn(2, 5, _EMBED_DIM)
    x_cls = torch.randn(2, 1, _EMBED_DIM)
    padding_mask = torch.zeros(2, 5, dtype=torch.bool)

    with pytest.raises(RuntimeError, match="encoder-stack only"):
        adapter(x, x_cls=x_cls, padding_mask=padding_mask)


def test_weaver_block_adapter_forwards_without_x_cls():
    """With ``x_cls=None`` (weaver's encoder-stack call) the adapter
    delegates to the wrapped block and preserves the ``(B, N, C)`` shape.

    **Validates: Requirements 1.4**
    """
    adapter = _small_adapted_block().eval()
    x = torch.randn(2, 5, _EMBED_DIM)
    padding_mask = torch.zeros(2, 5, dtype=torch.bool)

    with torch.no_grad():
        out = adapter(x, x_cls=None, padding_mask=padding_mask)

    assert out.shape == x.shape
    assert torch.isfinite(out).all()


# ---------------------------------------------------------------------------
# build_variant_part block substitution (Requirement 1.4)
# ---------------------------------------------------------------------------

_SMALL_CONFIG = dict(
    input_dim=8,
    num_classes=4,
    embed_dims=(16, 32, 16),
    pair_embed_dims=(8, 8),
    num_heads=2,
    num_layers=2,
    num_cls_layers=1,
    dropout=0.1,
    expansion_factor=2,
)


def test_build_variant_part_baseline_returns_stock_weaver_blocks():
    """``build_variant_part("baseline")`` returns a stock weaver model:
    every encoder block is weaver's ``Block``, none are adapters.

    **Validates: Requirements 1.4**
    """
    torch.manual_seed(0)
    model = build_variant_part("baseline", **_SMALL_CONFIG)

    assert len(model.blocks) == _SMALL_CONFIG["num_layers"]
    for i, block in enumerate(model.blocks):
        assert isinstance(block, WeaverBlock), (
            f"baseline blocks[{i}] is {type(block).__name__}, "
            "expected weaver Block"
        )
        assert not isinstance(block, WeaverBlockAdapter)
    for i, cls_block in enumerate(model.cls_blocks):
        assert isinstance(cls_block, WeaverBlock)


def test_build_variant_part_sparsemax_substitutes_encoder_stack_only():
    """A variant arm replaces every encoder block with a
    ``WeaverBlockAdapter`` wrapping the variant block, while ``cls_blocks``
    stay stock weaver ``Block``s (encoder-only substitution).

    **Validates: Requirements 1.4**
    """
    torch.manual_seed(0)
    model = build_variant_part("sparsemax", **_SMALL_CONFIG)

    assert len(model.blocks) == _SMALL_CONFIG["num_layers"]
    for i, adapter in enumerate(model.blocks):
        assert isinstance(adapter, WeaverBlockAdapter), (
            f"sparsemax blocks[{i}] is {type(adapter).__name__}, "
            "expected WeaverBlockAdapter"
        )
        assert isinstance(adapter.block, SparsemaxAttentionBlock)
    for i, cls_block in enumerate(model.cls_blocks):
        assert isinstance(cls_block, WeaverBlock), (
            f"cls_blocks[{i}] is {type(cls_block).__name__}, "
            "expected stock weaver Block"
        )
        assert not isinstance(cls_block, WeaverBlockAdapter)
