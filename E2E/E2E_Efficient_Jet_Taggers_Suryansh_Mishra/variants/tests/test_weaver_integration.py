"""Property-based test for weaver integration (design "Correctness Properties").

Feature: self-contained-kernels-migration.

This module holds Property 3: every ablation arm assembled by
``build_variant_part`` (``variants.VARIANTS``, including wrapper arms
``lloca`` and ``n8``) completes a CPU forward pass on random valid
``(x, v, mask)`` batches and returns finite logits of shape
``(B, num_classes)``.

Inputs (small weaver configs plus matched physically valid batches) are drawn
from the shared ``weaver_batches`` strategy in ``variants.tests.strategies``
(task 3.7). Models run in eval mode: weaver models contain dropout and
BatchNorm, so eval keeps the forward deterministic and avoids BatchNorm
batch-size-1 training quirks — Property 3 quantifies over forward-pass
validity, not training behavior (train-mode gradients are Property 2's job).
"""

from __future__ import annotations

import pytest
import torch
from hypothesis import given, settings
from hypothesis import strategies as st

from variants import VARIANTS, build_variant_part
from variants.tests.strategies import weaver_batches

VARIANT_NAMES = sorted(VARIANTS)


# ---------------------------------------------------------------------------
# Property 3
# ---------------------------------------------------------------------------

# Feature: self-contained-kernels-migration, Property 3: Variant-substituted weaver models complete a valid forward pass
@pytest.mark.parametrize("variant", VARIANT_NAMES)
@settings(max_examples=100, deadline=None)
@given(
    batch=weaver_batches(),
    seed=st.integers(min_value=0, max_value=2**32 - 1),
)
def test_variant_substituted_weaver_models_complete_forward_pass(
    variant, batch, seed
):
    """``build_variant_part(variant, **config)`` yields a weaver model that
    completes a CPU forward pass on a random valid ``(x, v, mask)`` batch and
    returns finite logits of shape ``(B, num_classes)``.

    **Validates: Requirements 1.4**
    """
    # Deterministic model construction per example (same pattern as the
    # block-level properties): all parameter init flows through this seed.
    torch.manual_seed(seed)
    model = build_variant_part(variant, **batch.config)
    model.eval()

    with torch.no_grad():
        logits = model(batch.x, v=batch.v, mask=batch.mask)

    B = batch.x.shape[0]
    num_classes = batch.config["num_classes"]

    assert logits.shape == (B, num_classes), (
        f"variant '{variant}' returned logits of shape {tuple(logits.shape)}, "
        f"expected {(B, num_classes)} "
        f"(P={batch.x.shape[-1]}, lengths={batch.lengths}, "
        f"config={batch.config})"
    )
    assert torch.isfinite(logits).all(), (
        f"variant '{variant}' produced non-finite logits "
        f"(B={B}, P={batch.x.shape[-1]}, lengths={batch.lengths}, "
        f"config={batch.config})"
    )
