"""Self-contained architectural variants of the weaver-core ParticleTransformer.

Depends only on this package, PyTorch, and the pinned weaver-core release.

Layout
------
=========================  ======================================================
:mod:`variants.blocks`     interchangeable encoder sub-blocks — one swap per arm
                           (softmax / sparsemax / differential v1 / v2 attention,
                           dense and Mixture-of-Experts feed-forward,
                           U-as-rotation attention)
:mod:`variants.lloca`      the LLoCa arm: local Lorentz frame construction,
                           frame-aware attention, and the assembled model
:mod:`variants.n8`         the N8 arm: factorized Minkowski + rotary pair extras
:mod:`variants.weaver_adapter`  the factory: :func:`build_variant_part` returns a
                           ready ParT for any arm, all called as
                           ``model(x, v=v, mask=mask)``
:mod:`variants.optim`      Lion / Lookahead / EMA
:mod:`variants.lgatr_model`  the separate L-GATr classifier arm
=========================  ======================================================

Everything public is re-exported here, so ``from variants import X`` reaches any
symbol regardless of which subpackage owns it.

Entry point for the ablation::

    from variants import VARIANTS, build_variant_part
    model = build_variant_part("lloca", input_dim=16, num_classes=10)

The weaver-core compatibility assertion below is a deliberate minimal local
copy (not imported from ``part_kernels``) so that ``variants`` stays
importable independently of ``part_kernels``.
"""

REQUIRED_PIN = 'weaver-core @ git+https://github.com/hqucms/weaver-core.git@154db69'

#: Minimal set of weaver attributes the variants package relies on.
#: Dotted names are resolved with getattr, never called — pure existence check.
_REQUIRED_TARGETS = [
    ('weaver.nn.model.ParticleTransformer', 'ParticleTransformer'),
    ('weaver.nn.model.ParticleTransformer', 'PairEmbed'),
    ('weaver.nn.model.ParticleTransformer', 'PairEmbed._forward_sparse'),
    ('weaver.nn.model.ParticleTransformer', 'Block'),
]


def assert_weaver_compat() -> None:
    """Assert the installed weaver-core contains every required target.

    Raises ImportError naming REQUIRED_PIN if weaver-core is missing or any
    target is absent (Requirement 3.3). The message lists every missing target.
    """
    import importlib

    missing = []
    for module_path, dotted_attr in _REQUIRED_TARGETS:
        try:
            obj = importlib.import_module(module_path)
        except ImportError:
            missing.append(f'{module_path} (module not importable)')
            continue
        for part in dotted_attr.split('.'):
            obj = getattr(obj, part, None)
            if obj is None:
                missing.append(dotted_attr)
                break

    if missing:
        raise ImportError(
            'variants requires weaver-core pinned to a commit that contains '
            f'PairEmbed._forward_sparse (missing: {", ".join(missing)}). '
            'PyPI releases up to v0.4.17 do NOT contain it. '
            'Install the verified pin:\n'
            f'    pip install --no-deps "{REQUIRED_PIN}"'
        )


assert_weaver_compat()

from . import blocks, lloca, n8, tied
from .blocks import (
    DifferentialAttentionBlock,
    DifferentialAttentionV2Block,
    Feedforward,
    MoEFeedforward,
    SoftmaxAttentionBlock,
    SparsemaxAttentionBlock,
    URotaryAttentionBlock,
    sparsemax,
)
from .lloca import (
    FramesNet,
    LLoCaAttentionBlock,
    LLoCaParT,
    frames_from_vectors,
    invert_frames,
)
from .n8 import FactorizedAttentionBlock, N8ParT
from .tied import (
    DepthModulation,
    MoRRouter,
    TiedSchedule,
    build_tied_blocks,
    tied_cost_report,
)
from .weaver_adapter import (
    DEFAULT_MOE_PRESET,
    MOE_PRESETS,
    VARIANTS,
    WeaverBlockAdapter,
    build_variant_part,
    collect_moe_aux_loss,
)
from .lgatr_model import LGATrJetClassifier

__all__ = [
    'REQUIRED_PIN',
    'assert_weaver_compat',
    # subpackages
    'blocks',
    'lloca',
    'n8',
    # blocks and supporting modules
    'Feedforward',
    'SparsemaxAttentionBlock',
    'sparsemax',
    'SoftmaxAttentionBlock',
    'MoEFeedforward',
    'DifferentialAttentionBlock',
    'DifferentialAttentionV2Block',
    'URotaryAttentionBlock',
    # LLoCa arm
    'FramesNet',
    'frames_from_vectors',
    'invert_frames',
    'LLoCaAttentionBlock',
    'LLoCaParT',
    'N8ParT',
    'FactorizedAttentionBlock',
    # weaver adaptation
    'VARIANTS',
    'MOE_PRESETS',
    'DEFAULT_MOE_PRESET',
    'WeaverBlockAdapter',
    'build_variant_part',
    'collect_moe_aux_loss',
    # LGATr arm
    'LGATrJetClassifier',
]
