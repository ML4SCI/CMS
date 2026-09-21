"""LLoCa — Lorentz Local Canonicalization for the Particle Transformer.

Reference: Spinner, Favaro, Lippmann, Pitz, Gerhartz, Plehn, Hamprecht,
"Lorentz Local Canonicalization: How to Make Any Network Lorentz-Equivariant",
`arXiv:2505.20280 <https://arxiv.org/abs/2505.20280>`_.  Companion paper
arXiv:2508.14898.  Upstream: https://github.com/heidelberg-hepml/lloca

Rather than building equivariance into the layers, LLoCa predicts a local
Lorentz frame per particle and expresses that particle's features in its own
frame, which makes them invariant.  An ordinary (non-equivariant) backbone can
then process them, and messages between particles are transported through the
frame-to-frame transformation.

The three modules split along the boundary of "geometry", "layer" and
"assembly", and are separated because only the first is pure mathematics and
therefore exactly testable:

=================  ==============================================================
:mod:`frames`      Minkowski algebra, the boost/Gram-Schmidt frame construction
                   (Alg. 1), and the equivariant frame predictor ``FramesNet``
:mod:`attention`   the encoder block, which transports tensorial messages
                   between frames (Eq. 11)
:mod:`part`        ``LLoCaParT``: predicts frames, canonicalizes the token
                   features, and delegates to a weaver ``ParticleTransformer``
=================  ==============================================================

The frame algebra in :mod:`frames` satisfies exact identities (``L g L^T = g``,
``L(Lambda v) = L(v) Lambda^-1``) that are verified to ~1e-14 in
``variants/tests/test_lloca_properties.py``.  A subtly wrong frame still
produces finite logits and trains normally -- it just silently stops being
equivariant -- so those identities are the arm's real correctness contract.
"""

from .attention import LLoCaAttentionBlock, default_representation
from .frames import (
    MINKOWSKI_SIGNATURE,
    REFERENCE_PARTICLES,
    FramesNet,
    apply_metric,
    boost_matrix,
    frames_from_vectors,
    gram_schmidt_3d,
    invert_frames,
    minkowski_dot,
    minkowski_norm,
    to_time_first,
    to_weaver_order,
)
from .part import NUM_KINEMATIC_FEATURES, LLoCaParT, local_kinematic_features

__all__ = [
    # geometry
    "MINKOWSKI_SIGNATURE",
    "REFERENCE_PARTICLES",
    "minkowski_dot",
    "minkowski_norm",
    "apply_metric",
    "to_time_first",
    "to_weaver_order",
    "boost_matrix",
    "gram_schmidt_3d",
    "frames_from_vectors",
    "invert_frames",
    "FramesNet",
    # layer
    "LLoCaAttentionBlock",
    "default_representation",
    # assembled arm
    "LLoCaParT",
    "NUM_KINEMATIC_FEATURES",
    "local_kinematic_features",
]
