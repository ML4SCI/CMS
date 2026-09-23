"""part_kernels: Custom GPU kernels for the Particle Transformer.

Drop-in optimization for weaver's ``ParticleTransformer``.

Quick start (eval, fused kernels)::

    from weaver.nn.model.ParticleTransformer import ParticleTransformer
    from part_kernels import OptimizedParticleTransformer

    orig = ParticleTransformer(...).cuda()
    opt = OptimizedParticleTransformer.from_pretrained(orig).eval()

Best performance (compile)::

    from part_kernels import optimize_part_model

    orig = ParticleTransformer(...).cuda()
    model, stats = optimize_part_model(orig, compile_mode="reduce-overhead")

Import safety (Requirements 2.6, 8.3, 3.3):

- ``import part_kernels`` eagerly runs :func:`_compat.assert_weaver_compat`
  so a wrong weaver-core pin fails loudly at import time.
- All public entry points are exposed via lazy module ``__getattr__``
  (PEP 562), so importing the package never imports ``triton``. GPU-only
  names (Triton kernels, fused modules) resolve on first attribute access
  and raise only then if triton is unavailable.
"""

from __future__ import annotations

import importlib
from typing import Any

from . import _compat

# Req 3.3: assert the weaver-core pin before exposing any public name.
_compat.assert_weaver_compat()

#: Public name -> defining submodule (relative to this package). Resolved
#: lazily on first attribute access so ``import part_kernels`` never touches
#: triton (Req 2.6, 8.3).
_LAZY_EXPORTS = {
    # runtime helpers (CPU-importable; GPU paths dispatch internally)
    "optimize_part_model": ".runtime",
    "unpatch_part_model": ".runtime",
    # autograd entry point (CPU-importable; triton imported inside dispatch)
    "fused_attention_with_bias": ".autograd.attention",
    # GPU-only modules/kernels (accessing these on CPU raises ImportError)
    "CUDAGraphInferenceWrapper": ".layers.cuda_graph_wrapper",
    "FusedPairMLP": ".layers.fused_pair_mlp",
    "OptimizedBlock": ".layers.optimized_model",
    "OptimizedPairEmbed": ".layers.optimized_model",
    "OptimizedParticleTransformer": ".layers.optimized_model",
    "fused_pairwise_lv_fts": ".triton.pairwise_kernel",
}

__all__ = sorted(_LAZY_EXPORTS)


def __getattr__(name: str) -> Any:
    """Lazily resolve public exports (PEP 562)."""
    try:
        module_name = _LAZY_EXPORTS[name]
    except KeyError:
        raise AttributeError(
            f"module {__name__!r} has no attribute {name!r}"
        ) from None
    module = importlib.import_module(module_name, __name__)
    value = getattr(module, name)
    globals()[name] = value  # cache: __getattr__ fires once per name
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(_LAZY_EXPORTS))
