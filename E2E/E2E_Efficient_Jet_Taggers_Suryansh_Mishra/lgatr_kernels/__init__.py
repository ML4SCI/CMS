"""lgatr_kernels: Custom GPU kernels for the L-GATr architecture.

Drop-in optimization for the official ``lgatr`` package.

Quick start (eager, 1.5x)::

    from lgatr_kernels import patch_lgatr, fuse_equi_linear_layers
    patch_lgatr()
    model = LorentzGATr(config).cuda()
    fuse_equi_linear_layers(model)

Best performance (compile, 2.5x)::

    from lgatr_kernels import optimize_lgatr_model
    model = LorentzGATr(config).cuda()
    model, stats = optimize_lgatr_model(
        model, use_compile_patches=True, compile_mode="reduce-overhead",
    )

Import policy
-------------
All public entry points are exposed via lazy module ``__getattr__`` (PEP 562), so importing
this package never imports ``triton``. GPU-only names resolve on first attribute access and
raise only then if triton is unavailable.

This mirrors ``part_kernels``, which already documents the same policy in its own docstring.
Before 2026-09-04 this module imported all four submodules eagerly, and ``primitives`` ->
``autograd.geometric_product`` -> ``triton.geometric_product_kernel`` -> ``import triton``, so
the package was **un-importable on any machine without triton** — which made all six test files
under ``lgatr_kernels/tests/`` fail at *collection* on CPU, not merely skip. A test that cannot
be collected cannot be reported as skipped, so the suite silently lost six files.
"""

from __future__ import annotations

import importlib
from typing import Any

#: Public name -> submodule that defines it. Nothing here is imported until first access.
_LAZY_EXPORTS = {
    # CPU-importable: pure-python patching/fusion helpers that dispatch internally
    "fuse_equi_linear_layers": ".layers",
    "FusedEquiLinear": ".layers",
    "patch_lgatr_compile": ".compile_patches",
    "optimize_lgatr_model": ".runtime",
    "LorentzGATrGraphWrapper": ".runtime",
    # GPU-only: pulls in the Triton geometric-product kernel, so raises on CPU
    "patch_lgatr": ".primitives",
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
