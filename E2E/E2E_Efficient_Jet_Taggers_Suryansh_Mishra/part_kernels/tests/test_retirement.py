"""Retirement checks for the legacy borrowed-tree code paths (Requirements 2.4, 2.5).

Task 5.5 deleted (DD2 -- deletion, no legacy flag):

- ``runtime.py::optimize_src_part_model`` (patcher for the borrowed
  ``src.models`` ParticleTransformer reimplementation)
- ``triton/pairwise_kernel.py::_pairwise_eta_kernel`` and
  ``fused_pairwise_eta_fts`` (eta-feature math from the borrowed
  ``src.models.processor._get_interaction``)

These tests pin that retirement in place:

1. The legacy names are absent from ``part_kernels.__all__`` and attribute
   access on the package raises ``AttributeError``.
2. The legacy identifier strings do not appear in the source of the modules
   that used to define/export them (catches textual re-introduction, e.g.
   a revert or copy-paste). The ``tests/`` directory is deliberately NOT
   scanned: ``test_parity.py`` may reference the old names until its own
   rewrite (task 5.7) lands.
3. Nothing legacy is importable: ``from part_kernels.runtime import
   optimize_src_part_model`` raises ``ImportError``; ``from
   part_kernels.triton.pairwise_kernel import fused_pairwise_eta_fts``
   raises ``ImportError`` in either failure mode -- name gone (GPU box with
   triton installed) or the module-scope ``import triton`` failing with
   ``ModuleNotFoundError`` on this CPU-only machine (a subclass of
   ``ImportError``). Success is a failure either way.
4. Sanity: the surviving public API (``optimize_part_model``,
   ``unpatch_part_model``, ``fused_attention_with_bias``) is still exported
   and resolvable on CPU, so retirement did not over-delete.

Unlike ``test_compat.py`` (which loads ``_compat.py`` directly from its file
path), these tests import the real package: exercising the actual
``part_kernels`` namespace IS the point of a retirement check, and
``import part_kernels`` is now CPU-safe (lazy PEP 562 exports).
"""

from pathlib import Path

import pytest

import part_kernels as part_kernels

# The tests/ conftest shim stubs the package only when the real import fails
# (pre-import-safety environments). Retirement can only be verified against
# the real package.
if getattr(part_kernels, "_PART_KERNELS_STUBBED", False):
    pytest.skip(
        "part_kernels could not be imported; retirement checks need the real package",
        allow_module_level=True,
    )

_PKG_DIR = Path(__file__).resolve().parents[1]  # .../part_kernels

#: Legacy identifiers deleted by task 5.5. NOTE: no substring collisions with
#: surviving names (fused_pairwise_lv_fts / _pairwise_lv_kernel differ).
_LEGACY_NAMES = (
    "optimize_src_part_model",
    "fused_pairwise_eta_fts",
    "_pairwise_eta_kernel",
)

#: Modules that used to define or export the legacy names. tests/ is
#: intentionally excluded (see module docstring).
_SCANNED_SOURCES = (
    _PKG_DIR / "__init__.py",
    _PKG_DIR / "runtime.py",
    _PKG_DIR / "triton" / "pairwise_kernel.py",
)


# =====================================================================
# 1. Legacy names absent from the package namespace and __all__
# =====================================================================

@pytest.mark.parametrize("name", ["optimize_src_part_model", "fused_pairwise_eta_fts"])
def test_legacy_name_not_in_all(name):
    """Deleted entry points must not be advertised in __all__ (Req 2.4)."""
    assert name not in part_kernels.__all__


@pytest.mark.parametrize("name", ["optimize_src_part_model", "fused_pairwise_eta_fts"])
def test_legacy_name_not_gettable(name):
    """Attribute access must raise AttributeError, not lazily resolve (Req 2.4, 2.5)."""
    with pytest.raises(AttributeError):
        getattr(part_kernels, name)


def test_legacy_names_not_in_dir():
    """The lazy-export __dir__ must not list any legacy name."""
    listing = set(dir(part_kernels))
    for name in _LEGACY_NAMES:
        assert name not in listing


# =====================================================================
# 2. Source-level scan: legacy identifiers do not reappear in the
#    modules that used to hold them
# =====================================================================

@pytest.mark.parametrize(
    "source_path", _SCANNED_SOURCES, ids=lambda p: str(p.relative_to(_PKG_DIR))
)
def test_legacy_identifiers_absent_from_source(source_path):
    """The deleted identifiers must not appear textually in the source files
    that defined or exported them -- catches re-introduction (Req 2.4)."""
    assert source_path.is_file(), f"expected source file missing: {source_path}"
    text = source_path.read_text()
    for name in _LEGACY_NAMES:
        assert name not in text, (
            f"legacy identifier {name!r} reappeared in {source_path}"
        )


# =====================================================================
# 3. Nothing legacy is importable
# =====================================================================

def test_optimize_src_part_model_not_importable_from_runtime():
    """part_kernels.runtime imports fine on CPU, but the legacy patcher is
    gone: importing the name must raise ImportError (Req 2.4, 2.5)."""
    with pytest.raises(ImportError):
        from part_kernels.runtime import optimize_src_part_model  # noqa: F401


def test_fused_pairwise_eta_fts_not_importable_from_triton_module():
    """Either failure mode proves the name is unreachable; success would mean
    the legacy kernel came back (Req 2.4, 2.5).

    - CPU-only machine (triton not installed): the module-scope
      ``import triton`` raises ModuleNotFoundError (subclass of ImportError).
    - GPU box with triton installed: the module imports but the deleted name
      raises a plain ImportError.
    """
    with pytest.raises(ImportError):
        from part_kernels.triton.pairwise_kernel import (  # noqa: F401
            fused_pairwise_eta_fts,
        )


# =====================================================================
# 4. Surviving public API sanity: retirement did not over-delete
# =====================================================================

@pytest.mark.parametrize(
    "name", ["optimize_part_model", "unpatch_part_model", "fused_attention_with_bias"]
)
def test_current_public_api_exported_and_resolvable(name):
    """The weaver-targeted API stays in __all__ and resolves on CPU (Req 2.4)."""
    assert name in part_kernels.__all__
    obj = getattr(part_kernels, name)
    assert callable(obj)
