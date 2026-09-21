"""Unit tests for ``part_kernels/_compat.py`` (Requirements 2.2, 3.3).

Covers:
- ``assert_weaver_compat`` happy path against the installed weaver-core pin.
- ``assert_weaver_compat`` failure path when ``PairEmbed._forward_sparse`` is
  missing: the ``ImportError`` message must name the missing target and the
  exact ``pip install --no-deps`` pin command.
- ``has_triton`` returns ``False`` on this CPU-only machine and never raises.
- ``get_in_proj_params`` handles both weaver attention layouts (0.5.3
  ``in_proj`` nn.Linear and 0.4.x ``in_proj_weight`` tensor attributes) and
  raises ``AttributeError`` when neither is present.

NOTE: ``_compat`` is loaded directly from its file path instead of via
``import part_kernels`` because the package ``__init__`` is not yet
import-safe on CPU (module-scope triton imports; fixed by a concurrent task).
Loading the single module keeps these tests independent of that work.
"""

import importlib.util
from pathlib import Path

import pytest
import torch

_COMPAT_PATH = Path(__file__).resolve().parents[1] / "_compat.py"


def _load_compat():
    """Load ``part_kernels/_compat.py`` as a standalone module.

    Deliberately avoids ``import part_kernels``: the package ``__init__``
    imports triton at module scope and fails on this CPU-only machine.
    ``_compat`` itself only needs stdlib + the installed weaver-core.
    """
    spec = importlib.util.spec_from_file_location("part_kernels_compat", _COMPAT_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_compat = _load_compat()


# =====================================================================
# assert_weaver_compat
# =====================================================================

def test_assert_weaver_compat_happy_path():
    """With the installed pin (154db69) every patch target exists: no raise."""
    assert _compat.assert_weaver_compat() is None


def test_assert_weaver_compat_missing_forward_sparse(monkeypatch):
    """Deleting PairEmbed._forward_sparse must raise ImportError whose message
    names the missing target and the exact --no-deps pin install command."""
    from weaver.nn.model.ParticleTransformer import PairEmbed

    # monkeypatch restores the real method after the test.
    monkeypatch.delattr(PairEmbed, "_forward_sparse")

    with pytest.raises(ImportError) as excinfo:
        _compat.assert_weaver_compat()

    msg = str(excinfo.value)
    assert "PairEmbed._forward_sparse" in msg
    assert "pip install --no-deps" in msg
    assert "weaver-core @ git+https://github.com/hqucms/weaver-core.git@154db69" in msg


def test_assert_weaver_compat_recovers_after_restore(monkeypatch):
    """The failure-path deletion is transient: once restored, compat passes.

    Guards against the previous test leaking state into later suites.
    """
    from weaver.nn.model.ParticleTransformer import PairEmbed

    with monkeypatch.context() as m:
        m.delattr(PairEmbed, "_forward_sparse")
        with pytest.raises(ImportError):
            _compat.assert_weaver_compat()

    # monkeypatch context exited -> method restored -> happy path again.
    assert hasattr(PairEmbed, "_forward_sparse")
    assert _compat.assert_weaver_compat() is None


# =====================================================================
# has_triton
# =====================================================================

def test_has_triton_false_on_cpu_and_never_raises(monkeypatch):
    """On this CPU-only machine has_triton() is False; repeated (cached)
    calls return the same value without raising."""
    # Reset the module-level cache so this test exercises the real probe,
    # not a value cached by an earlier call.
    monkeypatch.setattr(_compat, "_HAS_TRITON", None)

    result = _compat.has_triton()
    assert result is False
    # Second call hits the cache and must agree.
    assert _compat.has_triton() is False


# =====================================================================
# get_in_proj_params
# =====================================================================

def test_get_in_proj_params_weaver_053_layout():
    """Real weaver (pin 154db69) Attention: in_proj is an nn.Linear."""
    from weaver.nn.model.ParticleTransformer import Attention

    attn = Attention(16, 2)
    weight, bias = _compat.get_in_proj_params(attn)

    assert weight is attn.in_proj.weight
    assert bias is attn.in_proj.bias
    assert weight.shape == (3 * 16, 16)
    assert bias.shape == (3 * 16,)


def test_get_in_proj_params_weaver_04x_layout():
    """0.4.x nn.MultiheadAttention-style layout: plain tensor attributes."""

    class Fake04xAttention:
        def __init__(self):
            self.in_proj_weight = torch.randn(3 * 8, 8)
            self.in_proj_bias = torch.randn(3 * 8)

    attn = Fake04xAttention()
    weight, bias = _compat.get_in_proj_params(attn)

    assert weight is attn.in_proj_weight
    assert bias is attn.in_proj_bias


def test_get_in_proj_params_neither_layout_raises():
    """An object exposing neither layout raises a descriptive AttributeError."""

    class NotAnAttention:
        pass

    with pytest.raises(AttributeError, match="in_proj"):
        _compat.get_in_proj_params(NotAnAttention())
