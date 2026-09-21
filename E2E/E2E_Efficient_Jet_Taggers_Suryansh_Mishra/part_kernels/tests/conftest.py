"""Collection shim for ``part_kernels/tests`` on CPU-only machines.

pytest creates a ``Package`` collector for ``part_kernels`` (it has an
``__init__.py``) and imports that ``__init__`` during test setup. Until the
package ``__init__`` is import-safe on CPU (module-scope triton imports,
being fixed by the concurrent import-safety task), that import errors out
every test in this directory -- even tests that never touch the package,
like ``test_compat.py`` which loads ``_compat.py`` directly from its file
path.

Behavior:

- If ``import part_kernels`` succeeds (task 5.3 landed, or a GPU box with
  triton), this shim does nothing and the real package is used.
- Otherwise it installs a placeholder module under the package name so
  pytest's Package-setup import is served from ``sys.modules`` instead of
  executing the failing ``__init__``. The placeholder keeps a real
  ``__path__`` so genuine submodules (e.g. ``part_kernels.tests``) still
  resolve. Tests that need the real package attributes must check
  ``getattr(part_kernels, "_PART_KERNELS_STUBBED", False)`` and skip.

NOTE: this directory intentionally has no ``__init__.py``; the conftest must
load as a standalone module, not as ``part_kernels.tests.conftest`` (which
would import the failing package ``__init__`` before the shim could run).
"""

import sys
import types
from pathlib import Path

_PKG_DIR = Path(__file__).resolve().parents[1]  # .../part_kernels
_PKG_PARENT = _PKG_DIR.parent                   # .../gsoc_26 (repo root)

if str(_PKG_PARENT) not in sys.path:
    sys.path.insert(0, str(_PKG_PARENT))

try:
    import part_kernels as part_kernels  # noqa: F401
except Exception:
    # Drop partially-initialized modules left behind by the failed import.
    for _name in [m for m in sys.modules
                  if m == "part_kernels" or m.startswith("part_kernels.")]:
        del sys.modules[_name]
    _stub = types.ModuleType("part_kernels")
    _stub.__file__ = str(_PKG_DIR / "__init__.py")
    _stub.__path__ = [str(_PKG_DIR)]
    _stub._PART_KERNELS_STUBBED = True
    sys.modules["part_kernels"] = _stub
