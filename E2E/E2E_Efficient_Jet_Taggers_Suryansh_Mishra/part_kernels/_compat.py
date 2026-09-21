"""weaver-core pin assertion, triton/CUDA availability, and Attention attribute shim.

Single source of truth for the weaver-core internals that ``part_kernels``
patches. ``PATCH_TARGETS`` is consumed both by the import-time assertion in
``part_kernels/__init__.py`` (Req 3.3) and by ``optimize_part_model``'s
startup assertion (Req 2.2), so a target that disappears after an upstream
re-pin can never be silently skipped.

The required pin is a git commit rather than a PyPI release because no PyPI
release up to and including v0.4.17 contains ``PairEmbed._forward_sparse``
(verified present at commit ``154db69``).
"""

from __future__ import annotations

import importlib

__all__ = [
    "REQUIRED_PIN",
    "PATCH_TARGETS",
    "assert_weaver_compat",
    "has_triton",
    "get_in_proj_params",
]

REQUIRED_PIN = 'weaver-core @ git+https://github.com/hqucms/weaver-core.git@154db69'

#: Every attribute the patcher and adapter rely on. Names are checked with
#: getattr, not called -- pure existence assertion.
#: Format: (module_path, dotted_attribute).
PATCH_TARGETS = [
    ('weaver.nn.model.ParticleTransformer', 'ParticleTransformer'),
    ('weaver.nn.model.ParticleTransformer', 'PairEmbed'),
    ('weaver.nn.model.ParticleTransformer', 'PairEmbed._forward_sparse'),
    ('weaver.nn.model.ParticleTransformer', 'PairEmbed._forward_dense'),
    ('weaver.nn.model.ParticleTransformer', 'PairEmbed.forward'),
    ('weaver.nn.model.ParticleTransformer', 'pairwise_lv_fts_pp'),
    ('weaver.nn.model.ParticleTransformer', 'Block'),
    ('weaver.nn.model.ParticleTransformer', 'Block.forward'),
    ('weaver.nn.model.ParticleTransformer', 'Attention'),
    ('weaver.nn.model.ParticleTransformer', 'Attention.forward'),
]


def _target_exists(module: object, dotted_attribute: str) -> bool:
    """Walk a dotted attribute path (e.g. ``PairEmbed._forward_sparse``)."""
    obj = module
    for part in dotted_attribute.split('.'):
        try:
            obj = getattr(obj, part)
        except AttributeError:
            return False
    return True


def assert_weaver_compat() -> None:
    """Assert every ``PATCH_TARGETS`` entry exists in the installed weaver-core.

    Raises ``ImportError`` naming :data:`REQUIRED_PIN` if weaver-core is
    missing or any patch target is absent (Req 3.3, 2.2). The message lists
    every missing target and the exact install command.
    """
    missing: list[str] = []
    module_cache: dict[str, object | None] = {}
    import_errors: dict[str, str] = {}

    for module_path, dotted_attribute in PATCH_TARGETS:
        if module_path not in module_cache:
            try:
                module_cache[module_path] = importlib.import_module(module_path)
            except Exception as exc:  # ImportError or anything raised on import
                module_cache[module_path] = None
                import_errors[module_path] = f"{type(exc).__name__}: {exc}"
        module = module_cache[module_path]
        if module is None or not _target_exists(module, dotted_attribute):
            missing.append(dotted_attribute)

    if not missing:
        return

    lines = [
        "part_kernels requires weaver-core pinned to a commit that contains",
        f"PairEmbed._forward_sparse (missing: {', '.join(missing)}).",
    ]
    for module_path, err in sorted(import_errors.items()):
        lines.append(f"(could not import {module_path}: {err})")
    lines += [
        "PyPI releases up to v0.4.17 do NOT contain it. Install the verified pin:",
        f'    pip install --no-deps "{REQUIRED_PIN}"',
    ]
    raise ImportError("\n".join(lines))


_HAS_TRITON: bool | None = None


def has_triton() -> bool:
    """True only if ``import triton`` succeeds AND ``torch.cuda.is_available()``.

    Never raises; the result is cached after the first call. On a CPU-only
    machine this returns ``False`` so all fused dispatch sites fall back to
    the weaver reference math (Req 8.3).
    """
    global _HAS_TRITON
    if _HAS_TRITON is None:
        try:
            import triton  # noqa: F401
            import torch

            _HAS_TRITON = bool(torch.cuda.is_available())
        except Exception:
            _HAS_TRITON = False
    return _HAS_TRITON


def get_in_proj_params(attn):
    """Return ``(weight, bias)`` of an attention module's fused QKV projection.

    Handles both weaver attention layouts:

    - 0.5.3 (pin ``154db69``): ``attn.in_proj`` is an ``nn.Linear`` ->
      ``(attn.in_proj.weight, attn.in_proj.bias)``.
    - 0.4.x (``nn.MultiheadAttention``-style): plain ``attn.in_proj_weight`` /
      ``attn.in_proj_bias`` tensor attributes.

    Raises ``AttributeError`` if neither layout is present.
    """
    in_proj = getattr(attn, 'in_proj', None)
    if in_proj is not None and hasattr(in_proj, 'weight'):
        return in_proj.weight, in_proj.bias
    if hasattr(attn, 'in_proj_weight'):
        return attn.in_proj_weight, getattr(attn, 'in_proj_bias', None)
    raise AttributeError(
        f"{type(attn).__name__} exposes neither 'in_proj' (weaver 0.5.3 nn.Linear) "
        "nor 'in_proj_weight' (weaver 0.4.x tensor); cannot extract the QKV "
        "input-projection parameters."
    )
