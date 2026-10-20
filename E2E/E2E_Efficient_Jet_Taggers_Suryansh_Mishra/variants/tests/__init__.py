"""Test package for the self-contained ``variants`` package.

Unlike ``part_kernels/tests`` (kept init-less because the ``part_kernels``
package ``__init__`` was not import-safe on CPU), this directory IS a
package: ``import variants`` succeeds on CPU (weaver-core is pinned and
installed), so pytest's package-collection import of ``variants/__init__.py``
is safe, and the shared Hypothesis strategies are importable as
``variants.tests.strategies`` by every property-test module.
"""
