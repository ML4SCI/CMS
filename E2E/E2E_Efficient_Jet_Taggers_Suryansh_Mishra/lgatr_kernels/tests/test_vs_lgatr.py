"""Tests comparing our kernels against the actual lgatr package."""

import torch
import pytest
pytest.importorskip("lgatr")

# `lgatr` alone is not enough: the `..autograd.*` imports below pull in the Triton kernels, and
# `lgatr` is CPU-installable while `triton` is not. Guarding only `lgatr` therefore let this file
# reach a module-level `import triton` and fail collection on CPU. Every test here also allocates
# on `device="cuda"`, so CUDA is required too. (Added 2026-09-04.)
pytest.importorskip("triton", reason="lgatr_kernels Triton kernels are GPU-only")
if not torch.cuda.is_available():
    pytest.skip("lgatr_kernels kernel tests require CUDA", allow_module_level=True)

from lgatr.primitives.linear import equi_linear as lgatr_equi_linear
from lgatr.primitives.bilinear import geometric_product as lgatr_gp
from lgatr.primitives.normalization import equi_layer_norm as lgatr_equi_layer_norm
from ..autograd.equi_linear import triton_equi_linear
from ..autograd.geometric_product import triton_geometric_product
from ..autograd.equi_layernorm import triton_equi_layer_norm


class TestVsLGATrEquiLinear:
    def test_forward_match(self):
        x = torch.randn(64, 16, 16, device="cuda", dtype=torch.float32)
        w = torch.randn(32, 16, 10, device="cuda", dtype=torch.float32)
        torch.testing.assert_close(triton_equi_linear(x, w), lgatr_equi_linear(x, w), atol=1e-4, rtol=1e-4)

    def test_forward_match_batched(self):
        x = torch.randn(4, 32, 8, 16, device="cuda", dtype=torch.float32)
        w = torch.randn(16, 8, 10, device="cuda", dtype=torch.float32)
        torch.testing.assert_close(triton_equi_linear(x, w), lgatr_equi_linear(x, w), atol=1e-4, rtol=1e-4)

    def test_backward_match(self):
        x1 = torch.randn(32, 8, 16, device="cuda", dtype=torch.float32, requires_grad=True)
        x2 = x1.detach().clone().requires_grad_(True)
        w1 = torch.randn(16, 8, 10, device="cuda", dtype=torch.float32, requires_grad=True)
        w2 = w1.detach().clone().requires_grad_(True)
        lgatr_equi_linear(x1, w1).sum().backward()
        triton_equi_linear(x2, w2).sum().backward()
        torch.testing.assert_close(x2.grad, x1.grad, atol=1e-3, rtol=1e-3)
        torch.testing.assert_close(w2.grad, w1.grad, atol=1e-3, rtol=1e-3)

class TestVsLGATrGeometricProduct:
    def test_forward_match(self):
        x = torch.randn(256, 16, device="cuda", dtype=torch.float32)
        y = torch.randn(256, 16, device="cuda", dtype=torch.float32)
        torch.testing.assert_close(triton_geometric_product(x, y), lgatr_gp(x, y), atol=1e-5, rtol=1e-5)

    def test_forward_match_batched(self):
        x = torch.randn(4, 32, 8, 16, device="cuda", dtype=torch.float32)
        y = torch.randn(4, 32, 8, 16, device="cuda", dtype=torch.float32)
        torch.testing.assert_close(triton_geometric_product(x, y), lgatr_gp(x, y), atol=1e-5, rtol=1e-5)

    def test_backward_match(self):
        x1 = torch.randn(64, 16, device="cuda", dtype=torch.float32, requires_grad=True)
        x2 = x1.detach().clone().requires_grad_(True)
        y1 = torch.randn(64, 16, device="cuda", dtype=torch.float32, requires_grad=True)
        y2 = y1.detach().clone().requires_grad_(True)
        lgatr_gp(x1, y1).sum().backward()
        triton_geometric_product(x2, y2).sum().backward()
        torch.testing.assert_close(x2.grad, x1.grad, atol=1e-4, rtol=1e-4)
        torch.testing.assert_close(y2.grad, y1.grad, atol=1e-4, rtol=1e-4)

class TestVsLGATrLayerNorm:
    def test_forward_match(self):
        x = torch.randn(32, 16, 16, device="cuda", dtype=torch.float32)
        torch.testing.assert_close(triton_equi_layer_norm(x), lgatr_equi_layer_norm(x), atol=1e-4, rtol=1e-4)

    def test_forward_match_batched(self):
        x = torch.randn(4, 32, 8, 16, device="cuda", dtype=torch.float32)
        torch.testing.assert_close(triton_equi_layer_norm(x), lgatr_equi_layer_norm(x), atol=1e-4, rtol=1e-4)
