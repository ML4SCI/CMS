"""Tests for the fused geometric product kernel."""

import torch
import pytest
import pytest

# GPU-only: these exercise the Triton kernels directly and allocate on ``device="cuda"``.
# Guard BEFORE the kernel imports -- a module-level ``import triton`` failure is a collection
# ERROR, not a skip, and an uncollectable file is not reported as skipped, so all five of these
# silently disappeared from the suite on any CPU machine. ``importorskip`` turns that into a
# visible skip. (Added 2026-09-04 alongside making ``lgatr_kernels/__init__`` lazy.)
pytest.importorskip("triton", reason="lgatr_kernels Triton kernels are GPU-only")
_torch = pytest.importorskip("torch")
if not _torch.cuda.is_available():
    pytest.skip("lgatr_kernels kernel tests require CUDA", allow_module_level=True)

from ..codegen.cayley_table import compute_cayley_table
from ..triton.geometric_product_kernel import geometric_product_forward
from ..autograd.geometric_product import triton_geometric_product


def _build_gp_tensor(device="cuda", dtype=torch.float64):
    table = compute_cayley_table()
    gp = torch.zeros(16, 16, 16, device=device, dtype=dtype)
    for entry in table:
        gp[entry.c, entry.a, entry.b] = entry.sign
    return gp

def _reference_gp(x, y, gp_tensor):
    return torch.einsum("ijk,...j,...k->...i", gp_tensor, x, y)

@pytest.fixture
def gp_tensor():
    return _build_gp_tensor()

class TestGeometricProductForward:
    def test_basic_correctness(self, gp_tensor):
        x = torch.randn(128, 16, device="cuda", dtype=torch.float64)
        y = torch.randn(128, 16, device="cuda", dtype=torch.float64)
        torch.testing.assert_close(geometric_product_forward(x, y), _reference_gp(x, y, gp_tensor), atol=1e-10, rtol=1e-10)

    def test_batched(self, gp_tensor):
        x = torch.randn(4, 32, 8, 16, device="cuda", dtype=torch.float64)
        y = torch.randn(4, 32, 8, 16, device="cuda", dtype=torch.float64)
        torch.testing.assert_close(geometric_product_forward(x, y), _reference_gp(x, y, gp_tensor), atol=1e-10, rtol=1e-10)

    def test_zero_bivector(self, gp_tensor):
        x = torch.randn(64, 16, device="cuda", dtype=torch.float64)
        y = torch.randn(64, 16, device="cuda", dtype=torch.float64)
        out = geometric_product_forward(x, y, zero_bivector=True)
        assert (out[:, 5:11] == 0).all()
        ref = _reference_gp(x, y, gp_tensor); ref[:, 5:11] = 0
        torch.testing.assert_close(out, ref, atol=1e-10, rtol=1e-10)

    def test_scalar_times_scalar(self):
        x = torch.zeros(1, 16, device="cuda", dtype=torch.float64)
        y = torch.zeros(1, 16, device="cuda", dtype=torch.float64)
        x[0, 0] = 3.0; y[0, 0] = 5.0
        assert abs(geometric_product_forward(x, y)[0, 0].item() - 15.0) < 1e-10

    def test_vector_self_product(self):
        x = torch.zeros(1, 16, device="cuda", dtype=torch.float64)
        x[0, 1] = 1.0  # e0: e0*e0 = +1
        assert abs(geometric_product_forward(x, x)[0, 0].item() - 1.0) < 1e-10
        x = torch.zeros(1, 16, device="cuda", dtype=torch.float64)
        x[0, 2] = 1.0  # e1: e1*e1 = -1
        assert abs(geometric_product_forward(x, x)[0, 0].item() - (-1.0)) < 1e-10

class TestGeometricProductAutograd:
    def test_gradcheck(self):
        x = torch.randn(8, 16, device="cuda", dtype=torch.float64, requires_grad=True)
        y = torch.randn(8, 16, device="cuda", dtype=torch.float64, requires_grad=True)
        assert torch.autograd.gradcheck(triton_geometric_product, (x, y), eps=1e-6, atol=1e-4)

    def test_gradcheck_batched(self):
        x = torch.randn(2, 4, 16, device="cuda", dtype=torch.float64, requires_grad=True)
        y = torch.randn(2, 4, 16, device="cuda", dtype=torch.float64, requires_grad=True)
        assert torch.autograd.gradcheck(triton_geometric_product, (x, y), eps=1e-6, atol=1e-4)
