"""Tests for the GEMM-based equivariant linear."""

import torch
import pytest

# This file was previously hidden: pytest aborted the whole run on the sibling collection
# errors, so it never executed. With `lgatr_kernels/__init__` made lazy it now imports fine and
# its 13 `device="cuda"` allocations surface as runtime failures instead. Same guard as the
# sibling kernel tests. (Added 2026-09-04.)
if not torch.cuda.is_available():
    pytest.skip("lgatr_kernels kernel tests require CUDA", allow_module_level=True)

from ..codegen.cayley_table import compute_linear_basis_sparse
from ..triton.equi_linear_gemm import gemm_equi_linear


def _build_basis_tensor(device="cuda", dtype=torch.float64):
    bases = compute_linear_basis_sparse()
    basis = torch.zeros(10, 16, 16, device=device, dtype=dtype)
    for b_idx, mapping in enumerate(bases):
        for (inp, out), weight in mapping.items():
            basis[b_idx, out, inp] = weight
    return basis

def _reference_equi_linear(x, weight, basis):
    return torch.einsum("yxa, aij, ...xj -> ...yi", weight.to(x.dtype), basis, x)

@pytest.fixture
def basis():
    return _build_basis_tensor()

class TestEquiLinearForward:
    def test_basic_correctness(self, basis):
        x = torch.randn(64, 8, 16, device="cuda", dtype=torch.float64)
        w = torch.randn(16, 8, 10, device="cuda", dtype=torch.float64)
        torch.testing.assert_close(gemm_equi_linear(x, w), _reference_equi_linear(x, w, basis), atol=1e-5, rtol=1e-5)

    def test_batched_3d(self, basis):
        x = torch.randn(2, 16, 4, 16, device="cuda", dtype=torch.float64)
        w = torch.randn(8, 4, 10, device="cuda", dtype=torch.float64)
        torch.testing.assert_close(gemm_equi_linear(x, w), _reference_equi_linear(x, w, basis), atol=1e-5, rtol=1e-5)

    def test_single_channel(self, basis):
        x = torch.randn(32, 1, 16, device="cuda", dtype=torch.float64)
        w = torch.randn(1, 1, 10, device="cuda", dtype=torch.float64)
        torch.testing.assert_close(gemm_equi_linear(x, w), _reference_equi_linear(x, w, basis), atol=1e-5, rtol=1e-5)

    def test_grade_projection(self, basis):
        x = torch.randn(16, 1, 16, device="cuda", dtype=torch.float64)
        w = torch.zeros(1, 1, 10, device="cuda", dtype=torch.float64)
        w[0, 0, 0] = 1.0
        out = gemm_equi_linear(x, w)
        expected = torch.zeros_like(out)
        expected[..., 0, 0] = x[..., 0, 0]
        torch.testing.assert_close(out, expected, atol=1e-5, rtol=1e-5)

class TestEquiLinearAutograd:
    def test_gradcheck(self):
        x = torch.randn(8, 4, 16, device="cuda", dtype=torch.float64, requires_grad=True)
        w = torch.randn(6, 4, 10, device="cuda", dtype=torch.float64, requires_grad=True)
        assert torch.autograd.gradcheck(gemm_equi_linear, (x, w), eps=1e-4, atol=1e-2, rtol=1e-2)

    def test_gradcheck_batched(self):
        x = torch.randn(2, 4, 2, 16, device="cuda", dtype=torch.float64, requires_grad=True)
        w = torch.randn(3, 2, 10, device="cuda", dtype=torch.float64, requires_grad=True)
        assert torch.autograd.gradcheck(gemm_equi_linear, (x, w), eps=1e-4, atol=5e-3, rtol=5e-3)
