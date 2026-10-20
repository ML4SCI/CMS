"""Tests for the fused scalar-gated GELU kernel."""

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

from ..triton.gated_gelu_kernel import gated_gelu_forward
from ..autograd.gated_gelu import triton_gated_gelu


def _reference_gated_gelu(x):
    return torch.nn.functional.gelu(x[..., 0:1]) * x


class TestGatedGELUForward:
    def test_basic_correctness(self):
        x = torch.randn(128, 16, device="cuda", dtype=torch.float32)
        torch.testing.assert_close(gated_gelu_forward(x), _reference_gated_gelu(x), atol=1e-5, rtol=1e-5)

    def test_batched(self):
        x = torch.randn(4, 32, 8, 16, device="cuda", dtype=torch.float32)
        torch.testing.assert_close(gated_gelu_forward(x), _reference_gated_gelu(x), atol=1e-5, rtol=1e-5)

    def test_zero_input(self):
        x = torch.zeros(16, 16, device="cuda", dtype=torch.float64)
        assert (gated_gelu_forward(x) == 0).all()

class TestGatedGELUAutograd:
    def test_gradcheck(self):
        x = torch.randn(8, 16, device="cuda", dtype=torch.float64, requires_grad=True)
        assert torch.autograd.gradcheck(triton_gated_gelu, (x,), eps=1e-4, atol=5e-3, rtol=5e-3)

    def test_gradcheck_batched(self):
        x = torch.randn(2, 4, 16, device="cuda", dtype=torch.float64, requires_grad=True)
        assert torch.autograd.gradcheck(triton_gated_gelu, (x,), eps=1e-4, atol=5e-3, rtol=5e-3)
