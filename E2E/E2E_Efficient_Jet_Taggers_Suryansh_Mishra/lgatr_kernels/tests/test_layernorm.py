"""Tests for the fused equivariant LayerNorm kernel."""

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

from ..codegen.cayley_table import INNER_PRODUCT_SIGNS
from ..triton.equi_layernorm_kernel import equi_layernorm_forward
from ..autograd.equi_layernorm import triton_equi_layer_norm


def _reference_equi_layernorm(x, gain=1.0, epsilon=0.01):
    m = torch.tensor(INNER_PRODUCT_SIGNS, device=x.device, dtype=x.dtype)
    grade_ranges = [(0, 1), (1, 5), (5, 11), (11, 15), (15, 16)]
    total = torch.zeros(*x.shape[:-1], 1, device=x.device, dtype=x.dtype)
    for start, end in grade_ranges:
        sq = (x[..., start:end] * x[..., start:end] * m[start:end]).sum(dim=-1, keepdim=True)
        total = total + sq.abs()
    mean_norm = total.mean(dim=-2, keepdim=True).clamp(min=epsilon)
    return gain * x * torch.rsqrt(mean_norm)


class TestEquiLayerNormForward:
    def test_basic_correctness(self):
        x = torch.randn(32, 8, 16, device="cuda", dtype=torch.float64)
        torch.testing.assert_close(equi_layernorm_forward(x), _reference_equi_layernorm(x), atol=1e-5, rtol=1e-5)

    def test_batched(self):
        x = torch.randn(4, 16, 8, 16, device="cuda", dtype=torch.float64)
        torch.testing.assert_close(equi_layernorm_forward(x), _reference_equi_layernorm(x), atol=1e-5, rtol=1e-5)

    def test_gain(self):
        x = torch.randn(16, 4, 16, device="cuda", dtype=torch.float64)
        torch.testing.assert_close(equi_layernorm_forward(x, gain=2.0), 2.0 * equi_layernorm_forward(x, gain=1.0), atol=1e-10, rtol=1e-10)

    def test_scale_invariance(self):
        x = torch.randn(32, 8, 16, device="cuda", dtype=torch.float64) * 100
        torch.testing.assert_close(equi_layernorm_forward(x), equi_layernorm_forward(x * 0.01), atol=0.5, rtol=0.1)

class TestEquiLayerNormAutograd:
    def test_gradcheck(self):
        x = torch.randn(4, 4, 16, device="cuda", dtype=torch.float64, requires_grad=True)
        assert torch.autograd.gradcheck(lambda x: triton_equi_layer_norm(x, gain=1.0, epsilon=0.01), (x,), eps=1e-4, atol=5e-3, rtol=5e-3)
