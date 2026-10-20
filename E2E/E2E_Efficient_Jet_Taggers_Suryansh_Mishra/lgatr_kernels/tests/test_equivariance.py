"""Tests for Lorentz equivariance of all kernels."""

import torch
import pytest
import math
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

from ..autograd.equi_linear import triton_equi_linear
from ..autograd.geometric_product import triton_geometric_product
from ..autograd.equi_layernorm import triton_equi_layer_norm


def _random_rotation_4d(device="cuda", dtype=torch.float64):
    axis = torch.randn(3, device=device, dtype=dtype)
    axis = axis / axis.norm()
    angle = torch.rand(1, device=device, dtype=dtype).item() * 2 * math.pi
    K = torch.tensor([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]], device=device, dtype=dtype)
    R3 = torch.eye(3, device=device, dtype=dtype) + math.sin(angle) * K + (1 - math.cos(angle)) * (K @ K)
    R = torch.eye(4, device=device, dtype=dtype)
    R[1:, 1:] = R3
    return R

def _random_boost(device="cuda", dtype=torch.float64, max_rapidity=1.0):
    d = torch.randn(3, device=device, dtype=dtype); d = d / d.norm()
    r = torch.rand(1, device=device, dtype=dtype).item() * max_rapidity
    g = math.cosh(r); bg = math.sinh(r)
    L = torch.eye(4, device=device, dtype=dtype)
    L[0, 0] = g
    for i in range(3):
        L[0, i+1] = -bg * d[i]; L[i+1, 0] = -bg * d[i]
        for j in range(3): L[i+1, j+1] += (g - 1) * d[i] * d[j]
    return L

def _random_lorentz_transform(device="cuda", dtype=torch.float64):
    return _random_boost(device, dtype) @ _random_rotation_4d(device, dtype)

def _lorentz_transform_multivector(mv, Lambda):
    device, dtype = mv.device, mv.dtype
    T = torch.zeros(16, 16, device=device, dtype=dtype)
    T[0, 0] = 1.0
    T[1:5, 1:5] = Lambda
    bv = [(0,1),(0,2),(0,3),(1,2),(1,3),(2,3)]
    for oi, (a,b) in enumerate(bv):
        for ii, (i,j) in enumerate(bv):
            T[5+oi, 5+ii] = Lambda[a,i]*Lambda[b,j] - Lambda[a,j]*Lambda[b,i]
    tv = [(0,1,2),(0,1,3),(0,2,3),(1,2,3)]
    for oi, (a,b,c) in enumerate(tv):
        for ii, (i,j,k) in enumerate(tv):
            T[11+oi, 11+ii] = (Lambda[a,i]*(Lambda[b,j]*Lambda[c,k]-Lambda[b,k]*Lambda[c,j])
                               -Lambda[a,j]*(Lambda[b,i]*Lambda[c,k]-Lambda[b,k]*Lambda[c,i])
                               +Lambda[a,k]*(Lambda[b,i]*Lambda[c,j]-Lambda[b,j]*Lambda[c,i]))
    T[15, 15] = torch.det(Lambda).sign()
    return torch.einsum("ij,...j->...i", T, mv)


class TestLorentzEquivariance:
    def test_equi_linear_equivariance(self):
        L = _random_lorentz_transform()
        x = torch.randn(8, 4, 16, device="cuda", dtype=torch.float64)
        w = torch.randn(6, 4, 10, device="cuda", dtype=torch.float64)
        Lx = _lorentz_transform_multivector(x, L)
        torch.testing.assert_close(triton_equi_linear(Lx, w), _lorentz_transform_multivector(triton_equi_linear(x, w), L), atol=1e-6, rtol=1e-6)

    def test_geometric_product_equivariance(self):
        L = _random_lorentz_transform()
        x = torch.randn(16, 16, device="cuda", dtype=torch.float64)
        y = torch.randn(16, 16, device="cuda", dtype=torch.float64)
        Lx = _lorentz_transform_multivector(x, L); Ly = _lorentz_transform_multivector(y, L)
        torch.testing.assert_close(triton_geometric_product(Lx, Ly), _lorentz_transform_multivector(triton_geometric_product(x, y), L), atol=1e-6, rtol=1e-6)

    def test_layernorm_equivariance(self):
        L = _random_lorentz_transform()
        x = torch.randn(8, 4, 16, device="cuda", dtype=torch.float64)
        Lx = _lorentz_transform_multivector(x, L)
        torch.testing.assert_close(triton_equi_layer_norm(Lx), _lorentz_transform_multivector(triton_equi_layer_norm(x), L), atol=1e-6, rtol=1e-6)

    def test_rotation_only_equivariance(self):
        L = _random_rotation_4d()
        x = torch.randn(16, 16, device="cuda", dtype=torch.float64)
        y = torch.randn(16, 16, device="cuda", dtype=torch.float64)
        torch.testing.assert_close(triton_geometric_product(_lorentz_transform_multivector(x, L), _lorentz_transform_multivector(y, L)),
                                   _lorentz_transform_multivector(triton_geometric_product(x, y), L), atol=1e-6, rtol=1e-6)

    def test_boost_equivariance(self):
        L = _random_boost()
        x = torch.randn(16, 16, device="cuda", dtype=torch.float64)
        y = torch.randn(16, 16, device="cuda", dtype=torch.float64)
        torch.testing.assert_close(triton_geometric_product(_lorentz_transform_multivector(x, L), _lorentz_transform_multivector(y, L)),
                                   _lorentz_transform_multivector(triton_geometric_product(x, y), L), atol=1e-6, rtol=1e-6)
