"""Custom autograd Function for the GEMM-based equivariant linear map."""

from ..triton.equi_linear_gemm import gemm_equi_linear

triton_equi_linear = gemm_equi_linear
