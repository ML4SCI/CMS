"""Fused L-GATr layer replacements."""
from .fused_linear import FusedEquiLinear, fuse_equi_linear_layers
from .cuda_graph_wrapper import CUDAGraphWrapper
