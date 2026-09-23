"""nn.Module wrappers for optimized Particle Transformer components."""

from .cuda_graph_wrapper import CUDAGraphInferenceWrapper
from .fused_pair_mlp import FusedPairMLP
from .optimized_model import (
    OptimizedBlock,
    OptimizedPairEmbed,
    OptimizedParticleTransformer,
)
