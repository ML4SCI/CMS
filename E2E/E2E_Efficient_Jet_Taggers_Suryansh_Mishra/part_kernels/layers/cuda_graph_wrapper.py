"""
CUDA Graph inference wrapper for OptimizedParticleTransformer.

Captures the full forward pass into a single CUDA graph and replays
it for each batch, eliminating per-batch kernel launch overhead.

Requirements for graph capture:
  - fixed batch size N and particle count P
  - trimming disabled (for_inference=True or trim=False)
  - model in eval() mode
  - all buffers preallocated with static shapes
"""
import torch
import torch.nn as nn


class CUDAGraphInferenceWrapper(nn.Module):
    """Wraps an OptimizedParticleTransformer for CUDA-graph-accelerated
    inference with fixed N and P."""

    def __init__(self, model: nn.Module, batch_size: int, max_particles: int,
                 input_dim: int, num_classes: int, device: str = 'cuda'):
        super().__init__()
        self.model = model
        self.N = batch_size
        self.P = max_particles
        self.device = device
        self._graph = None
        self._captured = False

        self.static_x    = torch.zeros(batch_size, input_dim, max_particles, device=device)
        self.static_v    = torch.zeros(batch_size, 4, max_particles, device=device)
        self.static_mask = torch.ones(batch_size, 1, max_particles, device=device, dtype=torch.bool)
        self.static_out  = torch.zeros(batch_size, num_classes, device=device)

    def _warmup(self, n_warmup: int = 3):
        """Run a few eager forward passes to stabilize CUDA state."""
        for _ in range(n_warmup):
            _ = self.model(self.static_x, self.static_v, self.static_mask)
        torch.cuda.synchronize()

    def capture(self, n_warmup: int = 3):
        """Capture the forward pass as a CUDA graph."""
        self.model.eval()
        self._warmup(n_warmup)

        self._graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self._graph):
            self.static_out = self.model(self.static_x, self.static_v, self.static_mask)
        self._captured = True

    def forward(self, x: torch.Tensor, v: torch.Tensor,
                mask: torch.Tensor) -> torch.Tensor:
        if not self._captured:
            self.capture()

        N_actual = x.size(0)
        P_actual = x.size(2)
        assert N_actual <= self.N, f"Batch {N_actual} exceeds captured {self.N}"
        assert P_actual <= self.P, f"P {P_actual} exceeds captured {self.P}"

        self.static_x.zero_()
        self.static_v.zero_()
        self.static_mask.zero_()

        self.static_x[:N_actual, :, :P_actual].copy_(x)
        self.static_v[:N_actual, :, :P_actual].copy_(v)
        self.static_mask[:N_actual, :, :P_actual].copy_(mask)

        self._graph.replay()
        torch.cuda.synchronize()

        return self.static_out[:N_actual].clone()
