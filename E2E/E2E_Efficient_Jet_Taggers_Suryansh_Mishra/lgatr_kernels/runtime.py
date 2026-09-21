"""High-level runtime helpers for optimized LorentzGATr workflows."""

from __future__ import annotations

from typing import Any

import torch
from torch import nn

from .compile_patches import patch_lgatr_compile
from .layers import fuse_equi_linear_layers
from .primitives import patch_lgatr


def optimize_lgatr_model(
    model: nn.Module,
    *,
    use_compile_patches: bool = False,
    compile_mode: str | None = None,
) -> tuple[nn.Module, dict[str, Any]]:
    """Apply the recommended LGATr runtime optimizations to a model.

    Parameters
    ----------
    model:
        A LorentzGATr-style model instance.
    use_compile_patches:
        If True, enables the compile-friendly cache / einsum patches before
        optional ``torch.compile``.
    compile_mode:
        If not None, wraps the model in ``torch.compile(model, mode=...)``.

    Returns
    -------
    (model, stats):
        The optimized model and a small metadata dictionary.
    """

    patch_lgatr()
    if use_compile_patches:
        patch_lgatr_compile()

    equilinear_fused = fuse_equi_linear_layers(model)
    stats = {
        "equilinear_fused": equilinear_fused,
        "compile_patches": use_compile_patches,
        "compile_mode": compile_mode,
    }

    if compile_mode is not None:
        model = torch.compile(model, mode=compile_mode)

    return model, stats


class LorentzGATrGraphWrapper(nn.Module):
    """CUDA Graph wrapper for project-level ``LorentzGATr`` models.

    This wrapper captures the graphable portion of the forward pass after the
    dynamic preprocessing stage (`processor` + padding mask construction).
    """

    def __init__(
        self,
        model: nn.Module,
        batch_size: int,
        *,
        max_particles: int = 128,
        num_classes: int = 10,
    ) -> None:
        super().__init__()
        self.inner = model
        self.inner.eval()

        device = next(model.parameters()).device
        self.static_mv = torch.zeros(batch_size, max_particles, 16, device=device)
        self.static_pad = torch.zeros(batch_size, max_particles, device=device)
        self.static_out = torch.zeros(batch_size, num_classes, device=device)
        self.graph: torch.cuda.CUDAGraph | None = None

    def _graphed_forward(self, mv: torch.Tensor, pad_mask: torch.Tensor) -> torch.Tensor:
        m = self.inner
        batch = mv.size(0)
        x = m.encoder(mv)
        x = m.proj(x)
        x_cls = m.cls_token.expand(batch, -1, -1)
        for layer in m.decoder:
            x_cls = layer(x, x_cls, pad_mask)
        x_cls = m.layernorm(x_cls).squeeze(1)
        return m.classifier(x_cls)

    def capture(self, n_warmup: int = 8) -> None:
        with torch.no_grad():
            for _ in range(n_warmup):
                self._graphed_forward(self.static_mv, self.static_pad)
        torch.cuda.synchronize()

        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph):
            self.static_out = self._graphed_forward(self.static_mv, self.static_pad)
        torch.cuda.synchronize()

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.graph is None:
            self.capture()

        batch = x.size(0)
        pad = (x[..., 3] == 0).float()
        mv, _ = self.inner.processor(x)

        self.static_mv.zero_()
        self.static_pad.zero_()
        self.static_mv[:batch].copy_(mv)
        self.static_pad[:batch].copy_(pad[:batch])

        self.graph.replay()
        torch.cuda.synchronize()
        return self.static_out[:batch].clone()
