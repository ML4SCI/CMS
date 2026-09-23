"""CUDA Graph wrapper for L-GATr inference with zero-overhead replay."""

import torch
import torch.nn as nn


class CUDAGraphWrapper(nn.Module):
    def __init__(self, model, batch_size, n_items, in_mv_channels, in_s_channels, warmup_iters=5):
        super().__init__()
        self.model = model
        self._mv_static = torch.randn(batch_size, n_items, in_mv_channels, 16, device='cuda')
        self._s_static = torch.randn(batch_size, n_items, in_s_channels, device='cuda')
        self._out_mv = None
        self._out_s = None

        model.eval()
        with torch.no_grad():
            for _ in range(warmup_iters):
                model(self._mv_static, self._s_static)
                torch.cuda.synchronize()

        self.graph = torch.cuda.CUDAGraph()
        with torch.no_grad(), torch.cuda.graph(self.graph):
            self._out_mv, self._out_s = model(self._mv_static, self._s_static)

    @torch.no_grad()
    def forward(self, multivectors, scalars):
        self._mv_static.copy_(multivectors)
        self._s_static.copy_(scalars)
        self.graph.replay()
        return self._out_mv.clone(), self._out_s.clone()
