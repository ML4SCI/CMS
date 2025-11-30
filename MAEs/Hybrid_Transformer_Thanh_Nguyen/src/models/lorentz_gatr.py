from typing import Tuple, Dict, Optional

import torch
from torch import nn, Tensor
from lgatr import LGATr
from lgatr.interface import extract_vector

from .classifier import ClassAttentionBlock, Classifier
from .processor import ParticleProcessor
from ..configs import LGATrConfig
    

class LGATrEncoder(nn.Module):
    """
    .. References::
        Johann Brehmer, Víctor Bresó, Pim de Haan, Tilman Plehn, Huilin Qu, Jonas Spinner, and Jesse Thaler.
        [A Lorentz-Equivariant Transformer for All of the LHC](https://arxiv.org/abs/2411.00446).
        *arXiv preprint arXiv:2411.00446*, 2024.

        Jonas Spinner, Victor Bresó, Pim De Haan, Tilman Plehn, Jesse Thaler, and Johann Brehmer.
        [Lorentz-Equivariant Geometric Algebra Transformers for High-Energy Physics](https://arxiv.org/abs/2405.14806).
        *Advances in Neural Information Processing Systems*, 37:22178-22205, 2024.

        Johann Brehmer, Pim De Haan, Sönke Behrends, and Taco Cohen.
        [Geometric Algebra Transformer](https://arxiv.org/abs/2305.18415).
        *Advances in Neural Information Processing Systems*, 36:35472-35496, 2023.
    """
    def __init__(
        self,
        num_layers: int = 8,
        hidden_mv_channels: int = 8,
        in_s_channels: Optional[int] = None,
        out_s_channels: Optional[int] = None,
        hidden_s_channels: Optional[int] = 16,
        attention: Dict = {},
        mlp: Dict = {},
        reinsert_mv_channels: Optional[Tuple[int]] = None,
        reinsert_s_channels: Optional[Tuple[int]] = None,
        dropout: Optional[float] = None
    ):
        super(LGATrEncoder, self).__init__()
        self.encoder = LGATr(
            num_blocks=num_layers,
            in_mv_channels=1,
            out_mv_channels=1,
            hidden_mv_channels=hidden_mv_channels,
            in_s_channels=in_s_channels,
            out_s_channels=out_s_channels,
            hidden_s_channels=hidden_s_channels,
            attention=attention,
            mlp=mlp,
            reinsert_mv_channels=reinsert_mv_channels,
            reinsert_s_channels=reinsert_s_channels,
            dropout_prob=dropout
        )

    def forward(self, x: Tensor) -> Tensor:
        B, N, F = x.shape  # (batch_size, max_num_particles, num_particle_features)
        x = x.view(B, N, 1, F)  # for compatibility with LGATr
        x, _ = self.encoder(x)  # (B, N, out_mv_channels, 16)
        x = x.view(B, N, -1)  # (B, N, 16)

        return x
    

class LorentzGATr(nn.Module):
    """
    .. References::
        Johann Brehmer, Víctor Bresó, Pim de Haan, Tilman Plehn, Huilin Qu, Jonas Spinner, and Jesse Thaler.
        [A Lorentz-Equivariant Transformer for All of the LHC](https://arxiv.org/abs/2411.00446).
        *arXiv preprint arXiv:2411.00446*, 2024.

        Jonas Spinner, Victor Bresó, Pim De Haan, Tilman Plehn, Jesse Thaler, and Johann Brehmer.
        [Lorentz-Equivariant Geometric Algebra Transformers for High-Energy Physics](https://arxiv.org/abs/2405.14806).
        *Advances in Neural Information Processing Systems*, 37:22178-22205, 2024.

        Johann Brehmer, Pim De Haan, Sönke Behrends, and Taco Cohen.
        [Geometric Algebra Transformer](https://arxiv.org/abs/2305.18415).
        *Advances in Neural Information Processing Systems*, 36:35472-35496, 2023.
    """
    def __init__(
        self,
        config: Optional[LGATrConfig] = None,
        # Parameters below can override config if supplied explicitly
        num_classes: Optional[int] = None,
        embed_dim: Optional[int] = None,
        num_heads: Optional[int] = None,
        num_layers: Optional[int] = None,
        num_cls_layers: Optional[int] = None,
        num_mlp_layers: Optional[int] = None,
        hidden_dim: Optional[int] = None,
        hidden_mv_channels: Optional[int] = None,
        in_s_channels: Optional[int] = None,
        out_s_channels: Optional[int] = None,
        hidden_s_channels: Optional[int] = None,
        attention: Optional[Dict] = None,
        mlp: Optional[Dict] = None,
        reinsert_mv_channels: Optional[Tuple[int]] = None,
        reinsert_s_channels: Optional[Tuple[int]] = None,
        dropout: Optional[float] = None,
        expansion_factor: Optional[int] = None,
        max_num_particles: Optional[int] = None,
        num_particle_features: Optional[int] = None,  # (pT, eta, phi, energy)
        mask: Optional[bool] = None,
        weights: Optional[str] = None,
        inference: Optional[bool] = False,
    ):
        super(LorentzGATr, self).__init__()

        # Use config if provided, otherwise use defaults
        if config is not None:
            self.num_classes = num_classes if num_classes is not None else config.num_classes
            self.embed_dim = embed_dim if embed_dim is not None else config.embed_dim
            self.num_heads = num_heads if num_heads is not None else config.num_heads
            self.num_layers = num_layers if num_layers is not None else config.num_layers
            self.num_cls_layers = num_cls_layers if num_cls_layers is not None else config.num_cls_layers
            self.num_mlp_layers = num_mlp_layers if num_mlp_layers is not None else config.num_mlp_layers
            self.hidden_dim = hidden_dim if hidden_dim is not None else config.hidden_dim
            self.hidden_mv_channels = hidden_mv_channels if hidden_mv_channels is not None else config.hidden_mv_channels
            self.in_s_channels = in_s_channels if in_s_channels is not None else config.in_s_channels
            self.out_s_channels = out_s_channels if out_s_channels is not None else config.out_s_channels
            self.hidden_s_channels = hidden_s_channels if hidden_s_channels is not None else config.hidden_s_channels
            self.attention = attention if attention is not None else config.attention
            self.mlp = mlp if mlp is not None else config.mlp
            self.reinsert_mv_channels = reinsert_mv_channels if reinsert_mv_channels is not None else config.reinsert_mv_channels
            self.reinsert_s_channels = reinsert_s_channels if reinsert_s_channels is not None else config.reinsert_s_channels
            self.dropout = dropout if dropout is not None else config.dropout
            self.expansion_factor = expansion_factor if expansion_factor is not None else config.expansion_factor
            self.max_num_particles = max_num_particles if max_num_particles is not None else config.max_num_particles
            self.num_particle_features = num_particle_features if num_particle_features is not None else config.num_particle_features
            self.mask = mask if mask is not None else config.mask
            self.weights = weights if weights is not None else config.weights
            self.inference = inference if inference is not None else config.inference
        else:
            self.num_classes = num_classes if num_classes is not None else 10
            self.embed_dim = embed_dim if embed_dim is not None else 128
            self.num_heads = num_heads if num_heads is not None else 8
            self.num_layers = num_layers if num_layers is not None else 8
            self.num_cls_layers = num_cls_layers if num_cls_layers is not None else 2
            self.num_mlp_layers = num_mlp_layers if num_mlp_layers is not None else 0
            self.hidden_dim = hidden_dim if hidden_dim is not None else 256
            self.hidden_mv_channels = hidden_mv_channels if hidden_mv_channels is not None else 8
            self.in_s_channels = in_s_channels if in_s_channels is not None else None
            self.out_s_channels = out_s_channels if out_s_channels is not None else None
            self.hidden_s_channels = hidden_s_channels if hidden_s_channels is not None else 16
            self.attention = attention if attention is not None else {}
            self.mlp = mlp if mlp is not None else {}
            self.reinsert_mv_channels = reinsert_mv_channels if reinsert_mv_channels is not None else None
            self.reinsert_s_channels = reinsert_s_channels if reinsert_s_channels is not None else None
            self.dropout = dropout if dropout is not None else None
            self.expansion_factor = expansion_factor if expansion_factor is not None else 4
            self.max_num_particles = max_num_particles if max_num_particles is not None else 128
            self.num_particle_features = num_particle_features if num_particle_features is not None else 4
            self.mask = mask if mask is not None else False
            self.weights = weights if weights is not None else None
            self.inference = inference if inference is not None else False

        # Initialize the class token
        self.cls_token = nn.Parameter(torch.zeros(1, 1, self.embed_dim), requires_grad=True)
        nn.init.normal_(self.cls_token, mean=0.0, std=1.0)

        self.processor = ParticleProcessor(to_multivector=True)
        self.encoder = LGATrEncoder(
            num_layers=self.num_layers,
            hidden_mv_channels=self.hidden_mv_channels,
            in_s_channels=self.in_s_channels,
            out_s_channels=self.out_s_channels,
            hidden_s_channels=self.hidden_s_channels,
            attention=self.attention,
            mlp=self.mlp,
            reinsert_mv_channels=self.reinsert_mv_channels,
            reinsert_s_channels=self.reinsert_s_channels,
            dropout=self.dropout
        )

        # For self-supervised learning
        self.fc = nn.Linear(self.max_num_particles * 16, self.num_particle_features)

        # For classification
        self.proj = nn.Linear(16, self.embed_dim)
        self.decoder = nn.ModuleList([
            ClassAttentionBlock(
                embed_dim=self.embed_dim,
                num_heads=self.num_heads,
                dropout=0.0,  # paper didn't use dropout in the class attention blocks
                expansion_factor=self.expansion_factor
            ) for _ in range(self.num_cls_layers)
        ])
        self.layernorm = nn.LayerNorm(self.embed_dim)
        self.classifier = Classifier(
            num_classes=self.num_classes,
            input_dim=self.embed_dim,
            hidden_dim=self.hidden_dim,
            num_layers=self.num_mlp_layers,
            dropout=self.dropout,
        )
        self.act = nn.Softmax(dim=1) if self.inference else nn.Identity()

        # Load pretrained weights if provided
        if self.weights is not None:
            state_dict = torch.load(self.weights)
            filtered_state = {
                k[len("encoder.") :]: v
                for k, v in state_dict.items()
                if k.startswith("encoder.")
            }
            self.encoder.load_state_dict(filtered_state, strict=False)

    def forward(self, x: Tensor, mask_idx: Optional[Tensor] = None) -> Tensor:
        B, N, F = x.shape  # (batch_size, max_num_particles, num_particle_features)

        # Ignore padding particles in query
        padding_mask = (x[..., 3] == 0).float()  # (B, N)

        # Set the masked indices to 0.0 so they are not ignored in MultiheadAttention()
        if mask_idx is not None:
            batch_indices = torch.arange(B, device=x.device)
            padding_mask[batch_indices, mask_idx.view(-1)] = 0.0

        # Encoder
        x, _ = self.processor(x)
        x = self.encoder(x)  # (B, N, 16)

        # Classification (no masking in this case)
        if not self.mask:
            x_cls = self.cls_token.expand(B, -1, -1)
            x = self.proj(x)  # (B, N, embed_dim)

            # Decoder with class attention blocks
            for layer in self.decoder:
                x_cls = layer(x, x_cls, padding_mask)

            # MLP head for classification
            x_cls = self.layernorm(x_cls).squeeze(1)
            x_cls = self.classifier(x_cls)
            output = self.act(x_cls)  # (B, num_classes)

            return output
        else:
            x = x.view(B, -1)  # (B, N * 16)
            x = self.fc(x)  # (B, F)

            return x