from typing import List, Tuple, Dict, Optional

import torch
from torch import nn, Tensor
from lgatr.interface import extract_vector
from lgatr.layers import EquiLinear

from .classifier import ClassAttentionBlock, Classifier
from .particle_transformer import ParticleAttentionBlock, gate_kwargs_from_config
from .processor import DenormalizeMixin, InteractionEmbedding, ParticleProcessor
from .utils import load_encoder_weights
from ..configs import LorentzParTConfig


class LorentzParTEncoder(nn.Module):
    def __init__(
        self,
        embed_dim: int = 128,
        num_heads: int = 8,
        num_layers: int = 8,
        in_s_channels: Optional[int] = None,
        out_s_channels: Optional[int] = None,
        dropout: float = 0.1,
        expansion_factor: int = 4,
        pair_embed_dims: List[int] = [64, 64, 64],
        attention_config: Optional[Dict] = None
    ):
        super(LorentzParTEncoder, self).__init__()
        self.equilinear = EquiLinear(
            in_mv_channels=1,
            out_mv_channels=1,
            in_s_channels=in_s_channels,
            out_s_channels=out_s_channels
        )
        self.proj = nn.Linear(16, embed_dim)
        self.interaction_embed = InteractionEmbedding(
            num_interaction_features=4,
            pair_embed_dims=pair_embed_dims + [num_heads]
        )

        # attention_config decides whether the blocks use the physics-aware gate
        gate_kwargs = gate_kwargs_from_config(attention_config)
        self.encoder = nn.ModuleList([
            ParticleAttentionBlock(
                embed_dim=embed_dim,
                num_heads=num_heads,
                dropout=dropout,
                expansion_factor=expansion_factor,
                **gate_kwargs
            ) for _ in range(num_layers)
        ])

    def forward(self, x: Tensor, padding_mask: Tensor, U: Tensor, p4: Optional[Tensor] = None) -> Tensor:
        B, N, F = x.shape  # (batch_size, max_num_particles, 16)

        # Embed interaction features
        U = self.interaction_embed(U)  # (B * num_heads, N, N)

        # Pass through the EquiLinear layer
        x = x.view(B, N, 1, F)
        x, _ = self.equilinear(x)  # (B, N, 1, 16)
        x = x.view(B, N, 16)

        # Project input features to embedding dimension
        x = self.proj(x)  # (B, N, embed_dim)

        # Encoder with particle attention blocks (p4 feeds the mass gate)
        for layer in self.encoder:
            x = layer(x, padding_mask, U, p4=p4)  # (B, N, embed_dim)

        return x  # (B, N, embed_dim)


class LorentzParT(DenormalizeMixin, nn.Module):
    """
    Particle Transformer model with EquiLinear layer for jet classification and self-supervised learning.

    Parameters
    ----------
    config: LorentzParTConfig, optional
        Configuration object for the LorentzParT model.
    max_num_particles: int, optional
        Maximum number of particles per jet.
    num_particle_features: int, optional
        Number of features for each particle: pT, eta, phi, and energy.
    num_classes: int, optional
        Number of output classes for classification: 10 equally distributed classes in JetClass.
    embed_dim: int, optional
        Dimensionality of the embedding space.
    num_heads: int, optional
        Number of attention heads in the transformer.
    num_layers: int, optional
        Number of layers in the transformer.
    num_cls_layers: int, optional
        Number of layers in the classification head.
    num_mlp_layers: int, optional
        Number of layers in the MLP head.
    hidden_dim: int, optional
        Dimensionality of the hidden layers.
    hidden_mv_channels - reinsert_s_channels: optional
        EquiLinear layer's configurations.
    attention: Dict, optional
        Physics-aware gate settings, e.g. `{'use_gating': True}` (see `gate_kwargs_from_config`).
    dropout: float, optional
        Dropout rate for the model.
    expansion_factor: int, optional
        Expansion factor for the feedforward layers in LorentzParTEncoder.
    pair_embed_dims: List[int], optional
        Dimensionality of the pair embeddings for the interaction features.
    mask: bool, optional
        Indicates whether the model is for self-supervised learning or classification.
    weights: str, optional
        Path to the pretrained weights.
    inference: bool, optional
        Whether to use the model for inference.
    norm_stats: Dict[str, Tuple[float, float]], optional
        The (mean, std) per feature used by JetClassDataset. Lets the mass gate see GeV.
    normalize: List[bool], optional
        The normalize flags given to JetClassDataset. Default is `[True, False, False, True]`.
    debug: bool, optional
        Print the normalisation and the jet mass reaching the gate for the first `debug_batches` passes.
    debug_batches: int, optional
        Number of forward passes to report when `debug` is True.

    .. References::
        Huilin Qu, Congqiao Li, and Sitian Qian.
        [Particle Transformer for Jet Tagging](https://arxiv.org/abs/2202.03772).
        In *Proceedings of the 39th International Conference on Machine Learning*, pages 18281-18292, 2022.

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
        config: Optional[LorentzParTConfig] = None,
        # Parameters below can override config if supplied explicitly
        max_num_particles: Optional[int] = None,
        num_particle_features: Optional[int] = None,
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
        pair_embed_dims: Optional[List[int]] = None,
        mask: Optional[bool] = None,
        weights: Optional[str] = None,
        inference: Optional[bool] = False,
        # Mass-gate normalisation
        norm_stats: Optional[Dict[str, Tuple[float, float]]] = None,
        normalize: Optional[List[bool]] = None,
        debug: bool = False,
        debug_batches: int = 2
    ):
        super(LorentzParT, self).__init__()

        # Use config if provided, otherwise use defaults
        if config is not None:
            self.max_num_particles = max_num_particles if max_num_particles is not None else config.max_num_particles
            self.num_particle_features = num_particle_features if num_particle_features is not None else config.num_particle_features
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
            self.pair_embed_dims = pair_embed_dims if pair_embed_dims is not None else config.pair_embed_dims
            self.mask = mask if mask is not None else config.mask
            self.weights = weights if weights is not None else config.weights
            self.inference = inference if inference is not None else config.inference
        else:
            self.max_num_particles = max_num_particles if max_num_particles is not None else 128
            self.num_particle_features = num_particle_features if num_particle_features is not None else 4
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
            self.mlp = mlp if mlp is not None else None
            self.reinsert_mv_channels = reinsert_mv_channels if reinsert_mv_channels is not None else None
            self.reinsert_s_channels = reinsert_s_channels if reinsert_s_channels is not None else None
            self.dropout = dropout if dropout is not None else 0.1
            self.expansion_factor = expansion_factor if expansion_factor is not None else 4
            self.pair_embed_dims = pair_embed_dims if pair_embed_dims is not None else [64, 64, 64]
            self.mask = mask if mask is not None else False
            self.weights = weights if weights is not None else None
            self.inference = inference if inference is not None else False

        # Initialize the class token
        self.cls_token = nn.Parameter(torch.zeros(1, 1, self.embed_dim), requires_grad=True)
        nn.init.normal_(self.cls_token, mean=0.0, std=1.0)

        self.processor = ParticleProcessor(to_multivector=True, verbose=debug)
        self.encoder = LorentzParTEncoder(
            embed_dim=self.embed_dim,
            num_heads=self.num_heads,
            num_layers=self.num_layers,
            in_s_channels=self.in_s_channels,
            out_s_channels=self.out_s_channels,
            dropout=self.dropout,
            expansion_factor=self.expansion_factor,
            pair_embed_dims=self.pair_embed_dims,
            attention_config=self.attention
        )

        # For self-supervised learning
        self.fc = nn.Linear(self.max_num_particles * self.embed_dim, 16)
        self.equilinear = EquiLinear(
            in_mv_channels=1,
            out_mv_channels=1,
            in_s_channels=self.in_s_channels,
            out_s_channels=self.out_s_channels
        )

        # For classification
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

        # Inverse of the dataset normalisation, so the mass gate sees GeV
        self._setup_denormalize(
            norm_stats, normalize, debug, debug_batches,
            gated=gate_kwargs_from_config(self.attention)['gate_type'] is not None
        )

        # Load pretrained weights if provided
        if self.weights is not None:
            load_encoder_weights(self.encoder, self.weights)

    def forward(self, x: Tensor, mask_idx: Optional[Tensor] = None) -> Tensor:
        B, N, F = x.shape  # (batch_size, max_num_particles, num_particle_features)

        # Recover un-normalized (pT, eta, phi, E) in GeV for the invariant-mass gate.
        # The network itself still consumes the normalized tensor.
        x_raw = self.denormalize(x) if self.norm_stats is not None else None

        # Ignore padding particles in query
        padding_mask = (x[..., 3] == 0).float()  # (B, N)

        # Set the masked indices to 0.0 so they are not ignored in MultiheadAttention()
        if mask_idx is not None:
            batch_indices = torch.arange(x.size(0), device=x.device)
            padding_mask[batch_indices, mask_idx] = 0.0

        # Process particles. The processor also returns the PHYSICAL 4-momentum
        # p4 = (E, px, py, pz), converted from collider coordinates and with padded
        # rows zeroed. Do NOT substitute x.clone() here: that tensor is
        # (pT, eta, phi, E), so the mass gate would compute a negative m^2 that
        # gets clamped to zero.
        x, U, p4 = self.processor(x, x_raw)
        self._log_mass_gate(x_raw, p4)

        # Pass through equilinear layer and particle attention blocks
        x = self.encoder(x, padding_mask, U, p4=p4)

        # Classification (no masking in this case)
        if not self.mask:
            x_cls = self.cls_token.expand(B, -1, -1)

            # Decoder with class attention blocks
            for layer in self.decoder:
                x_cls = layer(x, x_cls, padding_mask)

            # MLP head for classification
            x_cls = self.layernorm(x_cls).squeeze(1)
            x_cls = self.classifier(x_cls)
            output = self.act(x_cls)  # (B, num_classes)

            return output
        else:
            x = x.view(B, -1)  # (B, N * embed_dim)
            x = self.fc(x)  # (B, 16)
            x = x.view(B, 1, 1, 16)
            x, _ = self.equilinear(x)  # (B, 1, 1, 16)
            x = x.view(B, 16)
            x = extract_vector(x)  # (B, F)

            return x
