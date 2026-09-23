from typing import List, Tuple, Dict, Optional

import torch
from torch import nn, Tensor

from .classifier import ClassAttentionBlock, Classifier
from .feedforward import Feedforward
from .processor import DenormalizeMixin, InteractionEmbedding, ParticleProcessor
from .utils import load_encoder_weights
from ..configs import ParticleTransformerConfig


GATE_TYPES = (None, 'headwise', 'elementwise')
GATE_CONFIG_KEYS = ('use_gating', 'gate_type', 'use_mass_bias', 'num_mass_freqs', 'mass_freq_range', 'identity_init_gate')


def gate_kwargs_from_config(attention: Optional[Dict]) -> Dict:
    """
    Translate the `attention` section of a model config into ParticleAttentionBlock arguments.

    attention:
        use_gating: True                # switch the physics-aware gate on
        gate_type: 'headwise'           # 'headwise' (default) or 'elementwise'
        use_mass_bias: True             # condition the gate on the jet invariant mass
        num_mass_freqs: 16              # Fourier features of the jet mass
        mass_freq_range: [0.02, 0.5]    # frequencies in 1/GeV
        identity_init_gate: True        # gate starts as an exact identity
    """
    attention = attention or {}
    unknown = set(attention) - set(GATE_CONFIG_KEYS)
    if unknown:
        raise ValueError(f"Unknown attention option(s) {sorted(unknown)}; expected a subset of {list(GATE_CONFIG_KEYS)}")

    if not attention.get('use_gating', False):
        return {'gate_type': None}

    kwargs = {'gate_type': attention.get('gate_type', 'headwise')}
    for key in GATE_CONFIG_KEYS[2:]:
        if key in attention:
            kwargs[key] = attention[key]
    if 'mass_freq_range' in kwargs:
        kwargs['mass_freq_range'] = tuple(kwargs['mass_freq_range'])

    return kwargs


class ParticleAttentionBlock(nn.Module):
    """
    Particle attention block with an optional physics-aware gate on the attention output.

    With `gate_type=None` this is the plain ParT block. Otherwise the attention output
    is multiplied by sigmoid(gate), where the gate is conditioned on the token, the
    pooled pairwise interactions U and (optionally) the jet invariant mass computed
    from the physical 4-momenta p4.
    """
    def __init__(
        self,
        embed_dim: int = 128,
        num_heads: int = 8,
        dropout: float = 0.1,
        expansion_factor: int = 4,
        gate_type: Optional[str] = None,
        num_mass_freqs: int = 16,
        mass_freq_range: Tuple[float, float] = (0.02, 0.5),
        use_mass_bias: bool = True,
        identity_init_gate: bool = True,
    ):
        super(ParticleAttentionBlock, self).__init__()
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        if gate_type not in GATE_TYPES:
            raise ValueError(f"gate_type must be one of {GATE_TYPES}, got {gate_type!r}")

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.gate_type = gate_type
        self.use_mass_bias = use_mass_bias
        self.identity_init_gate = identity_init_gate

        self.layernorm1 = nn.LayerNorm(embed_dim)
        self.pmha = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        self.layernorm2 = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)
        self.feedforward = Feedforward(
            embed_dim=embed_dim,
            expansion_factor=expansion_factor,
            dropout=dropout
        )

        # Physics-aware gate (only built when enabled, so ungated blocks keep the plain ParT parameters)
        if self.gate_type is not None:
            # Project pooled interaction head-features to match token embedding dimensions
            self.physics_proj = nn.Linear(num_heads, embed_dim)

            # Global invariant mass -> embedding bias.
            # Fourier features on m (GeV), NOT log1p(m^2). Periods span ~1/0.02 = 314 GeV
            # down to ~1/0.5 = 12.6 GeV, so the 10.8 GeV W/Z separation is resolvable.
            if self.use_mass_bias:
                self.register_buffer(
                    "m_freqs", torch.linspace(mass_freq_range[0], mass_freq_range[1], num_mass_freqs)
                )
                mass_feat_dim = 2 * num_mass_freqs + 1  # sin + cos + linear term
                self.mass_proj = nn.Linear(mass_feat_dim, embed_dim)

            gate_dim = num_heads if self.gate_type == 'headwise' else embed_dim
            self.gate_proj = nn.Linear(embed_dim, gate_dim)

            # Zero-init so sigmoid(0) = 0.5 uniformly; combined with the *2.0 rescale
            # in forward() the block starts as an exact identity to the ungated model.
            if identity_init_gate:
                nn.init.zeros_(self.gate_proj.weight)
                nn.init.zeros_(self.gate_proj.bias)

    def compute_mass(self, p4: Tensor) -> Tensor:
        """
        p4 : (B, N, 4) PHYSICAL 4-momentum (E, px, py, pz) in GeV, padded rows zeroed.
        returns (B, 1) invariant mass in GeV.
        """
        # Mask padded slots explicitly. If normalization ever leaks into padded
        # rows they are non-zero, and the spurious contribution scales with the
        # number of pads -- i.e. with multiplicity, which is class-correlated.
        pad = (p4.abs().sum(dim=-1, keepdim=True) > 0).float()  # (B, N, 1)
        p4m = p4 * pad

        energy_sum = p4m[..., 0].sum(dim=1, keepdim=True)  # (B, 1)
        momentum_sum = p4m[..., 1:].sum(dim=1)  # (B, 3)

        m2 = energy_sum ** 2 - momentum_sum.norm(dim=-1, keepdim=True) ** 2
        return m2.clamp(min=1e-6).sqrt()  # (B, 1) GeV

    def encode_mass(self, m: Tensor) -> Tensor:
        """(B, 1) mass in GeV -> (B, 1, embed_dim) bias vector."""
        ang = m * self.m_freqs  # (B, F)
        m_feat = torch.cat([ang.sin(), ang.cos(), m / 100.0], dim=-1)  # (B, 2F+1)
        return self.mass_proj(m_feat).unsqueeze(1)  # (B, 1, D)

    def forward(
        self,
        x: Tensor,
        padding_mask: Tensor,
        U: Optional[Tensor] = None,
        p4: Optional[Tensor] = None,
        mass_override: Optional[Tensor] = None,
        return_gate: bool = False,
    ) -> Tensor:
        """
        x             : (B, N, D) token embeddings
        padding_mask  : (B, N) key padding mask for MHA
        U             : (B*H, N, N) pairwise interaction bias
        p4            : (B, N, 4) PHYSICAL (E, px, py, pz) in GeV
        mass_override : (B, 1) optional mass, for the shuffled-invariant ablation
        return_gate   : also return (gate values, jet mass seen by the gate)
        """
        residual = x
        B, N, _ = x.shape

        # Pre-norm attention with pairwise interaction bias
        x_norm = self.layernorm1(x)
        x_attn, _ = self.pmha(x_norm, x_norm, x_norm, key_padding_mask=padding_mask, attn_mask=U)

        gate_val = None
        gate_mass = None
        if self.gate_type is not None:
            # Pairwise-interaction conditioning
            x_gating_input = x_norm
            if U is not None:
                u_pooled = U.view(B, self.num_heads, N, N).sum(dim=3).transpose(1, 2)  # (B, N, H)
                x_gating_input = x_gating_input + self.physics_proj(u_pooled)  # (B, N, D)

            # Global invariant-mass conditioning
            if self.use_mass_bias and (p4 is not None or mass_override is not None):
                gate_mass = mass_override if mass_override is not None else self.compute_mass(p4)
                x_gating_input = x_gating_input + self.encode_mass(gate_mass)

            # Physics-conditioned gate
            scale = 2.0 if self.identity_init_gate else 1.0
            if self.gate_type == 'headwise':
                gate_val = torch.sigmoid(self.gate_proj(x_gating_input)).unsqueeze(-1)  # (B, N, H, 1)
                x_attn = x_attn.reshape(B, N, self.num_heads, self.head_dim)
                x_attn = (x_attn * scale * gate_val).reshape(B, N, self.embed_dim)
            else:
                gate_val = torch.sigmoid(self.gate_proj(x_gating_input))  # (B, N, D)
                x_attn = x_attn * scale * gate_val

        # Residual + feedforward
        x = self.layernorm2(x_attn)
        x = self.dropout(x)
        x = x + residual
        x = self.feedforward(x)

        if return_gate:
            return x, gate_val, gate_mass

        return x


class ParticleTransformerEncoder(nn.Module):
    def __init__(
        self,
        embed_dim: int = 128,
        num_heads: int = 8,
        num_layers: int = 8,
        dropout: float = 0.1,
        expansion_factor: int = 4,
        pair_embed_dims: List[int] = [64, 64, 64],
        attention_config: Optional[Dict] = None
    ):
        super(ParticleTransformerEncoder, self).__init__()
        self.proj = nn.Linear(4, embed_dim)
        self.interaction_embed = InteractionEmbedding(
            num_interaction_features=4,
            pair_embed_dims=pair_embed_dims + [num_heads]
        )
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
        B, N, F = x.shape  # (batch_size, max_num_particles, num_particle_features)

        # Embed interaction features
        U = self.interaction_embed(U)  # (B * num_heads, N, N)

        # Project input features to embedding dimension
        x = self.proj(x)  # (B, N, embed_dim)

        # Encoder with particle attention blocks (p4 feeds the mass gate)
        for layer in self.encoder:
            x = layer(x, padding_mask, U, p4=p4)  # (B, N, embed_dim)

        return x  # (B, N, embed_dim)


class ParticleTransformer(DenormalizeMixin, nn.Module):
    """
    Particle Transformer model for jet classification and self-supervised learning.

    Parameters
    ----------
    config: ParticleTransformerConfig, optional
        Configuration object for the Particle Transformer model.
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
    dropout: float, optional
        Dropout rate for the model.
    expansion_factor: int, optional
        Expansion factor for the feedforward layers in ParticleTransformerEncoder.
    pair_embed_dims: List[int], optional
        Dimensionality of the pair embeddings for the interaction features.
    attention: Dict, optional
        Physics-aware gate settings, e.g. `{'use_gating': True}` (see `gate_kwargs_from_config`).
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
    """
    def __init__(
        self,
        config: Optional[ParticleTransformerConfig] = None,
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
        dropout: Optional[float] = None,
        expansion_factor: Optional[int] = None,
        pair_embed_dims: Optional[List[int]] = None,
        attention: Optional[Dict] = None,
        mask: Optional[bool] = None,
        weights: Optional[str] = None,
        inference: Optional[bool] = False,
        # Mass-gate normalisation
        norm_stats: Optional[Dict[str, Tuple[float, float]]] = None,
        normalize: Optional[List[bool]] = None,
        debug: bool = False,
        debug_batches: int = 2
    ):
        super(ParticleTransformer, self).__init__()

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
            self.dropout = dropout if dropout is not None else config.dropout
            self.expansion_factor = expansion_factor if expansion_factor is not None else config.expansion_factor
            self.pair_embed_dims = pair_embed_dims if pair_embed_dims is not None else config.pair_embed_dims
            self.attention = attention if attention is not None else config.attention
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
            self.dropout = dropout if dropout is not None else 0.1
            self.expansion_factor = expansion_factor if expansion_factor is not None else 4
            self.pair_embed_dims = pair_embed_dims if pair_embed_dims is not None else [64, 64, 64]
            self.attention = attention if attention is not None else {}
            self.mask = mask if mask is not None else False
            self.weights = weights if weights is not None else None
            self.inference = inference if inference is not None else False

        # Initialize the class token
        self.cls_token = nn.Parameter(torch.zeros(1, 1, self.embed_dim), requires_grad=True)
        nn.init.normal_(self.cls_token, mean=0.0, std=1.0)

        self.processor = ParticleProcessor(verbose=debug)
        self.encoder = ParticleTransformerEncoder(
            embed_dim=self.embed_dim,
            num_heads=self.num_heads,
            num_layers=self.num_layers,
            dropout=self.dropout,
            expansion_factor=self.expansion_factor,
            pair_embed_dims=self.pair_embed_dims,
            attention_config=self.attention
        )

        # For self-supervised learning
        self.fc = nn.Linear(self.max_num_particles * self.embed_dim, self.num_particle_features)

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

        # Process particles to get interaction embeddings and the physical 4-momenta
        x, U, p4 = self.processor(x, x_raw)
        self._log_mass_gate(x_raw, p4)

        # Pass through particle attention blocks
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
            x = self.fc(x)  # (B, F)

            return x
