from typing import List, Optional

import torch
from torch import nn, Tensor

from .classifier import ClassAttentionBlock, Classifier
from .feedforward import Feedforward
from .processor import InteractionEmbedding, ParticleProcessor
from ..configs import ParticleTransformerConfig


class ParticleAttentionBlock(nn.Module):
    def __init__(
        self,
        embed_dim: int = 128,
        num_heads: int = 8,
        dropout: float = 0.1,
        expansion_factor: int = 4,
    ):
        super(ParticleAttentionBlock, self).__init__()
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"

        self.embed_dim = embed_dim
        self.num_heads = num_heads

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

    def forward(self, x: Tensor, padding_mask: Tensor, U: Optional[Tensor] = None) -> Tensor:
        residual = x
        x = self.layernorm1(x)
        x, _ = self.pmha(x, x, x, key_padding_mask=padding_mask, attn_mask=U)
        x = self.layernorm2(x)
        x = self.dropout(x)

        x += residual
        x = self.feedforward(x)

        return x
    

class ParticleTransformerEncoder(nn.Module):
    def __init__(
        self,
        embed_dim: int = 128,
        num_heads: int = 8,
        num_layers: int = 8,
        dropout: float = 0.1,
        expansion_factor: int = 4,
        pair_embed_dims: List[int] = [64, 64, 64]
    ):
        super(ParticleTransformerEncoder, self).__init__()
        self.proj = nn.Linear(4, embed_dim)
        self.interaction_embed = InteractionEmbedding(
            num_interaction_features=4,
            pair_embed_dims=pair_embed_dims + [num_heads]
        )
        self.encoder = nn.ModuleList([
            ParticleAttentionBlock(
                embed_dim=embed_dim,
                num_heads=num_heads,
                dropout=dropout,
                expansion_factor=expansion_factor
            ) for _ in range(num_layers)
        ])
    
    def forward(self, x: Tensor, padding_mask: Tensor, U: Tensor) -> Tensor:
        B, N, F = x.shape  # (batch_size, max_num_particles, num_particle_features)

        # Embed interaction features
        U = self.interaction_embed(U)  # (B * num_heads, N, N)

        # Project input features to embedding dimension
        x = self.proj(x)  # (B, N, embed_dim)

        # Encoder with particle attention blocks
        for layer in self.encoder:
            x = layer(x, padding_mask, U)  # (B, N, embed_dim)

        return x  # (B, N, embed_dim)
    

class ParticleTransformer(nn.Module):
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
    mask: bool, optional
        Indicates whether the model is for self-supervised learning or classification.
    weights: str, optional
        Path to the pretrained weights.
    inference: bool, optional
        Whether to use the model for inference.

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
        mask: Optional[bool] = None,
        weights: Optional[str] = None,
        inference: Optional[bool] = False
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
            self.mask = mask if mask is not None else False
            self.weights = weights if weights is not None else None
            self.inference = inference if inference is not None else False

        # Initialize the class token
        self.cls_token = nn.Parameter(torch.zeros(1, 1, self.embed_dim), requires_grad=True)
        nn.init.normal_(self.cls_token, mean=0.0, std=1.0)

        self.processor = ParticleProcessor()
        self.encoder = ParticleTransformerEncoder(
            embed_dim=self.embed_dim,
            num_heads=self.num_heads,
            num_layers=self.num_layers,
            dropout=self.dropout,
            expansion_factor=self.expansion_factor,
            pair_embed_dims=self.pair_embed_dims
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
            batch_indices = torch.arange(x.size(0), device=x.device)
            padding_mask[batch_indices, mask_idx] = 0.0

        # Process particles to get interaction embeddings and multivectors (if applicable)
        x, U = self.processor(x)

        # Pass through equilinear layer and particle attention blocks
        x = self.encoder(x, padding_mask, U)

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