import torch
from torch import nn, Tensor
import torch.nn.functional as F

from .feedforward import Feedforward


class ClassAttentionBlock(nn.Module):
    def __init__(
        self,
        embed_dim: int = 128,
        num_heads: int = 8,
        dropout: float = 0.0,
        expansion_factor: int = 4,
    ):
        super(ClassAttentionBlock, self).__init__()
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"

        self.embed_dim = embed_dim
        self.num_heads = num_heads

        self.layernorm1 = nn.LayerNorm(embed_dim)
        self.mha = nn.MultiheadAttention(
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

    def forward(self, x: Tensor, x_cls: Tensor, padding_mask: Tensor) -> Tensor:
        B, N, D = x.shape  # (batch_size, max_num_particles, embed_dim)

        # Prepend the class token to the input
        with torch.no_grad():
            padding_mask = torch.cat((torch.zeros_like(padding_mask[:, :1]), padding_mask), dim=1)

        residual = x_cls
        x = torch.cat((x_cls, x), dim=1)  # (B, N + 1, D)
        x = self.layernorm1(x)
        x, _ = self.mha(x_cls, x, x, key_padding_mask=padding_mask)
        x = self.layernorm2(x)
        x = self.dropout(x)

        x += residual
        x = self.feedforward(x)

        return x


class Classifier(nn.Module):
    """
    MLP head with a cosine-similarity output layer.

    The final layer compares the L2-normalised features with L2-normalised class
    weights (no bias), so the logits are bounded cosine similarities times `scale`.
    """
    def __init__(
        self,
        num_classes: int,
        input_dim: int,
        hidden_dim: int,
        num_layers: int = 2,
        dropout: float = 0.25,
        scale: float = 30.0
    ):
        super(Classifier, self).__init__()
        self.scale = scale

        layers = []
        for i in range(num_layers):
            layers.append(nn.Linear(input_dim if i == 0 else hidden_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))

        self.hidden_layers = nn.Sequential(*layers)

        # Dimension produced by the last hidden layer
        final_in_dim = input_dim if num_layers == 0 else hidden_dim

        # Learnable class weights (no bias) instead of a final nn.Linear
        self.weight = nn.Parameter(torch.empty(num_classes, final_in_dim))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, x: Tensor) -> Tensor:
        x = self.hidden_layers(x)

        # L2 normalize both the latent features and the class weights
        x_norm = F.normalize(x, p=2, dim=1)
        w_norm = F.normalize(self.weight, p=2, dim=1)

        # Bounded cosine similarity, scaled
        return self.scale * F.linear(x_norm, w_norm)
