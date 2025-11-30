import torch
from torch import nn, Tensor

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
    def __init__(
        self,
        num_classes: int,
        input_dim: int,
        hidden_dim: int,
        num_layers: int = 2,
        dropout: float = 0.25
    ):
        super(Classifier, self).__init__()
        layers = []
        for i in range(num_layers):
            layers.append(nn.Linear(input_dim if i == 0 else hidden_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
        
        layers.append(nn.Linear(input_dim if num_layers == 0 else hidden_dim, num_classes))
        self.layers = nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        return self.layers(x)