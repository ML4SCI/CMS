from torch import nn, Tensor


class Feedforward(nn.Module):
    def __init__(
        self,
        embed_dim: int = 128,
        expansion_factor: int = 4,
        dropout: float = 0.1,
    ):
        super(Feedforward, self).__init__()
        self.embed_dim = embed_dim
        self.dim_feedforward = embed_dim * expansion_factor

        self.layernorm1 = nn.LayerNorm(embed_dim)
        self.linear1 = nn.Linear(embed_dim, self.dim_feedforward)
        self.act = nn.GELU()
        self.dropout1 = nn.Dropout(dropout)

        self.layernorm2 = nn.LayerNorm(self.dim_feedforward)
        self.linear2 = nn.Linear(self.dim_feedforward, embed_dim)
        self.dropout2 = nn.Dropout(dropout)
        
    def forward(self, x: Tensor) -> Tensor:
        residual = x
        x = self.layernorm1(x)
        x = self.linear1(x)
        x = self.act(x)
        x = self.dropout1(x)

        x = self.layernorm2(x)
        x = self.linear2(x)
        x = self.dropout2(x)
        x += residual

        return x