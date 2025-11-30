from models.shared_layers import CustomActivationFunction
import torch
from torch import nn, Tensor
import math

# Linear Embedding
class LinearEmbedding(nn.Module):
    def __init__(self, vars, d_model, linear: bool = False):
        super(LinearEmbedding, self).__init__()
        self.embedding_layer = nn.Linear(vars, d_model)
        self.relu = nn.ReLU()
        self.linear = linear

    def forward(self, x: Tensor):
        if self.linear:
            return self.embedding_layer(x)
        else:
            return self.relu(self.embedding_layer(x))

# Sine positional encoding
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_seq_len=6, base=10000.):
        super(PositionalEncoding, self).__init__()
        self.d_model = d_model

        # Create a 2D positional encoding matrix
        pe = torch.zeros(max_seq_len, d_model)
        position = torch.arange(0, max_seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(base) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        # Add an extra dimension for the batch size
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Input Shape: (batch_size, max_seq_len, d_model)
        Output Shape: (batch_size, max_seq_len, d_model)
        """
        x = x * math.sqrt(self.d_model)
        x = x + self.pe[:, :x.size(1), :]
        return x

# Transformer Autoencoder
class TransformerAutoencoder(nn.Module):
    def __init__(self, d_model, num_heads, num_layers, d_ff, max_seq_len, output_vars, dropout, device):
        super(TransformerAutoencoder, self).__init__()
        self.custom_act = CustomActivationFunction()
        self.trans = nn.Transformer(d_model=d_model, nhead=num_heads,
                                    num_encoder_layers=num_layers, num_decoder_layers=num_layers,
                                    dim_feedforward=d_ff, dropout=dropout,
                                    activation=self.custom_act, custom_encoder=None,
                                    custom_decoder=None, layer_norm_eps=1e-05,
                                    batch_first=True, norm_first=False,
                                    device=device, dtype=None)
        self.embedding = LinearEmbedding(output_vars + (output_vars % 3), d_model)
        self.pos_enc = PositionalEncoding(d_model, max_seq_len, base=100)
        self.dense = nn.Linear(d_model, 128)
        self.output = nn.Linear(128, output_vars + (output_vars % 3))

    def forward(self, src):
        src_mask = (src[:,:,0] == 0)
        src = torch.where(src == 999, torch.tensor(1, dtype=src.dtype, device=src.device), src)        
        src = self.embedding(src)
        src = self.pos_enc(src)
        tgt = self.trans.encoder(src, src_key_padding_mask=src_mask)
        return self.output(self.custom_act(self.dense(self.trans.decoder(src, tgt, tgt_key_padding_mask=src_mask))))