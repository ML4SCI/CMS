import torch
from torch import nn

# Custom activation function
class CustomActivationFunction(nn.Module):
    def forward(self, x):
        return torch.relu(x + torch.sin(x) ** 2)