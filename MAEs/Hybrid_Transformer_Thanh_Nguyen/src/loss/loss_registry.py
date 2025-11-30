from torch import nn

from .conservation_loss import ConservationLoss


LOSS_REGISTRY = {
    'conservation_loss': ConservationLoss,
    'cross_entropy_loss': nn.CrossEntropyLoss,
    'bce_with_logits_loss': nn.BCEWithLogitsLoss,
    'mse_loss': nn.MSELoss,
    # Add more loss functions here as needed
}