from typing import List

import torch
from torch import nn, Tensor
import torch.nn.functional as F


class ConservationLoss(nn.Module):
    """
    Loss function for the masked model to learn Conservation of Momentum and Energy.

    Parameters
    ----------
    beta: float
        The weight to reward the model for predicting extreme eta.
    gamma: float
        The weight to encourage the model to predict pT and energy near zero to offset a known positive bias:
        `nn.MSELoss()` produced a positive bias on pT and energy because they have right-skewed distributions.
    loss_coef: List[float]
        Coefficients for each loss term.
    reduction: str
        Reduction method to apply to the loss.

    Returns
    -------
    Tensor
        The computed loss.

    .. References::
        Eric Reinhardt.
        [GSOC 2023 with ML4SCI: Reconstruction and Classification of Particle Collisions with Masked Transformer Autoencoders](https://medium.com/@eric0reinhardt/gsoc-2023-with-ml4sci-reconstruction-and-classification-of-particle-collisions-with-masked-bab8b38958df). 
        *Medium*, 2023.
    """
    def __init__(
        self,
        alpha: float = 0.1,
        beta: float = 1.0,
        gamma: float = 0.5,
        loss_coef: List[float] = [0.25, 0.25, 0.25, 0.25],  # [pT, eta, phi, energy]
        reduction: str = 'mean'
    ):
        super(ConservationLoss, self).__init__()
        self.beta = beta
        self.gamma = gamma
        self.loss_coef = loss_coef
        self.reduction = reduction

    def _pT_loss(self, pT_pred: Tensor, pT_true: Tensor) -> Tensor:
        # self.gamma = torch.tensor([self.gamma], device=pT_pred.device)
        # loss = F.mse_loss(pT_pred, pT_true, reduction='none')
        # above_zero = self.gamma / (1 + torch.exp(3 * -pT_pred)) - self.gamma
        # below_zero = self.gamma / (1 + torch.exp(-self.gamma))
        # loss += torch.where(pT_pred >= 0, above_zero, below_zero)

        # return loss.mean()
        return torch.sqrt(F.mse_loss(pT_pred, pT_true, reduction=self.reduction))

    def _eta_loss(self, eta_pred: Tensor, eta_true: Tensor) -> Tensor:
        # self.beta = torch.tensor([self.beta], device=eta_pred.device)
        # loss = F.mse_loss(eta_pred, eta_true, reduction='none') - eta_pred**2 * self.beta

        # return loss.mean()
        return F.l1_loss(eta_pred, eta_true, reduction=self.reduction)

    def _phi_loss(self, phi_pred: Tensor, phi_true: Tensor) -> Tensor:
        # Compute sine and cosine for both predictions and targets
        sin_pred, cos_pred = torch.sin(phi_pred), torch.cos(phi_pred)
        sin_true, cos_true = torch.sin(phi_true), torch.cos(phi_true)

        # Compute cosine similarity between predicted and true values
        cos_sim = cos_true * cos_pred + sin_true * sin_pred
        loss = (1.0 - cos_sim).mean()

        return loss

    def _energy_loss(self, E_pred: Tensor, E_true: Tensor) -> Tensor:
        # self.gamma = torch.tensor([self.gamma], device=E_pred.device)
        # loss = F.mse_loss(E_pred, E_true, reduction='none')
        # above_zero = self.gamma / (1 + torch.exp(3 * -E_pred)) - self.gamma
        # below_zero = self.gamma / (1 + torch.exp(-self.gamma))
        # loss += torch.where(E_pred >= 0, above_zero, below_zero)

        # return loss.mean()
        return torch.sqrt(F.mse_loss(E_pred, E_true, reduction=self.reduction))

    def forward(self, pred: Tensor, target: Tensor) -> Tensor:
        # Extract pT, eta, phi, E from pred and target
        pT_pred, eta_pred, phi_pred, E_pred = pred[:, 0], pred[:, 1], pred[:, 2], pred[:, 3]
        pT_true, eta_true, phi_true, E_true = target[:, 0], target[:, 1], target[:, 2], target[:, 3]

        # Compute individual losses
        pT_loss = self._pT_loss(pT_pred, pT_true)
        eta_loss = self._eta_loss(eta_pred, eta_true)
        phi_loss = self._phi_loss(phi_pred, phi_true)
        energy_loss = self._energy_loss(E_pred, E_true)

        # # Momentum vector components
        # px_pred = pT_pred * torch.cos(phi_pred)
        # py_pred = pT_pred * torch.sin(phi_pred)
        # pz_pred = pT_pred * torch.sinh(eta_pred)
        # pred_vec = torch.stack([px_pred, py_pred, pz_pred], dim=1)

        # px_target = pT_true * torch.cos(phi_true)
        # py_target = pT_true * torch.sin(phi_true)
        # pz_target = pT_true * torch.sinh(eta_true)
        # target_vec = torch.stack([px_target, py_target, pz_target], dim=1)

        # # Angular loss
        # cos_sim = F.cosine_similarity(pred_vec, target_vec, dim=1)
        # angular_loss = (1.0 - cos_sim).mean()

        # # Magnitude loss
        # pred_norm = pred_vec.norm(dim=1)
        # target_norm = target_vec.norm(dim=1)
        # denom = target_norm.mean() + 1e-6
        # mag_loss = F.l1_loss(pred_norm, target_norm, reduction=self.reduction) / denom

        # # Momentum loss
        # momentum_loss = self.alpha * angular_loss + (1 - self.alpha) * mag_loss

        # Total loss
        loss = 0.0
        loss += self.loss_coef[0] * pT_loss
        loss += self.loss_coef[1] * eta_loss
        loss += self.loss_coef[2] * phi_loss
        loss += self.loss_coef[3] * energy_loss
        # loss += self.loss_coef[4] * momentum_loss

        return loss, (pT_loss, eta_loss, phi_loss, energy_loss)