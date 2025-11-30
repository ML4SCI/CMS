from typing import List, Tuple
import torch
from torch import nn, Tensor
from lgatr.interface import embed_vector


class ParticleProcessor(nn.Module):
    def __init__(self, to_multivector: bool = False):
        super(ParticleProcessor, self).__init__()
        self.to_multivector = to_multivector

    def _get_interaction(self, x: Tensor) -> Tensor:
        # Identify the padding particles (assume padding particles have zero energy)
        mask = x[..., 3] > 0  # (B, N)

        # Extract kinematic features
        pT = x[..., 0]
        eta = x[..., 1]
        phi = x[..., 2]
        energy = x[..., 3]

        # Compute the momentum 3-vectors
        px = pT * torch.cos(phi)
        py = pT * torch.sin(phi)
        pz = pT * torch.sinh(eta)
        momentum = torch.stack((px, py, pz), dim=-1)  # (B, N, 3)

        # Compute physics-inspired pairwise features
        eps = 1e-8  # to avoid log(0) and division by zero
        eta_diff = eta.unsqueeze(2) - eta.unsqueeze(1)
        phi_diff = ((phi.unsqueeze(2) - phi.unsqueeze(1)) + torch.pi) % (2 * torch.pi) - torch.pi
        min_pT = torch.minimum(pT.unsqueeze(2), pT.unsqueeze(1))
        pT_sum = pT.unsqueeze(2) + pT.unsqueeze(1)
        energy_sum = energy.unsqueeze(2) + energy.unsqueeze(1)
        momentum_sum = momentum.unsqueeze(2) + momentum.unsqueeze(1)

        delta = torch.sqrt(eta_diff**2 + phi_diff**2)
        kT = min_pT * delta
        z = min_pT / (pT_sum + eps)
        m2 = energy_sum**2 - momentum_sum.norm(dim=-1)**2

        if torch.isnan(delta).any():
            raise ValueError("NaN detected in delta calculation.")
        if torch.isnan(kT).any():
            raise ValueError("NaN detected in kT calculation.")
        if torch.isnan(z).any():
            raise ValueError("NaN detected in z calculation.")
        if torch.isnan(m2).any():
            raise ValueError("NaN detected in m2 calculation.")

        # Take the logarithm of the features
        ln_delta = torch.log(torch.clamp(delta, min=eps))
        ln_kT = torch.log(torch.clamp(kT, min=eps))
        ln_z = torch.log(torch.clamp(z, min=eps))
        ln_m2 = torch.log(torch.clamp(m2, min=eps))

        if torch.isnan(ln_delta).any():
            raise ValueError("NaN detected in ln_delta calculation.")
        if torch.isnan(ln_kT).any():
            raise ValueError("NaN detected in ln_kT calculation.")
        if torch.isnan(ln_z).any():
            raise ValueError("NaN detected in ln_z calculation.")
        if torch.isnan(ln_m2).any():
            raise ValueError("NaN detected in ln_m2 calculation.")

        # Combine the features into a single tensor
        U_vals = torch.stack((ln_delta, ln_kT, ln_z, ln_m2), dim=-1)  # (B, N, N, 4)

        # Initialize U with large negative values
        U = torch.full_like(U_vals, fill_value=-1e9)

        # Determine valid pairs (both particles are not padding)
        valid_pairs = mask.unsqueeze(2) & mask.unsqueeze(1)

        # Fill U only for valid pairs
        U[valid_pairs] = U_vals[valid_pairs]

        # Zero out diagonal elements (self-interactions)
        idx = torch.arange(U.size(1), device=U.device)
        U[:, idx, idx, :] = 0

        return U  # (B, N, N, 4)

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        B, N, F = x.shape  # (batch_size, max_num_particles, num_particle_features)

        # Compute interaction embeddings
        U = self._get_interaction(x)  # (B, N, N, 4)

        if self.to_multivector:
            # Convert (pT, eta, phi, E) to (E, px, py, pz) for EquiLinear layer
            # x = torch.stack([
            #     x[..., 3],  # E
            #     x[..., 0] * torch.cos(x[..., 2]),  # px = pT * cos(phi)
            #     x[..., 0] * torch.sin(x[..., 2]),  # py = pT * sin(phi)
            #     x[..., 0] * torch.sinh(x[..., 1]),  # pz = pT * sinh(eta)
            # ], dim=-1)
            
            # Lorentz-equivariant embedding
            x = x.view(B, N, 1, F)  # for compatibility with the EquiLinear layer
            x = embed_vector(x)  # (B, N, 1, 16)
            x = x.view(B, N, 16)

        return x, U
    

class InteractionEmbedding(nn.Module):
    def __init__(
        self,
        num_interaction_features: int = 4,
        pair_embed_dims: List[int] = [64, 64, 64, 8]
    ):
        super(InteractionEmbedding, self).__init__()
        input_dim = num_interaction_features
        layers = [nn.BatchNorm1d(input_dim)]
        for dim in pair_embed_dims:
            layers.extend([
                nn.Conv1d(input_dim, dim, kernel_size=1),
                nn.BatchNorm1d(dim),
                nn.GELU()
            ])
            input_dim = dim
            
        self.embed = nn.Sequential(*layers)

    def forward(self, U: Tensor) -> Tensor:
        B, N, _, F = U.shape  # (batch_size, max_num_particles, max_num_particles, num_interaction_features)
        U = U.view(B, N * N, F).transpose(1, 2)  # (B, F, N * N)
        U = self.embed(U)  # (B, num_heads, N * N)
        U = U.view(B * U.shape[1], N, N)  # (B * num_heads, N, N)

        return U