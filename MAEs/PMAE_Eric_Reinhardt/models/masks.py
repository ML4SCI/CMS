import torch
import random
from torch import nn

class ParticleMask(nn.Module):
    def __init__(self, group_size=4):
        super(ParticleMask, self).__init__()
        self.group_size = group_size

    def forward(self, x):
        assert x.dim() == 3, "Input tensor must be 3-dimensional (batch_size, seq_len, dim_in)"
        batch, seq_len, features = x.size()
        assert features == self.group_size, "Sequence length must be divisible by group_size"

        mask = torch.ones(batch, seq_len, features, device=x.device)

        random_idxs = torch.randint(0, seq_len, (batch,), device=x.device)
        for b, idx in enumerate(random_idxs):
            mask[b, idx, :] = 0

        masked_x = x * mask

        sums = masked_x[:, :, 4].sum(dim=1)
        condition_met = sums >= 2

        replacement_values = torch.zeros_like(masked_x)
        for b, idx in enumerate(random_idxs):
            if condition_met[b]:
                replacement_values[b, idx, 3] = 999

        mask_for_replacement = (replacement_values != 0).float()
        final_output = (1 - mask_for_replacement) * masked_x + replacement_values

        return final_output

class SpecificParticleMask(nn.Module):
    def __init__(self, group_size=4, particle=0):
        super(SpecificParticleMask, self).__init__()
        self.group_size = group_size
        self.particle = particle

    def forward(self, x):
        assert x.dim() == 3, "Input tensor must be 3-dimensional (batch_size, seq_len, dim_in)"
        batch, seq_len, features = x.size()

        mask = torch.ones(batch, seq_len, features, device=x.device)

        for b in range(batch):
            mask[b, self.particle, :] = 0

        masked_x = x * mask

        sums = masked_x[:, :, 4].sum(dim=1)
        condition_met = sums >= 2

        replacement_values = torch.zeros_like(masked_x)
        for b in range(batch):
            if condition_met[b]:
                replacement_values[b, self.particle:, 3] = 999

        mask_for_replacement = (replacement_values != 0).float()
        final_output = (1 - mask_for_replacement) * masked_x + replacement_values

        return final_output

class KinematicMask(nn.Module):
    def __init__(self, mask_count):
        super(KinematicMask, self).__init__()
        self.mask_count = mask_count

    def forward(self, x):
        assert x.dim() == 3, "Input tensor must be 3-dimensional (batch_size, seq_len, dim_in)"
        batch_size, seq_len, _ = x.size()
        assert self.mask_count <= seq_len, "Mask count must be less than or equal to the sequence length"

        # Generate a mask tensor with the same shape as the input tensor
        mask = torch.ones(batch_size, seq_len, 1, device=x.device)

        # Generate a set of unique random indices to mask for each sample in the batch
        for b in range(batch_size):
            mask_indices = set()
            while len(mask_indices) < self.mask_count:
                mask_indices.add(random.randint(0, seq_len - 1))

            # Zero out the elements at the selected indices
            for idx in mask_indices:
                mask[b, idx] = 0

        return x * mask
