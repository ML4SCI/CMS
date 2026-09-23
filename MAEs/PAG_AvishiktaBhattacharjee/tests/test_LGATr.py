from typing import Dict

import pytest

import torch
from torch import Tensor

from src.models import LorentzGATr
from src.configs import LGATrConfig
from src.utils import set_seed

set_seed(42)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


@pytest.fixture
def dummy_data() -> Dict[str, Tensor]:
    batch_size = 4
    max_num_particles = 128
    num_particle_features = 4
    embed_dim = 128
    num_classes = 10

    # Simulated particle features: shape (B, N, F)
    x = torch.randn(batch_size, max_num_particles, num_particle_features)

    # For LorentzGATr: output classification
    y = torch.randint(0, num_classes, (batch_size, num_classes))

    # For LorentzGATr: masked particles' indices
    mask_idx = torch.randint(0, max_num_particles, (batch_size, 1))

    return {
        'x': x,
        'y': y,
        'mask_idx': mask_idx,
        'num_classes': num_classes,
        'embed_dim': embed_dim,
        'max_num_particles': max_num_particles,
        'num_particle_features': num_particle_features,
    }


def make_config(**overrides: Dict) -> LGATrConfig:
    default = {
        'num_classes': 10,
        'embed_dim': 128,
        'num_heads': 4,
        'num_layers': 2,
        'num_cls_layers': 1,
        'num_mlp_layers': 1,
        'hidden_dim': 32,
        'hidden_mv_channels': 8,
        'in_s_channels': 1,
        'out_s_channels': 1,
        'hidden_s_channels': 1,
        'attention': {},
        'mlp': {},
        'reinsert_mv_channels': None,
        'reinsert_s_channels': None,
        'dropout': 0.1,
        'expansion_factor': 2,
        'max_num_particles': 128,
        'num_particle_features': 4,
        'mask': False
    }
    default.update(overrides)

    return LGATrConfig(**default)


def test_lgatr_forward(dummy_data: Dict[str, Tensor]):
    """Test forward pass of LGATr."""
    config = make_config(
        num_classes=dummy_data['num_classes'],
        embed_dim=dummy_data['embed_dim'],
        max_num_particles=dummy_data['max_num_particles'],
        num_particle_features=dummy_data['num_particle_features'],
        num_heads=4,
        num_layers=2,
        num_cls_layers=1,
    )
    model = LorentzGATr(config=config).to(device)
    x = dummy_data['x']

    output = model(x.to(device))

    # Output shape should be (B, num_classes)
    assert output.shape == (x.size(0), dummy_data['num_classes'])


def test_masked_lgatr_forward(dummy_data: Dict[str, Tensor]):
    """Test forward pass of MaskedLGATr."""
    config = make_config(
        max_num_particles=dummy_data['max_num_particles'],
        num_particle_features=dummy_data['num_particle_features'],
        embed_dim=dummy_data['embed_dim'],
        num_heads=4,
        num_layers=2,
        mask=True
    )
    model = LorentzGATr(config=config).to(device)
    x = dummy_data['x']
    mask_idx = dummy_data['mask_idx']

    output = model(x.to(device), mask_idx)

    # Output shape should be (B, num_particle_features)
    assert output.shape == (x.size(0), dummy_data['num_particle_features'])


def test_lgatr_batch_first(dummy_data: Dict[str, Tensor]):
    """Check that LGATr supports batch_first."""
    config = make_config(
        num_classes=dummy_data['num_classes'],
        embed_dim=dummy_data['embed_dim'],
        max_num_particles=dummy_data['max_num_particles'],
        num_particle_features=dummy_data['num_particle_features'],
        num_heads=4,
        num_layers=2,
        num_cls_layers=1,
    )
    model = LorentzGATr(config=config).to(device)
    x = dummy_data['x']
    output = model(x.to(device))

    assert output.shape[0] == x.size(0)


def test_masked_lgatr_grad(dummy_data: Dict[str, Tensor]):
    """Ensure gradients flow through MaskedLGATr."""
    config = make_config(
        max_num_particles=dummy_data['max_num_particles'],
        num_particle_features=dummy_data['num_particle_features'],
        embed_dim=dummy_data['embed_dim'],
        num_heads=4,
        num_layers=2,
        mask=True
    )
    model = LorentzGATr(config=config).to(device)
    x = dummy_data['x'].requires_grad_()
    mask_idx = dummy_data['mask_idx']
    output = model(x.to(device), mask_idx)
    loss = output.sum()
    loss.backward()

    assert x.grad is not None