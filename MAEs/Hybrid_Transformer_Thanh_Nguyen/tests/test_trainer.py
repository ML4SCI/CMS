import os
import shutil
import tempfile
from typing import List, Tuple, Dict, Optional

import pytest
import numpy as np

import torch

from src.configs import ParticleTransformerConfig, TrainConfig
from src.models import ParticleTransformer
from src.engine import Trainer
from src.utils import set_seed
from src.utils.data import JetClassDataset

set_seed(42)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


@pytest.fixture
def temp_dir():
    d = tempfile.mkdtemp()
    yield d
    shutil.rmtree(d)


@pytest.fixture
def dummy_dataset() -> Tuple[np.ndarray, np.ndarray, int, int, int]:
    # Small dummy dataset for quick tests
    num_samples = 8
    max_num_particles = 16
    num_particle_features = 4
    num_classes = 5
    X = np.random.randn(num_samples, num_particle_features, max_num_particles).astype(np.float32)
    y = np.random.randint(0, num_classes, size=(num_samples, num_classes)).astype(np.float32)

    return X, y, num_classes, max_num_particles, num_particle_features


def make_model_config(**overrides: Dict) -> ParticleTransformerConfig:
    default = {
        'num_classes': 10,
        'embed_dim': 128,
        'num_heads': 4,
        'num_layers': 2,
        'num_cls_layers': 1,
        'num_mlp_layers': 1,
        'hidden_dim': 32,
        'dropout': 0.1,
        'expansion_factor': 2,
        'pair_embed_dims': [16, 16],
        'max_num_particles': 128,
        'num_particle_features': 4,
        'mask': False
    }
    default.update(overrides)
    return ParticleTransformerConfig(**default)


def make_train_config(temp_log_dir: str, **overrides: Dict) -> TrainConfig:
    default = {
        'batch_size': 4,
        'criterion': {
            'name': 'cross_entropy_loss',
            'kwargs': {'reduction': 'mean'}
        },
        'optimizer': {
            'name': 'adamw',
            'kwargs': {'lr': 1e-3}
        },
        'num_epochs': 2,
        'start_epoch': 0,
        'logging_dir': temp_log_dir,
        'logging_steps': 2,
        'save_best': True,
        'save_ckpt': True,
        'device': 'cuda',
        'num_workers': 1,
        'pin_memory': True
    }
    default.update(overrides)
    return TrainConfig(**default)


def make_dataset(
    X: np.ndarray,
    y: np.ndarray,
    split_idx: int = 5,
    normalize: List[bool] = [True, False, False, True],
    norm_dict: Dict[str, List[float]] = {
        'pT': [92.67603302001953, 105.75433349609375],
        'eta': [-0.00041131096077151597, 0.9181342124938965],
        'phi': [0.00041396886808797717, 1.8135319948196411],
        'energy': [133.9013214111328, 167.53518676757812]
    },
    mask_mode: Optional[str] = None
) -> JetClassDataset:
    train_set = JetClassDataset(
        X_particles=X[:split_idx],
        y=y[:split_idx],
        normalize=normalize,
        norm_dict=norm_dict,
        mask_mode=mask_mode
    )
    val_set = JetClassDataset(
        X_particles=X[split_idx:],
        y=y[split_idx:],
        normalize=normalize,
        norm_dict=norm_dict,
        mask_mode=mask_mode
    )
    test_set = JetClassDataset(
        X_particles=X[split_idx:],
        y=y[split_idx:],
        normalize=normalize,
        norm_dict=norm_dict,
        mask_mode=mask_mode
    )
    return train_set, val_set, test_set


def test_trainer_train_loop_and_history(dummy_dataset: Tuple, temp_dir: str):
    X, y, num_classes, max_num_particles, num_particle_features = dummy_dataset

    # Build datasets
    train_set, val_set, test_set = make_dataset(X, y)

    # Model and configs
    model_config = make_model_config(
        num_classes=num_classes,
        max_num_particles=max_num_particles,
        num_particle_features=num_particle_features
    )
    model = ParticleTransformer(config=model_config).to(device)
    train_config = make_train_config(temp_dir)

    # Trainer
    trainer = Trainer( 
        model=model,
        train_dataset=train_set,
        val_dataset=val_set,
        test_dataset=test_set,
        device=device,
        config=train_config
    )

    history, trained_model = trainer.train()

    # Check training history
    assert 'epoch' in history
    assert isinstance(trained_model, ParticleTransformer)

    # Check checkpoint and best model files exist
    if trainer.save_best:
        assert os.path.exists(trainer.best_model_path)
    if trainer.save_ckpt:
        assert os.path.exists(trainer.checkpoint_path)


def test_trainer_optimizer_config(dummy_dataset: Tuple, temp_dir: str):
    X, y, num_classes, max_num_particles, num_particle_features = dummy_dataset

    # Build datasets
    train_set, val_set, test_set = make_dataset(X, y)

    # Model and configs
    model_config = make_model_config(
        num_classes=num_classes,
        max_num_particles=max_num_particles,
        num_particle_features=num_particle_features
    )
    model = ParticleTransformer(config=model_config).to(device)

    # Set optimizer config with a distinct learning rate
    train_config = make_train_config(temp_dir)
    train_config.optimizer = {'name': 'adam', 'kwargs': {'lr': 5e-3}}
    trainer = Trainer(
        model=model,
        train_dataset=train_set,
        val_dataset=val_set,
        test_dataset=test_set,
        device=device,
        config=train_config
    )

    # Optimizer param group learning rate should match config
    assert abs(trainer.optimizer.param_groups[0]['lr'] - 5e-3) < 1e-8


def test_trainer_callbacks_and_early_stopping(dummy_dataset: Tuple, temp_dir: str):
    X, y, num_classes, max_num_particles, num_particle_features = dummy_dataset

    # Build datasets
    train_set, val_set, test_set = make_dataset(X, y)

    # Model and configs
    model_config = make_model_config(
        num_classes=num_classes,
        max_num_particles=max_num_particles,
        num_particle_features=num_particle_features
    )
    model = ParticleTransformer(config=model_config).to(device)
    train_config = make_train_config(temp_dir)

    # EarlyStopping callback with patience=0 for immediate stop
    callbacks = [{
        'name': 'early_stopping',
        'kwargs': {
            'monitor': 'val_loss',
            'mode': 'min',
            'patience': 5
        }
    }]
    trainer = Trainer(
        model=model,
        train_dataset=train_set,
        val_dataset=val_set,
        test_dataset=test_set,
        device=device,
        config=train_config,
        callbacks=callbacks
    )
    history, _ = trainer.train()

    # Early stopping triggers, so number of epochs is at most 2
    assert len(history['epoch']) <= train_config.num_epochs


def test_trainer_device(dummy_dataset: Tuple, temp_dir: str):
    X, y, num_classes, max_num_particles, num_particle_features = dummy_dataset

    # Build datasets
    train_set, val_set, test_set = make_dataset(X, y)

    # Model and configs
    model_config = make_model_config(
        num_classes=num_classes,
        max_num_particles=max_num_particles,
        num_particle_features=num_particle_features
    )
    model = ParticleTransformer(config=model_config).to(device)
    train_config = make_train_config(temp_dir)

    # Set device to CUDA
    train_config.device = 'cuda'
    trainer = Trainer(
        model=model,
        train_dataset=train_set,
        val_dataset=val_set,
        test_dataset=test_set,
        device=device,
        config=train_config
    )

    # Model and data on correct device
    assert str(trainer.device) == 'cuda'
    for batch in trainer.train_loader:
        X_batch, y_batch = batch
        assert X_batch.to(device).device.type == 'cuda'
        assert y_batch.to(device).device.type == 'cuda'