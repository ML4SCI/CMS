import os
import tempfile
from typing import Dict

import matplotlib
matplotlib.use('Agg')

import numpy as np
import pytest

import torch
import torch.nn.functional as F

from src.configs import LorentzParTConfig, ParticleTransformerConfig, TrainConfig
from src.engine import Trainer
from src.models import LorentzParT, ParticleTransformer
from src.models.particle_transformer import ParticleAttentionBlock, gate_kwargs_from_config
from src.utils import accuracy_metric_ce, set_seed
from src.models.processor import ParticleProcessor, denorm_constants
from src.utils.data import (
    JetClassDataset, assert_uniform_scaling, compute_norm_stats, shared_scale
)
from src.utils.viz import plot_confusion_matrix, plot_roc_curve

set_seed(42)

NORMALIZE = [True, False, False, True]
GATE = {'use_gating': True}
MAX_NUM_PARTICLES = 32


def make_jets(num_jets: int = 6, max_num_particles: int = MAX_NUM_PARTICLES, seed: int = 0) -> np.ndarray:
    """Massless constituents in the JetClass layout (num_jets, 4, max_num_particles), zero-padded."""
    rng = np.random.default_rng(seed)
    X = np.zeros((num_jets, 4, max_num_particles), dtype=np.float32)
    for j in range(num_jets):
        n = rng.integers(5, max_num_particles)
        pT = rng.exponential(20.0, n) + 0.5
        eta = rng.normal(0.0, 0.4, n)
        X[j, 0, :n] = pT
        X[j, 1, :n] = eta
        X[j, 2, :n] = rng.normal(0.0, 0.4, n)
        X[j, 3, :n] = pT * np.cosh(eta)

    return X


def jet_mass(X: np.ndarray) -> np.ndarray:
    pT, eta, phi, E = X[:, 0].astype(np.float64), X[:, 1], X[:, 2], X[:, 3]
    p = np.stack([(pT * np.cos(phi)).sum(1), (pT * np.sin(phi)).sum(1), (pT * np.sinh(eta)).sum(1)], axis=-1)
    m2 = E.sum(1) ** 2 - (p ** 2).sum(-1)

    return np.sqrt(np.clip(m2, 1e-6, None))


@pytest.fixture
def jets() -> Dict:
    X = make_jets()
    y = np.eye(10, dtype=np.float32)[np.arange(len(X)) % 10]
    norm_dict = compute_norm_stats(X)
    dataset = JetClassDataset(X, y, NORMALIZE, norm_dict)
    x = torch.stack([dataset[i][0] for i in range(len(dataset))])
    labels = torch.stack([dataset[i][1] for i in range(len(dataset))]).argmax(-1)

    return {'X': X, 'x': x, 'y': labels, 'norm_dict': norm_dict}


def lorentz_config(**overrides) -> LorentzParTConfig:
    default = dict(
        embed_dim=32, num_heads=4, num_layers=2, num_cls_layers=1, hidden_dim=32,
        pair_embed_dims=[8], max_num_particles=MAX_NUM_PARTICLES, attention=GATE, mask=False
    )
    default.update(overrides)

    return LorentzParTConfig(**default)


def part_config(**overrides) -> ParticleTransformerConfig:
    default = dict(
        embed_dim=32, num_heads=4, num_layers=2, num_cls_layers=1, hidden_dim=32,
        pair_embed_dims=[8], max_num_particles=MAX_NUM_PARTICLES, attention=GATE, mask=False
    )
    default.update(overrides)

    return ParticleTransformerConfig(**default)


def params_with_grad(model: torch.nn.Module, name: str):
    return [n for n, p in model.named_parameters() if name in n and p.grad is not None and p.grad.abs().sum() > 0]


def test_gate_config_parsing():
    assert gate_kwargs_from_config(None) == {'gate_type': None}
    assert gate_kwargs_from_config({}) == {'gate_type': None}
    assert gate_kwargs_from_config({'use_gating': True}) == {'gate_type': 'headwise'}

    kwargs = gate_kwargs_from_config({'use_gating': True, 'gate_type': 'elementwise', 'mass_freq_range': [0.01, 0.4]})
    assert kwargs == {'gate_type': 'elementwise', 'mass_freq_range': (0.01, 0.4)}

    # A typo must not silently switch the gate off
    with pytest.raises(ValueError):
        gate_kwargs_from_config({'use_gate': True})


def test_dataset_normalization_is_not_repeated(jets: Dict):
    X = make_jets()
    y = np.eye(10, dtype=np.float32)[np.arange(len(X)) % 10]
    dataset = JetClassDataset(X, y, NORMALIZE, jets['norm_dict'])

    first, second = dataset[0][0], dataset[0][0]
    assert torch.equal(first, second)
    assert np.array_equal(X, make_jets())


def test_gate_starts_as_identity():
    torch.manual_seed(0)
    plain = ParticleAttentionBlock(embed_dim=32, num_heads=4, dropout=0.0)
    gated = ParticleAttentionBlock(embed_dim=32, num_heads=4, dropout=0.0, gate_type='headwise')

    # Shared weights load; only the gate's own tensors are new
    result = gated.load_state_dict(plain.state_dict(), strict=False)
    assert not result.unexpected_keys
    assert {k.split('.')[0] for k in result.missing_keys} <= {'physics_proj', 'mass_proj', 'gate_proj', 'm_freqs'}

    x = torch.randn(2, 8, 32)
    padding_mask = torch.zeros(2, 8)
    U = torch.randn(2 * 4, 8, 8)
    p4 = torch.rand(2, 8, 4) * 50
    plain.eval()
    gated.eval()

    assert torch.allclose(plain(x, padding_mask, U), gated(x, padding_mask, U, p4=p4), atol=1e-5)


def test_ungated_block_matches_plain_part_parameters():
    block = ParticleAttentionBlock(embed_dim=32, num_heads=4)
    names = {n.split('.')[0] for n, _ in block.named_parameters()}

    assert names == {'layernorm1', 'pmha', 'layernorm2', 'feedforward'}


def test_denormalize_inverts_dataset(jets: Dict):
    model = LorentzParT(config=lorentz_config(), norm_stats=jets['norm_dict'], normalize=NORMALIZE)
    raw = torch.from_numpy(jets['X']).transpose(1, 2)  # (B, N, 4)
    rec = model.denormalize(jets['x'])
    valid = raw[..., 3] > 0

    assert torch.allclose(rec[valid], raw[valid], rtol=1e-5, atol=1e-4)
    assert (rec[~valid] == 0).all()


def test_gate_sees_physical_jet_mass(jets: Dict):
    model = LorentzParT(config=lorentz_config(), norm_stats=jets['norm_dict'], normalize=NORMALIZE)
    captured = {}
    hook = model.processor.register_forward_hook(lambda m, i, o: captured.update(p4=o[2]))
    model.eval()
    with torch.no_grad():
        model(jets['x'])
    hook.remove()

    mass = model.encoder.encoder[0].compute_mass(captured['p4']).squeeze(1).double().numpy()
    assert np.allclose(mass, jet_mass(jets['X']), rtol=1e-3)
    assert np.median(mass) > 1.0


def test_mass_encoder_separates_w_and_z():
    block = ParticleAttentionBlock(embed_dim=32, num_heads=4, gate_type='headwise')
    with torch.no_grad():
        feat = block.encode_mass(torch.tensor([[80.379], [91.188]])).squeeze(1)

    assert (feat[0] - feat[1]).norm() / feat.norm(dim=-1).mean() > 1e-3


@pytest.mark.parametrize('model_cls, config_fn', [(LorentzParT, lorentz_config), (ParticleTransformer, part_config)])
def test_gate_and_mass_branch_are_trainable(jets: Dict, model_cls, config_fn):
    model = model_cls(config=config_fn(), norm_stats=jets['norm_dict'], normalize=NORMALIZE)
    model.train()
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-2)

    F.cross_entropy(model(jets['x']), jets['y']).backward()
    assert params_with_grad(model, 'gate_proj')
    assert not params_with_grad(model, 'mass_proj')  # zero-initialised gate: expected at step 0

    optimizer.step()
    optimizer.zero_grad()
    F.cross_entropy(model(jets['x']), jets['y']).backward()
    assert params_with_grad(model, 'mass_proj')


@pytest.mark.parametrize('model_cls, config_fn, out_dim', [(LorentzParT, lorentz_config, 4), (ParticleTransformer, part_config, 4)])
def test_masked_pretraining_forward(jets: Dict, model_cls, config_fn, out_dim):
    model = model_cls(config=config_fn(mask=True), norm_stats=jets['norm_dict'], normalize=NORMALIZE)
    mask_idx = torch.zeros(len(jets['x']), 1, dtype=torch.int64)
    output = model(jets['x'], mask_idx)

    assert output.shape == (len(jets['x']), out_dim)
    output.sum().backward()
    assert params_with_grad(model, 'gate_proj')


@pytest.mark.parametrize('model_cls, config_fn', [(LorentzParT, lorentz_config), (ParticleTransformer, part_config)])
def test_pretrained_gate_weights_load(jets: Dict, model_cls, config_fn, capsys):
    pretrained = model_cls(config=config_fn(mask=True), norm_stats=jets['norm_dict'], normalize=NORMALIZE)
    with torch.no_grad():
        for block in pretrained.encoder.encoder:
            block.gate_proj.weight.normal_()

    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, 'pretrained.pt')
        torch.save(pretrained.state_dict(), path)
        classifier = model_cls(config=config_fn(weights=path), norm_stats=jets['norm_dict'], normalize=NORMALIZE)

    for src, dst in zip(pretrained.encoder.encoder, classifier.encoder.encoder):
        assert torch.equal(src.gate_proj.weight, dst.gate_proj.weight)
    assert '0 missing, 0 unexpected' in capsys.readouterr().out


def test_plots_are_saved_per_model(jets: Dict, tmp_path):
    X = make_jets(num_jets=20)
    y = np.eye(10, dtype=np.float32)[np.arange(20) % 10]
    dataset = JetClassDataset(X, y, NORMALIZE, jets['norm_dict'])
    model = ParticleTransformer(config=part_config(), norm_stats=jets['norm_dict'], normalize=NORMALIZE)
    config = TrainConfig(
        batch_size=10,
        logging_dir=str(tmp_path / 'logs'),
        save_fig=True,
        plots_dir=str(tmp_path / 'plots')
    )
    trainer = Trainer(
        model=model,
        train_dataset=dataset,
        val_dataset=dataset,
        test_dataset=dataset,
        device=torch.device('cpu'),
        metric=accuracy_metric_ce,
        config=config
    )
    trainer.evaluate(loss_type='cross_entropy', plot=[plot_roc_curve, plot_confusion_matrix])

    plots = tmp_path / 'plots' / 'ParticleTransformer'
    assert sorted(p.name for p in plots.iterdir()) == [
        f"{trainer.run_name}_confusion_matrix.png",
        f"{trainer.run_name}_roc_curve.png"
    ]


@pytest.mark.parametrize('model_cls, config_fn', [(LorentzParT, lorentz_config), (ParticleTransformer, part_config)])
def test_best_model_reloads_strictly(jets: Dict, model_cls, config_fn):
    trained = model_cls(config=config_fn(), norm_stats=jets['norm_dict'], normalize=NORMALIZE)
    rebuilt = model_cls(config=config_fn(), norm_stats=jets['norm_dict'], normalize=NORMALIZE)
    rebuilt.load_state_dict(trained.state_dict())  # strict, as in Trainer.load_best_model

    trained.eval()
    rebuilt.eval()
    with torch.no_grad():
        assert torch.allclose(trained(jets['x']), rebuilt(jets['x']))


# ---------------------------------------------------------------------------
# Normalisation: pT and E must share ONE scale, or m^2 = E^2 - |p|^2 goes negative
# ---------------------------------------------------------------------------
def test_shared_scale_is_used_for_both_pT_and_energy(jets: Dict):
    """pT and E must be divided by the same constant, else p/s is not a 4-vector."""
    s = shared_scale(jets['norm_dict'])
    scale, shift = denorm_constants(jets['norm_dict'], NORMALIZE)

    assert scale[0] == scale[3] == s, "pT and E must share one scale"
    assert shift[0] == shift[3] == 0.0, "a 4-vector may be scaled, never shifted"
    assert scale[1] == scale[2] == 1.0 and shift[1] == shift[2] == 0.0, "eta/phi untouched"


def test_both_dataset_branches_normalise_identically(jets: Dict):
    """
    The MAE branch (mask_mode set) and the classification branch (mask_mode=None)
    must apply the same transform, or pretraining and fine-tuning see different
    input distributions and denormalize() scales the classification inputs twice.
    """
    X, y = jets['X'], np.eye(10, dtype=np.float32)[np.arange(len(jets['X'])) % 10]
    cls_x = JetClassDataset(X, y, NORMALIZE, jets['norm_dict'], mask_mode=None)[0][0]
    mae_x = JetClassDataset(X, y, NORMALIZE, jets['norm_dict'], mask_mode='first')[0][0]

    s = shared_scale(jets['norm_dict'])
    expected_pT = torch.from_numpy(X[0, 0] / s)
    expected_E = torch.from_numpy(X[0, 3] / s)

    for name, got in (('classification', cls_x), ('MAE', mae_x)):
        # mask_mode zeroes the masked constituent by design, so compare the rest
        keep = got[:, 3] != 0
        assert keep.sum() > 1, f"{name} branch: nothing left to compare"
        assert torch.allclose(got[keep, 0], expected_pT[keep], atol=1e-5), f"{name} branch: pT not x/s"
        assert torch.allclose(got[keep, 3], expected_E[keep], atol=1e-5), f"{name} branch: E not x/s"


def test_normalised_particles_keep_a_non_negative_mass_squared(jets: Dict):
    """The signature test: two different scales send m^2 negative for every particle."""
    x = jets['x']  # normalised, straight out of JetClassDataset
    pT, eta, phi, E = x[..., 0], x[..., 1], x[..., 2], x[..., 3]
    valid = E != 0
    p2 = (pT * torch.cos(phi)) ** 2 + (pT * torch.sin(phi)) ** 2 + (pT * torch.sinh(eta)) ** 2
    m2 = (E ** 2 - p2)[valid]

    assert (m2 >= -1e-5).all(), (
        f"{(m2 < -1e-5).float().mean().item():.1%} of normalised particles have m^2 < 0 -- "
        "pT and E are on different scales"
    )


def test_pairwise_ln_m2_channel_is_not_clamped(jets: Dict):
    """U[..., 3] = ln(m_ij^2) was 100% clamped to ln(1e-8) under the two-scale scheme."""
    proc = ParticleProcessor(to_multivector=False)
    U = proc._get_interaction(jets['x'])

    real = jets['x'][..., 3] != 0
    pairs = (real.unsqueeze(2) & real.unsqueeze(1))
    pairs &= ~torch.eye(pairs.shape[1], dtype=torch.bool).unsqueeze(0)
    ln_m2 = U[..., 3][pairs]
    clamped = (ln_m2 <= float(np.log(1e-8)) + 1e-3).float().mean().item()

    assert clamped < 0.05, f"{clamped:.1%} of pairwise ln(m_ij^2) clamped to the floor"
    assert ln_m2.std().item() > 0.1, "pairwise mass channel carries no information"


def test_normalising_commutes_with_a_lorentz_boost(jets: Dict):
    """
    Uniform scaling is a change of units, so it must commute with any boost.
    Scaling E and the momenta differently does not.
    """
    s = shared_scale(jets['norm_dict'])
    X = jets['X']
    pT, eta, phi, E = (X[:, 0].astype(np.float64), X[:, 1].astype(np.float64),
                       X[:, 2].astype(np.float64), X[:, 3].astype(np.float64))
    p4 = np.stack([E, pT * np.cos(phi), pT * np.sin(phi), pT * np.sinh(eta)], axis=-1)

    beta = 0.6
    gamma = 1.0 / np.sqrt(1 - beta ** 2)
    L = np.array([[gamma, 0, 0, -gamma * beta], [0, 1, 0, 0], [0, 0, 1, 0], [-gamma * beta, 0, 0, gamma]])

    scale_then_boost = (p4 / s) @ L.T
    boost_then_scale = (p4 @ L.T) / s
    assert np.abs(scale_then_boost - boost_then_scale).max() < 1e-9


def test_eta_or_phi_normalisation_is_rejected():
    """Scaling the angles breaks (pT, eta, phi, E) -> (E, px, py, pz) irreversibly."""
    for bad in ([True, True, False, True], [True, False, True, True]):
        with pytest.raises(ValueError, match='eta.*phi|must be False'):
            assert_uniform_scaling(bad)


def test_mass_gate_sees_physical_gev(jets: Dict):
    """denormalize -> processor -> p4 must reproduce the jet mass computed from raw GeV."""
    model = LorentzParT(config=lorentz_config(), norm_stats=jets['norm_dict'], normalize=NORMALIZE)
    model.eval()
    with torch.no_grad():
        _, _, p4 = model.processor(jets['x'], model.denormalize(jets['x']))
        m = (p4[..., 0].sum(1) ** 2 - p4[..., 1:].sum(1).pow(2).sum(-1)).clamp(min=1e-6).sqrt()

    assert torch.allclose(m.double(), torch.from_numpy(jet_mass(jets['X'])), rtol=2e-3)
