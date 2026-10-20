from typing import List, Dict, Optional, Tuple
import torch
from torch import nn, Tensor
from lgatr.interface import embed_vector

from ..utils.data.normalize import assert_uniform_scaling, shared_scale


FEATURE_NAMES = ['pT', 'eta', 'phi', 'energy']


class ParticleProcessor(nn.Module):
    """
    Converts collider-coordinate particle features into
      (a) Lorentz multivector tokens for the equivariant encoder (or a passthrough),
      (b) pairwise interaction features U,
      (c) physical 4-momentum p4 = (E, px, py, pz) for the invariant-mass gate.

    Input convention (matches JetClass / _get_interaction):
        slot 0 = pT   transverse momentum, GeV
        slot 1 = eta  pseudorapidity, dimensionless
        slot 2 = phi  azimuthal angle, radians
        slot 3 = E    energy, GeV
    """

    def __init__(self, to_multivector: bool = False, verbose: bool = False):
        super(ParticleProcessor, self).__init__()
        self.to_multivector = to_multivector
        self.verbose = verbose
        self._reported = False
        self._warned_raw = False

    @staticmethod
    def to_cartesian(x: Tensor) -> Tensor:
        """
        (pT, eta, phi, E) -> (E, px, py, pz).

        px = pT * cos(phi)
        py = pT * sin(phi)
        pz = pT * sinh(eta)

        This is the SAME conversion _get_interaction already performs for the
        pairwise features; it must also be applied before any m^2 computation.
        """
        pT, eta, phi, energy = x[..., 0], x[..., 1], x[..., 2], x[..., 3]
        px = pT * torch.cos(phi)
        py = pT * torch.sin(phi)
        pz = pT * torch.sinh(eta)
        return torch.stack((energy, px, py, pz), dim=-1).to(torch.float32)

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

        delta = torch.sqrt(eta_diff**2 + phi_diff**2 + eps)
        kT = min_pT * delta
        z = min_pT / (pT_sum + eps)
        m2 = energy_sum**2 - momentum_sum.norm(dim=-1)**2

        # Take the logarithm of the features
        ln_delta = torch.log(torch.clamp(delta, min=eps))
        ln_kT = torch.log(torch.clamp(kT, min=eps))
        ln_z = torch.log(torch.clamp(z, min=eps))
        ln_m2 = torch.log(torch.clamp(m2, min=eps))

        for name, t in (("ln_delta", ln_delta), ("ln_kT", ln_kT),
                        ("ln_z", ln_z), ("ln_m2", ln_m2)):
            if torch.isnan(t).any():
                raise ValueError(f"NaN detected in {name} calculation.")

        # Combine the features into a single tensor
        U_vals = torch.stack((ln_delta, ln_kT, ln_z, ln_m2), dim=-1)  # (B, N, N, 4)

        # Padded pairs stay at zero (not -1e9) so they do not dominate the
        # BatchNorm statistics of InteractionEmbedding
        U = torch.zeros_like(U_vals)

        # Determine valid pairs (both particles are not padding)
        valid_pairs = mask.unsqueeze(2) & mask.unsqueeze(1)
        U[valid_pairs] = U_vals[valid_pairs]

        # Zero out diagonal elements (self-interactions)
        idx = torch.arange(U.size(1), device=U.device)
        U[:, idx, idx, :] = 0

        return U  # (B, N, N, 4)

    def forward(self, x: Tensor, x_raw: Optional[Tensor] = None) -> Tuple[Tensor, Tensor, Tensor]:
        """
        x     : (B, N, 4) NORMALIZED (pT, eta, phi, E)  -- network input
        x_raw : (B, N, 4) UN-NORMALIZED (pT, eta, phi, E) in GeV.
                Required for a physically meaningful mass gate. If None, falls
                back to x and the returned p4 is NOT in GeV.

        returns (tokens, U, p4)
            tokens : (B, N, 16) multivector embeddings, or (B, N, 4) passthrough
            U      : (B, N, N, 4) pairwise interaction features
            p4     : (B, N, 4) physical (E, px, py, pz), padded rows zeroed
        """
        B, N, F = x.shape

        # Interaction features are computed on the normalized tensor, as upstream does.
        # This is valid because pT and E share ONE scale s, so the normalised particle
        # is p/s -- still a 4-vector. Each channel then differs from the GeV version by
        # a CONSTANT only:
        #     ln_delta : +0          (depends on eta, phi alone)
        #     ln_z     : +0          (a ratio of pT's, s cancels)
        #     ln_kT    : -ln(s)
        #     ln_m2    : -2 ln(s)    and m^2 -> m^2/s^2, so the sign is kept
        # InteractionEmbedding's leading BatchNorm1d removes those constants exactly.
        U = self._get_interaction(x)  # (B, N, N, 4)

        if self.verbose and not self._reported:
            self._reported = True
            self._report_interaction(x, U)

        # Physical 4-momentum for the invariant-mass gate
        src = x if x_raw is None else x_raw
        if x_raw is None and self.verbose and not self._warned_raw:
            self._warned_raw = True
            print("[processor] WARNING: x_raw is None -> p4 is in NORMALIZED units.")
            print("[processor]          The mass gate will see a dimensionless number, not GeV.")
            print("[processor]          Build the model with norm_stats=norm_dict.")
        p4 = self.to_cartesian(src)  # (B, N, 4)

        # Zero padded slots so downstream sums cannot pick up spurious mass.
        # Without this, normalized padding rows contribute (0 - mu)/sigma != 0,
        # and the error scales with the number of pads -- i.e. with multiplicity,
        # which is class-correlated.
        pad = (src[..., 3] > 0).unsqueeze(-1).to(p4.dtype)  # (B, N, 1)
        p4 = p4 * pad

        if self.to_multivector:
            # embed_vector expects a Lorentz 4-vector in (t, x, y, z) order.
            # Feeding raw (pT, eta, phi, E) puts an angle in the py slot and
            # energy in the pz slot, which breaks every downstream
            # Lorentz-algebraic operation.
            p4_tok = self.to_cartesian(x)  # normalized inputs
            p4_tok = p4_tok.view(B, N, 1, 4)
            x = embed_vector(p4_tok)  # (B, N, 1, 16)
            x = x.view(B, N, 16)

        return x, U, p4

    @staticmethod
    @torch.no_grad()
    def _report_interaction(x: Tensor, U: Tensor) -> None:
        m = x[..., 3] > 0
        pv = m.unsqueeze(2) & m.unsqueeze(1)
        eye = torch.eye(U.shape[1], dtype=torch.bool, device=U.device)
        sel = pv & ~eye
        if not sel.any():
            return

        ln_m2 = U[..., 3][sel]
        floor = float(torch.log(torch.tensor(1e-8)))
        frac = (ln_m2 <= floor + 1e-3).float().mean().item()
        print("[processor] _get_interaction input : NORMALIZED by the shared pT/E scale")
        print(f"[processor] U[...,3] = ln(m_ij^2)  : {frac:.1%} of valid pairs clamped to ln(1e-8)={floor:.2f}")
        print(f"[processor] U[...,3] spread        : std {ln_m2.std().item():.4f} over {int(sel.sum())} valid pairs")
        if frac > 0.5:
            print("[processor] -> WARNING: the pairwise mass channel is nearly constant.")
            print("[processor]    Expected ~0% now. A high fraction means pT and E are still")
            print("[processor]    being divided by different constants -- check shared_scale().")
        else:
            print("[processor] -> pairwise mass channel is live (m_ij^2 > 0, sign preserved).")
        print("[processor] the GLOBAL mass gate uses x_raw (GeV) instead.")


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


def denorm_constants(
    norm_stats: Dict[str, Tuple[float, float]],
    normalize: List[bool]
) -> Tuple[List[float], List[float]]:
    """
    Per-feature (scale, shift) such that x_raw = x * scale + shift undoes JetClassDataset.

    JetClassDataset applies, per feature and only where normalize[i] is True:
        i in (0, 3)  pT, E     ->  x / s               (ONE shared s, scale only)
        i in (1, 2)  eta, phi  ->  (x - mean) / std    (scale + shift)

    pT and E share a single constant so the particle stays a genuine 4-vector under
    normalisation; see `src.utils.data.normalize.shared_scale` for why two separate
    means send m^2 = E^2 - |p|^2 negative.
    """
    s = shared_scale(norm_stats)

    scale, shift = [], []
    for i, name in enumerate(FEATURE_NAMES):
        mean, std = norm_stats[name]
        if not normalize[i]:
            scale.append(1.0)
            shift.append(0.0)
        elif i in (0, 3):  # pT, E : divided by the SHARED scale
            scale.append(float(s))
            shift.append(0.0)
        else:  # eta, phi : standardised
            scale.append(float(std))
            shift.append(float(mean))

    return scale, shift


class DenormalizeMixin:
    """
    Gives a model access to physical (GeV) particle features for the invariant-mass gate.

    The model registers the inverse of the JetClassDataset normalisation as buffers
    (`_nscale`, `_nshift`) and exposes `denormalize()`. The network itself still
    consumes the normalized tensor.
    """

    def _setup_denormalize(
        self,
        norm_stats: Optional[Dict[str, Tuple[float, float]]],
        normalize: Optional[List[bool]],
        debug: bool = False,
        debug_batches: int = 2,
        gated: bool = True
    ) -> None:
        self.norm_stats = norm_stats
        self.debug = debug
        self.debug_batches = debug_batches
        self._dbg_seen = 0

        if norm_stats is None:
            self.normalize = None
            if self.debug and gated:
                print("[__init__] WARNING: norm_stats=None -> the mass gate will run on NORMALIZED units.")
                print(f"[__init__]          Build with {self.__class__.__name__}(config=..., norm_stats=norm_dict).")
            return

        self.normalize = list(normalize) if normalize is not None else [True, False, False, True]
        assert_uniform_scaling(self.normalize)
        scale, shift = denorm_constants(norm_stats, self.normalize)
        self.register_buffer('_nscale', torch.tensor(scale, dtype=torch.float32))
        self.register_buffer('_nshift', torch.tensor(shift, dtype=torch.float32))

        if self.debug:
            self._report_denormalize(scale, shift)

    def _report_denormalize(self, scale: List[float], shift: List[float]) -> None:
        print("=" * 72)
        print("[__init__] MASS-GATE NORMALISATION REGISTERED")
        print("=" * 72)
        s = shared_scale(self.norm_stats)
        print(f"  normalize flags        : {self.normalize}")
        print(f"  shared pT/E scale s    : {s:.6f}   (one constant for BOTH, keeps p a 4-vector)")
        print(f"  {'feature':<10}{'mean':>14}{'std':>14}{'scale':>12}{'shift':>10}")
        for i, name in enumerate(FEATURE_NAMES):
            mean, std = self.norm_stats[name]
            print(f"  {name:<10}{mean:>14.6f}{std:>14.6f}{scale[i]:>12.4f}{shift[i]:>10.4f}")

        # Prove the formula on a probe: a real 100 GeV / eta=0.4 particle
        pt, eta = 100.0, 0.4
        e = pt * float(torch.cosh(torch.tensor(eta)))
        pt_n = pt / s if self.normalize[0] else pt
        e_n = e / s if self.normalize[3] else e
        pt_rec = pt_n * scale[0] + shift[0]
        e_rec = e_n * scale[3] + shift[3]
        print()
        print("  round-trip check on a 100.00 GeV, eta=0.4 constituent:")
        print(f"    {'':<26}{'pT [GeV]':>12}{'E [GeV]':>12}")
        print(f"    {'truth':<26}{pt:>12.2f}{e:>12.2f}")
        print(f"    {'x*scale + shift':<26}{pt_rec:>12.2f}{e_rec:>12.2f}")
        assert abs(pt_rec - pt) < 1e-3 and abs(e_rec - e) < 1e-3, \
            "denormalize is not the inverse of JetClassDataset"

        # The signature check: m^2 of the NORMALISED particle must stay >= 0.
        # With two different scales this is negative for essentially every particle.
        m2_n = e_n ** 2 - (pt_n * float(torch.cosh(torch.tensor(eta)))) ** 2
        print(f"    normalised m^2 (must be >= 0)          : {m2_n:+.6e}")
        assert m2_n >= -1e-6, (
            "normalised m^2 is negative: pT and E are being divided by different "
            "constants, which is not a Lorentz transformation. See shared_scale()."
        )
        print("  OK: denormalize inverts the dataset transform and the signature holds.")
        print("=" * 72)

    def denormalize(self, x: Tensor) -> Tensor:
        """
        Undo the transform JetClassDataset applied, so the mass gate sees GeV.

        Padded rows are kept at exactly zero so downstream masking still works.
        """
        keep = (x[..., 3:4] != 0).to(x.dtype)  # (B, N, 1)
        return (x * self._nscale + self._nshift) * keep

    @torch.no_grad()
    def _log_mass_gate(self, x_raw: Optional[Tensor], p4: Tensor) -> None:
        """Prints the jet mass reaching the gate for the first `debug_batches` forward passes."""
        if not self.debug or self._dbg_seen >= self.debug_batches:
            return
        self._dbg_seen += 1

        valid = p4.abs().sum(-1) > 0
        m2 = p4[..., 0].sum(dim=1) ** 2 - p4[..., 1:].sum(dim=1).pow(2).sum(-1)
        m = m2.clamp(min=1e-6).sqrt()
        clamped = (m2 <= 1e-6).float().mean().item()
        units = "GeV" if self.norm_stats is not None else "NORMALIZED units"

        print("-" * 72)
        print(f"[forward {self._dbg_seen}/{self.debug_batches}] INVARIANT-MASS GATE  (units: {units})")
        print(f"  x_raw supplied to processor : {x_raw is not None}")
        if valid.any():
            if x_raw is not None:
                pt_raw = x_raw[..., 0][valid]
                print(f"  constituent pT   med/max    : {pt_raw.median().item():8.2f} / {pt_raw.max().item():8.2f}")
            print(f"  constituent E    med/max    : "
                  f"{p4[..., 0][valid].median().item():8.2f} / {p4[..., 0][valid].max().item():8.2f}")
        print(f"  jet mass  q25/med/q75       : "
              f"{m.quantile(0.25).item():8.2f} / {m.median().item():8.2f} / {m.quantile(0.75).item():8.2f}")
        print(f"  jet mass  min/max           : {m.min().item():8.2f} / {m.max().item():8.2f}")
        print(f"  fraction clamped at m^2<=0  : {clamped:8.4f}")
        if self.norm_stats is None:
            print("  VERDICT: gate is reading normalised units.")
        elif m.median().item() < 1.0:
            print("  VERDICT: masses are ~0 -- the gate is degenerate.")
        else:
            print("  VERDICT: masses are physical. m_W=80.4  m_Z=91.2  m_H=125.3  m_t=172.5 GeV")
        print("-" * 72)
