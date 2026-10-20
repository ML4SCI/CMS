"""LLoCa-ParT — a Lorentz-locally-canonicalized Particle Transformer.

Wraps a weaver ``ParticleTransformer`` so that

1. a local Lorentz frame is predicted for every particle
   (:class:`~variants.lloca.frames.FramesNet`),
2. the kinematic token features are re-expressed in those local frames, making
   them Lorentz-invariant,
3. the encoder blocks transport tensorial messages between frames
   (:class:`~variants.lloca.attention.LLoCaAttentionBlock`).

Reference: arXiv:2505.20280 Sec. 4 and App. D.3 ("LLoCa-ParT").

Deviations from the paper, deliberate and documented
----------------------------------------------------
``pair_bias_frame="global"`` (default)
    The paper additionally recomputes ParT's pairwise edge features (the
    attention bias) from four-momenta in local frames.  Doing so makes the bias
    asymmetric — pair ``(i, j)`` must be evaluated in the receiver's frame — and
    requires replacing weaver's ``PairEmbed`` with an ``O(N^2)`` four-vector
    transform.  This implementation keeps weaver's stock global ``pair_embed``
    instead, so that

    * the arm differs from the baseline in exactly the frame construction and
      the attention message transport, and
    * the pair bias is bit-identical to the baseline's.

    Of ParT's four pair features only ``m^2`` is Lorentz-invariant, so the
    global bias is a third symmetry-breaking channel alongside the paper's two
    (non-invariant scalars and reference vectors).  Since the paper already
    breaks the symmetry deliberately down to the beam-axis ``SO(2)`` subgroup
    (App. E, Tab. 6), this is a difference of degree, not of kind — but it does
    mean the assembled model is *not* exactly Lorentz-invariant.  Use
    ``symmetry_breaking=False`` **and** ``pair_embed_dims=None`` to obtain an
    exactly invariant model, which is what the equivariance test exercises.

Class attention
    Kept as stock weaver scalar attention, matching the paper ("retaining
    standard scalar message passing for class attention").

Precision
    The paper runs LLoCa-ParT in single precision throughout to avoid
    equivariance violations from AMP; frame construction itself runs in
    float64.  :class:`LLoCaParT` follows this, and the training entrypoint
    disables AMP for this arm.
"""

from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
from torch import Tensor, nn

from weaver.nn.model.ParticleTransformer import ParticleTransformer, SequenceTrimmer

from .attention import LLoCaAttentionBlock
from .frames import FramesNet, invert_frames, to_time_first

__all__ = ["LLoCaParT", "NUM_KINEMATIC_FEATURES", "local_kinematic_features"]

#: Number of leading token-feature channels replaced by local-frame kinematics.
#: The project's loader emits
#: ``[pt, eta, phi, energy, deta, dphi, d0val, d0err, dzval, dzerr, charge, PID x5]``
#: so channels 0-5 are kinematic and 6-15 (displacement, charge, PID) are
#: treated as scalars and passed through untouched, per App. D.3.
NUM_KINEMATIC_FEATURES = 6

_EPS = 1e-8


def _pt_eta_phi(p: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
    """Return ``(pt, eta, phi)`` from a time-first four-vector ``(..., 4)``."""
    px, py, pz = p[..., 1], p[..., 2], p[..., 3]
    pt = torch.sqrt(px * px + py * py + _EPS)
    eta = torch.asinh(pz / pt)
    phi = torch.atan2(py, px)
    return pt, eta, phi


def _delta_phi(a: Tensor, b: Tensor) -> Tensor:
    """Signed azimuthal difference wrapped to ``(-pi, pi]``."""
    return torch.remainder(a - b + torch.pi, 2 * torch.pi) - torch.pi


def local_kinematic_features(
    p_local: Tensor, jet_local: Tensor
) -> Tensor:
    """Build the six local-frame kinematic channels.

    All quantities are computed from four-momenta already expressed in each
    particle's own local frame, hence Lorentz-invariant.

    Channels, matching the slots they replace in the loader's feature vector:

    ===  ==========================================================
    0    ``log pT`` in the local frame
    1    ``eta`` in the local frame
    2    ``phi`` in the local frame
    3    ``log E`` in the local frame
    4    ``delta eta`` between particle and jet axis, both in frame ``L_i``
    5    ``delta phi`` between particle and jet axis, both in frame ``L_i``
    ===  ==========================================================

    Parameters
    ----------
    p_local : Tensor
        ``(B, P, 4)`` particle four-momenta in their own local frames.
    jet_local : Tensor
        ``(B, P, 4)`` the jet four-momentum transformed into each particle's
        local frame (so index ``i`` holds ``L_i p_jet``).

    Returns
    -------
    Tensor
        ``(B, P, 6)``.
    """
    pt, eta, phi = _pt_eta_phi(p_local)
    _, eta_jet, phi_jet = _pt_eta_phi(jet_local)
    energy = p_local[..., 0]

    return torch.stack(
        [
            torch.log(pt.clamp_min(_EPS)),
            eta,
            phi,
            torch.log(energy.abs().clamp_min(_EPS)),
            eta - eta_jet,
            _delta_phi(phi, phi_jet),
        ],
        dim=-1,
    )


class LLoCaParT(nn.Module):
    """Weaver ``ParticleTransformer`` made Lorentz-equivariant via LLoCa.

    Parameters
    ----------
    input_dim : int
        Number of per-particle input features (16 for this project's loader).
    num_classes : int
        Number of output classes.
    embed_dims, pair_embed_dims, num_heads, num_layers, num_cls_layers
        Forwarded to weaver ``ParticleTransformer``.
    dropout, expansion_factor
        LLoCa block hyperparameters.
    frames_hidden_dim : int
        Hidden width of the frame-prediction MLP ``phi``.  The paper uses 128;
        App. E reports 16 is nearly as good.  ``phi`` allocates a
        ``(B, P, P + 3, frames_hidden_dim)`` activation, which dominates this
        module's memory, so the default here is a memory-conscious 64.
    frames_dropout : float
        Dropout inside ``phi`` (App. E suggests 0.2 only in the low-data regime).
    frames_min_mass : float
        Mass regulator ``m_eps`` (App. D.1); ``5e-3`` for tagging.  Note this is
        a frame-dependent preprocessing step, so it introduces a tiny explicit
        symmetry breaking; set to ``0`` for exact-equivariance checks.
    symmetry_breaking : bool
        Append the time/beam reference particles to the frame-prediction input
        ("RV" of App. D.3).
    frames_use_nis : bool
        Feed the *global-frame* kinematic channels to the frame-prediction MLP
        as scalars.  These are only invariant under the beam-axis ``SO(2)``
        subgroup, so this is the paper's second symmetry-breaking mechanism
        ("NIS", App. D.3) and is enabled by default.  When ``False`` only the
        genuinely scalar channels (displacement, charge, PID) are used, which is
        required for exact Lorentz invariance.
    num_vector_channels : int or None
        Four-vector channels per attention head; ``None`` uses the paper's
        equal-mix default (8 scalars + 2 four-vectors for a 16-dim head).
    freeze_frames_net : bool
        Freeze ``phi`` after initialization.  App. E shows a fixed Frames-Net
        still beats the non-equivariant baseline, and it is cheaper.
    trim : bool
        Enable weaver's ``SequenceTrimmer``.  It is owned by *this* module
        rather than the inner transformer because it permutes and truncates
        tokens, and the frames must be built on the post-trim sequence.
    **weaver_kwargs
        Additional weaver ``ParticleTransformer`` keyword arguments.
    """

    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        *,
        embed_dims=(128, 512, 128),
        pair_embed_dims=(64, 64, 64),
        num_heads: int = 8,
        num_layers: int = 8,
        num_cls_layers: int = 2,
        dropout: float = 0.1,
        expansion_factor: int = 4,
        frames_hidden_dim: int = 64,
        frames_dropout: float = 0.0,
        frames_min_mass: float = 5e-3,
        symmetry_breaking: bool = True,
        frames_use_nis: bool = True,
        num_vector_channels: Optional[int] = None,
        freeze_frames_net: bool = False,
        trim: bool = True,
        for_inference: bool = False,
        **weaver_kwargs,
    ):
        super().__init__()
        if input_dim < NUM_KINEMATIC_FEATURES:
            raise ValueError(
                f"input_dim ({input_dim}) must be at least "
                f"{NUM_KINEMATIC_FEATURES} so the local kinematic block fits"
            )

        self.input_dim = input_dim
        self.for_inference = for_inference
        self.frames_use_nis = frames_use_nis

        # With NIS off, the frame predictor only sees the genuinely scalar
        # channels; the kinematic ones are not Lorentz-invariant.
        self._frames_scalar_slice = (
            slice(None) if frames_use_nis else slice(NUM_KINEMATIC_FEATURES, None)
        )
        frames_scalar_dim = (
            input_dim if frames_use_nis else input_dim - NUM_KINEMATIC_FEATURES
        )

        self.frames_net = FramesNet(
            scalar_dim=frames_scalar_dim,
            hidden_dim=frames_hidden_dim,
            symmetry_breaking=symmetry_breaking,
            dropout=frames_dropout,
            min_mass=frames_min_mass,
        )
        if freeze_frames_net:
            for param in self.frames_net.parameters():
                param.requires_grad_(False)

        # Trimming happens here, before frame prediction, so the inner model
        # must not trim again (it would permute tokens away from their frames).
        self.trimmer = SequenceTrimmer(enabled=trim and not for_inference)

        # fix_init is off for the same reason as the other variant arms:
        # weaver's fix_init_weight indexes layer.attn.out_proj / layer.fc2,
        # which LLoCa blocks do not have.  The equivalent rescale is applied
        # below.
        weaver_kwargs.pop("fix_init", None)
        self.part = ParticleTransformer(
            input_dim=input_dim,
            num_classes=num_classes,
            embed_dims=embed_dims,
            pair_embed_dims=pair_embed_dims,
            num_heads=num_heads,
            num_layers=num_layers,
            num_cls_layers=num_cls_layers,
            fix_init=False,
            trim=False,
            for_inference=for_inference,
            **weaver_kwargs,
        )

        embed_dim = embed_dims[-1] if len(embed_dims) > 0 else input_dim
        for i in range(num_layers):
            self.part.blocks[i] = _FrameAwareAdapter(
                LLoCaAttentionBlock(
                    embed_dim=embed_dim,
                    num_heads=num_heads,
                    dropout=dropout,
                    expansion_factor=expansion_factor,
                    num_vector_channels=num_vector_channels,
                )
            )
        # Plain list, deliberately not an nn.ModuleList: the blocks are already
        # registered through ``self.part.blocks``, and registering them twice
        # would make state_dict keys ambiguous.
        self._lloca_blocks = [adapter.block for adapter in self.part.blocks]
        self._apply_weaver_rescale()
        # Persistent, unlike every other buffer under variants/: these are
        # *data*-derived (installed from norm_stats.json by
        # :meth:`set_kinematic_norm`), not constants recomputed in __init__ the
        # way rope_inv_freq / head_metric / aux_loss are.  Registered
        # non-persistent they were absent from the checkpoint, so
        # ``load_state_dict`` could neither restore them nor complain -- and any
        # consumer that rebuilds from a checkpoint without re-reading
        # norm_stats.json (``ablation/rank_audit.py``, ``ablation/profile.py``)
        # silently fed raw ``log pT`` (~+5) to an embed trained on ~0-centred
        # inputs.  The gate travels with the affine for the same reason.
        self.register_buffer("kin_scale", torch.ones(NUM_KINEMATIC_FEATURES))
        self.register_buffer("kin_offset", torch.zeros(NUM_KINEMATIC_FEATURES))
        self.register_buffer(
            "kin_norm_enabled", torch.zeros((), dtype=torch.bool)
        )

    def set_kinematic_norm(self, scale, offset) -> None:
        """Apply an affine transform to local kinematics after canonicalization.

        Lab-frame ``NormStats`` already ran in the collate, but channels 0-5
        are overwritten here.  Without this, the token embed sees raw
        ``log pT`` next to z-scored displacement/PID.  ``scale``/``offset``
        should come from :meth:`ablation.data.NormStats.lloca_kinematic_affine`.
        """
        scale_t = torch.as_tensor(scale, dtype=self.kin_scale.dtype, device=self.kin_scale.device)
        offset_t = torch.as_tensor(
            offset, dtype=self.kin_offset.dtype, device=self.kin_offset.device
        )
        if scale_t.shape != (NUM_KINEMATIC_FEATURES,) or offset_t.shape != (
            NUM_KINEMATIC_FEATURES,
        ):
            raise ValueError(
                f"kinematic affine must have shape ({NUM_KINEMATIC_FEATURES},); "
                f"got scale={tuple(scale_t.shape)}, offset={tuple(offset_t.shape)}"
            )
        self.kin_scale.copy_(scale_t)
        self.kin_offset.copy_(offset_t)
        self.kin_norm_enabled.fill_(True)

    # ``kin_scale`` / ``kin_offset`` / ``kin_norm_enabled`` became persistent
    # after the first checkpoints were written.  Supply the defaults for an older
    # state_dict so it still loads under ``strict=True`` instead of reporting
    # three missing keys: a checkpoint predating them had no kinematic affine
    # installed, which is exactly what the defaults encode.  (``nn.Module``
    # shallow-copies ``state_dict`` before dispatching here, so this does not
    # mutate the caller's dict.)
    _KIN_NORM_BUFFERS = ("kin_scale", "kin_offset", "kin_norm_enabled")

    def _load_from_state_dict(self, state_dict, prefix, *args, **kwargs):
        for name in self._KIN_NORM_BUFFERS:
            key = prefix + name
            if key not in state_dict:
                state_dict[key] = getattr(self, name).detach().clone()
        return super()._load_from_state_dict(state_dict, prefix, *args, **kwargs)

    @property
    def lloca_blocks(self):
        """The LLoCa attention blocks, in encoder order."""
        return tuple(self._lloca_blocks)

    def _apply_weaver_rescale(self) -> None:
        """Replicate weaver's ``fix_init_weight`` on the LLoCa blocks.

        Layer ``i`` (0-based) has its attention output projection and FFN second
        linear divided by ``sqrt(2 * (i + 1))``, matching stock weaver.
        """
        with torch.no_grad():
            for layer_id, block in enumerate(self._lloca_blocks):
                factor = math.sqrt(2.0 * (layer_id + 1))
                block.out_proj.weight.data.div_(factor)
                block.feedforward.linear2.weight.data.div_(factor)

    def forward(
        self,
        x: Tensor,
        v: Optional[Tensor] = None,
        mask: Optional[Tensor] = None,
        uu: Optional[Tensor] = None,
        uu_idx: Optional[Tensor] = None,
    ) -> Tensor:
        """
        Parameters
        ----------
        x : Tensor
            ``(B, input_dim, P)`` token features (channel-first, as the loader
            emits them).
        v : Tensor
            ``(B, 4, P)`` **raw** four-vectors in weaver order
            ``[px, py, pz, E]``.  Required — LLoCa cannot build frames without
            four-momenta.
        mask : Tensor or None
            ``(B, 1, P)``, 1 for real particles.
        uu, uu_idx
            Optional extra pair features, forwarded to weaver.

        Returns
        -------
        Tensor
            ``(B, num_classes)`` logits (or probabilities if
            ``for_inference``).
        """
        if v is None:
            raise ValueError(
                "LLoCaParT requires four-vectors `v` to predict local frames"
            )

        # 1. Trim first: the trimmer permutes and truncates, and frames are
        #    indexed by token position.
        x, v, mask, uu = self.trimmer(x, v, mask, uu)
        mask_bool = mask.squeeze(1).bool()  # (B, P)

        # 2. Predict one local frame per particle.
        p = to_time_first(v.transpose(1, 2))  # (B, P, 4) time-first
        scalars = x[:, self._frames_scalar_slice].transpose(1, 2)  # (B, P, S)
        frames = self.frames_net(p, scalars, mask_bool)  # (B, P, 4, 4) float64
        frames_inv = invert_frames(frames)

        # 3. Canonicalize the kinematic features into the local frames.
        x = self._canonicalize(x, p, frames, mask_bool)

        # 4. Hand the frames to every encoder block, then run stock weaver.
        frames_c = frames.to(x.dtype)
        frames_inv_c = frames_inv.to(x.dtype)
        for block in self._lloca_blocks:
            block.set_frames(frames_c, frames_inv_c)

        return self.part(x, v=v, mask=mask, uu=uu, uu_idx=uu_idx)

    def _canonicalize(
        self, x: Tensor, p: Tensor, frames: Tensor, mask_bool: Tensor
    ) -> Tensor:
        """Replace channels ``0:6`` of ``x`` with local-frame kinematics."""
        p64 = p.to(frames.dtype)

        # Jet four-momentum: sum over real particles only.
        jet = (p64 * mask_bool.unsqueeze(-1)).sum(dim=1, keepdim=True)  # (B,1,4)

        # L_i p_i  and  L_i p_jet
        p_local = torch.matmul(frames, p64.unsqueeze(-1)).squeeze(-1)
        jet_local = torch.matmul(frames, jet.expand_as(p64).unsqueeze(-1)).squeeze(-1)

        local = local_kinematic_features(p_local, jet_local).to(x.dtype)
        local = local.transpose(1, 2)  # (B, 6, P)
        if bool(self.kin_norm_enabled):
            scale = self.kin_scale.to(dtype=local.dtype)
            offset = self.kin_offset.to(dtype=local.dtype)
            local = local * scale[None, :, None] + offset[None, :, None]
        local = local * mask_bool.unsqueeze(1).to(local.dtype)
        return torch.cat([local, x[:, NUM_KINEMATIC_FEATURES:]], dim=1)


class _FrameAwareAdapter(nn.Module):
    """Adapt :class:`LLoCaAttentionBlock` to weaver's ``Block`` call signature.

    Weaver calls encoder blocks as
    ``block(x, x_cls=None, padding_mask=..., attn_mask=...)``.  LLoCa blocks are
    encoder-only: class attention stays stock weaver, so a non-``None``
    ``x_cls`` is a wiring bug and raises rather than being silently mishandled.
    """

    def __init__(self, block: LLoCaAttentionBlock):
        super().__init__()
        self.block = block

    def set_frames(self, frames: Tensor, frames_inv: Tensor) -> None:
        self.block.set_frames(frames, frames_inv)

    def forward(
        self,
        x: Tensor,
        x_cls: Optional[Tensor] = None,
        padding_mask: Optional[Tensor] = None,
        attn_mask: Optional[Tensor] = None,
    ) -> Tensor:
        if x_cls is not None:
            raise RuntimeError(
                "LLoCa blocks are encoder-stack only; "
                "cls_blocks must remain standard weaver Blocks"
            )
        return self.block(x, padding_mask, attn_mask)
