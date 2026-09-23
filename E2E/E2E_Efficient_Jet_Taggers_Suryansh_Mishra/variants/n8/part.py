"""N8-ParT — Particle Transformer with a factorized pair bias.

Wraps a weaver ``ParticleTransformer`` so that

1. sequence trimming happens *here* (the trimmer permutes tokens, and the
   extra Q/K channels are built from post-trim four-momenta),
2. encoder blocks are :class:`~variants.n8.attention.FactorizedAttentionBlock`,
   which add rank-4 Minkowski and tensor-product rotary logits instead of a
   dense ``PairEmbed``,
3. class attention stays stock weaver.

``pair_embed_dims`` is forced off: N8's whole point is to not materialize
``U``.  A leftover dense bias would make the K6 screen measure the pair
pipeline against itself.
"""

from __future__ import annotations

import math
from typing import Optional

import torch
from torch import Tensor, nn

from weaver.nn.model.ParticleTransformer import ParticleTransformer, SequenceTrimmer

from .attention import FactorizedAttentionBlock

__all__ = ["N8ParT"]


class N8ParT(nn.Module):
    """Weaver ``ParticleTransformer`` with N8 factorized extras in the encoder.

    Parameters
    ----------
    input_dim, num_classes, embed_dims, num_heads, num_layers, num_cls_layers
        Forwarded to weaver ``ParticleTransformer``.
    pair_embed_dims
        Ignored.  Always ``None`` on the inner transformer.
    dropout, expansion_factor
        Encoder-block hyperparameters.
    minkowski, rotary, rotary_pairs, degree
        Extra-channel flags; see :class:`FactorizedAttentionBlock`.
    trim : bool
        Enable weaver's ``SequenceTrimmer``.  Owned by this wrapper because
        it permutes tokens and extras must be built on the post-trim sequence.
    **weaver_kwargs
        Additional weaver ``ParticleTransformer`` keyword arguments.
    """

    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        *,
        embed_dims=(128, 512, 128),
        pair_embed_dims=None,
        num_heads: int = 8,
        num_layers: int = 8,
        num_cls_layers: int = 2,
        dropout: float = 0.1,
        expansion_factor: int = 4,
        minkowski: bool = True,
        rotary: bool = True,
        rotary_pairs: int = 4,
        degree: int = 1,
        trim: bool = True,
        for_inference: bool = False,
        **weaver_kwargs,
    ):
        super().__init__()
        del pair_embed_dims  # N8 never builds PairEmbed
        self.for_inference = for_inference
        self.minkowski = bool(minkowski)
        self.rotary = bool(rotary)
        self.rotary_pairs = int(rotary_pairs)
        self.degree = int(degree)

        self.trimmer = SequenceTrimmer(enabled=trim and not for_inference)

        weaver_kwargs.pop("fix_init", None)
        weaver_kwargs.pop("minkowski", None)
        weaver_kwargs.pop("rotary", None)
        weaver_kwargs.pop("rotary_pairs", None)
        weaver_kwargs.pop("degree", None)
        self.part = ParticleTransformer(
            input_dim=input_dim,
            num_classes=num_classes,
            embed_dims=embed_dims,
            pair_embed_dims=None,
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
            self.part.blocks[i] = _MomentaAwareAdapter(
                FactorizedAttentionBlock(
                    embed_dim=embed_dim,
                    num_heads=num_heads,
                    dropout=dropout,
                    expansion_factor=expansion_factor,
                    minkowski=self.minkowski,
                    rotary=self.rotary,
                    rotary_pairs=self.rotary_pairs,
                    degree=self.degree,
                )
            )
        self._n8_blocks = [adapter.block for adapter in self.part.blocks]
        self._apply_weaver_rescale()

    @property
    def pair_embed(self):
        """Always ``None`` — N8 has no dense pair bias."""
        return self.part.pair_embed

    @property
    def n8_blocks(self):
        return tuple(self._n8_blocks)

    def _apply_weaver_rescale(self) -> None:
        with torch.no_grad():
            for layer_id, block in enumerate(self._n8_blocks):
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
            ``(B, input_dim, P)`` token features.
        v : Tensor
            ``(B, 4, P)`` raw four-vectors ``[px, py, pz, E]``.  Required.
        mask : Tensor or None
            ``(B, 1, P)``, 1 for real particles.
        """
        if v is None:
            raise ValueError(
                "N8ParT requires four-vectors `v` to build Minkowski/rotary extras"
            )

        x, v, mask, uu = self.trimmer(x, v, mask, uu)
        for block in self._n8_blocks:
            block.set_momenta(v, mask)

        return self.part(x, v=v, mask=mask, uu=uu, uu_idx=uu_idx)


class _MomentaAwareAdapter(nn.Module):
    """Adapt :class:`FactorizedAttentionBlock` to weaver's ``Block`` signature.

    Ignores ``attn_mask``: N8's extras replace the dense pair bias, and a
    leftover ``PairEmbed`` tensor must not be added on top.
    """

    def __init__(self, block: FactorizedAttentionBlock):
        super().__init__()
        self.block = block

    def set_momenta(self, v: Tensor, mask: Optional[Tensor] = None) -> None:
        self.block.set_momenta(v, mask)

    def forward(
        self,
        x: Tensor,
        x_cls: Optional[Tensor] = None,
        padding_mask: Optional[Tensor] = None,
        attn_mask: Optional[Tensor] = None,
    ) -> Tensor:
        if x_cls is not None:
            raise RuntimeError(
                "N8 blocks are encoder-stack only; "
                "cls_blocks must remain standard weaver Blocks"
            )
        return self.block(x, padding_mask, attn_mask)
