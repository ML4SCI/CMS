"""Jet classifier built on the OFFICIAL ``lgatr`` package (heidelberg-hepml/lgatr).

This module is the single source of truth for the LGATr ablation arm. It is
consumed by the ``lgatr_kernels/benchmarks/bench_e2e.py`` worker. It contains no
borrowed code — every LGATr building block (``LGATr``, ``embed_vector``,
``extract_scalar``) comes from the official package.

Architecture (design DD9)::

    embed_vector(4-vectors) -> LGATr(mv/s channels) -> extract_scalar
        -> masked mean pool over particles -> linear head

Input contract — a ``dataloader.ragged_loader`` batch, called with the same
convention as every other arm, ``model(x, v=v, mask=mask)``:

- ``x``: ``(B, 16, P)`` particle features — **accepted and ignored**; this arm
  is four-vector-only (``in_s_channels=None``). See
  :meth:`LGATrJetClassifier.forward` for why, and what it costs.
- ``v``: ``(B, 4, P)`` raw four-vectors ordered ``[px, py, pz, E]``
- ``mask``: ``(B, 1, P)`` — ``1.0`` for real particles, ``0.0`` for padding

Component-order note: ``lgatr.interface.embed_vector`` writes the four-vector
into multivector slots 1:5, whose Minkowski inner-product signs are
``(+, -, -, -)`` — i.e. the package expects **time-first** ``(E, px, py, pz)``.
The loader contract is ``[px, py, pz, E]``, so ``forward`` reorders the
components before embedding.

References
----------
- Spinner et al., "Lorentz-Equivariant Geometric Algebra Transformers for
  High-Energy Physics", NeurIPS 2024, arXiv:2405.14806.
- Brehmer et al., "A Lorentz-Equivariant Transformer for All of the LHC",
  arXiv:2411.00446.
- Official package: https://github.com/heidelberg-hepml/lgatr
"""

from typing import Optional

import torch
from torch import Tensor, nn

from lgatr import LGATr, embed_vector, extract_scalar

__all__ = ["LGATrJetClassifier"]


class LGATrJetClassifier(nn.Module):
    """Jet tagger on the OFFICIAL lgatr package (heidelberg-hepml/lgatr).

    embed_vector(4-vectors) -> LGATr(mv/s channels) -> extract_scalar
    -> masked mean pool -> linear head. No borrowed code.

    Parameters
    ----------
    num_classes : int
        Number of output classes (logit dimension).
    mv_channels : int
        Hidden (and output) multivector channels of the LGATr backbone.
        ``extract_scalar`` on the output multivectors yields per-particle
        features of this dimension.
    s_channels : int
        Hidden scalar channels of the LGATr backbone (auxiliary
        non-equivariant capacity; no scalar inputs/outputs are used).
    num_blocks : int
        Number of LGATr transformer blocks.
    num_heads : int
        Attention heads per block.
    """

    def __init__(
        self,
        num_classes: int,
        mv_channels: int = 8,
        s_channels: int = 16,
        num_blocks: int = 8,
        num_heads: int = 8,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.mv_channels = mv_channels

        self.lgatr = LGATr(
            num_blocks=num_blocks,
            in_mv_channels=1,
            out_mv_channels=mv_channels,
            hidden_mv_channels=mv_channels,
            in_s_channels=None,
            out_s_channels=None,
            hidden_s_channels=s_channels,
            attention={"num_heads": num_heads},
            mlp={},
        )
        self.head = nn.Linear(mv_channels, num_classes)

    def forward(
        self,
        x: Optional[Tensor] = None,
        v: Optional[Tensor] = None,
        mask: Optional[Tensor] = None,
    ) -> Tensor:
        """Classify jets from raw particle four-vectors.

        The signature matches the call convention every other ablation arm
        uses, ``model(x, v=v, mask=mask)``, so this model is a drop-in for the
        training harness and consumes ``ragged_loader`` batches unchanged.

        Parameters
        ----------
        x : Tensor or None
            ``(B, 16, P)`` particle features from the loader. **Deliberately
            unused.** L-GATr is a four-vector-only equivariant model: it is
            constructed with ``in_s_channels=None``, so the scalar channels of
            ``x`` (impact parameters, charge, PID one-hots) have no input path.
            Accepted purely for call-signature compatibility.

            This is a real modelling asymmetry against the ParT arms, which
            see all 16 features — it is not a bug, but it must be stated when
            comparing numbers. Feeding the genuinely Lorentz-scalar subset of
            ``x`` (channels 6-15) through ``scalars=`` would close it, at the
            cost of no longer being a pure four-vector baseline.
        v : Tensor
            ``(B, 4, P)`` raw four-vectors ``[px, py, pz, E]`` (loader
            contract; no normalization applied). Required.
        mask : Tensor
            ``(B, 1, P)`` padding mask — nonzero for real particles,
            ``0`` for padded slots. Required.

        Returns
        -------
        Tensor
            ``(B, num_classes)`` classification logits.

        Raises
        ------
        ValueError
            If ``v`` or ``mask`` is ``None``. Both are mandatory; they are
            keyword-defaulted only so ``x`` can occupy the first position.
        """
        if v is None or mask is None:
            raise ValueError(
                "LGATrJetClassifier requires both v (B,4,P) and mask (B,1,P); "
                f"got v={type(v).__name__}, mask={type(mask).__name__}. Call it "
                "as model(x, v=v, mask=mask) — the same convention as every "
                "other arm — or model(None, v=v, mask=mask) when there are no "
                "particle features to pass."
            )
        # (B, 4, P) [px, py, pz, E] -> (B, P, 4) [E, px, py, pz]
        # (lgatr's embed_vector expects time-first components; see module
        # docstring for the metric-signature check behind this reorder)
        v = v.permute(0, 2, 1)  # (B, P, 4)
        v = v[..., [3, 0, 1, 2]]  # [E, px, py, pz]

        # Embed into multivectors: (B, P, 16) -> add channel dim (B, P, 1, 16)
        mv = embed_vector(v).unsqueeze(-2)

        # Exclude padded particles from attention: torch SDPA bool mask,
        # True = attend. Shape (B, 1, 1, P) broadcasts over (heads, queries).
        attn_mask = mask.bool().unsqueeze(1)

        out_mv, _ = self.lgatr(mv, scalars=None, attn_mask=attn_mask)

        # (B, P, mv_channels, 1) -> (B, P, mv_channels)
        feats = extract_scalar(out_mv).squeeze(-1)

        # Masked mean pool over particles: (B, P, mv_channels) -> (B, mv_channels)
        m = mask.to(feats.dtype).permute(0, 2, 1)  # (B, P, 1)
        pooled = (feats * m).sum(dim=1) / m.sum(dim=1).clamp(min=1.0)

        return self.head(pooled)
