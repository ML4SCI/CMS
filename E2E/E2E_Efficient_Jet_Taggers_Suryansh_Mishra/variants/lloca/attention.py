"""LLoCa attention block — tensorial message passing between local frames.

Implements Eq. (11) of arXiv:2505.20280 (Lorentz Local Canonicalization):

    f_i^new = sum_j softmax_j( <q_i, rho(L_i L_j^-1) k_j> / sqrt(d) )
              * rho(L_i L_j^-1) v_j

Structurally this block is
:class:`~variants.blocks.attention_block.SoftmaxAttentionBlock` with exactly one
delta: queries/keys/values carry a Lorentz **representation** on their head
dimension, so they are transported between the sender's and receiver's local
frames before being combined, and the query/key contraction uses the Minkowski
product instead of the Euclidean one.

Two identities make this cheap:

* **Eq. (12)** — because the Minkowski product is Lorentz-invariant,
  ``<q_i, rho(L_i L_j^-1) k_j> = <rho(L_i^-1) q_i, rho(L_j^-1) k_j>``.  Mapping
  q/k into the *global* frame turns the pairwise transform into a plain
  ``O(N)`` operation followed by an ordinary dot product, so the ``(N, N)``
  score matrix is a single matmul.
* ``rho(L_i L_j^-1) = rho(L_i) rho(L_j^-1)`` — so values are mapped to the
  global frame, aggregated there, and the result mapped back with
  ``rho(L_i)``.  Again ``O(N)`` frame transforms, never ``O(N^2)``.

The Minkowski product is evaluated as a Euclidean dot product on
metric-folded queries (spatial components of the vector channels negated), so
the score computation stays a single ``matmul`` and could be swapped for a
fused SDPA kernel.

Representation layout
---------------------
Each head's ``head_dim`` channels are split into ``num_scalars`` scalar
channels followed by ``num_vectors`` four-vector channels::

    [ s_0 ... s_{S-1} | v_0^mu (4) | v_1^mu (4) | ... ]

with ``head_dim == num_scalars + 4 * num_vectors``.  The paper's default for a
16-dimensional head is eight scalars plus two four-vectors, which is what this
module picks automatically.  Table 2 of the paper shows any tensorial mix
clearly beats scalar-only message passing.

Note that only q/k/v carry the representation.  The block's hidden features are
Lorentz-*invariant* local features, so ``out_proj`` and the FFN are
unconstrained — the representation only has to be respected where messages
cross between frames.
"""

from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from ..blocks.feedforward import Feedforward
from .frames import MINKOWSKI_SIGNATURE

__all__ = ["LLoCaAttentionBlock", "default_representation"]


def default_representation(head_dim: int) -> Tuple[int, int]:
    """Pick the paper's default "equal mix" split for a head of ``head_dim``.

    Half the channels become scalars and half become four-vectors, i.e.
    ``num_vectors = head_dim // 8``.  For ``head_dim = 16`` this yields
    ``(8 scalars, 2 four-vectors)``.

    Falls back to a single four-vector when ``head_dim`` is too small to split
    evenly, and to scalars-only when ``head_dim < 4``.

    Returns
    -------
    tuple of int
        ``(num_scalars, num_vectors)``.
    """
    if head_dim < 4:
        return head_dim, 0
    num_vectors = max(1, head_dim // 8)
    while num_vectors > 0 and 4 * num_vectors > head_dim:
        num_vectors -= 1
    return head_dim - 4 * num_vectors, num_vectors


class LLoCaAttentionBlock(nn.Module):
    """Transformer block with LLoCa tensorial message passing.

    Drop-in replacement for the other variant blocks: same
    ``forward(x, padding_mask, U)`` signature.  Unlike them it additionally
    requires the per-particle local frames, which must be supplied via
    :meth:`set_frames` before each forward pass — :class:`LLoCaParT` does this
    once per step for every block.

    Parameters
    ----------
    embed_dim : int
        Model width.
    num_heads : int
        Number of attention heads.  ``embed_dim`` must be divisible by it.
    dropout : float
        Dropout rate.
    expansion_factor : int
        FFN expansion factor.
    num_vector_channels : int or None
        Number of four-vector channels per head.  ``None`` selects
        :func:`default_representation`.  ``0`` degrades to scalar-only
        messages, which the paper shows is markedly weaker (Table 2) — useful
        as an ablation of the tensorial messages themselves.

    Raises
    ------
    ValueError
        If the requested representation does not fit in ``head_dim``.
    """

    def __init__(
        self,
        embed_dim: int = 128,
        num_heads: int = 8,
        dropout: float = 0.1,
        expansion_factor: int = 4,
        num_vector_channels: Optional[int] = None,
    ):
        super().__init__()
        if embed_dim % num_heads != 0:
            raise ValueError(
                f"embed_dim ({embed_dim}) must be divisible by num_heads ({num_heads})"
            )

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        if num_vector_channels is None:
            num_scalars, num_vectors = default_representation(self.head_dim)
        else:
            num_vectors = int(num_vector_channels)
            num_scalars = self.head_dim - 4 * num_vectors
        if num_vectors < 0 or num_scalars < 0:
            raise ValueError(
                f"representation does not fit: head_dim={self.head_dim}, "
                f"num_vectors={num_vectors} needs {4 * num_vectors} channels"
            )
        self.num_scalars = num_scalars
        self.num_vectors = num_vectors

        self.layernorm1 = nn.LayerNorm(embed_dim)
        self.layernorm2 = nn.LayerNorm(embed_dim)

        # Kept byte-identical to SoftmaxAttentionBlock so cross-variant
        # comparisons remain single-delta.
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

        self.dropout = nn.Dropout(dropout)
        self.feedforward = Feedforward(
            embed_dim=embed_dim,
            expansion_factor=expansion_factor,
            dropout=dropout,
        )

        # Per-head metric used to fold the Minkowski signature into the query
        # so the score matrix is a plain matmul: ones on scalar channels,
        # (+1, -1, -1, -1) repeated over the four-vector channels.
        metric = torch.ones(self.head_dim)
        if num_vectors > 0:
            signature = torch.tensor(MINKOWSKI_SIGNATURE)
            metric[num_scalars:] = signature.repeat(num_vectors)
        self.register_buffer("head_metric", metric, persistent=False)

        self._frames: Optional[Tensor] = None
        self._frames_inv: Optional[Tensor] = None

    # -- frame plumbing ----------------------------------------------------

    def set_frames(self, frames: Tensor, frames_inv: Tensor) -> None:
        """Stash the local frames used by the next :meth:`forward`.

        Parameters
        ----------
        frames : Tensor
            ``(B, N, 4, 4)`` local frames ``L_i``.
        frames_inv : Tensor
            ``(B, N, 4, 4)`` inverses ``L_i^-1``.

        Notes
        -----
        Weaver calls its encoder blocks with a fixed signature that has no slot
        for extra tensors, so the frames are handed over out-of-band rather
        than as forward arguments.  They are cleared on use, which turns a
        missing :meth:`set_frames` call into an immediate error instead of a
        silent reuse of stale frames.
        """
        self._frames = frames
        self._frames_inv = frames_inv

    def _take_frames(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        if self._frames is None or self._frames_inv is None:
            raise RuntimeError(
                "LLoCaAttentionBlock.forward() called without frames; "
                "call set_frames(L, L_inv) before every forward pass "
                "(LLoCaParT does this for all blocks)"
            )
        frames, frames_inv = self._frames, self._frames_inv
        if frames.shape[1] != x.shape[1]:
            raise RuntimeError(
                f"frame/token length mismatch: frames have {frames.shape[1]} "
                f"particles but x has {x.shape[1]} tokens — frames must be "
                "built after sequence trimming"
            )
        return frames.to(x.dtype), frames_inv.to(x.dtype)

    # -- representation transforms ----------------------------------------

    def _transform(self, t: Tensor, matrices: Tensor) -> Tensor:
        """Apply ``rho(M)`` to the head features ``t``.

        Scalar channels pass through; every four-vector channel is contracted
        with ``M``.

        Parameters
        ----------
        t : Tensor
            ``(B, H, N, head_dim)``.
        matrices : Tensor
            ``(B, N, 4, 4)``.
        """
        if self.num_vectors == 0:
            return t
        scalars = t[..., : self.num_scalars]
        vectors = t[..., self.num_scalars :].unflatten(
            -1, (self.num_vectors, 4)
        )  # (B, H, N, V, 4)
        vectors = torch.einsum("bnij,bhnvj->bhnvi", matrices, vectors)
        return torch.cat([scalars, vectors.flatten(-2, -1)], dim=-1)

    # -- forward -----------------------------------------------------------

    def forward(
        self, x: Tensor, padding_mask: Optional[Tensor], U: Optional[Tensor] = None
    ) -> Tensor:
        """
        Parameters
        ----------
        x : Tensor
            ``(B, N, embed_dim)`` local (Lorentz-invariant) token features.
        padding_mask : Tensor or None
            ``(B, N)``, ``True`` marks padded positions.
        U : Tensor or None
            ParT pair-embedding bias, ``(B, num_heads, N, N)`` or the legacy
            ``(B * num_heads, N, N)`` layout.

        Returns
        -------
        Tensor
            ``(B, N, embed_dim)``.
        """
        frames, frames_inv = self._take_frames(x)
        self._frames = self._frames_inv = None  # consume; see set_frames()

        residual = x
        x_norm = self.layernorm1(x)
        B, N, C = x_norm.shape

        Q = self.q_proj(x_norm)
        K = self.k_proj(x_norm)
        V = self.v_proj(x_norm)

        # (B, N, H, d) -> (B, H, N, d)
        Q = Q.view(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        K = K.view(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        V = V.view(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)

        # --- DELTA vs SoftmaxAttentionBlock: move q/k/v to the global frame.
        # Eq. (12) lets the pairwise L_i L_j^-1 transform be replaced by one
        # per-token L^-1, after which scores are an ordinary dot product.
        Q = self._transform(Q, frames_inv)
        K = self._transform(K, frames_inv)
        V = self._transform(V, frames_inv)

        # Minkowski product via metric-folded query.
        scale = 1.0 / math.sqrt(self.head_dim)
        attn_scores = torch.matmul(Q * self.head_metric.to(Q.dtype), K.transpose(-2, -1))
        attn_scores = attn_scores * scale  # (B, H, N, N)

        if U is not None:
            attn_scores = attn_scores + U.view(B, self.num_heads, N, N)

        if padding_mask is not None:
            mask = padding_mask.bool().unsqueeze(1).unsqueeze(2)  # (B, 1, 1, N)
            attn_scores = attn_scores.masked_fill(mask, -1e9)

        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # Aggregate in the global frame, then map the result into each
        # receiver's own frame: rho(L_i) sum_j A_ij rho(L_j^-1) v_j.
        out = torch.matmul(attn_weights, V)  # (B, H, N, d)
        out = self._transform(out, frames)

        out = out.permute(0, 2, 1, 3).contiguous().view(B, N, C)
        out = self.out_proj(out)

        # Post-attention path identical to SoftmaxAttentionBlock.
        x_out = self.layernorm2(out)
        x_out = self.dropout(x_out)
        x_out = x_out + residual
        x_out = self.feedforward(x_out)

        return x_out
