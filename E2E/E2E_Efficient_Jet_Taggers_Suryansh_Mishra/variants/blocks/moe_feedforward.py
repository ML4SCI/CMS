"""Mixture-of-Experts Feedforward — configurable drop-in for Feedforward.

Provenance: user-authored; lifted from the project's earlier hybrid-transformer
tree (``src/models/moe_feedforward.py``) so the variants package is
self-contained, then generalized so the MoE arm can be matched to the dense
baseline on *either* parameters or FLOPs.

Sizing the experts
------------------
Let ``H = embed_dim * expansion_factor`` be the hidden width of the dense
baseline ``Feedforward`` (512 for ParT's 128-dim, 4x FFN), ``E`` the number of
routed experts, ``S`` the number of always-on shared experts and ``k`` the
number of routed experts activated per token.  Each expert is a two-layer FFN
of hidden width ``h``, so per token the MoE spends ``(k + S) * h`` of "hidden
width" and stores ``(E + S) * h``.  :class:`MoEFeedforward` therefore offers
three ways to pick ``h`` (``hidden_mode``):

=================  ====================  ==================  =================
``hidden_mode``    ``h``                 active FLOPs        FFN params
=================  ====================  ==================  =================
``"param_matched"``  ``H / (E + S)``     ``(k+S)/(E+S)`` x   ``1.0`` x
``"flop_matched"``   ``H / (k + S)``     ``1.0`` x           ``(E+S)/(k+S)`` x
``"full"``           ``H``               ``(k + S)`` x       ``(E + S)`` x
=================  ====================  ==================  =================

``"full"`` makes every expert *exactly* the size of the original ParT FFN.
Combined with ``top_k=1`` and no shared expert it spends exactly the baseline's
per-token compute while holding ``E`` times the FFN parameters — the standard
"same FLOPs, more capacity" MoE trade.  ``"param_matched"`` is the opposite
corner: identical parameter count, but only ``k/E`` of the baseline's compute.
Use :meth:`MoEFeedforward.cost_summary` to print the realized ratios.

Gating
------
``gate_mode`` selects how the router weight is formed:

``"topk_softmax"``
    Softmax over the ``k`` selected logits only (Mixtral-style).  Requires
    ``k >= 2``: with ``k == 1`` a softmax over one element is identically 1, so
    the router would receive **no gradient** and never learn to route.
``"full_softmax"``
    Softmax over all ``E`` logits, then gather the selected entries
    (Switch-Transformer-style).  Keeps the router differentiable at ``k == 1``.

The default picks ``"topk_softmax"`` for ``k >= 2`` and ``"full_softmax"`` for
``k == 1``.

Load balancing
--------------
After each forward pass ``self.aux_loss`` holds the Switch load-balancing loss
``E * sum_i f_i p_i``, where ``f_i`` is the fraction of tokens routed to expert
``i`` and ``p_i`` the mean router probability for it.  Training must add
``moe_aux_alpha * collect_moe_aux_loss(model)`` to the task loss or the experts
collapse; ``0.01`` is a typical coefficient.  Padded tokens are excluded from
``f`` and ``p`` when a ``padding_mask`` is supplied.

References
----------
- Switch Transformers (Fedus et al., 2021), arXiv:2101.03961
- GShard (Lepikhin et al., 2020), arXiv:2006.16668
- Mixtral of Experts (Jiang et al., 2024), arXiv:2401.04088
- DeepSeekMoE shared experts (Dai et al., 2024), arXiv:2401.06066
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn.functional as F
from torch import Tensor, nn

__all__ = ["MoEFeedforward", "HIDDEN_MODES", "GATE_MODES"]

#: Accepted values for ``hidden_mode``.
HIDDEN_MODES = ("param_matched", "flop_matched", "full")

#: Accepted values for ``gate_mode``.
GATE_MODES = ("topk_softmax", "full_softmax")


class _ExpertFFN(nn.Module):
    """Single expert: two-layer FFN matching ``Feedforward``'s inner structure.

    Deliberately mirrors :class:`~variants.blocks.feedforward.Feedforward` (both
    LayerNorms, GELU, both Dropouts) minus the residual connection, which the
    parent adds once after combining experts.
    """

    def __init__(self, embed_dim: int, hidden_dim: int, dropout: float):
        super().__init__()
        self.layernorm1 = nn.LayerNorm(embed_dim)
        self.linear1 = nn.Linear(embed_dim, hidden_dim)
        self.act = nn.GELU()
        self.dropout1 = nn.Dropout(dropout)

        self.layernorm2 = nn.LayerNorm(hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, embed_dim)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x: Tensor) -> Tensor:
        x = self.layernorm1(x)
        x = self.linear1(x)
        x = self.act(x)
        x = self.dropout1(x)

        x = self.layernorm2(x)
        x = self.linear2(x)
        x = self.dropout2(x)
        return x


class MoEFeedforward(nn.Module):
    """Mixture-of-Experts feedforward, drop-in for ``Feedforward``.

    Parameters
    ----------
    embed_dim : int
        Input / output dimensionality.
    expansion_factor : int
        Expansion factor of the *dense baseline* FFN; the baseline hidden width
        is ``embed_dim * expansion_factor``.
    dropout : float
        Dropout rate inside each expert.
    num_experts : int
        Number of routed experts ``E``.
    top_k : int
        Routed experts activated per token, ``k``.  Must satisfy ``k <= E``.
    hidden_mode : str
        One of :data:`HIDDEN_MODES` — how each expert's hidden width is derived
        from the baseline.  Ignored when ``expert_hidden`` is given.
    expert_hidden : int or None
        Explicit per-expert hidden width, overriding ``hidden_mode``.
    num_shared_experts : int
        Always-on experts ``S`` applied to every token with weight 1
        (DeepSeekMoE-style).  Useful to keep a dense "backbone" path while the
        routed experts specialize.
    gate_mode : str or None
        One of :data:`GATE_MODES`; ``None`` auto-selects (see module docstring).

    Raises
    ------
    ValueError
        On an inconsistent configuration — in particular ``top_k > num_experts``,
        a non-positive derived ``expert_hidden``, or ``top_k == 1`` combined with
        ``gate_mode="topk_softmax"`` (which would leave the router untrained).
    """

    #: Duck-typed flag: host blocks pass ``padding_mask`` when this is present,
    #: without needing to import this class (avoids a circular import).
    accepts_padding_mask = True

    def __init__(
        self,
        embed_dim: int = 128,
        expansion_factor: int = 4,
        dropout: float = 0.1,
        num_experts: int = 4,
        top_k: int = 2,
        hidden_mode: str = "param_matched",
        expert_hidden: Optional[int] = None,
        num_shared_experts: int = 0,
        gate_mode: Optional[str] = None,
    ):
        super().__init__()
        if top_k > num_experts:
            raise ValueError(
                f"top_k ({top_k}) must be <= num_experts ({num_experts})"
            )
        if top_k < 1:
            raise ValueError(f"top_k must be >= 1, got {top_k}")
        if num_shared_experts < 0:
            raise ValueError(
                f"num_shared_experts must be >= 0, got {num_shared_experts}"
            )
        if hidden_mode not in HIDDEN_MODES:
            raise ValueError(
                f"unknown hidden_mode {hidden_mode!r}; expected one of "
                f"{list(HIDDEN_MODES)}"
            )

        if gate_mode is None:
            gate_mode = "topk_softmax" if top_k > 1 else "full_softmax"
        if gate_mode not in GATE_MODES:
            raise ValueError(
                f"unknown gate_mode {gate_mode!r}; expected one of {list(GATE_MODES)}"
            )
        if top_k == 1 and gate_mode == "topk_softmax":
            raise ValueError(
                "top_k=1 with gate_mode='topk_softmax' leaves the router "
                "without gradient (softmax over a single logit is always 1.0); "
                "use gate_mode='full_softmax' instead"
            )

        self.embed_dim = embed_dim
        self.num_experts = num_experts
        self.top_k = top_k
        self.num_shared_experts = num_shared_experts
        self.hidden_mode = hidden_mode
        self.gate_mode = gate_mode

        self.base_hidden = embed_dim * expansion_factor
        if expert_hidden is None:
            if hidden_mode == "param_matched":
                expert_hidden = self.base_hidden // (num_experts + num_shared_experts)
            elif hidden_mode == "flop_matched":
                expert_hidden = self.base_hidden // (top_k + num_shared_experts)
            else:  # "full"
                expert_hidden = self.base_hidden
        if expert_hidden <= 0:
            raise ValueError(
                f"derived expert_hidden is {expert_hidden} (must be > 0): "
                f"embed_dim={embed_dim}, expansion_factor={expansion_factor}, "
                f"num_experts={num_experts}, top_k={top_k}, "
                f"num_shared_experts={num_shared_experts}, "
                f"hidden_mode={hidden_mode!r}"
            )
        self.expert_hidden = int(expert_hidden)

        # Router: projects to one logit per routed expert.
        self.router = nn.Linear(embed_dim, num_experts, bias=False)

        self.experts = nn.ModuleList(
            _ExpertFFN(embed_dim, self.expert_hidden, dropout)
            for _ in range(num_experts)
        )
        self.shared_experts = nn.ModuleList(
            _ExpertFFN(embed_dim, self.expert_hidden, dropout)
            for _ in range(num_shared_experts)
        )

        # Non-persistent buffer so it follows .to(device) and stays out of
        # state_dict; overwritten with a graph-connected scalar every forward.
        self.register_buffer("aux_loss", torch.zeros(()), persistent=False)

    # -- cost accounting ---------------------------------------------------

    @property
    def all_experts(self) -> tuple:
        """Routed and shared experts together, for weight-init bookkeeping."""
        return (*self.experts, *self.shared_experts)

    @property
    def active_hidden(self) -> int:
        """Hidden width actually evaluated per token."""
        return (self.top_k + self.num_shared_experts) * self.expert_hidden

    @property
    def total_hidden(self) -> int:
        """Hidden width stored across all experts."""
        return (self.num_experts + self.num_shared_experts) * self.expert_hidden

    @property
    def flop_ratio(self) -> float:
        """Active FFN FLOPs relative to the dense baseline (1.0 == matched)."""
        return self.active_hidden / self.base_hidden

    @property
    def param_ratio(self) -> float:
        """Stored FFN parameters relative to the dense baseline, router aside."""
        return self.total_hidden / self.base_hidden

    def cost_summary(self) -> str:
        """One-line description of the realized capacity/compute trade."""
        return (
            f"MoE(E={self.num_experts}, k={self.top_k}, "
            f"shared={self.num_shared_experts}, h={self.expert_hidden}, "
            f"mode={self.hidden_mode}, gate={self.gate_mode}) -> "
            f"FLOPs {self.flop_ratio:.3f}x, params {self.param_ratio:.3f}x "
            f"of dense h={self.base_hidden}"
        )

    def extra_repr(self) -> str:
        return self.cost_summary()

    # -- forward -----------------------------------------------------------

    def _compute_aux_loss(
        self, router_logits: Tensor, topk_indices: Tensor, valid: Optional[Tensor]
    ) -> Tensor:
        """Switch load-balancing loss ``E * sum_i f_i p_i``.

        ``f`` (the realized routing fractions) is a hard count and carries no
        gradient; ``p`` (mean router probability) does, so minimizing the
        product pushes the router toward a uniform assignment.

        Parameters
        ----------
        router_logits : Tensor
            ``(T, E)`` flattened logits.
        topk_indices : Tensor
            ``(T, k)`` selected expert indices.
        valid : Tensor or None
            ``(T,)`` bool mask of real (non-padded) tokens.
        """
        if valid is not None:
            if not valid.any():
                return router_logits.sum() * 0.0
            router_logits = router_logits[valid]
            topk_indices = topk_indices[valid]

        with torch.no_grad():
            one_hot = F.one_hot(topk_indices, self.num_experts).sum(dim=1).float()
            f = one_hot.mean(dim=0)  # (E,)

        p = torch.softmax(router_logits, dim=-1).mean(dim=0)  # (E,)
        return self.num_experts * (f.to(p.dtype) * p).sum()

    def forward(self, x: Tensor, padding_mask: Optional[Tensor] = None) -> Tensor:
        """
        Parameters
        ----------
        x : Tensor
            ``(B, N, embed_dim)``.
        padding_mask : Tensor or None
            ``(B, N)``, ``True`` marks padded positions.  Only used to exclude
            padded tokens from the load-balancing statistics; supplying it stops
            padding from dominating the router's balance target on ragged jets.

        Returns
        -------
        Tensor
            ``(B, N, embed_dim)``, residual connection included — identical
            semantics to ``Feedforward.forward``.
        """
        residual = x
        batch, seq_len, channels = x.shape

        flat_x = x.reshape(-1, channels)  # (T, C)
        router_logits = self.router(flat_x)  # (T, E)

        _, topk_indices = torch.topk(router_logits, self.top_k, dim=-1)  # (T, k)

        if self.gate_mode == "full_softmax":
            gates = torch.softmax(router_logits, dim=-1).gather(-1, topk_indices)
        else:
            gates = torch.softmax(
                router_logits.gather(-1, topk_indices), dim=-1
            )  # (T, k)

        valid = None
        if padding_mask is not None:
            valid = ~padding_mask.reshape(-1).bool()
        self.aux_loss = self._compute_aux_loss(router_logits, topk_indices, valid)

        out = torch.zeros_like(flat_x)

        # One pass per expert over the tokens routed to it.  Summing the gate
        # over the k slots handles the (degenerate) case where topk returns the
        # same expert twice.
        for expert_id, expert in enumerate(self.experts):
            selected = topk_indices == expert_id  # (T, k)
            token_ids = selected.any(dim=-1).nonzero(as_tuple=True)[0]
            if token_ids.numel() == 0:
                continue
            weight = (gates * selected).sum(dim=-1)[token_ids].unsqueeze(-1)
            out = out.index_add(0, token_ids, weight * expert(flat_x[token_ids]))

        # Shared experts see every token, with unit weight.
        for expert in self.shared_experts:
            out = out + expert(flat_x)

        return out.view(batch, seq_len, channels) + residual
