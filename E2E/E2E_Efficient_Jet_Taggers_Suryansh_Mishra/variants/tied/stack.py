"""Tied encoder stacks: schedules, per-depth modulation, and MoR routing.

Weaver's ``ParticleTransformer.forward`` iterates its own ``self.blocks``::

    for block in self.blocks:
        x = block(x, x_cls=None, padding_mask=padding_mask, attn_mask=attn_mask)

so tying is implemented by handing it a ``ModuleList`` of the right *length* whose entries share
parameters — no weaver source is modified, which is this package's standing constraint (DD6).
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Optional, Sequence

import torch
from torch import Tensor, nn

#: Tolerance for the per-jet capacity ceil. Absorbs float32 error in ``capacity * n_valid``
#: without changing any genuinely fractional case: it only bites within 1e-6 of an integer,
#: where rounding error is the cause rather than the intent.
_CEIL_TOL = 1e-6

#: Every schedule :class:`TiedSchedule` accepts. ``middle-cycle`` and ``head-unique`` keep the
#: entry (and, for middle-cycle, the exit) block unique -- the configurations the prior art and this
#: repo's own R0 data respectively point at.
_STRATEGIES = ("cycle", "sequence", "middle-cycle", "head-unique")

__all__ = [
    "TiedSchedule",
    "DepthModulation",
    "ModulatedBlock",
    "MoRRouter",
    "MoRBlock",
    "build_tied_blocks",
    "tied_cost_report",
    "shared_block_groups",
    "assert_shared_blocks_consistent",
]


# --------------------------------------------------------------------------------------------
# schedules
# --------------------------------------------------------------------------------------------
@dataclass(frozen=True)
class TiedSchedule:
    """Which unique block runs at each depth.

    ``depth`` forward applications, drawn from ``num_unique`` distinct blocks. The two strategies
    are the ones Mixture-of-Recursions distinguishes, and they are **not** equivalent:

    ``cycle``
        ``[0, 1, 2, 3, 0, 1, 2, 3]`` for depth 8, 4 unique — the group is *looped*. Each block
        sees its input twice at different stages of refinement, which is the setting where the
        "same function applied repeatedly" reading makes sense.
    ``sequence``
        ``[0, 0, 1, 1, 2, 2, 3, 3]`` — each block is applied consecutively before moving on. This
        stays closer to stock ParT's depthwise specialisation and is the gentler perturbation.
    ``middle-cycle``
        ``[0, 1, 2, 1, 2, 1, 2, 3]`` for depth 8, 4 unique — **entry and exit blocks stay unique**
        and only the middle is shared. Requires ``num_unique >= 3``. This is the configuration
        **four independent papers converge on**: Mixture-of-Recursions (arXiv:2507.10524) compares
        Cycle / Sequence / Middle-Cycle / Middle-Sequence and finds Middle-Cycle "consistently
        achieves the lowest validation loss". The mechanism is that the first block does
        embedding-like work and the last does readout-like work, neither of which is the iterative
        refinement the middle performs — so tying them is the expensive part of tying.
    ``head-unique``
        ``[0, 1, 1, 1, 1, 1, 1, 1]`` for depth 8, 2 unique — only block 0 stays unique. Requires
        ``num_unique >= 2``. Included because **this repo's own R0 data points at it**: the trained
        ffn block-similarity matrix is monotone in depth, with the deep blocks most similar to each
        other and block 0 the outlier, so "block 0 unique, deep blocks tied" is the highest-prior
        partial schedule from the measurement rather than from the literature.

    At ``num_unique == depth`` both reduce to stock ParT, which is the identity check the tests
    assert. At ``num_unique == 1`` both reduce to full ALBERT-style tying.
    """

    depth: int
    num_unique: int
    strategy: str = "cycle"

    def __post_init__(self) -> None:
        if self.depth < 1:
            raise ValueError(f"depth must be >= 1, got {self.depth}")
        if not 1 <= self.num_unique <= self.depth:
            raise ValueError(
                f"num_unique must be in [1, depth={self.depth}], got {self.num_unique}"
            )
        if self.strategy not in _STRATEGIES:
            raise ValueError(
                f"strategy must be one of {sorted(_STRATEGIES)}, got {self.strategy!r}"
            )
        if self.strategy == "middle-cycle" and self.num_unique < 3:
            raise ValueError(
                "middle-cycle keeps the FIRST and LAST blocks unique, so it needs num_unique >= 3 "
                f"(got {self.num_unique}); use 'head-unique' for a single unique entry block"
            )
        if self.strategy == "head-unique" and self.num_unique < 2:
            raise ValueError(
                f"head-unique needs num_unique >= 2 (got {self.num_unique})"
            )
        if self.strategy in ("middle-cycle", "head-unique") and self.depth < self.num_unique + 1:
            raise ValueError(
                f"{self.strategy} needs depth > num_unique to have a middle to share; "
                f"got depth={self.depth}, num_unique={self.num_unique}"
            )

    @property
    def indices(self) -> tuple[int, ...]:
        """Unique-block index used at each depth, length ``depth``."""
        if self.strategy == "cycle":
            return tuple(i % self.num_unique for i in range(self.depth))
        if self.strategy == "middle-cycle":
            # Block 0 and block depth-1 are their own; blocks 1..num_unique-2 cycle over the middle.
            middle = self.num_unique - 2
            return (0,) + tuple(1 + (i % middle) for i in range(self.depth - 2)) \
                   + (self.num_unique - 1,)
        if self.strategy == "head-unique":
            # Block 0 unique; everything deeper cycles over blocks 1..num_unique-1.
            rest = self.num_unique - 1
            return (0,) + tuple(1 + (i % rest) for i in range(self.depth - 1))
        # `sequence`: split depth into num_unique contiguous runs as evenly as possible, so a
        # depth that does not divide evenly spreads the remainder over the earliest blocks
        # rather than dumping it all on the last one.
        base, extra = divmod(self.depth, self.num_unique)
        out: list[int] = []
        for k in range(self.num_unique):
            out.extend([k] * (base + (1 if k < extra else 0)))
        return tuple(out)

    @property
    def is_stock(self) -> bool:
        """True when this schedule is indistinguishable from an untied stack."""
        return self.num_unique == self.depth


# --------------------------------------------------------------------------------------------
# per-depth modulation (R3)
# --------------------------------------------------------------------------------------------
class DepthModulation(nn.Module):
    """Learned per-depth affine ``x * (1 + gamma_d) + beta_d`` on a tied block's output.

    Tying forces every depth to use identical weights. This asks a narrower question: is what the
    depths actually need *different weights*, or merely a different *scale*? At
    ``2 * embed_dim`` parameters per depth (2,048 for d=128 at depth 8, against 199,688 for one
    real block) it recovers capacity at ~1% of the cost of untying, so a large gain here would
    mean depthwise specialisation is mostly a gain-control phenomenon.

    Initialised to the identity (``gamma = beta = 0``), so an untrained modulated stack is
    numerically identical to the unmodulated one — which the tests assert, because a
    "capacity recovery" mechanism that perturbs the model at init would confound its own effect.
    """

    def __init__(self, embed_dim: int, depth: int):
        super().__init__()
        self.gamma = nn.Parameter(torch.zeros(depth, embed_dim))
        self.beta = nn.Parameter(torch.zeros(depth, embed_dim))

    def forward(self, x: Tensor, depth_index: int) -> Tensor:
        return x * (1.0 + self.gamma[depth_index]) + self.beta[depth_index]


class ModulatedBlock(nn.Module):
    """Wraps a shared block so that depth ``d`` applies its own affine to the output.

    Holds the block by reference, so several ``ModulatedBlock``s over the same block share its
    parameters; only the modulation is per-depth.
    """

    def __init__(self, block: nn.Module, modulation: DepthModulation, depth_index: int):
        super().__init__()
        self.block = block
        self.modulation = modulation
        self.depth_index = int(depth_index)

    def forward(self, x: Tensor, x_cls: Optional[Tensor] = None,
                padding_mask: Optional[Tensor] = None,
                attn_mask: Optional[Tensor] = None) -> Tensor:
        out = self.block(x, x_cls=x_cls, padding_mask=padding_mask, attn_mask=attn_mask)
        return self.modulation(out, self.depth_index)


# --------------------------------------------------------------------------------------------
# Mixture-of-Recursions routing (R4)
# --------------------------------------------------------------------------------------------
class MoRRouter(nn.Module):
    """Expert-choice router with fixed **per-jet** capacity: which tokens get this recursion step.

    Mixture-of-Recursions (arXiv:2507.10524) pairs a shared block stack with a light router that
    gives each token its own recursion depth. Two routing modes exist; this is the
    **expert-choice** one, where the step selects its top-``capacity`` tokens rather than each
    token selecting steps.

    Capacity is a fraction of each jet's **own valid multiplicity**, i.e. ``ceil(capacity *
    n_valid_i)`` for jet ``i`` -- never a fraction of the padded tensor width. That distinction is
    not cosmetic. Jets are padded to a common ``N`` (128 here) while real multiplicity runs from a
    handful to 183, so taking ``ceil(capacity * N)`` would make the effective capacity a function
    of how much padding a jet happens to sit beside: at ``capacity=0.5`` and ``N=128``, a jet with
    20 valid particles gets 64 >= 20 slots and therefore *dense* depth, a 40-particle jet also
    runs dense, and only jets above 128 particles see a real 0.5. The arm would be recorded as
    "capacity 0.5" when no jet in it ever ran at 0.5, and the low-multiplicity jets -- the
    majority -- would silently be the dense baseline.

    Worst-case cost is still bounded and data-independent: ``n_valid <= N`` gives at most
    ``ceil(capacity * N)`` selected tokens per jet. That bound is what matters for the trigger,
    where the budget is worst-case rather than average, and it is preserved.

    The physics reason to expect this to *do* something: a jet's constituents are far more
    heterogeneous than an LLM's tokens. Multiplicity runs from a handful to 183 (mean ~39), and
    the leading few constituents carry most of the pT while soft radiation contributes little.
    "Which particles deserve more computation?" therefore has a physical answer, and whether the
    router finds it is checkable — which makes the router's *behaviour* a result even if the
    accuracy delta lands inside seed noise.
    """

    def __init__(self, embed_dim: int, capacity: float = 0.5):
        super().__init__()
        if not 0.0 < capacity <= 1.0:
            raise ValueError(f"capacity must be in (0, 1], got {capacity}")
        self.capacity = float(capacity)
        self.score = nn.Linear(embed_dim, 1)
        # Zero init => every logit is 0 at step 0, so the selected set is decided entirely by how
        # ties are broken. `topk` does NOT document a stable tie order and its choice varies by
        # backend, which would make an untrained MoR arm irreproducible across devices even at a
        # fixed seed -- so selection below uses a *stable* sort instead, whose tie order is
        # documented (lower index first). JetClass shards constituents in descending pT, so that
        # resolves ties toward the leading particles: a deliberate and physically sensible prior,
        # not an accident of the kernel.
        nn.init.zeros_(self.score.weight)
        nn.init.zeros_(self.score.bias)

    def forward(self, x: Tensor, padding_mask: Optional[Tensor] = None) -> tuple[Tensor, Tensor]:
        """Return ``(gate, keep)``.

        ``keep`` is ``(B, N)`` bool: tokens this step processes. ``gate`` is ``(B, N, 1)``, the
        router probability for kept tokens and 0 elsewhere — multiplied into the block's update so
        the router receives gradient. Without that the selection is discrete and the router would
        never learn, which is the standard MoE/MoR trick.
        """
        b, n, _ = x.shape
        logits = self.score(x).squeeze(-1)                     # (B, N)
        if padding_mask is not None:
            # padding_mask: True = padded. Padded slots must never consume capacity.
            logits = logits.masked_fill(padding_mask, float("-inf"))
            valid = (~padding_mask).sum(dim=1)                 # (B,)
        else:
            valid = torch.full((b,), n, device=x.device, dtype=torch.long)

        # Per-jet k, from each jet's own valid count. `clamp(min=1)` keeps a fully-padded row
        # well-defined (its one selection is masked off again below).
        #
        # float32 with a tolerance, not float64: MPS has no float64, and `valid` is a (B,) tensor
        # so precision is the only concern, not speed. The `- _CEIL_TOL` matters because
        # `capacity * valid` can land a hair *above* an exact integer (0.1 * 30 -> 3.0000001),
        # and a bare `ceil` would then hand out one extra slot for arithmetic reasons.
        k_per_jet = torch.ceil(
            valid.to(torch.float32) * self.capacity - _CEIL_TOL
        ).long()
        k_per_jet = torch.minimum(k_per_jet.clamp(min=1), valid.clamp(min=1))
        kmax = int(k_per_jet.max())

        # Stable sort, not topk: ties must break deterministically (see __init__).
        order = logits.sort(dim=1, descending=True, stable=True).indices[:, :kmax]
        within_budget = torch.arange(kmax, device=x.device).unsqueeze(0) < k_per_jet.unsqueeze(1)
        keep = torch.zeros(b, n, dtype=torch.bool, device=x.device)
        keep.scatter_(1, order, within_budget)
        if padding_mask is not None:
            # `-inf` slots can still be reached when a jet has fewer than kmax valid particles;
            # drop them so capacity is never spent on padding.
            keep &= ~padding_mask
        # Capacity actually used per jet, for reporting. Exact, not inferred from k.
        self._last_used = keep.sum(dim=1)

        probs = torch.sigmoid(logits).unsqueeze(-1)            # (B, N, 1)
        gate = probs * keep.unsqueeze(-1)
        return gate, keep


class MoRBlock(nn.Module):
    """One MoR recursion step: route, apply the shared block to kept tokens, gate the update.

    Tokens the router skips pass through **unchanged**, which is what makes the step cheaper than
    a dense one. The block is still called on the full tensor (a gather/scatter would be the
    genuinely sparse implementation and belongs in a kernel, not here), so this measures the
    *modelling* effect of adaptive depth rather than delivering its speedup. That distinction is
    stated because conflating the two is exactly how a "2x faster" claim gets made about code that
    is not faster.
    """

    def __init__(self, block: nn.Module, router: MoRRouter):
        super().__init__()
        self.block = block
        self.router = router

    def forward(self, x: Tensor, x_cls: Optional[Tensor] = None,
                padding_mask: Optional[Tensor] = None,
                attn_mask: Optional[Tensor] = None) -> Tensor:
        gate, _ = self.router(x, padding_mask)
        out = self.block(x, x_cls=x_cls, padding_mask=padding_mask, attn_mask=attn_mask)
        # Convex-ish blend: kept tokens take the block's output weighted by the router gate,
        # skipped tokens keep their input exactly. Written as a residual so gate=0 is provably
        # the identity.
        return x + gate * (out - x)


# --------------------------------------------------------------------------------------------
# assembly
# --------------------------------------------------------------------------------------------
#: Which parameters a tied arm shares. ``"block"`` shares whole blocks (199,688 each). The two
#: sub-block scopes exist because **ALBERT's own ablation says they are not equivalent**: at E=128,
#: sharing attention across layers costs +0.1 aggregate (i.e. free) while sharing the FFN costs
#: -1.4, and the paper states "most of the performance drop appears to come from sharing the
#: FFN-layer parameters" (arXiv:1909.11942 Table 4, Sec. 3.1).
#:
#: In ParT the split is 66,048 attention / 131,712 FFN of 199,688 per block, so attention is only
#: **33.1 %** — the ALBERT-safe sharing gives just 1.28x, and this repo's headline 2.87x is obtained
#: precisely by sharing the sublayer ALBERT identifies as costly. Running both scopes converts
#: "does tying work?" into the better-posed "which sublayer carries depthwise specialisation?",
#: which yields a result either way.
#:
#: Norms are deliberately NOT shared in the sub-block scopes: only the projection weights are, so
#: each depth keeps its own pre/post norms, ``w_resid`` and ``c_attn`` (1,928 params/depth, 0.9 % of
#: a block). That is the conservative reading of "share the attention parameters", and it keeps the
#: per-depth gain control the looped literature finds load-bearing.
TIE_SCOPES = ("block", "attn", "ffn")


def build_tied_blocks(
    blocks: Sequence[nn.Module],
    schedule: TiedSchedule,
    *,
    embed_dim: Optional[int] = None,
    depth_modulation: bool = False,
    mor_capacity: Optional[float] = None,
    scope: str = "block",
) -> nn.ModuleList:
    """Build the ``ModuleList`` weaver will iterate, of length ``schedule.depth``.

    Parameters
    ----------
    blocks
        At least ``schedule.num_unique`` freshly built blocks. Extras are ignored, so a caller can
        hand over a stock ``model.blocks`` of length 8 and tie it to any ``num_unique <= 8``.
    depth_modulation
        Wrap each entry in :class:`ModulatedBlock` with a shared :class:`DepthModulation` (R3).
    mor_capacity
        If set, wrap each entry in :class:`MoRBlock` with a fixed-capacity router (R4).

    Returns a list whose entries **share parameters** where the schedule repeats an index. Note
    ``nn.ModuleList`` accepts the same module object more than once, and ``parameters()`` dedupes
    by identity, so the reported parameter count is the shared one — while ``state_dict()`` does
    *not* dedupe and will contain one copy of the shared tensors per depth. That is harmless for
    correctness but inflates checkpoint size, and it is why the tests assert on
    ``parameters()`` rather than on ``state_dict()``.
    """
    if scope not in TIE_SCOPES:
        raise ValueError(f"scope must be one of {list(TIE_SCOPES)}, got {scope!r}")
    if scope != "block":
        # Sub-block scopes keep every depth's own block object and rebind only the shared
        # submodules, so depth-specific norms and gain controls survive. `schedule.indices` still
        # decides WHICH unique parameter set each depth uses.
        if len(blocks) < schedule.depth:
            raise ValueError(
                f"scope={scope!r} shares submodules across all {schedule.depth} depths, so it needs "
                f"{schedule.depth} blocks, got {len(blocks)}"
            )
        if depth_modulation or mor_capacity is not None:
            raise ValueError(
                "depth_modulation and mor_capacity apply to whole-block tying; combining them with "
                f"scope={scope!r} would confound sub-block sharing with a second mechanism"
            )
        attrs = ("attn",) if scope == "attn" else ("fc1", "fc2")
        out_sub: list[nn.Module] = list(blocks[: schedule.depth])
        for depth_index, block_index in enumerate(schedule.indices):
            source = out_sub[block_index]
            for attr in attrs:
                setattr(out_sub[depth_index], attr, getattr(source, attr))
        return nn.ModuleList(out_sub)

    if len(blocks) < schedule.num_unique:
        raise ValueError(
            f"need at least {schedule.num_unique} blocks for this schedule, got {len(blocks)}"
        )
    if depth_modulation and embed_dim is None:
        raise ValueError("embed_dim is required when depth_modulation=True")
    if depth_modulation and mor_capacity is not None:
        raise ValueError(
            "depth_modulation and mor_capacity are separate arms (R3 vs R4); combining them "
            "confounds two mechanisms in one measurement"
        )

    unique = list(blocks[: schedule.num_unique])
    modulation = (
        DepthModulation(int(embed_dim), schedule.depth) if depth_modulation else None
    )
    router = MoRRouter(int(embed_dim), mor_capacity) if mor_capacity is not None else None
    if router is not None and embed_dim is None:
        raise ValueError("embed_dim is required when mor_capacity is set")

    out: list[nn.Module] = []
    for depth_index, block_index in enumerate(schedule.indices):
        block = unique[block_index]
        if modulation is not None:
            out.append(ModulatedBlock(block, modulation, depth_index))
        elif router is not None:
            out.append(MoRBlock(block, router))
        else:
            out.append(block)
    return nn.ModuleList(out)


def shared_block_groups(model: nn.Module) -> list[list[int]]:
    """Groups of ``model.blocks`` indices that share one module. Only groups of size > 1."""
    blocks = getattr(model, "blocks", None)
    if blocks is None:
        return []
    groups: dict[int, list[int]] = {}
    for index, block in enumerate(blocks):
        target = getattr(block, "block", block)  # unwrap Modulated/MoR wrappers
        groups.setdefault(id(target), []).append(index)
    return [g for g in groups.values() if len(g) > 1]


def assert_shared_blocks_consistent(model: nn.Module, state: dict,
                                    source: str = "checkpoint") -> None:
    """Verify a state dict did not silently collapse this model's duplicated block keys.

    A tied stack's ``state_dict()`` emits one entry per *depth* for a block that exists once, so a
    state dict from a differently-tied or untied run has identical key names and shapes and loads
    without error -- keeping only the last duplicate of each shared block. This is the converse
    check, and it needs no config: for every group of indices that share a module, the
    corresponding entries in *state* must already agree. If they disagree, the load threw weights
    away, and the resulting model is a function of the checkpoint that equals no part of it.

    Complements :func:`ablation.config.assert_checkpoint_architecture`, which compares the recorded
    configs. This one still fires when the config is absent, wrong, or was itself the thing that
    drifted -- which is the failure mode this repository has actually had.

    A no-op when every block is distinct.
    """
    groups = shared_block_groups(model)
    if not groups:
        return

    mismatched: list[str] = []
    for group in groups:
        ref = group[0]
        pattern = re.compile(rf"(?P<pre>(?:.*\.)?)blocks\.{ref}\.(?P<rest>.+)$")
        for key, ref_value in state.items():
            match = pattern.match(key)
            if match is None or not isinstance(ref_value, Tensor):
                continue
            for other in group[1:]:
                twin = f"{match.group('pre')}blocks.{other}.{match.group('rest')}"
                value = state.get(twin)
                if not isinstance(value, Tensor):
                    continue
                if value.shape != ref_value.shape or not torch.equal(
                    value.detach().cpu(), ref_value.detach().cpu()
                ):
                    mismatched.append(f"    {key}  !=  {twin}")

    if mismatched:
        shown = mismatched[:8]
        more = f"\n    ... and {len(mismatched) - len(shown)} more" if len(mismatched) > len(shown) else ""
        raise ValueError(
            f"{source} disagrees across depths that this model ties together, so loading it "
            f"discarded weights (only the last duplicate survives):\n"
            + "\n".join(shown) + more
            + f"\n  Tied groups in the active model: {groups}. This most often means the "
            "checkpoint came from an untied or differently-tied run."
        )


def apply_residual_scale(model: nn.Module, schedule: TiedSchedule, lam: float,
                        scope: str = "block") -> dict:
    """Set each block's LayerScale gamma to ``lam / (N_b * sqrt(L))`` using its OWN loop count.

    The looped-transformer law ``eps = lambda / (N sqrt(L))`` is derived for ``L`` unique blocks each
    applied ``N`` times. A **single averaged** ``N = depth / num_unique`` is therefore wrong for any
    non-uniform schedule, and silently so:

        middle-cycle k=3 -> indices (0,1,1,1,1,1,1,2): block 1 runs SIX times, blocks 0 and 2 run
        once. The average N = 8/3 = 2.67 describes none of the three.

    This function computes ``N_b`` per unique block from ``schedule.indices`` and writes the
    corresponding gamma directly, so every block gets the scale its own reuse count implies.

    ``scope`` matters too. With ``scope="attn"`` only the attention projections are shared and the
    eight FFNs stay distinct, so the FFN branch is NOT looped and must keep the untied scale --
    applying the shared-branch value to both branches would over-damp a branch that has no
    cross-loop coherence to correct for. Only the shared branch is rescaled:

        scope="block" -> both ls1 (attention) and ls2 (FFN)
        scope="attn"  -> ls1 only
        scope="ffn"   -> ls2 only

    Returns a record of what was written, so a run's provenance carries the realised scales rather
    than a formula someone has to re-derive.
    """
    from collections import Counter

    counts = Counter(schedule.indices)
    unique = schedule.num_unique
    # Scale for a branch that is NOT shared. In a sub-block scope there are `depth` distinct
    # instances of that branch, each applied once, so its own (N, L) is (1, depth) and the scale is
    # lam/sqrt(depth) -- i.e. exactly what the untied baseline gets. Using lam/sqrt(num_unique) here
    # was wrong: at num_unique=1 it gave 1.0, leaving the unshared FFN branch 2.8x hotter than the
    # baseline it is supposed to match, which would have shown up as an "attention-only tying hurts"
    # result caused entirely by the control.
    untied = float(lam) / math.sqrt(max(schedule.depth, 1))
    written: dict[str, float] = {}

    branches = {"block": ("ls1", "ls2"), "attn": ("ls1",), "ffn": ("ls2",)}[scope]
    seen: set[int] = set()
    for depth_index, block_index in enumerate(schedule.indices):
        target = model.blocks[depth_index]
        inner = getattr(target, "block", target)         # unwrap Modulated/MoR wrappers
        if id(inner) in seen and scope == "block":
            continue                                     # shared object: write once
        seen.add(id(inner))
        n_b = counts[block_index]
        eps = float(lam) / (n_b * math.sqrt(max(unique, 1)))
        for name in ("ls1", "ls2"):
            layer = getattr(inner, name, None)
            if layer is None or not hasattr(layer, "gamma"):
                continue
            value = eps if name in branches else untied
            with torch.no_grad():
                layer.gamma.fill_(value)
            written[f"depth{depth_index}.{name}"] = value

    # The class-attention blocks are never tied, so they must carry the UNTIED scale in every arm --
    # exactly what `baseline_wave2` gets (lam / sqrt(depth)). Before this was written explicitly they
    # inherited the tied arm's uniform initial value from `block_params` (0.125 at k=1, 0.217 at
    # mc3, vs 0.354 on the baseline control), so the wave-2 comparison would have differed from its
    # control in the class readout, a place where no tying happens. Found by the wave-2
    # YAML->built-model test on 2026-09-09, not by the audit.
    for cls_index, cls_block in enumerate(getattr(model, "cls_blocks", None) or []):
        for name in ("ls1", "ls2"):
            layer = getattr(cls_block, name, None)
            if layer is None or not hasattr(layer, "gamma"):
                continue
            with torch.no_grad():
                layer.gamma.fill_(untied)
            written[f"cls{cls_index}.{name}"] = untied
    return {
        "lambda": float(lam),
        "scope": scope,
        "num_unique": unique,
        "applications_per_block": dict(counts),
        "untied_branch_scale": untied,
        "written": written,
    }


def tied_cost_report(model: nn.Module, schedule: TiedSchedule) -> dict:
    """Parameters and forward-application count, so a comparison can be checked for confounds.

    SMELT (arXiv:2609.01343) matches per-token FLOPs and non-embedding parameters exactly to
    isolate depth reuse, and that is the bar. Tying at fixed depth keeps ``block_applications``
    constant while cutting ``unique_block_params``: **FLOPs unchanged, parameters reduced, no
    speedup.** Any write-up claiming otherwise should be checked against this function's output.
    """
    seen: set[int] = set()
    unique_block_params = 0
    for block in model.blocks:
        target = getattr(block, "block", block)  # unwrap Modulated/MoR wrappers
        if id(target) in seen:
            continue
        seen.add(id(target))
        unique_block_params += sum(p.numel() for p in target.parameters())
    total = sum(p.numel() for p in model.parameters())
    return {
        "depth": schedule.depth,
        "num_unique": schedule.num_unique,
        "strategy": schedule.strategy,
        "schedule": list(schedule.indices),
        "unique_blocks_instantiated": len(seen),
        "unique_block_params": unique_block_params,
        "total_params": total,
        # Depth is unchanged by tying, so this is the FLOP-relevant count and it must match the
        # untied arm for the comparison to isolate parameter sharing.
        "block_applications": schedule.depth,
    }
