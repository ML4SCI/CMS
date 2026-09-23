"""Configuration for the ParT ablation runs.

One flat dataclass, loadable from YAML and overridable from the command line, so
that every run is reproducible from a single serialized artifact: the resolved
config is written next to the checkpoints.

Defaults follow the official ParT JetClass recipe (arXiv:2202.03772 and the
weaver-core training configs): 1M iterations at batch 512, Lookahead(RAdam),
peak LR 1e-3 held for the first 70% of training then decayed exponentially,
no weight decay, mixed precision.
"""

from __future__ import annotations

import dataclasses
import math
import json
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any, Mapping, Optional

__all__ = [
    "AblationConfig",
    "ARMS",
    "BASE_KEY",
    "load_config",
    "apply_overrides",
]

#: Ablation arms, in reporting order.  These are the keys accepted by
#: ``variants.build_variant_part``.  ``n8`` is a screen arm (not in
#: ``submit_all.sh``).
ARMS = (
    "baseline",
    "tied",
    "lowrank",
    "lloca",
    "moe",
    "sparsemax",
    "diff_v1",
    "diff_v2",
    "n8",
    "urot",
)

#: Arms whose encoder blocks are stock weaver ``Block``s and so can realise ``residual_scale_lambda``
#: through ``LayerScale`` (``ls1``/``ls2``). Mirrors ``variants.weaver_adapter._LAYER_SCALE_ARMS``;
#: kept here too so a bad YAML fails at config load rather than at model build on the cluster.
LAYER_SCALE_ARMS = frozenset({"baseline", "tied", "lowrank"})


@dataclass
class AblationConfig:
    """Everything needed to reproduce one arm's training run."""

    # -- identity ----------------------------------------------------------
    arm: str = "baseline"
    run_name: str = ""
    #: Human-readable experiment label (logged at start; does not change paths).
    experiment: str = "part_ablation_v1"
    seed: int = 42

    # -- data --------------------------------------------------------------
    #: Directory holding ragged CSR ``.pt`` shards for training
    #: (one shard per ROOT file, produced by ``preprocessing/convert_jetclass_ragged_pt.py``).
    train_pt_dir: str = ""
    #: Directory holding ragged CSR ``.pt`` shards for validation.
    val_pt_dir: str = ""
    #: Optional path to ``norm_stats.json`` (produced by
    #: ``preprocessing/compute_norm_stats.py``).  When set, continuous particle
    #: features are z-scored and pT/E are scale-only normalised before being
    #: fed to the model.  Leave empty to train on raw features.
    norm_stats_path: str = ""
    num_workers: int = 8
    prefetch_factor: int = 4
    #: Mix all 10 classes into every local training batch. JetClass shards are
    #: single-class, so processing whole shards sequentially causes catastrophic
    #: forgetting and misleading near-100% window accuracy.
    class_balanced_train: bool = True
    #: Cap the validation set (pooled across ranks). Use the full val_5M size
    #: (5_000_000) so all 10 classes appear — shards are one class per file, so
    #: a small cap can fill entirely from the first class (HToBB) and make
    #: AUC / QCD-rejection NaN. Gather cost at 5M is ~100 MB of fp16 probs.
    val_max_jets: int = 5_000_000
    #: Train-only Cambridge–Aachen coarsening (HERON §4 / J-JEPA R=0.2).
    #: Off by default so the architecture ablation stays unaugmented.
    ca_augment: bool = False
    #: Per-jet probability of applying any C/A coarsening.
    ca_augment_prob: float = 0.5
    #: Upper bound on the C/A radius; the actual cut is ``U(0, ca_rmax)``.
    ca_rmax: float = 0.2
    #: Never coarsen a jet below this multiplicity (weaver BatchNorm is
    #: degenerate on a 1-particle pair grid in train mode).
    ca_min_particles: int = 2

    # -- model -------------------------------------------------------------
    input_dim: int = 16
    num_classes: int = 10
    num_layers: int = 8
    num_cls_layers: int = 2
    num_heads: int = 8
    embed_dims: tuple = (128, 512, 128)
    pair_embed_dims: tuple = (64, 64, 64)
    dropout: float = 0.1
    expansion_factor: int = 4

    #: MoE arm only: a key of ``variants.MOE_PRESETS`` or ``null`` for the
    #: default (FLOP-matched to the dense baseline).
    moe_config: Optional[str] = None

    # --- arm "tied": weight-tied (looped) encoder stack, variants/tied/ ---
    #: Distinct encoder blocks to share across the 8 depths. 1 = full ALBERT-style tying
    #: (745,538 params, 2.87x smaller); 8 = exactly stock baseline and therefore the control.
    tie_num_unique: int = 1
    #: "cycle" loops the group ([0,1,0,1,...]); "sequence" runs each block consecutively
    #: ([0,0,1,1,...]). Not equivalent -- see variants/tied/stack.py.
    tie_strategy: str = "cycle"
    #: What the tied arm shares: "block" (whole blocks), "attn" (attention projections only) or
    #: "ffn" (fc1/fc2 only). ALBERT's ablation says these are NOT equivalent -- sharing attention is
    #: free (+0.1) and the damage is in the FFN (-1.4) -- and in ParT attention is only 33.1% of a
    #: block, so the ALBERT-safe scope gives 1.28x where whole-block tying gives 2.87x.
    tie_scope: str = "block"
    #: R3: learned per-depth affine on the tied block's output, 2*embed_dim per depth.
    tie_depth_modulation: bool = False
    #: R4: Mixture-of-Recursions expert-choice routing at this fixed capacity fraction.
    #: Fixed capacity keeps FLOPs and worst-case latency deterministic. None = no routing.
    tie_mor_capacity: Optional[float] = None
    #: Weight on the MoE load-balancing auxiliary loss. Without it the router
    #: collapses onto a single expert.
    moe_aux_alpha: float = 0.01

    #: lambda in the looped-transformer residual-scaling law  eps = lambda / (N * sqrt(L)),
    #: where L = number of UNIQUE blocks and N = applications per unique block. When set, it is
    #: passed to weaver as `layer_scale_init_values`, which installs a LayerScale on each residual
    #: BRANCH (ls1 after attention, ls2 after the FFN) -- the position arXiv:2606.18524 prescribes.
    #:
    #: WHY THIS EXISTS. Weaver's `fix_init` rescales block i's residual branches by
    #: 1/sqrt(2*(i+1)), which is the right law for a stack of DISTINCT blocks. `build_tied_blocks`
    #: then keeps `blocks[:num_unique]` -- the shallowest, LEAST-rescaled blocks -- and applies them
    #: N times, which is the opposite of what a looped network needs. Measured residual-stream growth
    #: over depth: baseline 2.23x, tied k=1 **6.12x**. The mechanism is cross-loop coherence: eight
    #: applications of the SAME weights emit near-parallel updates (pairwise cosine up to +0.995,
    #: all-pair mean +0.630) so the stream accumulates linearly instead of in quadrature, where the
    #: untied stack's updates are orthogonal (all-pair mean -0.001). That is why the law is 1/N and
    #: not 1/sqrt(N).
    #:
    #: The single formula covers BOTH arms -- baseline is just L=num_layers, N=1 -- which is what
    #: makes it a control rather than a favour to the tied arm. Measured depth-8 residual RMS at
    #: lambda=1: baseline 1.24, k=1 1.28, k=2 1.25, k=4 1.19, i.e. arm-independent.
    #:
    #: Defaults to None (weaver's stock behaviour) so existing runs stay reproducible. Set it on
    #: EVERY arm in a wave or on none of them.
    residual_scale_lambda: Optional[float] = None

    # --- arm "lowrank": rank-r factorized pair bias, variants/lowrank/ ---
    #: Rank of the factorized pair bias. The P5 operating curve, measured by truncating a TRAINED
    #: bias (logs/audit-2026-09-05): r=34 is accuracy-neutral (-0.0005), r=16 costs -0.0205,
    #: r=8 costs -0.0385. Those are a LOWER bound for this arm, because training with the
    #: constraint lets the network adapt rather than being deprived of structure it had learned.
    lowrank_rank: int = 32
    #: Per-particle MLP widths producing the factors. None -> (64, 64).
    lowrank_hidden: Optional[tuple] = None
    #: The `c[h] * delta_ij` self-pair term. P4/P5 measured it worth a FACTOR OF 2 in effective
    #: rank for one stored number per head, so it defaults ON; switching it off is the ablation.
    lowrank_self_pair: bool = True

    #: N8 arm only (K6 screen: Minkowski + tensor-product rotary).
    n8_minkowski: bool = True
    n8_rotary: bool = True
    n8_rotary_pairs: int = 4
    #: ``1`` = rank-4 Minkowski extras; ``2`` appends the 10 quadratic
    #: monomials (``r = 14`` follow-up, not the K6 job).
    n8_degree: int = 1

    #: U-rotary arm: ``False`` = one θ per head (d=128 ``urot`` job).
    #: ``True`` = independent RoPE-style frequencies per 2-plane (``urot_mf``).
    urot_per_plane: bool = False
    #: ``True`` = LLaMA RoPE apply: pool U to a per-token phase, rotate Q
    #: and K, then ordinary ``QKᵀ`` (``urot_rope``).  Cannot combine with
    #: ``urot_per_plane``.
    urot_rope_apply: bool = False

    #: LLoCa arm only.
    lloca_frames_hidden_dim: int = 64
    lloca_frames_dropout: float = 0.0
    lloca_frames_min_mass: float = 5e-3
    lloca_symmetry_breaking: bool = True
    lloca_use_nis: bool = True
    lloca_num_vector_channels: Optional[int] = None
    lloca_freeze_frames_net: bool = False

    # -- optimization ------------------------------------------------------
    #: Total optimizer steps. ParT's published run is 1e6 at batch 512.
    total_steps: int = 1_000_000
    #: Per-GPU batch size. The global batch is this times the world size times
    #: ``grad_accum_steps`` (default 4 x 256 = 1024).
    batch_size: int = 256
    grad_accum_steps: int = 1
    lr: float = 1e-3
    weight_decay: float = 0.0
    radam_betas: tuple = (0.95, 0.999)
    radam_eps: float = 1e-5
    lookahead_steps: int = 6
    lookahead_alpha: float = 0.5
    #: Fraction of ``total_steps`` at constant peak LR before exponential decay.
    lr_constant_fraction: float = 0.7
    #: LR multiplier reached at the end of the decay phase.
    lr_final_factor: float = 0.01
    warmup_steps: int = 2_000
    grad_clip: float = 1.0

    # -- precision ---------------------------------------------------------
    #: ``bf16``, ``fp16`` or ``fp32``. Forced to ``fp32`` for the LLoCa arm:
    #: reduced precision violates its exact equivariance (arXiv:2505.20280
    #: App. D.3 runs LLoCa-ParT in single precision throughout).
    precision: str = "bf16"
    allow_tf32: bool = True

    # -- evaluation / logging ---------------------------------------------
    eval_every: int = 25_000
    log_every: int = 200
    checkpoint_every: int = 10_000
    keep_last_checkpoints: int = 2
    output_dir: str = "runs"
    #: Signal efficiencies at which background rejection is reported.
    #: 0.3 added 2026-09-06 so results are directly comparable to arXiv:2608.16061, which reports
    #: ParT parameter-reduction results at 50 % and 30 % signal efficiency and measures per-class
    #: rejection degrading ~25x MORE than accuracy. An accuracy-only readout would report a
    #: parameter-reduction arm as a null when it is not. Adding a column keeps every existing
    #: comparison valid.
    rejection_efficiencies: tuple = (0.3, 0.5, 0.99)
    #: Write ``predictions/step_XXXXXXX.npz`` at each eval (fp16 probs + labels).
    #: Small on disk (~8 MB / 400k jets) and enough to recompute any metric.
    save_eval_predictions: bool = True

    # -- runtime -----------------------------------------------------------
    #: Stop after this many steps regardless of ``total_steps``; the LR schedule
    #: still spans ``total_steps``.  Used by the smoke test.
    max_steps: Optional[int] = None
    compile_model: bool = False
    #: Patch weaver ``PairEmbed`` / ``Attention`` with fused Triton kernels from
    #: ``part_kernels``.  Pairwise acceleration applies to every arm (stock
    #: ``pair_embed`` is shared).  Attention fusion only helps when
    #: ``part_kernels_attention`` is true *and* the call is eligible (no
    #: train-mode attention dropout — see ``part_kernels.runtime``).
    use_part_kernels: bool = False
    part_kernels_attention: bool = False
    resume: str = ""

    def __post_init__(self) -> None:
        if self.arm not in ARMS:
            raise ValueError(
                f"unknown arm {self.arm!r}; expected one of {list(ARMS)}"
            )
        if not self.run_name:
            self.run_name = self.arm
        if self.precision not in ("bf16", "fp16", "fp32"):
            raise ValueError(
                f"precision must be bf16, fp16 or fp32; got {self.precision!r}"
            )
        if self.n8_degree not in (1, 2):
            raise ValueError(f"n8_degree must be 1 or 2, got {self.n8_degree}")
        if self.n8_rotary_pairs < 1:
            raise ValueError(
                f"n8_rotary_pairs must be >= 1, got {self.n8_rotary_pairs}"
            )
        if self.arm == "lloca" and self.precision != "fp32":
            # Not silently corrected: an ablation comparing a "Lorentz-equivariant"
            # arm that is not actually equivariant would be a wrong result, and a
            # warning in a 24h SLURM log is easy to miss.
            raise ValueError(
                "the lloca arm requires precision='fp32': reduced precision "
                "breaks exact Lorentz equivariance (arXiv:2505.20280 App. D.3). "
                "Set precision: fp32 in the config for this arm."
            )
        if not 0.0 < self.lr_constant_fraction <= 1.0:
            raise ValueError("lr_constant_fraction must be in (0, 1]")
        if self.grad_accum_steps < 1:
            raise ValueError("grad_accum_steps must be >= 1")
        if self.residual_scale_lambda is not None and self.residual_scale_lambda <= 0:
            raise ValueError(
                f"residual_scale_lambda must be > 0, got {self.residual_scale_lambda}"
            )
        if self.residual_scale_lambda is not None and self.arm not in LAYER_SCALE_ARMS:
            # The scale is realised through weaver's LayerScale (Block.ls1/ls2), which only stock
            # weaver blocks have. On every other arm the key would be accepted, stored in the
            # checkpoint and silently ignored -- the `ffn2x` failure again (2026-09-09 audit, A1).
            # Fail at config load so it never reaches an sbatch.
            raise ValueError(
                f"residual_scale_lambda is set but arm {self.arm!r} replaces or wraps the stock "
                f"weaver blocks and has no LayerScale to apply it to; the field would be a no-op "
                f"recorded as if it applied. Supported arms: {sorted(LAYER_SCALE_ARMS)}."
            )

        if self.arm == "lowrank":
            if self.lowrank_rank < 1:
                raise ValueError(f"lowrank_rank must be >= 1, got {self.lowrank_rank}")
            if self.pair_embed_dims is None:
                raise ValueError(
                    "the `lowrank` arm replaces the pair bias, so pair_embed_dims must be set "
                    "(it determines that the base model HAS a pair bias to replace)"
                )
            if self.lowrank_hidden is not None:
                self.lowrank_hidden = tuple(self.lowrank_hidden)

        if self.arm == "tied":
            if not 1 <= self.tie_num_unique <= self.num_layers:
                raise ValueError(
                    f"tie_num_unique must be in [1, num_layers={self.num_layers}], "
                    f"got {self.tie_num_unique}"
                )
            if self.tie_scope not in ("block", "attn", "ffn"):
                raise ValueError(
                    f"tie_scope must be block/attn/ffn, got {self.tie_scope!r}"
                )
            # Imported rather than duplicated: a hardcoded list here would silently reject a
            # strategy the stack supports (which is exactly what happened when middle-cycle and
            # head-unique were added), and the config is the layer users actually touch.
            from variants.tied.stack import _STRATEGIES

            if self.tie_strategy not in _STRATEGIES:
                raise ValueError(
                    f"tie_strategy must be one of {sorted(_STRATEGIES)}, "
                    f"got {self.tie_strategy!r}"
                )
            if self.tie_mor_capacity is not None and not 0.0 < self.tie_mor_capacity <= 1.0:
                raise ValueError(
                    f"tie_mor_capacity must be in (0, 1], got {self.tie_mor_capacity}"
                )
            if self.tie_depth_modulation and self.tie_mor_capacity is not None:
                raise ValueError(
                    "tie_depth_modulation (R3) and tie_mor_capacity (R4) are separate arms; "
                    "combining them confounds two mechanisms in one measurement"
                )
        if not 0.0 <= self.ca_augment_prob <= 1.0:
            raise ValueError("ca_augment_prob must be in [0, 1]")
        if self.ca_rmax < 0.0:
            raise ValueError("ca_rmax must be >= 0")
        if self.ca_min_particles < 1:
            raise ValueError("ca_min_particles must be >= 1")
        self.embed_dims = tuple(self.embed_dims)
        self.pair_embed_dims = (
            tuple(self.pair_embed_dims) if self.pair_embed_dims else None
        )
        self.radam_betas = tuple(self.radam_betas)
        self.rejection_efficiencies = tuple(self.rejection_efficiencies)
        if self.arm == "urot" and self.pair_embed_dims is None:
            raise ValueError(
                "the urot arm consumes PairEmbed U as a pairwise rotation; "
                "pair_embed_dims must be set (this is not N8 / not K2)"
            )
        if self.urot_per_plane and self.urot_rope_apply:
            raise ValueError(
                "urot_per_plane and urot_rope_apply cannot both be True: "
                "RoPE apply uses geometric frequencies on a per-token phase"
            )

    # -- model construction ------------------------------------------------

    def residual_branch_scale(self) -> Optional[float]:
        """``eps = lambda / (N * sqrt(L))`` -- the looped-transformer residual-scaling law.

        ``L`` is the number of **unique** blocks and ``N`` the applications per unique block, so the
        single formula covers every arm: an untied stack is ``L = num_layers, N = 1``. That is what
        makes applying it a *control* rather than a favour to the tied arm -- measured depth-8
        residual RMS at ``lambda = 1`` is 1.24 (baseline), 1.28 (k=1), 1.25 (k=2), 1.19 (k=4).

        Returns ``None`` when ``residual_scale_lambda`` is unset, which leaves weaver's stock
        behaviour (no ``LayerScale``) untouched so existing runs stay reproducible.
        """
        if self.residual_scale_lambda is None:
            return None
        # Sub-block scopes ("attn"/"ffn") use L = tie_num_unique as well, even though 8 distinct
        # block objects survive and only one submodule is shared. The naive reading -- "the FFNs are
        # all distinct, so L = 8" -- is wrong, and it was checked rather than assumed. Measured
        # residual growth against baseline's 1.595 target:
        #     scope=attn  eps=0.125 (L=1) -> 1.373  (0.86x)     eps=0.354 (L=8) -> 2.655  (1.66x)
        #     scope=ffn   eps=0.125 (L=1) -> 1.399  (0.88x)     eps=0.354 (L=8) -> 2.778  (1.74x)
        # So L=1 is within 14% of baseline where L=8 is 66-74% off: sharing even ONE submodule across
        # depth is enough to make the residual updates coherent, which is itself a small result about
        # where the accumulation comes from.
        unique = self.tie_num_unique if self.arm == "tied" else self.num_layers
        unique = max(1, min(int(unique), self.num_layers))
        applications = self.num_layers / unique
        return float(self.residual_scale_lambda) / (applications * math.sqrt(unique))

    def model_kwargs(self) -> dict:
        """Arm-specific keyword arguments for ``build_variant_part``."""
        common = dict(
            input_dim=self.input_dim,
            num_classes=self.num_classes,
            embed_dims=self.embed_dims,
            pair_embed_dims=self.pair_embed_dims,
            num_heads=self.num_heads,
            num_layers=self.num_layers,
            num_cls_layers=self.num_cls_layers,
            dropout=self.dropout,
            expansion_factor=self.expansion_factor,
        )
        if self.residual_scale_lambda is not None:
            # `layer_scale_init_values` installs the LayerScale modules at all (weaver gates them on
            # a truthy value); `residual_scale_lambda` is what the per-block/per-branch rewrite in
            # `variants.tied.apply_residual_scale` actually uses. The value below is only the
            # UNIFORM-schedule approximation and is correct as an initial value for arms whose
            # schedule really is uniform; non-uniform schedules and sub-block scopes have their
            # gammas overwritten per block afterwards.
            common["layer_scale_init_values"] = self.residual_branch_scale()
            if self.arm == "tied":
                common["residual_scale_lambda"] = self.residual_scale_lambda

        if self.arm == "lowrank":
            common.update(
                lowrank_rank=self.lowrank_rank,
                lowrank_hidden=self.lowrank_hidden,
                lowrank_self_pair=self.lowrank_self_pair,
            )
        elif self.arm == "tied":
            common.update(
                tie_num_unique=self.tie_num_unique,
                tie_strategy=self.tie_strategy,
                tie_scope=self.tie_scope,
                tie_depth_modulation=self.tie_depth_modulation,
                tie_mor_capacity=self.tie_mor_capacity,
            )
        elif self.arm == "moe":
            common["moe_config"] = self.moe_config
        elif self.arm == "lloca":
            common.update(
                frames_hidden_dim=self.lloca_frames_hidden_dim,
                frames_dropout=self.lloca_frames_dropout,
                frames_min_mass=self.lloca_frames_min_mass,
                symmetry_breaking=self.lloca_symmetry_breaking,
                frames_use_nis=self.lloca_use_nis,
                num_vector_channels=self.lloca_num_vector_channels,
                freeze_frames_net=self.lloca_freeze_frames_net,
            )
        elif self.arm == "n8":
            common.update(
                minkowski=self.n8_minkowski,
                rotary=self.n8_rotary,
                rotary_pairs=self.n8_rotary_pairs,
                degree=self.n8_degree,
            )
        elif self.arm == "urot":
            common["per_plane"] = self.urot_per_plane
            common["rope_apply"] = self.urot_rope_apply
        return common

    @property
    def run_dir(self) -> Path:
        return Path(self.output_dir) / self.run_name

    def architecture_fingerprint(self) -> dict:
        """The fields that determine the *built* architecture, for checkpoint compatibility.

        Derived from :meth:`model_kwargs` rather than from a hand-written field list, deliberately.
        A hand-written list is a second source of truth that silently rots the moment an
        architecture field is added to the builder and not to the list -- which is the exact
        mechanism by which ``expansion_factor`` reached a config, never reached the model, and
        produced an "FFN 2x" arm byte-identical to baseline. Anything that changes the model must
        pass through ``model_kwargs``, so anything that changes the model is fingerprinted here.

        Tuples are normalized to lists because a fingerprint is compared against one that has
        round-tripped through JSON, where a tuple comes back as a list.
        """
        fp: dict = {"arm": self.arm}
        for key, value in self.model_kwargs().items():
            fp[key] = list(value) if isinstance(value, tuple) else value
        # Kernel fusion rewrites modules in place, so it is part of the architecture even though
        # it is not a `model_kwargs` entry.
        fp["use_part_kernels"] = bool(self.use_part_kernels)
        fp["part_kernels_attention"] = bool(self.part_kernels_attention)
        return fp

    def to_dict(self) -> dict:
        return {f.name: getattr(self, f.name) for f in fields(self)}

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            k: (list(v) if isinstance(v, tuple) else v)
            for k, v in self.to_dict().items()
        }
        path.write_text(json.dumps(payload, indent=2, sort_keys=True))


def config_from_payload(raw: dict) -> Optional["AblationConfig"]:
    """Rebuild an :class:`AblationConfig` from a checkpoint's stored ``config`` dict.

    Keys that no longer exist in the dataclass are dropped rather than raising, so a checkpoint
    written before a field was renamed stays readable.
    """
    if not raw:
        return None
    known = {f.name for f in fields(AblationConfig)}
    return AblationConfig(**{k: v for k, v in raw.items() if k in known})


def assert_checkpoint_architecture(payload: dict, config: "AblationConfig",
                                   path: Any = None) -> None:
    """Refuse to load a checkpoint whose architecture differs from *config*.

    ``load_state_dict`` is a name-and-shape check, and that is not enough to tell two of this
    repository's architectures apart. The sharpest case is weight tying: a ``tied`` stack holds the
    *same module object* at several ``blocks`` indices, so its ``state_dict`` has one key per depth
    with identical names and shapes to an untied stack's. Loading an untied 8-block checkpoint into
    a ``tied_k1`` model therefore reports "all keys matched successfully" and silently keeps only
    the *last* duplicate -- training then continues from a model that is a plausible-looking
    function of the checkpoint and equal to none of it. Nothing downstream can detect this, because
    the weights are finite, the loss is reasonable, and the recorded config says ``tied_k1``.

    So the architecture is compared explicitly, before the weights are touched. A checkpoint with
    no stored config cannot be checked; that is reported rather than assumed compatible.
    """
    stored = config_from_payload(payload.get("config") or {})
    where = f" ({path})" if path else ""
    if stored is None:
        raise ValueError(
            f"checkpoint{where} carries no `config`, so its architecture cannot be verified "
            "against the active one. Refusing to load: an incompatible load is silent for "
            "tied arms (identical key names, last duplicate wins). Re-save the checkpoint with "
            "its config, or pass an explicit override if you have verified compatibility."
        )

    want, have = config.architecture_fingerprint(), stored.architecture_fingerprint()
    if want == have:
        return
    diffs = [
        f"    {key}: checkpoint={have.get(key)!r}  active={want.get(key)!r}"
        for key in sorted(set(want) | set(have))
        if want.get(key) != have.get(key)
    ]
    raise ValueError(
        f"checkpoint{where} was trained with a different architecture than the active config, "
        "so loading it would produce a model that matches neither:\n"
        + "\n".join(diffs)
        + "\n  This is checked because `load_state_dict` cannot catch it: tied and untied stacks "
        "share key names and shapes, so a mismatched load succeeds silently."
    )


def _coerce(name: str, raw: str) -> Any:
    """Parse a ``key=value`` override into the field's declared type."""
    declared = {f.name: f for f in fields(AblationConfig)}
    if name not in declared:
        raise KeyError(
            f"unknown config key {name!r}; valid keys: "
            f"{', '.join(sorted(declared))}"
        )
    if raw.lower() in ("none", "null"):
        return None

    annotation = declared[name].type
    text = str(annotation)
    if "bool" in text:
        if raw.lower() not in ("true", "false", "1", "0", "yes", "no"):
            raise ValueError(f"{name}: expected a boolean, got {raw!r}")
        return raw.lower() in ("true", "1", "yes")
    if "tuple" in text:
        return tuple(
            float(p) if "." in p else int(p) for p in raw.split(",") if p != ""
        )
    if "int" in text and "Optional" not in text:
        return int(raw)
    if "float" in text:
        return float(raw)
    if "Optional[int]" in text:
        return int(raw)
    return raw


def apply_overrides(base: Mapping[str, Any], overrides) -> dict:
    """Merge ``key=value`` CLI overrides into a config mapping."""
    merged = dict(base)
    for item in overrides or ():
        if "=" not in item:
            raise ValueError(
                f"override {item!r} is not of the form key=value"
            )
        key, raw = item.split("=", 1)
        key = key.strip()
        merged[key] = _coerce(key, raw.strip())
    return merged


#: Key used by a YAML config to inherit from another one.  Keeps the six per-arm
#: configs down to just their genuine differences instead of six copies of the
#: shared recipe that can silently drift apart.
BASE_KEY = "_base"


def _load_yaml(path: Path, _seen=()) -> dict:
    """Load a YAML config, resolving a chain of ``_base`` includes."""
    import yaml  # local import: only needed when a YAML config is used

    resolved = path.resolve()
    if resolved in _seen:
        chain = " -> ".join(str(p) for p in (*_seen, resolved))
        raise ValueError(f"circular {BASE_KEY} include: {chain}")

    loaded = yaml.safe_load(resolved.read_text()) or {}
    if not isinstance(loaded, dict):
        raise ValueError(f"{path}: expected a YAML mapping at the top level")

    base_ref = loaded.pop(BASE_KEY, None)
    if base_ref is None:
        return loaded

    base_path = (resolved.parent / base_ref).resolve()
    if not base_path.exists():
        raise FileNotFoundError(
            f"{path}: {BASE_KEY} points at a missing file {base_path}"
        )
    merged = _load_yaml(base_path, (*_seen, resolved))
    merged.update(loaded)
    return merged


def load_config(path: Optional[str], overrides=()) -> AblationConfig:
    """Build a config from an optional YAML file plus CLI overrides.

    YAML is optional: ``load_config(None, ["arm=moe", "batch_size=64"])`` is a
    complete way to specify a run.  A YAML file may set ``_base`` to inherit from
    another config; keys in the child win.
    """
    payload: dict = {}
    if path:
        payload = _load_yaml(Path(path))

    unknown = set(payload) - {f.name for f in fields(AblationConfig)}
    if unknown:
        raise KeyError(
            f"{path}: unknown config keys {sorted(unknown)}; valid keys: "
            f"{', '.join(sorted(f.name for f in fields(AblationConfig)))}"
        )

    return AblationConfig(**apply_overrides(payload, overrides))
