"""Wave 0 dense-capacity screens (FFN 2× / PairEmbed 2× / width).

These files must stay stock ``arm: baseline`` with a new ``run_name`` and a
200k cap. A silent resume of ``runs/baseline`` would contaminate the screen.

**Every screen is asserted against the BUILT MODEL, not just the parsed YAML.**
That distinction is the whole reason this file was rewritten on 2026-09-04. The
previous ``test_ffn2x_doubles_expansion_and_caps_at_200k`` asserted only
``config.expansion_factor == 8`` — i.e. that the YAML file contained the number
it obviously contained — and never constructed a model. Meanwhile
``build_variant_part`` dropped ``expansion_factor`` for the baseline arm, so the
arm trained a width-512 FFN while its config, its checkpoint and its own
filename all recorded 1024. A test named "doubles expansion" passed for weeks
over an experiment that never ran. A config assertion tests the *request*; only
a tensor shape tests the *effect*.
"""

from __future__ import annotations

from pathlib import Path

import torch

from ablation.config import load_config
from variants import build_variant_part

_REPO_ROOT = Path(__file__).resolve().parents[2]
_CONFIGS = _REPO_ROOT / "ablation" / "configs"

#: Stock ParT at the pinned weaver commit. Any screen that fails to change this
#: number is not a capacity screen.
_BASELINE_PARAMS = 2_143_354


def _build(name: str):
    """Load a screen's YAML and build the model it actually specifies."""
    config = load_config(str(_CONFIGS / name))
    torch.manual_seed(0)
    model = build_variant_part(config.arm, **config.model_kwargs())
    return config, model


def _ffn_hidden(model) -> int:
    return int(model.blocks[0].fc1.out_features)


def _params(model) -> int:
    return sum(p.numel() for p in model.parameters())


def test_baseline_is_the_reference_param_count() -> None:
    """Anchor the other tests: if this drifts, every screen's delta is meaningless."""
    _, model = _build("baseline.yaml")
    assert _params(model) == _BASELINE_PARAMS
    assert _ffn_hidden(model) == 512


def test_ffn2x_doubles_expansion_and_caps_at_200k() -> None:
    config, model = _build("baseline_ffn2x.yaml")
    assert config.arm == "baseline"
    assert config.run_name == "baseline_ffn2x"
    assert config.expansion_factor == 8
    assert config.pair_embed_dims == (64, 64, 64)
    assert config.embed_dims == (128, 512, 128)
    assert config.max_steps == 200_000
    assert config.total_steps == 1_000_000

    # The assertions that actually matter: the request reached the weights.
    assert _ffn_hidden(model) == 1024, (
        "expansion_factor=8 must double the FFN hidden width to 1024. If this is 512 the "
        "config is being accepted and ignored, which is how baseline_ffn2x trained a model "
        "byte-identical to baseline while recording expansion_factor=8."
    )
    assert _params(model) > _BASELINE_PARAMS, (
        "a dense-capacity screen that leaves the parameter count unchanged has not changed "
        "capacity — see logs/cluster-audit-2026-08-30.md section 4.1"
    )


def test_pair2x_doubles_pair_mlp_and_caps_at_200k() -> None:
    config, model = _build("baseline_pair2x.yaml")
    assert config.arm == "baseline"
    assert config.run_name == "baseline_pair2x"
    assert config.pair_embed_dims == (128, 128, 128)
    assert config.expansion_factor == 4
    assert config.embed_dims == (128, 512, 128)
    assert config.max_steps == 200_000
    assert config.total_steps == 1_000_000

    # pair_embed_dims reaches the model via `common`, so this one always worked -- but
    # assert it anyway rather than trusting that it still does.
    assert _ffn_hidden(model) == 512, "pair2x must not touch the FFN"
    assert _params(model) > _BASELINE_PARAMS, "widening PairEmbed must add parameters"


def test_wide_widens_token_path_and_caps_at_200k() -> None:
    config, model = _build("baseline_wide.yaml")
    assert config.arm == "baseline"
    assert config.run_name == "baseline_wide"
    assert config.embed_dims == (160, 640, 160)
    assert config.num_heads == 8
    assert config.embed_dims[0] % config.num_heads == 0
    assert config.pair_embed_dims == (64, 64, 64)
    assert config.expansion_factor == 4
    assert config.max_steps == 200_000
    assert config.total_steps == 1_000_000

    # embed_dim 160 with expansion_factor 4 => FFN hidden 640, and d_head 160/8 = 20.
    assert _ffn_hidden(model) == 640
    assert _params(model) > _BASELINE_PARAMS


def test_every_capacity_screen_actually_changes_the_model() -> None:
    """The check whose absence let a phantom arm run for weeks.

    A "dense-capacity screen" whose built model is indistinguishable from baseline is not a
    screen. Asserted collectively so that adding a new screen to configs/ without changing
    anything measurable fails here rather than after the GPU-hours are spent.
    """
    _, base = _build("baseline.yaml")
    fingerprint = (_params(base), _ffn_hidden(base))
    for name in ("baseline_ffn2x.yaml", "baseline_pair2x.yaml", "baseline_wide.yaml"):
        _, model = _build(name)
        assert (_params(model), _ffn_hidden(model)) != fingerprint, (
            f"{name} builds a model indistinguishable from baseline "
            f"({fingerprint[0]:,} params, FFN {fingerprint[1]}): its config is being "
            "accepted and ignored"
        )
