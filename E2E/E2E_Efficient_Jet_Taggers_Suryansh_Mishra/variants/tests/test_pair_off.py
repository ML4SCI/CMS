"""Pair-bias-off wiring for the T0.4 screen (``baseline_nopair``).

The T0.4 gate runs the stock baseline with ``pair_embed_dims=None`` and asks
whether the model still reaches baseline accuracy/rejection at matched steps.
A silent failure mode would defeat the whole screen: if ``pair_embed_dims``
were dropped on the floor and the model still built a PairEmbed, the "pair
bias is not load-bearing" conclusion would be measuring nothing. These tests
pin that the switch actually removes the module (and its N² bias) in the
built model, and that every way of spelling "off" in the config layer —
YAML ``null``, ``None``, and the CLI ``--set pair_embed_dims=`` empty form
used by the smoke jobs — resolves to ``None``.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from ablation.config import AblationConfig, apply_overrides, load_config
from variants import build_variant_part
from variants.tests.strategies import _make_four_vectors, _make_lengths_masks

_REPO_ROOT = Path(__file__).resolve().parents[2]
_NOPAIR_CONFIG = _REPO_ROOT / "ablation" / "configs" / "baseline_nopair.yaml"

#: Tiny model for the forward pass — the point is the plumbing, not capacity.
_TINY = dict(
    input_dim=16,
    num_classes=10,
    embed_dims=(16, 32, 16),
    num_heads=2,
    num_layers=2,
    num_cls_layers=1,
)


def _synthetic_batch(num_particles: int, lengths: list[int]):
    """A loader-shaped ``(x, v, mask)`` batch, as ``_real_batch`` produces."""
    generator = torch.Generator()
    generator.manual_seed(0)
    batch = len(lengths)
    x = torch.randn(
        batch, _TINY["input_dim"], num_particles, generator=generator
    )
    v = _make_four_vectors(lengths, num_particles, generator)
    mask, _ = _make_lengths_masks(lengths, num_particles)
    return x, v, mask


def test_pair_off_builds_no_pair_embed_and_forwards() -> None:
    """``pair_embed_dims=None`` removes the PairEmbed and the forward still runs.

    Weaver's switch is ``pair_embed is None``: the encoder then passes
    ``attn_mask=None`` to every block, so the attention logits are QK^T alone
    with no O(B N² C_pair) bias. A forward pass on a real-shaped batch pins
    that the rest of the pipeline accepts a missing bias.
    """
    model = build_variant_part("baseline", pair_embed_dims=None, **_TINY).eval()

    assert model.pair_embed is None, (
        "pair_embed_dims=None must build no PairEmbed — a silent re-enable "
        "would make the T0.4 screen measure the pair pipeline against itself"
    )

    x, v, mask = _synthetic_batch(24, [24, 17, 8])
    with torch.no_grad():
        logits = model(x, v=v, mask=mask)

    assert logits.shape == (3, _TINY["num_classes"])
    assert torch.isfinite(logits).all()


def test_pair_on_reference_still_builds_the_bias() -> None:
    """The default arm keeps its PairEmbed — the switch must be opt-in only."""
    model = build_variant_part("baseline", **_TINY)
    assert model.pair_embed is not None


def test_yaml_null_round_trips_to_none(tmp_path: Path) -> None:
    """YAML ``null`` resolves to ``AblationConfig.pair_embed_dims is None``."""
    config_file = tmp_path / "nopair.yaml"
    config_file.write_text("arm: baseline\npair_embed_dims: null\n")
    config = load_config(str(config_file))
    assert config.pair_embed_dims is None


def test_empty_cli_override_round_trips_to_none() -> None:
    """``--set pair_embed_dims=`` (the smoke job's spelling) resolves to None."""
    config = AblationConfig(**apply_overrides({}, ["pair_embed_dims="]))
    assert config.pair_embed_dims is None


def test_baseline_nopair_config_is_the_pair_off_screen() -> None:
    """The committed T0.4 config keeps the stock baseline and the 200k cap.

    This is the file the real screen runs, so it must pin all three load-
    bearing facts at once: pair bias off, part kernels off (nothing to fuse),
    and the 200k screen cap with the untouched 1M LR schedule.
    """
    config = load_config(str(_NOPAIR_CONFIG))

    assert config.arm == "baseline"
    assert config.pair_embed_dims is None
    assert config.use_part_kernels is False
    assert config.max_steps == 200_000
    assert config.total_steps == 1_000_000  # matched-step LR schedule
    assert config.run_name == "baseline_nopair"
    assert config.experiment == "t0_pair_off"
    # Everything else inherits the untouched baseline recipe.
    assert config.num_layers == 8
    assert config.num_heads == 8


def test_baseline_ca_config_caps_the_screen() -> None:
    """The C/A screen config is also capped at 200k steps."""
    config = load_config(
        str(_REPO_ROOT / "ablation" / "configs" / "baseline_ca.yaml")
    )
    assert config.arm == "baseline"
    assert config.ca_augment is True
    assert config.max_steps == 200_000
    assert config.pair_embed_dims == (64, 64, 64)


@pytest.mark.parametrize("pair_embed_dims", [None, (8, 8)])
def test_model_kwargs_pass_the_switch_through(pair_embed_dims) -> None:
    """``model_kwargs()`` forwards the switch so the trainer path stays honest."""
    config = AblationConfig(
        arm="baseline",
        embed_dims=(16, 32, 16),
        pair_embed_dims=pair_embed_dims,
    )
    kwargs = config.model_kwargs()
    assert kwargs["pair_embed_dims"] is None if pair_embed_dims is None else (
        kwargs["pair_embed_dims"] == pair_embed_dims
    )
