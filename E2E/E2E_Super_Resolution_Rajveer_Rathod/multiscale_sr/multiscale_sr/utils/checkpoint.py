from __future__ import annotations

from torch import nn


def load_generator_state(generator: nn.Module, state_dict: dict) -> None:
    """Load a generator state_dict, tolerating missing keys from older checkpoints.

    Params added to ``Generator`` after a checkpoint was saved (e.g. ``lr_skip_alpha``)
    would otherwise fail strict loading; missing ones simply keep their fresh-init
    default (``lr_skip_alpha`` inits to 1.0, matching the old fixed-add behavior).
    """
    missing, unexpected = generator.load_state_dict(state_dict, strict=False)
    if missing:
        print(f"[ckpt] missing keys (using init defaults): {missing}")
    if unexpected:
        print(f"[ckpt] unexpected keys (ignored): {unexpected}")
