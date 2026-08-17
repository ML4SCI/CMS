"""Thin Weights & Biases wrapper with graceful degradation.

If --no-wandb is passed, or wandb is not installed / not logged in, every method
becomes a no-op so training never fails because of logging. Credentials are read
from a .env file in the package root via python-dotenv (WANDB_API_KEY, optional
WANDB_ENTITY) — matching the SwinIR convention.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np


class WandbLogger:
    def __init__(
        self,
        enabled: bool,
        project: str,
        run_name: str,
        config: dict[str, Any],
        entity: str | None = None,
        dir: Path | None = None,
    ) -> None:
        self.enabled = enabled
        self._run = None
        self._wandb = None
        if not enabled:
            return

        try:
            import wandb
        except ImportError:
            print("[wandb] not installed — logging disabled. `pip install wandb` to enable.")
            self.enabled = False
            return

        self._wandb = wandb
        try:
            self._run = wandb.init(
                project=project,
                entity=entity,
                name=run_name,
                config=config,
                dir=str(dir) if dir is not None else None,
            )
            print(f"[wandb] run initialized: {self._run.url}")
        except Exception as exc:  # network down, not logged in, etc.
            print(f"[wandb] init failed ({exc}); continuing without logging.")
            self.enabled = False
            self._run = None

    def log(self, metrics: dict[str, Any], step: int | None = None) -> None:
        if not self.enabled or self._run is None:
            return
        self._wandb.log(metrics, step=step)

    def log_image(self, key: str, image: np.ndarray, step: int | None = None, caption: str | None = None) -> None:
        if not self.enabled or self._run is None:
            return
        self._wandb.log({key: self._wandb.Image(image, caption=caption)}, step=step)

    def log_artifact(self, path: Path, name: str, type: str = "model", aliases: list[str] | None = None) -> None:
        if not self.enabled or self._run is None:
            return
        try:
            art = self._wandb.Artifact(name=name, type=type)
            art.add_file(str(path))
            self._run.log_artifact(art, aliases=aliases or [])
        except Exception as exc:
            print(f"[wandb] artifact upload failed ({exc}); skipping.")

    def set_summary(self, summary: dict[str, Any]) -> None:
        if not self.enabled or self._run is None:
            return
        for k, v in summary.items():
            self._run.summary[k] = v

    def finish(self) -> None:
        if not self.enabled or self._run is None:
            return
        self._run.finish()


def load_env(package_root: Path) -> None:
    """Load WANDB_* vars from <package_root>/.env if python-dotenv is available."""
    env_path = package_root / ".env"
    if not env_path.exists():
        return
    try:
        from dotenv import load_dotenv

        load_dotenv(env_path)
    except ImportError:
        pass
