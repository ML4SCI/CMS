"""Experiment directory layout and YAML config loading.

Run directory convention (created under experiments/):
    experiments/{YYYY-MM-DD}_{dataset}_{scale}x_{run_name}/
        checkpoints/   best.pt, latest.pt
        figures/       sample_epoch_{N}.png, metrics.png
        config.yaml    frozen copy of the resolved run config
        eval.json      final eval metrics
        metrics.jsonl  per-epoch metric log

CLI args override YAML values when both are present (handled in train.py).
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any

import yaml


@dataclass
class ExperimentPaths:
    root: Path
    checkpoints: Path
    figures: Path
    config_yaml: Path
    eval_json: Path
    metrics_jsonl: Path

    @property
    def best_ckpt(self) -> Path:
        return self.checkpoints / "best.pt"

    @property
    def latest_ckpt(self) -> Path:
        return self.checkpoints / "latest.pt"


def make_experiment_dir(
    experiments_root: Path,
    dataset_name: str,
    scale: int,
    run_name: str,
    run_date: date | None = None,
) -> ExperimentPaths:
    """Create experiments/{date}_{dataset}_{scale}x_{run_name}/ and subdirs."""
    d = (run_date or date.today()).isoformat()
    name = f"{d}_{dataset_name}_{scale}x_{run_name}"
    root = experiments_root / name
    checkpoints = root / "checkpoints"
    figures = root / "figures"
    checkpoints.mkdir(parents=True, exist_ok=True)
    figures.mkdir(parents=True, exist_ok=True)
    return ExperimentPaths(
        root=root,
        checkpoints=checkpoints,
        figures=figures,
        config_yaml=root / "config.yaml",
        eval_json=root / "eval.json",
        metrics_jsonl=root / "metrics.jsonl",
    )


def load_config(path: Path) -> dict[str, Any]:
    """Load a YAML config file into a plain dict."""
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    return data or {}


def save_config(path: Path, config: dict[str, Any]) -> None:
    """Write the resolved config as YAML (frozen copy in the run dir)."""
    path.write_text(yaml.safe_dump(config, sort_keys=True), encoding="utf-8")
