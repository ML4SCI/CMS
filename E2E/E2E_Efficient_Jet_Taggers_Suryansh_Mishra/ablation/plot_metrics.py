"""Plot training and validation curves from ``metrics.jsonl``.

Reads every ``<runs>/<arm>/metrics.jsonl`` and writes PNG summaries for one
arm or for all arms overlaid.

    python -m ablation.plot_metrics --runs $SCRATCH/part_ablation/runs
    python -m ablation.plot_metrics --runs runs --arm baseline --out plots/
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ablation.config import ARMS  # noqa: E402
from ablation.metrics import JETCLASS_LABELS  # noqa: E402

__all__ = ["load_series", "plot_arm", "plot_comparison"]


def _read_metrics(path: Path) -> Tuple[List[dict], List[dict], Optional[dict]]:
    """Return ``(train_rows, eval_rows, start_row)`` from one metrics file."""
    train_rows: List[dict] = []
    eval_rows: List[dict] = []
    start_row: Optional[dict] = None
    if not path.exists():
        return train_rows, eval_rows, start_row

    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue
        event = record.get("event")
        if event == "start":
            start_row = record
        elif event == "train":
            train_rows.append(record)
        elif event == "eval":
            eval_rows.append(record)
    return train_rows, eval_rows, start_row


def load_series(run_dir: Path) -> Optional[dict]:
    """Load time series for one run directory."""
    train_rows, eval_rows, start_row = _read_metrics(run_dir / "metrics.jsonl")
    if not train_rows and not eval_rows:
        return None

    config = {}
    config_path = run_dir / "config.json"
    if config_path.exists():
        config = json.loads(config_path.read_text())

    def _col(rows: Sequence[dict], key: str) -> np.ndarray:
        return np.asarray([r.get(key, float("nan")) for r in rows], dtype=np.float64)

    class_counts = [r.get("class_counts") for r in train_rows if r.get("class_counts")]
    class_min = (
        np.asarray([min(cc) for cc in class_counts], dtype=np.float64)
        if class_counts
        else np.array([])
    )
    class_max = (
        np.asarray([max(cc) for cc in class_counts], dtype=np.float64)
        if class_counts
        else np.array([])
    )

    per_class_keys = sorted(
        {name for row in eval_rows for name in (row.get("per_class") or {})}
    )
    per_class_rej50 = {
        name: _col(
            [
                {
                    "v": (row.get("per_class") or {})
                    .get(name, {})
                    .get("rej_50", float("nan"))
                }
                for row in eval_rows
            ],
            "v",
        )
        for name in per_class_keys
    }
    per_class_rej99 = {
        name: _col(
            [
                {
                    "v": (row.get("per_class") or {})
                    .get(name, {})
                    .get("rej_99", float("nan"))
                }
                for row in eval_rows
            ],
            "v",
        )
        for name in per_class_keys
    }
    per_class_auc = {
        name: _col(
            [
                {
                    "v": (row.get("per_class") or {})
                    .get(name, {})
                    .get("auc_vs_qcd", float("nan"))
                }
                for row in eval_rows
            ],
            "v",
        )
        for name in per_class_keys
    }

    return {
        "run": run_dir.name,
        "arm": config.get("arm", run_dir.name),
        "experiment": (start_row or {}).get("experiment") or config.get("experiment"),
        "git": (start_row or {}).get("git_commit_short"),
        "total_steps": config.get("total_steps"),
        "train": {
            "step": _col(train_rows, "step"),
            "loss": _col(train_rows, "loss"),
            "accuracy": _col(train_rows, "accuracy"),
            "lr": _col(train_rows, "lr"),
            "jets_per_sec": _col(train_rows, "jets_per_sec"),
            "class_min": class_min,
            "class_max": class_max,
        },
        "eval": {
            "step": _col(eval_rows, "step"),
            "accuracy": _col(eval_rows, "accuracy"),
            "auc": _col(eval_rows, "auc"),
            "num_jets": _col(eval_rows, "num_jets"),
            "per_class_rej50": per_class_rej50,
            "per_class_rej99": per_class_rej99,
            "per_class_auc": per_class_auc,
        },
    }


def _finite_xy(x: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    mask = np.isfinite(x) & np.isfinite(y)
    return x[mask], y[mask]


def _style_axis(ax, title: str, xlabel: str, ylabel: str) -> None:
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.3)


def plot_arm(series: dict, out_path: Path) -> None:
    """Write a multi-panel PNG for one arm."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    train = series["train"]
    eval_ = series["eval"]

    fig, axes = plt.subplots(3, 2, figsize=(14, 12), constrained_layout=True)
    title = f"{series['run']} ({series.get('experiment') or 'ablation'})"
    if series.get("git"):
        title += f"  git={series['git']}"
    fig.suptitle(title, fontsize=13)

    ax = axes[0, 0]
    x, y = _finite_xy(train["step"], train["loss"])
    ax.plot(x, y, linewidth=1.2)
    _style_axis(ax, "Train loss", "step", "loss")

    ax = axes[0, 1]
    x, y = _finite_xy(train["step"], train["accuracy"])
    ax.plot(x, y, linewidth=1.2, label="train")
    x, y = _finite_xy(eval_["step"], eval_["accuracy"])
    ax.plot(x, y, "o-", linewidth=1.2, markersize=4, label="val")
    ax.legend()
    _style_axis(ax, "Accuracy", "step", "accuracy")

    ax = axes[1, 0]
    x, y = _finite_xy(eval_["step"], eval_["auc"])
    ax.plot(x, y, "o-", linewidth=1.2, markersize=4)
    _style_axis(ax, "Validation macro AUC", "step", "AUC")

    ax = axes[1, 1]
    x, y = _finite_xy(train["step"], train["lr"])
    ax.plot(x, y, linewidth=1.2)
    _style_axis(ax, "Learning rate", "step", "lr")

    ax = axes[2, 0]
    x, y = _finite_xy(train["step"], train["jets_per_sec"])
    ax.plot(x, y, linewidth=1.2)
    _style_axis(ax, "Throughput", "step", "jets/s")

    ax = axes[2, 1]
    if train["class_min"].size:
        x, ymin = _finite_xy(train["step"], train["class_min"])
        _, ymax = _finite_xy(train["step"], train["class_max"])
        ax.plot(x, ymin, linewidth=1.2, label="min class count / log window")
        ax.plot(x, ymax, linewidth=1.2, label="max class count / log window")
        ax.legend(fontsize=8)
    _style_axis(ax, "Class balance (per rank, log window)", "step", "jets")

    fig.savefig(out_path, dpi=150)
    plt.close(fig)

    if eval_["per_class_rej50"]:
        n_class = len(eval_["per_class_rej50"])
        ncols = 5
        nrows = math.ceil(n_class / ncols)
        fig, axes = plt.subplots(nrows, ncols, figsize=(3.2 * ncols, 2.8 * nrows))
        axes = np.atleast_1d(axes).ravel()
        steps = eval_["step"]
        for ax, name in zip(axes, sorted(eval_["per_class_rej50"])):
            y50 = eval_["per_class_rej50"][name]
            y99 = eval_["per_class_rej99"].get(name, np.full_like(y50, np.nan))
            x, y = _finite_xy(steps, y50)
            ax.plot(x, y, "o-", markersize=3, label="rej@50%")
            x, y = _finite_xy(steps, y99)
            ax.plot(x, y, "o-", markersize=3, label="rej@99%")
            ax.set_title(name)
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=7)
        for ax in axes[len(eval_["per_class_rej50"]) :]:
            ax.axis("off")
        fig.suptitle(f"{series['run']}: background rejection vs QCD", fontsize=12)
        fig.savefig(out_path.with_name(out_path.stem + "_rejection.png"), dpi=150)
        plt.close(fig)


def plot_comparison(all_series: Sequence[dict], out_path: Path) -> None:
    """Overlay the main curves for every arm that has data."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 2, figsize=(14, 10), constrained_layout=True)
    fig.suptitle("Ablation comparison", fontsize=13)

    panels = [
        (axes[0, 0], "train", "loss", "Train loss"),
        (axes[0, 1], "train", "accuracy", "Train accuracy"),
        (axes[1, 0], "eval", "accuracy", "Validation accuracy"),
        (axes[1, 1], "eval", "auc", "Validation macro AUC"),
    ]
    for ax, split, key, title in panels:
        for series in all_series:
            block = series[split]
            x, y = _finite_xy(block["step"], block[key])
            if x.size == 0:
                continue
            ax.plot(x, y, linewidth=1.2, label=series["run"])
        ax.legend(fontsize=8)
        _style_axis(ax, title, "step", key)

    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def discover_series(runs_dir: Path, arms: Optional[Sequence[str]] = None) -> List[dict]:
    """Load metrics for canonical ablation arms only.

    By default only the six ``ARMS`` folder names are read. Archived reruns
    (``*_bak*``) are ignored unless explicitly passed via ``--arm``.
    """
    wanted = list(arms) if arms else list(ARMS)
    ordered: List[dict] = []
    for name in wanted:
        series = load_series(runs_dir / name)
        if series is not None:
            ordered.append(series)
    return ordered


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument(
        "--runs", default="runs", help="directory containing per-arm run folders"
    )
    parser.add_argument(
        "--out",
        default="",
        help="output directory for PNGs (default: <runs>/plots)",
    )
    parser.add_argument(
        "--arm",
        action="append",
        dest="arms",
        help="plot only these run/arm folder names (repeatable)",
    )
    parser.add_argument(
        "--html",
        nargs="?",
        const="",
        default=None,
        help="also write interactive Plotly HTML (default: logs/ablation-plots.html)",
    )
    args = parser.parse_args(argv)

    runs_dir = Path(args.runs)
    if not runs_dir.is_dir():
        parser.error(f"no such directory: {runs_dir}")

    out_dir = Path(args.out) if args.out else runs_dir / "plots"
    series_list = discover_series(runs_dir, args.arms)
    if not series_list:
        print(f"No metrics found under {runs_dir}", file=sys.stderr)
        return 1

    keep = {f"{series['run']}.png" for series in series_list}
    keep |= {f"{series['run']}_rejection.png" for series in series_list}
    keep.add("comparison.png")
    if out_dir.exists():
        for stale in out_dir.glob("*.png"):
            if stale.name not in keep:
                stale.unlink()
                print(f"removed stale {stale}")

    for series in series_list:
        out_path = out_dir / f"{series['run']}.png"
        plot_arm(series, out_path)
        print(f"wrote {out_path}")
        rej_path = out_path.with_name(out_path.stem + "_rejection.png")
        if rej_path.exists():
            print(f"wrote {rej_path}")

    comparison_path = out_dir / "comparison.png"
    plot_comparison(series_list, comparison_path)
    print(f"wrote {comparison_path}")

    if args.html is not None:
        from ablation.export_plotly import export_html

        html_path = Path(args.html) if args.html else Path("logs/ablation-plots.html")
        export_html(runs_dir, html_path, arms=args.arms)
        print(f"wrote {html_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
