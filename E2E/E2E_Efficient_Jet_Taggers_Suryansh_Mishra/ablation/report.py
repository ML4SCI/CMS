"""Collect the ablation results into one comparison table.

Reads every ``<runs>/<run_name>/metrics.jsonl`` and reports each run's best
evaluation alongside the baseline, so the arms are compared on a single harness
rather than on numbers copied out of separate logs.

    python -m ablation.report --runs $SCRATCH/part_ablation/runs
    python -m ablation.report --runs runs --json results.json

Columns
-------
``params``     total parameters, and the ratio to the baseline
``step``       step at which the best validation accuracy occurred
``acc``        best validation accuracy
``auc``        legacy macro one-vs-rest AUC at that step
``auc_ovo``    macro one-vs-one AUC (the ParT definition) when the run logged it
``rej@paper``  background rejection ``1/eps_B`` vs QCD at each class's ParT Table 1
               working point (50% hadronic, 99% Hqql, 99.5% Tbl), one column per
               class. Rejection is never averaged across classes: the paper does
               not, and a mean that skips ``inf`` silently drops the strongest
               class (2026-09-09 audit, A11). ``>N`` marks zero background passes
               with ``N`` the 95% lower bound.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Dict, List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ablation.config import ARMS  # noqa: E402
from ablation.metrics import PAPER_WORKING_POINTS  # noqa: E402

__all__ = ["load_run", "build_table", "format_table"]


def load_run(run_dir: Path) -> Optional[dict]:
    """Summarize one run directory, or ``None`` if it has no evaluations yet."""
    metrics_path = run_dir / "metrics.jsonl"
    if not metrics_path.exists():
        return None

    params: Optional[int] = None
    evaluations: List[dict] = []
    last_train: Optional[dict] = None

    for line in metrics_path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            # A job killed mid-write can leave a truncated final line; the rest
            # of the file is still usable.
            continue
        event = record.get("event")
        if event == "start":
            params = record.get("params")
        elif event == "eval":
            evaluations.append(record)
        elif event == "train":
            last_train = record

    if not evaluations:
        return None

    best = max(evaluations, key=lambda r: r.get("accuracy") or -1)
    config_path = run_dir / "config.json"
    config = json.loads(config_path.read_text()) if config_path.exists() else {}

    return {
        "run": run_dir.name,
        "arm": config.get("arm", run_dir.name),
        "params": params,
        "step": best.get("step"),
        "accuracy": best.get("accuracy"),
        "auc": best.get("auc"),
        "auc_ovo": best.get("auc_ovo"),
        "per_class": best.get("per_class", {}),
        "num_evals": len(evaluations),
        "last_step": (last_train or {}).get("step"),
        "total_steps": config.get("total_steps"),
        "precision": config.get("precision"),
        "moe_config": config.get("moe_config"),
    }


def _paper_rejection_cell(stats: dict, class_name: str, width: int = 9) -> str:
    """One class's rejection at its ParT Table 1 working point, or ``-`` if the run predates it.

    Prefers the ``rej_paper`` key written since 2026-09-09. Older logs only have ``rej_50`` /
    ``rej_99`` at every class; for those the paper point is available only where it is 50% or
    99%, and ``Tbl`` (99.5%) is shown as ``-`` rather than substituted.
    """
    value = stats.get("rej_paper")
    if value is None:
        eps = PAPER_WORKING_POINTS.get(class_name)
        legacy_key = {0.5: "rej_50", 0.99: "rej_99"}.get(eps)
        value = stats.get(legacy_key) if legacy_key else None
    if value is None or not isinstance(value, (int, float)):
        return f"{'-':>{width}}"
    if math.isinf(value):
        bound = (stats.get("rejection") or {}).get(
            f"rej_{PAPER_WORKING_POINTS.get(class_name, 0) * 100:g}", {}
        ).get("rejection_lower_bound_95")
        return f"{'>' + format(bound, '.0f'):>{width}}" if bound else f"{'inf':>{width}}"
    if math.isnan(value):
        return f"{'nan':>{width}}"
    return f"{value:>{width}.1f}"


def build_table(runs_dir: Path) -> List[dict]:
    """Summarize every run directory under ``runs_dir``, in canonical arm order.

    Directories are *discovered by scanning*, not looked up by :data:`ARMS`
    name. Runs are written to ``output_dir/run_name`` (see
    ``AblationConfig.run_dir``), and ``run_name`` routinely differs from the
    arm: several runs share one arm (a rerun, a different MoE preset), and the
    submit scripts create ``urot_rope``, ``urot_mf``, ``n8_k6_v2``,
    ``baseline_ca`` and friends. Indexing by arm dropped every one of those
    silently -- no warning, no error, just a short table.

    Archived reruns (``*_bak*``) are excluded *by name*, which is what the
    exclusion actually calls for; an arm allowlist is the wrong instrument for
    it. Rows are ordered by :data:`ARMS` position so the canonical arms keep
    their familiar order, with unrecognized arms appended alphabetically rather
    than discarded.
    """
    runs_dir = Path(runs_dir)
    if not runs_dir.is_dir():
        return []

    summaries: List[dict] = []
    for run_dir in sorted(p for p in runs_dir.iterdir() if p.is_dir()):
        if "_bak" in run_dir.name:
            continue
        summary = load_run(run_dir)
        if summary is not None:
            summaries.append(summary)

    arm_rank = {arm: i for i, arm in enumerate(ARMS)}
    summaries.sort(key=lambda s: (arm_rank.get(s["arm"], len(ARMS)), s["run"]))
    return summaries


def format_table(rows: List[dict]) -> str:
    if not rows:
        return (
            "No completed evaluations found.\n"
            "Each run needs at least one eval; check <runs>/<run_name>/metrics.jsonl."
        )

    baseline = next((r for r in rows if r["arm"] == "baseline"), None)
    base_params = (baseline or {}).get("params")
    base_acc = (baseline or {}).get("accuracy")

    header = (
        f"{'run':<14} {'params':>10} {'x base':>7} {'step':>9} "
        f"{'acc':>8} {'d acc':>8} {'auc':>9} {'auc_ovo':>9} {'prog':>7}"
    )
    lines = [header, "-" * len(header)]

    for row in rows:
        # Label by run name: several runs can share an arm (a rerun, a different
        # MoE preset), and collapsing them to the arm name hides which is which.
        label = row["run"]
        params = row.get("params")
        ratio = (
            f"{params / base_params:.3f}"
            if params and base_params
            else "-"
        )
        delta = (
            f"{row['accuracy'] - base_acc:+.4f}"
            if row.get("accuracy") is not None and base_acc is not None
            else "-"
        )
        progress = (
            f"{100.0 * row['last_step'] / row['total_steps']:.0f}%"
            if row.get("last_step") and row.get("total_steps")
            else "-"
        )
        auc_ovo = row.get("auc_ovo")
        lines.append(
            f"{label:<14} "
            f"{(params or 0):>10,} "
            f"{ratio:>7} "
            f"{(row.get('step') or 0):>9,} "
            f"{(row.get('accuracy') or float('nan')):>8.4f} "
            f"{delta:>8} "
            f"{(row.get('auc') or float('nan')):>9.5f} "
            f"{(auc_ovo if auc_ovo is not None else float('nan')):>9.5f} "
            f"{progress:>7}"
        )

    incomplete = [
        r["run"]
        for r in rows
        if r.get("last_step")
        and r.get("total_steps")
        and r["last_step"] < r["total_steps"]
    ]
    if incomplete:
        lines += [
            "",
            "Note: still training -- " + ", ".join(incomplete) + ".",
            "Arms compared at different steps are not comparable; wait for the "
            "'prog' column to read 100%.",
        ]

    # Per-class rejection at each class's own ParT Table 1 working point -- one number per class,
    # never averaged across classes, matching how ParT / LLoCa tables are reported.
    class_names = [n for n in PAPER_WORKING_POINTS if any(n in r["per_class"] for r in rows)]
    class_names += sorted(
        {n for r in rows for n in r["per_class"]} - set(PAPER_WORKING_POINTS)
    )
    if class_names:
        lines += [
            "",
            "Background rejection 1/eps_B vs QCD at each class's ParT Table 1 working point "
            "(50% hadronic, 99% Hqql, 99.5% Tbl); '-' = not logged at that point:",
        ]
        lines.append(
            f"  {'run':<14}"
            + "".join(
                f"{n + '@' + format(PAPER_WORKING_POINTS.get(n, float('nan')) * 100, 'g'):>11}"
                for n in class_names
            )
        )
        for row in rows:
            cells = "".join(
                _paper_rejection_cell(row["per_class"].get(n, {}), n, width=11)
                for n in class_names
            )
            lines.append(f"  {row['run']:<14}{cells}")

    return "\n".join(lines)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument(
        "--runs", default="runs", help="directory containing per-arm run folders"
    )
    parser.add_argument("--json", help="also write the raw rows to this JSON file")
    args = parser.parse_args(argv)

    runs_dir = Path(args.runs)
    if not runs_dir.is_dir():
        parser.error(f"no such directory: {runs_dir}")

    rows = build_table(runs_dir)
    print(format_table(rows))

    if args.json:
        Path(args.json).write_text(json.dumps(rows, indent=2))
        print(f"\nwrote {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
