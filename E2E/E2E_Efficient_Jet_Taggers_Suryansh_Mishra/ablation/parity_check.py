"""Compare two runs' training trajectories to verify part_kernels parity.

The part_kernels-optimized model must train as close to the pristine weaver
baseline as floating-point noise allows.  When the two runs share the same
seed, config and data order (only ``use_part_kernels`` differs), they consume
identical batches and identical dropout masks, so any divergence in the
per-step loss/accuracy comes from the fused kernels' numerics -- exactly the
property a smoke-level verification should gate on.

    python -m ablation.parity_check \
        --base runs/smoke/baseline --optimized runs/smoke/baseline_part

Exit code 0 on PASS, 1 on FAIL (any tolerance exceeded), so a Slurm wrapper can
use it as a gate.  Every threshold is a CLI flag because a meaningful value
depends on the run length: in a 60-step smoke the LR has barely left warmup, so
the loss sits near ln(C) and differences stay tiny, while a longer run shows
the natural amplification of kernel noise through the optimizer.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Dict, List, Optional

__all__ = ["load_events", "train_series", "eval_records", "compare", "main"]


def load_events(run_dir: Path) -> List[dict]:
    """Read ``metrics.jsonl`` as a list of records, skipping truncated lines."""
    path = run_dir / "metrics.jsonl"
    if not path.exists():
        raise FileNotFoundError(
            f"no metrics.jsonl under {run_dir} -- did that run complete?"
        )
    events = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            events.append(json.loads(line))
        except json.JSONDecodeError:
            # A job killed mid-write can leave a truncated final line.
            continue
    return events


def train_series(events: List[dict]) -> Dict[int, dict]:
    """Map step -> last train record at that step."""
    by_step: Dict[int, dict] = {}
    for ev in events:
        if ev.get("event") == "train":
            by_step[int(ev["step"])] = ev
    return by_step


def eval_records(events: List[dict]) -> List[dict]:
    return [ev for ev in events if ev.get("event") == "eval"]


def _finite_abs_diff(lhs, rhs) -> Optional[float]:
    """Abs difference of two finite numbers, or ``None`` if either is missing/NaN."""
    if lhs is None or rhs is None:
        return None
    lhs, rhs = float(lhs), float(rhs)
    if not (math.isfinite(lhs) and math.isfinite(rhs)):
        return None
    return abs(lhs - rhs)


def compare(
    base_dir: Path,
    optimized_dir: Path,
    loss_tol: float,
    loss_max_tol: float,
    acc_tol: float,
    eval_acc_tol: float,
    eval_auc_tol: float,
) -> int:
    base = train_series(load_events(base_dir))
    optimized = train_series(load_events(optimized_dir))

    steps = sorted(set(base) & set(optimized))
    if not steps:
        print(f"error: no aligned training steps between {base_dir} and {optimized_dir}")
        return 1

    print(f"{'step':>8}  {'base loss':>9}  {'opt loss':>9}  "
          f"{'|d loss|':>9}  {'base acc':>9}  {'opt acc':>9}  {'|d acc|':>9}")
    print("-" * 66)

    dloss = []
    dacc = []
    for step in steps:
        b, o = base[step], optimized[step]
        dl = abs(b["loss"] - o["loss"])
        da = abs(b.get("accuracy", 0.0) - o.get("accuracy", 0.0))
        dloss.append(dl)
        dacc.append(da)
        print(f"{step:>8}  {b['loss']:>9.5f}  {o['loss']:>9.5f}  {dl:>9.2e}  "
              f"{b.get('accuracy', float('nan')):>9.5f}  "
              f"{o.get('accuracy', float('nan')):>9.5f}  {da:>9.2e}")

    mean_dl = sum(dloss) / len(dloss)
    max_dl = max(dloss)
    mean_da = sum(dacc) / len(dacc)
    max_da = max(dacc)
    print("-" * 66)
    print(f"train: {len(steps)} logged steps aligned")
    print(f"  loss |d|  mean={mean_dl:.2e}  max={max_dl:.2e}  (tolerance mean<={loss_tol:g}, max<={loss_max_tol:g})")
    print(f"  acc  |d|  mean={mean_da:.2e}  max={max_da:.2e}  (tolerance max<={acc_tol:g})")

    # Eval records: compare every evaluation in the same order by step.
    # NaN metrics (e.g. AUC when the tiny smoke validation sample lacks some
    # classes) are not a divergence -- they are reported as n/a and skipped.
    base_evals = {int(ev["step"]): ev for ev in eval_records(load_events(base_dir))}
    opt_evals = {int(ev["step"]): ev for ev in eval_records(load_events(optimized_dir))}
    evals_ok = True
    for step in sorted(set(base_evals) & set(opt_evals)):
        b, o = base_evals[step], opt_evals[step]
        dacc = _finite_abs_diff(b.get("accuracy"), o.get("accuracy"))
        dauc = _finite_abs_diff(b.get("auc"), o.get("auc"))
        bad = []
        if dacc is not None and dacc > eval_acc_tol:
            bad.append(f"acc |d|={dacc:.4f} > {eval_acc_tol:g}")
        if dauc is not None and dauc > eval_auc_tol:
            bad.append(f"auc |d|={dauc:.4f} > {eval_auc_tol:g}")
        evals_ok &= not bad
        flag = "BAD" if bad else ("n/a" if dacc is None and dauc is None else "OK")
        acc_cell = (f"{b['accuracy']:.4f} vs {o['accuracy']:.4f} "
                    f"(|d|={dacc:.4f})") if dacc is not None else "n/a"
        auc_cell = (f"{b['auc']:.4f} vs {o['auc']:.4f} "
                    f"(|d|={dauc:.4f})") if dauc is not None else "n/a"
        print(f"eval @{step:>6}: acc {acc_cell}, auc {auc_cell}  [{flag:3s}]")

    ok = (
        mean_dl <= loss_tol
        and max_dl <= loss_max_tol
        and max_da <= acc_tol
        and evals_ok
    )
    print("\nVERDICT:", "PASS -- part_optimized trains as close to baseline "
                        "as floating-point noise allows." if ok
                        else "FAIL -- part_optimized diverges from baseline "
                             "beyond the configured tolerances.")
    return 0 if ok else 1


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--base", required=True, help="pristine baseline run dir")
    parser.add_argument("--optimized", required=True, help="part_kernels run dir")
    parser.add_argument("--loss-tol", type=float, default=3e-2,
                        help="max allowed mean |Δloss| per step")
    parser.add_argument("--loss-max-tol", type=float, default=1e-1,
                        help="max allowed single-step |Δloss|")
    parser.add_argument("--acc-tol", type=float, default=5e-2,
                        help="max allowed single-step |Δaccuracy|")
    parser.add_argument("--eval-acc-tol", type=float, default=5e-2,
                        help="max allowed |Δaccuracy| on a validation eval")
    parser.add_argument("--eval-auc-tol", type=float, default=5e-2,
                        help="max allowed |Δauc| on a validation eval")
    args = parser.parse_args(argv)
    return compare(
        Path(args.base),
        Path(args.optimized),
        args.loss_tol,
        args.loss_max_tol,
        args.acc_tol,
        args.eval_acc_tol,
        args.eval_auc_tol,
    )


if __name__ == "__main__":
    raise SystemExit(main())
