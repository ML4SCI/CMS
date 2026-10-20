"""Recompute validation metrics from saved eval prediction files.

Training writes ``<run_dir>/predictions/step_XXXXXXX.npz`` (fp16 probs + uint8
labels) at each eval.  This script re-runs :func:`ablation.metrics.summarize` so
you can change ``rejection_efficiencies`` without retraining.

    python -m ablation.recompute_metrics --run-dir $SCRATCH/part_ablation/runs/baseline
    python -m ablation.recompute_metrics --run-dir runs/baseline --efficiencies 0.3,0.5,0.95,0.99
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ablation.metrics import format_summary, summarize  # noqa: E402
from ablation.provenance import (  # noqa: E402
    load_eval_predictions,
    predictions_dir,
    stored_probs_dtype,
)

__all__ = ["main"]


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument(
        "--run-dir",
        required=True,
        help="run directory containing predictions/step_*.npz",
    )
    parser.add_argument(
        "--efficiencies",
        default="0.5,0.99",
        help="comma-separated signal efficiencies for background rejection "
        "(0.3 and each class's ParT Table 1 point are always added)",
    )
    parser.add_argument("--json", help="write all recomputed summaries to this file")
    args = parser.parse_args(argv)

    run_dir = Path(args.run_dir)
    pred_dir = predictions_dir(run_dir)
    if not pred_dir.is_dir():
        parser.error(f"no predictions directory: {pred_dir}")

    efficiencies = tuple(float(x) for x in args.efficiencies.split(",") if x)
    rows = []
    for path in sorted(pred_dir.glob("step_*.npz")):
        step, probs, labels = load_eval_predictions(path)
        dtype = str(stored_probs_dtype(path))
        summary = summarize(probs, labels, efficiencies)
        summary["step"] = step
        summary["stored_probs_dtype"] = dtype
        rows.append(summary)
        print(f"\n=== step {step:,}  (archived probs: {dtype}) ===")
        if dtype == "float16":
            print(
                "  NOTE: fp16 archive (~3 significant digits). Tied scores in the tail are an "
                "artefact of storage; read rej_99 / rej_99.5 with their achieved_signal_efficiency."
            )
        print(format_summary(summary))

    if args.json:
        Path(args.json).write_text(json.dumps(rows, indent=2))
        print(f"\nwrote {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
