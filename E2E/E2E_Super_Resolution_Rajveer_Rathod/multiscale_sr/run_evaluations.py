#!/usr/bin/env python3
"""Batch classification-eval harness for multiscale-SR checkpoints.

Runs the existing ``classification_eval.py`` on every trained checkpoint,
stores each run's figures + JSON in a DATE-organized tree under
``evaluations/<EVAL_DATE>/`` (kept separate from ``experiments/``), writes a
per-run ``REPORT.md``, then aggregates all runs into ``SUMMARY.md`` +
``master_metrics.csv``.

Does not retrain and does not modify any baseline code — it only shells out to
classification_eval.py with ``--out-dir`` redirected, then reads the emitted
JSON. One failing run is logged and skipped; it never aborts the batch.

Each run's ``val_ratio`` is read from its own ``config.yaml`` and forwarded
explicitly as ``--val-ratio`` — classification_eval.py's "test" split is a
row-half of the held-out file group determined by that ratio, so passing the
wrong one would silently score against a different held-out set than the one
the checkpoint actually trained against.

    cd multiscale_sr
    python run_evaluations.py --data-dir ../datasets
"""
from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
import time
from datetime import date
from pathlib import Path

HERE = Path(__file__).resolve().parent


# --------------------------------------------------------------------------- #
# discovery                                                                    #
# --------------------------------------------------------------------------- #
def _reached_epoch(run_dir: Path) -> int | None:
    mj = run_dir / "metrics.jsonl"
    if not mj.exists():
        return None
    last = None
    for line in mj.read_text().splitlines():
        line = line.strip()
        if line:
            last = line
    if last is None:
        return None
    try:
        return int(json.loads(last).get("epoch"))
    except Exception:
        return None


def _run_meta(run_dir: Path) -> dict:
    cfg = run_dir / "config.yaml"
    scale = epochs = None
    val_ratio = 0.33  # must match classification_eval.py's default as a fallback
    if cfg.exists():
        for ln in cfg.read_text().splitlines():
            if ln.startswith("scale:"):
                scale = int(ln.split(":")[1].strip())
            elif ln.startswith("epochs:"):
                epochs = int(ln.split(":")[1].strip())
            elif ln.startswith("val_ratio:"):
                val_ratio = float(ln.split(":")[1].strip())
    return {
        "name": run_dir.name,
        "scale": scale,
        "cfg_epochs": epochs,
        "val_ratio": val_ratio,
        "reached_epoch": _reached_epoch(run_dir),
        "checkpoint": str(run_dir / "checkpoints" / "best.pt"),
    }


def discover_runs(experiments: Path, include_smoke: bool) -> list[dict]:
    runs = []
    for run_dir in sorted(experiments.iterdir()):
        if not (run_dir / "checkpoints" / "best.pt").exists():
            continue
        if not include_smoke and "smoke" in run_dir.name:
            continue
        runs.append(_run_meta(run_dir))
    return runs


def training_status(meta: dict) -> str:
    """Coarse flag from epoch budget alone (refined after eval by recovery)."""
    reached, cfg = meta.get("reached_epoch"), meta.get("cfg_epochs")
    if reached is not None and cfg is not None and reached < cfg:
        return "partial"
    if (reached or 0) <= 5:
        return "undertrained"
    return "trained"


# --------------------------------------------------------------------------- #
# running one eval                                                             #
# --------------------------------------------------------------------------- #
def run_one(meta: dict, args, out_run_dir: Path) -> dict:
    classif_dir = out_run_dir / "classification"
    classif_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_run_dir / "eval.log"

    cmd = [
        sys.executable,
        str(HERE / "classification_eval.py"),
        "--checkpoint", meta["checkpoint"],
        "--data-dir", str(args.data_dir),
        "--out-dir", str(classif_dir),
        "--seed", str(args.seed),
        "--max-samples", str(args.max_samples),
        "--tagger-epochs", str(args.tagger_epochs),
        # Must match the value this checkpoint was actually trained with, else
        # "test" here would be drawn from a different held-out file group than
        # the one the run's config.yaml recorded (a real leak risk otherwise).
        "--val-ratio", str(meta["val_ratio"]),
    ]
    t0 = time.time()
    with log_path.open("w") as lf:
        proc = subprocess.run(cmd, stdout=lf, stderr=subprocess.STDOUT)
    dt = time.time() - t0
    return {"exit_code": proc.returncode, "seconds": round(dt, 1), "log": str(log_path)}


# --------------------------------------------------------------------------- #
# reporting                                                                    #
# --------------------------------------------------------------------------- #
def _load_result(out_run_dir: Path) -> dict | None:
    p = out_run_dir / "classification" / "classification_eval.json"
    if not p.exists():
        return None
    return json.loads(p.read_text())


def refine_status(meta_status: str, recovery: float | None) -> str:
    if recovery is not None and recovery < 0:
        return "undertrained (recovery<0: SR worse than bicubic)"
    if meta_status == "partial":
        return "partial (stopped before configured epochs)"
    if meta_status == "undertrained":
        return "undertrained (few epochs)"
    return "trained"


def write_report(meta: dict, res: dict, status: str, out_run_dir: Path) -> None:
    p = res["primary_fixed_hr_tagger"]
    auc, rej = p["auc"], p["bkg_rej_at_50"]
    eff = p["tagging_efficiency_sr_over_hr"]
    rec = p["recovery_fraction_lr_to_hr"]
    agree = p["score_agreement_vs_hr"]["sr"]
    ece = p["calibration_ece"]

    md = []
    md.append(f"# Classification-Eval Report — {meta['name']}\n")
    md.append(
        f"- **Scale:** {meta['scale']}x  |  **Config epochs:** {meta['cfg_epochs']}  |  "
        f"**Reached epoch:** {meta['reached_epoch']}\n"
        f"- **Checkpoint:** `{meta['checkpoint']}`\n"
        f"- **Status:** {status}\n"
        f"- **Tagger:** fixed HR-trained, {res['n_train']} train / {res['n_test']} test\n"
    )
    md.append("\n## Headline metrics (fixed HR tagger)\n")
    md.append("| Source | AUC | 1/εB @ 50% |\n|---|---|---|")
    md.append(f"| HR (truth) | {auc['hr']:.3f} | {rej['hr']:.2f} |")
    md.append(f"| LR (bicubic) | {auc['lr']:.3f} | {rej['lr']:.2f} |")
    md.append(f"| **SR (model)** | **{auc['sr']:.3f}** | **{rej['sr']:.2f}** |")
    md.append("")
    md.append(f"- **Tagging efficiency (AUC_SR/AUC_HR):** {eff*100:.1f}%")
    md.append(f"- **Recovery (LR→HR gap closed):** {rec*100:.1f}%")
    md.append(
        f"- **SR↔HR score agreement:** Pearson r={agree['pearson_r']:.3f}, "
        f"mean|Δ|={agree['mean_abs_score_diff']:.3f}"
    )
    md.append(f"- **Calibration ECE:** HR={ece['hr']:.3f} / LR={ece['lr']:.3f} / SR={ece['sr']:.3f}")

    md.append("\n## Interpretation\n")
    bullets = []
    if auc["sr"] >= auc["lr"]:
        bullets.append(
            f"SR AUC ({auc['sr']:.3f}) is **above the bicubic LR floor** "
            f"({auc['lr']:.3f}) — the model adds taggable information."
        )
    else:
        bullets.append(
            f"⚠️ SR AUC ({auc['sr']:.3f}) is **below the bicubic LR floor** "
            f"({auc['lr']:.3f}) — SR degrades taggability. Undertrained / not converged."
        )
    if rec >= 0:
        bullets.append(f"Recovery is **positive** ({rec*100:.1f}%): SR closes the LR→HR gap.")
    else:
        bullets.append(
            f"⚠️ Recovery is **negative** ({rec*100:.1f}%): needs more epochs, not a model fault."
        )
    if agree["pearson_r"] >= 0.7:
        bullets.append(
            f"SR fools the HR tagger like HR does (per-sample r={agree['pearson_r']:.3f})."
        )
    else:
        bullets.append(
            f"Per-sample SR↔HR agreement is weak (r={agree['pearson_r']:.3f})."
        )
    md += [f"- {b}" for b in bullets]

    md.append("\n## Figures\n")
    for fig in (
        "roc_overlay.png", "auc_summary_bar.png", "score_distributions.png",
        "confusion_matrices.png", "efficiency_vs_threshold.png",
        "calibration.png", "score_agreement.png",
        "energy_correlation.png", "pt_correlation.png",
    ):
        md.append(f"- [{fig}](classification/{fig})")

    md.append(
        "\n## Caveat\n"
        "Absolute AUC is capped by the small tagger (few epochs, ~4k samples); "
        "the **efficiency ratio** and **recovery** are the headline, not absolute AUC.\n"
    )
    (out_run_dir / "REPORT.md").write_text("\n".join(md))


def write_summary(rows: list[dict], out_date_dir: Path, args) -> None:
    # master_metrics.csv
    fields = [
        "run", "scale", "cfg_epochs", "reached_epoch", "status",
        "auc_hr", "auc_lr", "auc_sr", "efficiency", "recovery",
        "bkg_rej_sr_at_50", "sr_hr_pearson", "ece_sr",
    ]
    with (out_date_dir / "master_metrics.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in fields})

    ok = [r for r in rows if r.get("auc_sr") is not None]
    ok.sort(key=lambda r: (r["scale"] or 0, r["cfg_epochs"] or 0))

    md = [f"# Cross-Run Evaluation Summary — {out_date_dir.name}\n"]
    md.append(f"Evaluated {len(rows)} checkpoint(s); {len(ok)} succeeded. "
              f"Same parquet, same downsampling, same fixed HR tagger, seed={args.seed}.\n")

    md.append("## Master table (fixed HR tagger)\n")
    md.append("| Run | Scale | Cfg ep | Reached | AUC_SR | Efficiency | Recovery | Status |")
    md.append("|---|---|---|---|---|---|---|---|")
    for r in ok:
        md.append(
            f"| {r['run']} | {r['scale']}x | {r['cfg_epochs']} | {r['reached_epoch']} | "
            f"{r['auc_sr']:.3f} | {r['efficiency']*100:.1f}% | {r['recovery']*100:.1f}% | {r['status']} |"
        )

    # best per scale by efficiency
    md.append("\n## Best checkpoint per scale (by tagging efficiency)\n")
    for sc in sorted({r["scale"] for r in ok if r["scale"] is not None}):
        cand = [r for r in ok if r["scale"] == sc]
        best = max(cand, key=lambda r: r["efficiency"])
        md.append(f"- **{sc}x:** `{best['run']}` — efficiency {best['efficiency']*100:.1f}%, "
                  f"recovery {best['recovery']*100:.1f}% ({best['status']})")

    # AUC_HR consistency assertion
    md.append("\n## Sanity: AUC_HR consistency across runs\n")
    hrs = [r["auc_hr"] for r in ok if r.get("auc_hr") is not None]
    if hrs:
        spread = max(hrs) - min(hrs)
        flag = "OK" if spread <= 0.02 else "⚠️ >0.02 — check seed/data"
        md.append(f"- AUC_HR range across runs: {min(hrs):.3f}–{max(hrs):.3f} "
                  f"(spread {spread:.3f}) — {flag}. A fixed HR tagger should give ~identical "
                  f"AUC_HR regardless of the SR model.")

    # epoch effect per scale
    md.append("\n## Epoch effect (does more training help?)\n")
    for sc in sorted({r["scale"] for r in ok if r["scale"] is not None}):
        cand = sorted([r for r in ok if r["scale"] == sc], key=lambda r: r["cfg_epochs"] or 0)
        if len(cand) >= 2:
            trail = " → ".join(f"{r['cfg_epochs']}ep:{r['recovery']*100:.0f}%" for r in cand)
            lo, hi = cand[0]["recovery"], cand[-1]["recovery"]
            verdict = "still improving" if hi - lo > 0.05 else "plateaued"
            md.append(f"- **{sc}x recovery vs epochs:** {trail}  → {verdict}.")
        elif cand:
            md.append(f"- **{sc}x:** single run ({cand[0]['cfg_epochs']}ep) — no epoch sweep.")

    # honest bottom line
    md.append("\n## Bottom line\n")
    under = [r for r in ok if r["recovery"] < 0]
    good = [r for r in ok if r["recovery"] >= 0.8]
    parts = []
    if good:
        parts.append("Best-performing: " + ", ".join(f"{r['run']} ({r['recovery']*100:.0f}%)" for r in good) + ".")
    if under:
        parts.append("Needs retraining (recovery<0): " + ", ".join(r["run"] for r in under) + ".")
    md.append(" ".join(parts) if parts else "See table above.")
    md.append("")

    (out_date_dir / "SUMMARY.md").write_text("\n".join(md))


# --------------------------------------------------------------------------- #
# main                                                                         #
# --------------------------------------------------------------------------- #
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-dir", type=str, required=True)
    ap.add_argument("--experiments", type=str, default=str(HERE / "experiments"))
    ap.add_argument("--out-root", type=str, default=str(HERE / "evaluations"))
    ap.add_argument("--eval-date", type=str, default=date.today().isoformat())
    ap.add_argument("--runs", type=str, nargs="*", default=None,
                    help="Explicit run-dir names; else auto-discover.")
    ap.add_argument("--include-smoke", action="store_true")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max-samples", type=int, default=4000)
    ap.add_argument("--tagger-epochs", type=int, default=15)
    ap.add_argument("--skip-existing", action="store_true")
    args = ap.parse_args()
    args.data_dir = Path(args.data_dir)

    experiments = Path(args.experiments)
    out_date_dir = Path(args.out_root) / args.eval_date
    out_date_dir.mkdir(parents=True, exist_ok=True)

    all_runs = discover_runs(experiments, args.include_smoke)
    if args.runs:
        wanted = set(args.runs)
        all_runs = [m for m in all_runs if m["name"] in wanted]
    if not all_runs:
        print("No eligible runs found.", file=sys.stderr)
        sys.exit(1)

    print(f"[batch] eval date {args.eval_date} → {out_date_dir}")
    print(f"[batch] {len(all_runs)} run(s) to evaluate\n")

    manifest, rows = [], []
    for i, meta in enumerate(all_runs, 1):
        tag = f"{meta['scale']}x__{meta['name']}"
        out_run_dir = out_date_dir / tag
        print(f"[{i}/{len(all_runs)}] {tag}  (scale {meta['scale']}x, "
              f"{meta['reached_epoch']}/{meta['cfg_epochs']} ep)")

        existing = out_run_dir / "classification" / "classification_eval.json"
        if args.skip_existing and existing.exists():
            print("    skip (already evaluated)")
            run_info = {"exit_code": 0, "seconds": 0.0, "skipped": True}
        else:
            run_info = run_one(meta, args, out_run_dir)
            print(f"    exit={run_info['exit_code']} in {run_info['seconds']}s")

        res = _load_result(out_run_dir)
        base_status = training_status(meta)
        if res is None:
            print("    ⚠️ no JSON produced — see eval.log")
            rows.append({**{f"auc_{k}": None for k in ("hr", "lr", "sr")},
                         "run": meta["name"], "scale": meta["scale"],
                         "cfg_epochs": meta["cfg_epochs"], "reached_epoch": meta["reached_epoch"],
                         "status": base_status + " (EVAL FAILED)", "efficiency": None,
                         "recovery": None})
        else:
            p = res["primary_fixed_hr_tagger"]
            rec = p["recovery_fraction_lr_to_hr"]
            status = refine_status(base_status, rec)
            write_report(meta, res, status, out_run_dir)
            rows.append({
                "run": meta["name"], "scale": meta["scale"],
                "cfg_epochs": meta["cfg_epochs"], "reached_epoch": meta["reached_epoch"],
                "status": status,
                "auc_hr": p["auc"]["hr"], "auc_lr": p["auc"]["lr"], "auc_sr": p["auc"]["sr"],
                "efficiency": p["tagging_efficiency_sr_over_hr"], "recovery": rec,
                "bkg_rej_sr_at_50": p["bkg_rej_at_50"]["sr"],
                "sr_hr_pearson": p["score_agreement_vs_hr"]["sr"]["pearson_r"],
                "ece_sr": p["calibration_ece"]["sr"],
            })
            print(f"    efficiency={p['tagging_efficiency_sr_over_hr']*100:.1f}%  "
                  f"recovery={rec*100:.1f}%  [{status}]")
        manifest.append({"run": meta["name"], **run_info})

    (out_date_dir / "batch_manifest.json").write_text(
        json.dumps({"eval_date": args.eval_date, "args": vars(args) | {"data_dir": str(args.data_dir)},
                    "runs": manifest}, indent=2, default=str))
    write_summary(rows, out_date_dir, args)

    print(f"\n[batch] wrote SUMMARY.md + master_metrics.csv → {out_date_dir}")
    ok = [r for r in rows if r.get("auc_sr") is not None]
    for sc in sorted({r["scale"] for r in ok if r["scale"] is not None}):
        cand = [r for r in ok if r["scale"] == sc]
        best = max(cand, key=lambda r: r["efficiency"])
        print(f"  best {sc}x: {best['run']}  (eff {best['efficiency']*100:.1f}%, "
              f"rec {best['recovery']*100:.1f}%)")
    under = [r for r in ok if r["recovery"] is not None and r["recovery"] < 0]
    if under:
        print("  ⚠️ undertrained (recovery<0): " + ", ".join(r["run"] for r in under))


if __name__ == "__main__":
    main()
