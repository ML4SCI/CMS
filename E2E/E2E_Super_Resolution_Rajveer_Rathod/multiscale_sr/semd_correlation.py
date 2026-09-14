"""Correlate every physics metric against tagging efficiency, across seeds.

This is the decision point for the SEMD work (S6). The hypothesis under test:

    Every physics metric currently logged is invariant under pixel permutation,
    so none of them can see the geometric differences that drive jet tagging.
    SEMD can. Therefore SEMD should track tagging efficiency across seeds where
    the existing metrics do not.

The script is deliberately symmetric: SEMD gets no special treatment, and every
existing metric is scored the same way against the same four points. A result
where SEMD correlates no better than ``energy_response`` is a **null result and
must be reported as one** — the alternative, sweeping ``topk``/``omega_R``
until the ranking appears, would prove nothing on n=4.

Usage::

    python semd_correlation.py --eval-dir evaluations/2026-08-22 \\
        --out docs/semd_correlation.md

It reads the ``classification_eval.json`` written by ``classification_eval.py``
for each seed and emits a markdown table plus a machine-readable JSON.

Caveat that must survive into any writeup: **n=4**. With four points, a Spearman
rho of 1.0 has a two-sided permutation p-value of 1/12 ~ 0.083 — suggestive, not
significant. This script prints that p-value next to every rho so the number is
never quoted without it.
"""
from __future__ import annotations

import argparse
import itertools
import json
import math
from pathlib import Path


# Metrics pulled from each eval JSON. ``lower_is_better`` only affects how the
# expected sign is described in the report; the correlation itself is signed.
METRIC_SPECS: list[tuple[str, str, bool]] = [
    # (label, dotted path into classification_eval.json, lower_is_better)
    ("SEMD(SR,HR)", "semd.sr_vs_hr.mean", True),
    ("SEMD recovery", "semd.semd_recovery", False),
    ("energy mean ratio", "physics_correlation.energy_correlation_vs_hr.sr.mean_ratio", False),
    ("energy pearson r", "physics_correlation.energy_correlation_vs_hr.sr.pearson_r", False),
    ("energy |frac diff|", "physics_correlation.energy_correlation_vs_hr.sr.mean_abs_frac_diff", True),
    ("pT-E correlation", "physics_correlation.pt_energy_pearson_r.sr", False),
]


def _dig(obj, path: str):
    """Fetch a dotted path, returning None rather than raising on a miss."""
    cur = obj
    for key in path.split("."):
        if not isinstance(cur, dict) or key not in cur:
            return None
        cur = cur[key]
    return cur if isinstance(cur, (int, float)) else None


def _rank(xs: list[float]) -> list[float]:
    """Average ranks, so ties don't bias the correlation."""
    order = sorted(range(len(xs)), key=lambda i: xs[i])
    ranks = [0.0] * len(xs)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and xs[order[j + 1]] == xs[order[i]]:
            j += 1
        avg = (i + j) / 2.0 + 1.0
        for k in range(i, j + 1):
            ranks[order[k]] = avg
        i = j + 1
    return ranks


def _pearson(xs: list[float], ys: list[float]) -> float:
    n = len(xs)
    if n < 2:
        return float("nan")
    mx, my = sum(xs) / n, sum(ys) / n
    num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    dx = math.sqrt(sum((x - mx) ** 2 for x in xs))
    dy = math.sqrt(sum((y - my) ** 2 for y in ys))
    return num / (dx * dy) if dx > 0 and dy > 0 else float("nan")


def _spearman(xs: list[float], ys: list[float]) -> float:
    return _pearson(_rank(xs), _rank(ys))


def _perm_pvalue(xs: list[float], ys: list[float]) -> float:
    """Exact two-sided permutation p-value for Spearman rho.

    With n=4 there are only 24 permutations, so the exact null is cheap and far
    more honest than a large-sample approximation.
    """
    obs = _spearman(xs, ys)
    if math.isnan(obs):
        return float("nan")
    rx = _rank(xs)
    count = 0
    total = 0
    for perm in itertools.permutations(range(len(ys))):
        r = _pearson(rx, [_rank(ys)[i] for i in perm])
        total += 1
        if not math.isnan(r) and abs(r) >= abs(obs) - 1e-12:
            count += 1
    return count / total if total else float("nan")


def load_seed_results(eval_dir: Path, scale: int | None = None,
                      require_semd: bool = True) -> list[dict]:
    """Collect one record per ``classification_eval.json`` under ``eval_dir``.

    ``eval_dir`` is typically a Drive folder holding *every* run ever saved, of
    mixed scales, epoch budgets, and code vintages. Correlating across that set
    conflates scale with seed and silently mixes runs evaluated under different
    rulers, so filtering is on by default rather than opt-in.
    """
    records = []
    skipped_no_semd = 0
    skipped_scale = 0
    for path in sorted(eval_dir.rglob("classification_eval.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            print(f"[semd-corr] skipping unreadable {path}: {exc}")
            continue
        prim = payload.get("primary_fixed_hr_tagger", {})
        auc = prim.get("auc", {})
        if not {"hr", "lr", "sr"} <= set(auc):
            print(f"[semd-corr] skipping {path}: no primary AUC block")
            continue
        # classification_eval.py writes the fully-qualified key names; the
        # short aliases are accepted only as a fallback for hand-made fixtures.
        eff = prim.get("tagging_efficiency_sr_over_hr", prim.get("tagging_efficiency"))
        if eff is None and auc.get("hr"):
            eff = auc["sr"] / auc["hr"]
        recovery = prim.get("recovery_fraction_lr_to_hr")
        semd_block = payload.get("semd") or {}
        if require_semd and not semd_block.get("params"):
            skipped_no_semd += 1
            continue

        run_scale = payload.get("scale")
        if scale is not None and run_scale != scale:
            skipped_scale += 1
            continue

        # The run directory name: .../<run_name>/figures/classification/<file>
        run_name = path.parent.parent.parent.name
        records.append(
            {
                "path": str(path),
                "run": run_name,
                "scale": run_scale,
                "eval_seed": payload.get("eval_seed"),
                "auc_hr": auc["hr"],
                "auc_lr": auc["lr"],
                "auc_sr": auc["sr"],
                "tagging_efficiency": eff,
                "recovery_fraction": recovery,
                "metrics": {
                    label: _dig(payload, dotted) for label, dotted, _ in METRIC_SPECS
                },
                "semd_params": semd_block.get("params"),
            }
        )

    if skipped_no_semd:
        print(f"[semd-corr] skipped {skipped_no_semd} run(s) with no SEMD block — these "
              f"predate the SEMD code. Re-run classification_eval.py on those checkpoints "
              f"to include them (--allow-missing-semd to audit legacy metrics instead).")
    if skipped_scale:
        print(f"[semd-corr] skipped {skipped_scale} run(s) not at scale {scale}x")
    return records


def build_report(records: list[dict]) -> tuple[str, dict]:
    n = len(records)
    effs = [r["tagging_efficiency"] for r in records]

    lines = ["# SEMD correlation with tagging efficiency", ""]
    lines.append(f"**n = {n} checkpoints.** ")
    lines.append("")

    if n < 3:
        lines.append(
            "Fewer than 3 checkpoints — no correlation is computable. "
            "Run the seed sweep first."
        )
        return "\n".join(lines), {"n": n, "correlations": {}}

    # --- provenance ---
    hr_set = {round(r["auc_hr"], 12) for r in records}
    lines.append("## Provenance")
    lines.append("")
    lines.append("| Run | scale | eval_seed | AUC_HR | AUC_LR | AUC_SR | Tagging eff |")
    lines.append("|---|---|---|---|---|---|---|")
    for r in records:
        lines.append(
            f"| `{r['run']}` | {r.get('scale')}x | {r['eval_seed']} | {r['auc_hr']:.4f} | "
            f"{r['auc_lr']:.4f} | {r['auc_sr']:.4f} | {r['tagging_efficiency']*100:.1f}% |"
        )
    lines.append("")

    # --- contamination guards --------------------------------------------- #
    # These are the two ways this table silently stops meaning anything.
    scales = {r.get("scale") for r in records if r.get("scale") is not None}
    if len(scales) > 1:
        lines.append(
            f"> **STOP: this table mixes {len(scales)} scales ({sorted(scales)}).** Tagging "
            "difficulty depends strongly on scale, so any correlation below is partly "
            "'which scale is this' rather than 'which seed is this'. Re-run with "
            "`--scale <N>` before reading anything into these numbers."
        )
        lines.append("")

    missing_seed = [r for r in records if r["eval_seed"] is None]
    if missing_seed:
        lines.append(
            f"> **Warning: {len(missing_seed)} run(s) record no `eval_seed`.** These predate "
            "the Phase A eval-seed fix, so their tagger split and init moved with the "
            "training seed — model variance and evaluation noise are not separable in them."
        )
        lines.append("")
    if len(hr_set) == 1:
        lines.append(
            f"AUC_HR is identical across all {n} runs ({hr_set.pop():.4f}) — the frozen "
            "tagger is doing its job, so these differences are the generator's."
        )
    else:
        lines.append(
            f"> **Warning:** AUC_HR varies across runs ({sorted(hr_set)}). The ruler is "
            "not fixed, so every correlation below is confounded. Re-run with a single "
            "`--tagger-checkpoint` and one `--eval-seed` before interpreting anything."
        )
    lines.append("")

    # Runs that share a config (scale, AUC_HR, AUC_LR) and differ only by
    # training/eval seed are variance replicates, not independent samples of
    # "which config performs better". Counting them as independent n inflates
    # the correlation's apparent power without adding real information.
    config_key = lambda r: (r.get("scale"), round(r["auc_hr"], 6), round(r["auc_lr"], 6))
    config_counts: dict[tuple, int] = {}
    for r in records:
        config_counts[config_key(r)] = config_counts.get(config_key(r), 0) + 1
    repeat_groups = {k: v for k, v in config_counts.items() if v > 1}
    if repeat_groups:
        n_repeats = sum(repeat_groups.values())
        n_independent = len(config_counts)
        lines.append(
            f"> **Warning: {n_repeats} of {n} runs are seed repeats of "
            f"{len(repeat_groups)} config(s)** (same scale/AUC_HR/AUC_LR), leaving only "
            f"**{n_independent} independent configuration(s)**. Seed repeats estimate "
            "variance within a config; they are not additional evidence about which "
            "config or metric drives recovery. Treat the effective n for the correlation "
            f"below as {n_independent}, not {n}."
        )
        lines.append("")

    params = {json.dumps(r["semd_params"], sort_keys=True) for r in records if r["semd_params"]}
    if len(params) > 1:
        lines.append(
            "> **Warning:** SEMD parameters differ across runs, so SEMD values are not "
            f"comparable: {sorted(params)}"
        )
        lines.append("")
    elif params:
        lines.append(f"SEMD parameters (identical across runs): `{params.pop()}`")
        lines.append("")

    # --- correlation table ---
    lines.append("## Correlation against tagging efficiency")
    lines.append("")
    lines.append("| Metric | Spearman rho | perm. p | Pearson r | spread (CV) | Verdict |")
    lines.append("|---|---|---|---|---|---|")

    corr_out: dict[str, dict] = {}
    for label, _dotted, lower_better in METRIC_SPECS:
        vals = [r["metrics"].get(label) for r in records]
        if any(v is None for v in vals):
            lines.append(f"| {label} | — | — | — | — | absent from results JSON |")
            continue
        rho = _spearman(vals, effs)
        pval = _perm_pvalue(vals, effs)
        pear = _pearson(vals, effs)
        mean = sum(vals) / len(vals)
        sd = math.sqrt(sum((v - mean) ** 2 for v in vals) / max(len(vals) - 1, 1))
        cv = abs(sd / mean) if mean else float("nan")

        if math.isnan(rho):
            verdict = "constant — no signal"
        elif abs(rho) >= 0.8:
            direction = "tracks" if (rho < 0) == lower_better else "anti-tracks"
            verdict = f"**{direction} tagging**"
        elif abs(rho) >= 0.5:
            verdict = "weak"
        else:
            verdict = "blind"

        lines.append(
            f"| {label} | {rho:+.3f} | {pval:.3f} | {pear:+.3f} | {cv*100:.1f}% | {verdict} |"
        )
        corr_out[label] = {
            "spearman_rho": rho, "perm_p": pval, "pearson_r": pear,
            "cv": cv, "values": vals, "lower_is_better": lower_better,
        }

    lines.append("")
    lines.append(
        f"`perm. p` is the **exact** two-sided permutation p-value over all {math.factorial(n)} "
        f"orderings of {n} points. The smallest attainable value is "
        f"{2/math.factorial(n):.3f}, so with n={n} even a perfect rank correlation "
        "cannot reach p<0.05. These numbers rank hypotheses; they do not confirm one."
    )
    lines.append("")

    # --- verdict ---
    lines.append("## Verdict")
    lines.append("")
    semd_rho = corr_out.get("SEMD(SR,HR)", {}).get("spearman_rho")
    others = [
        abs(v["spearman_rho"]) for k, v in corr_out.items()
        if not k.startswith("SEMD") and not math.isnan(v["spearman_rho"])
    ]
    best_other = max(others) if others else 0.0

    # Rank correlation saturates easily at n=4, so a tie at |rho|=1.0 between
    # SEMD and an existing metric is common and is NOT evidence for SEMD. Break
    # such ties on Pearson r, which still discriminates once ranks are exhausted,
    # and require a real margin before claiming SEMD wins.
    tied = [
        k for k, v in corr_out.items()
        if not k.startswith("SEMD") and not math.isnan(v["spearman_rho"])
        and abs(abs(v["spearman_rho"]) - abs(semd_rho or 0.0)) < 1e-9
    ]
    semd_pearson = abs(corr_out.get("SEMD(SR,HR)", {}).get("pearson_r", float("nan")))
    best_other_pearson = max(
        (abs(v["pearson_r"]) for k, v in corr_out.items()
         if not k.startswith("SEMD") and not math.isnan(v["pearson_r"])),
        default=0.0,
    )

    if semd_rho is None or math.isnan(semd_rho):
        lines.append("SEMD is absent or constant — **inconclusive**. Re-run the evals.")
        answer = "inconclusive"
    elif abs(semd_rho) >= 0.8 and tied:
        lines.append(
            f"**Inconclusive — SEMD ties with {len(tied)} existing metric(s) at "
            f"|rho| = {abs(semd_rho):.3f}: {', '.join('`'+t+'`' for t in tied)}.**\n\n"
            f"With n={n} there are only {math.factorial(n)} possible orderings, so a "
            "perfect rank correlation is cheap and several metrics reach it by chance. "
            "A tie here is *not* evidence that SEMD adds information — it is evidence "
            f"that n={n} cannot separate these hypotheses."
            + (
                f"\n\nOn the tiebreaker (Pearson r, which still discriminates once ranks "
                f"saturate) SEMD gives {semd_pearson:.3f} vs {best_other_pearson:.3f} for the "
                "best existing metric — "
                + ("a point in SEMD's favour, but not a decisive one."
                   if semd_pearson > best_other_pearson + 0.05
                   else "no meaningful separation.")
            )
            + "\n\n**Do not enable `--lambda-semd` on this basis.** Add seeds until the "
            "ranks can actually be distinguished."
        )
        answer = "inconclusive_tie"
    elif abs(semd_rho) >= 0.8 and abs(semd_rho) > best_other + 0.1:
        lines.append(
            f"**Yes — SEMD sees what the pixel-wise metrics cannot.** SEMD reaches "
            f"|rho| = {abs(semd_rho):.3f} against tagging efficiency, while the best "
            f"existing physics metric manages only {best_other:.3f}. This is consistent "
            "with the metric-blindness hypothesis: the geometric information SEMD adds "
            "is the information tagging depends on.\n\n"
            f"It remains n={n}, p = "
            f"{corr_out['SEMD(SR,HR)']['perm_p']:.3f}. The honest next step is more seeds, "
            "not switching on `--lambda-semd` and declaring victory."
        )
        answer = "semd_correlates"
    else:
        lines.append(
            f"**No.** SEMD reaches |rho| = {abs(semd_rho):.3f}, against {best_other:.3f} for "
            "the best existing metric. On this evidence SEMD is not a better tagging "
            "surrogate than what we already log.\n\n"
            "**This is the finding — report it.** Do not sweep `topk`/`omega_R`/`threshold` "
            f"looking for a configuration that reproduces the known ranking; with n={n} "
            "that is curve-fitting to four points, and it would manufacture exactly the "
            "kind of result this investigation exists to avoid. Either the geometric "
            "hypothesis is wrong, or the top-K pixel approximation is too coarse to test "
            "it — and distinguishing those needs more seeds, not more knobs."
        )
        answer = "semd_does_not_correlate"

    return "\n".join(lines), {
        "n": n, "n_independent_configs": len(config_counts), "answer": answer,
        "correlations": corr_out,
        "records": [{k: v for k, v in r.items() if k != "metrics"} for r in records],
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--eval-dir", type=str, required=True,
                    help="Directory containing per-seed classification_eval.json files")
    ap.add_argument("--scale", type=int, default=None,
                    help="Keep only runs at this scale. Correlating across scales conflates "
                         "'which scale' with 'which seed' and is almost never what you want")
    ap.add_argument("--require-semd", action="store_true", default=True,
                    help="Drop runs with no SEMD block (i.e. produced before the SEMD code "
                         "landed). On by default: a table with no SEMD rows cannot answer "
                         "the question this script exists to answer")
    ap.add_argument("--allow-missing-semd", dest="require_semd", action="store_false",
                    help="Include pre-SEMD runs. Only for auditing the legacy metrics")
    ap.add_argument("--out", type=str, default=None, help="Markdown output path")
    ap.add_argument("--out-json", type=str, default=None, help="JSON output path")
    args = ap.parse_args()

    eval_dir = Path(args.eval_dir)
    if not eval_dir.exists():
        raise SystemExit(f"[semd-corr] no such directory: {eval_dir}")

    records = load_seed_results(eval_dir, scale=args.scale, require_semd=args.require_semd)
    if not records:
        raise SystemExit(f"[semd-corr] no classification_eval.json found under {eval_dir}")
    print(f"[semd-corr] loaded {len(records)} result(s)")

    md, payload = build_report(records)
    print()
    print(md)

    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(md + "\n", encoding="utf-8")
        print(f"\n[semd-corr] wrote {out}")
    if args.out_json:
        outj = Path(args.out_json)
        outj.parent.mkdir(parents=True, exist_ok=True)
        outj.write_text(json.dumps(payload, indent=2, default=float), encoding="utf-8")
        print(f"[semd-corr] wrote {outj}")


if __name__ == "__main__":
    main()
