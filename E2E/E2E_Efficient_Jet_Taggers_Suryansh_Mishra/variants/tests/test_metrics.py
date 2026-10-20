"""Tests for ``ablation.metrics`` -- the code that turns saved probabilities into the numbers
RESULTS.md quotes. Every case is hand-computable; no scikit-learn or torch needed.

Background (2026-09-09 audit, A11): the previous working-point extraction took a linearly
interpolated quantile of the signal scores and admitted every score ``>=`` it. On tied inputs that
passed 100% of the signal when 50% was requested, and probabilities were stored in fp16, which
manufactures ties. Zero background passes became ``inf`` and were silently dropped from means;
``Tbl``'s Table-1 point (99.5%) was never computed; the overall AUC was one-vs-rest where the
paper is one-vs-one; and the per-class AUC ranked by raw ``p_c`` rather than the discriminant.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from ablation.metrics import (
    EXTRA_EFFICIENCIES,
    JETCLASS_LABELS,
    PAPER_WORKING_POINTS,
    QCD_INDEX,
    background_rejection,
    background_rejection_detail,
    format_summary,
    macro_auc,
    macro_ovo_auc,
    qcd_discriminant,
    roc_operating_point,
    summarize,
)


# ---------------------------------------------------------------------------
# operating point
# ---------------------------------------------------------------------------

def test_operating_point_matches_hand_count_without_ties() -> None:
    signal = np.array([0.9, 0.8, 0.7, 0.6])
    background = np.array([0.85, 0.75, 0.10, 0.05])
    # 50%: two of four signal -> threshold 0.8; background >= 0.8 is {0.85} -> 1 of 4 passes.
    point = roc_operating_point(signal, background, 0.5)
    assert point["threshold"] == 0.8
    assert point["achieved_signal_efficiency"] == 0.5
    assert point["n_background_pass"] == 1
    # 75%: threshold 0.7; background >= 0.7 is {0.85, 0.75} -> 2 of 4.
    assert roc_operating_point(signal, background, 0.75)["n_background_pass"] == 2
    # 100%: threshold 0.6; still 2 of 4 (0.10 and 0.05 are below).
    assert roc_operating_point(signal, background, 1.0)["n_background_pass"] == 2


def test_operating_point_reports_tie_overshoot_instead_of_hiding_it() -> None:
    """Four tied signal scores: requesting 50% must admit the whole tie group and SAY so."""
    signal = np.array([0.9, 0.9, 0.9, 0.9])
    background = np.array([0.9, 0.9, 0.5, 0.4])
    point = roc_operating_point(signal, background, 0.5)
    assert point["achieved_signal_efficiency"] == 1.0      # overshoot is visible
    assert point["n_background_pass"] == 2                 # the two tied background events


def test_ties_between_signal_and_background_enter_together() -> None:
    """A background event with exactly the threshold score is admitted (one ROC point per score)."""
    signal = np.array([0.9, 0.8, 0.7, 0.6])
    background = np.array([0.8, 0.1])
    point = roc_operating_point(signal, background, 0.5)   # threshold 0.8
    assert point["threshold"] == 0.8
    assert point["n_background_pass"] == 1


@pytest.mark.parametrize("seed", range(5))
def test_achieved_efficiency_never_undershoots_and_is_monotone(seed: int) -> None:
    """Properties on fp16-quantised random scores (the regime that manufactured ties)."""
    rng = np.random.default_rng(seed)
    signal = rng.beta(5, 2, size=400).astype(np.float16).astype(np.float64)
    background = rng.beta(2, 5, size=600).astype(np.float16).astype(np.float64)
    last_pass = -1
    for eps in (0.3, 0.5, 0.9, 0.99, 0.995, 1.0):
        point = roc_operating_point(signal, background, eps)
        assert point["achieved_signal_efficiency"] >= eps - 1e-12
        assert point["n_background_pass"] >= last_pass    # more signal admitted -> more background
        last_pass = point["n_background_pass"]
    assert roc_operating_point(signal, background, 1.0)["achieved_signal_efficiency"] == 1.0


def test_operating_point_rejects_bad_efficiency() -> None:
    with pytest.raises(ValueError):
        roc_operating_point(np.ones(3), np.zeros(3), 0.0)
    with pytest.raises(ValueError):
        roc_operating_point(np.ones(3), np.zeros(3), 1.5)


# ---------------------------------------------------------------------------
# rejection
# ---------------------------------------------------------------------------

def _probs_from_scores(signal_disc: np.ndarray, background_disc: np.ndarray, signal_index: int = 1):
    """Build a (N, 10) probability matrix whose S-vs-QCD discriminant equals the given scores.

    ``p_S = d``, ``p_QCD = 1 - d``, everything else 0, so ``p_S / (p_S + p_QCD) = d`` exactly.
    """
    d = np.concatenate([signal_disc, background_disc]).astype(np.float64)
    probs = np.zeros((d.size, 10))
    probs[:, signal_index] = d
    probs[:, QCD_INDEX] = 1.0 - d
    labels = np.concatenate(
        [np.full(signal_disc.size, signal_index), np.full(background_disc.size, QCD_INDEX)]
    )
    return probs, labels


def test_rejection_is_background_count_over_passes() -> None:
    probs, labels = _probs_from_scores(
        np.array([0.9, 0.8, 0.7, 0.6]), np.array([0.85, 0.75, 0.10, 0.05])
    )
    rej = background_rejection(probs, labels, 1, (0.5, 0.75))
    assert rej[0.5] == 4.0          # 4 background / 1 pass
    assert rej[0.75] == 2.0         # 4 background / 2 passes


def test_zero_passes_is_a_lower_bound_not_infinite_rejection() -> None:
    probs, labels = _probs_from_scores(
        np.array([0.9, 0.8, 0.7, 0.6]), np.full(300, 0.1)
    )
    detail = background_rejection_detail(probs, labels, 1, (0.5,))[0.5]
    assert math.isinf(detail["rejection"])                 # legacy value kept
    assert detail["n_background_pass"] == 0
    assert detail["rejection_lower_bound_95"] == pytest.approx(300 / 3)
    assert detail["rejection_stat_rel_err"] is None


def test_rejection_carries_counting_error() -> None:
    probs, labels = _probs_from_scores(
        np.array([0.9, 0.8, 0.7, 0.6]), np.array([0.85, 0.75, 0.10, 0.05])
    )
    detail = background_rejection_detail(probs, labels, 1, (0.75,))[0.75]
    assert detail["n_background_pass"] == 2
    assert detail["rejection_stat_rel_err"] == pytest.approx(1 / math.sqrt(2))


def test_rejection_uses_only_signal_and_qcd_events() -> None:
    """Other-class events must not enter the ROC at all."""
    probs, labels = _probs_from_scores(np.array([0.9, 0.6]), np.array([0.7, 0.1]))
    # Add a third-class event with a huge discriminant-looking score; it must be ignored.
    extra = np.zeros((1, 10)); extra[0, 1] = 0.99; extra[0, QCD_INDEX] = 0.01
    probs = np.vstack([probs, extra]); labels = np.append(labels, 5)
    detail = background_rejection_detail(probs, labels, 1, (1.0,))[1.0]
    assert detail["n_signal"] == 2 and detail["n_background"] == 2


def test_missing_class_gives_nan() -> None:
    probs, labels = _probs_from_scores(np.array([0.9]), np.array([], dtype=float))
    assert math.isnan(background_rejection(probs, labels, 1, (0.5,))[0.5])


# ---------------------------------------------------------------------------
# summarize: working points, keys, AUCs
# ---------------------------------------------------------------------------

def _synthetic_ten_class(seed: int = 0, n_per_class: int = 60):
    rng = np.random.default_rng(seed)
    labels = np.repeat(np.arange(10), n_per_class)
    logits = rng.normal(size=(labels.size, 10))
    logits[np.arange(labels.size), labels] += 2.5           # informative but imperfect
    probs = np.exp(logits) / np.exp(logits).sum(axis=1, keepdims=True)
    return probs, labels


def test_summary_always_contains_paper_working_points_and_30_percent() -> None:
    probs, labels = _synthetic_ten_class()
    summary = summarize(probs, labels, efficiencies=(0.5, 0.99))
    per_class = summary["per_class"]
    assert set(per_class) == set(JETCLASS_LABELS) - {"QCD"}
    for name, entry in per_class.items():
        assert entry["paper_working_point"] == PAPER_WORKING_POINTS[name]
        for eps in EXTRA_EFFICIENCIES:
            assert f"rej_{eps * 100:g}" in entry
    # Tbl's Table-1 point is 99.5%, which the old int(round(eps*100)) key would have called rej_100.
    assert "rej_99.5" in per_class["Tbl"]
    assert per_class["Tbl"]["rej_paper"] == per_class["Tbl"]["rej_99.5"]
    assert per_class["Hqql"]["rej_paper"] == per_class["Hqql"]["rej_99"]
    assert per_class["Hbb"]["rej_paper"] == per_class["Hbb"]["rej_50"]
    # Detail block is keyed identically and its achieved efficiency is recorded.
    assert per_class["Tbl"]["rejection"]["rej_99.5"]["achieved_signal_efficiency"] >= 0.995


def test_summary_keeps_legacy_keys_with_unchanged_meaning() -> None:
    """Old metrics.jsonl consumers read auc / rej_50 / rej_99 / auc_vs_qcd; they must still exist and
    `auc` must still be the one-vs-rest number so old and new logs agree on what it means."""
    probs, labels = _synthetic_ten_class()
    summary = summarize(probs, labels)
    assert summary["auc"] == pytest.approx(macro_auc(probs, labels))
    assert summary["auc_ovo"] == pytest.approx(macro_ovo_auc(probs, labels))
    entry = summary["per_class"]["Hbb"]
    for key in ("auc_vs_qcd", "auc_vs_qcd_disc", "rej_50", "rej_99", "rej_30", "rejection"):
        assert key in entry


def test_ovo_auc_matches_hand_computation_on_three_classes() -> None:
    """Three classes, two events each, hand-checkable pairwise AUCs."""
    labels = np.array([0, 0, 1, 1, 2, 2])
    probs = np.array([
        [0.7, 0.2, 0.1],
        [0.6, 0.3, 0.1],
        [0.3, 0.6, 0.1],
        [0.4, 0.4, 0.2],
        [0.1, 0.1, 0.8],
        [0.2, 0.3, 0.5],
    ])
    # Pair (0,1): p0 on class-0 {0.7,0.6} vs class-1 {0.3,0.4} -> AUC 1; p1 on class-1 {0.6,0.4}
    # vs class-0 {0.2,0.3} -> AUC 1. Pair (0,2): p0 {0.7,0.6} vs {0.1,0.2} -> 1; p2 {0.8,0.5} vs
    # {0.1,0.1} -> 1. Pair (1,2): p1 {0.6,0.4} vs {0.1,0.3} -> 1; p2 {0.8,0.5} vs {0.1,0.2} -> 1.
    assert macro_ovo_auc(probs, labels) == pytest.approx(1.0)
    # Break one pair: make a class-2 event look like class 1 under p1.
    probs2 = probs.copy(); probs2[5] = [0.1, 0.7, 0.2]
    # Pair (1,2) p1: class-1 {0.6,0.4} vs class-2 {0.1,0.7}: pairs (0.6>0.1, 0.6<0.7, 0.4>0.1,
    # 0.4<0.7) -> 2/4 = 0.5; p2: class-2 {0.8,0.2} vs class-1 {0.1,0.2}: (0.8>0.1, 0.8>0.2,
    # 0.2>0.1, 0.2=0.2 tie -> 0.5) -> 3.5/4 = 0.875. Pair score (0.5+0.875)/2 = 0.6875.
    # Pairs (0,1) and (0,2) unchanged at 1. Macro: (1 + 1 + 0.6875) / 3.
    assert macro_ovo_auc(probs2, labels) == pytest.approx((1 + 1 + 0.6875) / 3)


def test_ovo_equals_ovr_on_balanced_classes_and_differs_otherwise() -> None:
    """Identity: with equal class counts, macro OVR == macro OVO (both are the mean of AUC_a(a,b)
    over ordered pairs). JetClass val_5M is balanced, so the definitional fix does not move the
    historical `auc` there; it matters for any unbalanced sample."""
    probs, labels = _synthetic_ten_class(seed=3)
    assert macro_ovo_auc(probs, labels) == pytest.approx(macro_auc(probs, labels), abs=1e-12)
    keep = np.ones(labels.size, dtype=bool)
    keep[(labels == 2)] = False; keep[np.flatnonzero(labels == 2)[:5]] = True   # class 2: 60 -> 5
    assert macro_ovo_auc(probs[keep], labels[keep]) != pytest.approx(
        macro_auc(probs[keep], labels[keep]), abs=1e-6
    )


def test_discriminant_auc_can_differ_from_raw_probability_auc() -> None:
    """Other-class mass reorders raw p_S but not p_S/(p_S+p_QCD)."""
    #            QCD   S    other
    probs = np.array([
        [0.10, 0.30, 0.60],    # signal: raw p_S 0.30, disc 0.75
        [0.55, 0.45, 0.00],    # QCD:    raw p_S 0.45, disc 0.45
    ])
    labels = np.array([1, QCD_INDEX])
    keep = np.ones(2, dtype=bool)
    from ablation.metrics import _binary_auc
    raw = _binary_auc(probs[:, 1], labels == 1)
    disc = _binary_auc(qcd_discriminant(probs, 1), labels == 1)
    assert raw == 0.0 and disc == 1.0


def test_format_summary_marks_paper_point_and_bounds() -> None:
    probs, labels = _synthetic_ten_class()
    text = format_summary(summarize(probs, labels))
    assert "auc_ovo=" in text and "rej_99.5" in text and "*" in text


# ---------------------------------------------------------------------------
# report.py: per-class at the paper point, never averaged, inf shown as a bound
# ---------------------------------------------------------------------------

def test_report_table_uses_paper_points_and_never_averages(tmp_path) -> None:
    import json
    from ablation.report import build_table, format_table

    probs, labels = _synthetic_ten_class()
    summary = summarize(probs, labels)
    # Force one class to zero background passes so `inf` appears with its bound.
    summary["per_class"]["Hbb"]["rej_paper"] = float("inf")
    summary["per_class"]["Hbb"]["rejection"]["rej_50"]["rejection"] = float("inf")
    summary["per_class"]["Hbb"]["rejection"]["rej_50"]["rejection_lower_bound_95"] = 20.0

    new_run = tmp_path / "baseline"
    new_run.mkdir()
    (new_run / "config.json").write_text(json.dumps({"arm": "baseline", "total_steps": 10}))
    (new_run / "metrics.jsonl").write_text(
        json.dumps({"event": "start", "params": 100}) + "\n"
        + json.dumps({"event": "eval", "step": 10, **summary}) + "\n"
    )
    # A legacy run (pre-2026-09-09 keys only): rej_50/rej_99 per class, no rej_paper, no auc_ovo.
    legacy = {
        name: {"auc_vs_qcd": 0.9, "rej_50": 100.0 + i, "rej_99": 5.0}
        for i, name in enumerate(JETCLASS_LABELS[1:])
    }
    old_run = tmp_path / "tied_k1"
    old_run.mkdir()
    (old_run / "config.json").write_text(json.dumps({"arm": "tied", "total_steps": 10}))
    (old_run / "metrics.jsonl").write_text(
        json.dumps({"event": "start", "params": 50}) + "\n"
        + json.dumps({"event": "eval", "step": 10, "accuracy": 0.8, "auc": 0.9,
                      "per_class": legacy}) + "\n"
    )

    text = format_table(build_table(tmp_path))
    assert "rej50" not in text and "rej99" not in text          # no cross-class means
    assert "auc_ovo" in text and "Tbl@99.5" in text and "Hqql@99" in text
    assert ">20" in text                                        # zero passes -> bound, not inf
    legacy_line = next(l for l in text.splitlines() if l.strip().startswith("tied_k1") and "@" not in l and "100." in l)
    assert "100.0" in legacy_line                               # Hbb rej_50 read from legacy key
    assert legacy_line.rstrip().endswith("-")                   # Tbl@99.5 not available in legacy
