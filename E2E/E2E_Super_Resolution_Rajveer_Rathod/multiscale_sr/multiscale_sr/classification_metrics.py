"""Classification-based evaluation metrics for super-resolution quality.

The headline question this module answers is *not* "do the pixels match?" (that's
PSNR/SSIM) but "does the super-resolved image preserve the **physics
classification** information a downstream tagger needs?". Everything here operates
on tagger scores (sigmoid probabilities in [0, 1]) and binary labels, so it is
shared by the fixed-HR-tagger and per-source analyses in classification_eval.py.

All functions degrade gracefully to a numpy fallback when sklearn is absent, and
return NaN rather than raising when a metric is undefined (e.g. a single class).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


# --------------------------------------------------------------------------- #
# ROC / AUC (re-exported convenience over tagger.roc_auc with extra HEP metrics)
# --------------------------------------------------------------------------- #
@dataclass
class WorkingPoint:
    """A single (threshold, signal-eff, background-rejection) operating point."""

    threshold: float
    signal_eff: float          # TPR: fraction of signal (y==1) kept
    background_rej: float       # 1 / FPR (inf if FPR==0)
    background_eff: float        # FPR: fraction of background (y==0) kept


def _binary_guard(labels: np.ndarray) -> bool:
    """True iff both classes are present (metrics are otherwise undefined)."""
    return len(np.unique(labels)) >= 2


def roc_points(scores: np.ndarray, labels: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """Return (fpr, tpr, thresholds, auc). NaN auc when only one class present."""
    if not _binary_guard(labels):
        z = np.array([0.0, 1.0])
        return z, z, np.array([1.0, 0.0]), float("nan")
    try:
        from sklearn.metrics import roc_auc_score, roc_curve

        fpr, tpr, thr = roc_curve(labels, scores)
        return fpr, tpr, thr, float(roc_auc_score(labels, scores))
    except ImportError:
        pass
    # numpy fallback: threshold sweep + trapezoid (mirrors tagger.roc_auc).
    order = np.argsort(-scores)
    s = scores[order]
    y = labels[order]
    thresholds = np.unique(s)[::-1]
    pos = max(int((labels == 1).sum()), 1)
    neg = max(int((labels == 0).sum()), 1)
    fpr_list, tpr_list, thr_list = [0.0], [0.0], [np.inf]
    for t in thresholds:
        pred = s >= t
        tpr_list.append(int((pred & (y == 1)).sum()) / pos)
        fpr_list.append(int((pred & (y == 0)).sum()) / neg)
        thr_list.append(float(t))
    fpr = np.asarray(fpr_list)
    tpr = np.asarray(tpr_list)
    auc = float(abs(np.trapezoid(tpr, fpr)))
    return fpr, tpr, np.asarray(thr_list), auc


def background_rejection_at_signal_eff(
    fpr: np.ndarray, tpr: np.ndarray, target_eff: float = 0.5
) -> float:
    """HEP-standard 1/eps_B at a fixed signal efficiency eps_S (e.g. ROC50 -> 50%).

    Interpolates the ROC at tpr == target_eff and returns 1/fpr there. A higher
    number means the tagger rejects more background at that signal efficiency, the
    metric particle physicists actually quote. Returns inf if fpr is 0 at that
    point, NaN if the ROC is degenerate.
    """
    if fpr.size < 2 or np.all(np.isnan(fpr)):
        return float("nan")
    # tpr from roc_curve is monotonically non-decreasing, so np.interp is valid.
    fpr_at = float(np.interp(target_eff, tpr, fpr))
    if fpr_at <= 0.0:
        return float("inf")
    return 1.0 / fpr_at


def best_threshold(fpr: np.ndarray, tpr: np.ndarray, thresholds: np.ndarray) -> float:
    """Youden's J optimal threshold (argmax tpr - fpr)."""
    if thresholds.size == 0:
        return 0.5
    j = tpr - fpr
    idx = int(np.argmax(j))
    thr = float(thresholds[idx])
    # roc_curve sets thresholds[0] = inf (predict-nothing point); avoid returning it.
    if not np.isfinite(thr):
        thr = 0.5
    return thr


def working_points(
    fpr: np.ndarray, tpr: np.ndarray, thresholds: np.ndarray,
    signal_effs: tuple[float, ...] = (0.3, 0.5, 0.7, 0.9),
) -> list[WorkingPoint]:
    """Background rejection at a few canonical signal efficiencies for a table."""
    out: list[WorkingPoint] = []
    for eff in signal_effs:
        fpr_at = float(np.interp(eff, tpr, fpr)) if fpr.size >= 2 else float("nan")
        rej = (1.0 / fpr_at) if fpr_at and fpr_at > 0 else float("inf")
        out.append(WorkingPoint(threshold=float("nan"), signal_eff=eff,
                                background_rej=rej, background_eff=fpr_at))
    return out


# --------------------------------------------------------------------------- #
# Threshold-dependent metrics (confusion matrix, accuracy, F1)
# --------------------------------------------------------------------------- #
@dataclass
class ConfusionStats:
    tp: int
    fp: int
    tn: int
    fn: int
    accuracy: float
    precision: float
    recall: float
    f1: float
    threshold: float

    @property
    def matrix(self) -> np.ndarray:
        """2x2 [[TN, FP], [FN, TP]] in sklearn row=true / col=pred order."""
        return np.array([[self.tn, self.fp], [self.fn, self.tp]], dtype=int)


def confusion_at_threshold(scores: np.ndarray, labels: np.ndarray, threshold: float) -> ConfusionStats:
    """Confusion matrix + accuracy/precision/recall/F1 at a fixed score threshold."""
    pred = scores >= threshold
    y = labels.astype(bool)
    tp = int((pred & y).sum())
    fp = int((pred & ~y).sum())
    tn = int((~pred & ~y).sum())
    fn = int((~pred & y).sum())
    total = max(tp + fp + tn + fn, 1)
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-12)
    return ConfusionStats(
        tp=tp, fp=fp, tn=tn, fn=fn,
        accuracy=(tp + tn) / total, precision=precision, recall=recall, f1=f1,
        threshold=float(threshold),
    )


# --------------------------------------------------------------------------- #
# Calibration (reliability curve + ECE) and score agreement
# --------------------------------------------------------------------------- #
@dataclass
class CalibrationCurve:
    bin_centers: np.ndarray = field(default_factory=lambda: np.array([]))
    bin_accuracy: np.ndarray = field(default_factory=lambda: np.array([]))
    bin_confidence: np.ndarray = field(default_factory=lambda: np.array([]))
    bin_counts: np.ndarray = field(default_factory=lambda: np.array([]))
    ece: float = float("nan")


def calibration_curve(scores: np.ndarray, labels: np.ndarray, n_bins: int = 10) -> CalibrationCurve:
    """Reliability curve + Expected Calibration Error.

    For each probability bin, compare mean predicted score (confidence) to the
    empirical fraction of positives (accuracy). A well-calibrated tagger sits on
    the diagonal; ECE is the count-weighted mean gap. Useful to check that SR
    images don't shift the tagger's confidence relative to HR.
    """
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    idx = np.clip(np.digitize(scores, edges[1:-1]), 0, n_bins - 1)
    centers, accs, confs, counts = [], [], [], []
    ece, n = 0.0, max(len(scores), 1)
    for b in range(n_bins):
        mask = idx == b
        c = int(mask.sum())
        centers.append((edges[b] + edges[b + 1]) / 2)
        if c == 0:
            accs.append(np.nan)
            confs.append(np.nan)
            counts.append(0)
            continue
        acc = float(labels[mask].mean())
        conf = float(scores[mask].mean())
        accs.append(acc)
        confs.append(conf)
        counts.append(c)
        ece += (c / n) * abs(acc - conf)
    return CalibrationCurve(
        bin_centers=np.asarray(centers), bin_accuracy=np.asarray(accs),
        bin_confidence=np.asarray(confs), bin_counts=np.asarray(counts), ece=ece,
    )


def score_agreement(scores_ref: np.ndarray, scores_other: np.ndarray) -> dict[str, float]:
    """How closely a tagger's per-sample scores on one source match another.

    Quantifies "does SR fool the HR-trained tagger the *same way* HR does, sample
    by sample?" — a stricter test than matching aggregate AUC. Returns Pearson r,
    Spearman rho (if scipy present), and mean-absolute score difference.
    """
    out: dict[str, float] = {}
    if scores_ref.size >= 2 and np.std(scores_ref) > 0 and np.std(scores_other) > 0:
        out["pearson_r"] = float(np.corrcoef(scores_ref, scores_other)[0, 1])
    else:
        out["pearson_r"] = float("nan")
    try:
        from scipy.stats import spearmanr

        out["spearman_rho"] = float(spearmanr(scores_ref, scores_other).correlation)
    except Exception:
        out["spearman_rho"] = float("nan")
    out["mean_abs_score_diff"] = float(np.mean(np.abs(scores_ref - scores_other)))
    return out
