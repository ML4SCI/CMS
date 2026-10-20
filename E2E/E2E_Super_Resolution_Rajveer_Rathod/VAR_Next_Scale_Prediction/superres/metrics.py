"""Physics-facing metrics.

Tagger (qg):      ROC AUC, background rejection 1/eps_B at fixed signal efficiency.
Regressor (calo): energy response R = E_pred / E_true -> bias (median R - 1) and
                  resolution sigma_eff / median, with sigma_eff = IQR / 1.349.
Distributions:    Wasserstein-1 distance in units of the reference std, KS statistic,
                  mean shift; per-event bias and resolution of matched residuals.
"""
import numpy as np
from scipy.stats import ks_2samp, wasserstein_distance
from sklearn.metrics import roc_auc_score, roc_curve

SIG_EFFS = (0.3, 0.5, 0.7)


def robust_sigma(x):
    q75, q25 = np.percentile(x, [75, 25])
    return float((q75 - q25) / 1.349)


def rejection_at(y, score, eff):
    fpr, tpr, _ = roc_curve(y, score)
    fb = float(np.interp(eff, tpr, fpr))
    return 1.0 / max(fb, 1e-6)


def classification_metrics(y, score, n_boot=100, seed=0):
    y = np.asarray(y).astype(int)
    score = np.asarray(score, dtype=np.float64)
    out = {"auc": float(roc_auc_score(y, score)),
           "accuracy": float(((score > 0) == y).mean())}  # score is a logit
    for e in SIG_EFFS:
        out[f"rej_at_eff{int(e * 100)}"] = rejection_at(y, score, e)
    rng = np.random.default_rng(seed)
    aucs, rejs = [], []
    for _ in range(n_boot):
        i = rng.integers(0, len(y), len(y))
        if len(np.unique(y[i])) < 2:
            continue
        aucs.append(roc_auc_score(y[i], score[i]))
        rejs.append(rejection_at(y[i], score[i], 0.5))
    if aucs:
        out["auc_boot_err"] = float(np.std(aucs))  # uncertainty from resampling the test set
        out["rej_at_eff50_boot_err"] = float(np.std(rejs))
    return out


def response_metrics(e_true, e_pred, n_bins=8):
    R = e_pred / e_true
    med = float(np.median(R))
    out = {"response_bias": med - 1.0, "resolution": robust_sigma(R) / med}
    edges = np.logspace(np.log10(e_true.min()), np.log10(e_true.max()) + 1e-9, n_bins + 1)
    binned = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (e_true >= lo) & (e_true < hi)
        if m.sum() < 20:
            continue
        mr = float(np.median(R[m]))
        binned.append({"e_lo": float(lo), "e_hi": float(hi), "n": int(m.sum()),
                       "bias": mr - 1.0, "resolution": robust_sigma(R[m]) / mr})
    out["binned"] = binned
    out["mean_binned_resolution"] = float(np.mean([b["resolution"] for b in binned])) if binned else out["resolution"]
    return out


def paired_metrics(ref, other):
    """Event by event (SR and HR are the same events): bias = mean(other - ref) / mean(ref),
    resolution = std(other - ref) / mean(ref), and the Pearson correlation."""
    ref = np.asarray(ref, dtype=np.float64)
    other = np.asarray(other, dtype=np.float64)
    ok = np.isfinite(ref) & np.isfinite(other)
    a, b = ref[ok], other[ok]
    scale = abs(a.mean()) + 1e-12
    r = float(np.corrcoef(a, b)[0, 1]) if a.std() > 0 and b.std() > 0 else float("nan")
    return {"mean_ref": float(a.mean()), "mean_pred": float(b.mean()), "bias": float((b - a).mean() / scale),
            "resolution": float((b - a).std() / scale), "pearson_r": r}


def compare_distributions(ref, other, relative=False):
    """ref/other: per-event observable values for the same events."""
    ref = np.asarray(ref, dtype=np.float64)
    other = np.asarray(other, dtype=np.float64)
    ok = np.isfinite(ref) & np.isfinite(other)
    a, b = ref[ok], other[ok]
    sd = a.std() + 1e-12
    out = {"w1_over_sigma": float(wasserstein_distance(a, b) / sd),
           "ks": float(ks_2samp(a, b).statistic),
           "mean_ref": float(a.mean()), "mean_pred": float(b.mean()),
           "mean_shift_over_sigma": float((b.mean() - a.mean()) / sd)}
    if relative:
        m = np.abs(a) > 1e-12
        r = b[m] / a[m] - 1.0
        out.update({"event_bias": float(np.median(r)), "event_resolution": robust_sigma(r), "residual": "relative"})
    else:
        r = b - a
        out.update({"event_bias": float(np.median(r) / sd), "event_resolution": robust_sigma(r) / sd,
                    "residual": "absolute/sigma_ref"})
    return out
