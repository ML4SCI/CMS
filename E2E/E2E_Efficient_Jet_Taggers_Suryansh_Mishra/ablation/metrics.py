"""Evaluation metrics for JetClass tagging.

Accuracy and AUC summarize overall performance, but the number the jet-tagging
literature actually compares is **background rejection at fixed signal
efficiency**: ``Rej_X% = 1 / eps_B`` at ``eps_S = X%``.  ParT, ParticleNet,
L-GATr and LLoCa all report it per signal class, so the ablation needs it to be
comparable with published tables.

Scores follow the JetClass convention (ParT, arXiv:2202.03772, Eq. 2): for
signal class ``c`` the discriminant is ``p_c / (p_c + p_QCD)`` evaluated on
events of class ``c`` and QCD only.

Definitions, and how each was checked (2026-09-09 audit, A11)
-------------------------------------------------------------
* **Working point.** ``eps_B`` is read off the ROC step function at the first
  threshold whose signal efficiency reaches the target.  Tied scores form one
  ROC point and are admitted together, so the achieved efficiency can exceed the
  target by at most one tie group; it is reported alongside.  The previous
  implementation took a linearly interpolated quantile of the signal scores and
  admitted every score ``>=`` it, which on tied inputs could pass 100% of the
  signal when 50% was requested.  Ties are not hypothetical here: probabilities
  were stored as fp16 (~3 significant digits) until the same date.
* **Zero background passes.** ``1/0`` is not infinite rejection, it is an
  unmeasured tail.  ``rej_*`` keeps ``inf`` for backward compatibility, and the
  ``rejection`` detail carries a 95% lower bound from the rule of three
  (``eps_B < 3 / N_B``).  Never average rejection across classes: the paper
  does not, and a mean that drops ``inf`` silently removes the strongest class.
* **Per-class working points.** ParT Table 1 reports each class at ONE point:
  50% for the seven hadronic classes, 99% for ``Hqql``, 99.5% for ``Tbl``.
  Those are always computed (:data:`PAPER_WORKING_POINTS`), together with 0.3
  for the parameter-reduction comparison in arXiv:2608.16061
  (:data:`EXTRA_EFFICIENCIES`), on top of whatever the config asks for.
* **AUC.** The paper's overall AUC is macro **one-vs-one** (sklearn
  ``multi_class='ovo', average='macro'``).  The legacy ``auc`` key is macro
  one-vs-rest and is kept unchanged so old and new ``metrics.jsonl`` agree on
  what that key means; the paper-comparable value is ``auc_ovo``.  With equal
  class counts the two are identical (both are the mean of ``AUC_a(a, b)`` over
  ordered pairs), and JetClass ``val_5M`` is balanced, so historical ``auc``
  values there are numerically the OVO number as well.  Likewise the legacy
  per-class ``auc_vs_qcd`` ranks by raw ``p_c``; the discriminant version is
  ``auc_vs_qcd_disc``.
"""

from __future__ import annotations

import math
from typing import Dict, Iterable, Mapping, Optional, Sequence

import numpy as np

__all__ = [
    "JETCLASS_LABELS",
    "QCD_INDEX",
    "PAPER_WORKING_POINTS",
    "EXTRA_EFFICIENCIES",
    "accuracy",
    "macro_auc",
    "macro_ovo_auc",
    "qcd_discriminant",
    "roc_operating_point",
    "background_rejection",
    "background_rejection_detail",
    "summarize",
    "format_summary",
]

#: JetClass class order as produced by the project's loader.
JETCLASS_LABELS: Sequence[str] = (
    "QCD",
    "Hbb",
    "Hcc",
    "Hgg",
    "H4q",
    "Hqql",
    "Zqq",
    "Wqq",
    "Tbqq",
    "Tbl",
)

#: Index of the background class in :data:`JETCLASS_LABELS`.
QCD_INDEX = 0

#: The signal efficiency at which ParT Table 1 (arXiv:2202.03772) reports each class.
PAPER_WORKING_POINTS: Mapping[str, float] = {
    "Hbb": 0.5,
    "Hcc": 0.5,
    "Hgg": 0.5,
    "H4q": 0.5,
    "Hqql": 0.99,
    "Zqq": 0.5,
    "Wqq": 0.5,
    "Tbqq": 0.5,
    "Tbl": 0.995,
}

#: Always computed in addition to the configured efficiencies. 0.3 is the working point
#: arXiv:2608.16061 uses to show rejection degrading ~25x more than accuracy under parameter
#: reduction, which is the reading the ``lowrank`` arm needs.
EXTRA_EFFICIENCIES: Sequence[float] = (0.3,)

#: Rule of three: with zero of ``N`` background events passing, ``eps_B < 3/N`` at ~95% CL.
_RULE_OF_THREE = 3.0


def accuracy(probs: np.ndarray, labels: np.ndarray) -> float:
    """Top-1 accuracy.

    Parameters
    ----------
    probs : ndarray
        ``(N, C)`` predicted class probabilities.
    labels : ndarray
        ``(N,)`` integer class indices.
    """
    if labels.size == 0:
        return float("nan")
    return float((probs.argmax(axis=1) == labels).mean())


def macro_auc(probs: np.ndarray, labels: np.ndarray) -> float:
    """One-vs-rest macro-averaged ROC AUC (legacy ``auc`` key; NOT the ParT definition).

    Kept with unchanged semantics so pre- and post-2026-09-09 ``metrics.jsonl`` agree on what
    ``auc`` means. The paper-comparable number is :func:`macro_ovo_auc`. Returns NaN when the
    batch does not contain at least two classes, which happens for tiny smoke-test evaluations.
    """
    present = np.unique(labels)
    if present.size < 2:
        return float("nan")

    scores = []
    for class_index in present:
        positive = labels == class_index
        scores.append(_binary_auc(probs[:, class_index], positive))
    return float(np.mean(scores))


def macro_ovo_auc(probs: np.ndarray, labels: np.ndarray) -> float:
    """One-vs-one macro-averaged ROC AUC, the ParT / weaver definition.

    Mirrors scikit-learn's ``roc_auc_score(..., multi_class="ovo", average="macro")``: for every
    unordered pair of present classes ``(a, b)``, restrict to events of those two classes, take
    the AUC of ``p_a`` for ``a`` against ``b`` and of ``p_b`` for ``b`` against ``a``, average the
    two, then average over pairs. Implemented here rather than imported so the training path has
    no scikit-learn dependency, and ties are handled by average ranks as in :func:`_binary_auc`.
    """
    present = np.unique(labels)
    if present.size < 2:
        return float("nan")

    pair_scores = []
    for i, a in enumerate(present):
        for b in present[i + 1:]:
            mask = (labels == a) | (labels == b)
            a_true = labels[mask] == a
            a_score = _binary_auc(probs[mask][:, a], a_true)
            b_score = _binary_auc(probs[mask][:, b], ~a_true)
            pair_scores.append(0.5 * (a_score + b_score))
    return float(np.mean(pair_scores))


def _binary_auc(scores: np.ndarray, positive: np.ndarray) -> float:
    """ROC AUC via the rank-sum (Mann-Whitney U) identity.

    Avoids a scikit-learn dependency in the training path and handles ties
    correctly by averaging tied ranks.
    """
    n_pos = int(positive.sum())
    n_neg = int(positive.size - n_pos)
    if n_pos == 0 or n_neg == 0:
        return float("nan")

    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty(scores.size, dtype=np.float64)
    ranks[order] = np.arange(1, scores.size + 1, dtype=np.float64)

    # Average ranks within tied score groups.
    sorted_scores = scores[order]
    ties_start = 0
    for index in range(1, sorted_scores.size + 1):
        if index == sorted_scores.size or sorted_scores[index] != sorted_scores[ties_start]:
            if index - ties_start > 1:
                span = order[ties_start:index]
                ranks[span] = ranks[span].mean()
            ties_start = index

    rank_sum = ranks[positive].sum()
    return float((rank_sum - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def qcd_discriminant(
    probs: np.ndarray, signal_index: int, background_index: int = QCD_INDEX
) -> np.ndarray:
    """``p_S / (p_S + p_B)`` per event (ParT Eq. 2); 0 where both probabilities are 0."""
    p_s = probs[:, signal_index].astype(np.float64)
    denominator = p_s + probs[:, background_index].astype(np.float64)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(denominator > 0, p_s / denominator, 0.0)


def roc_operating_point(
    signal_scores: np.ndarray, background_scores: np.ndarray, efficiency: float
) -> dict:
    """Background pass count at the first ROC point whose signal efficiency reaches the target.

    The ROC step function is built with tied scores grouped into one point (as
    ``sklearn.metrics.roc_curve(drop_intermediate=False)`` does), thresholds descending. The
    operating point is the highest threshold with ``TPR >= efficiency``, i.e. the smallest
    ``eps_B`` that achieves at least the requested ``eps_S``. Because a whole tie group enters
    together, the achieved efficiency can exceed the target; it is returned so the overshoot is
    visible instead of silent.

    Returns ``threshold``, ``achieved_signal_efficiency``, ``n_signal``, ``n_background``,
    ``n_background_pass``.
    """
    if not 0.0 < efficiency <= 1.0:
        raise ValueError(f"efficiency must be in (0, 1], got {efficiency}")
    n_sig = int(signal_scores.size)
    n_bg = int(background_scores.size)
    if n_sig == 0 or n_bg == 0:
        return {
            "threshold": float("nan"),
            "achieved_signal_efficiency": float("nan"),
            "n_signal": n_sig,
            "n_background": n_bg,
            "n_background_pass": 0,
        }

    scores = np.concatenate([signal_scores, background_scores]).astype(np.float64)
    is_signal = np.concatenate(
        [np.ones(n_sig, dtype=bool), np.zeros(n_bg, dtype=bool)]
    )
    order = np.argsort(-scores, kind="mergesort")          # descending, stable
    scores = scores[order]
    is_signal = is_signal[order]

    # Cumulative counts at each position, then keep the LAST position of every equal-score run:
    # that is the ROC point where the whole tie group has been admitted.
    cum_sig = np.cumsum(is_signal)
    cum_bg = np.cumsum(~is_signal)
    group_end = np.append(scores[1:] != scores[:-1], True)
    tpr = cum_sig[group_end] / n_sig
    bg_pass = cum_bg[group_end]
    thresholds = scores[group_end]

    # `tpr` is non-decreasing along descending thresholds and ends at exactly 1.0, so this always
    # lands inside the array for efficiency <= 1. A small tolerance absorbs the float division so
    # that e.g. 0.5 * n_sig events count as reaching 0.5 exactly.
    index = int(np.searchsorted(tpr, efficiency - 1e-12, side="left"))
    index = min(index, tpr.size - 1)
    return {
        "threshold": float(thresholds[index]),
        "achieved_signal_efficiency": float(tpr[index]),
        "n_signal": n_sig,
        "n_background": n_bg,
        "n_background_pass": int(bg_pass[index]),
    }


def _rejection_record(point: dict) -> dict:
    """Turn an operating point into the reported rejection fields."""
    n_bg = point["n_background"]
    n_pass = point["n_background_pass"]
    record = dict(point)
    if point["n_signal"] == 0 or n_bg == 0:
        record.update(rejection=float("nan"), rejection_lower_bound_95=None,
                      rejection_stat_rel_err=None)
    elif n_pass == 0:
        # Not infinite rejection: no background event was observed in the tail. Rule of three.
        record.update(
            rejection=float("inf"),
            rejection_lower_bound_95=float(n_bg / _RULE_OF_THREE),
            rejection_stat_rel_err=None,
        )
    else:
        record.update(
            rejection=float(n_bg / n_pass),
            rejection_lower_bound_95=None,
            # Counting error on the passing background: relative 1/sqrt(n).
            rejection_stat_rel_err=float(1.0 / math.sqrt(n_pass)),
        )
    return record


def background_rejection_detail(
    probs: np.ndarray,
    labels: np.ndarray,
    signal_index: int,
    efficiencies: Iterable[float],
    background_index: int = QCD_INDEX,
) -> Dict[float, dict]:
    """Rejection at each target efficiency with its operating-point bookkeeping.

    Each value is a dict with ``rejection`` (``1/eps_B``; ``inf`` when nothing passes, ``nan``
    when a class is absent), ``achieved_signal_efficiency``, ``threshold``, ``n_signal``,
    ``n_background``, ``n_background_pass``, ``rejection_lower_bound_95`` (finite only when
    nothing passes) and ``rejection_stat_rel_err`` (``1/sqrt(n_background_pass)``).
    """
    keep = (labels == signal_index) | (labels == background_index)
    scores = qcd_discriminant(probs[keep], signal_index, background_index)
    signal_mask = labels[keep] == signal_index
    signal_scores = scores[signal_mask]
    background_scores = scores[~signal_mask]

    out: Dict[float, dict] = {}
    for eps in efficiencies:
        point = roc_operating_point(signal_scores, background_scores, float(eps))
        out[float(eps)] = _rejection_record(point)
    return out


def background_rejection(
    probs: np.ndarray,
    labels: np.ndarray,
    signal_index: int,
    efficiencies: Iterable[float],
    background_index: int = QCD_INDEX,
) -> Dict[float, float]:
    """``1 / eps_B`` at each target signal efficiency (see :func:`background_rejection_detail`).

    Parameters
    ----------
    probs : ndarray
        ``(N, C)`` predicted probabilities.
    labels : ndarray
        ``(N,)`` integer class indices.
    signal_index : int
        Class treated as signal.
    efficiencies : iterable of float
        Target signal efficiencies, e.g. ``(0.5, 0.99)``.
    background_index : int
        Class treated as background (QCD by convention).

    Returns
    -------
    dict
        Maps each requested efficiency to ``1 / eps_B``; ``inf`` when no
        background event survives the threshold and ``nan`` when either class is
        missing from the sample.
    """
    detail = background_rejection_detail(
        probs, labels, signal_index, efficiencies, background_index
    )
    return {eps: record["rejection"] for eps, record in detail.items()}


def _eff_key(eps: float) -> str:
    """``0.5 -> 'rej_50'``, ``0.995 -> 'rej_99.5'``. The old ``int(round(eps*100))`` mapped 0.995
    to ``rej_100``, colliding with a genuine 100% point."""
    return f"rej_{eps * 100:g}"


def summarize(
    probs: np.ndarray,
    labels: np.ndarray,
    efficiencies: Iterable[float] = (0.5, 0.99),
    class_names: Sequence[str] = JETCLASS_LABELS,
) -> dict:
    """Full metric bundle for one evaluation pass.

    Returns a JSON-serializable dict with overall accuracy, ``auc`` (legacy macro OVR) and
    ``auc_ovo`` (paper), plus for every non-background class: ``auc_vs_qcd`` (legacy, raw
    ``p_c``), ``auc_vs_qcd_disc`` (discriminant), flat ``rej_<pct>`` values at every requested
    efficiency **plus** :data:`EXTRA_EFFICIENCIES` and the class's :data:`PAPER_WORKING_POINTS`
    entry, a ``paper_working_point`` / ``rej_paper`` pair for the Table-1-comparable number, and a
    ``rejection`` detail block keyed by the same ``rej_<pct>`` names.
    """
    requested = tuple(float(e) for e in efficiencies)
    out: dict = {
        "num_jets": int(labels.size),
        "accuracy": accuracy(probs, labels),
        "auc": macro_auc(probs, labels),
        "auc_ovo": macro_ovo_auc(probs, labels),
        "per_class": {},
    }

    for index, name in enumerate(class_names[: probs.shape[1]]):
        if index == QCD_INDEX:
            continue
        paper_eps: Optional[float] = PAPER_WORKING_POINTS.get(name)
        effs = sorted(
            {*requested, *(float(e) for e in EXTRA_EFFICIENCIES),
             *([float(paper_eps)] if paper_eps is not None else [])}
        )
        detail = background_rejection_detail(probs, labels, index, effs)

        keep = (labels == index) | (labels == QCD_INDEX)
        positive = labels[keep] == index
        entry: dict = {
            "auc_vs_qcd": _binary_auc(probs[keep][:, index], positive),
            "auc_vs_qcd_disc": _binary_auc(
                qcd_discriminant(probs[keep], index), positive
            ),
        }
        for eps in effs:
            entry[_eff_key(eps)] = detail[eps]["rejection"]
        entry["paper_working_point"] = paper_eps
        entry["rej_paper"] = detail[float(paper_eps)]["rejection"] if paper_eps is not None else None
        entry["rejection"] = {_eff_key(eps): detail[eps] for eps in effs}
        out["per_class"][name] = entry
    return out


def _fmt_rej(record: Optional[dict]) -> str:
    if record is None:
        return f"{'-':>10}"
    value = record["rejection"]
    if isinstance(value, float) and math.isinf(value):
        bound = record.get("rejection_lower_bound_95")
        return f"{'>' + format(bound, '.0f'):>10}" if bound else f"{'inf':>10}"
    if isinstance(value, float) and math.isnan(value):
        return f"{'nan':>10}"
    return f"{value:>10.1f}"


def format_summary(summary: Mapping) -> str:
    """Render :func:`summarize` output as a compact table for logs.

    One row per class; the column marked ``*`` is that class's ParT Table 1 working point.
    ``>N`` means no background event passed and ``N`` is the 95% lower bound on rejection.
    """
    per_class = summary.get("per_class", {})
    lines = [
        f"jets={summary['num_jets']:,}  "
        f"acc={summary['accuracy']:.4f}  auc_ovr={summary['auc']:.5f}  "
        f"auc_ovo={summary.get('auc_ovo', float('nan')):.5f}"
    ]
    eff_keys = sorted(
        {k for v in per_class.values() for k in v.get("rejection", {})},
        key=lambda k: float(k[len("rej_"):]),
    )
    lines.append(
        f"  {'class':<6} {'auc_disc':>9} " + " ".join(f"{k:>10}" for k in eff_keys)
    )
    for name, values in per_class.items():
        paper = values.get("paper_working_point")
        cells = []
        for key in eff_keys:
            cell = _fmt_rej(values.get("rejection", {}).get(key))
            if paper is not None and key == _eff_key(float(paper)):
                cell = cell[1:] + "*"
            cells.append(cell)
        lines.append(
            f"  {name:<6} {values.get('auc_vs_qcd_disc', float('nan')):>9.5f} " + " ".join(cells)
        )
    return "\n".join(lines)
