"""Run provenance (git) and compact eval artifacts for metric recomputation.

Eval checkpoints store only ``probs`` (float32; float16 before 2026-09-09) and
``labels`` (uint8) per step.
Any metric in :mod:`ablation.metrics` can be recomputed offline without keeping
every scalar in ``metrics.jsonl`` — including rejection at efficiencies other
than the ones used during training.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any, Mapping, Optional

import numpy as np

__all__ = [
    "git_provenance",
    "predictions_dir",
    "save_eval_predictions",
    "load_eval_predictions",
    "stored_probs_dtype",
    "provenance_json",
]


def _run_git(args: list[str], cwd: Path) -> Optional[str]:
    try:
        out = subprocess.run(
            ["git", *args],
            cwd=cwd,
            capture_output=True,
            text=True,
            check=True,
            timeout=5,
        )
        return out.stdout.strip()
    except (subprocess.SubprocessError, FileNotFoundError, OSError):
        return None


def find_git_root(start: Optional[Path] = None) -> Optional[Path]:
    """Walk parents from ``start`` (default: this file) looking for ``.git``."""
    here = (start or Path(__file__)).resolve()
    for parent in [here, *here.parents]:
        if (parent / ".git").exists():
            return parent
    return None


def git_provenance(start: Optional[Path] = None) -> dict[str, Any]:
    """Best-effort git metadata for reproducibility logging."""
    root = find_git_root(start)
    if root is None:
        return {
            "git_root": None,
            "git_commit": None,
            "git_commit_short": None,
            "git_branch": None,
            "git_dirty": None,
            "git_describe": None,
        }

    commit = _run_git(["rev-parse", "HEAD"], root)
    dirty_out = _run_git(["status", "--porcelain"], root)
    branch = _run_git(["branch", "--show-current"], root)
    describe = _run_git(["describe", "--always", "--dirty", "--tags"], root)

    return {
        "git_root": str(root),
        "git_commit": commit,
        "git_commit_short": commit[:7] if commit else None,
        "git_branch": branch or None,
        "git_dirty": bool(dirty_out) if dirty_out is not None else None,
        "git_describe": describe,
    }


def provenance_json(provenance: Mapping[str, Any]) -> str:
    return json.dumps(dict(provenance), indent=2, sort_keys=True)


def predictions_dir(run_dir: Path) -> Path:
    return Path(run_dir) / "predictions"


def save_eval_predictions(
    run_dir: Path,
    step: int,
    probs: np.ndarray,
    labels: np.ndarray,
    probs_dtype: type = np.float32,
) -> Path:
    """Write one eval's pooled predictions (metric-complete).

    ``probs`` are stored as ``float32`` since 2026-09-09. Archives before that date are
    ``float16`` (~3 significant digits), which rounds the S-vs-QCD discriminant into tied groups
    in the far tail where Rej99 / Rej99.5 are read; nothing offline can undo that, so
    :func:`load_eval_predictions` reports the stored dtype and recomputed metrics from old
    archives must carry it. Cost: 5M jets x 10 classes x 4 B = 200 MB per eval before
    compression, vs 100 MB for fp16.
    """
    import os
    import tempfile

    out_dir = predictions_dir(run_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"step_{step:07d}.npz"
    fd, tmp = tempfile.mkstemp(dir=out_dir, prefix=f"step_{step:07d}_", suffix=".npz")
    os.close(fd)
    try:
        np.savez_compressed(
            tmp,
            step=np.int64(step),
            probs=np.asarray(probs, dtype=probs_dtype),
            labels=np.asarray(labels, dtype=np.uint8),
        )
        os.replace(tmp, path)
    except Exception:
        if os.path.exists(tmp):
            os.unlink(tmp)
        raise
    return path


def load_eval_predictions(path: Path) -> tuple[int, np.ndarray, np.ndarray]:
    """Return ``(step, probs float32, labels int64)``. See :func:`stored_probs_dtype`."""
    payload = np.load(path)
    step = int(payload["step"])
    probs = payload["probs"].astype(np.float32)
    labels = payload["labels"].astype(np.int64)
    return step, probs, labels


def stored_probs_dtype(path: Path) -> np.dtype:
    """The dtype ``probs`` were archived in -- ``float16`` for pre-2026-09-09 runs."""
    return np.load(path)["probs"].dtype
