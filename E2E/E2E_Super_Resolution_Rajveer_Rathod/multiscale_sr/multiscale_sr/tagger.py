"""Jet tagger — a small CNN classifier used to measure tagging efficiency.

Motivation
----------
Pixel metrics (L1, PSNR) and energy response tell you whether the SR image
*looks* right. They do not tell you whether it preserves the physics needed for
the downstream task: classifying the jet (the `y` label in the parquet data,
q/g-style QCD tagging). "Tagging efficiency" answers that directly:

    train a tagger on the jet label, then compare its AUC when fed
    HR (ground truth), LR (bicubic upsampled), and SR (generator output).

If AUC(SR) approaches AUC(HR) and clearly beats AUC(LR), the super-resolution
is preserving class-discriminative structure, not just minimizing pixel error.

The tagger operates at HR resolution; LR inputs are bicubic-upsampled to HR so
all three sources share an identical tensor shape and the tagger is unchanged
across them.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor, nn


class JetTagger(nn.Module):
    """Compact CNN: (3, H, W) -> single class logit.

    Deliberately small — the point is a consistent, quick-to-train probe of
    class-discriminative content, not a state-of-the-art tagger. Same capacity
    is used for every image source so comparisons are fair.
    """

    def __init__(self, in_channels: int = 3, width: int = 32) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, width, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(width),
            nn.ReLU(inplace=True),
            nn.Conv2d(width, width * 2, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(width * 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(width * 2, width * 4, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(width * 4),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
        )
        self.head = nn.Linear(width * 4, 1)

    def forward(self, x: Tensor) -> Tensor:
        h = self.features(x).flatten(1)
        return self.head(h).squeeze(1)  # (B,) logits


def roc_auc(scores: np.ndarray, labels: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    """Return (fpr, tpr, auc). Uses sklearn if present, else a numpy fallback."""
    try:
        from sklearn.metrics import roc_auc_score, roc_curve

        # Degenerate case: only one class present -> AUC undefined.
        if len(np.unique(labels)) < 2:
            return np.array([0.0, 1.0]), np.array([0.0, 1.0]), float("nan")
        fpr, tpr, _ = roc_curve(labels, scores)
        return fpr, tpr, float(roc_auc_score(labels, scores))
    except ImportError:
        pass

    # numpy fallback (threshold sweep + trapezoid), mirrors SRGAN/plot_roc_auc.py.
    if len(np.unique(labels)) < 2:
        return np.array([0.0, 1.0]), np.array([0.0, 1.0]), float("nan")
    order = np.argsort(-scores)
    s = scores[order]
    y = labels[order]
    thresholds = np.unique(s)[::-1]
    pos = max((labels == 1).sum(), 1)
    neg = max((labels == 0).sum(), 1)
    fpr_list, tpr_list = [0.0], [0.0]
    for t in thresholds:
        pred = s >= t
        tp = (pred & (y == 1)).sum()
        fp = (pred & (y == 0)).sum()
        tpr_list.append(tp / pos)
        fpr_list.append(fp / neg)
    fpr = np.array(fpr_list)
    tpr = np.array(tpr_list)
    auc = float(abs(np.trapezoid(tpr, fpr)))
    return fpr, tpr, auc


def train_tagger(
    images: Tensor,
    labels: Tensor,
    device: torch.device,
    width: int = 32,
    epochs: int = 15,
    batch_size: int = 128,
    lr: float = 1e-3,
    seed: int = 0,
    verbose: bool = False,
) -> JetTagger:
    """Train a JetTagger on (N,3,H,W) images with binary labels (N,).

    Images are expected already normalized (same pipeline as the SR model).
    Returns the trained tagger in eval mode.
    """
    g = torch.Generator().manual_seed(seed)
    n = images.shape[0]
    tagger = JetTagger(in_channels=images.shape[1], width=width).to(device)
    opt = torch.optim.Adam(tagger.parameters(), lr=lr)
    loss_fn = nn.BCEWithLogitsLoss()

    tagger.train()
    for epoch in range(epochs):
        perm = torch.randperm(n, generator=g)
        running = 0.0
        for i in range(0, n, batch_size):
            idx = perm[i : i + batch_size]
            xb = images[idx].to(device)
            yb = labels[idx].to(device).float()
            opt.zero_grad(set_to_none=True)
            logits = tagger(xb)
            loss = loss_fn(logits, yb)
            loss.backward()
            opt.step()
            running += loss.item() * xb.shape[0]
        if verbose:
            print(f"  [tagger] epoch {epoch + 1:02d} loss={running / max(n, 1):.4f}")
    tagger.eval()
    return tagger


@torch.no_grad()
def tagger_scores(tagger: JetTagger, images: Tensor, device: torch.device, batch_size: int = 256) -> np.ndarray:
    """Return sigmoid probabilities (N,) for a set of images."""
    tagger.eval()
    out: list[np.ndarray] = []
    for i in range(0, images.shape[0], batch_size):
        xb = images[i : i + batch_size].to(device)
        out.append(torch.sigmoid(tagger(xb)).cpu().numpy())
    return np.concatenate(out) if out else np.array([], dtype=np.float32)


@torch.no_grad()
def eval_tagger_auc(tagger: JetTagger, images: Tensor, labels: Tensor, device: torch.device) -> float:
    """Convenience: AUC of a tagger on (images, labels)."""
    scores = tagger_scores(tagger, images, device)
    _, _, auc = roc_auc(scores, labels.cpu().numpy())
    return auc
