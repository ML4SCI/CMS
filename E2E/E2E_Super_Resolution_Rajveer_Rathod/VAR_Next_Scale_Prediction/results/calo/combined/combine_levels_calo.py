"""Combine the per-level figures of the CaloChallenge experiment into side-by-side images
(1x2x1 | 1x4x3 | 3x8x3), like results/output/combine_levels.py does for qg.

    python3 results/output_calo/combine_levels_calo.py [experiment_dir] [output_dir]

Evaluation figures (main and diagnostics), summary figures and VAR training curves: one image per figure,
levels side by side. Energy-regressor training curves: per seed (three levels) and as a seed x level grid.
Also writes level-trend charts: regressor resolution and bias per seed, and the SR mean shift per observable.
"""
import glob
import os
import re
import shutil
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image, ImageDraw, ImageFont

HERE = os.path.dirname(os.path.abspath(__file__))
EXP = sys.argv[1] if len(sys.argv) > 1 else glob.glob(os.path.join(HERE, "..", "results_calo_*", "calo_*"))[0]
OUT = sys.argv[2] if len(sys.argv) > 2 else HERE
LEVELS = ["pool1x2x1", "pool1x4x3", "pool3x8x3"]
# HR is 48 layers x 16 angular x 9 radial cells
LEVEL_LABEL = {"pool1x2x1": "1×2×1 pooling  (LR 48×8×9)", "pool1x4x3": "1×4×3 pooling  (LR 48×4×3)",
               "pool3x8x3": "3×8×3 pooling  (LR 16×2×3)"}
X_LAB = ["1×2×1", "1×4×3", "3×8×3"]
SEEDS = ["42", "43", "44"]
NAME = os.path.basename(os.path.normpath(EXP))           # e.g. calo_v2_2026-10-23_multiseed


def font(size, bold=False):
    for path in (["/System/Library/Fonts/HelveticaNeue.ttc"] if bold else []) + [
            "/System/Library/Fonts/Helvetica.ttc", "/Library/Fonts/Arial.ttf",
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"]:
        try:
            return ImageFont.truetype(path, size, index=1 if bold and path.endswith(".ttc") else 0)
        except OSError:
            continue
    return ImageFont.load_default(size)


def montage(paths, labels, title, out_path, vertical=None, cols=None):
    """Paste images in a row (or column, or grid with `cols`), each under its label, under a title."""
    imgs = [Image.open(p).convert("RGB") for p in paths]
    if vertical is None and cols is None:
        vertical = sum(i.width / i.height for i in imgs) / len(imgs) > 1.6
    cols = cols or (1 if vertical else len(imgs))
    rows = -(-len(imgs) // cols)
    if cols == 1:
        w = max(i.width for i in imgs)
        imgs = [i.resize((w, round(i.height * w / i.width)), Image.LANCZOS) for i in imgs]
    else:
        h = max(i.height for i in imgs)
        imgs = [i.resize((round(i.width * h / i.height), h), Image.LANCZOS) for i in imgs]
    cw, ch = max(i.width for i in imgs), max(i.height for i in imgs)
    unit = max(14, cw // 34)
    f_title, f_label = font(round(unit * 1.5), True), font(unit, True)
    pad, lab_h, title_h = unit, round(unit * 2.2), round(unit * 3)
    grid_w = cols * cw + (cols + 1) * pad
    title_w = round(ImageDraw.Draw(Image.new("RGB", (1, 1))).textlength(title, font=f_title)) + 2 * pad
    W = max(grid_w, title_w)
    x0 = (W - grid_w) // 2
    H = title_h + rows * (lab_h + ch) + (rows + 1) * pad
    canvas = Image.new("RGB", (W, H), "white")
    d = ImageDraw.Draw(canvas)
    d.text((W // 2, title_h // 2 + pad // 2), title, fill="#111111", font=f_title, anchor="mm")
    for k, (im, lab) in enumerate(zip(imgs, labels)):
        r, c = divmod(k, cols)
        x = x0 + pad + c * (cw + pad)
        y = title_h + pad + r * (lab_h + ch + pad)
        d.rounded_rectangle([x, y, x + cw, y + lab_h - unit // 3], radius=unit // 3, fill="#eef1f5")
        d.text((x + cw // 2, y + (lab_h - unit // 3) // 2), lab, fill="#1f2a37", font=f_label, anchor="mm")
        canvas.paste(im, (x + (cw - im.width) // 2, y + lab_h))
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    canvas.save(out_path, optimize=True)
    return out_path


def pretty(what):
    return what.replace("hist_", "histogram: ").replace("scatter_", "SR vs HR per event: ") \
               .replace("profile_", "profile: ").replace("_", " ")


written = []
L0 = LEVELS[0]

# 1. evaluation figures, main and diagnostics: one image per figure, levels side by side
for sub, folder in (("", "evaluation"), ("diagnostics", "evaluation_diagnostics")):
    pattern = os.path.join(EXP, "evaluation", f"sr_{L0}", "figures", sub, f"{NAME}_{L0}_*.png")
    for p0 in sorted(glob.glob(pattern)):
        what = re.sub(rf"^{NAME}_{L0}_|\.png$", "", os.path.basename(p0))
        paths = [p0.replace(L0, lv) for lv in LEVELS]
        if not all(map(os.path.exists, paths)):
            print("skip (missing level):", what)
            continue
        title = f"{pretty(what)}{'  (diagnostics)' if sub else ''}  —  {NAME}"
        written.append(montage(paths, [LEVEL_LABEL[lv] for lv in LEVELS], title,
                               os.path.join(OUT, folder, f"{what}_levels.png"),
                               vertical=False if what.startswith("examples") else None))

# 2. summary figures that exist per level
for what in ("cnn-regressor-2x3", "cnn-regressor-curves", "w1-heatmap"):
    paths = [os.path.join(EXP, "summary", "figures", f"{NAME}_{what}_{lv}.png") for lv in LEVELS]
    if all(map(os.path.exists, paths)):
        written.append(montage(paths, [LEVEL_LABEL[lv] for lv in LEVELS], f"{what.replace('-', ' ')}  —  {NAME}",
                               os.path.join(OUT, "summary", f"{what}_levels.png")))
# summary figures that already cover all levels: copied as they are
for what in ("cnn-regressor-mean_binned_resolution-vs-level", "energy-resolution-vs-E"):
    src = os.path.join(EXP, "summary", "figures", f"{NAME}_{what}.png")
    if os.path.exists(src):
        dst = os.path.join(OUT, "summary", f"{what}.png")
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        shutil.copy(src, dst)
        written.append(dst)

# 3. training curves: tokenizer (one model) and transformer per level
src = os.path.join(EXP, "models", "vqvae_tokenizer", "figures", f"{NAME}_vqvae_tokenizer_training-curves.png")
if os.path.exists(src):
    written.append(montage([src], ["VQ-VAE tokenizer (shared by all levels)"], f"Tokenizer training  —  {NAME}",
                           os.path.join(OUT, "training", "vqvae_tokenizer.png")))
paths = [os.path.join(EXP, "models", f"var_transformer_{lv}", "figures",
                      f"{NAME}_var_transformer_{lv}_training-curves.png") for lv in LEVELS]
written.append(montage(paths, [LEVEL_LABEL[lv] for lv in LEVELS], f"VAR transformer training  —  {NAME}",
                       os.path.join(OUT, "training", "var_transformer_levels.png")))


# 4. energy-regressor training curves, seed-wise
def rcurve(kind, seed, lv=None):
    run = f"cnn_regressor_{kind}{'-' + lv if lv else ''}_seed{seed}"
    return os.path.join(EXP, "models", run, "figures", f"{NAME}_{run}_training-curves.png")


for kind, desc in (("lr", "trained on LR input"), ("sr", "trained on SR output")):
    for s in SEEDS:
        written.append(montage([rcurve(kind, s, lv) for lv in LEVELS], [LEVEL_LABEL[lv] for lv in LEVELS],
                               f"CNN energy regressor {desc}, seed {s}  —  {NAME}",
                               os.path.join(OUT, "training", f"regressor_{kind}_seed{s}_levels.png")))
    written.append(montage([rcurve(kind, s, lv) for s in SEEDS for lv in LEVELS],
                           [f"seed {s}  ·  {LEVEL_LABEL[lv]}" for s in SEEDS for lv in LEVELS],
                           f"CNN energy regressor {desc}: seeds (rows) × levels (columns)  —  {NAME}",
                           os.path.join(OUT, "training", f"regressor_{kind}_seeds-x-levels.png"), cols=3))
written.append(montage([rcurve("hr", s) for s in SEEDS], [f"HR reference, seed {s}" for s in SEEDS],
                       f"CNN energy regressor trained on HR (no pooling)  —  {NAME}",
                       os.path.join(OUT, "training", "regressor_hr_seeds.png"), vertical=True))

# 5. level-trend charts built from the summary tables
SUM = os.path.join(EXP, "summary", f"{NAME}_")
INK, MUTED, GRID = "#1f2a37", "#5b6675", "#e3e7ec"
COL = {"hr": "#6b7280", "lr": "#2a6fdb", "sr": "#d1495b"}
plt.rcParams.update({"font.size": 11, "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED,
                     "ytick.color": MUTED, "axes.spines.top": False, "axes.spines.right": False})

runs = pd.read_csv(SUM + "cnn-regressor-runs.csv")
fig, axes = plt.subplots(1, 3, figsize=(16, 4.6))
for ax, metric, ylab in ((axes[0], "resolution", "energy resolution (lower is better)"),
                         (axes[1], "mean_binned_resolution", "resolution, mean over energy bins"),
                         (axes[2], "response_bias", "response bias (0 is unbiased)")):
    hr = runs[runs.method == "hr"][metric]
    ax.axhspan(hr.min(), hr.max(), color=COL["hr"], alpha=0.18, lw=0)
    ax.axhline(hr.mean(), color=COL["hr"], lw=2, label="HR (truth), 3 seeds")
    for m, lab in (("lr", "LR input"), ("sr", "SR output")):
        sub = runs[runs.method == m]
        mean = [sub[sub.level == lv][metric].mean() for lv in LEVELS]
        ax.plot(range(3), mean, color=COL[m], lw=2, marker="o", ms=8, label=f"{lab} (mean)", zorder=3)
        for i, lv in enumerate(LEVELS):
            v = sub[sub.level == lv][metric]
            ax.scatter([i + (0.06 if m == "sr" else -0.06)] * len(v), v, s=22, color=COL[m], alpha=0.55,
                       edgecolor="white", lw=0.8, zorder=4)
    ax.set_xticks(range(3), X_LAB)
    ax.set_xlabel("pooling level")
    ax.set_ylabel(ylab)
    ax.grid(axis="y", color=GRID)
axes[2].axhline(0, color=INK, lw=1)
axes[0].legend(frameon=False, loc="upper left")
fig.suptitle("CNN energy regressor vs pooling level (small dots = individual seeds 42/43/44)", color=INK)
fig.tight_layout()
p = os.path.join(OUT, "summary", "regressor_resolution_by_seed.png")
fig.savefig(p, dpi=160)
plt.close(fig)
written.append(p)

obs = pd.read_csv(SUM + "sr-observables.csv")
sr = obs[(obs.method == "sr") & (obs.role == "main")]
keys = [("e_total", "total energy"), ("n_hits", "hit cells"), ("depth_centroid", "depth centroid"),
        ("depth_width", "depth width"), ("r_centroid", "radial centroid"), ("r_width", "radial width"),
        ("frac_early", "early fraction"), ("frac_late", "late fraction")]
keys = [(k, lab) for k, lab in keys if k in set(sr.observable)]
fig, ax = plt.subplots(figsize=(13, 4.8))
w = 0.26
shades = ["#9cc0f0", "#4f8be0", "#1d4f9e"]
for i, lv in enumerate(LEVELS):
    vals = [sr[(sr.level == lv) & (sr.observable == k)].mean_shift_over_sigma.iloc[0] for k, _ in keys]
    bars = ax.bar([j + (i - 1) * w for j in range(len(keys))], vals, w - 0.03, color=shades[i], label=X_LAB[i])
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + (0.02 if v >= 0 else -0.02), f"{v:+.2f}", ha="center",
                va="bottom" if v >= 0 else "top", fontsize=8, color=MUTED)
ax.axhline(0, color=INK, lw=1)
ax.set_xticks(range(len(keys)), [lab for _, lab in keys])
ax.set_ylabel("(SR mean − HR mean) / σ_HR")
ax.grid(axis="y", color=GRID)
ax.legend(title="pooling", frameon=False, ncol=3, loc="lower left")
ax.set_title("Mean shift of the SR output from HR, per observable and level (0 = unbiased)", color=INK)
fig.tight_layout()
p = os.path.join(OUT, "summary", "sr_mean_shift_by_level.png")
fig.savefig(p, dpi=160)
plt.close(fig)
written.append(p)

print(f"{len(written)} images in {OUT}")
for p in written:
    print("  ", os.path.relpath(p, OUT))
