"""Combine the per-level figures of one experiment into side-by-side images (2x2 | 4x4 | 8x8).

    python3 results/output/combine_levels.py [experiment_dir] [output_dir]

Every figure that exists once per level is placed in one image, in level order, with a header naming the
figure and each column. Wide figures (tagger curves, training curves) are stacked top-to-bottom instead.
Tagger training curves are combined per seed (one image per input kind and seed) and as a seed x level grid.
Also writes the level-trend charts used in the slides (tagger AUC per seed, observable shifts per level).
"""
import glob
import os
import re
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image, ImageDraw, ImageFont

HERE = os.path.dirname(os.path.abspath(__file__))
EXP = sys.argv[1] if len(sys.argv) > 1 else glob.glob(os.path.join(HERE, "..", "results_qg_*", "qg_*"))[0]
OUT = sys.argv[2] if len(sys.argv) > 2 else HERE
LEVELS = ["pool2x2", "pool4x4", "pool8x8"]
LEVEL_LABEL = {"pool2x2": "2×2 pooling  (LR 64×64)", "pool4x4": "4×4 pooling  (LR 32×32)",
               "pool8x8": "8×8 pooling  (LR 16×16)"}
SEEDS = ["42", "43", "44"]
NAME = os.path.basename(os.path.normpath(EXP))           # e.g. qg_v2_2026-10-13_multiseed


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
    # one common cell size: scale every image to the same width (columns) or height (row)
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
    W = max(grid_w, title_w)                       # never clip the title on narrow (stacked) images
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

# 1. evaluation figures, main and diagnostics: one image per figure, 2x2 | 4x4 | 8x8
for sub, folder in (("", "evaluation"), ("diagnostics", "evaluation_diagnostics")):
    pattern = os.path.join(EXP, "evaluation", "sr_pool2x2", "figures", sub, f"{NAME}_pool2x2_*.png")
    for p2 in sorted(glob.glob(pattern)):
        what = re.sub(rf"^{NAME}_pool2x2_|\.png$", "", os.path.basename(p2))
        paths = [p2.replace("pool2x2", lv) for lv in LEVELS]
        if not all(map(os.path.exists, paths)):
            print("skip (missing level):", what)
            continue
        title = f"{pretty(what)}{'  (diagnostics)' if sub else ''}  —  {NAME}"
        written.append(montage(paths, [LEVEL_LABEL[lv] for lv in LEVELS], title,
                               os.path.join(OUT, folder, f"{what}_levels.png"),
                               vertical=False if what.startswith(("channels", "examples")) else None))

# 2. summary figures that exist per level
for what in ("resnet18-tagger-2x3", "resnet18-tagger-curves", "w1-heatmap"):
    paths = [os.path.join(EXP, "summary", "figures", f"{NAME}_{what}_{lv}.png") for lv in LEVELS]
    if all(map(os.path.exists, paths)):
        written.append(montage(paths, [LEVEL_LABEL[lv] for lv in LEVELS], f"{what.replace('-', ' ')}  —  {NAME}",
                               os.path.join(OUT, "summary", f"{what}_levels.png")))

# 3. transformer training curves per level
paths = [os.path.join(EXP, "models", f"var_transformer_{lv}", "figures",
                      f"{NAME}_var_transformer_{lv}_training-curves.png") for lv in LEVELS]
written.append(montage(paths, [LEVEL_LABEL[lv] for lv in LEVELS], f"VAR transformer training  —  {NAME}",
                       os.path.join(OUT, "training", "var_transformer_levels.png")))


# 4. tagger training curves, seed-wise
def tcurve(kind, seed, lv=None):
    run = f"resnet18_tagger_{kind}{'-' + lv if lv else ''}_seed{seed}"
    return os.path.join(EXP, "models", run, "figures", f"{NAME}_{run}_training-curves.png")


for kind, desc in (("lr", "trained on LR input"), ("sr", "trained on SR output")):
    for s in SEEDS:   # one image per seed: the three levels
        written.append(montage([tcurve(kind, s, lv) for lv in LEVELS], [LEVEL_LABEL[lv] for lv in LEVELS],
                               f"ResNet-18 tagger {desc}, seed {s}  —  {NAME}",
                               os.path.join(OUT, "training", f"tagger_{kind}_seed{s}_levels.png")))
    # all seeds at once: rows = seeds, columns = levels
    written.append(montage([tcurve(kind, s, lv) for s in SEEDS for lv in LEVELS],
                           [f"seed {s}  ·  {LEVEL_LABEL[lv]}" for s in SEEDS for lv in LEVELS],
                           f"ResNet-18 tagger {desc}: seeds (rows) × levels (columns)  —  {NAME}",
                           os.path.join(OUT, "training", f"tagger_{kind}_seeds-x-levels.png"), cols=3))
written.append(montage([tcurve("hr", s) for s in SEEDS], [f"HR reference, seed {s}" for s in SEEDS],
                       f"ResNet-18 tagger trained on HR (no pooling)  —  {NAME}",
                       os.path.join(OUT, "training", "tagger_hr_seeds.png"), vertical=True))

# 5. level-trend charts built from the summary tables
SUM = os.path.join(EXP, "summary", f"{NAME}_")
INK, MUTED, GRID = "#1f2a37", "#5b6675", "#e3e7ec"
COL = {"hr": "#6b7280", "lr": "#2a6fdb", "sr": "#d1495b"}
x_lab = ["2×2", "4×4", "8×8"]
plt.rcParams.update({"font.size": 11, "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED,
                     "ytick.color": MUTED, "axes.spines.top": False, "axes.spines.right": False})

runs = pd.read_csv(SUM + "resnet18-tagger-runs.csv")
fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))
for ax, metric, ylab in ((axes[0], "auc", "test AUC"), (axes[1], "rej_at_eff50", "gluon rejection at 50% quark eff.")):
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
    ax.set_xticks(range(3), x_lab)
    ax.set_xlabel("pooling level")
    ax.set_ylabel(ylab)
    ax.grid(axis="y", color=GRID)
axes[0].legend(frameon=False, loc="lower left")
fig.suptitle("ResNet-18 quark/gluon tagger vs pooling level (small dots = individual seeds 42/43/44)", color=INK)
fig.tight_layout()
p = os.path.join(OUT, "summary", "tagger_auc_rejection_by_seed.png")
fig.savefig(p, dpi=160)
plt.close(fig)
written.append(p)

obs = pd.read_csv(SUM + "sr-observables.csv")
sr = obs[obs.method == "sr"]
keys = [("sum_tracks", "track energy"), ("jet_pt", "jet pT"), ("jet_mass", "jet mass"), ("ptD", "pT_D"),
        ("girth", "girth"), ("n_hits_tracks", "track hits")]
fig, ax = plt.subplots(figsize=(12, 4.8))
w = 0.26
shades = ["#9cc0f0", "#4f8be0", "#1d4f9e"]
for i, lv in enumerate(LEVELS):
    vals = [sr[(sr.level == lv) & (sr.observable == k)].mean_shift_over_sigma.iloc[0] for k, _ in keys]
    bars = ax.bar([j + (i - 1) * w for j in range(len(keys))], vals, w - 0.03, color=shades[i], label=x_lab[i])
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + (0.04 if v >= 0 else -0.04), f"{v:+.2f}", ha="center",
                va="bottom" if v >= 0 else "top", fontsize=8, color=MUTED)
ax.axhline(0, color=INK, lw=1)
ax.set_xticks(range(len(keys)), [lab for _, lab in keys])
ax.set_ylabel("(SR mean − HR mean) / σ_HR")
ax.grid(axis="y", color=GRID)
ax.legend(title="pooling", frameon=False, ncol=3, loc="upper right")
ax.set_title("Mean shift of the SR output from HR, per observable and level (0 = unbiased)", color=INK)
fig.tight_layout()
p = os.path.join(OUT, "summary", "sr_mean_shift_by_level.png")
fig.savefig(p, dpi=160)
plt.close(fig)
written.append(p)

print(f"{len(written)} images in {OUT}")
for p in written:
    print("  ", os.path.relpath(p, OUT))
