"""Result layout and small I/O helpers.

Layout (all names are stable, so later stages can find earlier outputs):

    {drive_root}/                              e.g. MyDrive/GSoC_SuperRes/experiments or $SCRATCH/superres/experiments
      {dataset}/
        {date}_{version}_{tag}/                one experiment, e.g. 2026-10-06_v2_multiseed
          README_layout.md
          run_info.json                        start time, git commit, GPU, config of every session
          data_meta.json                       which events were cached (counts, scales, source files)
          models/
            vqvae_tokenizer/                   multi-scale VQ-VAE (one per experiment)
            var_transformer_{level}/           conditional VAR, one per LR level, e.g. var_transformer_pool2x2
            resnet18_tagger_{input}_seed{seed}/ qg quark/gluon tagger, e.g. resnet18_tagger_sr-pool2x2_seed42
            cnn_tagger_{input}_seed{seed}/     the same with tagger.arch=cnn
            cnn_regressor_{input}_seed{seed}/  calo incident-energy regressor
                config.json  history.csv  metrics.json  best.pt  last.pt  figures/
          evaluation/
            sr_{level}/                        physics observables, paired.csv, metrics.json, figures/
                figures/diagnostics/           the same with diagnostic methods added
          tuning/                              Optuna studies: {name}.db, {name}_trials.csv, {name}_best.json
          summary/                             {name}_report.md, tables (*.csv), figures/
                                               tagger files name the tagger, e.g. {name}_resnet18-tagger-2x3_pool2x2.csv

name    : {dataset}_{version}_{date}_{tag}, e.g. qg_v2_2026-10-06_multiseed. Every file name starts with it
          (figures: {name}_{model or level}_{what}.png), so a file copied elsewhere still says where it is from.
          '_' separates fields, '-' joins words inside a field.
input   : hr | lr-{level} | sr-{level} | uniform-{level} | srproj-{level}
Every figure is written as .png and .pdf.
"""
import csv
import datetime as _dt
import json
import os
import re
import shutil

import numpy as np

LAYOUT_README = __doc__


def slug(s):
    return re.sub(r"[^A-Za-z0-9.\-]+", "-", str(s)).strip("-")


def run_date(cfg):
    return slug(cfg.get("run_date") or "undated")


def run_tag(cfg):
    return slug(cfg.get("run_tag") or "untagged")


def version(cfg):
    return slug(cfg.get("version", "v1"))


def experiment_name(cfg):
    """{date}_{version}_{tag}: the experiment folder, e.g. 2026-10-06_v2_multiseed."""
    return f"{run_date(cfg)}_{version(cfg)}_{run_tag(cfg)}"


def prefix(cfg):
    """{dataset}_{version}_{date}_{tag}: the start of every file name, e.g. qg_v2_2026-10-06_multiseed."""
    return f"{slug(cfg['dataset'])}_{version(cfg)}_{run_date(cfg)}_{run_tag(cfg)}"


def model_tag(cfg):
    """Marks this experiment's model outputs in the shared local cache."""
    return prefix(cfg)


# stage -> (group folder, model name)
_RUNS = {"vqvae": ("models", "vqvae_tokenizer"), "var": ("models", "var_transformer"),
         "tagger": ("models", None), "sreval": ("evaluation", "sr")}


def tagger_arch(cfg):
    """cnn | resnet18 (runs from before tagger.arch existed are cnn)."""
    return "resnet18" if cfg.get("tagger", {}).get("arch", "cnn") == "resnet18" else "cnn"


def _tagger_name(cfg):
    task = "tagger" if cfg.get("tagger", {}).get("task", "classification") == "classification" else "regressor"
    return f"{tagger_arch(cfg)}_{task}"


def tagger_tag(cfg):
    """The tagger in summary file names, e.g. resnet18-tagger, cnn-regressor."""
    return _tagger_name(cfg).replace("_", "-")


def tagger_title(cfg):
    """The tagger in figure titles and report headings, e.g. ResNet-18 tagger."""
    return f"{ {'cnn': 'CNN', 'resnet18': 'ResNet-18'}[tagger_arch(cfg)]} {_tagger_name(cfg).split('_')[1]}"


def run_folder(cfg, stage, variant=None, seed=None):
    """Folder name of one trained model or evaluation, e.g. var_transformer_pool2x2, cnn_tagger_hr_seed42.
    The tokenizer and the transformers are trained once per experiment (config seed), so their names
    carry no seed; taggers are trained with several seeds."""
    group, name = _RUNS[stage]
    name = name or _tagger_name(cfg)
    if stage != "vqvae" and variant:
        name += f"_{slug(variant)}"
    if stage == "tagger" and seed is not None:
        name += f"_seed{seed}"
    return name


def run_path(cfg, stage, variant=None, seed=None):
    """Where a run lives, without creating it."""
    return os.path.join(experiment_root(cfg), _RUNS[stage][0], run_folder(cfg, stage, variant, seed))


def run_glob(cfg, stage):
    """Glob pattern matching every run of a stage in this experiment."""
    group, name = _RUNS[stage]
    return os.path.join(experiment_root(cfg), group, (name or _tagger_name(cfg)) + "*")


def run_name(cfg, stage, variant=None, seed=None):
    """Full label of a run: {prefix}_{folder}, e.g. qg_v2_2026-10-06_multiseed_cnn_tagger_hr_seed42."""
    return f"{prefix(cfg)}_{run_folder(cfg, stage, variant, seed)}"


def now():
    return _dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def _jsonable(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (set, tuple)):
        return list(o)
    return str(o)


def save_json(path, obj):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    tmp = f"{path}.{os.getpid()}.tmp"  # per process: parallel jobs may write the same file
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=2, default=_jsonable)
    os.replace(tmp, path)


def load_json(path):
    with open(path) as f:
        return json.load(f)


def experiment_root(cfg):
    """{drive_root}/{dataset}/{date}_{version}_{tag}: everything one experiment writes."""
    root = os.path.join(cfg["paths"]["drive_root"], cfg["dataset"], experiment_name(cfg))
    os.makedirs(root, exist_ok=True)
    readme = os.path.join(root, "README_layout.md")
    if not os.path.exists(readme):
        with open(readme, "w") as f:
            f.write("# Result layout\n\n```" + LAYOUT_README + "```\n")
    return root


dataset_root = experiment_root


def save_run_info(cfg, **session):
    """run_info.json: when the experiment started, plus one entry per session (code commit, GPU, config)."""
    path = os.path.join(experiment_root(cfg), "run_info.json")
    info = load_json(path) if os.path.exists(path) else {
        "experiment": experiment_name(cfg), "name": prefix(cfg), "started": now(), "sessions": []}
    info["sessions"].append({"time": now(), **session, "config": cfg})
    save_json(path, info)
    return info


def summary_dir(cfg, sub=""):
    d = os.path.join(dataset_root(cfg), "summary", sub)
    os.makedirs(d, exist_ok=True)
    return d


class RunDir:
    def __init__(self, cfg, stage, variant, seed=None):
        self.cfg = cfg
        self.name = run_name(cfg, stage, variant, seed)
        self.path = run_path(cfg, stage, variant, seed)
        self.fig_dir = os.path.join(self.path, "figures")
        os.makedirs(self.fig_dir, exist_ok=True)

    def file(self, name):
        return os.path.join(self.path, name)

    def exists(self, name):
        return os.path.exists(self.file(name))

    def save_config(self, extra=None):
        save_json(self.file("config.json"),
                  {"run": self.name, "created": now(), "config": self.cfg, **(extra or {})})

    def save_metrics(self, metrics):
        save_json(self.file("metrics.json"), {"run": self.name, "saved": now(), **metrics})

    def reset_history(self):
        if self.exists("history.csv"):
            os.remove(self.file("history.csv"))

    def log(self, row):
        path = self.file("history.csv")
        new = not os.path.exists(path)
        with open(path, "a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(row))
            if new:
                w.writeheader()
            w.writerow(row)

    def plot_history(self, groups):
        """Training curves from history.csv -> figures/{run name}_training-curves.png|pdf.
        groups: list of (panel title, [columns]); columns missing from the history are skipped."""
        path = self.file("history.csv")
        if not os.path.exists(path):
            return
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import pandas as pd
        h = pd.read_csv(path)
        groups = [(t, [c for c in cols if c in h]) for t, cols in groups]
        groups = [(t, cols) for t, cols in groups if cols]
        if h.empty or not groups:
            return
        fig, axes = plt.subplots(1, len(groups), figsize=(4.2 * len(groups), 3.3), squeeze=False)
        for ax, (title, cols) in zip(axes[0], groups):
            for c in cols:
                ax.plot(h["epoch"], h[c], "o-", ms=3, label=c)
            ax.set_title(title, fontsize=9)
            ax.set_xlabel("epoch")
            ax.legend(fontsize=7, frameon=False)
        fig.suptitle(self.name, fontsize=9)
        fig.tight_layout()
        save_figure(fig, self.fig_dir, f"{self.name}_training-curves")
        plt.close(fig)


def study_paths(cfg, stage, variant):
    name = run_name(cfg, stage, variant)  # e.g. ..._var_transformer_all, ..._cnn_tagger_hr
    drive_dir = os.path.join(dataset_root(cfg), "tuning")
    local_dir = os.path.join(cfg["paths"]["cache_root"], "optuna", model_tag(cfg))
    os.makedirs(drive_dir, exist_ok=True)
    os.makedirs(local_dir, exist_ok=True)
    return {
        "name": name,
        # SQLite locking is unreliable on the Drive FUSE mount, so the live DB is
        # local and copied to Drive after every trial.
        "db_local": os.path.join(local_dir, name + ".db"),
        "db_drive": os.path.join(drive_dir, name + ".db"),
        "trials_csv": os.path.join(drive_dir, name + "_trials.csv"),
        "best_json": os.path.join(drive_dir, name + "_best.json"),
    }


def copy_file(src, dst):
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    shutil.copy2(src, dst)


def save_figure(fig, directory, name):
    os.makedirs(directory, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(directory, f"{name}.{ext}"), dpi=150, bbox_inches="tight")
