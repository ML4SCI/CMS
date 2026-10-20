"""Stage runner.

    python -m superres.pipeline --dataset qg --stages download,prepare,train_vqvae
    python -m superres.pipeline --dataset calo --stages all --set vqvae.epochs=10
    python -m superres.pipeline --dataset qg --stages all --smoke        # quick end-to-end check

Stages (in order): download, prepare, tune_vqvae, train_vqvae, tune_var, train_var,
generate, eval_sr, tune_tagger, train_taggers, tagger_xeval, summarize

tagger_xeval: training curves of every tagger, and the 2x3 table (a tagger trained on HR only,
applied to HR / uniformly up-sampled LR / SR, next to a tagger trained on each input).
"""
import argparse

from . import tuning
from .config import apply_overrides, get_config, parse_set_args

ORDER = ["download", "prepare", "tune_vqvae", "train_vqvae", "tune_var", "train_var",
         "generate", "eval_sr", "tune_tagger", "train_taggers", "tagger_xeval", "summarize"]


def tagger_kinds(cfg, levels=None):
    kinds = ["hr"]
    for lvl in levels or cfg["data"]["levels"]:
        kinds += [f"lr:{lvl}", f"sr:{lvl}"]  # sr = model output as generated, no energy correction
    return kinds


def run_stage(cfg, stage, levels=None, kinds=None, seeds=None):
    levels = levels or list(cfg["data"]["levels"])
    if stage == "download":
        from .download import download
        return download(cfg)
    if stage == "prepare":
        from .data import prepare
        return prepare(cfg)
    if stage == "tune_vqvae":
        return tuning.tune_vqvae(cfg)
    if stage == "train_vqvae":
        from .train import train_vqvae
        return train_vqvae(tuning.with_tuned(cfg, "vqvae", "hr"))
    if stage == "tune_var":
        return tuning.tune_var(cfg)
    if stage == "train_var":
        from .train import train_var
        c = tuning.with_tuned(cfg, "var", "all")
        return {lvl: train_var(c, lvl) for lvl in levels}
    if stage == "generate":
        from .train import generate_sr
        c = tuning.with_tuned(cfg, "var", "all")
        for lvl in levels:
            generate_sr(c, lvl)
        return None
    if stage == "eval_sr":
        from .evaluate import evaluate_level
        c = apply_overrides(cfg, {"tagger.arch": "cnn"})  # C2ST uses the tagger CNN and its tuned settings
        c = tuning.with_tuned(c, "tagger", "hr")
        return {lvl: evaluate_level(c, lvl) for lvl in levels}
    if stage == "tune_tagger":
        return tuning.tune_tagger(cfg)
    if stage == "train_taggers":
        from .train import tagger_done, train_tagger
        c = tuning.with_tuned(cfg, "tagger", "hr")
        results = {}
        for kind in kinds or tagger_kinds(cfg, levels):
            for seed in seeds or cfg["tagger"]["seeds"]:
                if not c["tagger"].get("retrain", False) and tagger_done(c, kind, seed):
                    print(f"=== tagger on {kind}, seed {seed}: already trained - skipped (tagger.retrain=true to redo)")
                    continue
                print(f"\n=== tagger on {kind}, seed {seed} ===")
                results[(kind, seed)] = train_tagger(c, kind, seed)
        return results
    if stage == "tagger_xeval":
        from .evaluate import plot_tagger_curves, tagger_cross_eval
        c = tuning.with_tuned(cfg, "tagger", "hr")
        out = {}
        for lvl in levels:
            plot_tagger_curves(c, lvl, seeds)
            out[lvl] = tagger_cross_eval(c, lvl, seeds)
        return out
    if stage == "summarize":
        from .evaluate import summarize
        return summarize(cfg)
    raise ValueError(f"unknown stage {stage}; choose from {ORDER}")


def _session_info(stages, levels, kinds, seeds, overrides, smoke):
    """What run_info.json records for a batch job (the notebook records the same)."""
    import os
    import platform
    import subprocess

    def git(*args):
        try:
            return subprocess.run(["git", *args], capture_output=True, text=True, check=True,
                                  cwd=os.path.dirname(os.path.abspath(__file__))).stdout.strip()
        except Exception:
            return None

    import torch
    return {"launcher": "slurm" if os.environ.get("SLURM_JOB_ID") else "cli",
            "job_id": os.environ.get("SLURM_JOB_ID"), "array_task": os.environ.get("SLURM_ARRAY_TASK_ID"),
            "node": platform.node(), "branch": git("rev-parse", "--abbrev-ref", "HEAD"),
            "commit": git("rev-parse", "HEAD"),
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
            "stages": stages, "levels": levels, "kinds": kinds, "seeds": seeds, "smoke": smoke,
            "overrides": overrides}


def run(dataset, stages, overrides=None, smoke=False, levels=None, kinds=None, seeds=None, run_info=False):
    cfg = get_config(dataset, overrides, smoke=smoke)
    stages = ORDER if stages == ["all"] else stages
    if run_info:
        from . import io_utils
        io_utils.save_run_info(cfg, **_session_info(stages, levels, kinds, seeds, overrides, smoke))
    out = {}
    for s in stages:
        print(f"\n######## {dataset}: {s} ########")
        out[s] = run_stage(cfg, s, levels, kinds, seeds)
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", required=True, choices=["qg", "calo"])
    p.add_argument("--stages", default="all", help=f"comma list from {ORDER} or 'all'")
    p.add_argument("--levels", default=None, help="comma list of LR levels (default: all)")
    p.add_argument("--kinds", default=None, help="comma list of tagger inputs, e.g. hr,var:pool4x4")
    p.add_argument("--seeds", default=None, help="comma list of tagger seeds")
    p.add_argument("--set", nargs="*", default=[], help="config overrides key=value (JSON values)")
    p.add_argument("--smoke", action="store_true", help="tiny settings for a quick end-to-end check")
    p.add_argument("--run-info", action="store_true",
                   help="record this session (job id, node, commit, GPU, config) in the experiment's run_info.json")
    a = p.parse_args()
    split = lambda s: s.split(",") if s else None  # noqa: E731
    run(a.dataset, split(a.stages), parse_set_args(a.set), a.smoke, split(a.levels), split(a.kinds),
        [int(x) for x in split(a.seeds)] if a.seeds else None, a.run_info)


if __name__ == "__main__":
    main()
