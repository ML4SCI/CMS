"""Optuna studies (TPE sampler + median pruner), resumable across Colab sessions.

study, in {experiment}/tuning/ (name = {dataset}_{version}_{date}_{tag}, so each experiment has its own)
                                  objective (validation set)                                 direction
{name}_vqvae_tokenizer            MSE + median per-channel |dE/E| + |1 - predicted/true hits|  minimise
{name}_var_transformer_all        next-scale token cross-entropy                             minimise
{name}_{tagger}_hr                qg: ROC AUC / calo: mean binned resolution                 max / min

Trials use short schedules (optuna.trial_epochs) on a subset of the training
data (optuna.trial_max_train); the best parameters are stored as dotted config
overrides in *_best.json and picked up automatically by the full runs.
"""
import os

from . import io_utils
from .config import apply_overrides


def _sync(paths, study):
    if os.path.exists(paths["db_local"]):
        io_utils.copy_file(paths["db_local"], paths["db_drive"])
    try:
        study.trials_dataframe().to_csv(paths["trials_csv"], index=False)
    except Exception:
        pass
    try:
        best = study.best_trial
    except ValueError:
        return
    io_utils.save_json(paths["best_json"], {
        "study": paths["name"], "updated": io_utils.now(), "best_value": best.value,
        "best_trial": best.number, "params": best.params, "overrides": best.user_attrs.get("overrides", {}),
        "n_complete": sum(t.state.name == "COMPLETE" for t in study.trials),
        "n_pruned": sum(t.state.name == "PRUNED" for t in study.trials)})


def run_study(cfg, stage, variant, suggest, train_fn, n_trials, direction):
    import optuna
    paths = io_utils.study_paths(cfg, stage, variant)
    if not os.path.exists(paths["db_local"]) and os.path.exists(paths["db_drive"]):
        io_utils.copy_file(paths["db_drive"], paths["db_local"])  # resume from Drive
    study = optuna.create_study(
        study_name=paths["name"], storage=f"sqlite:///{paths['db_local']}", load_if_exists=True,
        direction=direction, sampler=optuna.samplers.TPESampler(seed=cfg["seed"]),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=3, n_warmup_steps=1))
    done = sum(t.state.name in ("COMPLETE", "PRUNED") for t in study.trials)
    remaining = max(0, n_trials - done)
    print(f"study {paths['name']}: {done} trials done, running {remaining} more")

    def objective(trial):
        overrides = suggest(trial)
        trial.set_user_attr("overrides", overrides)
        return train_fn(apply_overrides(cfg, overrides), trial)

    if remaining:
        study.optimize(objective, n_trials=remaining, gc_after_trial=True,
                       callbacks=[lambda s, t: _sync(paths, s)], catch=(RuntimeError,))
    _sync(paths, study)
    try:
        print(f"best {paths['name']}: {study.best_value:.5f} with {study.best_params}")
    except ValueError:
        print("no completed trials")
    return study


def load_best_overrides(cfg, stage, variant):
    path = io_utils.study_paths(cfg, stage, variant)["best_json"]
    if not os.path.exists(path):
        return {}
    return io_utils.load_json(path).get("overrides", {})


def with_tuned(cfg, stage, variant):
    if not cfg["optuna"].get("use_tuned", True):
        return cfg
    over = load_best_overrides(cfg, stage, variant)
    if over:
        print(f"using tuned {stage} params: {over}")
    return apply_overrides(cfg, over)


def _scaled(widths, factor):
    return [max(8, int(round(w * factor / 8)) * 8) for w in widths]


# ------------------------------------------------------------------ studies
def tune_vqvae(cfg):
    from .train import train_vqvae
    o = cfg["optuna"]
    base_w = cfg["vqvae"]["widths"]

    def suggest(t):
        return {"vqvae.lr": t.suggest_float("lr", 1e-4, 2e-3, log=True),
                "vqvae.codebook_size": t.suggest_categorical("codebook_size", [512, 1024, 2048, 4096]),
                "vqvae.z_ch": t.suggest_categorical("z_ch", [16, 32, 64]),
                "vqvae.beta": t.suggest_float("beta", 0.1, 0.5),
                "vqvae.energy_weight": t.suggest_float("energy_weight", 0.01, 1.0, log=True),
                "vqvae.widths": _scaled(base_w, t.suggest_categorical("width_factor", [0.5, 1.0, 2.0]))}

    def fn(c, trial):
        return train_vqvae(c, trial=trial, epochs=o["trial_epochs"]["vqvae"], max_train=o["trial_max_train"],
                           save=False, resume=False)

    return run_study(cfg, "vqvae", "hr", suggest, fn, o["vqvae_trials"], "minimize")


def tune_var(cfg, level=None):
    from .train import train_var
    o = cfg["optuna"]
    level = level or o["var_tune_level"]

    def suggest(t):
        depth = t.suggest_categorical("depth", [4, 6, 8])
        return {"var.depth": depth, "var.width": 64 * depth, "var.heads": depth,  # paper eq. (7)
                "var.lr": t.suggest_float("lr", 1e-4, 1e-3, log=True),
                "var.drop": t.suggest_float("drop", 0.0, 0.2),
                "var.weight_decay": t.suggest_float("weight_decay", 1e-3, 0.1, log=True),
                "var.label_smoothing": t.suggest_float("label_smoothing", 0.0, 0.1)}

    def fn(c, trial):
        return train_var(c, level, trial=trial, epochs=o["trial_epochs"]["var"], max_train=o["trial_max_train"],
                         save=False, resume=False)

    # one study, reused for every level (see with_tuned(cfg, "var", "all"))
    return run_study(cfg, "var", "all", suggest, fn, o["var_trials"], "minimize")


def tune_tagger(cfg):
    from .train import train_tagger
    o = cfg["optuna"]
    base_w = cfg["tagger"]["widths"]
    classify = cfg["tagger"]["task"] == "classification"

    arch = cfg["tagger"].get("arch", "cnn")
    key = "tagger.resnet18." if arch == "resnet18" else "tagger."  # where train_tagger reads them

    def suggest(t):
        out = {key + "lr": t.suggest_float("lr", 1e-4, 3e-3, log=True),
               key + "dropout": t.suggest_float("dropout", 0.0, 0.4),
               key + "weight_decay": t.suggest_float("weight_decay", 1e-5, 1e-1, log=True)}
        if arch == "cnn":
            out["tagger.widths"] = _scaled(base_w, t.suggest_categorical("width_factor", [0.5, 1.0, 2.0]))
        return out

    def fn(c, trial):
        return train_tagger(c, "hr", c["seed"], trial=trial, epochs=o["trial_epochs"]["tagger"],
                            max_train=o["trial_max_train"], save=False)

    # tuned on HR, then the SAME architecture is used for every input so that
    # differences between inputs come from the data, not from tuning effort
    return run_study(cfg, "tagger", "hr", suggest, fn, o["tagger_trials"], "maximize" if classify else "minimize")
