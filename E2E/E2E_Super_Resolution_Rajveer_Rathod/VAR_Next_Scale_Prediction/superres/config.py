"""Default configuration for both datasets.

Everything is a plain nested dict so it serialises to JSON and can be overridden
with dotted keys, e.g. {"vqvae.lr": 1e-3} (used by Optuna and the CLI --set flag).
"""
import copy
import json

CERNBOX_TOKEN = "EYgmOkI9BjwxNqy"
ZENODO_RECORD = "6366271"

# CaloChallenge Dataset 2 radial bin edges in mm (binning_dataset_2.xml).
CALO_R_EDGES = [0.0, 4.65, 9.3, 13.95, 18.6, 27.9, 37.2, 46.5, 55.8, 65.1]

BASE = {
    # Results go to {drive_root}/{dataset}/{run_date}_{version}_{run_tag}/ and every file name starts
    # with {dataset}_{version}_{run_date}_{run_tag}, so no experiment overwrites or mixes with another.
    # v1: single-head decoder, 16x16 latent, total-energy loss.
    # v2: hit/energy decoder heads, LR image into the decoder, 32x32 latent, per-channel energy loss.
    "version": "v2",
    # One experiment = one training campaign: its start date (YYYY-MM-DD) and a descriptive tag,
    # e.g. run_date "2026-10-06", run_tag "multiseed" -> experiment 2026-10-06_v2_multiseed.
    # Keep both to resume or re-evaluate an experiment; a new date or tag starts from scratch.
    "run_date": None,
    "run_tag": None,
    "seed": 42,
    "num_workers": 2,
    "amp": True,
    "paths": {
        # Results (checkpoints, metrics, figures, Optuna DBs) -> Google Drive.
        "drive_root": "/content/drive/MyDrive/GSoC_SuperRes/experiments",
        # Raw downloads and preprocessed memmaps -> fast local Colab disk.
        "raw_root": "/content/data/raw",
        "cache_root": "/content/data/cache",
    },
    "optuna": {
        "vqvae_trials": 12,
        "var_trials": 8,
        "tagger_trials": 12,
        "trial_epochs": {"vqvae": 4, "var": 4, "tagger": 4},
        "trial_max_train": 10000,
        # VAR hyper-parameters are tuned on the hardest level and reused.
        "var_tune_level": None,
        "use_tuned": True,
    },
}

DATASETS = {
    # ------------------------------------------------------------------ CMS jets
    "qg": {
        "data": {
            "cernbox_token": CERNBOX_TOKEN,
            "extra_urls": [],            # optional direct links, used if CERNBox listing fails
            "native_shape": [125, 125],
            "pad": [[1, 2], [1, 2]],     # 125 -> 128 so 2/4/8 pooling divides evenly
            "hr_shape": [128, 128],
            "channels": ["tracks", "ecal", "hcal"],
            "circular_dims": [],
            "levels": {"pool2x2": [2, 2], "pool4x4": [4, 4], "pool8x8": [8, 8]},
            "max_samples": 40000,
            "split": [0.8, 0.1, 0.1],
            "pixel_size": 0.0174,        # approx. ECAL crystal size in (eta, phi)
            "center_index": 63,          # jet axis pixel in padded 128 grid
            "occupancy_threshold": 1e-3, # pixel counted as "hit" above this (data units)
            "signal_label": 1,
        },
        "vqvae": {
            "widths": [32, 64, 128],
            "strides": [[2, 2], [2, 2]],           # 128 -> 32x32 latent: one token per 4x4 pixels
            "n_res": 1,
            "z_ch": 32,
            "codebook_size": 1024,
            "beta": 0.25,
            "scales": [[1, 1], [2, 2], [3, 3], [4, 4], [6, 6], [8, 8], [12, 12], [16, 16], [20, 20],
                       [24, 24], [32, 32]],        # 2530 tokens per image
            "lr": 3e-4,
            "weight_decay": 0.0,
            "batch_size": 64,
            "epochs": 30,
            "energy_weight": 0.3,                  # per-channel |log E_pred - log E_true|
            "hit_weight": 1.0,                     # hit / no-hit cross-entropy
            "hit_pos_weight": 1.0,                 # >1 favours predicting hits
            "hit_threshold": 0.5,                  # a cell is "hit" when p(hit) exceeds this
            "energy_mask": "soft",                 # energy loss counts p(hit) (soft) or the 0/1 output cut
                                                   # with straight-through gradients (hard); see vqvae_losses
            "lr_decoder": True,                    # decoder also sees the LR image (cycled over levels)
            "dead_thresh": 0.01,
            "patience": 6,
        },
        "var": {
            "depth": 6,
            "width": 384,                # paper rule w = 64 * depth
            "heads": 6,
            "mlp_ratio": 4.0,
            "drop": 0.05,
            "lr_enc_widths": [32, 64, 128],    # one more entry than vqvae.strides
            "lr": 3e-4,
            "weight_decay": 0.05,
            "label_smoothing": 0.0,
            "batch_size": 32,            # 2530 tokens per image
            "epochs": 40,
            "patience": 8,
            "temperature": 0.8,          # only used by decode "sample"
            "top_k": 50,
            "decode": "greedy",          # which output is "the" SR result: greedy | sample
            # outputs the generate stage writes; any decode other than `decode` is a diagnostic
            # and is written for the test split only
            "decodes": ["greedy", "sample"],
            "gen_splits": ["train", "val", "test"],
            "gen_overwrite": False,      # set true after retraining, to replace cached outputs
            "gen_batch_size": 128,
            "gen_max_per_split": None,
            "sr_dtype": "float16",       # halves disk use; pixel values are far below fp16 max
        },
        "tagger": {
            "task": "classification",
            "arch": "resnet18",          # resnet18 | cnn; named in run folders, summary files and titles
            "widths": [32, 64, 128, 128],    # cnn only (also the C2ST classifier)
            # put on top of the settings here when arch = resnet18: 11 M parameters on 32k jets overfit
            # after 2-3 epochs with the CNN's settings, so stronger weight decay, more dropout, a lower lr
            # and random eta / phi reflections of the training images
            "resnet18": {"lr": 5e-4, "weight_decay": 0.05, "dropout": 0.3, "augment": True},
            "dropout": 0.1,
            "lr": 1e-3,
            "weight_decay": 1e-4,
            "batch_size": 128,
            "epochs": 20,
            "patience": 5,
            "max_train": None,
            "seeds": [42],
            "retrain": False,            # True: retrain taggers that already finished
        },
        "eval": {"max_events": 4000, "c2st_max": 4000, "c2st_epochs": 5, "nsub_max": 2000,
                 "hist_bins": 40, "n_examples": 4,
                 "channel_scale": "linear",   # per-channel event figures: linear | log
                 # main figures and tables: truth vs coarse vs model output as generated
                 "methods": ["lr", "sr"],
                 # extra rows / figures kept apart under figures/diagnostics
                 "diagnostics": ["vqrec", "sr-greedy", "sr-sample", "uniform", "srproj"],
                 "c2st_methods": ["sr"]},
    },
    # ------------------------------------------------------- CaloChallenge DS2
    "calo": {
        "data": {
            "zenodo_record": ZENODO_RECORD,
            "files": ["dataset_2_1.hdf5", "dataset_2_2.hdf5"],
            "native_shape": [45, 16, 9],   # (layer, angular, radial)
            "pad": [[0, 3], [0, 0], [0, 0]],
            "hr_shape": [48, 16, 9],
            "channels": ["energy"],
            "circular_dims": [1],          # the angular axis is periodic
            "levels": {"pool1x2x1": [1, 2, 1], "pool1x4x3": [1, 4, 3], "pool3x8x3": [3, 8, 3]},
            "max_samples": None,           # from dataset_2_1 (train + val)
            "val_frac": 0.1,
            "test_max": None,              # from dataset_2_2
            "r_edges": CALO_R_EDGES,
            "voxel_threshold_mev": 0.0151, # 15.15 keV readout threshold used in CaloChallenge
        },
        "vqvae": {
            "widths": [32, 64, 128],
            "strides": [[2, 2, 1], [2, 2, 1]],     # (48,16,9) -> (12,4,9) latent
            "n_res": 1,
            "z_ch": 32,
            "codebook_size": 1024,
            "beta": 0.25,
            "scales": [[1, 1, 1], [2, 1, 2], [3, 2, 3], [6, 2, 5], [9, 3, 7], [12, 4, 9]],
            "lr": 3e-4,
            "weight_decay": 0.0,
            "batch_size": 128,
            "epochs": 30,
            "energy_weight": 0.3,
            "hit_weight": 1.0,
            "hit_pos_weight": 1.0,
            "hit_threshold": 0.5,
            "lr_decoder": True,
            "dead_thresh": 0.01,
            "patience": 6,
        },
        "var": {
            "depth": 6,
            "width": 384,
            "heads": 6,
            "mlp_ratio": 4.0,
            "drop": 0.05,
            "lr_enc_widths": [32, 64, 128],
            "lr": 3e-4,
            "weight_decay": 0.05,
            "label_smoothing": 0.0,
            "batch_size": 128,
            "epochs": 40,
            "patience": 8,
            "temperature": 0.8,          # only used by decode "sample"
            "top_k": 50,
            "decode": "greedy",          # which output is "the" SR result: greedy | sample
            # outputs the generate stage writes; any decode other than `decode` is a diagnostic
            # and is written for the test split only
            "decodes": ["greedy", "sample"],
            "gen_splits": ["train", "val", "test"],
            "gen_overwrite": False,      # set true after retraining, to replace cached outputs
            "gen_batch_size": 256,
            "gen_max_per_split": None,
            "sr_dtype": "float32",       # core voxels of TeV showers can exceed fp16 range
        },
        "tagger": {
            "task": "regression",          # regress incident energy (energy calibration)
            "arch": "cnn",                 # cnn only: resnet18 is 2-D
            "widths": [32, 64, 128],
            "dropout": 0.1,
            "lr": 1e-3,
            "weight_decay": 1e-4,
            "batch_size": 256,
            "epochs": 20,
            "patience": 5,
            "max_train": None,
            "seeds": [42],
            "retrain": False,            # True: retrain taggers that already finished
        },
        "eval": {"max_events": 10000, "c2st_max": 10000, "c2st_epochs": 5, "hist_bins": 40,
                 "n_examples": 3, "methods": ["lr", "sr"],
                 "diagnostics": ["vqrec", "sr-greedy", "sr-sample", "uniform", "srproj"],
                 "c2st_methods": ["sr"]},
    },
}

# Tiny settings to check the whole pipeline end-to-end in a few minutes.
SMOKE = {
    "qg": {"data.max_samples": 600, "vqvae.epochs": 2, "var.epochs": 2, "tagger.epochs": 2,
           "var.depth": 2, "var.width": 128, "var.heads": 2, "eval.max_events": 200,
           "eval.c2st_max": 200, "eval.c2st_epochs": 1, "eval.nsub_max": 100,
           "optuna.vqvae_trials": 2, "optuna.var_trials": 2, "optuna.tagger_trials": 2,
           "optuna.trial_epochs": {"vqvae": 1, "var": 1, "tagger": 1},
           "optuna.trial_max_train": 200},
    "calo": {"data.max_samples": 2000, "data.test_max": 500, "vqvae.epochs": 2, "var.epochs": 2,
             "tagger.epochs": 2, "var.depth": 2, "var.width": 128, "var.heads": 2,
             "eval.max_events": 300, "eval.c2st_max": 300, "eval.c2st_epochs": 1,
             "optuna.vqvae_trials": 2, "optuna.var_trials": 2, "optuna.tagger_trials": 2,
             "optuna.trial_epochs": {"vqvae": 1, "var": 1, "tagger": 1},
             "optuna.trial_max_train": 500},
}


def _deep_merge(a, b):
    out = copy.deepcopy(a)
    for k, v in b.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def apply_overrides(cfg, overrides):
    """Set dotted keys, e.g. {"vqvae.lr": 1e-3, "paths.drive_root": "/x"}."""
    cfg = copy.deepcopy(cfg)
    for key, value in (overrides or {}).items():
        node = cfg
        parts = key.split(".")
        for p in parts[:-1]:
            node = node.setdefault(p, {})
        node[parts[-1]] = value
    return cfg


def get_config(dataset, overrides=None, smoke=False):
    if dataset not in DATASETS:
        raise ValueError(f"dataset must be one of {list(DATASETS)}")
    cfg = _deep_merge(BASE, DATASETS[dataset])
    cfg["dataset"] = dataset
    if cfg["optuna"]["var_tune_level"] is None:
        cfg["optuna"]["var_tune_level"] = list(cfg["data"]["levels"])[-1]
    if smoke:
        cfg = apply_overrides(cfg, SMOKE[dataset])
        cfg["smoke"] = True
    return apply_overrides(cfg, overrides)


def parse_set_args(items):
    """['vqvae.lr=1e-3', 'data.max_samples=null'] -> dict (values parsed as JSON when possible)."""
    out = {}
    for item in items or []:
        key, _, raw = item.partition("=")
        try:
            out[key] = json.loads(raw)
        except json.JSONDecodeError:
            out[key] = raw
    return out
