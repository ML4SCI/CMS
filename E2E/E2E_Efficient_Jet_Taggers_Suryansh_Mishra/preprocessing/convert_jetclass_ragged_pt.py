"""
Sharded ragged JetClass ROOT → CSR .pt converter.

One ``.pt`` shard per ROOT file (~100k jets, CSR-packed, ~310 MB), plus a
PyTorch ``RaggedShardDataset`` with dynamic batch-padding collate and fail-closed
CSR verification gates (structural invariants + ROOT round-trip spot checks).

Designed to run on Perlmutter against CFS/PSCRATCH paths.  CPU/I/O bound;
parallel conversion via ``ProcessPoolExecutor``, resume-safe via skip-existing,
atomic writes via ``.pt.tmp`` → rename.

On-disk shard schema
--------------------
::

    {
      "x":            Tensor[float32]  (N_particles_total, 16)  ALL_PARTICLE_FEATURES
      "v":            Tensor[float32]  (N_particles_total,  4)  raw [px, py, pz, E]
      "offsets":      Tensor[int64]    (N_jets + 1,)            CSR row pointers
      "jet":          Tensor[float32]  (N_jets, 10)             jet_* scalar branches
      "y":            Tensor[float32]  (N_jets, 10)             one-hot labels
      "n_particles":  Tensor[int32]    (N_jets,)                per-jet particle count
    }

Usage
-----
::

    # Smoke test (local or NERSC)
    python convert_jetclass_ragged_pt.py \\
      --root-dir /pscratch/sd/o/omasho/jetclass/Pythia \\
      --out-dir  /pscratch/sd/o/omasho/jetclass/pt_ragged_smoke \\
      --splits train_100M --max-files 2 --num-workers 2

    python convert_jetclass_ragged_pt.py --verify-only \\
      --root-dir /pscratch/sd/o/omasho/jetclass/Pythia \\
      --out-dir  /pscratch/sd/o/omasho/jetclass/pt_ragged_smoke \\
      --splits train_100M --verify-all

    # Full convert + post-verify sample
    python convert_jetclass_ragged_pt.py \\
      --root-dir /pscratch/sd/o/omasho/jetclass/Pythia \\
      --out-dir  /pscratch/sd/o/omasho/jetclass/pt_ragged \\
      --splits train_100M val_5M test_20M --num-workers 8

    python convert_jetclass_ragged_pt.py --verify-only \\
      --root-dir /pscratch/sd/o/omasho/jetclass/Pythia \\
      --out-dir  /pscratch/sd/o/omasho/jetclass/pt_ragged \\
      --splits train_100M val_5M test_20M --verify-shards 20 --verify-jets 32
"""

from __future__ import annotations

import argparse
import os
import sys
from collections import OrderedDict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import awkward as ak
import numpy as np
import uproot
import vector
from tqdm import tqdm

import torch
from torch import Tensor

# --- Canonical loader module; thin re-exports for backward compat ----------
import sys as _sys
_PARENT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PARENT_DIR not in _sys.path:
    _sys.path.insert(0, _PARENT_DIR)

from dataloader.ragged_loader import (  # noqa: E402
    EVENTS_PER_SHARD as _EVENTS_PER_SHARD_CANONICAL,
    RaggedShardDataset,
    ragged_collate,
    create_ragged_dataloader,
)

vector.register_awkward()

# ---------------------------------------------------------------------------
# Constants — canonical definition of the 16-feature particle ordering.
# This is the authoritative source: ``x`` columns in every ``.pt`` shard follow
# it, so ``ablation.data.NormStats`` channel indices are defined against it.
# (It was previously mirrored from a padding loader that has since been removed;
# the ragged CSR path is now the only pipeline.)
# ---------------------------------------------------------------------------

ALL_PARTICLE_FEATURES: Tuple[str, ...] = (
    # Kinematics (indices 0-3) — pt/eta/phi derived, energy from ROOT
    "part_pt",
    "part_eta",
    "part_phi",
    "part_energy",
    # Relative kinematics (indices 4-5) — pre-computed in ROOT
    "part_deta",
    "part_dphi",
    # Displacement / impact parameters (indices 6-9)
    "part_d0val",
    "part_d0err",
    "part_dzval",
    "part_dzerr",
    # Identification (indices 10-15)
    "part_charge",
    "part_isChargedHadron",
    "part_isNeutralHadron",
    "part_isPhoton",
    "part_isElectron",
    "part_isMuon",
)

#: Features already present in the ROOT file (no derivation needed).
#  pt, eta, phi are derived via vector; everything else is read verbatim.
_PARTICLE_ROT_FEATURES: Tuple[str, ...] = (
    "part_energy",
    "part_deta",
    "part_dphi",
    "part_d0val",
    "part_d0err",
    "part_dzval",
    "part_dzerr",
    "part_charge",
    "part_isChargedHadron",
    "part_isNeutralHadron",
    "part_isPhoton",
    "part_isElectron",
    "part_isMuon",
)

#: Raw Lorentz four-vector source branches, weaver order [px, py, pz, E].
RAW_VECTOR_BRANCHES: Tuple[str, ...] = (
    "part_px",
    "part_py",
    "part_pz",
    "part_energy",
)

#: All ROOT particle branches read from the tree.
_ROT_PARTICLE_BRANCHES: Tuple[str, ...] = (
    "part_px",
    "part_py",
    "part_pz",
    "part_energy",
    "part_deta",
    "part_dphi",
    "part_d0val",
    "part_d0err",
    "part_dzval",
    "part_dzerr",
    "part_charge",
    "part_isChargedHadron",
    "part_isNeutralHadron",
    "part_isPhoton",
    "part_isElectron",
    "part_isMuon",
)

#: Jet scalar branches (fixed order).
JET_BRANCHES: Tuple[str, ...] = (
    "jet_pt",
    "jet_eta",
    "jet_phi",
    "jet_energy",
    "jet_sdmass",
    "jet_tau1",
    "jet_tau2",
    "jet_tau3",
    "jet_tau4",
    "jet_nparticles",
)

#: Index of ``jet_nparticles`` within JET_BRANCHES (0-based).
_JET_NPARTICLES_IDX: int = 9

#: One-hot label branches (10-class JetClass).
LABEL_NAMES: Tuple[str, ...] = (
    "label_QCD",
    "label_Hbb",
    "label_Hcc",
    "label_Hgg",
    "label_H4q",
    "label_Hqql",
    "label_Zqq",
    "label_Wqq",
    "label_Tbqq",
    "label_Tbl",
)

#: Non-padded global maximum across the full 125M-jet dataset.
_GLOBAL_MAX_PARTICLES: int = 183

#: Jets per ROOT file / .pt shard (from ragged_loader).
_EVENTS_PER_SHARD: int = _EVENTS_PER_SHARD_CANONICAL

# ---------------------------------------------------------------------------
# ROOT → CSR .pt  (no padding, no clipping)
# ---------------------------------------------------------------------------


def _read_root_file_csr(filepath: str) -> Dict[str, np.ndarray]:
    """Read a single JetClass ``.root`` file and return CSR-packed numpy arrays.

    Nothing is padded or clipped — every particle in every jet is kept, which
    is the point of the CSR layout. Padding happens later and dynamically, at
    collate time, to the max multiplicity in each batch.

    Parameters
    ----------
    filepath : str
        Path to a single ``.root`` file.

    Returns
    -------
    dict
        ``{"x", "v", "offsets", "jet", "y", "n_particles"}`` — all values are
        numpy arrays matching the on-disk shard schema.
    """
    tree = uproot.open(filepath)["tree"]

    # Read all needed branches in one call.  uproot ≥5 accepts a list of
    # branch names for *filter_name*; the fallback reads everything and
    # selects afterwards.
    all_branches = list(_ROT_PARTICLE_BRANCHES) + list(JET_BRANCHES) + list(LABEL_NAMES)
    try:
        table = tree.arrays(filter_name=all_branches)
    except Exception:
        # Fallback: read all branches, then select (older uproot / weird files)
        table = tree.arrays()
        keep = set(all_branches)
        for key in list(table.keys()):
            if key not in keep:
                del table[key]

    # --- Derive pt, eta, phi from the Lorentz 4-vector -----------------------
    p4 = vector.zip(
        {
            "px": table["part_px"],
            "py": table["part_py"],
            "pz": table["part_pz"],
            "energy": table["part_energy"],
        }
    )
    _derived = {
        "part_pt": p4.pt,
        "part_eta": p4.eta,
        "part_phi": p4.phi,
    }

    # --- Build CSR offsets from the jagged counts ----------------------------
    counts = ak.num(table["part_px"])  # int64 per-jet particle count
    offsets = np.empty(len(counts) + 1, dtype=np.int64)
    offsets[0] = 0
    np.cumsum(counts, out=offsets[1:])  # in-place into pre-allocated int64

    # --- Flatten & stack particle features -----------------------------------
    x_cols = []
    for name in ALL_PARTICLE_FEATURES:
        if name in _derived:
            col = ak.flatten(_derived[name])
        else:
            col = ak.flatten(table[name])
        x_cols.append(ak.to_numpy(col).astype(np.float32))
    x = np.column_stack(x_cols)  # (N_particles_total, 16)

    # --- Raw four-vectors ----------------------------------------------------
    v_cols = [
        ak.to_numpy(ak.flatten(table[n])).astype(np.float32)
        for n in RAW_VECTOR_BRANCHES
    ]
    v = np.column_stack(v_cols)  # (N_particles_total, 4)

    # --- Jet scalar features -------------------------------------------------
    jet_cols = [ak.to_numpy(table[n]).astype(np.float32) for n in JET_BRANCHES]
    jet = np.column_stack(jet_cols)  # (N_jets, 10)

    # --- One-hot labels ------------------------------------------------------
    y_cols = [ak.to_numpy(table[n]).astype(np.float32) for n in LABEL_NAMES]
    y = np.column_stack(y_cols)  # (N_jets, 10)

    n_particles = np.asarray(counts, dtype=np.int32)  # (N_jets,)

    return {
        "x": x,
        "v": v,
        "offsets": offsets,
        "jet": jet,
        "y": y,
        "n_particles": n_particles,
    }


def _convert_one_file(
    root_path: str,
    out_dir: str,
    force: bool = False,
    skip_verify: bool = False,  # noqa: FBT001
) -> Tuple[str, int, List[str]]:
    """Convert one ROOT file → CSR ``.pt`` shard.

    Returns ``(out_path, n_jets, errors)``.  Errors is a list of strings;
    empty means success.

    The structural verification (§A of the CSR gate) is run in-memory
    **before** the atomic save.  If it fails the ``.pt`` is never written.
    """
    basename = Path(root_path).stem  # e.g. "HToBB_000"
    out_path = os.path.join(out_dir, f"{basename}.pt")
    tmp_path = out_path + ".tmp"

    # --- Resume: skip existing unless forced --------------------------------
    if os.path.exists(out_path) and not force:
        return out_path, -1, []  # -1 means "skipped"

    # Clean stale tmp file from a previous crash
    if os.path.exists(tmp_path):
        os.unlink(tmp_path)

    # --- Read & pack --------------------------------------------------------
    try:
        arrays = _read_root_file_csr(root_path)
    except Exception as exc:
        return out_path, 0, [f"{root_path}: ROOT read failed: {exc}"]

    n_jets = len(arrays["y"])
    errors: List[str] = []

    # --- §A structural check (in-memory, before save) -----------------------
    if not skip_verify:
        errors = _verify_shard_arrays(arrays, source=basename)
    if errors:
        return out_path, n_jets, errors

    # --- Atomic write -------------------------------------------------------
    os.makedirs(out_dir, exist_ok=True)
    try:
        # Convert to torch tensors for torch.save
        tensors = {k: torch.from_numpy(v) for k, v in arrays.items()}
        torch.save(tensors, tmp_path, _use_new_zipfile_serialization=True)
        os.rename(tmp_path, out_path)
    except Exception as exc:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)
        return out_path, n_jets, [f"{basename}: save failed: {exc}"]

    return out_path, n_jets, []


# ---------------------------------------------------------------------------
# CSR verification gate (fail-closed)
# ---------------------------------------------------------------------------


def _verify_shard_arrays(
    data: Dict[str, Any],
    source: str = "<unknown>",
    allow_empty: bool = True,  # noqa: FBT001
) -> List[str]:
    """Structural CSR invariant checks (gate §A).

    Works with both numpy arrays (pre-save) and torch tensors (post-load).

    Returns a list of error strings; empty = pass.
    """
    errors: List[str] = []

    def _np(arr: Any) -> np.ndarray:
        """Normalize to numpy for consistent checking."""
        if isinstance(arr, Tensor):
            return arr.detach().cpu().numpy()
        return np.asarray(arr)

    x = _np(data["x"])
    v = _np(data["v"])
    offsets = _np(data["offsets"])
    jet = _np(data["jet"])
    y = _np(data["y"])
    n_p = _np(data["n_particles"])

    N_jets = len(y)
    N_part = x.shape[0]
    _tag = f"[{source}]"

    # -- shapes --
    if x.ndim != 2:
        errors.append(f"{_tag} x.ndim={x.ndim}, expected 2")
    elif x.shape[1] != 16:
        errors.append(f"{_tag} x.shape[1]={x.shape[1]}, expected 16 features")
    if v.ndim != 2:
        errors.append(f"{_tag} v.ndim={v.ndim}, expected 2")
    elif v.shape[1] != 4:
        errors.append(f"{_tag} v.shape[1]={v.shape[1]}, expected 4")
    if x.shape[0] != v.shape[0]:
        errors.append(
            f"{_tag} x rows ({x.shape[0]}) != v rows ({v.shape[0]})"
        )

    # -- offsets --
    if offsets.dtype != np.int64:
        errors.append(
            f"{_tag} offsets.dtype={offsets.dtype}, expected int64"
        )
    if len(offsets) != N_jets + 1:
        errors.append(
            f"{_tag} len(offsets)={len(offsets)}, expected N_jets+1={N_jets + 1}"
        )
    else:
        if offsets[0] != 0:
            errors.append(f"{_tag} offsets[0]={offsets[0]}, expected 0")
        if offsets[-1] != N_part:
            errors.append(
                f"{_tag} offsets[-1]={offsets[-1]}, expected N_part={N_part}"
            )
        # monotonic (non-decreasing; allows empty jets)
        if not np.all(offsets[1:] >= offsets[:-1]):
            n_bad = int((offsets[1:] < offsets[:-1]).sum())
            errors.append(f"{_tag} offsets not non-decreasing ({n_bad} violations)")

    # -- counts --
    if len(offsets) == N_jets + 1:
        expected_n = np.diff(offsets)
        if not np.array_equal(n_p, expected_n):
            bad = np.where(n_p != expected_n)[0]
            errors.append(
                f"{_tag} n_particles != diff(offsets) at {len(bad)} indices: "
                f"{list(bad[:5])}"
            )
    if n_p.dtype != np.int32:
        errors.append(
            f"{_tag} n_particles.dtype={n_p.dtype}, expected int32"
        )

    # -- jet / y shapes --
    if jet.shape != (N_jets, 10):
        errors.append(f"{_tag} jet.shape={jet.shape}, expected ({N_jets}, 10)")
    if y.shape != (N_jets, 10):
        errors.append(f"{_tag} y.shape={y.shape}, expected ({N_jets}, 10)")

    # -- one-hot --
    y_sums = y.sum(axis=1)
    if not np.allclose(y_sums, 1.0):
        bad = np.where(~np.isclose(y_sums, 1.0))[0]
        errors.append(
            f"{_tag} y not one-hot at {len(bad)} rows: {list(bad[:5])}"
        )

    # -- jet_nparticles matches n_particles --
    if jet.shape == (N_jets, 10):
        if not np.allclose(jet[:, _JET_NPARTICLES_IDX], n_p, atol=0):
            bad = np.where(
                ~np.isclose(jet[:, _JET_NPARTICLES_IDX], n_p, atol=0)
            )[0]
            errors.append(
                f"{_tag} jet_nparticles != n_particles at {len(bad)} indices"
            )

    # -- finite --
    for arr_name, arr in [("x", x), ("v", v), ("jet", jet)]:
        if not np.all(np.isfinite(arr)):
            errors.append(f"{_tag} {arr_name} has non-finite values")

    # -- mask guard: E > 0 for all recorded particles --
    #  (particles with E <= 0 would silently zero the collate mask,
    #   causing mask.sum != n_particles at train time)
    e_chan = v[:, 3]
    n_e_bad = int((e_chan <= 0).sum())
    if n_e_bad > 0:
        errors.append(
            f"{_tag} v[:,3] (energy) has {n_e_bad} non-positive values"
        )

    # -- no silent clip --
    max_n = int(n_p.max())
    if max_n > _GLOBAL_MAX_PARTICLES:
        errors.append(
            f"{_tag} n_particles.max()={max_n} > {_GLOBAL_MAX_PARTICLES} "
            f"(known global max)"
        )

    # -- allow_empty guard (informational) --
    if not allow_empty:
        n_empty = int((n_p == 0).sum())
        if n_empty > 0:
            errors.append(f"{_tag} {n_empty} empty jets found")

    return errors


def _verify_root_roundtrip(
    pt_path: str,
    root_path: str,
    jet_indices: "np.ndarray",
    rtol: float = 1e-5,
    atol: float = 1e-5,
) -> List[str]:
    """ROOT round-trip spot check on specific jet indices (gate §B).

    Loads the ``.pt`` shard, re-opens the matching ``.root``, and compares
    per-particle values and per-jet scalars for the sampled jets.
    """
    errors: List[str] = []
    basename = Path(pt_path).stem

    # Load shard
    try:
        data = torch.load(pt_path, map_location="cpu", weights_only=True)
    except Exception as exc:
        return [f"{basename}: torch.load failed: {exc}"]

    # Open ROOT
    try:
        tree = uproot.open(root_path)["tree"]
        table = tree.arrays(
            filter_name=list(_ROT_PARTICLE_BRANCHES)
            + list(JET_BRANCHES)
            + list(LABEL_NAMES)
        )
    except Exception as exc:
        return [f"{basename}: ROOT re-open failed: {exc}"]

    # Derived features from ROOT for comparison
    p4 = vector.zip(
        {
            "px": table["part_px"],
            "py": table["part_py"],
            "pz": table["part_pz"],
            "energy": table["part_energy"],
        }
    )
    _derived_root = {
        "part_pt": p4.pt,
        "part_eta": p4.eta,
        "part_phi": p4.phi,
    }

    x = data["x"]
    v = data["v"]
    offsets = data["offsets"]
    y = data["y"]
    jet = data["jet"]

    for i in jet_indices:
        i = int(i)
        tag = f"{basename} jet {i}"

        # --- particle count ---
        n_root = int(ak.num(table["part_px"])[i])
        n_csr = int(offsets[i + 1] - offsets[i])
        if n_root != n_csr:
            errors.append(
                f"{tag}: ROOT n_part={n_root}, CSR n_part={n_csr}"
            )
            continue

        if n_root == 0:
            continue  # nothing else to compare for empty jets

        # --- raw four-vectors ---
        v_csr = v[offsets[i] : offsets[i + 1]].numpy()  # (n, 4)
        v_root = np.column_stack(
            [
                ak.to_numpy(table["part_px"][i]),
                ak.to_numpy(table["part_py"][i]),
                ak.to_numpy(table["part_pz"][i]),
                ak.to_numpy(table["part_energy"][i]),
            ]
        ).astype(np.float32)
        if not np.allclose(v_csr, v_root, rtol=rtol, atol=atol):
            max_diff = np.abs(v_csr - v_root).max()
            errors.append(
                f"{tag}: v mismatch (max abs diff={max_diff:.2e})"
            )

        # --- derived kinematics (x indices 0–2: pt, eta, phi) ---
        for col_idx, feat_name in enumerate(
            ["part_pt", "part_eta", "part_phi"]
        ):
            csr_vals = x[offsets[i] : offsets[i + 1], col_idx].numpy()
            root_vals = ak.to_numpy(_derived_root[feat_name][i]).astype(
                np.float32
            )
            if not np.allclose(csr_vals, root_vals, rtol=rtol, atol=atol):
                max_diff = np.abs(csr_vals - root_vals).max()
                errors.append(
                    f"{tag}: {feat_name} mismatch (max abs diff={max_diff:.2e})"
                )

        # --- precomputed relative feature (part_deta, index 4 in x) ---
        deta_csr = x[offsets[i] : offsets[i + 1], 4].numpy()
        deta_root = ak.to_numpy(table["part_deta"][i]).astype(np.float32)
        if not np.allclose(deta_csr, deta_root, rtol=rtol, atol=atol):
            max_diff = np.abs(deta_csr - deta_root).max()
            errors.append(
                f"{tag}: part_deta mismatch (max abs diff={max_diff:.2e})"
            )

        # --- label ---
        y_csr_argmax = int(y[i].argmax())
        label_root = np.array(
            [float(table[n][i]) for n in LABEL_NAMES], dtype=np.float32
        )
        y_root_argmax = int(label_root.argmax())
        if y_csr_argmax != y_root_argmax:
            errors.append(
                f"{tag}: label mismatch CSR={y_csr_argmax} "
                f"ROOT={y_root_argmax}"
            )

        # --- jet_pt ---
        jet_pt_csr = float(jet[i, 0])
        jet_pt_root = float(table["jet_pt"][i])
        if not np.isclose(jet_pt_csr, jet_pt_root, rtol=rtol, atol=atol):
            errors.append(
                f"{tag}: jet_pt mismatch CSR={jet_pt_csr:.6f} "
                f"ROOT={jet_pt_root:.6f}"
            )

        # --- jet_nparticles ---
        jet_np_csr = int(jet[i, _JET_NPARTICLES_IDX])
        jet_np_root = int(table["jet_nparticles"][i])
        if jet_np_csr != jet_np_root:
            errors.append(
                f"{tag}: jet_nparticles mismatch CSR={jet_np_csr} "
                f"ROOT={jet_np_root}"
            )

    return errors


# ---------------------------------------------------------------------------
# PyTorch Dataset + Collate — imported from ragged_loader.py
# ---------------------------------------------------------------------------
# RaggedShardDataset, ragged_collate, and create_ragged_dataloader are
# imported at the top of this file from ml4sci_26/dataloader/ragged_loader.py.
# They are re-exported here so that existing callers and the smoke check
# continue to work without changes.


# ---------------------------------------------------------------------------
# Converter CLI
# ---------------------------------------------------------------------------


def _iter_root_files(
    root_dir: str, split: str, max_files: Optional[int] = None
) -> List[str]:
    """List sorted ``.root`` files in ``root_dir/split/``."""
    split_dir = os.path.join(root_dir, split)
    if not os.path.isdir(split_dir):
        return []
    files = sorted(
        os.path.join(split_dir, f)
        for f in os.listdir(split_dir)
        if f.endswith(".root")
    )
    if max_files is not None:
        files = files[:max_files]
    return files


def _convert_split(
    root_dir: str,
    out_dir: str,
    split: str,
    num_workers: int = 8,
    max_files: Optional[int] = None,
    force: bool = False,
) -> Tuple[int, int, int, int, List[str]]:
    """Convert all ROOT files in one split to CSR .pt shards.

    Returns ``(n_total, n_ok, n_skipped, n_failed, all_errors)``.
    """
    files = _iter_root_files(root_dir, split, max_files)
    if not files:
        print(f"  [{split}] No .root files found — skipping")
        return 0, 0, 0, 0, []

    split_out = os.path.join(out_dir, split)
    os.makedirs(split_out, exist_ok=True)

    n_ok, n_skipped, n_failed = 0, 0, 0
    all_errors: List[str] = []

    if num_workers <= 1:
        # Sequential
        for fp in tqdm(files, desc=f"  {split}", unit="file"):
            out_path, n_jets, errs = _convert_one_file(
                fp, split_out, force=force
            )
            if n_jets < 0:
                n_skipped += 1
            elif errs:
                n_failed += 1
                all_errors.extend(errs)
                tqdm.write(f"  FAIL {out_path}: {errs[0]}")
            else:
                n_ok += 1
        return len(files), n_ok, n_skipped, n_failed, all_errors

    # Parallel via ProcessPoolExecutor
    with ProcessPoolExecutor(max_workers=num_workers) as ex:
        futures = {
            ex.submit(_convert_one_file, fp, split_out, force): fp
            for fp in files
        }
        pbar = tqdm(
            total=len(files), desc=f"  {split}", unit="file"
        )
        for future in as_completed(futures):
            fp = futures[future]
            try:
                out_path, n_jets, errs = future.result()
            except Exception as exc:
                n_failed += 1
                all_errors.append(f"{fp}: worker exception: {exc}")
                tqdm.write(f"  EXCEPTION {fp}: {exc}")
            else:
                if n_jets < 0:
                    n_skipped += 1
                elif errs:
                    n_failed += 1
                    all_errors.extend(errs)
                    tqdm.write(f"  FAIL {out_path}: {errs[0]}")
                else:
                    n_ok += 1
            pbar.update(1)
        pbar.close()

    return len(files), n_ok, n_skipped, n_failed, all_errors


def _verify_pass(args: argparse.Namespace) -> int:
    """``--verify-only`` path: load existing ``.pt`` shards and run gates §A+§B."""
    root_dir = args.root_dir
    out_dir = args.out_dir
    splits = args.splits
    verify_jets = args.verify_jets
    verify_shards: Optional[int] = args.verify_shards
    verify_all: bool = args.verify_all
    seed: int = args.seed

    rng = np.random.default_rng(seed)
    total_errors = 0

    for split in splits:
        split_out = os.path.join(out_dir, split)
        if not os.path.isdir(split_out):
            print(f"  [{split}] No output directory — skipping")
            continue

        pt_files = sorted(
            os.path.join(split_out, f)
            for f in os.listdir(split_out)
            if f.endswith(".pt")
        )
        if not pt_files:
            print(f"  [{split}] No .pt files — skipping")
            continue

        # Determine how many shards to deep-check
        if verify_all or args.max_files is not None:
            n_deep = len(pt_files)
        elif verify_shards is not None:
            n_deep = min(verify_shards, len(pt_files))
        else:
            n_deep = min(20, len(pt_files))

        deep_indices = set(
            rng.choice(len(pt_files), size=n_deep, replace=False)
        )

        n_ok, n_fail = 0, 0
        for idx, pt_path in enumerate(
            tqdm(pt_files, desc=f"  verify {split}", unit="shard")
        ):
            basename = Path(pt_path).stem
            errors: List[str] = []

            # §A: structural invariants
            try:
                data = torch.load(
                    pt_path, map_location="cpu", weights_only=True
                )
            except Exception as exc:
                errors.append(f"{basename}: torch.load failed: {exc}")
                n_fail += 1
                total_errors += len(errors)
                for e in errors:
                    tqdm.write(f"  FAIL {e}")
                continue

            errs_a = _verify_shard_arrays(data, source=basename)
            errors.extend(errs_a)

            # §B: ROOT round-trip on sampled shards
            if idx in deep_indices and not errs_a:
                root_path = os.path.join(
                    root_dir, split, f"{basename}.root"
                )
                if os.path.exists(root_path):
                    n_jets = len(data["y"])
                    n_sample = min(verify_jets, n_jets)
                    jet_indices = rng.choice(
                        n_jets, size=n_sample, replace=False
                    )
                    errs_b = _verify_root_roundtrip(
                        pt_path, root_path, jet_indices
                    )
                    errors.extend(errs_b)
                else:
                    errors.append(
                        f"{basename}: matching .root not found at {root_path}"
                    )

            if errors:
                n_fail += 1
                total_errors += len(errors)
                for e in errors:
                    tqdm.write(f"  FAIL {e}")
            else:
                n_ok += 1

        print(
            f"  [{split}] verify: {n_ok} ok, {n_fail} failed "
            f"(deep-checked {n_deep} shards)"
        )

    if total_errors > 0:
        print(f"\nVERIFY FAILED: {total_errors} error(s) across all splits")
        return 1
    print("\nVERIFY PASSED: all shards ok")
    return 0


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Sharded ragged JetClass ROOT → CSR .pt converter",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--root-dir",
        required=True,
        help="Directory containing split subdirectories (e.g. …/Pythia)",
    )
    parser.add_argument(
        "--out-dir",
        required=True,
        help="Destination for .pt shards (split subdirs created automatically)",
    )
    parser.add_argument(
        "--splits",
        nargs="*",
        default=["train_100M", "val_5M", "test_20M"],
        help="Split subdirectories to convert (default: train_100M val_5M test_20M)",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=8,
        help="Number of ProcessPoolExecutor workers (default: 8)",
    )
    parser.add_argument(
        "--max-files",
        type=int,
        default=None,
        metavar="N",
        help="Convert only the first N .root files per split (smoke test)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing .pt files",
    )
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="Skip conversion; run CSR verification gates on existing .pt files",
    )
    parser.add_argument(
        "--verify-jets",
        type=int,
        default=32,
        metavar="N",
        help="Jets sampled per shard for ROOT round-trip check (default: 32)",
    )
    parser.add_argument(
        "--verify-shards",
        type=int,
        default=None,
        metavar="N",
        help=(
            "Shards sampled per split for ROOT round-trip. "
            "Default: all for --max-files runs, 20 otherwise"
        ),
    )
    parser.add_argument(
        "--verify-all",
        action="store_true",
        help="Deep-check ALL shards (overrides --verify-shards default)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for verify spot-check reproducibility (default: 42)",
    )
    args = parser.parse_args(argv)

    # --- Verify-only path ---
    if args.verify_only:
        return _verify_pass(args)

    # --- Convert path ---
    root_dir = args.root_dir
    out_dir = args.out_dir
    splits = args.splits

    if not os.path.isdir(root_dir):
        print(f"ERROR: --root-dir not found: {root_dir}", file=sys.stderr)
        return 1

    os.makedirs(out_dir, exist_ok=True)

    total_files = 0
    total_ok = 0
    total_skipped = 0
    total_failed = 0
    all_errors: List[str] = []

    for split in splits:
        n_files, n_ok, n_skipped, n_failed, errs = _convert_split(
            root_dir=root_dir,
            out_dir=out_dir,
            split=split,
            num_workers=args.num_workers,
            max_files=args.max_files,
            force=args.force,
        )
        total_files += n_files
        total_ok += n_ok
        total_skipped += n_skipped
        total_failed += n_failed
        all_errors.extend(errs)
        print(
            f"  [{split}] {n_files} files: "
            f"{n_ok} ok, {n_skipped} skipped, {n_failed} failed"
        )

    print(
        f"\nTotal: {total_files} files, "
        f"{total_ok} ok, {total_skipped} skipped, {total_failed} failed"
    )

    if total_failed > 0:
        print(f"\n{total_failed} file(s) failed conversion:", file=sys.stderr)
        for e in all_errors[:20]:
            print(f"  {e}", file=sys.stderr)
        if len(all_errors) > 20:
            print(f"  ... and {len(all_errors) - 20} more", file=sys.stderr)
        return 1

    # --- Post-convert verify pass ---
    # Always run, even when every file was skipped (resume).  A prior crashed
    # run may have left bad shards on disk that a skip-everything pass would
    # silently accept without a separate --verify-only invocation.
    if total_ok == 0:
        print(
            "\nAll files skipped (already exist) — "
            "running verify pass on existing shards"
        )
    else:
        print("\nRunning post-convert verification sample ...")
    # Override verify-shards for max-files runs
    if args.max_files is not None:
        args.verify_shards = None  # --verify-all will catch all
        args.verify_all = True
    return _verify_pass(args)


# ---------------------------------------------------------------------------
# Convenience: create a DataLoader from a shard directory
# ---------------------------------------------------------------------------
# create_ragged_dataloader is imported from ragged_loader.py at the top.


# ---------------------------------------------------------------------------
# Smoke self-check (runs when executed directly with no arguments on a laptop)
# ---------------------------------------------------------------------------


def _smoke_self_check() -> int:
    """Lightweight self-tests that don't need ROOT files.

    Verifies: shard schema round-trip, dataset slicing, collate shapes,
    structural verify on synthetic data, verify detects injected errors.
    """
    import tempfile

    print("=== Smoke self-check (synthetic data) ===")

    # --- Build a tiny synthetic shard ---
    N_jets = 10
    counts = np.array([3, 5, 2, 0, 7, 4, 1, 6, 8, 9], dtype=np.int32)
    N_part = int(counts.sum())
    offsets = np.empty(N_jets + 1, dtype=np.int64)
    offsets[0] = 0
    np.cumsum(counts, out=offsets[1:])

    rng = np.random.default_rng(42)
    x = rng.normal(size=(N_part, 16)).astype(np.float32)
    v = np.abs(rng.normal(size=(N_part, 4)).astype(np.float32))  # E > 0
    jet = np.column_stack(
        [
            rng.normal(size=N_jets).astype(np.float32)
            for _ in range(9)
        ]
        + [counts.astype(np.float32)]
    )
    y = np.zeros((N_jets, 10), dtype=np.float32)
    y[np.arange(N_jets), rng.integers(0, 10, size=N_jets)] = 1.0

    data = {
        "x": x,
        "v": v,
        "offsets": offsets,
        "jet": jet,
        "y": y,
        "n_particles": counts,
    }
    print(f"  Synthetic shard: {N_jets} jets, {N_part} particles")

    # --- §A: structural verify on clean data ---
    errs = _verify_shard_arrays(data, source="synthetic")
    assert len(errs) == 0, f"Clean verify failed: {errs}"
    print("  §A clean verify: OK")

    # --- §A: verify catches injected errors ---
    bad_data = {
        "x": x.copy(),
        "v": v.copy(),
        "offsets": offsets.copy(),
        "jet": jet.copy(),
        "y": y.copy(),
        "n_particles": counts.copy(),
    }
    # Inject: offset mismatch
    bad_data["offsets"][-1] += 1
    errs = _verify_shard_arrays(bad_data, source="bad")
    assert len(errs) > 0, "Should have caught offset mismatch"
    print(f"  §A caught injected errors: {len(errs)} error(s)")

    # --- Save / load round-trip ---
    with tempfile.TemporaryDirectory() as tmp:
        pt_path = os.path.join(tmp, "test.pt")
        tensors = {k: torch.from_numpy(v) for k, v in data.items()}
        torch.save(tensors, pt_path, _use_new_zipfile_serialization=True)

        loaded = torch.load(pt_path, map_location="cpu", weights_only=True)
        assert torch.allclose(loaded["x"], tensors["x"])
        assert torch.allclose(loaded["offsets"], tensors["offsets"])
        print("  Save/load round-trip: OK")

        # --- Dataset slicing ---
        # Write to a temp dir and instantiate dataset
        ds = RaggedShardDataset(tmp, events_per_shard=N_jets)
        assert len(ds) == N_jets
        print(f"  Dataset len={len(ds)}: OK")

        # Check first and last items
        for idx in [0, N_jets - 1]:
            x_i, v_i, mask_i, y_i, n_p_i = ds[idx]
            n_expected = int(counts[idx])
            assert x_i.shape == (n_expected, 16), f"idx={idx}: x shape {x_i.shape}"
            assert v_i.shape == (n_expected, 4)
            assert mask_i.shape == (n_expected, 1)
            assert y_i.shape == (10,)
            assert int(n_p_i) == n_expected
            # mask should be all 1s (E > 0 for all real particles)
            assert mask_i.sum() == n_expected
        print("  Dataset slicing: OK")

        # --- Collate ---
        batch = [ds[i] for i in range(4)]  # jets with 3,5,2,0 particles
        # Drop the empty jet (idx=3) for collate — it's valid but P_max=0
        # causes an empty tensor; test both paths
        batch_no_empty = [ds[i] for i in [0, 1, 2, 5]]  # 3,5,2,4 particles
        x_b, v_b, m_b, y_b = ragged_collate(batch_no_empty)
        assert x_b.shape == (4, 16, 5), f"x_batch shape {x_b.shape}"
        assert v_b.shape == (4, 4, 5)
        assert m_b.shape == (4, 1, 5)
        assert y_b.shape == (4, 10)
        # Verify mask sum per row
        for b_idx, orig_idx in enumerate([0, 1, 2, 5]):
            assert int(m_b[b_idx].sum()) == int(counts[orig_idx]), (
                f"row {b_idx}: mask sum {int(m_b[b_idx].sum())} != "
                f"n_p {int(counts[orig_idx])}"
            )
        print("  Collate shapes & mask guard: OK")

    print("=== All smoke self-checks passed ===\n")
    return 0


if __name__ == "__main__":
    # If run with no arguments, run smoke self-check (safe on laptop)
    if len(sys.argv) == 1:
        raise SystemExit(_smoke_self_check())
    raise SystemExit(main())
