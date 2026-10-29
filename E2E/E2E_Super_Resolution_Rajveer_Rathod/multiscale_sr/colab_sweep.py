"""Colab sweep controller: isolated attempts, checkpoint recovery and truthful status.

The notebook is configuration/UI; this script owns the testable control flow.
Run from the multiscale_sr project directory. One controller per sweep at a time.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time
import uuid

from multiscale_sr.recovery import atomic_json, sha256, verified_copies


def digest(payload):
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:16]


def dataset_manifest(root):
    files = sorted(p for p in root.iterdir() if p.suffix in (".parquet", ".hdf5"))
    if not files:
        raise FileNotFoundError(f"No dataset files in {root}")
    # Metadata identity, not a content checksum. Document same-size/mtime caveat.
    return [{"name": p.name, "size": p.stat().st_size, "mtime_ns": p.stat().st_mtime_ns}
            for p in files]


def stream(cmd, log_path, phase):
    """Unbuffered output with durable logs and an explicit subprocess outcome."""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"[{phase}] {' '.join(map(str, cmd))}", flush=True)
    with log_path.open("a", buffering=1) as log:
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                text=True, bufsize=1, env={**os.environ, "PYTHONUNBUFFERED": "1"})
        try:
            for line in proc.stdout:
                print(line, end="", flush=True)
                log.write(line)
            code = proc.wait()
            log.write(f"\n[{phase}] returncode={code}; negative codes indicate POSIX signals\n")
            return code
        except BaseException:
            proc.terminate()
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait()
            raise


def valid_evaluation(path):
    try:
        p = json.loads(path.read_text())
        auc = p["primary_fixed_hr_tagger"]["auc"]
        return (all(isinstance(auc[s], (int, float)) and math.isfinite(auc[s])
                    and 0 <= auc[s] <= 1 for s in ("hr", "lr", "sr"))
                and p["n_train"] > 0 and p["n_test"] > 0
                and p["semd"]["params"]["n_samples"] > 0
                and all(math.isfinite(p["semd"][s]["mean"])
                        for s in ("sr_vs_hr", "lr_vs_hr")))
    except (OSError, ValueError, KeyError, TypeError):
        return False


def training_complete(attempt):
    try:
        meta = json.loads((attempt / "trained.json").read_text())
        best = attempt / "artifacts/checkpoints/best.pt"
        return sha256(best) == meta["best_sha256"]
    except (OSError, ValueError, KeyError, TypeError):
        return False


def completed(attempt, identity, eval_id):
    try:
        meta = json.loads((attempt / "run_identity.json").read_text())
        out = attempt / "evaluations" / eval_id
        marker = json.loads((out / "complete.json").read_text())
        return (meta == identity and training_complete(attempt)
                and marker["eval_id"] == eval_id
                and marker["best_sha256"] == sha256(attempt / "artifacts/checkpoints/best.pt")
                and marker["eval_sha256"] == sha256(out / "classification_eval.json")
                and valid_evaluation(out / "classification_eval.json"))
    except (OSError, ValueError, KeyError, TypeError):
        return False


def matching_attempts(seed_root, identity):
    matches = []
    for p in sorted(seed_root.glob("attempt_*"), reverse=True):
        if not p.is_dir() or p.name.endswith(".partial"):
            continue
        try:
            if json.loads((p / "run_identity.json").read_text()) == identity:
                matches.append(p)
        except (OSError, ValueError):
            pass
    return matches


def run_seed(args, sweep_root, fingerprint, seed, eval_id, eval_options, tagger_path):
    identity = {"training_fingerprint": fingerprint, "run_variant": args.variant,
                "seed": seed, "scale": args.scale, "epochs": args.epochs}
    seed_root = sweep_root / f"seed_{seed}"
    matches = matching_attempts(seed_root, identity)
    for attempt in matches:
        if completed(attempt, identity, eval_id):
            return {"status": "complete", "action": "reused", "path": str(attempt)}
    attempt = next((p for p in matches if training_complete(p) or
                    verified_copies(p / "checkpoints", "latest")), None)
    if attempt is None:
        attempt = seed_root / f"attempt_{time.time_ns()}_{uuid.uuid4().hex[:8]}"
        atomic_json(attempt / "run_identity.json", identity)
    local = Path(args.local_root) / sweep_root.name / f"seed_{seed}" / attempt.name
    action = "evaluate_only"
    if not training_complete(attempt):
        copies = verified_copies(attempt / "checkpoints", "latest")
        # Every execution gets its own local directory, so lost/corrupt local
        # state cannot be mistaken for an existing successful training result.
        local = local / f"session_{uuid.uuid4().hex[:8]}"
        train_cmd = [sys.executable, "-u", "train.py", "--config", args.config,
            "--data-dir", str(Path(args.data_dir).resolve()), "--scale", str(args.scale),
            "--epochs", str(args.epochs), "--seed", str(seed), "--run-dir", str(local),
            "--run-name", f"{args.variant}_{fingerprint}_seed_{seed}",
            "--run-fingerprint", fingerprint, "--recovery-dir", str(attempt),
            "--checkpoint-seconds", str(args.checkpoint_seconds),
            "--cache", "--cache-dir", args.cache_dir, "--semd-chunk", str(args.semd_chunk)]
        if args.smoke:
            train_cmd += smoke_training_options()
        if args.no_wandb:
            train_cmd.append("--no-wandb")
        action = "resume" if copies else "train"
        if copies:
            local.mkdir(parents=True, exist_ok=True)
            resume = local / "resume.pt"
            shutil.copyfile(copies[0][1], resume)
            if sha256(resume) != copies[0][0]["sha256"]:
                raise IOError("Local resume copy failed verification")
            train_cmd += ["--resume", str(resume)]
            import torch
            restored = torch.load(resume, map_location="cpu", weights_only=True)
            best = verified_copies(attempt / "checkpoints", "best")
            best_path = None
            for _, candidate in best:
                candidate_state = torch.load(candidate, map_location="cpu", weights_only=True)
                if (candidate_state["global_step"] <= restored["global_step"]
                        and candidate_state["best_val"] == restored["best_val"]):
                    best_path = candidate
                    break
            if best_path:
                (local / "checkpoints").mkdir(exist_ok=True)
                shutil.copyfile(best_path, local / "checkpoints/best.pt")
            elif math.isfinite(restored["best_val"]):
                if (restored["epoch_complete"] and restored["history"]
                        and restored["history"][-1]["val_l1"] == restored["best_val"]):
                    (local / "checkpoints").mkdir(exist_ok=True)
                    shutil.copyfile(resume, local / "checkpoints/best.pt")
                else:
                    raise ValueError("No verified best checkpoint compatible with the recovery point. "
                                     "Preserve this attempt and use a new RUN_VARIANT to retrain.")
            del restored
            if best:
                del candidate_state
        atomic_json(attempt / "status.json", {"phase": action, "local": str(local)})
        code = stream(train_cmd, attempt / f"train-{time.time_ns()}.log", action)
        if code:
            return {"status": "failed", "phase": "training", "returncode": code,
                    "path": str(attempt)}
        # Exit code alone is insufficient: verify the epoch budget and checkpoint.
        import torch
        final = torch.load(local / "checkpoints/latest.pt", map_location="cpu", weights_only=True)
        if (not final["epoch_complete"] or final["epoch"] != args.epochs
                or final["args"]["seed"] != seed):
            raise ValueError("Training exited without completing the requested seed/epoch budget")
        del final
        best_path = local / "checkpoints/best.pt"
        best = torch.load(best_path, map_location="cpu", weights_only=True)
        if best["args"]["run_fingerprint"] != fingerprint:
            raise ValueError("Best checkpoint identity mismatch")
        del best
        # Unique staging area; never remove old artifact directories to retry.
        staging = attempt / f"artifacts_{uuid.uuid4().hex[:8]}.partial"
        shutil.copytree(local, staging, ignore=shutil.ignore_patterns("resume.pt"))
        if sha256(staging / "checkpoints/best.pt") != sha256(best_path):
            raise IOError("Artifact copy failed verification")
        if (attempt / "artifacts").exists():
            (attempt / "artifacts").rename(attempt / f"artifacts_previous_{time.time_ns()}")
        staging.rename(attempt / "artifacts")
        atomic_json(attempt / "trained.json", {"best_sha256": sha256(best_path),
                    "epochs": args.epochs, "seed": seed})
    best_path = attempt / "artifacts/checkpoints/best.pt"
    out = attempt / "evaluations" / eval_id
    # Failed evaluation outputs are retained in a unique attempt directory.
    staging = out.parent / f"{eval_id}_{uuid.uuid4().hex[:8]}.partial"
    atomic_json(attempt / "status.json", {"phase": "evaluation"})
    cmd = [sys.executable, "-u", "classification_eval.py", "--checkpoint", str(best_path),
           "--data-dir", args.data_dir, "--out-dir", str(staging),
           "--tagger-checkpoint", str(tagger_path), *eval_options]
    code = stream(cmd, attempt / f"eval-{time.time_ns()}.log", "evaluation")
    if code or not valid_evaluation(staging / "classification_eval.json"):
        return {"status": "failed", "phase": "evaluation", "returncode": code,
                "path": str(attempt)}
    evaluated = json.loads((staging / "classification_eval.json").read_text())
    if evaluated["scale"] != args.scale or evaluated["eval_seed"] != args.eval_seed:
        raise ValueError("Evaluation scale/seed does not match the requested measurement protocol")
    if out.exists():
        out.rename(out.with_name(out.name + f"_previous_{time.time_ns()}.partial"))
    staging.rename(out)
    atomic_json(out / "complete.json", {"eval_id": eval_id,
        "best_sha256": sha256(best_path), "eval_sha256": sha256(out / "classification_eval.json")})
    atomic_json(attempt / "status.json", {"phase": "complete"})
    return {"status": "complete", "action": action, "path": str(attempt)}


def smoke_training_options():
    return ["--max-train-batches", "2", "--max-val-batches", "1", "--max-stats-batches", "1",
            "--gen-channels", "8", "--gen-blocks", "1", "--batch-size", "2", "--semd-topk", "8"]


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", required=True)
    p.add_argument("--data-dir", required=True)
    p.add_argument("--drive-root", required=True)
    p.add_argument("--local-root", default="/content/experiments")
    p.add_argument("--cache-dir", default="/content/.sr_cache")
    p.add_argument("--variant", default="learnable_lr_skip_v1")
    p.add_argument("--scale", type=int, required=True)
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--seeds", type=int, nargs="+", default=[123, 456, 789, 999])
    p.add_argument("--eval-seed", type=int, default=0)
    p.add_argument("--semd-topk", type=int, default=128)
    p.add_argument("--semd-omega-R", type=float, default=1.0)
    p.add_argument("--semd-chunk", type=int, default=1)
    p.add_argument("--semd-max-samples", type=int, default=1000)
    p.add_argument("--checkpoint-seconds", type=float, default=300)
    p.add_argument("--no-wandb", action="store_true")
    p.add_argument("--result-file", default=None)
    p.add_argument("--smoke", action="store_true",
                   help="Explicitly isolated smoke experiment: tiny model/batch caps; requires epochs=1")
    args = p.parse_args(argv)
    if not re.fullmatch(r"[A-Za-z0-9_-]+", args.variant):
        p.error("variant must contain only letters, digits, underscore or hyphen")
    if len(set(args.seeds)) != len(args.seeds):
        p.error("seeds must be unique")
    if args.smoke:
        if args.epochs != 1:
            p.error("--smoke requires --epochs 1")
        args.variant += "_smoke"
    os.environ["MULTISCALE_SR_NUM_WORKERS"] = "0"
    import torch
    from train import resolve_args
    resolved = vars(resolve_args(["--config", args.config, "--data-dir", str(Path(args.data_dir).resolve()),
            "--scale", str(args.scale), "--epochs", str(args.epochs), "--cache",
            "--semd-chunk", str(args.semd_chunk),
            *(smoke_training_options() if args.smoke else [])]))
    for key in ("seed", "run_name", "run_dir", "recovery_dir", "resume", "experiments_root",
                "cache_dir", "checkpoint_seconds", "no_wandb", "wandb_project", "wandb_entity"):
        resolved.pop(key, None)
    identity = {"files": {str(f): sha256(f) for f in sorted([
                    Path("train.py"), Path("colab_sweep.py"), *Path("multiscale_sr").rglob("*.py")])},
                "config": resolved, "data": dataset_manifest(Path(args.data_dir))}
    fingerprint = digest(identity)
    sweep_root = Path(args.drive_root) / f"{args.variant}_{args.scale}x_{fingerprint}"
    sweep_root.mkdir(parents=True, exist_ok=True)
    eval_options = ["--eval-seed", str(args.eval_seed), "--semd-topk", str(args.semd_topk),
        "--semd-omega-R", str(args.semd_omega_R), "--semd-chunk", str(args.semd_chunk),
        "--semd-max-samples", str(args.semd_max_samples), "--val-ratio", str(resolved["val_ratio"])]
    if args.smoke:
        eval_options += ["--max-samples", "32", "--tagger-epochs", "1", "--batch-size", "2"]
    eval_id = digest({"source": sha256(Path("classification_eval.py")), "options": eval_options})
    # Ruler shared across SR seeds; scoped to data, preprocessing and eval protocol.
    tagger_path = sweep_root / "taggers" / f"hr_{eval_id}.pt"
    revision = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True)
    atomic_json(sweep_root / "provenance.json", {**identity, "git_commit": revision.stdout.strip() or None})
    result = {"training_fingerprint": fingerprint, "eval_id": eval_id,
              "sweep_root": str(sweep_root), "results": {}, "seeds_completed": []}
    def save_summary():
        result["seeds_completed"] = [int(s) for s, r in result["results"].items() if r["status"] == "complete"]
        atomic_json(sweep_root / f"summary_{eval_id}.json", result)
        if args.result_file:
            atomic_json(Path(args.result_file), result)
    save_summary()
    print(f"[sweep] {sweep_root}; evaluation={eval_id}", flush=True)
    for seed in args.seeds:
        try:
            result["results"][str(seed)] = run_seed(args, sweep_root, fingerprint, seed,
                                                  eval_id, eval_options, tagger_path)
        except Exception as exc:
            import traceback
            traceback.print_exc()
            (sweep_root / f"error_seed_{seed}_{time.time_ns()}.log").write_text(traceback.format_exc())
            result["results"][str(seed)] = {"status": "failed", "error": f"{type(exc).__name__}: {exc}"}
        save_summary()
        print(f"[seed {seed}] {result['results'][str(seed)]}", flush=True)
    success = len(result["seeds_completed"]) == len(args.seeds)
    print("All requested seeds completed and persisted." if success else
          f"Sweep incomplete: {len(result['seeds_completed'])}/{len(args.seeds)} complete; inspect summary/logs.")
    return 0 if success else 1


if __name__ == "__main__":
    raise SystemExit(main())
