"""Verified rotating checkpoint copies and lightweight crash diagnostics."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import sys
import threading
import time
import uuid


def atomic_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".partial")
    with tmp.open("w") as f:
        json.dump(payload, f, indent=2)
        f.flush()
        os.fsync(f.fileno())
    tmp.replace(path)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def verified_copies(root: Path, kind: str) -> list[tuple[dict, Path]]:
    valid = []
    for slot in (0, 1):
        try:
            meta = json.loads((root / f"{kind}-{slot}.json").read_text())
            path = root / f"{kind}-{slot}.pt"
            if (meta["slot"] == slot and isinstance(meta["time_ns"], int)
                    and path.stat().st_size == meta["size"] and sha256(path) == meta["sha256"]):
                valid.append((meta, path))
        except (OSError, ValueError, KeyError, TypeError):
            pass
    return sorted(valid, key=lambda pair: pair[0]["time_ns"], reverse=True)


def publish_checkpoint(source: Path, root: Path, kind: str) -> Path:
    """Keep the previous verified generation until its replacement is verified.

    Drive is not assumed to implement atomic rename. A copy is trusted only if
    its separate checksum record matches. An interrupted older-slot overwrite
    leaves the newest verified slot available. Single writer per run required.
    """
    root.mkdir(parents=True, exist_ok=True)
    valid = verified_copies(root, kind)
    slot = 1 - valid[0][0]["slot"] if valid else 0
    target = root / f"{kind}-{slot}.pt"
    expected = sha256(source)
    shutil.copyfile(source, target)
    if sha256(target) != expected:
        raise IOError(f"Checkpoint verification failed: {target}")
    atomic_json(root / f"{kind}-{slot}.json", {
        "slot": slot, "time_ns": time.time_ns(), "size": source.stat().st_size,
        "sha256": expected,
    })
    return target


def save_checkpoint(payload: dict, path: Path, recovery_dir: Path | None = None) -> None:
    import torch
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".partial")
    with tmp.open("wb") as f:
        torch.save(payload, f)
        f.flush()
        os.fsync(f.fileno())
    tmp.replace(path)
    if recovery_dir:
        # A failed publish is fatal and visible; local checkpoint remains usable.
        publish_checkpoint(path, recovery_dir / "checkpoints", path.stem)


class Diagnostics:
    def __init__(self, root: Path, device, recovery_dir: Path | None = None):
        self.root, self.device, self.recovery_dir = root, device, recovery_dir
        self.state = {"phase": "initializing"}
        self.stop = threading.Event()
        self.started = time.monotonic()
        self.name = f"resources-{uuid.uuid4().hex[:8]}.jsonl"
        self.thread = threading.Thread(target=self._loop, daemon=True)

    def start(self):
        self.thread.start()
        return self

    def phase(self, phase, **fields):
        self.state = {**self.state, **fields, "phase": phase}
        print(f"[phase] {self.state}", flush=True)

    def _loop(self):
        import torch
        while not self.stop.is_set():
            record = {**self.state, "time": time.time(),
                      "elapsed_s": time.monotonic() - self.started,
                      "python": sys.version.split()[0], "torch": torch.__version__,
                      "platform": platform.platform(), "pid": os.getpid()}
            try:
                import psutil
                record.update(rss_bytes=psutil.Process().memory_info().rss,
                              ram_available_bytes=psutil.virtual_memory().available)
            except ImportError:
                record["ram_monitor"] = "install psutil for RSS/available RAM"
            try:
                record["disk_free_bytes"] = shutil.disk_usage(self.root).free
                if self.device.type == "cuda":
                    record.update(cuda_allocated=torch.cuda.memory_allocated(self.device),
                                  cuda_reserved=torch.cuda.memory_reserved(self.device),
                                  cuda_peak=torch.cuda.max_memory_allocated(self.device),
                                  gpu=torch.cuda.get_device_name(self.device))
                for folder in (self.root, self.recovery_dir):
                    if folder:
                        folder.mkdir(parents=True, exist_ok=True)
                        with (folder / self.name).open("a") as f:
                            f.write(json.dumps(record) + "\n")
                            f.flush()
            except Exception as exc:
                print(f"[diagnostics] {type(exc).__name__}: {exc}", flush=True)
            self.stop.wait(30)

    def close(self):
        self.stop.set()
        self.thread.join(timeout=1)
