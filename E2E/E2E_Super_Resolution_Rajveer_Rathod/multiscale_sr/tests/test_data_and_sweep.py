import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import torch

from multiscale_sr.data.normalization import batch_to_tensor, split_files
from multiscale_sr.data.cache import ensure_hr_cache, _cache_is_valid


@pytest.mark.parametrize("dtype", [pa.float32(), pa.float64()])
@pytest.mark.parametrize("channels_last", [False, True])
def test_arrow_offsets_and_layout(dtype, channels_last):
    x = np.arange(4*3*5*5).reshape(4,3,5,5).astype(np.float32)
    stored = x.transpose(0,2,3,1) if channels_last else x
    arr = pa.array(stored.tolist(), type=pa.list_(pa.list_(pa.list_(dtype))))
    actual = batch_to_tensor(arr.slice(1,2))
    torch.testing.assert_close(actual, torch.from_numpy(x[1:3]))


def make_parquet(path, n=64, side=32):
    rng = np.random.default_rng(4)
    hr = rng.random((n,3,side,side), dtype=np.float32)
    # Every held-out parity subset contains both classes.
    table = pa.table({"X_jets_LR": hr[:,:,::2,::2].tolist(), "X_jets": hr.tolist(),
                      "pt": list(range(n)), "m0": [1.]*n, "y": [(i//2)%2 for i in range(n)]})
    pq.write_table(table, path)


def test_cache_interruption_and_source_change(tmp_path, monkeypatch):
    import multiscale_sr.data.cache as cache
    path = tmp_path / "data.parquet"
    make_parquet(path, n=4, side=8)
    root = tmp_path / "cache"
    ensure_hr_cache([path], root)
    assert _cache_is_valid(root, [path])
    make_parquet(path, n=6, side=8)
    assert not _cache_is_valid(root, [path])
    original = cache.batch_to_tensor
    def fail(*args):
        raise RuntimeError("decode interrupted")
    monkeypatch.setattr(cache, "batch_to_tensor", fail)
    with pytest.raises(RuntimeError):
        ensure_hr_cache([path], root)
    assert not _cache_is_valid(root, [path])
    monkeypatch.setattr(cache, "batch_to_tensor", original)
    ensure_hr_cache([path], root)
    assert _cache_is_valid(root, [path])
    with pytest.raises(ValueError, match="disjoint"):
        split_files([path], .33)


def test_full_sweep_cli_smoke(tmp_path):
    """Real subprocesses: parquet -> training -> eval -> verified reuse."""
    data = tmp_path / "data"
    data.mkdir()
    for i in range(3):
        make_parquet(data / f"{i}.parquet")
    config = tmp_path / "config.yaml"
    config.write_text("scale: 16\nhr_size: 32\nval_ratio: 0.33\n")
    result = tmp_path / "result.json"
    cmd = [sys.executable, "colab_sweep.py", "--config", str(config), "--data-dir", str(data),
           "--drive-root", str(tmp_path / "drive"), "--local-root", str(tmp_path / "local"),
           "--cache-dir", str(tmp_path / "cache"), "--variant", "regression",
           "--scale", "16", "--epochs", "1", "--seeds", "123", "--smoke",
           "--semd-topk", "8", "--semd-max-samples", "8", "--no-wandb", "--result-file", str(result)]
    import os
    env = {**os.environ, "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1",
           "MULTISCALE_SR_DEVICE": "cpu", "MPLCONFIGDIR": str(tmp_path / "mpl")}
    first = subprocess.run(cmd, capture_output=True, text=True, env=env, timeout=180)
    assert first.returncode == 0, first.stdout + first.stderr
    payload = json.loads(result.read_text())
    assert payload["results"]["123"]["status"] == "complete"
    second = subprocess.run(cmd, capture_output=True, text=True, env=env, timeout=30)
    assert second.returncode == 0, second.stdout + second.stderr
    assert json.loads(result.read_text())["results"]["123"]["action"] == "reused"
    # Removing the eval completion marker must retry evaluation, not training.
    run = Path(payload["results"]["123"]["path"])
    (run / "evaluations" / payload["eval_id"] / "complete.json").unlink()
    third = subprocess.run(cmd, capture_output=True, text=True, env=env, timeout=180)
    assert third.returncode == 0, third.stdout + third.stderr
    assert json.loads(result.read_text())["results"]["123"]["action"] == "evaluate_only"
    # A lost final-artifact marker recovers from Drive checkpoints, including
    # when all training epochs had already finished before the interruption.
    (run / "trained.json").rename(run / "trained.previous.json")
    fourth = subprocess.run(cmd, capture_output=True, text=True, env=env, timeout=180)
    assert fourth.returncode == 0, fourth.stdout + fourth.stderr
    assert json.loads(result.read_text())["results"]["123"]["action"] == "resume"
