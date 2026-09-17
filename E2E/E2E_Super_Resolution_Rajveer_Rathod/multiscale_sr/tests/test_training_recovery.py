"""Exercise actual GAN updates and interruption recovery on small CPU tensors."""
from dataclasses import replace
from pathlib import Path

import pytest
import torch
from torch.utils.data import DataLoader

import train as training
from multiscale_sr.data.normalization import ChannelStats
from multiscale_sr.models import Discriminator
from multiscale_sr.utils.env import resolve_env


@pytest.mark.parametrize("interrupt_at", ["mid_epoch", "epoch_boundary", "validation"])
def test_resume_matches_uninterrupted(tmp_path, monkeypatch, interrupt_at):
    torch.set_num_threads(1)
    # Keep the architecture/optimizer/loss path real; use a smaller width for speed.
    monkeypatch.setattr(training, "Discriminator", lambda: Discriminator(base_channels=4))
    env = replace(resolve_env(), device=torch.device("cpu"), num_workers=0,
                  persistent_workers=False, prefetch_factor=None)
    monkeypatch.setattr(training, "resolve_env", lambda: env)
    rng = torch.Generator().manual_seed(90)
    samples = []
    for i in range(5):
        hr = torch.rand(3, 32, 32, generator=rng)
        samples.append({"hr": hr, "lr": torch.nn.functional.interpolate(hr[None], size=(16,16), mode="area")[0],
                        "y": torch.tensor(i % 2)})
    stats = ChannelStats(torch.zeros(3), torch.ones(3))
    def loader(**kwargs):
        return DataLoader(samples, batch_size=2, shuffle=kwargs["split"] == "train"), stats
    monkeypatch.setattr(training, "get_dataloader", loader)
    monkeypatch.setattr(training, "render_sample_grid", lambda *a, **k: None)
    monkeypatch.setattr(training, "render_metrics_plot", lambda *a, **k: None)
    real_save = training.save_checkpoint
    def run(name, resume=None):
        argv = ["--data-dir", str(tmp_path), "--dataset-format", "parquet", "--cache",
            "--scale", "16", "--hr-size", "32", "--epochs", "2", "--gen-channels", "4",
            "--gen-blocks", "1", "--batch-size", "2", "--semd-topk", "4",
            "--no-wandb", "--adv-warmup-epochs", "0", "--adv-ramp-epochs", "1",
            "--checkpoint-seconds", "0.000001", "--run-dir", str(tmp_path / name)]
        if resume:
            argv += ["--resume", str(resume)]
        try:
            training.train(training.resolve_args(argv))
        finally:
            for r in training._ACTIVE_RESOURCES:
                r.close() if hasattr(r, "close") else r.finish()
            training._ACTIVE_RESOURCES.clear()
    run("baseline")
    raised = False
    def interrupted_save(payload, path, recovery):
        nonlocal raised
        real_save(payload, path, recovery)
        steps = payload["progress"][-1] if payload["progress"] else None
        condition = ((interrupt_at == "mid_epoch" and steps == 1)
                     or (interrupt_at == "validation" and steps == 3 and not payload["epoch_complete"])
                     or (interrupt_at == "epoch_boundary" and payload["epoch_complete"]))
        if path.stem == "latest" and payload["epoch"] == 1 and condition and not raised:
            raised = True
            raise RuntimeError("simulated runtime termination")
    monkeypatch.setattr(training, "save_checkpoint", interrupted_save)
    with pytest.raises(RuntimeError, match="termination"):
        run("interrupted")
    monkeypatch.setattr(training, "save_checkpoint", real_save)
    run("interrupted", tmp_path / "interrupted/checkpoints/latest.pt")
    a = torch.load(tmp_path / "baseline/checkpoints/latest.pt", weights_only=True)
    b = torch.load(tmp_path / "interrupted/checkpoints/latest.pt", weights_only=True)
    assert a["global_step"] == b["global_step"] == 6
    assert a["history"] == b["history"]
    for name in ("generator", "discriminator"):
        for k in a[name]:
            torch.testing.assert_close(a[name][k], b[name][k], rtol=0, atol=0)
    for name in ("optimizer_g", "optimizer_d"):
        for k, state in a[name]["state"].items():
            for field, value in state.items():
                torch.testing.assert_close(value, b[name]["state"][k][field], rtol=0, atol=0)
