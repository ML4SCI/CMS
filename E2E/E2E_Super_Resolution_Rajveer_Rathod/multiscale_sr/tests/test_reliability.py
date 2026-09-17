import gc
import json
from pathlib import Path
import random
import threading
import time
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from multiscale_sr.spectral import _spectral_cross, semd_images, cumulative_spectral
from multiscale_sr.models import Generator
from multiscale_sr.recovery import publish_checkpoint, verified_copies, atomic_json, sha256
from multiscale_sr.utils.env import prefetch_generator
from multiscale_sr.training_state import capture_rng, restore_rng, validate_resume, adversarial_weight, noise_std
from multiscale_sr.experiment import make_experiment_dir
from colab_sweep import matching_attempts, completed, valid_evaluation, stream
from train import resolve_args

torch.set_num_threads(1)


def dense_cross(a, x, b, y):
    am, ap = cumulative_spectral(x)
    bm, bp = cumulative_spectral(y)
    overlap = (torch.minimum(ap[:, :, None], bp[:, None, :]) -
               torch.maximum(am[:, :, None], bm[:, None, :])).relu()
    return (a[:, :, None] * b[:, None, :] * overlap).sum((1, 2))


@pytest.mark.parametrize("seed", range(4))
def test_cross_values_and_gradients(seed):
    torch.manual_seed(seed)
    inputs = [torch.rand(2, n, dtype=torch.double, requires_grad=True)
              for n in (13, 13, 17, 17)]
    actual = _spectral_cross(*inputs)
    expected = dense_cross(*inputs)
    torch.testing.assert_close(actual, expected)
    for a, b in zip(torch.autograd.grad(actual.sum(), inputs),
                    torch.autograd.grad(expected.sum(), inputs)):
        torch.testing.assert_close(a, b, atol=1e-10, rtol=1e-10)


def test_cross_zero_weights_and_ties():
    a = torch.arange(6, dtype=torch.double)[None]
    x = torch.tensor([[0., 2., 0., 1., 0., 3.]], dtype=torch.double)
    torch.testing.assert_close(_spectral_cross(a, x, a, x), dense_cross(a, x, a, x))


@pytest.mark.parametrize("topk", [1, 8, 128, 1000])
def test_semd_boundaries(topk):
    torch.manual_seed(3)
    x = torch.rand(2, 3, 8, 8, requires_grad=True)
    y = torch.rand_like(x)
    d = semd_images(x, y, topk=topk)
    assert torch.isfinite(d).all() and (d >= 0).all()
    d.sum().backward()
    assert torch.isfinite(x.grad).all()
    torch.testing.assert_close(d, semd_images(y, x.detach(), topk=topk), rtol=2e-5, atol=0.1)
    torch.testing.assert_close(semd_images(x.detach(), x.detach(), topk=topk), torch.zeros(2), atol=0.01, rtol=0)
    assert semd_images(torch.zeros_like(x), torch.zeros_like(x), topk=topk).eq(0).all()
    with pytest.raises(ValueError):
        semd_images(x, y, chunk=0)


@pytest.mark.parametrize("scale", [16, 32, 64])
def test_generator_parameter_and_roundtrip(scale):
    torch.manual_seed(5)
    model = Generator(base_channels=4, num_blocks=1)
    opt = torch.optim.Adam(model.parameters(), lr=0.01)
    x = torch.rand(1, 3, scale, scale)
    out = model(x, (128, 128))
    assert out.shape == (1, 3, 128, 128)
    out.square().mean().backward()
    assert torch.isfinite(model.lr_skip_alpha.grad)
    assert model.lr_skip_alpha.grad.abs() > 0
    old = model.lr_skip_alpha.detach().clone()
    opt.step()
    assert not torch.equal(old, model.lr_skip_alpha)
    clone = Generator(base_channels=4, num_blocks=1)
    clone.load_state_dict(model.state_dict(), strict=True)
    torch.testing.assert_close(clone(x, (128, 128)), model(x, (128, 128)))
    clone.lr_skip = False
    clone(x, (128, 128)).mean().backward()
    assert clone.lr_skip_alpha.grad is None


def test_prefetch_propagates_errors_and_closes():
    def data():
        yield 1
        raise ValueError("decode failed")
    it = prefetch_generator(data())
    assert next(it) == 1
    with pytest.raises(ValueError, match="decode failed"):
        next(it)
    before = threading.active_count()
    threads = []
    for _ in range(12):
        it = prefetch_generator(iter(range(10000)), maxsize=1)
        threads.append(it._thread)
        next(it)
        del it
    gc.collect()
    for thread in threads:
        thread.join(timeout=2)
        assert not thread.is_alive()
    assert threading.active_count() <= before


def test_verified_rotation_survives_corruption(tmp_path):
    source = tmp_path / "source.pt"
    root = tmp_path / "drive"
    source.write_bytes(b"first")
    first = publish_checkpoint(source, root, "latest")
    source.write_bytes(b"second")
    second = publish_checkpoint(source, root, "latest")
    assert verified_copies(root, "latest")[0][1] == second
    second.write_bytes(b"truncated")
    assert verified_copies(root, "latest")[0][1] == first


def test_failed_copy_preserves_previous(tmp_path, monkeypatch):
    import multiscale_sr.recovery as recovery
    source = tmp_path / "source"
    source.write_bytes(b"good")
    first = publish_checkpoint(source, tmp_path / "drive", "latest")
    def fail(*args):
        raise OSError("disk full")
    monkeypatch.setattr(recovery.shutil, "copyfile", fail)
    with pytest.raises(OSError):
        publish_checkpoint(source, tmp_path / "drive", "latest")
    assert verified_copies(tmp_path / "drive", "latest")[0][1] == first


def test_identity_and_completion(tmp_path):
    identity = {"seed": 123, "training_fingerprint": "v1", "run_variant": "alpha"}
    root = tmp_path / "seed_123"
    for name in ("attempt_old", "attempt_new", "attempt_new.partial"):
        atomic_json(root / name / "run_identity.json", identity)
    matches = matching_attempts(root, identity)
    assert len(matches) == 2
    assert not matching_attempts(tmp_path / "seed_1234", identity)
    assert not matching_attempts(root, {**identity, "training_fingerprint": "v2"})
    run = root / "attempt_new"
    assert not completed(run, identity, "eval")
    best = run / "artifacts/checkpoints/best.pt"
    best.parent.mkdir(parents=True)
    best.write_bytes(b"test checkpoint")
    atomic_json(run / "trained.json", {"best_sha256": sha256(best)})
    evaluation = run / "evaluations/eval/classification_eval.json"
    payload = {"n_train": 10, "n_test": 4,
               "primary_fixed_hr_tagger": {"auc": {s: .8 for s in ("hr", "lr", "sr")}},
               "semd": {"params": {"n_samples": 4},
                        **{s: {"mean": 1.} for s in ("sr_vs_hr", "lr_vs_hr")}}}
    atomic_json(evaluation, payload)
    atomic_json(evaluation.parent / "complete.json", {"eval_id": "eval",
                "best_sha256": sha256(best), "eval_sha256": sha256(evaluation)})
    assert completed(run, identity, "eval")
    assert not completed(run, identity, "different_eval")
    evaluation.write_text("{}")
    assert not completed(run, identity, "eval")
    assert not valid_evaluation(evaluation)


def test_rng_schedule_and_resume_validation():
    state = capture_rng()
    expected = (random.random(), np.random.rand(), torch.rand(1))
    restore_rng(state)
    actual = (random.random(), np.random.rand(), torch.rand(1))
    assert expected == actual
    args = SimpleNamespace(seed=123, scale=64, adv_warmup_epochs=3,
        adv_ramp_epochs=5, lambda_adv=1., d_input_noise=.05, epochs=40)
    assert adversarial_weight(args, 4) == .2
    assert adversarial_weight(args, 20) == 1.
    assert noise_std(args, 40) == 0.
    ckpt = {"recovery_version": 1, "args": vars(args).copy()}
    validate_resume(ckpt, args)
    args.seed = 456
    with pytest.raises(ValueError, match="seed"):
        validate_resume(ckpt, args)
    with pytest.raises(ValueError, match="legacy"):
        validate_resume({}, args)


def test_explicit_default_cli_overrides_yaml(tmp_path):
    cfg = tmp_path / "config.yaml"
    cfg.write_text("epochs: 2\nseed: 123\n")
    args = resolve_args(["--config", str(cfg), "--data-dir", str(tmp_path), "--scale", "64",
                         "--epochs", "50", "--seed", "42"])
    assert args.epochs == 50 and args.seed == 42


def test_refuse_experiment_overwrite(tmp_path):
    make_experiment_dir(tmp_path, "data", 64, "run")
    with pytest.raises(FileExistsError):
        make_experiment_dir(tmp_path, "data", 64, "run")


def test_subprocess_failure_is_logged(tmp_path):
    import sys
    log = tmp_path / "failure.log"
    assert stream([sys.executable, "-c", "print('failure'); raise SystemExit(7)"], log, "test") == 7
    assert "returncode=7" in log.read_text()


def test_sweep_continues_and_summary_is_truthful(tmp_path, monkeypatch):
    import colab_sweep
    (tmp_path / "data.parquet").write_bytes(b"identity-only fixture")
    config = tmp_path / "config.yaml"
    config.write_text("scale: 64\n")
    seen = []
    def run(args, root, fingerprint, seed, *rest):
        seen.append(seed)
        if seed == 123:
            raise OSError("Drive unavailable")
        return {"status": "complete", "action": "reused", "path": str(root / f"seed_{seed}")}
    monkeypatch.setattr(colab_sweep, "run_seed", run)
    result = tmp_path / "result.json"
    code = colab_sweep.main(["--config", str(config), "--data-dir", str(tmp_path),
        "--drive-root", str(tmp_path / "drive"), "--scale", "64", "--epochs", "1",
        "--seeds", "123", "456", "--result-file", str(result)])
    assert code == 1 and seen == [123, 456]
    assert json.loads(result.read_text())["seeds_completed"] == [456]


def test_normalization_and_parameter_shape_mismatch_rejected():
    from multiscale_sr.utils.checkpoint import load_generator_state
    model = Generator(base_channels=4, num_blocks=1)
    old = model.state_dict()
    del old["stem.0.weight"]
    with pytest.raises(RuntimeError, match="architecture"):
        load_generator_state(model, old)


def test_semd_default_topk_large_image():
    # This would materialize several GB per sample in the original implementation.
    torch.manual_seed(8)
    a = torch.rand(2, 3, 128, 128)
    b = torch.rand_like(a)
    out = semd_images(a, b, topk=128)
    assert out.shape == (2,) and torch.isfinite(out).all()


def test_collection_cap_empty_validation_and_semd_loss():
    from multiscale_sr.engine import collect_tagging_tensors, evaluate, semd_loss
    from multiscale_sr.data.normalization import ChannelStats
    model = Generator(base_channels=4, num_blocks=1)
    stats = ChannelStats(torch.zeros(3), torch.ones(3))
    batch = {"lr": torch.rand(2,3,8,8), "hr": torch.rand(2,3,16,16), "y": torch.tensor([0,1])}
    data = collect_tagging_tensors(model, [batch, batch], stats, torch.device("cpu"), max_samples=3)
    assert data["hr"].shape[0] == 3 and not data["sr"].requires_grad
    with pytest.raises(ValueError, match="no samples"):
        evaluate(model, [], stats, torch.device("cpu"))
    x = torch.rand(2,3,8,8, requires_grad=True)
    loss = semd_loss(x, torch.rand_like(x), topk=8)
    loss.backward()
    assert torch.isfinite(x.grad).all()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA runtime not available locally")
def test_repeated_cuda_validation_has_no_live_allocation_growth():
    from multiscale_sr.engine import evaluate
    from multiscale_sr.data.normalization import ChannelStats
    model = Generator(base_channels=4, num_blocks=1).cuda()
    stats = ChannelStats(torch.zeros(3), torch.ones(3))
    batch = {"lr": torch.rand(2,3,64,64), "hr": torch.rand(2,3,128,128)}
    allocated = []
    for _ in range(5):
        evaluate(model, [batch], stats, torch.device("cuda"), semd_topk=128)
        torch.cuda.synchronize()
        allocated.append(torch.cuda.memory_allocated())
    assert max(allocated[1:]) == min(allocated[1:])
