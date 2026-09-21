"""Numerical parity tests: part_kernels fused kernels / weaver patcher vs weaver-core.

The installed weaver-core pin (git ``154db69``) is the SOLE reference for
every test in this file (Req 2.3): ``pairwise_lv_fts_pp``, ``PairEmbed`` and
``ParticleTransformer`` from ``weaver.nn.model.ParticleTransformer``.

Device handling (Req 8.2, 8.3):

- ``DEVICE`` adapts to the machine (``'cuda'`` if available, else ``'cpu'``).
- Kernel-level tests (``test_pairwise_kernel``, ``test_attention_kernel``,
  ``test_attention_backward``) measure Triton numerics and are skipped
  cleanly when triton + CUDA are unavailable.
- Patch-parity tests (``test_pair_embed_patch_parity``,
  ``test_model_patch_parity``) run on ANY device: on CPU the patch dispatches
  to weaver's own reference math so parity is near-exact (<= 1e-5 max-abs);
  on GPU the Triton kernels execute (<= 5e-2 max-abs).
- Property 4 (Hypothesis) generalizes the patch-parity checks over random
  small weaver configs and random valid ``(x, v, mask)`` batches, in both
  eval and train modes, and additionally proves ``unpatch_part_model``
  restores the exact pristine behavior.

Run under pytest (from repo root, project venv):

    .venv/bin/python -m pytest part_kernels/tests/test_parity.py -v

or as a script:

    .venv/bin/python part_kernels/tests/test_parity.py
"""
from __future__ import annotations

import copy
import math
import sys
from pathlib import Path

import pytest
import torch
from hypothesis import given, settings
from hypothesis import strategies as st

# Standalone-script bootstrap: make `part_kernels` importable when this file
# runs outside pytest (under pytest the tests/ conftest does the same thing).
_PKG_PARENT = Path(__file__).resolve().parents[2]  # .../gsoc_26 (repo root)
if str(_PKG_PARENT) not in sys.path:
    sys.path.insert(0, str(_PKG_PARENT))

from ml4sci_26.part_kernels._compat import has_triton  # noqa: E402  (CPU-safe submodule)
from variants.tests.strategies import weaver_batches  # noqa: E402  (shared strategies, task 3.7)

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
EPS = 1e-8

#: Max-abs tolerance for patched-vs-pristine parity: near-exact on CPU (the
#: patch delegates to weaver reference math), fp32-Triton tolerance on GPU.
PARITY_TOL = 5e-2 if DEVICE == 'cuda' else 1e-5

#: GPU-only kernel tests skip cleanly on machines without triton + CUDA
#: (Req 8.3: skipped, never error).
requires_triton = pytest.mark.skipif(
    not has_triton(),
    reason="requires triton + CUDA; fused-kernel numerics are GPU-only",
)

# Small weaver config shared by the patch-parity tests.
INPUT_DIM = 8
NUM_CLASSES = 5


# =====================================================================
# Helpers (weaver = sole reference)
# =====================================================================

def _ref_pairwise(v):
    """Reference pair-feature grid via weaver's own pairwise_lv_fts_pp."""
    from weaver.nn.model.ParticleTransformer import pairwise_lv_fts_pp

    xi = v.unsqueeze(-1)
    xj = v.unsqueeze(-2)
    return pairwise_lv_fts_pp(xi, xj, num_outputs=4, eps=EPS)


def _make_physical_4vec(N, P, device):
    """Physically valid four-vectors (B, 4, P) with E > |p| ([px, py, pz, E])."""
    p3 = torch.randn(N, 3, P, device=device) * 10
    mass = torch.rand(N, 1, P, device=device) * 0.5 + 0.1
    E = (p3.square().sum(dim=1, keepdim=True) + mass.square()).sqrt()
    return torch.cat([p3, E], dim=1)


def _make_mask(N, P, device, dtype=torch.float32):
    """(B, 1, P) padding mask with random valid lengths in [1, P].

    Real particle = 1, padded = 0 (weaver's mask convention). Every jet keeps
    at least one valid particle.
    """
    lengths = torch.randint(1, P + 1, (N,), device=device)
    idx = torch.arange(P, device=device).unsqueeze(0)
    return (idx < lengths.unsqueeze(1)).unsqueeze(1).to(dtype)


def _small_part_model():
    """Small weaver ParticleTransformer for CPU-fast patch-parity checks.

    Dropout is zeroed everywhere (block_params) and the sequence trimmer is
    disabled (trim=False) so train-mode forwards are deterministic and
    directly comparable between the patched and pristine models.
    """
    from weaver.nn.model.ParticleTransformer import ParticleTransformer

    return ParticleTransformer(
        input_dim=INPUT_DIM,
        num_classes=NUM_CLASSES,
        pair_input_dim=4,
        pair_extra_dim=0,
        embed_dims=(32, 32),
        pair_embed_dims=(16, 16),
        num_heads=4,
        num_layers=2,
        num_cls_layers=1,
        block_params=dict(dropout=0.0, attn_dropout=0.0, activation_dropout=0.0),
        trim=False,
        for_inference=False,
        use_amp=False,
    ).to(DEVICE)


def _patched_and_pristine(seed):
    """Build a small weaver model, return (patched, pristine, stats).

    The pristine copy is deep-copied BEFORE patching so both models carry
    identical parameters and buffers; `optimize_part_model` patches in place.
    """
    from ml4sci_26.part_kernels import optimize_part_model

    torch.manual_seed(seed)
    pristine = _small_part_model()
    patched = copy.deepcopy(pristine)
    patched, stats = optimize_part_model(patched)
    return patched, pristine, stats


def _max_abs_diff(a, b):
    return (a - b).abs().max().item()


def _ref_attention(Q, K, V, bias, pad_mask, scale, num_heads):
    """Reference softmax attention with additive bias + bool padding mask."""
    S = torch.matmul(Q, K.transpose(-2, -1)) * scale
    if bias is not None:
        S = S + bias
    if pad_mask is not None:
        NH = Q.size(0)
        N = NH // num_heads
        expanded = pad_mask.unsqueeze(1).expand(N, num_heads, -1)
        expanded = expanded.reshape(NH, 1, Q.size(1))
        S = S.masked_fill(expanded.expand_as(S), float('-inf'))
    attn = torch.softmax(S, dim=-1)
    attn = torch.nan_to_num(attn)
    return torch.matmul(attn, V)


# =====================================================================
# 1. Pairwise Lorentz-feature kernel vs weaver pairwise_lv_fts_pp (GPU-only)
# =====================================================================

@requires_triton
def test_pairwise_kernel():
    from ml4sci_26.part_kernels.triton.pairwise_kernel import fused_pairwise_lv_fts

    torch.manual_seed(42)
    for N, P in [(1, 16), (4, 32), (8, 64), (16, 128)]:
        v = _make_physical_4vec(N, P, DEVICE)

        ref = _ref_pairwise(v)
        opt = fused_pairwise_lv_fts(v, eps=EPS)

        assert ref.shape == opt.shape, f"Shape mismatch: {ref.shape} vs {opt.shape}"

        finite = ref.isfinite() & opt.isfinite()
        diff = (ref - opt).abs()
        max_diff = diff[finite].max().item() if finite.any() else 0.0
        print(f"  pairwise N={N:3d} P={P:3d}  max_abs={max_diff:.2e}")
        assert max_diff < 1e-2, f"pairwise kernel diverges: max_abs={max_diff:.2e}"


# =====================================================================
# 2. Fused attention kernel vs torch reference (GPU-only)
# =====================================================================

@requires_triton
def test_attention_kernel():
    from ml4sci_26.part_kernels.autograd.attention import fused_attention_with_bias

    torch.manual_seed(42)
    for N, H, P, D in [(2, 8, 32, 16), (4, 8, 64, 16), (8, 8, 128, 16)]:
        NH = N * H
        Q = torch.randn(NH, P, D, device=DEVICE, dtype=torch.float32)
        K = torch.randn(NH, P, D, device=DEVICE, dtype=torch.float32)
        V = torch.randn(NH, P, D, device=DEVICE, dtype=torch.float32)
        bias = torch.randn(NH, P, P, device=DEVICE, dtype=torch.float32) * 0.1
        pad_mask = torch.zeros(N, P, device=DEVICE, dtype=torch.bool)
        pad_mask[:, -P // 4:] = True

        scale = 1.0 / math.sqrt(D)

        ref = _ref_attention(Q, K, V, bias, pad_mask, scale, H)
        opt = fused_attention_with_bias(Q, K, V, bias, pad_mask, scale, H)

        max_diff = _max_abs_diff(ref, opt)
        print(f"  attention N={N} H={H} P={P:3d} D={D}  max_abs={max_diff:.2e}")
        assert max_diff < 1e-2, f"attention kernel diverges: max_abs={max_diff:.2e}"


@requires_triton
def test_attention_backward():
    from ml4sci_26.part_kernels.autograd.attention import fused_attention_with_bias

    torch.manual_seed(42)
    N, H, P, D = 2, 8, 32, 16
    NH = N * H
    scale = 1.0 / math.sqrt(D)

    Q = torch.randn(NH, P, D, device=DEVICE).requires_grad_(True)
    K = torch.randn(NH, P, D, device=DEVICE).requires_grad_(True)
    V = torch.randn(NH, P, D, device=DEVICE).requires_grad_(True)
    bias = (torch.randn(NH, P, P, device=DEVICE) * 0.1).requires_grad_(True)

    out = fused_attention_with_bias(Q, K, V, bias, None, scale, H)
    out.sum().backward()

    for name, t in [('Q', Q), ('K', K), ('V', V), ('bias', bias)]:
        assert t.grad is not None, f"missing gradient for {name}"
        assert t.grad.isfinite().all(), f"non-finite gradient for {name}"


# =====================================================================
# 3. PairEmbed patch parity (device-adaptive: CPU stub or GPU Triton)
# =====================================================================

@pytest.mark.parametrize("mode", ["train", "eval"])
def test_pair_embed_patch_parity(mode):
    """Patched PairEmbed.forward vs pristine weaver PairEmbed.forward.

    On CPU this validates the patch's sparse-style gather -> embed -> scatter
    plumbing against weaver's own path; on GPU it additionally validates the
    Triton pair-feature numerics. In train mode BatchNorm running statistics
    must also stay equivalent (both paths restrict BN statistics to valid
    pairs).
    """
    patched, pristine, stats = _patched_and_pristine(seed=0)
    assert stats["pair_embed_instances"] == 1

    training = mode == "train"
    pristine.train(training)
    patched.train(training)

    torch.manual_seed(123)
    B, P = 4, 24
    v = _make_physical_4vec(B, P, DEVICE)
    mask = _make_mask(B, P, DEVICE, dtype=torch.bool)

    with torch.no_grad():
        ref = pristine.pair_embed(v, uu=None, mask=mask)
        out = patched.pair_embed(v, uu=None, mask=mask)

    assert out.shape == ref.shape, f"Shape mismatch: {out.shape} vs {ref.shape}"
    max_diff = _max_abs_diff(out, ref)
    print(f"  pair_embed[{mode}] device={DEVICE}  max_abs={max_diff:.2e}")
    assert max_diff <= PARITY_TOL, (
        f"pair_embed patch diverges in {mode} mode: "
        f"max_abs={max_diff:.2e} > tol={PARITY_TOL:g}"
    )

    if training:
        # Valid-pairs-only BN semantics: the train-mode forward must have
        # updated running statistics identically on both models.
        ref_buffers = dict(pristine.pair_embed.named_buffers())
        for name, buf in patched.pair_embed.named_buffers():
            assert name in ref_buffers
            assert torch.allclose(
                buf.float(), ref_buffers[name].float(),
                atol=PARITY_TOL, rtol=0,
            ), f"BatchNorm buffer {name!r} diverged after train-mode forward"


# =====================================================================
# 4. Full-model patch parity (device-adaptive)
# =====================================================================

@pytest.mark.parametrize("mode", ["train", "eval"])
def test_model_patch_parity(mode):
    """optimize_part_model-patched weaver model vs pristine copy.

    Train mode is deterministic because the model config zeroes all dropout
    and disables the sequence trimmer; parity tolerance is 1e-5 max-abs on
    CPU and 5e-2 on GPU.
    """
    patched, pristine, stats = _patched_and_pristine(seed=1)
    assert stats["pair_embed_instances"] == 1
    assert stats["attention_instances"] == 2

    training = mode == "train"
    pristine.train(training)
    patched.train(training)

    torch.manual_seed(456)
    B, P = 4, 24
    x = torch.randn(B, INPUT_DIM, P, device=DEVICE)
    v = _make_physical_4vec(B, P, DEVICE)
    mask = _make_mask(B, P, DEVICE)

    with torch.no_grad():
        ref = pristine(x, v=v, mask=mask)
        out = patched(x, v=v, mask=mask)

    assert out.shape == (B, NUM_CLASSES)
    assert out.shape == ref.shape
    assert out.isfinite().all()
    max_diff = _max_abs_diff(out, ref)
    print(f"  model[{mode}] device={DEVICE}  max_abs={max_diff:.2e}")
    assert max_diff <= PARITY_TOL, (
        f"full-model patch diverges in {mode} mode: "
        f"max_abs={max_diff:.2e} > tol={PARITY_TOL:g}"
    )


# =====================================================================
# 5. Property 4: patched model behaviorally equivalent to pristine
# =====================================================================

def _weaver_model_from_config(config):
    """Build a plain weaver ParticleTransformer from a `weaver_configs` dict.

    The shared ``weaver_batches`` strategy (variants/tests/strategies.py)
    emits configs shaped for ``build_variant_part``; two keys are consumed by
    the variants factory only and are not weaver kwargs: ``dropout`` and
    ``expansion_factor``. This adapter drops them and instead zeroes all
    dropout via ``block_params`` (train-mode determinism, Property 4 requires
    dropout disabled) and disables the sequence trimmer (``trim=False``) so
    train-mode forwards are directly comparable — the same choices as
    `_small_part_model`, generalized over the drawn config.
    """
    from weaver.nn.model.ParticleTransformer import ParticleTransformer

    cfg = dict(config)
    cfg.pop("dropout", None)
    cfg.pop("expansion_factor", None)
    return ParticleTransformer(
        **cfg,
        pair_input_dim=4,
        pair_extra_dim=0,
        block_params=dict(dropout=0.0, attn_dropout=0.0, activation_dropout=0.0),
        trim=False,
        for_inference=False,
        use_amp=False,
    ).to(DEVICE)


# Feature: self-contained-kernels-migration, Property 4: Patched weaver model is behaviorally equivalent to the pristine model
@settings(max_examples=100, deadline=None)
@given(
    batch=weaver_batches(),
    seed=st.integers(min_value=0, max_value=2**32 - 1),
)
def test_patched_weaver_model_behaviorally_equivalent_to_pristine(batch, seed):
    """For any small weaver config and valid ``(x, v, mask)`` batch, the model
    patched by ``optimize_part_model`` produces outputs equal to the pristine
    weaver model within device-appropriate tolerance (``PARITY_TOL``) in both
    eval and train modes (dropout disabled), and ``unpatch_part_model``
    restores the exact pristine behavior.

    Check order (deliberate, to keep the unpatch check bit-exact):

    1. eval parity — no state mutation, so parameters/buffers stay identical;
    2. unpatch restoration — the restored model must reproduce the pristine
       eval output exactly (``torch.equal``): unpatching removes the
       per-instance forward overrides, so identical class code runs on
       bit-identical state;
    3. re-patch, then train parity — train forwards mutate BatchNorm running
       statistics, so they run last. A batch whose pair grid has exactly one
       valid pair (B=1, one valid particle) makes weaver's own BatchNorm
       raise in train mode; behavioral equivalence there means the patched
       model raises the same error.

    **Validates: Requirements 2.1, 8.2**
    """
    from ml4sci_26.part_kernels import optimize_part_model, unpatch_part_model

    x, v, mask = batch.x.to(DEVICE), batch.v.to(DEVICE), batch.mask.to(DEVICE)

    # Deterministic model construction per example: all parameter init flows
    # through the Hypothesis-drawn seed.
    torch.manual_seed(seed)
    pristine = _weaver_model_from_config(batch.config)
    patched = copy.deepcopy(pristine)
    patched, stats = optimize_part_model(patched)

    assert stats["pair_embed_instances"] == 1, stats
    assert stats["attention_instances"] == batch.config["num_layers"], stats

    ctx = (
        f"config={batch.config}, lengths={batch.lengths}, "
        f"B={x.shape[0]}, P={x.shape[-1]}, seed={seed}"
    )

    # --- 1. eval parity ---------------------------------------------------
    pristine.eval()
    patched.eval()
    with torch.no_grad():
        ref_eval = pristine(x, v=v, mask=mask)
        out_eval = patched(x, v=v, mask=mask)

    assert out_eval.shape == ref_eval.shape
    assert out_eval.isfinite().all(), f"non-finite patched eval output ({ctx})"
    max_diff = _max_abs_diff(out_eval, ref_eval)
    assert max_diff <= PARITY_TOL, (
        f"patched model diverges in eval mode: max_abs={max_diff:.2e} > "
        f"tol={PARITY_TOL:g} ({ctx})"
    )

    # --- 2. unpatch restoration (bit-exact: no buffers mutated yet) --------
    restored = unpatch_part_model(patched)
    restored.eval()
    with torch.no_grad():
        out_restored = restored(x, v=v, mask=mask)
    assert torch.equal(out_restored, ref_eval), (
        f"unpatch_part_model did not restore pristine behavior: "
        f"max_abs={_max_abs_diff(out_restored, ref_eval):.2e} ({ctx})"
    )

    # --- 3. train parity (dropout disabled; BN stats mutate, so last) ------
    patched, _ = optimize_part_model(restored)
    pristine.train()
    patched.train()
    try:
        with torch.no_grad():
            ref_train = pristine(x, v=v, mask=mask)
    except ValueError:
        # Degenerate pair grid (a single valid pair): weaver's own BatchNorm
        # rejects it in train mode. Equivalence = the patched model does too.
        with pytest.raises(ValueError):
            with torch.no_grad():
                patched(x, v=v, mask=mask)
        return

    with torch.no_grad():
        out_train = patched(x, v=v, mask=mask)

    assert out_train.shape == ref_train.shape
    assert out_train.isfinite().all(), f"non-finite patched train output ({ctx})"
    max_diff = _max_abs_diff(out_train, ref_train)
    assert max_diff <= PARITY_TOL, (
        f"patched model diverges in train mode: max_abs={max_diff:.2e} > "
        f"tol={PARITY_TOL:g} ({ctx})"
    )


# =====================================================================
# Script mode
# =====================================================================

def main():
    gpu = has_triton()
    print(f"device={DEVICE}  triton={gpu}  parity_tol={PARITY_TOL:g}")
    results: dict[str, str] = {}

    def run(name, fn, *args, skip=False):
        if skip:
            print(f"  {name:40s} SKIP (no triton/CUDA)")
            results[name] = "SKIP"
            return
        try:
            fn(*args)
        except AssertionError as exc:
            print(f"  {name:40s} FAIL ({exc})")
            results[name] = "FAIL"
        else:
            print(f"  {name:40s} PASS")
            results[name] = "PASS"

    print("=== Kernel-level parity (GPU-only) ===")
    run("pairwise_kernel", test_pairwise_kernel, skip=not gpu)
    run("attention_kernel", test_attention_kernel, skip=not gpu)
    run("attention_backward", test_attention_backward, skip=not gpu)

    print("=== Patch parity (device-adaptive) ===")
    for mode in ("train", "eval"):
        run(f"pair_embed_patch_parity[{mode}]", test_pair_embed_patch_parity, mode)
        run(f"model_patch_parity[{mode}]", test_model_patch_parity, mode)

    print("=== Property 4 (Hypothesis, 100 examples) ===")
    # The @given wrapper makes the property runnable with no arguments.
    run("property4_patch_behavioral_equivalence",
        test_patched_weaver_model_behaviorally_equivalent_to_pristine)

    failed = [name for name, status in results.items() if status == "FAIL"]
    print("=" * 60)
    for name, status in results.items():
        print(f"  {name:40s} {status}")
    print("=" * 60)
    print("Overall:", "ALL PASS" if not failed else f"FAILURES: {', '.join(failed)}")
    return 1 if failed else 0


if __name__ == '__main__':
    sys.exit(main())
