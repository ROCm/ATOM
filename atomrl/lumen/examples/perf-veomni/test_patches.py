"""Essential checks for veomni_patches/ (one GPU for all but the last).

HIP_VISIBLE_DEVICES=0 CUDA_VISIBLE_DEVICES=0 python -m pytest -q test_patches.py
"""

import os
import sys
import types

import pytest
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

_CUDA = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


def _snr(ref, got):
    ref = ref.float()
    return (
        10 * torch.log10(ref.pow(2).mean() / (got.float() - ref).pow(2).mean()).item()
    )


# --- aiter_conv -------------------------------------------------------------------


@_CUDA
def test_conv_uses_flydsl_for_bf16_and_torch_otherwise():
    from veomni_patches import aiter_conv

    if aiter_conv.load() is None:
        pytest.skip("aiter flydsl_conv_implicit unavailable (overlay_aiter.py --apply)")
    torch.manual_seed(0)
    x = torch.randn(1, 96, 4, 24, 24, device="cuda")
    w = torch.randn(96, 96, 3, 3, 3, device="cuda") * 0.05
    before = aiter_conv.conv_calls()
    lookups = sum(aiter_conv.tuned_lookup_stats()["calls"].values())
    fp32 = aiter_conv.conv(x, w, None, 1, 1, 1)
    bf16 = aiter_conv.conv(x.bfloat16(), w.bfloat16(), None, 1, 1, 1)
    after = aiter_conv.conv_calls()
    assert (
        after["torch"] == before["torch"] + 1
        and after["flydsl"] == before["flydsl"] + 1
    )
    assert sum(aiter_conv.tuned_lookup_stats()["calls"].values()) == lookups + 1
    assert torch.equal(fp32, F.conv3d(x, w, None, 1, 1, 1))
    assert _snr(fp32, bf16) > 30


# --- sdpa --------------------------------------------------------------------------


@_CUDA
def test_triton_forward_routes_dit_attention_only():
    pytest.importorskip("aiter.ops.triton.mha")
    from veomni_patches import sdpa

    sdpa.install_triton_forward(min_seq=256)
    try:
        torch.manual_seed(0)
        q, k, v = (
            torch.randn(
                1, 4, 512, 128, device="cuda", dtype=torch.bfloat16, requires_grad=True
            )
            for _ in range(3)
        )
        g = torch.randn_like(q)
        routed = sdpa.stats()["routed"]
        out = F.scaled_dot_product_attention(q, k, v)
        assert sdpa.stats()["routed"] == routed + 1
        grads = torch.autograd.grad(out, [q, k, v], g)
        qf, kf, vf = (t.detach().float().requires_grad_(True) for t in (q, k, v))
        ref = F.scaled_dot_product_attention(qf, kf, vf)
        ref_grads = torch.autograd.grad(ref, [qf, kf, vf], g.float())
        for a, b in zip((ref, *ref_grads), (out, *grads)):
            assert _snr(a, b) > 30
        stock = sdpa.stats()["stock"]
        with torch.no_grad():
            got = F.scaled_dot_product_attention(q, k, v, is_causal=True)
            want = sdpa._orig_sdpa(q, k, v, is_causal=True)
        assert sdpa.stats()["stock"] == stock + 1 and torch.equal(got, want)
    finally:
        sdpa.uninstall_triton_forward()


# --- rmsnorm -----------------------------------------------------------------------


@_CUDA
def test_rmsnorm_fuses_long_weighted_inputs_only():
    norm_mod = pytest.importorskip("diffusers.models.normalization")
    from veomni_patches import rmsnorm

    norm = norm_mod.RMSNorm(128, eps=1e-6).cuda().to(torch.bfloat16)
    long_x = torch.randn(1, 2048, 24, 128, device="cuda", dtype=torch.bfloat16)
    short_x = torch.randn(1, 13, 24, 128, device="cuda", dtype=torch.bfloat16)
    want_short = norm(short_x)
    ref_long = rmsnorm._rmsnorm_weighted(long_x.float(), norm.weight.float(), 1e-6)
    assert rmsnorm.patch_diffusers_rmsnorm(min_seq=1024)
    try:
        before = rmsnorm.stats()
        got_long = norm(long_x)
        assert torch.equal(norm(short_x), want_short)
        after = rmsnorm.stats()
        assert (
            after["fused"] == before["fused"] + 1
            and after["fallback"] == before["fallback"] + 1
        )
        assert _snr(ref_long, got_long) > 50
    finally:
        rmsnorm.unpatch_diffusers_rmsnorm()


# --- local_adamw -------------------------------------------------------------------


def _train_adamw(steps, patched):
    from veomni_patches.local_adamw import (
        install_local_shard_adamw,
        uninstall_local_shard_adamw,
    )

    torch.manual_seed(0)
    ps = [
        torch.nn.Parameter(torch.randn(s, device="cuda"))
        for s in ((64, 32), (32,), (7,))
    ]
    ps.append(
        torch.nn.Parameter(torch.randn(8, 8, device="cuda", dtype=torch.bfloat16))
    )
    opt = torch.optim.AdamW(
        [{"params": ps[:2]}, {"params": ps[2:], "weight_decay": 0.0}],
        lr=1e-3,
        fused=True,
    )
    if patched:
        install_local_shard_adamw()
    try:
        gen = torch.Generator(device="cuda").manual_seed(1)
        for _ in range(steps):
            for p in ps:
                p.grad = torch.randn(
                    p.shape, device="cuda", dtype=p.dtype, generator=gen
                )
            opt.step()
    finally:
        uninstall_local_shard_adamw()
    return [p.detach().clone() for p in ps], opt


@_CUDA
def test_local_adamw_is_bit_identical():
    want, _ = _train_adamw(4, patched=False)
    got, opt = _train_adamw(4, patched=True)
    assert "_lumen_local_cache" in opt.__dict__
    assert all(torch.equal(a, b) for a, b in zip(want, got))


# --- qwen_offline (CPU) ------------------------------------------------------------


def test_qwen_offline_fix_reads_vae_config(monkeypatch):
    from veomni_patches import qwen_offline

    cond = types.ModuleType(qwen_offline._MODULE)

    class QwenImageConditionModel:
        def __init__(self, vae):
            self.vae = vae
            self.config = types.SimpleNamespace(
                base_model_path="/models/qwen", vae_subfolder="vae"
            )

        def _normalize_latents(self, latents):
            return ("original", latents)

    QwenImageConditionModel.__module__ = qwen_offline._MODULE
    cond.QwenImageConditionModel = QwenImageConditionModel
    monkeypatch.setitem(sys.modules, qwen_offline._MODULE, cond)

    class AutoencoderKLQwenImage:
        @staticmethod
        def load_config(path, subfolder=None):
            return {"latents_mean": [1.0, 2.0], "latents_std": [2.0, 4.0], "z_dim": 2}

    monkeypatch.setitem(
        sys.modules,
        "diffusers",
        types.SimpleNamespace(AutoencoderKLQwenImage=AutoencoderKLQwenImage),
    )
    assert qwen_offline.install_qwen_offline_fix()
    out = QwenImageConditionModel(vae=None)._normalize_latents(
        torch.full((1, 2, 1, 1, 1), 5.0)
    )
    assert torch.equal(out.flatten(), torch.tensor([2.0, 0.75]))
    assert QwenImageConditionModel(vae=object())._normalize_latents("x") == (
        "original",
        "x",
    )
