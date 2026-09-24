# SPDX-License-Identifier: MIT
"""Fused Engram gate: FP64 oracle, masks, fallback and graph replay."""

import pytest
import torch

if not torch.cuda.is_available():
    # Before the import, which reaches Triton. As a `pytestmark` this ran
    # after it, so a CPU runner failed collection instead of skipping.
    pytest.skip("requires GPU", allow_module_level=True)

from atom.model_ops.engram.device.gate import (
    engram_post_wkv,
    engram_post_wkv_reference,
)


def oracle(hidden, kv, weight, mask, eps):
    hc, dim = hidden.shape[-2:]
    h = hidden.double()
    k = kv[..., : hc * dim].reshape_as(hidden).double()
    v = kv[..., hc * dim :].double()
    score = ((h * weight.double()) * k).sum(-1)
    score *= torch.rsqrt(h.square().mean(-1) + eps) * torch.rsqrt(
        k.square().mean(-1) + eps
    )
    score *= dim**-0.5
    gate = torch.sigmoid(torch.copysign(score.abs().clamp_min(1e-6).sqrt(), score))
    if mask is not None:
        gate = gate.masked_fill(~mask.unsqueeze(-1), 0)
    return h + gate.unsqueeze(-1) * v.unsqueeze(-2)


def inputs(shape, dtype, seed=5, scale=1):
    torch.manual_seed(seed)
    hc, dim = shape[-2:]
    hidden = (torch.randn(shape, device="cuda") * scale).to(dtype)
    kv = (torch.randn(*shape[:-2], (hc + 1) * dim, device="cuda") * scale).to(dtype)
    weight = torch.randn(hc, dim, device="cuda", dtype=torch.float32)
    mask = torch.rand(shape[:-2], device="cuda") > 0.25
    return hidden, kv, weight, mask


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16, torch.float32])
@pytest.mark.parametrize("shape", [(6, 4, 5120), (96, 4, 5120), (2, 3, 2, 37)])
def test_fused_against_fp64_and_original(dtype, shape, monkeypatch):
    monkeypatch.setenv("ATOM_ENGRAM_FUSED_GATE", "1")
    h, kv, w, mask = inputs(shape, dtype)
    actual = engram_post_wkv(h, kv, w, mask)
    original = engram_post_wkv_reference(h, kv, w, mask)
    exact = oracle(h, kv, w, mask, 1e-20)
    # Final dtype rounding + an FP32 arithmetic budget, scaled by operands
    # rather than just the output (which can cancel almost to zero).
    hc, dim = h.shape[-2:]
    v = kv[..., hc * dim :].double().unsqueeze(-2)
    budget = (
        torch.finfo(dtype).eps * exact.abs()
        + 4e-6 * (h.double().abs() + v.abs())
        + 1e-7
    )
    assert torch.all((actual.double() - exact).abs() <= budget)
    fused_rms = (actual.double() - exact).square().mean().sqrt()
    old_rms = (original.double() - exact).square().mean().sqrt()
    assert fused_rms <= old_rms * 1.02 + 1e-7
    assert torch.equal(actual[~mask], h[~mask])
    assert actual.shape == h.shape and actual.dtype == h.dtype


@pytest.mark.parametrize("scale", [0, 1e-10, 100])
def test_small_norms_zero_and_large_inputs(scale, monkeypatch):
    monkeypatch.setenv("ATOM_ENGRAM_FUSED_GATE", "1")
    h, kv, w, mask = inputs((6, 4, 5120), torch.float32, scale=scale)
    for active in (None, mask, torch.zeros_like(mask)):
        actual = engram_post_wkv(h, kv, w, active)
        exact = oracle(h, kv, w, active, 1e-20)
        torch.testing.assert_close(
            actual.double(), exact, rtol=2e-5, atol=max(scale * 3e-6, 1e-15)
        )


def test_gate_clamp_and_both_signs(monkeypatch):
    monkeypatch.setenv("ATOM_ENGRAM_FUSED_GATE", "1")
    h = torch.ones(6, 4, 5120, device="cuda")
    kv = torch.ones(6, 5 * 5120, device="cuda")
    for scale in (0, 1e-10, -1e-10, 1e-4, -1e-4, 1, -1):
        w = torch.full((4, 5120), scale, device="cuda")
        actual = engram_post_wkv(h, kv, w)
        exact = oracle(h, kv, w, None, 1e-20)
        torch.testing.assert_close(actual.double(), exact, rtol=2e-6, atol=1e-7)


def test_strided_and_disabled_paths_match_reference(monkeypatch):
    h, kv, w, mask = inputs((12, 4, 5120), torch.bfloat16)
    monkeypatch.setenv("ATOM_ENGRAM_FUSED_GATE", "1")
    assert torch.equal(
        engram_post_wkv(h[::2], kv[::2], w, mask[::2]),
        engram_post_wkv_reference(h[::2], kv[::2], w, mask[::2]),
    )
    monkeypatch.setenv("ATOM_ENGRAM_FUSED_GATE", "0")
    assert torch.equal(
        engram_post_wkv(h, kv, w, mask), engram_post_wkv_reference(h, kv, w, mask)
    )


def test_empty_and_bad_mask():
    h, kv, w, mask = inputs((0, 4, 5120), torch.bfloat16)
    assert engram_post_wkv(h, kv, w, mask).shape == h.shape
    with pytest.raises(ValueError, match="mask"):
        engram_post_wkv(h, kv, w, torch.ones(1, device="cuda", dtype=torch.bool))


def test_capture_replay_reads_changed_inputs(monkeypatch):
    monkeypatch.setenv("ATOM_ENGRAM_FUSED_GATE", "1")
    h, kv, w, mask = inputs((96, 4, 5120), torch.float32)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        engram_post_wkv(h, kv, w, mask)
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        actual = engram_post_wkv(h, kv, w, mask)
    for iteration in range(4):
        h.add_(0.25)
        kv.mul_(0.75)
        mask.logical_not_()
        graph.replay()
        expected = engram_post_wkv_reference(h, kv, w, mask)
        torch.testing.assert_close(actual, expected, rtol=2e-5, atol=2e-6)
