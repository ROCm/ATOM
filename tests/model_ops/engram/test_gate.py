# SPDX-License-Identifier: MIT
"""Engram gates: full/TP layouts, FP64 oracle, masks and graph replay."""

from types import SimpleNamespace

import pytest
import torch

if not torch.cuda.is_available():
    # Before the import, which reaches Triton. As a `pytestmark` this ran
    # after it, so a CPU runner failed collection instead of skipping.
    pytest.skip("requires GPU", allow_module_level=True)

from atom.model_ops.engram.device.gate import (
    _engram_apply_gates_kernel,
    engram_post_wkv,
    engram_post_wkv_reference,
    engram_post_wkv_tp,
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


@pytest.mark.parametrize("tp", [2, 4])
@pytest.mark.parametrize("dim", [38, 5120])
def test_apply_gates_reuses_binary_across_token_counts_and_graph(tp, dim):
    local_hc, value_start = 4 // tp, dim // tp
    # Include M=1, aligned/unaligned M and the observed prefill JIT triggers.
    counts = [1, 6, 7, 16, 129, 192]
    if dim == 38:
        counts += [768, 12237, 16260]
    compiled = None
    for m in counts:
        h = torch.randn(m, 4, dim, device="cuda", dtype=torch.bfloat16)
        kv = torch.randn(m, (local_hc + 1) * dim, device="cuda", dtype=h.dtype)
        gates = torch.rand(tp, m, local_hc, device="cuda")
        output = torch.empty_like(h)

        def launch(h=h, kv=kv, gates=gates, output=output, m=m):
            return _engram_apply_gates_kernel[((h.numel() + 1023) // 1024,)](
                h,
                kv,
                gates,
                output,
                m,
                HC=4,
                DIM=dim,
                LOCAL_HC=local_hc,
                BLOCK=1024,
                VALUE_START=value_start,
                num_warps=4,
                enable_fp_fusion=False,
            )

        def check(h=h, kv=kv, gates=gates, output=output, m=m):
            gate = gates.permute(1, 0, 2).reshape(m, 4)
            value = torch.roll(kv[:, local_hc * dim :], value_start, -1)
            expected = (h.float() + gate[..., None] * value.float()[:, None]).to(
                h.dtype
            )
            assert torch.equal(output, expected)

        kernel = launch()
        if compiled is None:
            compiled = kernel
        assert kernel is compiled
        check()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        assert launch() is compiled
    for _ in range(2):
        h.add_(0.25)
        gates.mul_(0.75)
        graph.replay()
        check()


@pytest.mark.parametrize("tp", [2, 4])
@pytest.mark.parametrize("shape", [(3, 4, 5120), (2, 3, 4, 64)])
@pytest.mark.parametrize("fused", [False, True])
@pytest.mark.parametrize("masked", [False, True])
@pytest.mark.parametrize("rotated_value", [False, True])
def test_tp_gate_and_complete_residual(
    tp, shape, fused, masked, rotated_value, monkeypatch
):
    monkeypatch.setenv("ATOM_ENGRAM_FUSED_GATE", str(int(fused)))
    torch.manual_seed(713)
    hc, dim = shape[-2:]
    hidden = torch.randn(shape, device="cuda", dtype=torch.bfloat16)
    kv = torch.randn(*shape[:-2], (hc + 1) * dim, device="cuda", dtype=torch.bfloat16)
    weight = torch.randn(hc, dim, device="cuda")
    mask = torch.rand(shape[:-2], device="cuda") > 0.3 if masked else None
    # Independent FP64 gate oracle; round to the communicated FP32 dtype.
    h = hidden.double()
    k = kv[..., : hc * dim].reshape_as(hidden).double()
    dot = ((h * weight.double()) * k).sum(-1)
    dot *= torch.rsqrt(h.square().mean(-1) + 1e-20)
    dot *= torch.rsqrt(k.square().mean(-1) + 1e-20) * dim**-0.5
    gate = torch.sigmoid(torch.copysign(dot.abs().clamp_min(1e-6).sqrt(), dot)).float()
    if mask is not None:
        gate.masked_fill_(~mask.unsqueeze(-1), 0)
    c = hc // tp
    gathered = torch.cat(
        [gate[..., r * c : (r + 1) * c].reshape(-1, c) for r in range(tp)]
    )
    expected = (
        hidden.float() + gate.unsqueeze(-1) * kv[..., hc * dim :].float().unsqueeze(-2)
    ).to(hidden.dtype)
    for rank in range(tp):
        value_start = rank * (dim // tp) if rotated_value else 0
        part = torch.cat(
            (
                kv[..., rank * c * dim : (rank + 1) * c * dim],
                torch.roll(kv[..., hc * dim :], -value_start, -1),
            ),
            dim=-1,
        )

        def all_gather(local, *, dim, use_custom, rank=rank):
            assert local.dtype == torch.float32 and dim == 0
            assert not use_custom  # This test supplies a mock collective.
            torch.testing.assert_close(
                local,
                gate[..., rank * c : (rank + 1) * c].reshape(-1, c),
                rtol=2e-5,
                atol=2e-6,
            )
            return gathered

        group = SimpleNamespace(
            world_size=tp, rank_in_group=rank, all_gather=all_gather
        )
        actual = engram_post_wkv_tp(
            hidden, part, weight, group, mask, value_start=value_start
        )
        assert torch.equal(actual, expected)
        if mask is not None:
            assert torch.equal(actual[~mask], hidden[~mask])
    # The original full path remains the numerical reference at TP1.
    torch.testing.assert_close(
        expected,
        engram_post_wkv_reference(hidden, kv, weight, mask),
        rtol=1e-2,
        atol=1e-2,
    )


@pytest.mark.parametrize("tp", [2, 4])
@pytest.mark.parametrize("shape", [(1, 4, 5120), (15, 4, 5120), (2, 3, 4, 128)])
@pytest.mark.parametrize("fused", [False, True])
@pytest.mark.parametrize("masked", [False, True])
def test_gathered_key_value_shards_need_no_reorder(
    tp, shape, fused, masked, monkeypatch
):
    monkeypatch.setenv("ATOM_ENGRAM_FUSED_GATE", str(int(fused)))
    torch.manual_seed(914)
    hc, dim = shape[-2:]
    h = torch.randn(shape, device="cuda", dtype=torch.bfloat16)
    kv = torch.randn(*shape[:-2], (hc + 1) * dim, device="cuda", dtype=torch.bfloat16)
    w = torch.randn(hc, dim, device="cuda")
    mask = torch.rand(shape[:-2], device="cuda") > 0.4 if masked else None
    c, v = hc // tp * dim, dim // tp
    gathered = torch.cat(
        [
            torch.cat(
                (
                    kv[..., r * c : (r + 1) * c],
                    kv[..., hc * dim + r * v : hc * dim + (r + 1) * v],
                ),
                -1,
            )
            for r in range(tp)
        ],
        -1,
    )
    expected = engram_post_wkv(h, kv, w, mask)
    actual = engram_post_wkv(h, gathered, w, mask, kv_tp_size=tp)
    assert torch.equal(actual, expected)
    if mask is not None:
        assert torch.equal(actual[~mask], h[~mask])


def test_empty_and_invalid_local_projection():
    group = SimpleNamespace(world_size=4, rank_in_group=0)
    h = torch.empty(0, 4, 64, device="cuda", dtype=torch.bfloat16)
    weight = torch.ones(4, 64, device="cuda")
    assert (
        engram_post_wkv_tp(h, torch.empty(0, 128, device="cuda"), weight, group).shape
        == h.shape
    )
    with pytest.raises(ValueError, match="local KV"):
        engram_post_wkv_tp(h, torch.empty(0, 256, device="cuda"), weight, group)
    with pytest.raises(ValueError, match="mask"):
        engram_post_wkv_tp(
            h,
            torch.empty(0, 128, device="cuda"),
            weight,
            group,
            torch.ones(1, device="cuda", dtype=torch.bool),
        )


def test_tp8_rejected_by_sharded_gate_entry_points():
    group = SimpleNamespace(world_size=8, rank_in_group=0)
    h = torch.empty(1, 8, 64, device="cuda")
    weight = torch.ones(8, 64, device="cuda")
    with pytest.raises(ValueError, match="TP2 or TP4"):
        engram_post_wkv_tp(h, torch.empty(1, 128, device="cuda"), weight, group)
    with pytest.raises(ValueError, match="TP2 or TP4"):
        engram_post_wkv(h, torch.empty(1, 9 * 64, device="cuda"), weight, kv_tp_size=8)
