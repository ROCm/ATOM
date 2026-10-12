# SPDX-License-Identifier: MIT
"""Qwen3.8 Flash-Next decode HC parity, fallback and graph replay."""

from types import SimpleNamespace

import pytest
import torch

pytest.importorskip("aiter")
pytest.importorskip("triton")

from atom.model_ops import linear
from atom.model_ops.qwen4_exp.hyperconnection import Qwen4ExpHyperConnection
from atom.model_ops.qwen4_exp.ops import hc_flydsl


def test_served_bf16_default_keeps_hc_dispatch_eligible(monkeypatch):
    if not torch.cuda.is_available():
        pytest.skip("needs a ROCm GPU")
    if torch.cuda.get_device_properties(0).gcnArchName.split(":")[0] != "gfx950":
        pytest.skip("the decode specialization targets gfx950")
    monkeypatch.setattr(
        linear, "get_tp_group", lambda: SimpleNamespace(rank_in_group=0, world_size=1)
    )
    previous_dtype = torch.get_default_dtype()
    try:
        torch.set_default_dtype(torch.bfloat16)
        module = Qwen4ExpHyperConnection(2560, 4, 320).cuda()
    finally:
        torch.set_default_dtype(previous_dtype)
    with torch.no_grad():
        for parameter in module.parameters():
            parameter.normal_(0, 0.02)
    module.process_weights_after_loading()
    monkeypatch.setenv("ATOM_QWEN38_MONO_HC", "1")
    x = torch.randn(4, 10240, device="cuda", dtype=torch.bfloat16)
    assert module.hc_norm.weight.dtype == torch.float32
    assert module._mono_supported(x, 8)


@pytest.fixture
def hc(monkeypatch):
    if not torch.cuda.is_available():
        pytest.skip("needs a ROCm GPU")
    if torch.cuda.get_device_properties(0).gcnArchName.split(":")[0] != "gfx950":
        pytest.skip("the decode specialization targets gfx950")
    monkeypatch.setattr(
        linear, "get_tp_group", lambda: SimpleNamespace(rank_in_group=0, world_size=1)
    )
    torch.manual_seed(7)
    module = Qwen4ExpHyperConnection(2560, 4, 320).cuda()
    with torch.no_grad():
        for parameter in module.parameters():
            parameter.normal_(0, 0.02)
    return module


def _assert_deferred_streams_close(actual, expected, rows, has_pending):
    if rows in (12, 16, 18, 24) and has_pending:
        # The deferred combine has only a few BF16 ties among the stream values.
        # Bound both their count and their bit distance instead of relaxing
        # the tolerance for the whole residual stream.
        differing = actual != expected
        assert int(differing.sum()) <= 8
        actual_bits = actual.view(torch.int16).to(torch.int32)
        expected_bits = expected.view(torch.int16).to(torch.int32)
        assert torch.all((actual_bits[differing] - expected_bits[differing]).abs() == 1)
    else:
        torch.testing.assert_close(actual, expected, atol=1e-5, rtol=0)


@pytest.mark.parametrize("rows", [1, 2, 4, 8, 16])
def test_fused_hc_matches_native_and_large_batch_falls_back(hc, monkeypatch, rows):
    x = (
        torch.randn(rows, 10240, device="cuda", dtype=torch.bfloat16) * 0.5
    ).contiguous()
    block = (
        torch.randn(rows, 2560, device="cuda", dtype=torch.bfloat16) * 0.5
    ).contiguous()

    monkeypatch.setenv("ATOM_QWEN38_MONO_HC", "0")
    assert not hc._mono_supported(x, 4)
    reference_mixed, reference_residuals = hc.mix(x)
    reference = hc.combine(block, reference_residuals)
    monkeypatch.setenv("ATOM_QWEN38_MONO_HC", "1")
    assert hc._mono_supported(x, 4) == (rows <= 4)
    assert hc._mono_supported(x, 16)
    mixed, residuals = hc.mix(x)
    actual = hc.combine(block, residuals)

    torch.testing.assert_close(mixed, reference_mixed, atol=0.003, rtol=0.02)
    torch.testing.assert_close(residuals[1], reference_residuals[1], atol=0, rtol=0)
    torch.testing.assert_close(actual, reference, atol=0.004, rtol=0.02)


@pytest.mark.parametrize("rows", [4, 8, 16])
def test_graph_replay_reads_changed_input(hc, monkeypatch, rows):
    monkeypatch.setenv("ATOM_QWEN38_MONO_HC", "1")
    x = torch.zeros(rows, 10240, device="cuda", dtype=torch.bfloat16)
    block = torch.zeros(rows, 2560, device="cuda", dtype=torch.bfloat16)
    hc.combine(block, hc.mix(x)[1])
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        result = hc.combine(block, hc.mix(x)[1])

    for seed in (13, 29):
        torch.manual_seed(seed)
        x.copy_(torch.randn_like(x) * 0.5)
        block.copy_(torch.randn_like(block) * 0.5)
        graph.replay()
        torch.cuda.synchronize()
        monkeypatch.setenv("ATOM_QWEN38_MONO_HC", "0")
        expected = hc.combine(block, hc.mix(x)[1])
        monkeypatch.setenv("ATOM_QWEN38_MONO_HC", "1")
        torch.testing.assert_close(result, expected, atol=0.004, rtol=0.02)


@pytest.mark.parametrize("rows", [32, 64, 128])
def test_large_batch_combine_matches_native(hc, monkeypatch, rows):
    x = (
        torch.randn(rows, 10240, device="cuda", dtype=torch.bfloat16) * 0.5
    ).contiguous()
    block = (
        torch.randn(rows, 2560, device="cuda", dtype=torch.bfloat16) * 0.5
    ).contiguous()
    normed = hc.hc_norm(x)
    monkeypatch.setenv("ATOM_QWEN38_MONO_HC", "0")
    reference = hc.combine(block, (x, normed))
    monkeypatch.setenv("ATOM_QWEN38_MONO_HC", "1")
    assert not hc._mono_supported(x, 16)
    assert hc._mono_supported(x, 128)
    actual = hc.combine(block, (x, normed))
    torch.testing.assert_close(actual, reference, atol=0.004, rtol=0.02)


def test_large_batch_combine_graph_replay(hc, monkeypatch):
    rows = 64
    monkeypatch.setenv("ATOM_QWEN38_MONO_HC", "1")
    x = torch.zeros(rows, 10240, device="cuda", dtype=torch.bfloat16)
    normed = torch.zeros_like(x)
    block = torch.zeros(rows, 2560, device="cuda", dtype=torch.bfloat16)
    hc.combine(block, (x, normed))
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        actual = hc.combine(block, (x, normed))
    for seed in (11, 17):
        torch.manual_seed(seed)
        x.copy_(torch.randn_like(x) * 0.5)
        normed.copy_(torch.randn_like(normed) * 0.5)
        block.copy_(torch.randn_like(block) * 0.5)
        graph.replay()
        torch.cuda.synchronize()
        monkeypatch.setenv("ATOM_QWEN38_MONO_HC", "0")
        reference = hc.combine(block, (x, normed))
        monkeypatch.setenv("ATOM_QWEN38_MONO_HC", "1")
        torch.testing.assert_close(actual, reference, atol=0.004, rtol=0.02)


@pytest.mark.parametrize("rows", [3, 4, 6, 8, 9, 12, 16, 18, 24])
@pytest.mark.parametrize("has_pending", [False, True])
def test_served_mix_fused_matches_native(hc, monkeypatch, rows, has_pending):
    hc.process_weights_after_loading()
    x = (
        torch.randn(rows, 10240, device="cuda", dtype=torch.bfloat16) * 0.5
    ).contiguous()
    pending = None
    if has_pending:
        pending = (
            (
                torch.randn(rows, 2560, device="cuda", dtype=torch.bfloat16) * 0.5
            ).contiguous(),
            (
                torch.randn(rows, 4, device="cuda", dtype=torch.bfloat16) * 0.5
            ).contiguous(),
        )
    monkeypatch.setenv("ATOM_QWEN38_MONO_HC", "0")
    expected, (expected_streams, expected_raw) = hc.mix_fused(x, pending)
    monkeypatch.setenv("ATOM_QWEN38_MONO_HC", "1")
    actual, (actual_streams, actual_raw) = hc.mix_fused(x, pending)
    torch.testing.assert_close(actual, expected, atol=0.004, rtol=0.02)
    _assert_deferred_streams_close(actual_streams, expected_streams, rows, has_pending)
    torch.testing.assert_close(actual_raw, expected_raw, atol=0.004, rtol=0.02)


@pytest.mark.parametrize("rows", [3, 8, 12, 16, 24])
def test_served_mix_fused_graph_replay(hc, monkeypatch, rows):
    hc.process_weights_after_loading()
    x = torch.zeros(rows, 10240, device="cuda", dtype=torch.bfloat16)
    block = torch.zeros(rows, 2560, device="cuda", dtype=torch.bfloat16)
    raw = torch.zeros(rows, 4, device="cuda", dtype=torch.bfloat16)
    monkeypatch.setenv("ATOM_QWEN38_MONO_HC", "1")
    hc.mix_fused(x, (block, raw))
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        actual, (streams, inject) = hc.mix_fused(x, (block, raw))
    for seed in (11, 17):
        torch.manual_seed(seed)
        x.copy_(torch.randn_like(x) * 0.5)
        block.copy_(torch.randn_like(block) * 0.5)
        raw.copy_(torch.randn_like(raw) * 0.5)
        graph.replay()
        torch.cuda.synchronize()
        monkeypatch.setenv("ATOM_QWEN38_MONO_HC", "0")
        expected, (expected_streams, expected_inject) = hc.mix_fused(x, (block, raw))
        monkeypatch.setenv("ATOM_QWEN38_MONO_HC", "1")
        torch.testing.assert_close(actual, expected, atol=0.004, rtol=0.02)
        _assert_deferred_streams_close(streams, expected_streams, rows, True)
        torch.testing.assert_close(inject, expected_inject, atol=0.004, rtol=0.02)


@pytest.mark.parametrize(
    "rows, uses_flydsl", [(3, True), (12, True), (15, False), (24, True), (32, False)]
)
def test_mtp_verification_shape_dispatch(hc, monkeypatch, rows, uses_flydsl):
    hc.process_weights_after_loading()
    x = torch.randn(rows, 10240, device="cuda", dtype=torch.bfloat16)
    pending = (
        torch.randn(rows, 2560, device="cuda", dtype=torch.bfloat16),
        torch.randn(rows, 4, device="cuda", dtype=torch.bfloat16),
    )
    calls = []
    original = hc_flydsl.mix_fused

    def trace(*args):
        calls.append(True)
        return original(*args)

    monkeypatch.setattr(hc_flydsl, "mix_fused", trace)
    monkeypatch.setenv("ATOM_QWEN38_MONO_HC", "1")
    hc.mix_fused(x, pending)
    assert bool(calls) == uses_flydsl


@pytest.mark.parametrize("rows", [4, 16])
def test_final_mixer_without_injection(hc, monkeypatch, rows):
    hc.block_inject_weight = None
    hc.process_weights_after_loading()
    x = (
        torch.randn(rows, 10240, device="cuda", dtype=torch.bfloat16) * 0.5
    ).contiguous()
    pending = (
        (
            torch.randn(rows, 2560, device="cuda", dtype=torch.bfloat16) * 0.5
        ).contiguous(),
        (torch.randn(rows, 4, device="cuda", dtype=torch.bfloat16) * 0.5).contiguous(),
    )
    monkeypatch.setenv("ATOM_QWEN38_MONO_HC", "0")
    expected, (expected_streams, expected_raw) = hc.mix_fused(x, pending)
    monkeypatch.setenv("ATOM_QWEN38_MONO_HC", "1")
    actual, (actual_streams, actual_raw) = hc.mix_fused(x, pending)
    assert expected_raw is None and actual_raw is None
    torch.testing.assert_close(actual, expected, atol=0.004, rtol=0.02)
    _assert_deferred_streams_close(actual_streams, expected_streams, rows, True)
