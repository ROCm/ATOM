# SPDX-License-Identifier: MIT
"""Qwen3.8 Flash-Next decode HC parity, fallback and graph replay."""

from types import SimpleNamespace

import pytest
import torch

pytest.importorskip("aiter")
pytest.importorskip("triton")

from atom.model_ops import linear
from atom.model_ops.qwen4_exp.hyperconnection import Qwen4ExpHyperConnection


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
