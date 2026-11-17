# SPDX-License-Identifier: MIT
"""Kimi-K3 on gfx1250: E8M0 per-1x128 activation scales handed to the GEMMs row-major."""

import importlib
from types import SimpleNamespace

import pytest
import torch

aiter = pytest.importorskip("aiter")
QuantType, dtypes = aiter.QuantType, aiter.dtypes

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU GEMM")

linear_mod = importlib.import_module("atom.model_ops.linear")
layernorm_mod = importlib.import_module("atom.model_ops.layernorm")
LayerQuantConfig = importlib.import_module("atom.quant_spec").LayerQuantConfig
kimi_k3 = importlib.import_module("atom.models.kimi_k3.model")

K = 7168
G = K // 128


def _row_major_supported():
    return linear_mod._aiter_takes_row_major_x_scale()


def _quant_config(e8m0):
    target = LayerQuantConfig(
        quant_type=QuantType.per_1x128, quant_dtype=dtypes.fp8, is_dynamic=True
    )
    return SimpleNamespace(
        online_quant=True,
        blockscale_e8m0_scale=e8m0,
        get_layer_quant_config=lambda *_a, use_online_quant=False, **_k: (
            target if use_online_quant else LayerQuantConfig()
        ),
    )


def _layer(monkeypatch, n, source, e8m0=True):
    monkeypatch.setenv("ATOM_ONLINE_QUANT_STREAMING", "0")
    monkeypatch.setenv("ATOM_FP8_BLOCKSCALE_WEIGHT_PRESHUFFLE", "1")
    monkeypatch.setattr(
        linear_mod,
        "get_tp_group",
        lambda: SimpleNamespace(rank_in_group=0, world_size=1),
    )
    monkeypatch.setattr(
        linear_mod,
        "get_current_atom_config",
        lambda: SimpleNamespace(torch_dtype=torch.bfloat16),
    )
    layer = linear_mod.ColumnParallelLinear(
        K, n, quant_config=_quant_config(e8m0), prefix="layers.3.in_proj"
    )
    layer.weight.data = source.clone()
    layer.process_weights_after_loading()
    return layer.cuda()


def _norm(monkeypatch):
    monkeypatch.setattr(
        layernorm_mod, "get_tensor_model_parallel_world_size", lambda: 1
    )
    target = LayerQuantConfig(
        quant_type=QuantType.per_1x128, quant_dtype=dtypes.fp8, is_dynamic=True
    )
    cfg = SimpleNamespace(
        online_quant=False,
        blockscale_e8m0_scale=True,
        get_layer_quant_config=lambda *_a, **_k: target,
    )
    norm = layernorm_mod.RMSNorm(
        K, eps=1e-6, fused_quant=True, quant_config=cfg, prefix="layers.3.in_proj"
    )
    norm.weight.data = 1 + 0.1 * torch.randn(K)
    return norm.cuda()


def _col_bytes_as_rows(scale, rows):
    """A column-major (K//128, M) scale (viewed as (M, K//128)) as [M, K//128]."""
    return scale.view(torch.uint8).reshape(-1).reshape(G, rows).t()


def _grouped(monkeypatch, ns):
    """A norm and a column-major baseline norm with the same weight, and per N a
    Linear in the norm's row-major x_scale group plus a baseline Linear."""
    torch.manual_seed(0)
    srcs = [torch.randn(n, K, dtype=torch.bfloat16) * 0.02 for n in ns]
    row_lin = [_layer(monkeypatch, n, s) for n, s in zip(ns, srcs)]
    col_lin = [_layer(monkeypatch, n, s) for n, s in zip(ns, srcs)]
    row_norm, col_norm = _norm(monkeypatch), _norm(monkeypatch)
    row_norm.weight.data.copy_(col_norm.weight.data)
    assert kimi_k3._row_major_x_scale(row_norm, row_lin)
    return row_norm, col_norm, row_lin, col_lin


def test_the_group_takes_row_major_only_where_every_gemm_does(monkeypatch):
    if not _row_major_supported():
        pytest.skip("needs gfx1250 and an aiter with the row-major x_scale query")
    row_norm, _, row_lin, _ = _grouped(monkeypatch, [2176, 1536])
    key = row_norm._row_major_key
    assert key and all(c.x_scale_row_major_key == key for c in row_lin)
    query = linear_mod._aiter_row_major_x_scale_query()
    for m in (1, 58, 256, 545, 1024):
        want = query(m, 2176, K) and query(m, 1536, K)
        assert linear_mod.row_major_x_scale_for(key, m) == want
    assert not linear_mod.row_major_x_scale_for(0, 545)


@pytest.mark.parametrize("rows", [1, 58, 256, 545, 1024])
def test_norm_to_gemm_bytes_match_the_column_major_chain(monkeypatch, rows):
    if not _row_major_supported():
        pytest.skip("needs gfx1250 and an aiter with the row-major x_scale query")
    row_norm, col_norm, row_lin, col_lin = _grouped(monkeypatch, [2176, 1536])
    x = torch.randn(rows, K, device="cuda", dtype=torch.bfloat16)
    yc, sc = col_norm(x)
    yr, sr = row_norm(x)
    assert torch.equal(yc.view(torch.uint8), yr.view(torch.uint8))
    if linear_mod.row_major_x_scale_for(row_norm._row_major_key, rows):
        assert torch.equal(sr.view(torch.uint8), _col_bytes_as_rows(sc, rows))
    else:
        assert torch.equal(sr.view(torch.uint8), sc.view(torch.uint8))
    for r, c in zip(row_lin, col_lin):
        assert torch.equal(r(yr, sr).view(torch.int16), c(yc, sc).view(torch.int16))
        # a Linear quantizing for itself stays column-major
        assert torch.equal(r(x).view(torch.int16), c(x).view(torch.int16))


def test_a_norm_and_its_consumers_switch_together_or_not_at_all(monkeypatch):
    if not _row_major_supported():
        pytest.skip("needs gfx1250 and an aiter with the row-major x_scale query")
    source = torch.randn(2176, K, dtype=torch.bfloat16) * 0.02
    e8m0 = _layer(monkeypatch, 2176, source)
    fp32 = _layer(monkeypatch, 2176, source, e8m0=False)
    norm = _norm(monkeypatch)

    assert not kimi_k3._row_major_x_scale(norm, [e8m0, fp32])
    assert not e8m0.x_scale_row_major_key and not norm._row_major_key
    assert not kimi_k3._row_major_x_scale(norm, [None])

    assert kimi_k3._row_major_x_scale(norm, [e8m0])
    assert e8m0.x_scale_row_major_key == norm._row_major_key != 0
    assert norm._aiter_transpose_scale
