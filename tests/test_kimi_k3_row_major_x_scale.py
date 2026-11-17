# SPDX-License-Identifier: MIT
"""Kimi-K3 on gfx1250: E8M0 per-1x128 activation scales handed to the GEMMs row-major.

Needs an aiter with ``gemm_a8w8_blockscale_bpreshuffle_prefers_row_major_x_scale``
(ROCm/aiter#8079, or #8088 on the Kimi-K3 branch)."""

import importlib
from types import SimpleNamespace

import pytest
import torch

aiter = pytest.importorskip("aiter")
QuantType, dtypes = aiter.QuantType, aiter.dtypes

if not torch.cuda.is_available():
    pytest.skip("GPU GEMM", allow_module_level=True)
from aiter.jit.utils.chip_info import get_gfx_runtime

if not get_gfx_runtime().startswith("gfx1250"):
    pytest.skip("the row-major x_scale GEMM is gfx1250 only", allow_module_level=True)

from aiter.ops.gemm_op_a8w8 import (
    gemm_a8w8_blockscale_bpreshuffle_prefers_row_major_x_scale as prefers_row_major,
)

linear_mod = importlib.import_module("atom.model_ops.linear")
layernorm_mod = importlib.import_module("atom.model_ops.layernorm")
layout_mod = importlib.import_module("atom.model_ops.kimi_k3.x_scale_layout")
LayerQuantConfig = importlib.import_module("atom.quant_spec").LayerQuantConfig

K = 7168
G = K // 128
ROWS = (1, 58, 256, 545, 1024)


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
    """A Linear that online quantisation turns into per_1x128 at load, so at
    construction it is still unquantized (as in the model before loading)."""
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
    return layer


def _load(*layers):
    for layer in layers:
        layer.process_weights_after_loading()
        layer.cuda()


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
    Linear sharing the norm's x_scale layout plus a baseline Linear. The layout
    is shared before loading, as the decoder layer does in __init__."""
    torch.manual_seed(0)
    srcs = [torch.randn(n, K, dtype=torch.bfloat16) * 0.02 for n in ns]
    row_lin = [_layer(monkeypatch, n, s) for n, s in zip(ns, srcs)]
    col_lin = [_layer(monkeypatch, n, s) for n, s in zip(ns, srcs)]
    row_norm, col_norm = _norm(monkeypatch), _norm(monkeypatch)
    row_norm.weight.data.copy_(col_norm.weight.data)
    layout_mod.share_x_scale_layout(row_norm, row_lin)
    _load(*row_lin, *col_lin)
    return row_norm, col_norm, row_lin, col_lin


def test_the_layout_is_row_major_only_where_every_gemm_prefers_it(monkeypatch):
    row_norm, _, row_lin, _ = _grouped(monkeypatch, [2176, 1536])
    key = row_norm.x_scale_row_major_key
    assert key and all(c.x_scale_row_major_key == key for c in row_lin)
    for m in ROWS:
        want = prefers_row_major(m, 2176, K) and prefers_row_major(m, 1536, K)
        assert linear_mod.x_scale_row_major(key, m) == want
    # the tuned tables pick row-major at decode with concurrency 128
    assert linear_mod.x_scale_row_major(key, 545)
    assert not linear_mod.x_scale_row_major(0, 545)


@pytest.mark.parametrize("rows", ROWS)
def test_norm_to_gemm_bytes_match_the_column_major_chain(monkeypatch, rows):
    row_norm, col_norm, row_lin, col_lin = _grouped(monkeypatch, [2176, 1536])
    x = torch.randn(rows, K, device="cuda", dtype=torch.bfloat16)
    yc, sc = col_norm(x)
    yr, sr = row_norm(x)
    assert torch.equal(yc.view(torch.uint8), yr.view(torch.uint8))
    if linear_mod.x_scale_row_major(row_norm.x_scale_row_major_key, rows):
        assert torch.equal(sr.view(torch.uint8), _col_bytes_as_rows(sc, rows))
    else:
        assert torch.equal(sr.view(torch.uint8), sc.view(torch.uint8))
    for r, c in zip(row_lin, col_lin):
        assert torch.equal(r(yr, sr).view(torch.int16), c(yc, sc).view(torch.int16))
        # a Linear quantizing for itself stays column-major
        assert torch.equal(r(x).view(torch.int16), c(x).view(torch.int16))


def test_a_norm_and_its_consumers_switch_together_or_not_at_all(monkeypatch):
    source = torch.randn(2176, K, dtype=torch.bfloat16) * 0.02
    e8m0 = _layer(monkeypatch, 2176, source)
    fp32 = _layer(monkeypatch, 2176, source, e8m0=False)
    norm = _norm(monkeypatch)
    layout_mod.share_x_scale_layout(norm, [e8m0, fp32])
    _load(e8m0, fp32)
    key = norm.x_scale_row_major_key
    assert key and e8m0.x_scale_row_major_key == fp32.x_scale_row_major_key == key
    assert not any(linear_mod.x_scale_row_major(key, m) for m in ROWS)

    other = _norm(monkeypatch)
    layout_mod.share_x_scale_layout(other, [e8m0, torch.nn.Identity()])
    assert not other.x_scale_row_major_key


def test_column_major_everywhere_when_aiter_prefers_it(monkeypatch):
    monkeypatch.setattr(layout_mod, "_prefers_row_major", lambda m, n, k: False)
    row_norm, col_norm, row_lin, col_lin = _grouped(monkeypatch, [2176])
    key = row_norm.x_scale_row_major_key
    assert not any(linear_mod.x_scale_row_major(key, m) for m in ROWS)
    x = torch.randn(545, K, device="cuda", dtype=torch.bfloat16)
    (yc, sc), (yr, sr) = col_norm(x), row_norm(x)
    assert torch.equal(sr.view(torch.uint8), sc.view(torch.uint8))
    assert torch.equal(
        row_lin[0](yr, sr).view(torch.int16), col_lin[0](yc, sc).view(torch.int16)
    )
