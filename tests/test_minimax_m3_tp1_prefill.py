# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2026 ATOM Project Contributors

"""MiniMax-M3.1 TP1 prefill path (atom/model_ops/minimax_m3/tp1_prefill.py):
where it runs and the ops' fake shapes on any host; the MoE layer's hand-off
where the model imports; on a gfx950 GPU with aiter's m3_tp1_prefill ops, each
op against the generic path it replaces, byte for byte."""

import logging
from types import SimpleNamespace

import pytest
import torch
from conftest import gpu_stack_import

from atom.model_ops.minimax_m3 import tp1_prefill as T

_FP8 = torch.float8_e4m3fn


@pytest.fixture
def ops(monkeypatch):
    monkeypatch.setattr(T, "kernels_available", lambda: True)
    monkeypatch.setenv("ATOM_M3_TP1_PREFILL", "1")


_PRODUCER = {"kv_connector": "mooncake", "kv_role": "kv_producer"}


def _deployment(**over):
    cfg = {
        "tensor_parallel_size": 1,
        "kv_transfer_config": _PRODUCER,
        "dp": 1,
        "enable_dp_attention": False,
        "prefill_context_parallel_size": 1,
        "sequence_parallel_size": 1,
    }
    cfg.update(over)
    return SimpleNamespace(
        parallel_config=SimpleNamespace(data_parallel_size=cfg.pop("dp")), **cfg
    )


@pytest.mark.parametrize(
    "over, want",
    [
        ({}, True),
        ({"tensor_parallel_size": 2}, False),
        ({"kv_transfer_config": {}}, False),  # a server that decodes
        ({"kv_transfer_config": dict(_PRODUCER, kv_role="kv_consumer")}, False),
        ({"kv_transfer_config": {"kv_connector": "lmcache"}}, False),  # offload
    ],
)
def test_path_needs_the_switch_a_tp1_producer_and_the_ops(ops, monkeypatch, over, want):
    assert T.path_enabled(_deployment(**over)) is want
    monkeypatch.setenv("ATOM_M3_TP1_PREFILL", "0")
    assert not T.path_enabled(_deployment(**over))
    monkeypatch.setenv("ATOM_M3_TP1_PREFILL", "1")
    monkeypatch.setattr(T, "kernels_available", lambda: False)
    assert not T.path_enabled(_deployment(**over))


@pytest.mark.parametrize(
    "over, hidden, indexer_cp, plugin, want",
    [
        ({}, 6144, False, False, True),
        ({}, 4096, False, False, False),  # the norms are built for hidden 6144
        ({"dp": 2}, 6144, False, False, False),
        ({"enable_dp_attention": True}, 6144, False, False, False),
        ({"prefill_context_parallel_size": 2}, 6144, False, False, False),
        ({"sequence_parallel_size": 2}, 6144, False, False, False),
        ({}, 6144, True, False, False),
        ({}, 6144, False, True, False),  # plugin mode patches the attention
    ],
)
def test_model_runs_it_only_on_the_measured_deployment(
    ops, monkeypatch, over, hidden, indexer_cp, plugin, want
):
    import atom.distributed.indexer_cp as icp
    from atom.plugin import prepare

    monkeypatch.setattr(icp, "indexer_cp_enabled", lambda: indexer_cp)
    monkeypatch.setattr(prepare, "is_plugin_mode", lambda: plugin)
    assert T.model_enabled(_deployment(**over), hidden) is want
    assert not T.model_enabled(_deployment(tensor_parallel_size=2, **over), hidden)


@pytest.mark.parametrize(
    "k, need",
    [
        (None, "aiter's m3_tp1_prefill ops"),  # an aiter without the ops
        (SimpleNamespace(is_supported=lambda: False), "a gfx950 GPU"),
    ],
)
def test_model_logs_what_the_path_needs(monkeypatch, caplog, k, need):
    monkeypatch.setenv("ATOM_M3_TP1_PREFILL", "1")
    monkeypatch.setattr(T, "_K", k)
    with caplog.at_level(logging.INFO, logger="atom"):
        assert not T.model_enabled(_deployment(), 6144)
    assert f"MiniMax-M3 TP1 prefill path off: it needs {need}" in caplog.text


@pytest.mark.parametrize(
    "attrs,ok",
    [
        ({"native_a8_group_rows": 1, "params_dtype": _FP8}, True),
        ({"native_a8_group_rows": 1, "params_dtype": _FP8, "preshuffle": "0"}, False),
        ({"native_a8_group_rows": 32, "params_dtype": _FP8}, False),
        ({"native_a8_group_rows": None, "params_dtype": _FP8}, False),
        ({"native_a8_group_rows": 1, "params_dtype": torch.bfloat16}, False),
        ({"native_a8_group_rows": 1, "params_dtype": _FP8, "runs_bf16": True}, False),
        ({"native_a8_group_rows": 1, "params_dtype": _FP8, "bias": object()}, False),
    ],
)
def test_linear_supported_takes_native_mxfp8_1x32_only(monkeypatch, attrs, ok):
    """A native MXFP8 1x32 linear stored preshuffled (the GEMM's B layout)."""
    monkeypatch.setenv("ATOM_GROUP32_WEIGHT_PRESHUFFLE", attrs.pop("preshuffle", "1"))
    attrs.setdefault("bias", None)
    assert T.linear_supported(SimpleNamespace(**attrs)) is ok


def test_fake_impls_give_the_real_shapes():
    m, h = 300, 6144
    x = torch.empty(m, h, dtype=torch.bfloat16, device="meta")
    res_out, xq, xs = T._norm_mxfp8_fake(x, x, x, None, 1e-6, 9856)
    assert res_out.shape == (m, h) and xq.dtype == _FP8
    assert xs.shape == (512, h // 32) and xs.dtype == torch.uint8
    outs = T._norm_moe_input_fake(x, x, None, 1e-6, 6144)
    assert [(tuple(t.shape), t.dtype) for t in outs] == [
        ((m, h), torch.bfloat16),
        ((m, h), torch.bfloat16),
        ((m, h), _FP8),
        ((512, h // 32), torch.uint8),
    ]
    assert T._materialize_fake(x, x).shape == (m, h)
    w = torch.empty(9856, h, dtype=_FP8, device="meta")
    assert T._linear_q_fake(xq, xs, w, None, None).shape == (m, 9856)
    assert T._linear_bf16_fake(x, w, None, None).shape == (m, 9856)
    assert T._mlp_fake(xq, xs, "x").shape == (m, h)
    assert [t.shape for t in T._moe_ffn_fake(x, xq, xs, None, "x")] == [(m, h)] * 2


def _w4_index(rows: int, k: int) -> torch.Tensor:
    """Byte position of scale (row, group) in shuffle_scale_w4, from the 8-wave
    GEMM reader's formula (aiter ScalePreshuffledS2R)."""
    r = torch.arange(rows).view(-1, 1)
    g = torch.arange(k // 32).view(1, -1)
    rb, rr, kb, gg = r // 32, r % 32, g // 8, g % 8
    return (
        (rb * (k // 256) + kb) * 256
        + ((gg % 4) * 16 + rr % 16) * 4
        + (gg // 4) * 2
        + rr // 16
    )


@pytest.mark.parametrize("rows,k", [(1, 3072), (37, 6144), (256, 8192), (1000, 12288)])
def test_shuffle_scale_w4_follows_the_reader_formula(rows, k):
    scale = torch.randint(0, 255, (rows, k // 32), dtype=torch.uint8)
    w4 = T.shuffle_scale_w4(scale.view(torch.float8_e8m0fnu)).view(torch.uint8)
    assert w4.numel() == T.scale_rows(rows) * k // 32
    assert torch.equal(w4[_w4_index(rows, k).view(-1)], scale.view(-1))
    # rows past `rows` (up to the 256 padding) read the neutral exponent 127
    pad = torch.ones(w4.numel(), dtype=torch.bool)
    pad[_w4_index(rows, k).view(-1)] = False
    assert bool((w4[pad] == 127).all())


@pytest.mark.parametrize("m,want", [(1, 256), (256, 256), (257, 512), (8192, 8192)])
def test_scale_rows_pad_to_256(m, want):
    assert T.scale_rows(m) == want


def test_psc_layout_needs_an_unsplit_8wave_row_1024_rows_and_k_256(monkeypatch):
    rows = {(8192, 9856, 6144): (1, 4), (1000, 9856, 6144): (2, 8)}
    monkeypatch.setattr(T, "_psc_row", lambda m, n, k: rows.get((m, n, k)))
    assert T.psc_layout(8192, 9856, 6144)
    assert not T.psc_layout(8192, 6144, 6144)  # no unsplit 8-wave row
    assert not T.psc_layout(1000, 9856, 6144)  # below 1024 rows
    rows[(8192, 9856, 6000)] = (1, 4)
    assert not T.psc_layout(8192, 9856, 6000)  # K not a multiple of 256


@pytest.fixture(scope="module")
def model():
    return gpu_stack_import("atom.models.minimax_m3.model")


def test_a_moe_layer_defers_its_fold_only_to_a_fused_next_layer(model):
    """dense, MoE, MoE (generic), MoE, MoE (the stage's last layer)."""
    fused = [True, True, False, True, True]
    layers = []
    for moe, ok in zip([False, True, True, True, True], fused):
        layer = SimpleNamespace(is_moe_layer=moe, tp1=False, tp1_defer=False)
        layer.enable_tp1 = lambda prefix, layer=layer, ok=ok: setattr(layer, "tp1", ok)
        layers.append(layer)
    stage = SimpleNamespace(start_layer=0, end_layer=5, layers=layers)
    model.MiniMaxM3Model._enable_tp1(stage, "model.layers")
    assert [layer.tp1 for layer in layers] == fused
    assert [layer.tp1_defer for layer in layers] == [False, False, False, True, False]


def _fake_ops(monkeypatch, seen):
    def rec(name, out):
        def op(*args):
            seen.append((name, args))
            return out(*args)

        monkeypatch.setattr(T, name, op)

    rec("linear_bf16", lambda x, w, *_: x.new_full((x.shape[0], w.shape[0]), 1.0))
    rec("linear_q", lambda xq, xs, w, *_: xq.new_full((xq.shape[0], w.shape[0]), 1.0))
    rec("norm_mxfp8", lambda x, s, res, w, eps, n: (res + 1, res, res))
    rec("norm_moe_input", lambda x, res, w, eps, n: (x, res + 2, x, x))
    rec("moe_ffn", lambda x, q8, s8, logits, name: (x + 3, x + 4))
    rec("mlp", lambda q8, s8, name: q8 + 5)
    rec("materialize", lambda r, s: r * 2 + s)


def _moe_layer(defer: bool):
    """A MoE layer of hidden 8 and 4 q columns, as `_forward_tp1` reads it."""

    def linear(n, k):
        weight = SimpleNamespace(shape=(n, k))
        return SimpleNamespace(weight=weight, weight_scale=None, weight_scale_w4=None)

    norm = SimpleNamespace(weight=None, variance_epsilon=1e-6)
    return SimpleNamespace(
        self_attn=SimpleNamespace(
            q_size=4,
            qkv_proj=linear(12, 8),
            o_proj=linear(8, 4),
            attend=lambda positions, qkv: qkv[:, :4],
        ),
        input_layernorm=norm,
        post_attention_layernorm=norm,
        is_moe_layer=True,
        block_sparse_moe=SimpleNamespace(
            gate=lambda x: x,
            shared_experts=SimpleNamespace(gate_up_proj=linear(16, 8)),
        ),
        tp1_defer=defer,
        tp1_ffn_name="model.layers.1.tp1_ffn",
    )


@pytest.mark.parametrize("defer", [True, False])
def test_a_moe_layer_hands_on_its_pair_or_folds_it(model, monkeypatch, defer):
    seen = []
    _fake_ops(monkeypatch, seen)
    layer = _moe_layer(defer)
    pair = (torch.zeros(3, 8), torch.ones(3, 8))
    out, residual = model.MiniMaxM3DecoderLayer._forward_tp1(
        layer, None, pair, torch.zeros(3, 8), False
    )
    # the previous MoE layer's pair reaches the input norm unfolded
    name, args = seen[0]
    assert name == "norm_mxfp8" and args[0] is pair[0] and args[1] is pair[1]
    assert [n for n, _ in seen] == [
        "norm_mxfp8",
        "linear_q",
        "linear_bf16",
        "norm_moe_input",
        "moe_ffn",
    ] + ([] if defer else ["materialize"])
    # o_proj gives 1s, the MoE (1 + 3, 1 + 4): the pair, or 4 * 2 + 5 folded
    if defer:
        assert [float(t[0, 0]) for t in out] == [4.0, 5.0]
    else:
        assert torch.equal(out, torch.full((3, 8), 13.0))
    assert torch.equal(residual, torch.full((3, 8), 3.0))


def test_dspark_taps_fold_a_deferred_pair(model, monkeypatch):
    monkeypatch.setattr(T, "materialize", lambda r, s: r * 2 + s)
    monkeypatch.setattr(model, "get_tensor_model_parallel_world_size", lambda: 1)
    r, s, res = torch.ones(2, 4), torch.full((2, 4), 3.0), torch.full((2, 4), 10.0)
    tap = model.MiniMaxM3DecoderLayer.aux_hidden_state(((r, s), res))
    assert torch.equal(tap, torch.full((2, 4), 15.0))


# ---------------------------------------------------------------------------
# GPU: each op against the generic path it replaces
# ---------------------------------------------------------------------------


def _gfx950_with_ops() -> bool:
    if not torch.cuda.is_available():
        return False
    from aiter.jit.utils.chip_info import get_gfx

    return get_gfx() == "gfx950" and T.kernels_available()


gpu = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="needs a gfx950 GPU and aiter's ops"
)
_MS = (1, 1000, 3001, 8192)


@pytest.fixture
def dev():
    if not _gfx950_with_ops():
        pytest.skip("needs a gfx950 GPU and aiter's m3_tp1_prefill ops")
    return "cuda"


def _acts(m, h, seed, dev, scale=1.0):
    g = torch.Generator(device=dev).manual_seed(seed)
    x = torch.randn(m, h, device=dev, generator=g) * scale
    x[:, ::97] *= 400.0  # massive-activation columns
    return x.to(torch.bfloat16)


def _bytes_equal(a, b) -> bool:
    return torch.equal(
        a.contiguous().view(torch.uint8), b.contiguous().view(torch.uint8)
    )


def _generic_norm(h, residual):
    from atom.model_ops.layernorm import GemmaRMSNorm

    norm = GemmaRMSNorm(h.shape[1], eps=1e-6)
    norm = norm.to(device=h.device, dtype=torch.bfloat16)
    with torch.no_grad():
        norm.weight.copy_(torch.randn(h.shape[1], device=h.device) * 0.1)
    return norm, norm(h, residual)


def _row_scales(xs, m, k, n_out) -> torch.Tensor:
    """Rows < m of e8m0 bytes in the layout `psc_layout` names for an [m, k]
    input of an n_out-row weight, row-major (the 256-row padding is unset)."""
    if not T.psc_layout(m, n_out, k):
        return xs[:m]
    return xs.view(-1)[_w4_index(m, k).to(xs.device)]


def _scales_equal(xs, ref, m, n_out) -> bool:
    """``xs`` as `_row_scales` reads it == `quantize_fp8`'s row-major ``ref``."""
    return _bytes_equal(_row_scales(xs, m, ref.shape[1] * 32, n_out), ref)


@gpu
@pytest.mark.parametrize("m", _MS + (8191, 16384))
def test_norms_equal_add_rmsnorm_then_quantize_fp8(dev, m):
    from atom.model_ops.blockscale import quantize_fp8

    h = 6144
    x, s, res = (_acts(m, h, seed, dev) for seed in (0, 1, 2))
    for shared in (None, s):
        hin = x if shared is None else x * 2 + shared
        norm, (normed, res_out) = _generic_norm(hin, res)
        q, qs = quantize_fp8(normed)
        for n_out in (9856, 24576):  # qkv, a dense gate_up
            got = T.norm_mxfp8(x, shared, res, norm.weight, 1e-6, n_out)
            assert _bytes_equal(got[0], res_out) and _bytes_equal(got[1], q)
            assert _scales_equal(got[2], qs, m, n_out)
    normed, res_out, q8, s8 = T.norm_moe_input(x, res, norm.weight, 1e-6, 6144)
    norm_ref, res_ref = norm(x, res)
    q, qs = quantize_fp8(norm_ref)
    assert _bytes_equal(normed, norm_ref) and _bytes_equal(res_out, res_ref)
    assert _bytes_equal(q8, q) and _scales_equal(s8, qs, m, 6144)
    assert _bytes_equal(T.materialize(x, s), x * 2 + s)


def _mxfp8_weight(n, k, seed, dev):
    """A preshuffled MXFP8 1x32 weight as LinearBase holds it: (weight, the
    GEMM's weight scale, its shuffle_scale_w4 copy)."""
    from aiter.ops.shuffle import shuffle_mxfp8_bpreshuffle_scale, shuffle_weight

    g = torch.Generator(device=dev).manual_seed(seed)
    w = (torch.randn(n, k, device=dev, generator=g) * 0.5).to(_FP8)
    codes = torch.randint(118, 125, (n, k // 32), device=dev, generator=g)
    ws = codes.to(torch.uint8).view(torch.float8_e8m0fnu)
    return (
        shuffle_weight(w, layout=(16, 16)),
        shuffle_mxfp8_bpreshuffle_scale(ws),
        T.shuffle_scale_w4(ws),
    )


# qkv, o_proj, shared gate_up, shared down, dense gate_up, dense down
_LINEARS = [(9856, 6144), (6144, 8192), (6144, 6144), (6144, 3072), (24576, 6144)]


@gpu
@pytest.mark.parametrize("n,k", _LINEARS + [(6144, 12288)])
def test_linears_equal_native_quant_linear(dev, n, k):
    """Row-major scales and, where aiter's row is the 8-wave kernel, preshuffled
    ones (ragged M included): the generic linear's bytes."""
    from atom.model_ops.blockscale import native_quant_linear

    w, ws, ws_w4 = _mxfp8_weight(n, k, n + k, dev)
    for m in (1000, 1024, 3001, 8191, 8192, 16384):
        x = _acts(m, k, m, dev)
        want = native_quant_linear(
            x, w, ws, weight_group_rows=1, weight_preshuffled=True
        )
        assert _bytes_equal(T.linear_bf16(x, w, ws, ws_w4), want), m
        xq, xs = T.quant_mxfp8(x, n)
        assert _bytes_equal(T.linear_q(xq, xs, w, ws, ws_w4), want), m


@gpu
@pytest.mark.parametrize("inter", [3072, 12288])
@pytest.mark.parametrize("m", (37, 1000, 3001, 8192))
def test_mlp_equals_the_generic_mlp(dev, inter, m):
    from atom.model_ops.blockscale import native_quant_linear, quantize_fp8
    from atom.model_ops.swiglu_oai import swiglu_oai_split

    h = 6144

    def lin(n, k, seed):
        weight, weight_scale, weight_scale_w4 = _mxfp8_weight(n, k, seed, dev)
        return SimpleNamespace(
            weight=weight, weight_scale=weight_scale, weight_scale_w4=weight_scale_w4
        )

    gu, dn = lin(2 * inter, h, inter), lin(h, inter, inter + 1)
    mlp = SimpleNamespace(
        gate_up_proj=gu,
        down_proj=dn,
        swiglu_alpha=1.702,
        swiglu_beta=1.0,
        swiglu_limit=7.0,
    )
    x = _acts(m, h, 3, dev)

    def linear(t, lin):
        return native_quant_linear(
            t,
            lin.weight,
            lin.weight_scale,
            weight_group_rows=1,
            weight_preshuffled=True,
        )

    act = swiglu_oai_split(linear(x, gu), alpha=1.702, beta=1.0, limit=7.0)
    want = linear(act, dn)
    q8, s8 = T.quant_mxfp8(x, 2 * inter)
    assert _bytes_equal(T.mlp_forward(mlp, q8, s8), want)
    if m >= 1024:  # the SwiGLU's w4 scales, as the down GEMM reads them
        from atom.model_ops.minimax_m3.tp1_swiglu import swiglu_oai_mxfp8_w4

        y, s = swiglu_oai_mxfp8_w4(linear(x, gu), 1.702, 1.0, 7.0)
        q, qs = quantize_fp8(act)
        assert _bytes_equal(y, q)
        rows = _w4_index(m, inter).view(-1).to(dev)
        assert _bytes_equal(s.view(-1)[rows], qs.view(torch.uint8).view(-1))


@gpu
def test_the_ops_compile_to_the_eager_bytes(dev):
    """The fake impls under torch.compile (dynamic rows): the traced chain gives
    the eager bytes."""
    h = 6144
    norm, _ = _generic_norm(_acts(8, h, 0, dev), _acts(8, h, 1, dev))
    w, ws, ws_w4 = _mxfp8_weight(9856, h, 7, dev)

    def chain(x, shared, res):
        res, xq, xs = T.norm_mxfp8(x, shared, res, norm.weight, 1e-6, 9856)
        y = T.linear_q(xq, xs, w, ws, ws_w4)[:, :h]
        normed, res, q8, s8 = T.norm_moe_input(y, res, norm.weight, 1e-6, 6144)
        return T.materialize(normed, res), q8, s8

    compiled = torch.compile(chain, dynamic=True, fullgraph=True)
    for m in (1000, 3001, 8192):
        args = [_acts(m, h, seed, dev) for seed in (4, 5, 6)]
        (out_c, q_c, s_c), (out_e, q_e, s_e) = compiled(*args), chain(*args)
        assert _bytes_equal(out_c, out_e) and _bytes_equal(q_c, q_e)
        assert _bytes_equal(*(_row_scales(s, m, h, 6144) for s in (s_c, s_e)))
