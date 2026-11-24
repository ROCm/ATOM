# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""`model_options.attn_out_quant_fuse` of the DeepSeek-V4.1 plugin model: a
decode step's MXFP8 attention is called through the entry that also writes
the inverse RoPE group quant of its output, wo_a's input pair, over fp32
twins of the model's BF16 RoPE tables. CPU only: AITER's ops and the cache
are stood in for; what is checked is which call a step takes, with which
tables and buffers, and that every other step takes the calls it took."""

from types import SimpleNamespace

import pytest
import torch
from aiter_stub import atom_without_aiter

attention = None
model = None
rotary = None
host = None
recipe = None


@pytest.fixture(scope="module", autouse=True)
def _modules():
    global attention, model, rotary, host, recipe
    with atom_without_aiter() as load:
        host = load("atom.plugin_host")
        recipe = load("atom.plugin_recipe")
        rotary = load("atom.model_ops.deepseek_v41.rotary")
        attention = load("atom.models.deepseek_v41_plugin.attention")
        model = load("atom.models.deepseek_v41_plugin.model")
        yield


@pytest.fixture
def option(monkeypatch):
    def set_option(on=True):
        plan = SimpleNamespace(
            model_options=recipe.ModelOptions(attn_out_quant_fuse=on)
        )
        monkeypatch.setattr(host, "_plan", plan)

    monkeypatch.setattr(host, "_plan", None)
    monkeypatch.setattr(host, "owns", lambda seam, rows=None: False)
    if hasattr(attention._attn_out_quant_fuse_supported, "state"):
        monkeypatch.delattr(attention._attn_out_quant_fuse_supported, "state")
    return set_option


def _rope(dtype=torch.bfloat16, positions=32, pairs=32, twins=True):
    cos = torch.rand(positions, pairs).to(dtype)
    sin = torch.rand(positions, pairs).to(dtype)
    rope = SimpleNamespace(cos_cache=cos, sin_cache=sin)
    if twins:
        rope.cos_fp32, rope.sin_fp32 = cos.float(), sin.float()
    return rope


# -- the fp32 twins of the RoPE tables -------------------------------------------


def test_bf16_tables_have_no_fp32_form_until_the_twins_are_built():
    rope = rotary.RotaryEmbedding(64, 48, base=10000.0)
    assert rope.cos_cache.dtype == torch.bfloat16
    assert rotary.fp32_tables(rope) is None
    assert rotary.fp32_tables(rope, widened=True) is None
    cos, sin = rope.build_fp32_tables()
    assert rope.build_fp32_tables()[0] is cos  # built once
    for twin, cache in ((cos, rope.cos_cache), (sin, rope.sin_cache)):
        assert twin.dtype == torch.float32 and twin.is_contiguous()
        assert twin.shape == (cache.shape[0], cache.shape[-1])
        # value for value: each BF16 entry widened, nothing recomputed
        assert torch.equal(twin, cache.float().view_as(twin))
    assert rotary.fp32_tables(rope) is None
    got = rotary.fp32_tables(rope, widened=True)
    assert got[0] is cos and got[1] is sin
    assert "cos_fp32" not in rope.state_dict()


def test_fp32_tables_of_their_own_come_back_as_views():
    rope = _rope(torch.float32, twins=False)
    rope.cos_cache = rope.cos_cache.view(32, 1, 1, 32)
    rope.sin_cache = rope.sin_cache.view(32, 1, 1, 32)
    cos, sin = rotary.fp32_tables(rope)
    assert cos.shape == (32, 32) and cos.data_ptr() == rope.cos_cache.data_ptr()
    assert sin.data_ptr() == rope.sin_cache.data_ptr()


@pytest.mark.parametrize(
    "change",
    [
        lambda r: setattr(r, "cos_cache", r.cos_cache[:, :16]),
        lambda r: setattr(r, "sin_cache", r.sin_cache.float()),
        lambda r: setattr(r, "cos_fp32", r.cos_fp32[:8]),
        lambda r: setattr(r, "sin_fp32", r.sin_fp32.double()),
        lambda r: setattr(r, "cos_fp32", None),
    ],
)
def test_tables_that_are_not_what_a_kernel_indexes_are_refused(change):
    rope = _rope()
    change(rope)
    assert rotary.fp32_tables(rope, widened=True) is None


def test_the_twins_are_built_with_the_model_only_where_a_layer_fuses():
    built = []
    ropes = [
        SimpleNamespace(
            build_fp32_tables=lambda: built.append(1)
            or (torch.zeros(4), torch.zeros(4))
        )
        for _ in range(2)
    ]

    def layers(*flags):
        return [
            SimpleNamespace(attn=SimpleNamespace(_fuse_attn_out_quant=f)) for f in flags
        ]

    assert not model.build_out_quant_rope_tables(layers(False, False), ropes)
    assert built == []
    assert model.build_out_quant_rope_tables(layers(False, True), ropes)
    assert built == [1, 1]


# -- the gate -----------------------------------------------------------------


def test_off_by_default(option):
    assert not recipe.ModelOptions().attn_out_quant_fuse
    assert not attention._attn_out_quant_fuse_supported(64, 512, 8)


def test_on_only_for_the_shape_and_arch_aiter_fuses(option, monkeypatch):
    option()
    from aiter.jit.utils import chip_info

    monkeypatch.setattr(chip_info, "get_gfx_runtime", lambda: "gfx950")
    assert attention._attn_out_quant_fuse_supported(64, 512, 8)
    assert not attention._attn_out_quant_fuse_supported(32, 512, 8)
    assert not attention._attn_out_quant_fuse_supported(64, 512, 4)
    monkeypatch.setattr(chip_info, "get_gfx_runtime", lambda: "gfx942")
    assert not attention._attn_out_quant_fuse_supported(64, 512, 8)
    monkeypatch.setattr(chip_info, "get_gfx_runtime", lambda: "gfx950")
    monkeypatch.setattr(attention, "mla_sparse_prefill_v41_inv_rope_quant", None)
    assert not attention._attn_out_quant_fuse_supported(64, 512, 8)


# -- which steps take the fused entry -------------------------------------------


def _layer(fuse=True):
    layer = SimpleNamespace(
        _fuse_attn_out_quant=fuse,
        heads=8,
        head_dim=64,
        groups=2,
        wo_a=SimpleNamespace(runs_bf16=False),
        _attend_out_seam=attention.attend_out_native,
        spec=SimpleNamespace(layer_id=5, kv_owner=5),
        attn_sink="sink",
        softmax_scale=0.125,
    )
    for name in ("_attn_out_quant_tail", "_attend_out_tail", "_quantize_tail"):
        method = getattr(attention.Attention, name)
        setattr(layer, name, method.__get__(layer))
    return layer


def _step_inputs(rows=3):
    query = torch.zeros(1, rows, 8, 64)
    cache = SimpleNamespace(
        kv_fp8=True,
        packed=False,
        rope_positions=lambda step: torch.arange(rows)[None],
    )
    step = SimpleNamespace(decode=True, requests=rows)
    prepared = (False, ("q_fp8", "q_scale"), None, None)
    return query, cache, step, prepared


def test_a_decode_step_of_the_mxfp8_cache_gets_the_pair_and_the_twins(option):
    layer, rope = _layer(), _rope()
    query, cache, step, prepared = _step_inputs()
    tail = layer._attend_out_tail(query, cache, step, None, rope, prepared)
    positions, cos, sin, x_fp8, x_scale, cos32, sin32 = tail
    assert positions.tolist() == [0, 1, 2] and positions.dtype == torch.int64
    assert cos is rope.cos_cache and sin is rope.sin_cache
    assert cos32 is rope.cos_fp32 and sin32 is rope.sin_fp32
    assert x_fp8.shape == (3, 2, 8 * 64 // 2)
    assert x_scale.shape == (3, 2, 8 * 64 // 2 // 32)
    assert x_scale.dtype == torch.float8_e8m0fnu


def test_off_no_tail(option):
    layer, rope = _layer(fuse=False), _rope()
    query, cache, step, prepared = _step_inputs()
    assert layer._attend_out_tail(query, cache, step, None, rope, prepared) is None


@pytest.mark.parametrize(
    "change",
    [
        lambda q, c, s, p, r, layer: setattr(s, "decode", False),
        lambda q, c, s, p, r, layer: setattr(c, "kv_fp8", False),
        lambda q, c, s, p, r, layer: setattr(c, "packed", True),
        lambda q, c, s, p, r, layer: setattr(layer.wo_a, "runs_bf16", True),
        lambda q, c, s, p, r, layer: delattr(r, "cos_fp32"),
        lambda q, c, s, p, r, layer: setattr(
            c, "rope_positions", lambda step: torch.arange(3, dtype=torch.int32)[None]
        ),
    ],
)
def test_any_other_step_keeps_its_two_calls(option, change):
    layer, rope = _layer(), _rope()
    query, cache, step, prepared = _step_inputs()
    change(query, cache, step, prepared, rope, layer)
    assert layer._attend_out_tail(query, cache, step, None, rope, prepared) is None


def test_bounded_replay_live_preparation_and_a8w4_keep_their_calls(option):
    layer, rope = _layer(), _rope()
    query, cache, step, prepared = _step_inputs()
    tail = layer._attend_out_tail
    assert tail(query, cache, step, object(), rope, prepared) is None
    assert tail(query, cache, step, None, rope, None) is None
    a8w4 = (True, *prepared[1:])
    assert tail(query, cache, step, None, rope, a8w4) is None


@pytest.mark.parametrize(
    "seam", ["v41.attn.attend", "v41.attn.attend_out", "v41.attn.out_full"]
)
def test_an_owner_of_a_span_it_covers_keeps_its_calls(option, monkeypatch, seam):
    layer, rope = _layer(), _rope()
    query, cache, step, prepared = _step_inputs()
    monkeypatch.setattr(host, "owns", lambda name, rows=None: name == seam)
    assert layer._attn_out_quant_tail(query, cache, step, None, rope, prepared) is None


# -- the calls ------------------------------------------------------------------


def test_a_step_without_requests_quantizes_the_rows_into_the_pair(option, monkeypatch):
    layer, rope = _layer(), _rope()
    query, cache, step, prepared = _step_inputs()
    tail = layer._attend_out_tail(query, cache, step, None, rope, prepared)
    seen = {}

    def inverse_rope_group_quant(o, positions, cos, sin, **kwargs):
        seen.update(o=tuple(o.shape), positions=positions, cos=cos, sin=sin, **kwargs)

    monkeypatch.setattr(attention, "inverse_rope_group_quant", inverse_rope_group_quant)
    layer._quantize_tail(query, tail)
    assert seen["o"] == (3, 8, 64)
    assert seen["cos"] is rope.cos_cache and seen["sin"] is rope.sin_cache
    assert seen["x_fp8"] is tail[3] and seen["x_scale"] is tail[4]
    assert seen["num_groups"] == 2 and seen["quant_group_size"] == 32


def test_the_decode_attention_is_the_entry_that_writes_the_pair(option, monkeypatch):
    layer, rope = _layer(), _rope()
    query, cache, step, prepared = _step_inputs()
    tail = layer._attend_out_tail(query, cache, step, None, rope, prepared)
    values = torch.zeros(1, 3, 8, 64)
    scales = torch.zeros(1, 3, 8, 2)
    prepared = (False, (values, scales), None, "window")
    cache.attention_indices = lambda spec, step, a8w4: (
        "pre",
        "pre_ptr",
        "ext",
        "ext_ptr",
    )
    cache.main_view = lambda owner: "pool"
    cache.ring_view = lambda layer_id: "ring"
    written = []
    cache.write_window = lambda layer_id, window, step: written.append(window)
    step.attention_inputs = None
    calls = []

    def fused(*args, **kwargs):
        calls.append((args, kwargs))

    monkeypatch.setattr(attention, "sparse_attn_v41_paged_2src_inv_rope_quant", fused)
    monkeypatch.setattr(
        attention,
        "sparse_attn_v41_paged_2src",
        lambda *a, **k: pytest.fail("the unfused attention ran"),
    )
    layer._attend_out_seam = lambda *a, **k: pytest.fail("the seam's native ran")
    attend = attention.Attention._attend
    result = getattr(attend, "__wrapped__", attend)(
        layer, query, cache, step, None, prepared, tail=tail
    )
    assert result is None and written == ["window"]
    ((args, kwargs),) = calls
    attending = args[0]
    assert attending[0].shape == (3, 8, 64) and attending[1].shape == (3, 8, 2)
    assert args[1:9] == (
        "pool",
        "pre",
        "pre_ptr",
        "ring",
        "ext",
        "ext_ptr",
        "sink",
        0.125,
    )
    assert args[9] is tail[0]
    # the fp32 twins, not the model's BF16 tables
    assert args[10] is rope.cos_fp32 and args[11] is rope.sin_fp32
    assert kwargs["out"].shape == (3, 8, 64)
    assert kwargs["out"].data_ptr() == query.data_ptr()
    assert kwargs["x_fp8"] is tail[3] and kwargs["x_scale"] is tail[4]
    assert kwargs["num_groups"] == 2
