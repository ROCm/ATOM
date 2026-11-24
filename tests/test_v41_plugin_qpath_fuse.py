# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""`model_options.qpath_fuse` of the DeepSeek-V4.1 plugin model: which halves
of the RoPE / window launch a decode step moves onto the q/kv norm (`kv`) and
onto wq_b (`q`), with which rotation form, and what the attention then calls.
CPU only: the kernels are stood in for; what is checked is the plan for a
step, the arguments of the fused launches, the launch that is left for a half
that stays behind, and that a step outside the plan takes the calls it took."""

from types import SimpleNamespace

import pytest
import torch
from aiter_stub import atom_without_aiter

qpath = None
attention = None
host = None
recipe = None


@pytest.fixture(scope="module", autouse=True)
def _modules():
    global qpath, attention, host, recipe
    with atom_without_aiter() as load:
        host = load("atom.plugin_host")
        recipe = load("atom.plugin_recipe")
        attention = load("atom.models.deepseek_v41_plugin.attention")
        qpath = load("atom.models.deepseek_v41_plugin.qpath_fuse")
        yield


CU = 256  # the reference groups 8 heads a program from 64 rows of 64 heads


@pytest.fixture
def fuse(monkeypatch):
    """Turn items on, on a gfx950 whose AITER fuses wq_b at every row count."""

    def set_items(*names):
        plan = SimpleNamespace(model_options=recipe.ModelOptions(qpath_fuse=names))
        monkeypatch.setattr(host, "_plan", plan)

    monkeypatch.setattr(host, "_plan", None)
    monkeypatch.setattr(host, "owns", lambda seam, rows=None: False)
    monkeypatch.setattr(qpath, "_logged", set())
    monkeypatch.setattr(qpath, "COUNTS", dict.fromkeys(qpath.COUNTS, 0))
    monkeypatch.setattr(qpath, "_gfx950", lambda: True)
    monkeypatch.setattr(qpath, "_aiter_norm_may_be_served", lambda: False)
    monkeypatch.setattr(
        qpath, "_aiter_rope_quant", lambda: ("rope_quant_gemm", lambda *a: True)
    )
    monkeypatch.setattr(qpath, "reference_fingerprint", lambda: "0123456789abcdef")
    monkeypatch.setattr(
        qpath,
        "reference_program_heads",
        lambda rows, heads, device: 8 if rows * heads // 8 >= 2 * CU else 1,
    )
    return set_items


def _weight():
    weight = torch.empty(64 * 512, 1280, dtype=torch.float8_e4m3fn)
    weight.is_shuffled = True
    return weight


def _attn():
    norm = type("DualRMSNormMXFP8", (), {})()
    wq_b = SimpleNamespace(
        weight=_weight(),
        weight_scale=torch.empty(64 * 512 // 32, 1280 // 32),
        native_a8_group_rows=32,
        bias=None,
    )
    return SimpleNamespace(
        heads=64, head_dim=512, qk_norm=norm, wq_b=wq_b, _qpath_fuse_state=None
    )


def _rope(dtype=torch.bfloat16):
    return SimpleNamespace(
        rope_dim=64,
        cos_cache=torch.zeros(128, 32, dtype=dtype),
        sin_cache=torch.zeros(128, 32, dtype=dtype),
    )


def _step(rows, decode=True):
    return SimpleNamespace(
        decode=decode, running_tokens=rows, positions=torch.arange(rows)
    )


def _cache():
    return SimpleNamespace(kv_fp8=True, window_fp8=True, packed=False)


def _plan(rows, *, rope=None, attn=None, cache=None, step=None, queries=None):
    return qpath.plan_forms(
        attn or _attn(),
        torch.zeros(1, rows, 8),
        cache or _cache(),
        step or _step(rows),
        rope or _rope(),
        queries,
    )


# -- the option ---------------------------------------------------------------


def test_the_recipe_takes_the_two_items_in_a_fixed_order():
    assert recipe.ModelOptions().qpath_fuse == ()
    options = recipe.ModelOptions.from_mapping({"qpath_fuse": ["q", "kv", "q"]})
    assert options.qpath_fuse == ("kv", "q")
    for bad in (["k"], "kv,q", ["kv", 1], [True], None):
        with pytest.raises(recipe.RecipeRefused):
            recipe.ModelOptions.from_mapping({"qpath_fuse": bad})


def test_off_by_default_the_plan_reads_nothing(fuse):
    assert qpath.items() == () and not qpath.enabled("kv")
    assert qpath.plan_forms(None, None, None, None, None, None) == (None, None)


# -- the reference's layout ------------------------------------------------------


def test_the_reference_groups_heads_only_where_that_leaves_programs(monkeypatch):
    from aiter.jit.utils import chip_info

    monkeypatch.setattr(chip_info, "get_cu_num", lambda: CU)
    heads = qpath.reference_program_heads
    cpu = torch.device("cpu")
    assert [heads(rows, 64, cpu) for rows in (1, 63, 64, 300)] == [1, 1, 8, 8]
    assert heads(1000, 60, cpu) == 1  # 8 does not divide the heads
    monkeypatch.setattr(chip_info, "get_cu_num", lambda: 0)
    assert heads(64, 64, cpu) is None


# -- the plan ------------------------------------------------------------------


@pytest.mark.parametrize(
    "rows, tables, expected",
    [
        (6, torch.bfloat16, ("G", "G")),  # a program a head
        (64, torch.bfloat16, ("P", "G")),  # grouped heads
        (6, torch.float32, ("P", "P")),
        (64, torch.float32, ("P", "G")),
    ],
)
def test_both_items_name_the_form_of_the_reference_launch(fuse, rows, tables, expected):
    fuse("kv", "q")
    assert _plan(rows, rope=_rope(tables)) == expected
    assert qpath.COUNTS["rope_window_launches"] == 0
    assert qpath.plan(
        _attn(), torch.zeros(1, rows, 8), _cache(), _step(rows), _rope(tables), None
    ) == (True, True)


def test_below_the_first_checked_row_count_wq_b_keeps_its_launch(fuse):
    fuse("kv", "q")
    assert qpath.Q_MIN_ROWS == 5
    assert _plan(4) == ("G", None)
    assert _plan(5) == ("G", "G")
    assert qpath.COUNTS["rope_window_launches"] == 1


def test_a_row_count_aiter_does_not_fuse_keeps_wq_b(fuse, monkeypatch):
    fuse("kv", "q")
    monkeypatch.setattr(
        qpath, "_aiter_rope_quant", lambda: ("op", lambda rows, *a: rows != 64)
    )
    assert _plan(64) == ("P", None)
    monkeypatch.setattr(qpath, "_aiter_rope_quant", lambda: (None, None))
    assert _plan(6) == ("G", None)


def test_a_half_left_behind_must_round_as_the_full_launch(fuse):
    """One item alone: the other half is launched by the program that runs it
    alone, and the item fuses only where that program has the same form."""
    fuse("q")
    assert _plan(6) == (None, "G")  # kv alone: G, as beside a program a head
    assert _plan(64) == (None, None)  # kv alone: G, beside grouped heads: P
    fuse("kv")
    assert _plan(6) == ("G", None)
    assert _plan(64) == ("P", None)
    fuse("kv")
    assert _plan(6, rope=_rope(torch.float32)) == ("P", None)


def test_an_aiter_provider_of_the_norm_keeps_the_norm(fuse, monkeypatch):
    fuse("kv", "q")
    monkeypatch.setattr(qpath, "_aiter_norm_may_be_served", lambda: True)
    assert _plan(6) == (None, "G")


@pytest.mark.parametrize(
    "change",
    [
        {"step": _step(6, decode=False)},
        {"queries": object()},
        {"cache": SimpleNamespace(kv_fp8=True, window_fp8=True, packed=True)},
        {"cache": SimpleNamespace(kv_fp8=True, window_fp8=False, packed=False)},
        {"cache": SimpleNamespace(kv_fp8=False, window_fp8=True, packed=False)},
        {"step": SimpleNamespace(decode=True, running_tokens=0, positions=None)},
        {
            "rope": SimpleNamespace(
                rope_dim=32, cos_cache=torch.zeros(8, 16), sin_cache=torch.zeros(8, 16)
            )
        },
        {"rope": _rope(torch.float64)},
        {
            "step": SimpleNamespace(
                decode=True,
                running_tokens=6,
                positions=torch.arange(6, dtype=torch.int32),
            )
        },
        {
            "step": SimpleNamespace(
                decode=True, running_tokens=6, positions=torch.arange(5)
            )
        },
    ],
)
def test_any_other_step_keeps_todays_launches(fuse, change):
    fuse("kv", "q")
    assert _plan(6, **change) == (None, None)
    assert qpath.COUNTS["unfused"] == 1


def test_an_owner_of_the_norm_seam_keeps_its_calls(fuse, monkeypatch):
    fuse("kv", "q")
    monkeypatch.setattr(
        host, "owns", lambda seam, rows=None: seam == "v41.attn.qk_norm"
    )
    assert _plan(6) == (None, None)


def test_a_layer_of_another_shape_or_norm_is_left_alone(fuse):
    fuse("kv", "q")
    attn = _attn()
    attn.heads = 32
    assert _plan(6, attn=attn) == (None, None)
    attn = _attn()
    attn.qk_norm = type("DualRMSNormBF16", (), {})()
    assert _plan(6, attn=attn) == (None, None)
    assert attn._qpath_fuse_state == (False, False)


def test_a_wq_b_that_is_not_the_native_gemm_keeps_its_launch(fuse):
    fuse("kv", "q")
    for change in (
        lambda w: setattr(w, "native_a8_group_rows", None),
        lambda w: setattr(w, "bias", torch.zeros(1)),
        lambda w: setattr(w.weight, "is_shuffled", False),
        lambda w: setattr(w, "is_output_padded", True),
    ):
        attn = _attn()
        change(attn.wq_b)
        assert _plan(6, attn=attn) == ("G", None)
        assert attn._qpath_fuse_state == (True, False)


def test_the_log_says_whether_the_reference_kernel_is_the_recorded_one(
    fuse, monkeypatch, caplog
):
    """The fused launches compute the recorded forms on any build; whether
    that is bit for bit this tree's own launch is what the line reports."""
    fuse("kv", "q")
    with caplog.at_level("INFO", logger="atom"):
        assert _plan(6) == ("G", "G")
    assert "is not the recorded reference (fingerprint 0123456789abcdef)" in caplog.text
    caplog.clear()
    monkeypatch.setattr(qpath, "_logged", set())
    proven = next(iter(qpath.PROVEN_FINGERPRINTS))
    monkeypatch.setattr(qpath, "reference_fingerprint", lambda: proven)
    with caplog.at_level("INFO", logger="atom"):
        assert _plan(6) == ("G", "G")
    assert f"the recorded reference kernel (fingerprint {proven})" in caplog.text


# -- the launches ----------------------------------------------------------------


def _layer(monkeypatch):
    """The parts of an attention layer `_norm_query_qpath` / `_qpath_rest`
    read, recording what each launch is handed."""
    calls = []
    norm = SimpleNamespace(
        q_norm=SimpleNamespace(weight="gq", eps=1e-6),
        k_norm=SimpleNamespace(weight="gk", eps=2e-6),
        amax_floor=1e-10,
    )

    def norm_call(q, kv):
        calls.append(("norm",))
        return torch.zeros(1, 3, 1280), torch.zeros(1, 3, 40), "kv_normed"

    layer = SimpleNamespace(
        heads=4,
        head_dim=64,
        spec=SimpleNamespace(layer_id=7),
        calls=calls,
    )
    layer.qk_norm = _CallableNorm(norm, norm_call)

    def wq_b(qr, x_scale=None):
        calls.append(("wq_b", qr, x_scale))
        return torch.zeros(1, 3, 4 * 64)

    wq_b.weight, wq_b.weight_scale = "wq_b.weight", "wq_b.scale"
    layer.wq_b = wq_b
    for name in ("_norm_query_qpath", "_qpath_rest"):
        setattr(layer, name, getattr(attention.Attention, name).__get__(layer))

    def kv_window(q, kv, gq, gk, eps_q, eps_k, floor, cos, sin, positions, ring, **kw):
        calls.append(
            (
                "kv_window",
                tuple(q.shape),
                tuple(kv.shape),
                gq,
                gk,
                eps_q,
                eps_k,
                floor,
                cos,
                sin,
                positions,
                ring,
                kw,
            )
        )
        rows = q.shape[0]
        return (
            torch.zeros(rows, q.shape[1], dtype=torch.uint8),
            torch.zeros(rows, q.shape[1] // 32, dtype=torch.uint8),
            torch.zeros(rows, kv.shape[1]),
        )

    import sys
    import types

    ops = types.ModuleType("atom.model_ops.deepseek_v41.qpath_fuse_ops")
    ops.dual_rmsnorm_mxfp8_kv_window = kv_window
    monkeypatch.setitem(sys.modules, ops.__name__, ops)

    def rope_quant_gemm(
        x, w, x_scale, w_scale, positions, cos, sin, q_fp8, q_scale, **kw
    ):
        calls.append(
            (
                "rope_quant_gemm",
                tuple(x.shape),
                w,
                tuple(x_scale.shape),
                w_scale,
                positions,
                cos,
                sin,
                tuple(q_fp8.shape),
                tuple(q_scale.shape),
                kw,
            )
        )
        return torch.zeros(x.shape[0], 4 * 64)

    monkeypatch.setattr(qpath, "_aiter_rope_quant", lambda: (rope_quant_gemm, None))
    return layer


class _CallableNorm:
    def __init__(self, norm, call):
        self.q_norm, self.k_norm = norm.q_norm, norm.k_norm
        self.amax_floor = norm.amax_floor
        self._call = call

    def __call__(self, q, kv):
        return self._call(q, kv)


def _launch_inputs(rows=3):
    step = SimpleNamespace(positions=torch.arange(rows))
    rope = _rope()
    cache = SimpleNamespace(
        ring_view=lambda layer: ("ring", layer),
        geometry=SimpleNamespace(window=lambda layer, pages: ("window", layer, pages)),
        num_pages=11,
        calls=[],
    )
    cache.query_fp8 = lambda query: ("query_fp8", tuple(query.shape))
    cache.rope_quant_window = lambda *a, **k: cache.calls.append(("rope_kv", a, k))
    cache.rotate_query = lambda *a: cache.calls.append(("rotate_query", a))
    q_lora = torch.zeros(1, rows, 1280)
    kv_pre = torch.zeros(1, rows, 512)
    return q_lora, kv_pre, cache, step, rope


def test_both_fused_the_norm_stores_the_kv_row_and_wq_b_writes_the_query(
    fuse, monkeypatch
):
    layer = _layer(monkeypatch)
    q_lora, kv_pre, cache, step, rope = _launch_inputs()
    query, qr, qr_scale, kv_normed, query_fp8 = layer._norm_query_qpath(
        q_lora, kv_pre, cache, step, rope, "G", "P"
    )
    kv_call, gemm_call = layer.calls
    assert kv_call[:8] == (
        "kv_window",
        (3, 1280),
        (3, 512),
        "gq",
        "gk",
        1e-6,
        2e-6,
        1e-10,
    )
    # the model's own tables, in place
    assert kv_call[8].data_ptr() == rope.cos_cache.data_ptr()
    assert kv_call[9].data_ptr() == rope.sin_cache.data_ptr()
    assert kv_call[10] is step.positions
    assert kv_call[11] == (("ring", 7), step, ("window", 7, 11))
    assert kv_call[12] == {"rope_dim": 64, "fuse_cos": True}  # form G
    assert gemm_call[:5] == (
        "rope_quant_gemm",
        (3, 1280),
        "wq_b.weight",
        (3, 40),
        "wq_b.scale",
    )
    assert gemm_call[5] is step.positions
    assert gemm_call[8:10] == ((1, 3, 4, 64), (1, 3, 4, 2))
    assert gemm_call[10] == {"head_dim": 64, "rope_dim": 64}  # form P: the default
    assert query.shape == (1, 3, 4, 64)
    assert qr.shape == (1, 3, 1280) and qr_scale.shape == (1, 3, 40)
    assert qr_scale.dtype == torch.float8_e8m0fnu and kv_normed.shape == (1, 3, 512)
    assert query_fp8[0].shape == (1, 3, 4, 64) and query_fp8[1].shape == (1, 3, 4, 2)
    prepared = layer._qpath_rest(
        query, kv_normed, cache, step, rope, query_fp8, "G", "P"
    )
    assert prepared == (False, query_fp8, None, None) and cache.calls == []


def test_a_half_that_is_not_fused_runs_the_launch_it_has_alone(fuse, monkeypatch):
    layer = _layer(monkeypatch)
    q_lora, kv_pre, cache, step, rope = _launch_inputs()
    # kv fused, q not: the norm stores the KV row, wq_b is its own call and the
    # query is rotated alone
    query, _, _, kv_normed, query_fp8 = layer._norm_query_qpath(
        q_lora, kv_pre, cache, step, rope, "P", None
    )
    assert [c[0] for c in layer.calls] == ["kv_window", "wq_b"]
    assert layer.calls[0][12] == {"rope_dim": 64}
    assert query_fp8 == ("query_fp8", (1, 3, 4, 64))
    layer._qpath_rest(query, kv_normed, cache, step, rope, query_fp8, "P", None)
    assert [c[0] for c in cache.calls] == ["rotate_query"]
    assert cache.calls[0][1][1:] == (rope, step.positions, query_fp8)

    # q fused, kv not: the norm is its own call and the KV row is rotated,
    # quantized and stored alone
    layer.calls.clear()
    cache.calls.clear()
    query, _, _, kv_normed, query_fp8 = layer._norm_query_qpath(
        q_lora, kv_pre, cache, step, rope, None, "G"
    )
    assert [c[0] for c in layer.calls] == ["norm", "rope_quant_gemm"]
    assert layer.calls[1][10] == {"head_dim": 64, "rope_dim": 64, "fuse_cos": True}
    layer._qpath_rest(query, kv_normed, cache, step, rope, query_fp8, None, "G")
    ((name, args, kwargs),) = cache.calls
    assert name == "rope_kv" and args[:3] == (7, None, "kv_normed")
    assert args[3:] == (rope, step) and kwargs == {"a8w4": False}


# -- the forward -----------------------------------------------------------------


def _forward_layer(monkeypatch, plan):
    """An attention layer whose every step records its name: the order of the
    launches of one forward."""
    order = []

    def step_of(name, result=None):
        def run(*args, **kwargs):
            order.append(name)
            return result

        return run

    query = torch.zeros(1, 3, 4, 64)
    layer = SimpleNamespace(
        spec=SimpleNamespace(layer_id=7),
        heads=4,
        head_dim=64,
        index_stream=None,
        compress_stream=None,
        wqkv_a=SimpleNamespace(output_sizes=(1280, 512)),
        project_qkv="project_qkv",
        _fork_compress=step_of("compress", False),
        _use_fused_attention=lambda *a: False,
        qk_norm=step_of("qk_norm", ("qr", "qr_scale", "kv_normed")),
        _fork_select=step_of("select"),
        wq_b=step_of("wq_b", query.flatten(-2)),
        _prepare_attention=step_of("rope_window", "prepared"),
        _norm_query_qpath=step_of(
            "norm+wq_b", (query, "qr", "qr_scale", "kv_normed", "query_fp8")
        ),
        _qpath_rest=step_of("rest", "prepared"),
        _attend_out_tail=lambda *a: None,
        _attend=step_of("attend", query),
        _project_out=step_of("project_out", "out"),
    )
    monkeypatch.setattr(
        attention.plugin_transition,
        "end",
        lambda *a: order.append("wqkv_a") or ("q_lora", "kv_pre"),
    )
    monkeypatch.setattr(attention, "join_stream", lambda *a: None)
    monkeypatch.setattr(attention.qpath_fuse, "plan_forms", lambda *a: plan)
    cache = SimpleNamespace(rope_positions=lambda step: "positions")
    out = attention.Attention.forward(layer, "hidden", None, cache, "step", "rope")
    assert out == "out"
    return order


def test_off_the_forward_launches_what_it_launched(fuse, monkeypatch):
    order = _forward_layer(monkeypatch, ("G", "G"))  # no item: the plan is not asked
    assert order == [
        "compress",
        "wqkv_a",
        "qk_norm",
        "select",
        "wq_b",
        "rope_window",
        "attend",
        "project_out",
    ]
    fuse("kv", "q")
    assert _forward_layer(monkeypatch, (None, None)) == order


@pytest.mark.parametrize("plan", [("G", "G"), ("P", None), (None, "G")])
def test_a_planned_step_has_no_rope_window_launch_of_both_halves(
    fuse, monkeypatch, plan
):
    fuse("kv", "q")
    assert _forward_layer(monkeypatch, plan) == [
        "compress",
        "wqkv_a",
        "norm+wq_b",
        "select",
        "rest",
        "attend",
        "project_out",
    ]


def test_fused_attention_serves_its_steps_first(fuse, monkeypatch):
    """`model_options.fused_attn` rotates the query itself: the plan is not
    asked on a step it serves."""
    fuse("kv", "q")
    asked = []
    monkeypatch.setattr(
        attention.qpath_fuse, "plan_forms", lambda *a: asked.append(1) or ("G", "G")
    )
    layer = SimpleNamespace(
        spec=SimpleNamespace(layer_id=7),
        wqkv_a=SimpleNamespace(output_sizes=(1280, 512)),
        project_qkv=None,
        _fork_compress=lambda *a: False,
        _use_fused_attention=lambda *a: True,
        _prepare_fused_attention=lambda *a: (_ for _ in ()).throw(StopIteration),
    )
    monkeypatch.setattr(attention.plugin_transition, "end", lambda *a: ("q", "kv"))
    with pytest.raises(StopIteration):
        attention.Attention.forward(layer, "hidden", None, None, "step", "rope")
    assert asked == []
