# SPDX-License-Identifier: MIT
"""A8W4 host contracts without importing AITER or initializing a device."""

import ast
from dataclasses import replace
from pathlib import Path
from types import MethodType, SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
OPS = "atom/model_ops/attentions/deepseek_v41/"


def functions(path, names, scope=None, *, class_name=None):
    """Load the host boundary, as the vLLM copy tests do, without GPU imports."""
    tree = ast.parse((ROOT / path).read_text())
    body = tree.body
    if class_name:
        body = next(
            n.body for n in body if isinstance(n, ast.ClassDef) and n.name == class_name
        )
    selected = [n for n in body if isinstance(n, ast.FunctionDef) and n.name in names]
    assert {n.name for n in selected} == set(names)
    for node in selected:
        node.decorator_list = []
    namespace = {"torch": torch, **(scope or {})}
    exec(  # noqa: S102 - only host functions from this repository.
        compile(ast.Module(body=selected, type_ignores=[]), str(ROOT / path), "exec"),
        namespace,
    )
    return namespace


@pytest.fixture
def api():
    return functions(
        OPS + "a8w4_attention.py",
        ["supports_a8w4_attention", "sparse_attn_v41_a8w4_paged_2src"],
        {
            "envs": SimpleNamespace(ATOM_DSV41_A8W4_ATTN="auto"),
            "get_mla_sparse_prefill_v41_backend": Mock(return_value="opus"),
            "mla_sparse_prefill_v41": Mock(side_effect=lambda *a, **kw: kw["out"]),
        },
    )


@pytest.mark.parametrize("backend", ["opus", "asm", None])
@pytest.mark.parametrize("rows", [0, 4, 65536, 65537])
def test_support_query_uses_current_shape_and_intended_dtypes(api, backend, rows):
    query = SimpleNamespace(
        shape=torch.Size((2, rows, 64, 512)),
        device=torch.device("cuda:3"),
        dtype=torch.bfloat16,
    )
    probe = api["get_mla_sparse_prefill_v41_backend"]
    probe.return_value = backend
    assert api["supports_a8w4_attention"](query, 0.125) is (backend is not None)
    probe.assert_called_once_with(
        query.device,
        (2 * rows, 64, 512),
        unified_kv_dtype=torch.uint8,
        softmax_scale=0.125,
        q_dtype=torch.float8_e4m3fn,
        q_scale_dtype=torch.float8_e8m0fnu,
        unified_kv_scale_dtype=torch.float8_e4m3fn,
        kv_dtype=torch.float8_e4m3fn,
        kv_scale_dtype=torch.float8_e8m0fnu,
        layout="contiguous",
    )


def test_missing_query_keeps_bf16_adapter(api):
    api["get_mla_sparse_prefill_v41_backend"] = None
    assert api["supports_a8w4_attention"](object(), 512**-0.5) is False


def test_off_needs_no_device_probe(api):
    api["envs"].ATOM_DSV41_A8W4_ATTN = "off"
    assert api["supports_a8w4_attention"](object(), 512**-0.5) is False
    api["get_mla_sparse_prefill_v41_backend"].assert_not_called()
    api["envs"].ATOM_DSV41_A8W4_ATTN = "unknown"
    with pytest.raises(ValueError, match="auto or off"):
        api["supports_a8w4_attention"](object(), 512**-0.5)


def test_query_errors_are_not_silently_treated_as_fallback(api):
    api["get_mla_sparse_prefill_v41_backend"].side_effect = ValueError("bad metadata")
    query = SimpleNamespace(shape=torch.Size((1, 2, 64, 512)), device="cuda")
    with pytest.raises(ValueError, match="bad metadata"):
        api["supports_a8w4_attention"](query, 512**-0.5)


@pytest.mark.parametrize("rows", [0, 5])
@pytest.mark.parametrize("splits", [None, 1, 4])
def test_adapter_preserves_source_storage_and_csr(api, rows, splits):
    owner = torch.empty(rows * 288, dtype=torch.uint8)
    pool = (owner[: rows * 256].view(rows, 256), owner[rows * 256 :].view(rows, 32))
    window = torch.empty(7 * 528, dtype=torch.uint8)
    swa = (window[: 7 * 512].view(7, 512), window[7 * 512 :].view(7, 16))
    q = torch.empty(2, 64, 512, dtype=torch.float8_e4m3fn)
    qs = torch.empty(2, 64, 16, dtype=torch.float8_e8m0fnu)
    pi, pp = torch.empty(0, dtype=torch.int32), torch.zeros(3, dtype=torch.int32)
    wi, wp = (
        torch.tensor([0, 2, 6], dtype=torch.int32),
        torch.tensor([0, 1, 3], dtype=torch.int32),
    )
    sink, out = torch.zeros(64), torch.empty(2, 64, 512, dtype=torch.bfloat16)
    result = api["sparse_attn_v41_a8w4_paged_2src"](
        (q, qs),
        pool,
        pi,
        pp,
        swa,
        wi,
        wp,
        sink,
        512**-0.5,
        out=out,
        num_kv_splits=splits,
    )
    assert result is out
    args, kwargs = api["mla_sparse_prefill_v41"].call_args
    assert args[0] is q and args[1] is qs
    for at, source, dtype in (
        (2, pool[0], torch.uint8),
        (3, pool[1], torch.float8_e4m3fn),
        (6, swa[0], torch.float8_e4m3fn),
        (7, swa[1], torch.float8_e8m0fnu),
    ):
        assert args[at].data_ptr() == source.data_ptr()
        assert args[at].dtype == dtype and args[at].is_contiguous()
    assert all(args[i] is value for i, value in ((4, pi), (5, pp), (8, wi), (9, wp)))
    assert kwargs == {"out": out, "num_kv_splits": splits}


@pytest.mark.parametrize("supported", [False, True])
@pytest.mark.parametrize("packed", [False, True])
def test_only_packed_cache_queries_a8w4_support(supported, packed):
    probe = Mock(return_value=supported)
    scope = functions(
        "atom/models/deepseek_v41/attention.py",
        ["_use_a8w4_attention"],
        {"supports_a8w4_attention": probe},
        class_name="Attention",
    )
    query = object()
    assert scope["_use_a8w4_attention"](
        SimpleNamespace(softmax_scale=512**-0.5), query, SimpleNamespace(packed=packed)
    ) is (packed and supported)
    if packed:
        probe.assert_called_once_with(query, 512**-0.5)
    else:
        probe.assert_not_called()


@pytest.mark.parametrize("bounded", [False, True])
def test_support_is_requeried_before_each_q_producer(api, bounded):
    events = []

    def probe(device, shape, **kwargs):
        events.append(("probe", shape[0]))
        return "asm" if shape[0] <= 65536 else None

    api["get_mla_sparse_prefill_v41_backend"].side_effect = probe
    scope = functions(
        "atom/models/deepseek_v41/attention.py",
        ["_use_a8w4_attention", "_attention_inputs"],
        {"supports_a8w4_attention": api["supports_a8w4_attention"]},
        class_name="Attention",
    )
    q8 = object()

    def prepare(query, *, a8w4):
        events.append(("Q8" if a8w4 else "BF16", query.shape[1]))
        return q8 if a8w4 else None

    cache = SimpleNamespace(
        packed=True,
        query_fp8=prepare,
        rope_quant_window=Mock(return_value=(object(), object())),
        rotate_query=Mock(),
    )
    owner = SimpleNamespace(softmax_scale=512**-0.5, spec=SimpleNamespace(layer_id=0))
    owner._use_a8w4_attention = MethodType(scope["_use_a8w4_attention"], owner)
    queries = SimpleNamespace(positions=object()) if bounded else None
    for rows, supported in ((65536, True), (65537, False), (4, True)):
        query = SimpleNamespace(shape=torch.Size((1, rows, 64, 512)), device="cuda:1")
        prepared = scope["_attention_inputs"](
            owner, query, object(), cache, object(), object(), queries
        )
        assert prepared[0] is supported
        assert prepared[1] is (q8 if supported else None)
        assert cache.rope_quant_window.call_args.kwargs["a8w4"] is supported
        if bounded:
            assert cache.rotate_query.call_args.args[-1] is prepared[1]
        else:
            assert cache.rope_quant_window.call_args.args[-1] is prepared[1]
    assert events == [
        ("probe", 65536),
        ("Q8", 65536),
        ("probe", 65537),
        ("BF16", 65537),
        ("probe", 4),
        ("Q8", 4),
    ]


@pytest.mark.parametrize("plugin", [False, True])
@pytest.mark.parametrize("a8w4", [False, True])
@pytest.mark.parametrize("decode", [False, True])
@pytest.mark.parametrize("bounded", [False, True])
def test_native_and_plugin_use_same_sources_and_commit_order(
    plugin, a8w4, decode, bounded
):
    events, calls = [], []

    def attend(*args, **kwargs):
        events.append("attention")
        calls.append((args, kwargs))
        return kwargs["out"].fill_(3)

    scope = {
        "sparse_attn_v41_a8w4_paged_2src": attend,
        "packed_prefill": attend,
        "packed_decode": lambda *args: attend(*args, out=args[0]),
        "join_stream": lambda *args: events.append("join"),
    }
    if plugin:
        scope = functions(
            "atom/plugin/vllm/v41_core_overrides.py", ["_attend_block"], scope
        )
    else:
        scope = functions(
            "atom/models/deepseek_v41/attention.py",
            [
                "forward",
                "_prepare_attention",
                "_prepare_attention_live",
                "_attention_inputs",
                "_attend",
            ],
            scope,
            class_name="Attention",
        )
    query = torch.zeros(1, 2, 64, 512, dtype=torch.bfloat16)
    q8 = (
        torch.empty_like(query, dtype=torch.float8_e4m3fn),
        torch.empty(1, 2, 64, 16, dtype=torch.uint8),
    )
    csa = (
        torch.empty(3, 256, dtype=torch.uint8),
        torch.empty(3, 32, dtype=torch.uint8),
    )
    ring = (
        torch.empty(8, 512, dtype=torch.uint8),
        torch.empty(8, 16, dtype=torch.uint8),
    )
    staged = (
        torch.empty(10, 512, dtype=torch.uint8),
        torch.empty(10, 16, dtype=torch.uint8),
    )
    csr = tuple(torch.zeros(3, dtype=torch.int32) for _ in range(4))
    step = SimpleNamespace(decode=decode, requests=(object(),))
    cache = SimpleNamespace(
        packed=True,
        kv_fp8=False,
        current_step=step,
        query_fp8=Mock(return_value=q8 if a8w4 else None),
        rope_quant_window=Mock(
            side_effect=lambda *a, **k: (
                events.append("prepare") or (staged if a8w4 else query),
                object(),
            )
        ),
        rotate_query=Mock(),
        attention_indices=Mock(return_value=csr),
        main_view=Mock(return_value=csa),
        ring_view=Mock(return_value=ring),
        write_window=Mock(side_effect=lambda *args: events.append("commit")),
        rope_positions=Mock(return_value=object()),
    )
    model = SimpleNamespace(
        heads=64,
        head_dim=512,
        spec=SimpleNamespace(layer_id=0, kv_owner=0),
        index_stream=None,
        compress_stream=None,
        attn_sink=torch.zeros(64),
        softmax_scale=512**-0.5,
        _fork_compress=Mock(return_value=False),
        project_qkv=Mock(return_value=(None, None)),
        qk_norm=Mock(return_value=(None, None, None)),
        _fork_select=Mock(),
        wq_b=Mock(return_value=query.flatten(-2)),
        _use_a8w4_attention=Mock(return_value=a8w4),
        _project_out=Mock(side_effect=lambda value, *args: value),
    )
    queries = (
        SimpleNamespace(
            take=Mock(return_value=(None, None)),
            indices=Mock(return_value=csr),
            positions=object(),
        )
        if bounded
        else None
    )
    if plugin:
        result = torch.empty_like(query)
        scope["_attend_block"](model, result, None, None, cache, step, None, queries)
    else:
        for name in (
            "_prepare_attention",
            "_prepare_attention_live",
            "_attention_inputs",
            "_attend",
        ):
            setattr(model, name, MethodType(scope[name], model))
        result = scope["forward"](model, None, None, cache, step, None, queries)
    assert events == ["prepare", "join"] + (
        ["commit", "attention"] if decode else ["attention", "commit"]
    )
    ((args, kwargs),) = calls
    if a8w4:
        assert args[1] is csa and args[4] is (ring if decode else staged)
        assert args[0][0].data_ptr() == q8[0].data_ptr()
    else:
        assert args[1] == (csa, ring)
        assert args[0].dtype == torch.bfloat16
    assert "backend" not in kwargs
    assert cache.query_fp8.call_args.kwargs == {"a8w4": a8w4}
    assert cache.rope_quant_window.call_args.kwargs == {"a8w4": a8w4}
    index_call = queries.indices if bounded else cache.attention_indices
    assert index_call.call_args.kwargs == {"a8w4": a8w4}
    assert torch.all(result == 3)


@pytest.mark.parametrize("kv_fp8,a8w4", [(False, False), (False, True), (True, False)])
@pytest.mark.parametrize("decode", [False, True])
@pytest.mark.parametrize("bounded", [False, True])
def test_preparation_only_runs_live_for_variable_sources(kv_fp8, a8w4, decode, bounded):
    scope = functions(
        "atom/models/deepseek_v41/attention.py",
        ["_prepare_attention"],
        class_name="Attention",
    )
    prepared = object()
    model = SimpleNamespace(
        _use_a8w4_attention=Mock(return_value=a8w4),
        _prepare_attention_live=Mock(),
        _attention_inputs=Mock(return_value=prepared),
    )
    cache, step = SimpleNamespace(kv_fp8=kv_fp8), SimpleNamespace(decode=decode)
    args = (object(), object(), cache, step, object(), object() if bounded else None)
    result = scope["_prepare_attention"](model, *args)
    live = bounded or (not decode and (kv_fp8 or a8w4))
    if live:
        model._prepare_attention_live.assert_called_once_with(*args)
        model._attention_inputs.assert_not_called()
        assert result is None
    else:
        model._prepare_attention_live.assert_not_called()
        model._attention_inputs.assert_called_once_with(*args)
        assert result is prepared


def test_bounded_queries_do_not_reuse_full_step_a8w4_indptrs():
    from dataclasses import dataclass, field

    @dataclass
    class Step:
        group_indices: dict = field(default_factory=dict)
        a8w4_indptrs: dict = field(default_factory=dict)
        window_rows: object = None

    scope = functions(
        "atom/models/deepseek_v41/bounded_replay.py",
        ["indices"],
        {"replace": replace},
        class_name="QueryRows",
    )
    original = Step({0: object()}, {2: object()}, object())
    query = SimpleNamespace(index=original, ptr_rows=torch.tensor([1, 3, 4]))
    pptr, eptr = torch.tensor([0, 0, 2, 2, 5]), torch.tensor([0, 0, 3, 3, 7])
    cache = SimpleNamespace(
        attention_indices=Mock(return_value=(object(), pptr, object(), eptr))
    )
    _, pp, _, wp = scope["indices"](query, cache, object(), a8w4=True)
    child = cache.attention_indices.call_args.args[1]
    assert (
        child.group_indices == {}
        and child.a8w4_indptrs == {}
        and child.window_rows is None
    )
    assert (
        original.a8w4_indptrs
        and original.group_indices
        and original.window_rows is not None
    )
    assert pp.tolist() == [0, 2, 5] and wp.tolist() == [0, 3, 7]


@pytest.mark.parametrize("row_ids", [[5, 0, 4, 3, 5, 1], [1, 2, 0], [4, 6, 5], []])
def test_mono_reference_gathers_before_packing_sources(row_ids, monkeypatch):
    pack = functions(OPS + "packed_rows.py", ["pack_rows"])["pack_rows"]
    gather = functions(
        "atom/models/deepseek_v41/mono/check.py",
        ["_gather_key_rows"],
        {"pack_rows": pack},
    )["_gather_key_rows"]
    data = torch.arange(7 * 32).view(7, 32).to(torch.uint8)
    scales = torch.arange(7, dtype=torch.uint8).view(7, 1)
    main, window = (data[:4], scales[:4]), (data[4:], scales[4:])
    ids = torch.tensor(row_ids, dtype=torch.int32)
    expected = pack(data[ids.long()], scales[ids.long()])
    cat = torch.cat
    calls = []

    def selected_only(parts, dim=0):
        assert dim == -1
        assert all(p.shape[0] == ids.numel() for p in parts)
        calls.append(tuple(p.shape for p in parts))
        return cat(parts, dim=dim)

    monkeypatch.setattr(torch, "cat", selected_only)
    torch.testing.assert_close(gather(main, window, ids), expected)
    assert len(calls) == 1
