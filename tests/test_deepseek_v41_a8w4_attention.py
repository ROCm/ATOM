# SPDX-License-Identifier: MIT
"""A8W4 host contracts without importing AITER or initializing a device."""

import ast
import struct
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
        ["select_a8w4_backend", "sparse_attn_v41_a8w4_paged_2src"],
        {
            "struct": struct,
            "envs": SimpleNamespace(ATOM_DSV41_A8W4_ATTN="auto"),
            "get_gfx_runtime": Mock(return_value="gfx950"),
            "mla_sparse_prefill_v41": Mock(side_effect=lambda *a, **kw: kw["out"]),
            "opus_mla_v41_prefill_a8w4_gfx950_fwd": object(),
            "asm_mla_sparse_prefill_v41_fwd": object(),
            "is_gfx1250_asm_supported": Mock(return_value=True),
        },
    )


@pytest.mark.parametrize(
    "gfx,heads,dim,stepping,scale,expected",
    [
        *(
            ("gfx950", h, 512, True, 512**-0.5, "opus")
            for h in (2, 4, 8, 16, 32, 64, 128)
        ),
        ("gfx1250", 64, 512, True, 512**-0.5, "asm"),
        ("gfx1250", 32, 512, True, 512**-0.5, None),
        ("gfx1250", 64, 512, False, 512**-0.5, None),
        ("gfx1250", 64, 512, True, 0.125, None),
        ("gfx950", 64, 128, True, 128**-0.5, None),
        ("gfx942", 64, 512, True, 512**-0.5, None),
    ],
)
def test_backend_eligibility(api, gfx, heads, dim, stepping, scale, expected):
    api["get_gfx_runtime"].return_value = gfx
    api["is_gfx1250_asm_supported"].return_value = stepping
    assert api["select_a8w4_backend"](heads, dim, scale) == expected


@pytest.mark.parametrize("backend", ["opus", "asm"])
def test_missing_backend_keeps_old_adapter(api, backend):
    api["get_gfx_runtime"].return_value = "gfx950" if backend == "opus" else "gfx1250"
    symbol = (
        "opus_mla_v41_prefill_a8w4_gfx950_fwd"
        if backend == "opus"
        else "asm_mla_sparse_prefill_v41_fwd"
    )
    api[symbol] = None
    assert api["select_a8w4_backend"](64, 512, 512**-0.5) is None


def test_off_needs_no_arch_probe(api):
    api["envs"].ATOM_DSV41_A8W4_ATTN = "off"
    assert api["select_a8w4_backend"](64, 512, 512**-0.5) is None
    api["get_gfx_runtime"].assert_not_called()
    api["envs"].ATOM_DSV41_A8W4_ATTN = "unknown"
    with pytest.raises(ValueError, match="auto or off"):
        api["select_a8w4_backend"](64, 512, 512**-0.5)


@pytest.mark.parametrize("backend", ["opus", "asm"])
@pytest.mark.parametrize("rows", [0, 5])
@pytest.mark.parametrize("splits", [None, 1, 4])
def test_adapter_preserves_source_storage_and_csr(api, backend, rows, splits):
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
        backend=backend,
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
    assert kwargs == {"out": out, "num_kv_splits": splits, "backend": backend}


@pytest.mark.parametrize(
    "backend,rows,packed,expected",
    [
        ("asm", 65536, True, "asm"),
        ("asm", 65537, True, None),
        ("opus", 65537, True, "opus"),
        ("opus", 2, False, None),
        (None, 2, True, None),
    ],
)
def test_shape_fallback_precedes_q_preparation(backend, rows, packed, expected):
    scope = functions(
        "atom/models/deepseek_v41/attention.py",
        ["_a8w4_attention_backend"],
        class_name="Attention",
    )
    query = SimpleNamespace(shape=torch.Size((1, rows, 64, 512)))
    assert (
        scope["_a8w4_attention_backend"](
            SimpleNamespace(_a8w4_backend=backend),
            query,
            SimpleNamespace(packed=packed),
        )
        == expected
    )


@pytest.mark.parametrize("plugin", [False, True])
@pytest.mark.parametrize("backend", ["opus", "asm"])
@pytest.mark.parametrize("decode", [False, True])
@pytest.mark.parametrize("bounded", [False, True])
def test_native_and_plugin_use_same_sources_and_commit_order(
    plugin, backend, decode, bounded
):
    events, calls = [], []

    def attend(*args, **kwargs):
        events.append("attention")
        calls.append((args, kwargs))
        return kwargs["out"].fill_(3)

    scope = {
        "sparse_attn_v41_a8w4_paged_2src": attend,
        "join_stream": lambda *args: events.append("join"),
    }
    if plugin:
        scope = functions(
            "atom/plugin/vllm/v41_core_overrides.py", ["_attend_block"], scope
        )
    else:
        scope = functions(
            "atom/models/deepseek_v41/attention.py",
            ["forward", "_prepare_attention", "_attend"],
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
        query_fp8=Mock(return_value=q8),
        rope_quant_window=Mock(
            side_effect=lambda *a, **k: (events.append("prepare") or staged, object())
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
        _a8w4_attention_backend=Mock(return_value=backend),
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
        model._prepare_attention = MethodType(scope["_prepare_attention"], model)
        model._attend = MethodType(scope["_attend"], model)
        result = scope["forward"](model, None, None, cache, step, None, queries)
    assert events == ["prepare", "join"] + (
        ["commit", "attention"] if decode else ["attention", "commit"]
    )
    ((args, kwargs),) = calls
    assert args[1] is csa and args[4] is (ring if decode else staged)
    assert args[0][0].data_ptr() == q8[0].data_ptr()
    assert kwargs["backend"] == backend
    assert cache.query_fp8.call_args.kwargs == {"a8w4": True}
    assert cache.rope_quant_window.call_args.kwargs == {"a8w4": True}
    index_call = queries.indices if bounded else cache.attention_indices
    assert index_call.call_args.kwargs == {"a8w4": True}
    assert torch.all(result == 3)


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
