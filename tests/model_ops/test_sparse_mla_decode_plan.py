# SPDX-License-Identifier: MIT
"""Sparse decode's builder/consumer contract; CPU stubs stop at GPU operations."""

from __future__ import annotations

import ast
from itertools import pairwise
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

WORK = (
    "work_meta_data",
    "work_indptr",
    "work_info_set",
    "reduce_indptr",
    "reduce_final_map",
    "reduce_partial_map",
)
BUILD = {
    "_sparse_dcp_decode_plan",
    "set_mla_persistent_worker_buffers",
    "_set_mla_persistent_worker_buffers_sparse_mtp",
}
DECODE = {
    "_forward_decode",
    "_sparse_dcp_decode_work_buffers",
    "_rebuild_sparse_dcp_persistent_metadata",
}


def methods(path, names, namespace):
    tree = ast.parse((Path(__file__).parents[2] / path).read_text(encoding="utf-8"))
    nodes = [
        n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name in names
    ]
    assert {n.name for n in nodes} == names
    exec(  # noqa: S102 -- real production bodies without importing GPU dependencies
        compile(ast.Module(body=nodes, type_ignores=[]), path, "exec"), namespace
    )
    return {name: namespace[name] for name in names}


@pytest.fixture
def runtime():
    ns = {
        "torch": torch,
        "PERSISTENT_WORK_BUFFERS": WORK,
        "_MLA_SPLIT_BUDGET_AUTO": -1,
        "envs": SimpleNamespace(ATOM_MLA_PAGE_SIZE=1),
        "triton": SimpleNamespace(next_power_of_2=lambda n: 1 << (n - 1).bit_length()),
    }
    ns["get_mla_decode_shape_support"] = lambda *a, **kw: SimpleNamespace(
        native=True, supported=True
    )
    ns["get_mla_decode_fwd_max_splits"] = lambda *a: 256
    ns["Builder"] = type(
        "Builder", (), methods("atom/model_ops/attentions/aiter_mla.py", BUILD, ns)
    )
    ns["Attention"] = type(
        "Attention", (), methods("atom/model_ops/attention_mla.py", DECODE, ns)
    )
    calls = []

    def sizes(rows, max_q, heads, dtype_q, dtype_kv, *, is_sparse, fast_mode):
        assert max_q == 1 and is_sparse is False and fast_mode is True
        groups = ns["get_mla_decode_fwd_max_splits"](heads, max_q, dtype_q, dtype_kv)
        return [
            (2, torch.int64),
            (groups + 1, torch.int32),
            ((rows + groups, 8), torch.int32),
            (rows + 1, torch.int32),
            ((rows, 2), torch.int32),
            (rows + 2 * groups, torch.int32),
        ]

    def planner(qptr, kvptr, last, heads, _, causal, wmd, info, wi, ri, rfm, rpm, **kw):
        rows = qptr.numel() - 1
        groups = ns["get_mla_decode_fwd_max_splits"](
            heads, 1, kw["dtype_q"], kw["dtype_kv"]
        )
        assert wi.numel() >= groups + 1, "planner work-indptr allocation is too small"
        calls.append((heads, kvptr.clone(), kw, info))
        info.zero_()
        info[:rows, 0] = torch.arange(rows)
        info[:rows, 1] = -1
        info[:rows, 2] = torch.arange(rows)
        info[:rows, 3] = torch.arange(1, rows + 1)
        info[:rows, 4] = kvptr[:-1] - kvptr[0]
        info[:rows, 5] = kvptr[1:] - kvptr[0]
        wi.copy_(torch.arange(wi.numel()).clamp_max(rows))
        for tensor in (wmd, ri, rfm, rpm):
            tensor.fill_(len(calls))

    def patch(info, wi, ptr, rows):
        count = int(wi[-1])
        ids = info[:count, 0].long()
        assert count <= rows and (ids < rows).all()
        info[:count, 4] = ptr[ids] - ptr[0]
        info[:count, 5] = ptr[ids + 1] - ptr[0]

    def launch(q, kv, out, qptr, ptr, indices, last, *a, **kw):
        info, wi = kw["work_info_set"], kw["work_indptr"]
        lse = torch.full(q.shape[:2], -torch.inf)
        out.zero_()
        for row, _, start, end, lo, hi, *_ in info[: int(wi[-1])].tolist():
            assert start == row and end == row + 1
            if hi > lo:
                keys = kv.flatten(0, 2)[indices[lo:hi].long()].float()
                scores = q[row].float() @ keys.T * kw["sm_scale"]
                out[row] = (scores.softmax(-1) @ keys[:, : out.shape[-1]]).to(out.dtype)
                lse[row] = scores.logsumexp(-1)
        return out, lse

    ns.update(
        get_mla_metadata_v1=planner,
        get_mla_metadata_info_v1=sizes,
        decode_update_mla_metadata_v1=planner,
        patch_work_kv_ranges=patch,
        mla_decode_fwd=launch,
        calls=calls,
    )
    return ns


def setup(ns, rows=16, max_q=1, dtype=torch.bfloat16, device="cpu", sizes=None):
    builder = ns["Builder"].__new__(ns["Builder"])
    builder.__dict__.update(
        is_sparse=True,
        sparse_dcp_metadata_rebuild=True,
        block_size=1,
        persistent_num_heads=64,
        padded_num_attention_heads=16,
        dtype_q=dtype,
        dtype_kv=dtype,
        index_topk_out=8,
        kv_shuffled=False,
    )
    if sizes is None:
        sizes = [
            (2, torch.int64),
            (257, torch.int32),
            ((rows + 256, 8), torch.int32),
            (rows + 1, torch.int32),
            ((rows, 2), torch.int32),
            (rows + 512, torch.int32),
        ]
    var = {
        prefix + name: torch.empty(shape, dtype=dt, device=device)
        for prefix in ("", "sparse_mtp_")
        for name, (shape, dt) in zip(WORK, sizes)
    }
    ptr = torch.arange(rows + 1, dtype=torch.int32, device=device)
    for name, value in {
        "cu_seqlens_q": ptr,
        "sparse_cu_seqlens_q": ptr,
        "kv_indptr": ptr * 16,
        "sparse_kv_indptr": ptr * 8,
        "kv_last_page_lens": torch.ones(rows, dtype=torch.int32, device=device),
        "sparse_kv_last_page_lens": torch.ones(rows, dtype=torch.int32, device=device),
    }.items():
        var[name] = SimpleNamespace(gpu=value)
    builder.model_runner = SimpleNamespace(forward_vars=var)
    md = SimpleNamespace(
        max_seqlen_q=max_q,
        causal=True,
        kv_indices=None,
        **{
            name: value.gpu
            for name, value in var.items()
            if isinstance(value, SimpleNamespace)
        },
    )
    result = (
        builder._set_mla_persistent_worker_buffers_sparse_mtp(rows)
        if max_q > 1
        else builder.set_mla_persistent_worker_buffers(rows, 1)
    )
    md.__dict__.update(result)
    layer = ns["Attention"].__new__(ns["Attention"])
    if isinstance(layer, torch.nn.Module):
        torch.nn.Module.__init__(layer)
    layer.__dict__.update(
        num_heads=64,
        kv_lora_rank=512 if device == "cuda" else 4,
        dtype=torch.bfloat16,
        use_seg_mla=False,
        use_triton_mla=False,
        kv_shuffled=False,
        is_sparse_mla=True,
        dcp_world_size=8,
        sparse_dcp_metadata_rebuild=True,
        owns_sparse_indexer=True,
        scale=0.25,
        _q_scale=torch.ones(1, device=device),
        _k_scale=torch.ones(1, device=device),
    )
    layer._pad_decode_query_heads = lambda x: x
    layer._restore_decode_query_heads = lambda x, _: x
    layer.decode_is_persistent = lambda: True
    return builder, md, layer


def selection(md, rows, seed, device="cpu"):
    lengths = (torch.arange(rows, device=device) + seed) % 5
    lengths[0] = 0
    md.dcp_sparse_kv_indptr = torch.cat((lengths.new_zeros(1), lengths.cumsum(0))).int()
    md.sparse_kv_indices = (
        (torch.arange(rows * 8, device=device) * 7 + seed) % 97
    ).int()
    return lengths


def reference(q, kv, md, rank, scale):
    out = torch.zeros((*q.shape[:2], rank), device=q.device)
    lse = torch.full(q.shape[:2], -torch.inf, device=q.device)
    ptr = md.dcp_sparse_kv_indptr.tolist()
    for row, (lo, hi) in enumerate(pairwise(ptr)):
        if hi > lo:
            keys = kv.float()[md.sparse_kv_indices[lo:hi].long()].reshape(
                -1, q.shape[-1]
            )
            scores = torch.einsum("hd,kd->hk", q[row].float(), keys) * scale
            lse[row] = torch.logsumexp(scores, -1)
            out[row] = torch.einsum(
                "hk,kd->hd", torch.exp(scores - lse[row, :, None]), keys[:, :rank]
            )
    return out, lse


@pytest.mark.parametrize("max_q,rows", [(1, 16), (8, 128), (8, 256)])
def test_three_owners_and_shared_layers_reuse_one_builder_plan(runtime, max_q, rows):
    _, md, layer = setup(runtime, rows, max_q)
    call = runtime["calls"][0]
    assert call[0] == 64 and torch.equal(call[1], torch.arange(rows + 1))
    assert call[2]["kv_granularity"] == 8 and call[2]["max_split_per_batch"] == 1
    torch.manual_seed(7)
    q, kv = torch.randn(rows + 2, 64, 8).bfloat16(), torch.randn(97, 1, 8).bfloat16()
    for owner in range(3):
        selection(md, rows, owner)
        for owns in (True, False):
            layer.owns_sparse_indexer = owns
            out, lse = layer._forward_decode(q, kv, md, return_lse=True)
            expected, expected_lse = reference(q[:rows], kv, md, 4, layer.scale)
            torch.testing.assert_close(
                out[:rows], expected.bfloat16(), atol=0.016, rtol=0.01
            )
            torch.testing.assert_close(lse[:rows], expected_lse)
    assert len(runtime["calls"]) == 1


@pytest.mark.parametrize(
    "change",
    ["folded", "unknown", "unsupported", "q_dtype", "kv_dtype", "heads", "rows"],
)
@pytest.mark.parametrize("max_q", [1, 8])
def test_fallback_preserves_native_template(runtime, change, max_q):
    _, md, layer = setup(runtime, 16, max_q)
    selection(md, 16, 1)
    prefix = "sparse_mtp_" if max_q > 1 else ""
    saved = [getattr(md, prefix + name).clone() for name in WORK]
    q, kv = torch.randn(16, 64, 8).bfloat16(), torch.randn(97, 1, 8).bfloat16()
    if change in ("folded", "unknown", "unsupported"):
        runtime["get_mla_decode_shape_support"] = lambda *a, **kw: SimpleNamespace(
            native=change != "folded",
            supported={"folded": True, "unknown": None, "unsupported": False}[change],
        )
    elif change == "heads":
        q = q[:, :32]
    elif change == "rows":
        q = q[:12]
    elif change == "q_dtype":
        q = q.float()
    else:
        kv = kv.float()
    layer._forward_decode(q, kv, md, return_lse=True)
    assert len(runtime["calls"]) == 2
    assert runtime["calls"][-1][0] == q.shape[1]
    assert runtime["calls"][-1][3] is not getattr(md, prefix + "work_info_set")
    for name, value in zip(WORK, saved):
        torch.testing.assert_close(getattr(md, prefix + name), value)
    runtime["get_mla_decode_shape_support"] = lambda *a, **kw: SimpleNamespace(
        native=True, supported=True
    )
    selection(md, 16, 3)
    layer.owns_sparse_indexer = False
    q, kv = torch.randn(16, 64, 8).bfloat16(), kv.bfloat16()
    out, lse = layer._forward_decode(q, kv, md, return_lse=True)
    expected, expected_lse = reference(q, kv, md, 4, layer.scale)
    torch.testing.assert_close(out, expected.bfloat16(), atol=0.016, rtol=0.01)
    torch.testing.assert_close(lse, expected_lse)
    assert len(runtime["calls"]) == 2


@pytest.mark.parametrize(
    "rows,groups,capacity",
    [(0, 256, 257), (257, 304, 305), (65, 64, 65), (16, 256, 513)],
)
def test_unqualified_capacity_keeps_layer_rebuild(runtime, rows, groups, capacity):
    builder, _, _ = setup(runtime, max(1, rows))
    runtime["get_mla_decode_fwd_max_splits"] = lambda *a: groups
    assert builder._sparse_dcp_decode_plan(rows, 64, torch.empty(capacity)) is None


@pytest.mark.parametrize(
    "native,supported", [(False, True), (True, None), (True, False)]
)
def test_unqualified_builder_keeps_existing_geometry(runtime, native, supported):
    runtime["get_mla_decode_shape_support"] = lambda *a, **kw: SimpleNamespace(
        native=native, supported=supported
    )
    _, md, layer = setup(runtime)
    assert md.sparse_dcp_decode_plan is None
    assert runtime["calls"][0][2]["kv_granularity"] == 16
    for owner in range(3):
        selection(md, 16, owner)
        layer._forward_decode(
            torch.randn(16, 64, 8).bfloat16(),
            torch.randn(97, 1, 8).bfloat16(),
            md,
            return_lse=True,
        )
    assert len(runtime["calls"]) == 4


def test_next_step_refreshes_rows_without_reusing_old_template(runtime):
    builder, _, _ = setup(runtime, 128, 8)
    next_step = builder._set_mla_persistent_worker_buffers_sparse_mtp(7)
    assert next_step["sparse_mtp_dcp_decode_plan"][0] == 7
    assert next_step["sparse_mtp_num_tokens"] == 7
    assert int(next_step["sparse_mtp_work_indptr"][-1]) == 7
    assert len(runtime["calls"]) == 2
    updated = builder.set_mla_persistent_worker_buffers(7, 1, only_update=True)
    assert updated["sparse_dcp_decode_plan"] is None
    assert runtime["calls"][-1][2]["kv_granularity"] == 16


def test_fallback_sizes_for_consumers_with_more_workgroups(runtime):
    builder, md, layer = setup(runtime)
    fp8 = torch.float8_e4m3fn
    runtime["get_mla_decode_fwd_max_splits"] = lambda heads, _, qdt, kdt: (
        512 if heads == 64 and qdt == kdt == fp8 else 256
    )
    builder.persistent_num_heads = 96
    builder.dtype_q = builder.dtype_kv = fp8
    md.__dict__.update(builder.set_mla_persistent_worker_buffers(16, 1))
    selection(md, 16, 2)
    original = md.work_info_set.clone()
    buffers = layer._sparse_dcp_decode_work_buffers(
        md,
        torch.empty(16, 64, 8, dtype=fp8),
        torch.empty(97, 1, 8, dtype=fp8),
        md.cu_seqlens_q,
        md.dcp_sparse_kv_indptr,
        md.sparse_kv_last_page_lens,
        "",
    )
    assert buffers[1].numel() == 513 and md.work_indptr.numel() == 257
    torch.testing.assert_close(md.work_info_set, original)


@pytest.mark.skipif(
    not torch.version.hip or not torch.cuda.is_available(), reason="requires ROCm GPU"
)
@pytest.mark.parametrize("max_q,rows", [(1, 16), (8, 128), (8, 256)])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float8_e4m3fn])
def test_native_decode_and_graph_against_direct_attention(
    max_q, rows, dtype, monkeypatch
):
    from aiter import get_mla_metadata_info_v1

    from atom.model_ops import attention_mla
    from atom.model_ops.attentions import aiter_mla

    sizes = get_mla_metadata_info_v1(rows, 1, 64, dtype, dtype, is_sparse=True)
    ns = {
        "Builder": aiter_mla.AiterMLAMetadataBuilder,
        "Attention": attention_mla.MLAAttention,
    }
    builds = []
    original = aiter_mla.get_mla_metadata_v1

    def build(*a, **kw):
        builds.append(1)
        return original(*a, **kw)

    monkeypatch.setattr(aiter_mla, "get_mla_metadata_v1", build)
    _, md, layer = setup(ns, rows, max_q, dtype, "cuda", sizes)
    plan = getattr(
        md, "sparse_mtp_dcp_decode_plan" if max_q > 1 else "sparse_dcp_decode_plan"
    )
    if plan is None:
        pytest.skip("runtime does not qualify this native plan geometry")
    prefix = "sparse_mtp_" if max_q > 1 else ""
    wi = getattr(md, prefix + "work_indptr")
    assert int(wi[-1]) == rows and int((wi[1:] - wi[:-1]).max()) == 1
    monkeypatch.setattr(
        attention_mla,
        "get_mla_metadata_v1",
        lambda *a, **kw: pytest.fail("layer rebuilt native plan"),
    )
    torch.manual_seed(92)
    q = (torch.randn(rows + 2, 64, 576, device="cuda") / 4).to(dtype)
    kv = (torch.randn(97, 1, 576, device="cuda") / 4).to(dtype)
    layer.scale = 576**-0.5
    selection(md, rows, 0, "cuda")
    layer._forward_decode(q, kv, md, return_lse=True)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        out, lse = layer._forward_decode(q, kv, md, return_lse=True)
    for owner in range(3):
        next_md = SimpleNamespace()
        selection(next_md, rows, owner, "cuda")
        md.dcp_sparse_kv_indptr.copy_(next_md.dcp_sparse_kv_indptr)
        md.sparse_kv_indices.copy_(next_md.sparse_kv_indices)
        graph.replay()
        expected, expected_lse = reference(q[:rows], kv, md, 512, layer.scale)
        # Raw empty-rank partials may be NaN; the DCP merge uses owned counts
        # to drop them. Check every live row, including those following empties.
        live = md.dcp_sparse_kv_indptr.diff() > 0
        torch.testing.assert_close(
            out[:rows][live].float(), expected[live], atol=0.015, rtol=0.03
        )
        torch.testing.assert_close(
            lse[:rows][live], expected_lse[live], atol=0.02, rtol=0.01
        )
    assert len(builds) == 1
