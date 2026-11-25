# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2026 ATOM Project Contributors
"""Sparse decode's builder/consumer contract; CPU stubs stop at GPU operations."""

from __future__ import annotations

import ast
import base64
import json
import math
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
    "_persistent_work_buffer_sizes",
    "_active_work_indptr",
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
        "_mla_seg_meta_kwargs": dict,
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
        assert fast_mode is True and (is_sparse or max_q == 1)
        groups = ns["get_mla_decode_fwd_max_splits"](heads, max_q, dtype_q, dtype_kv)
        rows *= max_q if is_sparse else 1
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
            heads, kw["max_seqlen_qo"], kw["dtype_q"], kw["dtype_kv"]
        )
        assert wi.numel() >= groups + 1, "planner work-indptr allocation is too small"
        calls.append((heads, kvptr.clone(), kw, info, (wmd, wi, info, ri, rfm, rpm)))
        info.zero_()
        info[:rows, 0] = torch.arange(rows)
        info[:rows, 1] = -1
        info[:rows, 2] = torch.arange(rows)
        info[:rows, 3] = torch.arange(1, rows + 1)
        info[:rows, 4] = kvptr[:-1] - kvptr[0]
        info[:rows, 5] = kvptr[1:] - kvptr[0]
        wi[: groups + 1].copy_(torch.arange(groups + 1).clamp_max(rows))
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
                keys = kv.float().flatten(0, 2)[indices[lo:hi].long()]
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


def setup(
    ns,
    rows=16,
    max_q=1,
    dtype=torch.bfloat16,
    device="cpu",
    sizes=None,
    heads=64,
    allocation_q=1,
):
    builder = ns["Builder"].__new__(ns["Builder"])
    builder.__dict__.update(
        is_sparse=True,
        sparse_dcp_metadata_rebuild=True,
        block_size=1,
        persistent_num_heads=heads,
        max_bs=rows,
        padded_num_attention_heads=16,
        dtype_q=dtype,
        dtype_kv=dtype,
        index_topk_out=8,
        kv_shuffled=False,
    )
    if sizes is None:
        sizes = builder._persistent_work_buffer_sizes(allocation_q)
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
        num_heads=heads,
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


def native_fp8_probability_scale(support, q_dtype, kv_dtype):
    if (
        q_dtype == kv_dtype == torch.float8_e4m3fn
        and getattr(support, "arch", None) == "gfx950"
        and getattr(support, "backend", None) == "asm"
        and getattr(support, "asm_status", None) == "ok"
        and getattr(support, "native", None) is True
        and getattr(support, "supported", None) is True
        and getattr(support, "kernel_num_heads", None) == 64
        # AITER's query returns the generated map key: arch + symbol.
        and getattr(support, "asm_kernel", None)
        == "gfx950_ZN5aiter43mla_a8w8_qh16_qseqlen4_gqaratio16_lse_v3_psE"
    ):
        return 448.0
    return None


def reference(q, kv, md, rank, scale, *, fp8_probability_scale=None):
    if fp8_probability_scale is not None:
        assert q.dtype == kv.dtype == torch.float8_e4m3fn
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
            if fp8_probability_scale is None:
                out[row] = torch.einsum(
                    "hk,kd->hd", torch.exp(scores - lse[row, :, None]), keys[:, :rank]
                )
            else:
                # ROCm/aiter#6422: round scaled P before PV, retaining the FP32 Z.
                weights = torch.exp(scores - scores.amax(-1, keepdim=True))
                denominator = weights.sum(-1, keepdim=True)
                weights = (
                    (weights * fp8_probability_scale).to(torch.float8_e4m3fn).float()
                )
                value = torch.einsum("hk,kd->hd", weights, keys[:, :rank])
                out[row] = (
                    (value / denominator / fp8_probability_scale).bfloat16().float()
                )
    return out, lse


def test_fp8_reference_matches_saved_native_vectors():
    data = json.loads(
        Path(__file__).with_name("sparse_mla_fp8_contract.json").read_text()
    )

    def tensor(name, dtype, width):
        raw = bytearray(base64.b64decode(data[name]))
        return (
            torch.frombuffer(raw, dtype=torch.uint8).view(dtype).reshape(-1, 1, width)
        )

    q = tensor("q_fp8", torch.float8_e4m3fn, 576)
    kv = tensor("kv_fp8", torch.float8_e4m3fn, 576)
    native = tensor("native_bf16", torch.bfloat16, 512).float()
    md = SimpleNamespace(
        dcp_sparse_kv_indptr=torch.tensor(data["kv_indptr"], dtype=torch.int32),
        sparse_kv_indices=torch.tensor(data["kv_indices"], dtype=torch.int32),
    )
    original, old_lse = reference(q, kv, md, 512, data["scale"])
    failures = (~torch.isclose(native, original, atol=0.015, rtol=0.03)).nonzero()
    assert failures.tolist() == [
        [row, 0, dim] for row, dim in enumerate(data["failed_dims"])
    ]
    torch.testing.assert_close(
        original[torch.arange(3), 0, data["failed_dims"]],
        torch.tensor(data["gpu_reference"]),
        atol=2e-7,
        rtol=0,
    )
    corrected, lse = reference(q, kv, md, 512, data["scale"], fp8_probability_scale=448)
    torch.testing.assert_close(native, corrected, atol=0.015, rtol=0.03)
    torch.testing.assert_close(lse, old_lse, atol=0, rtol=0)
    assert torch.equal(corrected, corrected.bfloat16().float())


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float8_e4m3fn])
def test_probability_reference_keeps_default_attention(dtype):
    q = torch.tensor([[[0, 1]]], dtype=dtype)
    kv = torch.tensor([[[1, 0]], [[1, -0.5]]], dtype=dtype)
    md = SimpleNamespace(
        dcp_sparse_kv_indptr=torch.tensor([0, 2]),
        sparse_kv_indices=torch.tensor([0, 1]),
    )
    out, lse = reference(q, kv, md, 1, 1)
    torch.testing.assert_close(out, torch.ones(1, 1, 1))
    torch.testing.assert_close(lse, torch.tensor([[math.log1p(math.exp(-0.5))]]))


def test_probability_reference_uses_unrounded_denominator():
    q = torch.tensor([[[0, 1]]], dtype=torch.float8_e4m3fn)
    kv = torch.tensor([[[1, 0]], [[1, -0.5]]], dtype=torch.float8_e4m3fn)
    md = SimpleNamespace(
        dcp_sparse_kv_indptr=torch.tensor([0, 2]),
        sparse_kv_indices=torch.tensor([0, 1]),
    )
    out, lse = reference(q, kv, md, 1, 1, fp8_probability_scale=448)
    # 448*exp(-0.5) rounds to 256; the denominator still uses exp(-0.5).
    expected = torch.tensor((448 + 256) / 448 / (1 + math.exp(-0.5))).bfloat16().float()
    torch.testing.assert_close(out, expected.expand_as(out), atol=0, rtol=0)
    torch.testing.assert_close(lse, torch.tensor([[math.log1p(math.exp(-0.5))]]))


@pytest.mark.parametrize(
    "change",
    [
        {},
        {"arch": "gfx942"},
        {"backend": "hk"},
        {"backend": "opus"},
        {"backend": "flydsl_ps1"},
        {"native": False},
        {"supported": None},
        {"asm_status": "unknown"},
        {"kernel_num_heads": 16},
        {"asm_kernel": "_ZN5aiter43mla_a8w8_qh16_qseqlen4_gqaratio16_lse_v3_psE"},
        {"asm_kernel": None},
    ],
)
def test_native_probability_scale_is_route_specific(change):
    values = {
        "arch": "gfx950",
        "backend": "asm",
        "asm_status": "ok",
        "native": True,
        "supported": True,
        "kernel_num_heads": 64,
        "asm_kernel": "gfx950_ZN5aiter43mla_a8w8_qh16_qseqlen4_gqaratio16_lse_v3_psE",
    }
    values.update(change)
    support = SimpleNamespace(**values)
    fp8 = torch.float8_e4m3fn
    assert native_fp8_probability_scale(support, fp8, fp8) == (None if change else 448)
    assert native_fp8_probability_scale(support, torch.bfloat16, fp8) is None
    assert native_fp8_probability_scale(support, fp8, torch.bfloat16) is None
    assert native_fp8_probability_scale(SimpleNamespace(), fp8, fp8) is None


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
    [(0, 256, 257), (257, 304, 305), (65, 64, 65), (16, 512, 257)],
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
    assert all(call[3] is md.work_info_set for call in runtime["calls"])


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
    expected = runtime["get_mla_metadata_info_v1"](
        16, 1, 64, fp8, fp8, is_sparse=False, fast_mode=True
    )
    for buffer, (shape, dtype) in zip(buffers, expected, strict=True):
        assert tuple(buffer.shape) == (shape if isinstance(shape, tuple) else (shape,))
        assert buffer.dtype == dtype
    torch.testing.assert_close(md.work_info_set, original)


@pytest.mark.parametrize("heads,allocation_q", [(16, 4), (64, 8)])
def test_spec_allocation_covers_both_occupancies_and_publishes_active_prefix(
    runtime, heads, allocation_q
):
    runtime["get_mla_decode_fwd_max_splits"] = lambda h, q, *_: (
        512 if h * q == 64 else 256
    )
    dtype = torch.float8_e4m3fn
    builder, md, layer = setup(
        runtime, heads=heads, allocation_q=allocation_q, dtype=dtype
    )
    backing = builder.model_runner.forward_vars["work_indptr"]
    assert backing.numel() == 513
    backing.fill_(-12345)
    md.__dict__.update(builder.set_mla_persistent_worker_buffers(16, 1))
    groups = runtime["get_mla_decode_fwd_max_splits"](heads, 1)
    assert md.sparse_dcp_decode_plan is not None
    assert md.work_indptr.numel() == groups + 1
    assert int(md.work_indptr[-1]) == 16
    assert torch.all(backing[groups + 1 :] == -12345)
    selection(md, 16, 2)
    layer._forward_decode(
        torch.randn(16, heads, 8).to(dtype),
        torch.randn(97, 1, 8).to(dtype),
        md,
        return_lse=True,
    )
    assert len(runtime["calls"]) == 2


@pytest.mark.parametrize("heads,actual_q", [(64, 1), (64, 8), (16, 4), (16, 8)])
def test_every_planner_publishes_active_workers(runtime, heads, actual_q):
    runtime["get_mla_decode_fwd_max_splits"] = lambda h, q, *_: (
        512 if h * q == 64 else 256
    )
    fp8 = torch.float8_e4m3fn
    builder, _, _ = setup(runtime, heads=heads, allocation_q=8, dtype=fp8)
    var = builder.model_runner.forward_vars
    # Every configured width must fit all buffers, including an interior peak.
    for q_len in range(1, 9):
        sizes = runtime["get_mla_metadata_info_v1"](
            16, q_len, heads, fp8, fp8, is_sparse=True, fast_mode=True
        )
        for name, (shape, dtype) in zip(WORK, sizes, strict=True):
            dims = shape if isinstance(shape, tuple) else (shape,)
            assert all(a >= b for a, b in zip(var[name].shape, dims, strict=True))
            assert var[name].dtype == dtype
    var["cu_seqlens_q"].gpu = torch.arange(17, dtype=torch.int32) * actual_q
    backing = var["work_indptr"]
    backing.fill_(-12345)
    result = builder.set_mla_persistent_worker_buffers(16, actual_q)
    groups = runtime["get_mla_decode_fwd_max_splits"](heads, actual_q)
    assert (
        result["work_indptr"].numel() == groups + 1
    ), "unwritten capacity launches workers"
    assert torch.all(backing[groups + 1 :] == -12345)
    for name, buffer in zip(WORK, runtime["calls"][-1][4], strict=True):
        assert buffer is result[name] and buffer.data_ptr() == var[name].data_ptr()
    assert (result["sparse_dcp_decode_plan"] is not None) == (actual_q == 1)

    def update(*args, **kwargs):
        # The incremental updater reads the terminal entry of the original plan.
        assert kwargs["max_seqlen_qo"] == actual_q
        assert args[8].numel() == groups + 1 and int(args[8][-1]) == 16

    runtime["decode_update_mla_metadata_v1"] = update
    updated = builder.set_mla_persistent_worker_buffers(16, actual_q, only_update=True)
    assert updated["work_indptr"].numel() == groups + 1


@pytest.mark.parametrize(
    "native,supported", [(False, True), (True, None), (True, False)]
)
def test_unqualified_mtp_still_publishes_active_workers(runtime, native, supported):
    runtime["get_mla_decode_fwd_max_splits"] = lambda h, q, *_: (
        512 if h * q == 64 else 256
    )
    runtime["get_mla_decode_shape_support"] = lambda *a, **kw: SimpleNamespace(
        native=native, supported=supported
    )
    builder, _, _ = setup(runtime, heads=16, allocation_q=4, max_q=8)
    backing = builder.model_runner.forward_vars["sparse_mtp_work_indptr"]
    backing.fill_(-12345)
    result = builder._set_mla_persistent_worker_buffers_sparse_mtp(16)
    assert result["sparse_mtp_dcp_decode_plan"] is None
    assert result["sparse_mtp_work_indptr"].numel() == 257
    assert torch.all(backing[257:] == -12345)


def test_decode_ubatch_publishes_the_plan(runtime):
    builder, _, _ = setup(runtime)
    var = builder.model_runner.forward_vars
    var.update({"ub0_" + name: value for name, value in tuple(var.items())})
    path = Path(__file__).parents[2] / "atom/model_ops/attentions/aiter_mla.py"
    method = next(
        n
        for n in ast.walk(ast.parse(path.read_text()))
        if isinstance(n, ast.FunctionDef) and n.name == "finish_ubatch"
    )
    cls = ast.ClassDef(
        name="Child",
        bases=[ast.Name(id="Parent", ctx=ast.Load())],
        keywords=[],
        body=[method],
        decorator_list=[],
    )
    ns = {"Parent": type("Parent", (), {"finish_ubatch": lambda *a: None})}
    exec(  # noqa: S102 -- actual caller, with only its parent boundary stubbed
        compile(
            ast.fix_missing_locations(ast.Module(body=[cls], type_ignores=[])),
            str(path),
            "exec",
        ),
        ns,
    )
    child = ns["Child"]()
    child.model_runner = builder.model_runner
    child.set_mla_persistent_worker_buffers = builder.set_mla_persistent_worker_buffers
    child._decode_index_rows = lambda *a, **kw: None
    child._mtp_index_rows = lambda *a: None
    md = SimpleNamespace()
    child.finish_ubatch(0, SimpleNamespace(running_bs=16, running_tokens=16), md, 1)
    assert md.sparse_dcp_decode_plan is not None
    assert md.work_indptr.data_ptr() == var["ub0_work_indptr"].data_ptr()


@pytest.mark.skipif(
    not torch.version.hip or not torch.cuda.is_available(), reason="requires ROCm GPU"
)
@pytest.mark.parametrize("max_q,rows", [(1, 16), (8, 128), (8, 256)])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float8_e4m3fn])
def test_native_decode_and_graph_against_direct_attention(
    max_q, rows, dtype, monkeypatch
):
    from aiter import get_mla_decode_shape_support, get_mla_metadata_info_v1

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
    probability_scale = None
    if dtype == torch.float8_e4m3fn:
        support = get_mla_decode_shape_support(
            64, 1, dtype, dtype, persistent=True, return_lse=True, page_size=1
        )
        probability_scale = native_fp8_probability_scale(support, dtype, dtype)
        if probability_scale is not None:
            assert (getattr(md, prefix + "work_info_set")[:rows, 1] == -1).all()
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
        expected, expected_lse = reference(
            q[:rows], kv, md, 512, layer.scale, fp8_probability_scale=probability_scale
        )
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
