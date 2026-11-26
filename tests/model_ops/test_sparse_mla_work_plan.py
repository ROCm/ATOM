# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2026 ATOM Project Contributors
"""Native-plan reuse and folded-plan rebuild on aiter's persistent MLA kernel."""

import pytest
import torch

pytestmark = pytest.mark.skipif(
    not torch.version.hip or not torch.cuda.is_available(), reason="requires a ROCm GPU"
)


@pytest.mark.parametrize("rows", [300, 16384])
@pytest.mark.parametrize("cache_dtype", ["bf16", "fp8"])
def test_patched_forward_plan_matches_per_layer_rebuild(monkeypatch, cache_dtype, rows):
    from types import SimpleNamespace

    from aiter import dtypes, get_mla_metadata_info_v1, get_mla_metadata_v1
    from aiter.mla import mla_decode_fwd
    from aiter.ops.attention import get_mla_decode_shape_support

    from atom.model_ops import attention_mla
    from atom.model_ops.attention_mla import MLAAttention
    from atom.model_ops.attentions.aiter_mla import AiterMLAMetadataBuilder
    from atom.model_ops.attentions.sparse_mla_head_groups import (
        PERSISTENT_WORK_BUFFERS,
    )

    fp8 = cache_dtype == "fp8"
    dtype = dtypes.fp8 if fp8 else torch.bfloat16
    heads, topk, slots, scale = 96, 2048, 1 << 17, 576**-0.5
    gen = torch.Generator(device="cuda").manual_seed(500 + rows)
    q = (torch.randn(rows, heads, 576, generator=gen, device="cuda") / 2).to(dtype)
    kv = (torch.randn(slots, 1, 1, 576, generator=gen, device="cuda") / 2).to(dtype)
    qptr = torch.arange(rows + 1, dtype=torch.int32, device="cuda")
    last = torch.ones(rows, dtype=torch.int32, device="cuda")
    unit = torch.ones(1, dtype=torch.float32, device="cuda")
    sizes = get_mla_metadata_info_v1(
        rows, 1, heads, dtype, dtype, is_sparse=True, fast_mode=True
    )

    def work_buffers(prefix=""):
        return {
            prefix + name: torch.empty(shape, dtype=dt, device="cuda")
            for name, (shape, dt) in zip(PERSISTENT_WORK_BUFFERS, sizes)
        }

    def indptr(counts):
        return torch.cat([counts.new_zeros(1), counts.cumsum(0)]).int()

    # The forward's top-k counts; each layer owns a subset, at least the
    # packer's one dummy slot a row.
    selected = torch.randint(1, topk + 1, (rows,), generator=gen, device="cuda")
    selected[::5] = topk
    meta = SimpleNamespace(
        cu_seqlens_q=qptr,
        max_seqlen_q=1,
        kv_indptr=None,
        kv_indices=None,
        kv_last_page_lens=None,
        sparse_cu_seqlens_q=qptr,
        sparse_kv_indptr=indptr(selected),
        sparse_kv_last_page_lens=last,
    )
    builder = SimpleNamespace(
        dcp_sparse_prefill_nonps=False,
        dcp_world_size=8,
        block_size=1,
        dtype_q=dtype,
        dtype_kv=dtype,
        index_topk_out=topk,
        _sparse_prefill_kv_granularity=topk,
        model_runner=SimpleNamespace(forward_vars=work_buffers("sparse_prefill_")),
    )
    AiterMLAMetadataBuilder._sparse_prefill_work_plan(builder, meta, "", heads, last)

    monkeypatch.setenv("ATOM_MLA_PAGE_SIZE", "1")
    monkeypatch.setattr(
        attention_mla,
        "get_forward_context",
        lambda: SimpleNamespace(context=SimpleNamespace(scheduled_bs=rows)),
    )

    def no_layer_plan(*args, **kwargs):
        raise AssertionError("a prefill layer planned its own work")

    support = get_mla_decode_shape_support(
        heads, 1, dtype, dtype, persistent=True, return_lse=True, page_size=1
    )
    layer_plans = []

    def layer_plan(*args, **kwargs):
        layer_plans.append(args[1].clone())
        return get_mla_metadata_v1(*args, **kwargs)

    reuse = support.supported is True and support.native
    monkeypatch.setattr(
        attention_mla, "get_mla_metadata_v1", no_layer_plan if reuse else layer_plan
    )
    attn = MLAAttention.__new__(MLAAttention)
    torch.nn.Module.__init__(attn)
    for name, value in {
        "is_sparse_mla": True,
        "dcp_world_size": 8,
        "num_heads": heads // 8,
        "use_seg_mla": False,
        "kv_lora_rank": 512,
        "dtype": torch.bfloat16,
        "kv_cache_dtype": cache_dtype if fp8 else "auto",
        "_q_scale": unit,
        "_k_scale": unit,
        "scale": scale,
        "dcp_sparse_prefill_num_heads": heads,
        "dcp_sparse_prefill_persistent": True,
        "dcp_sparse_prefill_plan_native": reuse,
        "sparse_dcp_metadata_rebuild": True,
        "dcp_persistent_supported": True,
        "dcp_sparse_prefill_nonps": False,
        "owns_sparse_indexer": True,
        "sparse_prefill_head_groups": None,
        "_sparse_prefill_kv_granularity": topk,
    }.items():
        setattr(attn, name, value)

    for _layer in range(2):
        share = torch.rand(rows, generator=gen, device="cuda")
        owned = (selected * share).ceil().clamp(min=1).int()
        kvptr = indptr(owned)
        # Sized for the forward's candidates, so a plan still holding their
        # ranges reads wrong keys rather than past the end.
        indices = torch.randint(
            0, slots, (int(meta.sparse_kv_indptr[-1]),), generator=gen, device="cuda"
        ).int()
        meta.dcp_sparse_kv_indptr = kvptr
        meta.sparse_kv_indices = indices
        out, lse = attn._forward_prefill_mla(q, kv, meta, return_lse=True)

        # What each layer used to plan: its own owned ranges, BF16 rows whole.
        work = work_buffers()
        get_mla_metadata_v1(
            qptr,
            kvptr,
            last,
            heads,
            1,
            True,
            work["work_meta_data"],
            work["work_info_set"],
            work["work_indptr"],
            work["reduce_indptr"],
            work["reduce_final_map"],
            work["reduce_partial_map"],
            page_size=1,
            dtype_q=dtype,
            dtype_kv=dtype,
            kv_granularity=16 if fp8 else topk,
            max_seqlen_qo=1,
            uni_seqlen_qo=1,
            fast_mode=1,
            max_split_per_batch=-1,
        )
        expected = torch.empty(rows, heads, 512, dtype=torch.bfloat16, device="cuda")
        _, expected_lse = mla_decode_fwd(
            q,
            kv,
            expected,
            qptr,
            kvptr,
            indices,
            last,
            1,
            page_size=1,
            num_kv_splits=2,
            sm_scale=scale,
            q_scale=unit if fp8 else None,
            kv_scale=unit if fp8 else None,
            return_lse=True,
            **work,
        )
        if fp8:
            # The rebuild split FP8 rows over CUs; whole rows reduce in order.
            torch.testing.assert_close(out, expected, rtol=0.02, atol=0.02)
            torch.testing.assert_close(lse, expected_lse, rtol=0, atol=1e-4)
        else:
            assert torch.equal(out, expected)
            assert torch.equal(lse, expected_lse)

    assert len(layer_plans) == (0 if reuse else 2)


@pytest.mark.parametrize("rows", [64, 257])
@pytest.mark.parametrize("cache_dtype", ["bf16", "fp8"])
@pytest.mark.parametrize("heads", [16, 96])
def test_patch_stops_at_the_plans_last_item(cache_dtype, rows, heads):
    """A PCP replan's dummy query has no keys, and the planner gives such a
    row no work item when it lands on a nearly full CU. The patch rewrites
    the plan's items only, not one a row past its end."""
    from types import SimpleNamespace

    from aiter import dtypes, get_mla_metadata_info_v1
    from aiter.ops.attention import get_mla_decode_shape_support

    from atom.model_ops.attentions.aiter_mla import AiterMLAMetadataBuilder
    from atom.model_ops.attentions.sparse_mla_head_groups import (
        PERSISTENT_WORK_BUFFERS,
    )
    from atom.model_ops.sparse_mla_work_plan import patch_work_kv_ranges

    dtype = dtypes.fp8 if cache_dtype == "fp8" else torch.bfloat16
    support = get_mla_decode_shape_support(
        heads, 1, dtype, dtype, persistent=True, return_lse=True, page_size=1
    )
    if support.supported is not True or not support.native:
        pytest.skip("range patching requires a native, unfolded plan")
    topk = 2048
    gen = torch.Generator(device="cuda").manual_seed(500 + rows)
    sizes = get_mla_metadata_info_v1(
        rows, 1, heads, dtype, dtype, is_sparse=True, fast_mode=True
    )
    work = {
        "sparse_prefill_" + name: torch.empty(shape, dtype=dt, device="cuda")
        for name, (shape, dt) in zip(PERSISTENT_WORK_BUFFERS, sizes)
    }
    # Leftovers whose row lies before the indptr.
    work["sparse_prefill_work_info_set"].fill_(-1)
    selected = torch.full((rows,), topk, dtype=torch.int32, device="cuda")
    selected[-1] = 0

    def indptr(counts):
        return torch.cat([counts.new_zeros(1), counts.cumsum(0)]).int()

    meta = SimpleNamespace(
        sparse_cu_seqlens_q=torch.arange(rows + 1, dtype=torch.int32, device="cuda"),
        sparse_kv_indptr=indptr(selected),
    )
    builder = SimpleNamespace(
        dcp_sparse_prefill_nonps=False,
        dcp_world_size=8,
        block_size=1,
        dtype_q=dtype,
        dtype_kv=dtype,
        index_topk_out=topk,
        _sparse_prefill_kv_granularity=topk,
        model_runner=SimpleNamespace(forward_vars=work),
    )
    last = torch.ones(rows, dtype=torch.int32, device="cuda")
    AiterMLAMetadataBuilder._sparse_prefill_work_plan(builder, meta, "", heads, last)
    items = int(meta.sparse_prefill_work_indptr[-1])
    assert items == rows - 1, "the zero-key row kept its work item"

    info = meta.sparse_prefill_work_info_set
    leftovers = info[items:rows].clone()
    owned = (selected * torch.rand(rows, generator=gen, device="cuda")).ceil().int()
    kvptr = indptr(owned)
    patch_work_kv_ranges(info, meta.sparse_prefill_work_indptr, kvptr, rows)
    assert torch.equal(info[items:rows], leftovers)
    planned = info[:items, 0].long()
    assert torch.equal(info[:items, 4], kvptr[planned])
    assert torch.equal(info[:items, 5], kvptr[planned + 1])


def test_patch_rejects_a_work_info_width_it_does_not_write():
    """The patch writes MlaWorkInfo fields by offset, so a resized aiter
    struct fails before the kernel writes into the wrong fields."""
    from atom.model_ops.sparse_mla_work_plan import patch_work_kv_ranges

    info = torch.zeros(4, 9, dtype=torch.int32, device="cuda")
    indptr = torch.arange(5, dtype=torch.int32, device="cuda")
    with pytest.raises(AssertionError, match="MlaWorkInfo layout changed"):
        patch_work_kv_ranges(info, indptr, indptr, 4)
    assert not info.any()
