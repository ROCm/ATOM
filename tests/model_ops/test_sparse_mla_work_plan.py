# SPDX-License-Identifier: MIT
"""A DCP sparse MLA work plan, made once a prefill forward or decode step and
patched per layer, against the per-layer rebuild it replaces, on aiter's
persistent MLA kernel."""

from itertools import pairwise

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
        model_runner=SimpleNamespace(forward_vars=work_buffers("sparse_prefill_")),
    )
    AiterMLAMetadataBuilder._sparse_prefill_work_plan(builder, meta, "", heads, last)

    monkeypatch.setenv("ATOM_MLA_PAGE_SIZE", "1")
    monkeypatch.setattr(
        attention_mla,
        "get_forward_context",
        lambda: SimpleNamespace(context=SimpleNamespace(scheduled_bs=rows)),
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
        "sparse_dcp_metadata_rebuild": True,
        "dcp_persistent_supported": True,
        "dcp_sparse_prefill_nonps": False,
        "owns_sparse_indexer": True,
        "sparse_prefill_head_groups": None,
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


@pytest.mark.parametrize("rows", [64, 257])
@pytest.mark.parametrize("cache_dtype", ["bf16", "fp8"])
def test_patch_stops_at_the_plans_last_item(cache_dtype, rows):
    """A PCP replan's dummy query has no keys, and the planner gives such a
    row no work item when it lands on a nearly full CU. The patch rewrites
    the plan's items only, not one a row past its end."""
    from types import SimpleNamespace

    from aiter import dtypes, get_mla_metadata_info_v1

    from atom.model_ops.attentions.aiter_mla import AiterMLAMetadataBuilder
    from atom.model_ops.attentions.sparse_mla_head_groups import (
        PERSISTENT_WORK_BUFFERS,
    )
    from atom.model_ops.sparse_mla_work_plan import patch_work_kv_ranges

    dtype = dtypes.fp8 if cache_dtype == "fp8" else torch.bfloat16
    heads, topk = 96, 2048
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


@pytest.mark.parametrize("q_len", [8, 1])
@pytest.mark.parametrize("cache_dtype", ["bf16", "fp8"])
def test_patched_decode_step_plan_matches_per_layer_rebuild(
    monkeypatch, cache_dtype, q_len
):
    """A decode step's plan, made once by the builder and patched by each
    owning layer's `_forward_decode`, against the plan each layer used to
    rebuild: a verify step's sparse_mtp_ plan (q_len 8) or a plain decode
    step's (q_len 1). Rows owning no keys get an empty work item, which the
    DCP merge drops by owned count; they must not disturb the other rows."""
    from types import SimpleNamespace

    from aiter import dtypes, get_mla_metadata_info_v1, get_mla_metadata_v1
    from aiter.mla import mla_decode_fwd

    from atom.model_ops import attention_mla
    from atom.model_ops.attention_mla import MLAAttention
    from atom.model_ops.attentions.aiter_mla import AiterMLAMetadataBuilder
    from atom.model_ops.attentions.sparse_mla_head_groups import (
        PERSISTENT_WORK_BUFFERS,
    )
    from atom.utils.forward_context import AttentionMetaData

    fp8 = cache_dtype == "fp8"
    dtype = dtypes.fp8 if fp8 else torch.bfloat16
    # Kimi-K3.1 at DCP8: 16 requests verifying 8 tokens, 96 gathered heads.
    rows, heads, topk, slots, scale = 128, 96, 2048, 1 << 17, 576**-0.5
    gen = torch.Generator(device="cuda").manual_seed(518)
    q = (torch.randn(rows, heads, 576, generator=gen, device="cuda") / 2).to(dtype)
    kv = (torch.randn(slots, 1, 576, generator=gen, device="cuda") / 2).to(dtype)
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

    # Past index_topk tokens of context every query selects topk keys.
    selected = indptr(torch.full((rows,), topk, dtype=torch.int32, device="cuda"))
    prefix = "sparse_mtp_" if q_len > 1 else ""
    builder = SimpleNamespace(
        dcp_world_size=8,
        block_size=1,
        index_topk_out=topk,
        is_sparse=True,
        kv_shuffled=False,
        sparse_mtp_num_heads=heads,
        persistent_num_heads=heads,
        dtype_q=dtype,
        dtype_kv=dtype,
        model_runner=SimpleNamespace(
            forward_vars=work_buffers(prefix)
            | {
                "cu_seqlens_q": SimpleNamespace(gpu=qptr),
                "sparse_cu_seqlens_q": SimpleNamespace(gpu=qptr),
                "sparse_kv_indptr": SimpleNamespace(gpu=selected),
                "sparse_kv_last_page_lens": SimpleNamespace(gpu=last),
            }
        ),
    )
    meta = AttentionMetaData(max_seqlen_q=q_len)
    if q_len > 1:
        plan = AiterMLAMetadataBuilder._set_mla_persistent_worker_buffers_sparse_mtp(
            builder, rows
        )
    else:
        plan = AiterMLAMetadataBuilder.set_mla_persistent_worker_buffers(
            builder, rows, 1
        )
    for name, value in plan.items():
        setattr(meta, name, value)
    work_indptr = getattr(meta, prefix + "work_indptr")
    assert int(work_indptr[-1]) == rows, "not one work item a row"
    meta.cu_seqlens_q = qptr
    meta.sparse_cu_seqlens_q = qptr
    meta.sparse_kv_indptr = selected
    meta.sparse_kv_last_page_lens = last

    monkeypatch.setenv("ATOM_MLA_PAGE_SIZE", "1")
    monkeypatch.setattr(
        attention_mla, "get_dp_group", lambda: SimpleNamespace(world_size=1)
    )
    attn = MLAAttention.__new__(MLAAttention)
    torch.nn.Module.__init__(attn)
    for name, value in {
        "is_sparse_mla": True,
        "dcp_world_size": 8,
        "dcp_rank": 0,
        "dcp_head_pad": 0,
        "use_seg_mla": False,
        "use_triton_mla": False,
        "kv_shuffled": False,
        "kv_lora_rank": 512,
        "qk_rope_head_dim": 64,
        "dtype": torch.bfloat16,
        "_q_scale": unit if fp8 else None,
        "_k_scale": unit if fp8 else None,
        "scale": scale,
        "_dpa_persistent_supported": True,
        "dcp_persistent_supported": True,
        "sparse_dcp_metadata_rebuild": True,
        "owns_sparse_indexer": True,
    }.items():
        setattr(attn, name, value)

    for _layer in range(2):
        owned = torch.randint(0, 512, (rows,), generator=gen, device="cuda").int()
        owned[:12] = torch.tensor([0, 0, 0, 1, 2, 3, 5, 15, 17, topk, topk, 0])
        kvptr = indptr(owned)
        # Sized for the step's candidates, so a plan still holding their
        # ranges reads wrong keys rather than past the end.
        indices = torch.randint(
            0, slots, (int(selected[-1]),), generator=gen, device="cuda"
        ).int()
        meta.dcp_sparse_kv_indptr = kvptr
        meta.sparse_kv_indices = indices
        out, lse = attn._forward_decode(q, kv, meta, return_lse=True)

        # What each layer used to plan: its own owned ranges, rows split.
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
            max_seqlen_qo=1,
            uni_seqlen_qo=1,
            fast_mode=1,
            max_split_per_batch=-1,
        )
        expected = torch.empty(rows, heads, 512, dtype=torch.bfloat16, device="cuda")
        _, expected_lse = mla_decode_fwd(
            q,
            kv.view(-1, 1, 1, 576),
            expected,
            qptr,
            kvptr,
            indices,
            last,
            1,
            page_size=1,
            num_kv_splits=16,
            sm_scale=scale,
            q_scale=unit if fp8 else None,
            kv_scale=unit if fp8 else None,
            return_lse=True,
            **work,
        )
        live = owned > 0
        torch.testing.assert_close(out[live], expected[live], rtol=0.02, atol=0.02)
        torch.testing.assert_close(lse[live], expected_lse[live], rtol=0, atol=1e-4)


@pytest.mark.parametrize("q_len", [8, 1])
@pytest.mark.parametrize("cache_dtype", ["bf16", "fp8"])
def test_decode_step_past_the_cus_runs_every_live_row(monkeypatch, cache_dtype, q_len):
    """Two rows a CU, each under 8 tokens of context: at DCP8 a rank owns one
    key of such a row or none, so the step's plan puts empty work items ahead
    of live ones on a CU. Every row owning keys must match an fp32 reference,
    through a full layer, a shared one, and a full one over other keys."""
    from types import SimpleNamespace

    from aiter import dtypes, get_mla_metadata_info_v1

    from atom.model_ops import attention_mla
    from atom.model_ops.attention_mla import MLAAttention
    from atom.model_ops.attentions.aiter_mla import AiterMLAMetadataBuilder
    from atom.model_ops.attentions.sparse_mla_head_groups import (
        PERSISTENT_WORK_BUFFERS,
    )
    from atom.utils.forward_context import AttentionMetaData

    fp8 = cache_dtype == "fp8"
    dtype = dtypes.fp8 if fp8 else torch.bfloat16
    rows = 2 * torch.cuda.get_device_properties().multi_processor_count
    heads, topk, slots, scale = 96, 2048, 1 << 12, 576**-0.5
    gen = torch.Generator(device="cuda").manual_seed(570)
    q = (torch.randn(rows, heads, 576, generator=gen, device="cuda") / 2).to(dtype)
    kv = (torch.randn(slots, 1, 576, generator=gen, device="cuda") / 2).to(dtype)
    qptr = torch.arange(rows + 1, dtype=torch.int32, device="cuda")
    last = torch.ones(rows, dtype=torch.int32, device="cuda")
    unit = torch.ones(1, dtype=torch.float32, device="cuda")
    sizes = get_mla_metadata_info_v1(
        rows, 1, heads, dtype, dtype, is_sparse=True, fast_mode=True
    )

    def indptr(counts):
        return torch.cat([counts.new_zeros(1), counts.cumsum(0)]).int()

    context = torch.randint(1, 8, (rows,), generator=gen, device="cuda").int()
    selected = indptr(context)
    prefix = "sparse_mtp_" if q_len > 1 else ""
    builder = SimpleNamespace(
        dcp_world_size=8,
        block_size=1,
        index_topk_out=topk,
        is_sparse=True,
        kv_shuffled=False,
        sparse_mtp_num_heads=heads,
        persistent_num_heads=heads,
        dtype_q=dtype,
        dtype_kv=dtype,
        model_runner=SimpleNamespace(
            forward_vars={
                prefix + name: torch.empty(shape, dtype=dt, device="cuda")
                for name, (shape, dt) in zip(PERSISTENT_WORK_BUFFERS, sizes)
            }
            | {
                "cu_seqlens_q": SimpleNamespace(gpu=qptr),
                "sparse_cu_seqlens_q": SimpleNamespace(gpu=qptr),
                "sparse_kv_indptr": SimpleNamespace(gpu=selected),
                "sparse_kv_last_page_lens": SimpleNamespace(gpu=last),
            }
        ),
    )
    meta = AttentionMetaData(max_seqlen_q=q_len)
    if q_len > 1:
        plan = AiterMLAMetadataBuilder._set_mla_persistent_worker_buffers_sparse_mtp(
            builder, rows
        )
    else:
        plan = AiterMLAMetadataBuilder.set_mla_persistent_worker_buffers(
            builder, rows, 1
        )
    for name, value in plan.items():
        setattr(meta, name, value)
    meta.cu_seqlens_q = qptr
    meta.sparse_cu_seqlens_q = qptr
    meta.sparse_kv_indptr = selected
    meta.sparse_kv_last_page_lens = last

    # Rank r of 8 owns position p of a row's selection when p % 8 == r.
    first_rank, second_rank = 3, 5
    work_indptr = getattr(meta, prefix + "work_indptr").tolist()
    item_rows = getattr(meta, prefix + "work_info_set")[: work_indptr[-1], 0].tolist()
    empty = (context <= first_rank).tolist()
    assert any(
        flags != sorted(flags)
        for flags in (
            [empty[row] for row in item_rows[a:b]] for a, b in pairwise(work_indptr)
        )
    ), "the plan puts no empty work item ahead of a live one"

    monkeypatch.setenv("ATOM_MLA_PAGE_SIZE", "1")
    monkeypatch.setattr(
        attention_mla, "get_dp_group", lambda: SimpleNamespace(world_size=1)
    )

    def layer(owns_indexer):
        attn = MLAAttention.__new__(MLAAttention)
        torch.nn.Module.__init__(attn)
        for name, value in {
            "is_sparse_mla": True,
            "dcp_world_size": 8,
            "dcp_rank": 0,
            "dcp_head_pad": 0,
            "use_seg_mla": False,
            "use_triton_mla": False,
            "kv_shuffled": False,
            "kv_lora_rank": 512,
            "qk_rope_head_dim": 64,
            "dtype": torch.bfloat16,
            "_q_scale": unit if fp8 else None,
            "_k_scale": unit if fp8 else None,
            "scale": scale,
            "_dpa_persistent_supported": True,
            "dcp_persistent_supported": True,
            "sparse_dcp_metadata_rebuild": True,
            "owns_sparse_indexer": owns_indexer,
        }.items():
            setattr(attn, name, value)
        return attn

    def reference(kvptr, indices):
        counts = (kvptr[1:] - kvptr[:-1]).long()
        cols = torch.arange(int(counts.max()), device="cuda")
        at = (kvptr[:-1, None] + cols).clamp(max=indices.numel() - 1)
        keys = kv.float().view(slots, 576)[indices[at].long()]
        s = torch.einsum("rhd,rld->rhl", q.float(), keys) * scale
        s = s.masked_fill(cols >= counts[:, None, None], float("-inf"))
        return torch.einsum(
            "rhl,rld->rhd", s.softmax(-1), keys[..., :512]
        ), s.logsumexp(-1)

    for owns_indexer, rank in (
        (True, first_rank),
        (False, first_rank),
        (True, second_rank),
    ):
        if owns_indexer:
            owned = (context > rank).int()
            kvptr = indptr(owned)
            # Sized for the step's candidates, so a plan still holding their
            # ranges reads wrong keys rather than past the end.
            indices = torch.randint(
                0, slots, (int(selected[-1]),), generator=gen, device="cuda"
            ).int()
            meta.dcp_sparse_kv_indptr = kvptr
            meta.sparse_kv_indices = indices
        out, lse = layer(owns_indexer)._forward_decode(q, kv, meta, return_lse=True)
        expected, expected_lse = reference(kvptr, indices)
        live = owned > 0
        torch.testing.assert_close(
            out[live].float(), expected[live], rtol=0.02, atol=0.02
        )
        torch.testing.assert_close(
            lse.view(rows, heads)[live], expected_lse[live], rtol=0, atol=1e-3
        )
