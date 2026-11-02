# SPDX-License-Identifier: MIT
"""Compose NoPE index selection, physical slots and MLA against a math oracle.

These tests cover the attention core with synthetic projected inputs, not the
full transformer or checkpoint projections. No torch.compile is involved.
"""

import pytest
import torch

pytestmark = pytest.mark.skipif(
    not torch.version.hip or not torch.cuda.is_available(), reason="requires a ROCm GPU"
)


@pytest.mark.parametrize("length", [1024, 2048, 4096, 32768, 131072])
def test_indexer_to_sparse_mla_matches_attention_reference(length):
    from aiter import (
        QuantType,
        dtypes,
        get_hip_quant,
        get_mla_metadata_info_v1,
        get_mla_metadata_v1,
        indexer_k_quant_and_cache,
    )
    from aiter.mla import mla_decode_fwd, mla_prefill_fwd

    from atom.model_ops.attention_mla import (
        triton_convert_req_index_to_global_index,
        triton_convert_req_index_to_global_index_dsa_prefill,
    )
    from atom.model_ops.indexer import IndexCache, fp8_rows, score_topk

    gen = torch.Generator(device="cuda").manual_seed(731)
    block, topk, rows = 128, 2048, 2
    heads, index_dim, mla_heads, mla_dim, value_dim = 32, 128, 16, 576, 512
    pages = (length + block - 1) // block

    def normal(*shape):
        return torch.randn(*shape, generator=gen, device="cuda").bfloat16()

    keys = normal(length, index_dim)
    query = normal(rows, heads, index_dim)
    weights = normal(rows, heads).float() / 64
    ends = torch.tensor([length - 37, length], dtype=torch.int32, device="cuda")
    table = torch.randperm(pages, generator=gen, device="cuda").int()[None]
    tokens = torch.arange(length, device="cuda")
    slots = table[0, tokens // block].long() * block + tokens % block

    cache = torch.zeros(pages, block, 144, dtype=torch.uint8, device="cuda")
    indexer_k_quant_and_cache(keys, cache, slots, index_dim, "ue8m0", preshuffle=True)
    q8, qs = get_hip_quant(QuantType.per_1x128)(
        query.reshape(-1, index_dim), quant_dtype=dtypes.fp8
    )
    q8 = q8.view_as(query)
    weights = weights * qs.reshape(rows, heads)
    native_scores = []
    selected = score_topk(
        q8,
        weights,
        IndexCache(cache, None, block),
        fp8_rows(ends, pages * block, None, table.expand(rows, -1).contiguous()),
        topk=topk,
        stable=True,
        on_band=lambda values, span: native_scores.append(values.clone()),
    )
    # Independent FP8 reconstruction, without the production cache gather.
    scale = 2.0 ** torch.ceil(
        torch.log2(
            keys.float().abs().amax(-1, keepdim=True).clamp_min(1e-4)
            / torch.finfo(dtypes.fp8).max
        )
    )
    ref_keys = ((keys.float() / scale).to(dtypes.fp8).float() * scale).cpu().float()
    ref_query = q8.cpu().float()

    scores = (
        torch.einsum("rhd,kd->rhk", ref_query, ref_keys).relu()
        * weights.cpu().float()[:, :, None]
    ).sum(1)
    # Head weights can cancel. Compare native scores with independent CPU FP32
    # reductions, then check top-k against native scores; near-tied candidates
    # may exchange order after FP32 rounding.
    native_scores = torch.cat(native_scores).cpu().float()
    u = torch.finfo(torch.float32).eps / 2
    operations = index_dim + heads + 3
    gamma = operations * u / (1 - operations * u)
    absolute_terms = torch.einsum("rhd,kd->rhk", ref_query.abs(), ref_keys.abs())
    bound = gamma * (absolute_terms * weights.cpu().float().abs()[:, :, None]).sum(1)
    for row, end in enumerate(ends.tolist()):
        error = (native_scores[row, :end] - scores[row, :end]).abs()
        assert (error <= bound[row, :end]).all()
        assert error.norm() / scores[row, :end].norm() < 1e-4
    selection_scores = native_scores
    counts = ends.clamp_max(topk)
    ptr = torch.cat(
        [torch.zeros(1, dtype=torch.int32, device="cuda"), counts.cumsum(0).int()]
    )
    qptr = torch.arange(rows + 1, dtype=torch.int32, device="cuda")
    expected_physical = []
    for row, end in enumerate(ends.tolist()):
        expected = torch.argsort(
            selection_scores[row, :end], descending=True, stable=True
        )[:topk]
        actual = selected[row, : min(end, topk)].long()
        assert torch.equal(actual.cpu().sort().values, expected.sort().values)
        assert (selected[row, min(end, topk) :] == -1).all()
        expected_physical.append(slots[actual])
    expected_physical = torch.cat(expected_physical).int()

    # Separate logical K/V from the shuffled physical cache used by the kernel.
    logical_kv = normal(length, mla_dim)
    kv = torch.full(
        (pages * block, 1, 1, mla_dim),
        float("nan"),
        dtype=torch.bfloat16,
        device="cuda",
    )
    kv[slots, 0, 0] = logical_kv
    mla_q = normal(rows, mla_heads, mla_dim)
    # 12 real heads per rank, padded to the assembly kernel's width of 16.
    # The padded head outputs are discarded.
    mla_q[:, 12:] = 0
    scale = 192**-0.5  # An unabsorbed qk dimension of 128 + 64.
    expected_out = []
    for row, end in enumerate(ends.tolist()):
        idx = torch.argsort(selection_scores[row, :end], descending=True, stable=True)[
            :topk
        ]
        # Canonical logical order makes the all-visible CPU reference
        # use the same FP32 reduction order as the dense reference.
        chosen = logical_kv.cpu()[idx.sort().values].float()
        probabilities = (mla_q[row].cpu().float() @ chosen.T * scale).softmax(-1)
        expected_out.append(probabilities @ chosen[:, :value_dim])
    expected_out = torch.stack(expected_out)

    last = torch.ones(rows, dtype=torch.int32, device="cuda")
    dense_ptr = torch.arange(rows + 1, dtype=torch.int32, device="cuda") * length
    dense_indices = slots.int().repeat(rows)
    for mode in ("prefill", "decode", "decode_persistent"):
        if mode == "prefill":
            indices = triton_convert_req_index_to_global_index_dsa_prefill(
                qptr,
                ptr,
                torch.zeros(rows, dtype=torch.int32, device="cuda"),
                selected,
                table,
                torch.tensor([0, length], dtype=torch.int32, device="cuda"),
                NUM_TOPK_TOKENS=topk,
                PAGE_SIZE=block,
            )
        else:
            indices = triton_convert_req_index_to_global_index(
                qptr, dense_ptr, ptr, dense_indices, selected, NUM_TOPK_TOKENS=topk
            )
        assert torch.equal(indices[: expected_physical.numel()], expected_physical)
        output = torch.empty(
            rows, mla_heads, value_dim, dtype=torch.bfloat16, device="cuda"
        )
        if mode == "prefill":
            mla_prefill_fwd(mla_q, kv, output, qptr, ptr, indices, last, 1, scale)
        else:
            metadata = {}
            if mode == "decode_persistent":
                names = (
                    "work_meta_data",
                    "work_indptr",
                    "work_info_set",
                    "reduce_indptr",
                    "reduce_final_map",
                    "reduce_partial_map",
                )
                info = get_mla_metadata_info_v1(
                    rows,
                    1,
                    mla_heads,
                    mla_q.dtype,
                    kv.dtype,
                    is_sparse=True,
                    fast_mode=True,
                    max_split_per_batch=-1,
                )
                metadata = {
                    name: torch.empty(shape, dtype=dtype, device="cuda")
                    for name, (shape, dtype) in zip(names, info)
                }
                get_mla_metadata_v1(
                    qptr,
                    ptr,
                    last,
                    mla_heads,
                    1,
                    True,
                    metadata["work_meta_data"],
                    metadata["work_info_set"],
                    metadata["work_indptr"],
                    metadata["reduce_indptr"],
                    metadata["reduce_final_map"],
                    metadata["reduce_partial_map"],
                    page_size=1,
                    kv_granularity=16,
                    max_seqlen_qo=1,
                    uni_seqlen_qo=1,
                    fast_mode=True,
                    max_split_per_batch=-1,
                    dtype_q=mla_q.dtype,
                    dtype_kv=kv.dtype,
                )
            mla_decode_fwd(
                mla_q,
                kv,
                output,
                qptr,
                ptr,
                indices,
                last,
                1,
                sm_scale=scale,
                num_kv_splits=16,
                **metadata,
            )
        # Includes BF16 output rounding and the assembly softmax/matmul error.
        torch.testing.assert_close(
            output.cpu().float(), expected_out, rtol=0.02, atol=0.002
        )
        assert (output.cpu().float() - expected_out).norm() / expected_out.norm() < 0.01

        if length <= topk:
            # Independent dense attention over all causally visible keys.
            dense = []
            for row, end in enumerate(ends.tolist()):
                visible = logical_kv[:end].cpu().float()
                p = (mla_q[row].cpu().float() @ visible.T * scale).softmax(-1)
                dense.append(p @ visible[:, :value_dim])
            torch.testing.assert_close(expected_out, torch.stack(dense), rtol=0, atol=0)
            # Also run the native kernel over the dense, chronological CSR.
            # Sparse score order and chronological order can round differently.
            dense_output = torch.empty_like(output)
            dense_kernel_indices = torch.cat(
                [slots[:end] for end in ends.tolist()]
            ).int()
            if mode == "prefill":
                mla_prefill_fwd(
                    mla_q,
                    kv,
                    dense_output,
                    qptr,
                    ptr,
                    dense_kernel_indices,
                    last,
                    1,
                    scale,
                )
            else:
                mla_decode_fwd(
                    mla_q,
                    kv,
                    dense_output,
                    qptr,
                    ptr,
                    dense_kernel_indices,
                    last,
                    1,
                    sm_scale=scale,
                    num_kv_splits=16,
                    **metadata,
                )
            torch.testing.assert_close(output, dense_output, rtol=0.02, atol=0.002)


@pytest.mark.parametrize("candidate_width", [2048, 2176])
def test_bf16_sparse_prefill_is_independent_of_batch_composition(candidate_width):
    """The same ordered candidate row must not acquire batch-dependent splits.

    Include the former 384+10 tail and candidates concentrated on one rank.
    The second width also covers a pooled indexer's padded output capacity.
    This exercises production metadata and native MLA, with a CPU FP32 oracle.
    """
    from itertools import accumulate
    from types import SimpleNamespace

    import triton
    from aiter import get_mla_metadata_info_v1
    from aiter.mla import mla_decode_fwd

    from atom.model_ops.attention_mla import MLAAttention

    lengths = [1, 63, 64, 127, 128, 255, 256, 394, 479, 511, 512, 1025, candidate_width]
    gen = torch.Generator(device="cuda").manual_seed(3165 + candidate_width)
    fixed = (
        torch.randn(len(lengths), 96, 576, generator=gen, device="cuda") * 0.125
    ).bfloat16()
    kv = (
        torch.randn(candidate_width, 1, 1, 576, generator=gen, device="cuda") * 0.125
    ).bfloat16()
    names = (
        "work_meta_data",
        "work_indptr",
        "work_info_set",
        "reduce_indptr",
        "reduce_final_map",
        "reduce_partial_map",
    )
    first = None
    impl = SimpleNamespace(
        _sparse_prefill_kv_granularity=triton.next_power_of_2(candidate_width)
    )
    for rows in (65, 97, 162, 481):
        repeats = (rows + len(lengths) - 1) // len(lengths)
        q = fixed.repeat(repeats, 1, 1)[:rows].contiguous()
        counts = (lengths * repeats)[:rows]
        kvptr = torch.tensor(
            [0] + list(accumulate(counts)), dtype=torch.int32, device="cuda"
        )
        indices = torch.cat(
            [torch.arange(count, dtype=torch.int32, device="cuda") for count in counts]
        )
        qptr = torch.arange(rows + 1, dtype=torch.int32, device="cuda")
        last = torch.ones(rows, dtype=torch.int32, device="cuda")
        sizes = get_mla_metadata_info_v1(
            rows,
            1,
            96,
            q.dtype,
            kv.dtype,
            is_sparse=True,
            fast_mode=True,
            max_split_per_batch=-1,
        )
        work = {
            name: torch.empty(shape, dtype=dtype, device="cuda")
            for name, (shape, dtype) in zip(names, sizes)
        }
        meta = SimpleNamespace(
            **{"sparse_prefill_" + name: value for name, value in work.items()}
        )
        MLAAttention._rebuild_sparse_dcp_persistent_metadata(
            impl, meta, q, kv, qptr, kvptr, last, 96, work_prefix="sparse_prefill_"
        )
        # Each live row has one whole interval, even at the smallest batch.
        info = work["work_info_set"][: int(work["work_indptr"][-1])].cpu()
        starts = kvptr.cpu().tolist()
        assert len(info) == rows
        for row, count in enumerate(counts):
            row_work = info[info[:, 0] == row]
            assert len(row_work) == 1
            assert row_work[0, 4:6].tolist() == [starts[row], starts[row] + count]
        output = torch.empty(rows, 96, 512, dtype=torch.bfloat16, device="cuda")
        mla_decode_fwd(
            q,
            kv,
            output,
            qptr,
            kvptr,
            indices,
            last,
            1,
            page_size=1,
            sm_scale=576**-0.5,
            **work,
        )
        actual = output[: len(lengths)].cpu()
        assert torch.isfinite(actual).all()
        if first is None:
            first = actual
            q_cpu, kv_cpu = fixed.cpu().float(), kv.cpu().float().reshape(-1, 576)
            for row, count in enumerate(lengths):
                reference = (q_cpu[row] @ kv_cpu[:count].T * (576**-0.5)).softmax(
                    -1
                ) @ kv_cpu[:count, :512]
                torch.testing.assert_close(
                    actual[row].float(), reference, rtol=0.02, atol=0.003
                )
        else:
            assert torch.equal(actual, first), (candidate_width, rows)


@pytest.mark.parametrize(
    "rows,independent", [(1, True), (7, True), (32, True), (33, True), (7, False)]
)
def test_small_prefill_head_groups_refresh_graph_metadata(
    monkeypatch, rows, independent
):
    """Exercise production dispatch when Q, CSR boundaries and dummy rows change.

    Compare against the original whole-row dispatch and check a CPU FP32
    reference. Include the automatic size boundary and IndexShare exclusion.
    """
    from itertools import accumulate
    from types import SimpleNamespace

    from aiter import get_mla_metadata_info_v1

    from atom.model_ops import attention_mla
    from atom.model_ops.attention_mla import MLAAttention
    from atom.model_ops.attentions.sparse_mla_head_groups import (
        MAX_HEAD_GROUP_ROWS,
        MLA_WORK_NAMES,
        SparsePrefillHeadGroups,
    )

    monkeypatch.setenv("ATOM_MLA_PAGE_SIZE", "1")
    monkeypatch.setattr(
        attention_mla,
        "get_forward_context",
        lambda: SimpleNamespace(context=SimpleNamespace(scheduled_bs=1)),
    )
    gen = torch.Generator().manual_seed(7341)
    q_cpu = (torch.randn(rows, 96, 576, generator=gen) * 0.125).bfloat16()
    kv_cpu = (torch.randn(2049, 1, 1, 576, generator=gen) * 0.125).bfloat16()
    kv_cpu[0].zero_()
    q, kv = q_cpu.cuda(), kv_cpu.cuda()
    ptr = torch.empty(rows + 1, dtype=torch.int32, device="cuda")
    indices = torch.empty(rows * 2048, dtype=torch.int32, device="cuda")

    def make_attn(allow_groups):
        attn = MLAAttention.__new__(MLAAttention)
        torch.nn.Module.__init__(attn)
        for name, value in {
            "is_sparse_mla": True,
            "dcp_world_size": 4,
            "use_seg_mla": False,
            "num_heads": 24,
            "kv_lora_rank": 512,
            "dtype": torch.bfloat16,
            "kv_cache_dtype": "auto",
            "scale": 576**-0.5,
            "dcp_sparse_prefill_num_heads": 128,
            "dcp_sparse_prefill_persistent": True,
            "sparse_dcp_metadata_rebuild": True,
            "dcp_persistent_supported": True,
            "dcp_sparse_prefill_nonps": False,
            "owns_sparse_indexer": True,
            # What the metadata builder binds when no layer shares top-k.
            "sparse_prefill_head_groups": (
                SparsePrefillHeadGroups("cuda") if allow_groups else None
            ),
            "_sparse_prefill_kv_granularity": 2048,
            "sparse_kv_indices_buffer": indices,
            "dcp_sparse_kv_indptr_buffer": ptr,
        }.items():
            setattr(attn, name, value)
        sizes = get_mla_metadata_info_v1(
            rows, 1, 128, q.dtype, kv.dtype, is_sparse=True, fast_mode=True
        )
        work = {
            f"sparse_prefill_{name}": torch.empty(shape, dtype=dtype, device="cuda")
            for name, (shape, dtype) in zip(MLA_WORK_NAMES, sizes)
        }
        meta = SimpleNamespace(
            cu_seqlens_q=torch.tensor([0, rows], dtype=torch.int32, device="cuda"),
            kv_indptr=None,
            kv_indices=None,
            kv_last_page_lens=None,
            max_seqlen_q=rows,
            sparse_cu_seqlens_q=torch.arange(
                rows + 1, dtype=torch.int32, device="cuda"
            ),
            sparse_kv_indptr=ptr,
            sparse_kv_last_page_lens=torch.ones(rows, dtype=torch.int32, device="cuda"),
            **work,
        )
        return attn, meta

    candidate, meta = make_attn(independent)
    baseline, base_meta = make_attn(False)
    lengths_template = (1, 31, 33, 394, 1025, 2048, 0)

    def update(state):
        lengths = [
            lengths_template[(row + state) % len(lengths_template)]
            for row in range(rows)
        ]
        packed = [
            (
                torch.arange(1, length + 1, dtype=torch.int32).roll(state)
                if length
                else torch.zeros(1, dtype=torch.int32)
            )
            for length in lengths
        ]
        offsets = [0] + list(accumulate(max(length, 1) for length in lengths))
        ptr.copy_(torch.tensor(offsets, dtype=torch.int32))
        indices[: offsets[-1]].copy_(torch.cat(packed))
        q.copy_((q_cpu.float() * (1.0, 0.5, -1.0)[state]).bfloat16())
        return lengths, packed

    update(0)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            candidate._forward_prefill_mla(q, kv, meta, return_lse=True)
    torch.cuda.current_stream().wait_stream(stream)
    if independent:
        # The grouped path serves batches of at most MAX_HEAD_GROUP_ROWS rows.
        assert candidate.sparse_prefill_head_groups.applies(
            torch.empty(rows, 128, 576, dtype=q.dtype, device="cuda"),
            torch.empty(rows, 128, 512, dtype=q.dtype, device="cuda"),
            kv,
            2048,
        ) == (rows <= MAX_HEAD_GROUP_ROWS)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        captured = candidate._forward_prefill_mla(q, kv, meta, return_lse=True)
    for state in range(3):
        lengths, packed = update(state)
        expected = baseline._forward_prefill_mla(q, kv, base_meta, return_lse=True)
        eager = candidate._forward_prefill_mla(q, kv, meta, return_lse=True)
        graph.replay()
        torch.cuda.synchronize()
        for original, value, replay in zip(expected, eager, captured):
            assert torch.equal(value, original)
            assert torch.equal(replay, original)
        row = next((row for row, length in enumerate(lengths) if length > 0), None)
        if row is not None:
            query = q[row].cpu().float()
            values = kv_cpu.view(-1, 576)[packed[row].long()].float()
            reference = (query @ values.T * candidate.scale).softmax(-1) @ values[
                :, :512
            ]
            torch.testing.assert_close(
                captured[0][row].cpu().float(), reference, rtol=0.02, atol=0.003
            )
