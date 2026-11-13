# SPDX-License-Identifier: MIT
"""Persistent attention producers through real H2D owners and GPU consumers."""

import os
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from atom.utils import CpuGpuBuffer
from atom.utils.forward_context import (
    AttentionMetaData,
    get_published_dcp_local_context_lens,
)
from atom.utils.h2d import BufferPublication, PublicationError
from atom.utils.tbo import cut_decode_ubatches, cut_prefill_ubatches

pytestmark = pytest.mark.skipif(
    os.environ.get("RUN_H2D_GPU_TESTS") != "1", reason="set RUN_H2D_GPU_TESTS=1"
)


def make_runner(
    monkeypatch,
    kind="mla",
    *,
    pp=1,
    q=1,
    dcp=1,
    sparse=False,
    fp4=False,
    kpool=1,
    tbo=True,
    base=None,
    bind=True,
):
    from atom.distributed import ulysses_sp
    from atom.model_engine.model_runner import ModelRunner
    from atom.model_ops.attentions import aiter_attention, aiter_mla, backends

    for module in (ulysses_sp, backends):
        monkeypatch.setattr(module, "attn_head_shard_size", lambda: 1)
    monkeypatch.setenv("ATOM_MLA_PAGE_SIZE", "1")
    monkeypatch.setenv("ATOM_USE_UNIFIED_ATTN", "0")
    monkeypatch.setenv("ATOM_USE_TRITON_MLA", "0")
    for module in (backends, aiter_attention, aiter_mla):
        if hasattr(module, "get_tp_group"):
            monkeypatch.setattr(
                module, "get_tp_group", lambda: SimpleNamespace(world_size=1)
            )
        if hasattr(module, "get_dcp_world_size"):
            monkeypatch.setattr(module, "get_dcp_world_size", lambda: dcp)
        if hasattr(module, "get_dcp_rank"):
            monkeypatch.setattr(module, "get_dcp_rank", lambda: dcp - 1)
    runner = ModelRunner.__new__(ModelRunner)
    runner.device = torch.device("cuda", 0)
    runner.max_bs = 4
    runner.max_num_batched_tokens = 64
    # the FP4 index cache's e8m0 swizzle describes 64-row blocks
    runner.block_size = 64 if fp4 else 16
    runner.kv_cache_dtype = "bf16"
    runner.num_spec_tokens = q - 1
    runner.has_mla_indexer = sparse
    runner.use_mrope = False
    runner.enforce_eager = True
    runner.arange_np = np.arange(64, dtype=np.int64)
    runner.config = SimpleNamespace(
        pipeline_parallel_size=pp,
        max_model_len=128,
        max_num_batched_tokens=runner.max_num_batched_tokens,
        enable_tbo=tbo,
        enable_tbo_decode=tbo,
        prefill_context_parallel_size=1,
        kv_cache_dtype="bf16",
        index_cache_dtype="fp4" if fp4 else "fp8",
        attn_prefill_chunk_size=0,
        compilation_config=SimpleNamespace(static_forward_context={}),
        speculative_config=SimpleNamespace(num_speculative_tokens=q - 1),
        hf_config=SimpleNamespace(
            num_attention_heads=16,
            num_key_value_heads=1,
            index_topk=8,
            index_kpool=kpool,
            index_head_dim=128,
            index_n_heads=32,
            indexer_compress_ratio=4,
            ngram_size=2,
            ple_layer_ids=[0],
            eos_token_id=2,
        ),
    )
    runner.tokenID_processor = SimpleNamespace(
        num_rejected=None, verify_ctx_lens=lambda ctx: ctx
    )
    runner.forward_vars = {
        name: CpuGpuBuffer(64, dtype=torch.int32, device=runner.device)
        for name in ("input_ids", "decode_src", "batch_info")
    }
    runner.forward_vars["positions"] = CpuGpuBuffer(
        64, dtype=torch.int64, device=runner.device, publication_group="positions"
    )
    runner.forward_vars["mtp_k"] = q - 1
    cls = base or (
        aiter_mla.AiterMLAMetadataBuilder
        if kind == "mla"
        else aiter_attention.AiterAttentionMetadataBuilder
    )
    builder = cls(model_runner=runner)
    # These tests execute the host producers and real CSR/compressed-slot
    # consumers. Attention worker scheduling is covered by its own suite.
    if kind == "mla":
        builder.set_mla_persistent_worker_buffers = lambda *a, **kw: {}
        builder._set_mla_persistent_worker_buffers_sparse_mtp = lambda *a, **kw: {}
    runner.attn_metadata_builder = builder
    runner._init_forward_vars_ring()
    if bind:
        runner._init_h2d_publication()
    return runner, builder


def decode_batch(count, q=1, step=0):
    lens = np.array([17 + 3 * i + step for i in range(count)], dtype=np.int32)
    return SimpleNamespace(
        req_ids=list(range(count)),
        total_seqs_num_decode=count,
        total_tokens_num_decode=count * q,
        total_seqs_num_prefill=0,
        total_tokens_num_prefill=0,
        total_seqs_num=count,
        total_tokens_num=count * q,
        context_lens=lens,
        num_scheduled_tokens=np.full(count, q, dtype=np.int32),
        block_tables=[
            np.asarray(
                [5 + 8 * i + j for j in range((int(n) + 15) // 16)], dtype=np.int32
            )
            for i, n in enumerate(lens)
        ],
        last_block_num_tokens=[(int(n) - 1) % 16 + 1 for n in lens],
        is_first_decode_without_local_prefill=[False] * count,
        is_dummy_run=False,
    )


def publish_query_prefix(builder, batch, running_bs, running_tokens):
    """`prepare_model`'s query-prefix producer, published on its own."""
    builder.publish_query_metadata(
        np.asarray(batch.num_scheduled_tokens[: batch.total_seqs_num], np.int32),
        running_bs,
        running_tokens,
    )


def start_decode(runner, builder, batch, q):
    runner._advance_forward_vars()
    runner._gate_staging_reuse()
    publish_query_prefix(builder, batch, 4, 4 * q)


def finish(runner):
    runner._mark_staging_h2d_enqueued()
    runner._record_forward_vars_event()


def stage_ubatch_decode(builder, batch, slices, q, attn_metadata=None):
    """`prepare_decode`'s micro-batch staging, over the step's decode spans."""
    return builder.stage_ubatch_decode(
        AttentionMetaData() if attn_metadata is None else attn_metadata,
        *builder.decode_spans(batch),
        slices,
        q,
    )


def assert_repeat_does_not_write(runner, call):
    before = {
        name: buf.cpu.clone()
        for name, buf in runner.forward_vars.items()
        if isinstance(buf, CpuGpuBuffer)
    }
    with pytest.raises(PublicationError, match="republish_reason"):
        call()
    for name, expected in before.items():
        assert torch.equal(runner.forward_vars[name].cpu, expected), name


def test_kpool_verify_publishes_the_mtp_layers_rows_a_token_behind(monkeypatch):
    """A pooled indexer's verify: the target's rows see the pools position p+1
    closes, the MTP layer's -- draft step 0, on the same metadata, its
    positions a token behind -- the pools p closes."""
    q = 3
    runner, builder = make_runner(
        monkeypatch, "mla", q=q, sparse=True, tbo=False, kpool=4
    )
    batch = decode_batch(2, q)
    start_decode(runner, builder, batch, q)
    md, _ = builder.prepare_decode(batch, 4, 4 * q, q)
    positions = np.concatenate([np.arange(n - q, n) for n in batch.context_lens])
    rows = len(positions)
    assert md.index_rows.row_ends[:rows].tolist() == ((positions + 1) // 4).tolist()
    assert md.mtp_index_rows.row_ends[:rows].tolist() == (positions // 4).tolist()
    finish(runner)


def test_sparse_mtp_verify_plans_only_the_real_tokens(monkeypatch):
    """A cudagraph-padded sparse MTP verify plans its per-token work over the
    scheduled tokens, and says so in the metadata. The MTP drafter's step 0
    reuses this plan with the target's UNPADDED stream (scheduled tokens rows);
    the persistent kernel reads q and writes o for every work item, so a plan
    over the padded count ran past both (#287)."""
    q = 3
    runner, builder = make_runner(monkeypatch, "mla", q=q, sparse=True, tbo=False)
    planned = []

    def plan(num_tokens, *a, **kw):
        planned.append(num_tokens)
        return {"sparse_mtp_num_tokens": num_tokens}

    builder._set_mla_persistent_worker_buffers_sparse_mtp = plan
    batch = decode_batch(2, q)
    start_decode(runner, builder, batch, q)
    attn_metadata, _ = builder.prepare_decode(batch, 4, 4 * q, q)
    assert planned == [2 * q]
    # What MLAAttention._forward_decode checks q's rows against.
    assert attn_metadata.sparse_mtp_num_tokens == 2 * q
    finish(runner)


@pytest.mark.parametrize("pp", [1, 2])
@pytest.mark.parametrize(
    "kind,q,dcp,sparse,tbo,fp4",
    [
        ("mha", 1, 1, False, True, False),
        ("mha", 3, 1, False, True, False),
        ("mla", 1, 1, False, True, False),
        ("mla", 3, 2, False, True, False),
        ("mla", 1, 2, True, True, False),
        ("mla", 3, 2, True, False, False),
        ("mla", 3, 1, True, True, True),
        ("mla", 1, 2, True, True, True),
    ],
)
def test_decode_csr_tbo_delayed_slot_reuse(
    monkeypatch, pp, kind, q, dcp, sparse, tbo, fp4
):
    runner, builder = make_runner(
        monkeypatch,
        kind,
        pp=pp,
        q=q,
        dcp=dcp,
        sparse=sparse,
        fp4=fp4,
        tbo=tbo,
    )
    direct = []
    copy = BufferPublication._copy

    def record_direct(binding, count):
        direct.append(binding.name)
        copy(binding, count)

    monkeypatch.setattr(BufferPublication, "_copy", record_direct)
    saved = []
    for step, count in enumerate((3, 1, 4, 2, 1, 3)):
        # A TBO runner's decode is a split one (`ForwardMode.decode_ubatch_slices`).
        slices = cut_decode_ubatches(4, 4 * q, np.full(count, q)) if tbo else None
        batch = decode_batch(count, q, step)
        start_decode(runner, builder, batch, q)
        torch.cuda._sleep(3_000_000)
        direct.clear()
        md, _ = builder.prepare_decode(batch, 4, 4 * q, q, slices)
        # The whole step's metadata is one upload, no buffer its own copy.
        assert direct == [], direct
        names = ["kv_indptr"]
        if kind == "mla":
            names.append("kv_last_page_lens")
        if dcp > 1:
            names.append("g_kv_indptr")
        if sparse:
            names.append("sparse_kv_indptr")
        if sparse and dcp > 1:
            names.append("dcp_local_context_lens")
        # DCP's decode scores inside its fused exchange, off no row ends.
        reads_row_ends = fp4 and dcp == 1
        if reads_row_ends:
            names.append("index_row_ends")
        for name in names:
            assert (
                runner.forward_vars[name]._publication._epoch == runner.h2d_owner.epoch
            )
        lengths = batch.context_lens
        local = (lengths + dcp - 1 - builder.dcp_rank) // dcp
        pages = (local + builder.block_size - 1) // builder.block_size
        expected = [0] + np.cumsum(pages).tolist() + [int(pages.sum())] * (4 - count)
        saved.append((md.kv_indptr.clone(), expected))
        # The CSR generator consumes the freshly published indptr/table on GPU.
        indices = []
        for table, n in zip(batch.block_tables, pages):
            indices.extend(
                table[j // builder.block_ratio] * builder.block_ratio
                + j % builder.block_ratio
                for j in range(int(n))
            )
        saved.append((md.kv_indices[: len(indices)].clone(), indices))
        if dcp > 1:
            expected_global = (
                [0] + np.cumsum(lengths).tolist() + [int(lengths.sum())] * (4 - count)
            )
            saved.append((md.g_kv_indptr.clone(), expected_global))
        if sparse and dcp > 1:
            expected_local = [
                (int(n) - q + j + 1 + dcp - 1 - builder.dcp_rank) // dcp
                for n in lengths
                for j in range(q)
            ] + [0] * ((4 - count) * q)
            saved.append((md.dcp_local_context_lens.clone(), expected_local))
        if reads_row_ends:
            # the FP4 indexer's bound: row j of a request sees context - q + j + 1
            expected_rows = [int(n) - q + j + 1 for n in lengths for j in range(q)]
            saved.append(
                (
                    runner.forward_vars["index_row_ends"].gpu[: 4 * q].clone(),
                    expected_rows + [0] * ((4 - count) * q),
                )
            )
        if sparse:
            effective = [min(int(n) - q + j + 1, 8) for n in lengths for j in range(q)]
            expected_sparse = (
                [0]
                + np.cumsum(effective).tolist()
                + [sum(effective)] * ((4 - count) * q)
            )
            saved.append(
                (
                    runner.forward_vars["sparse_kv_indptr"].gpu[: 4 * q + 1].clone(),
                    expected_sparse,
                )
            )
        if tbo:
            # Each ubatch holds its scheduled requests first, padded out to its
            # own running_bs: count=1 leaves ubatch 1 all padding.
            for ub, ub_slice in enumerate(slices):
                lo = ub_slice.request_slice.start
                n = ub_slice.request_slice.stop - lo
                p = f"ub{ub}_"
                width = ub_slice.running_bs + 1
                expected_ub = [
                    expected[lo + min(j, n)] - expected[lo] for j in range(width)
                ]
                saved.append(
                    (
                        runner.forward_vars[p + "kv_indptr"].gpu[:width].clone(),
                        expected_ub,
                    )
                )
                saved.append(
                    (
                        runner.forward_vars[p + "context_lens"].gpu[:n].clone(),
                        runner.forward_vars["context_lens"].np[lo : lo + n].tolist(),
                    )
                )
                saved.append(
                    (
                        runner.forward_vars[p + "cu_seqlens_q"].gpu[:width].clone(),
                        [min(j, n) * q for j in range(width)],
                    )
                )
            assert_repeat_does_not_write(
                runner,
                lambda batch=batch, slices=slices: stage_ubatch_decode(
                    builder, batch, slices, q
                ),
            )
        assert_repeat_does_not_write(
            runner,
            lambda count=count, step=step, slices=slices: builder.prepare_decode(
                decode_batch(count, q, step + 50), 4, 4 * q, q, slices
            ),
        )
        finish(runner)
    torch.cuda.synchronize()
    for actual, expected in saved:
        assert actual.cpu().tolist() == expected


@pytest.mark.parametrize("dcp, sparse", [(2, True), (1, True), (2, False)])
def test_a_capture_publishes_its_dcp_local_lens(monkeypatch, dcp, sparse):
    """A sparse DCP capture's per-token local lengths are its layout's one key
    a row, published with it; without sparse DCP there are none."""
    runner, builder = make_runner(monkeypatch, "mla", dcp=dcp, sparse=sparse, tbo=False)
    runner._advance_forward_vars()
    runner._gate_staging_reuse()
    builder.publish_query_metadata(np.ones(4, dtype=np.int32), 4, 4)
    md, _ = builder.build_for_cudagraph_capture(4)
    torch.cuda.synchronize()
    if dcp > 1 and sparse:
        assert md.dcp_local_context_lens.tolist() == [1] * 4
        assert get_published_dcp_local_context_lens(md, 4) is md.dcp_local_context_lens
    else:
        assert md.dcp_local_context_lens is None
    finish(runner)


@pytest.mark.parametrize("prefill_first", [False, True])
def test_a_tbo_capture_reads_its_layout_not_a_split_prefills(
    monkeypatch, prefill_first
):
    """A split prefill stages its micro-batches into the same `ub{i}_` rows a
    decode micro-batch reads -- per token where a decode's are per request. A
    capture that follows it (the warmup prefill, then the graphs) records over
    the capture layout, not the prefill's leftovers, which once walked the
    sparse decode off its selection."""
    from atom.model_ops.attentions import aiter_mla

    monkeypatch.setattr(aiter_mla, "get_mla_metadata_v1", lambda *a, **kw: None)
    runner, builder = make_runner(monkeypatch, "mla", sparse=True, tbo=True)
    if prefill_first:
        counts = np.asarray([20], dtype=np.int32)
        batch = decode_batch(1)
        batch.total_seqs_num_decode = batch.total_tokens_num_decode = 0
        batch.total_seqs_num_prefill = 1
        batch.total_tokens_num_prefill = batch.total_tokens_num = 20
        batch.num_scheduled_tokens = counts
        batch.context_lens = counts.copy()
        batch.num_cached_tokens = [0]
        runner._advance_forward_vars()
        runner._gate_staging_reuse()
        publish_query_prefix(builder, batch, 1, 20)
        slices = cut_prefill_ubatches(1, counts, 20)
        builder.prepare_prefill(batch, 1, 20, ubatch_slices=slices)
        finish(runner)
    runner._advance_forward_vars()
    runner._gate_staging_reuse()
    slices = cut_decode_ubatches(4, 4, np.ones(4, dtype=np.int32))
    builder.publish_query_metadata(np.ones(4, dtype=np.int32), 4, 4, slices)
    md, _ = builder.build_for_cudagraph_capture(4, decode_ubatch_slices=slices)
    torch.cuda.synchronize()
    for ub in md.ubatches:
        assert ub.sparse_kv_indptr.tolist() == [0, 1, 2]
        assert ub.kv_indptr.tolist() == [0, 1, 2]
        assert ub.context_lens.tolist() == [1, 1]
    finish(runner)


@pytest.mark.parametrize("kind", ["mla", "mha"])
def test_tbo_empty_batch_and_preflight_all_sources(monkeypatch, kind):
    runner, builder = make_runner(monkeypatch, kind)
    batch = decode_batch(0)
    slices = cut_decode_ubatches(4, 4, np.zeros(1))
    runner._gate_staging_reuse()
    md = AttentionMetaData()
    for group in stage_ubatch_decode(builder, batch, slices, 1, md):
        group.publish()
    builder.finish_ubatch_decode(md, batch.total_seqs_num_decode, slices, 1)
    finish(runner)
    torch.cuda.synchronize()
    for ub in range(2):
        var = runner.forward_vars
        assert var[f"ub{ub}_kv_indptr"].gpu[:3].tolist() == [0, 0, 0]
        assert var[f"ub{ub}_slot_mapping"].gpu[:2].tolist() == [-1, -1]
    runner._gate_staging_reuse()
    runner.forward_vars["ub1_kv_indptr"].copy_to_gpu(0)
    assert_repeat_does_not_write(
        runner,
        lambda: stage_ubatch_decode(builder, batch, slices, 1),
    )
    finish(runner)


@pytest.mark.parametrize("dcp,q", [(1, 1), (2, 3)])
def test_kimi_capture_has_one_publication_and_rejects_rewrite(monkeypatch, dcp, q):
    from atom.model_ops.attentions.kimi_mla_gdn_attn import (
        KimiAiterMLAGDNMetadataBuilder,
    )

    runner, builder = make_runner(monkeypatch, q=q, dcp=dcp, tbo=False)
    # Reuse production MLA allocation; Kimi's pool construction is unrelated.
    builder.__class__ = KimiAiterMLAGDNMetadataBuilder
    builder._build_gdn_capture_metadata = lambda bs: None
    runner.h2d_owner.begin()
    md, _ = builder.build_for_cudagraph_capture(3)
    saved = md.kv_indptr.clone()
    assert (
        runner.forward_vars["kv_indptr"]._publication._epoch == runner.h2d_owner.epoch
    )
    assert_repeat_does_not_write(runner, lambda: builder.build_for_cudagraph_capture(2))
    finish(runner)
    torch.cuda.synchronize()
    assert saved.tolist() == [0, 1, 2, 3]


@pytest.mark.parametrize("kind", ["mla", "mha"])
@pytest.mark.parametrize("pp", [1, 2])
@pytest.mark.parametrize("q,replay", [(1, False), (4, False), (4, True)])
def test_gdn_decode_publishes_slots_with_attention_metadata(
    monkeypatch, kind, pp, q, replay
):
    """Kimi (MLA + GDN) and Qwen3-Next (MHA + GDN): one upload a decode."""
    from atom.model_ops.attentions.gdn_attn import GDNAttentionMetadataBuilder
    from atom.model_ops.attentions.kimi_mla_gdn_attn import (
        KimiAiterMLAGDNMetadataBuilder,
    )

    monkeypatch.setenv("ATOM_ENABLE_REPLAYSSM", str(int(replay)))
    runner, builder = make_runner(monkeypatch, kind, pp=pp, q=q, tbo=False, bind=False)
    # Reuse production attention allocation; the hybrids' pools are unrelated.
    builder.__class__ = (
        KimiAiterMLAGDNMetadataBuilder if kind == "mla" else GDNAttentionMetadataBuilder
    )
    runner.drafter = SimpleNamespace(mtp_k=q - 1, runner=runner)
    runner.tokenID_processor.num_bonus = None
    runner.replayssm_write_pos = torch.full(
        (64,), -1, dtype=torch.int32, device=runner.device
    )
    builder._init_gdn_state(runner)
    runner._init_forward_vars_ring()
    runner._init_h2d_publication()
    assert {"spec_state_indices", "kv_indptr", "block_tables"} <= {
        b.name for b in runner.h2d_groups["decode_metadata"].members
    }

    direct = []
    copy = BufferPublication._copy

    def record_direct(binding, count):
        direct.append(binding.name)
        copy(binding, count)

    monkeypatch.setattr(BufferPublication, "_copy", record_direct)
    spec = q > 1
    saved = []
    for step, count in enumerate((3, 1, 4, 2)):
        batch = decode_batch(count, q, step)
        width = 1 if replay or not spec else q
        batch.state_slots = [
            [step + 3 * i + 1 + 7 * j for j in range(width)] for i in range(count)
        ]
        batch.state_fork_srcs = []
        start_decode(runner, builder, batch, q)
        torch.cuda._sleep(3_000_000)
        direct.clear()
        md, _ = builder.prepare_decode(batch, 4, 4 * q, q)
        assert direct == [], direct
        gdn = md.gdn_metadata
        var = runner.forward_vars
        firsts = [slots[0] for slots in batch.state_slots] + [-1] * (4 - count)
        if spec:
            rows = np.full((4, q), -1, dtype=np.int32)
            for i, slots in enumerate(batch.state_slots):
                rows[i] = 0
                rows[i, : len(slots)] = slots
            saved.append((var["spec_state_indices"].gpu[:4].clone(), rows.tolist()))
            assert gdn.spec_state_indices_tensor.data_ptr() == (
                var["spec_state_indices"].gpu.data_ptr()
            )
        else:
            saved.append((var["non_spec_state_indices_in"].gpu[:4].clone(), firsts))
        if replay or not spec:
            saved.append((var["non_spec_state_indices"].gpu[:4].clone(), firsts))
        finish(runner)
    torch.cuda.synchronize()
    for actual, expected in saved:
        assert actual.cpu().tolist() == expected


def make_qwen_runner(monkeypatch, pp=1, *, bind=True):
    from atom.model_ops.attentions.gdn_attn import (
        GDNAttentionMetadataBuilder,
        GDNStateMixin,
    )
    from atom.model_ops.attentions.qwen4_exp_attn import Qwen4ExpMetadataBuilder

    runner, _ = make_runner(monkeypatch, "mha", pp=1, tbo=False, bind=False)
    # The GDN state is not allocated here: publish none of it.
    monkeypatch.setattr(
        GDNStateMixin,
        "h2d_group_members",
        lambda self: super(GDNStateMixin, self).h2d_group_members(),
    )

    # Invoke the real Qwen allocation while using the already allocated base.
    def init(self, model_runner, **kwargs):
        self.model_runner = model_runner
        self.device = model_runner.device
        self.max_bs = model_runner.max_bs
        self.max_num_batched_tokens = model_runner.max_num_batched_tokens

    monkeypatch.setattr(GDNAttentionMetadataBuilder, "__init__", init)
    # Bind a fresh runner after all backend fields have been allocated.
    builder = Qwen4ExpMetadataBuilder(runner)
    runner.attn_metadata_builder = builder
    runner.config.pipeline_parallel_size = pp
    runner.ple_conv_state = torch.zeros(8, 2, device=runner.device)
    runner.ple_ngram_state = torch.zeros(8, 2, device=runner.device)
    runner._init_forward_vars_ring()
    if bind:
        runner._init_h2d_publication()
    return runner, builder


@pytest.mark.parametrize("pp", [1, 2])
def test_qsa_ple_ragged_padding_delayed_graph_consumer(monkeypatch, pp):
    runner, builder = make_qwen_runner(monkeypatch, pp)
    graphs, outputs = [], []
    # Isolated consumer graphs per slot; production PP still requires eager.
    for var in runner._fv_ring:
        out = torch.empty(12, dtype=torch.int64, device=runner.device)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            torch.add(
                var["qsa_logical_positions"].gpu[:12],
                var["qsa_token_to_req"].gpu[:12],
                out=out,
            )
        graphs.append(graph)
        outputs.append(out)
    saved = []
    for step, counts in enumerate(([3, 1, 4], [1], [], [4, 4, 4], [2, 3])):
        runner._advance_forward_vars()
        runner._gate_staging_reuse()
        var = runner.forward_vars
        n = sum(counts)
        positions = list(range(step, step + n))
        var["positions"].np[:n] = positions
        var["slot_mapping"].gpu[:12] = torch.arange(12, device=runner.device) + 64
        state_indices = torch.arange(
            len(counts), dtype=torch.int32, device=runner.device
        )
        md = SimpleNamespace(
            slot_mapping=var["slot_mapping"].gpu[:12],
            block_tables=var["block_tables"].gpu,
            context_lens=var["context_lens"].gpu,
            cu_seqlens_q=var["cu_seqlens_q"].gpu,
            max_seqlen_k=128,
            gdn_metadata=SimpleNamespace(
                spec_state_indices_tensor=None,
                non_spec_state_indices_tensor=state_indices,
                non_spec_state_indices_in_tensor=state_indices,
                num_accepted_tokens=None,
            ),
        )
        cached = [i % 2 for i in range(len(counts))]
        batch = SimpleNamespace(num_cached_tokens=cached)
        var["ple_has_initial_state"].gpu.fill_(True)
        torch.cuda._sleep(3_000_000)
        group = builder.model_runner.h2d_groups["prefill_metadata"]
        args = (12, np.asarray(counts, dtype=np.int64), n, group)
        builder._stage_qsa(*args)
        builder._stage_ple(batch, len(counts), True, group)
        group.publish()
        qsa = builder._finish_qsa(md, len(counts), 12)
        ple = builder._finish_ple(md, len(counts))
        graphs[runner._fv_idx].replay()
        ids = [i for i, count in enumerate(counts) for _ in range(count)]
        saved.append(
            (
                outputs[runner._fv_idx].clone(),
                [p + i for p, i in zip(positions, ids)] + [-2] * (12 - n),
            )
        )
        saved.append(
            (
                qsa.compressed_slot_mapping.clone(),
                [
                    (64 + i) // 4 if (p + 1) % 4 == 0 else -1
                    for i, p in enumerate(positions)
                ]
                + [-1] * (12 - n),
            )
        )
        saved.append((ple.has_initial_state.clone(), [bool(x) for x in cached]))
        saved.append(
            (
                var["ple_has_initial_state"].gpu[len(counts) :].clone(),
                [True] * (4 - len(counts)),
            )
        )
        assert_repeat_does_not_write(
            runner, lambda args=args: builder._stage_qsa(*args)
        )
        assert_repeat_does_not_write(
            runner,
            lambda batch=batch, counts=counts, group=group: builder._stage_ple(
                batch, len(counts), True, group
            ),
        )
        finish(runner)
    torch.cuda.synchronize()
    for actual, expected in saved:
        assert actual.cpu().tolist() == expected


def test_qsa_capture_rejected_before_raw_source_write(monkeypatch):
    runner, builder = make_qwen_runner(monkeypatch)
    runner.h2d_owner.begin()
    before = runner.forward_vars["qsa_token_to_req"].cpu.clone()
    # Preflight must fail before a host fill even for an empty publication.
    graph = torch.cuda.CUDAGraph()
    group = runner.h2d_groups["prefill_metadata"]
    with torch.cuda.graph(graph), pytest.raises(PublicationError, match="capture"):
        builder._stage_qsa(0, np.array([], dtype=np.int64), 0, group)
    assert torch.equal(before, runner.forward_vars["qsa_token_to_req"].cpu)
    finish(runner)


@pytest.mark.parametrize("pp", [1, 2])
@pytest.mark.parametrize("cached", [False, True])
def test_mla_sparse_prefill_prefixes_and_tail(monkeypatch, pp, cached):
    from atom.model_ops.attentions import aiter_mla

    runner, builder = make_runner(monkeypatch, pp=pp, sparse=True, tbo=False)
    monkeypatch.setattr(aiter_mla, "get_mla_metadata_v1", lambda *a, **kw: None)
    saved = []
    for step in range(4):
        counts = np.asarray([3, 2] if cached else [10 + step, 3], dtype=np.int32)
        prefix = np.asarray([9 + step, 5] if cached else [0, 0], dtype=np.int32)
        lens = counts + prefix
        tokens = int(counts.sum())
        batch = decode_batch(2)
        batch.total_seqs_num_decode = batch.total_tokens_num_decode = 0
        batch.total_seqs_num_prefill = 2
        batch.total_tokens_num_prefill = batch.total_tokens_num = tokens
        batch.num_scheduled_tokens = counts
        batch.context_lens = lens
        batch.num_cached_tokens = prefix.tolist()
        batch.last_block_num_tokens = ((lens - 1) % 16 + 1).tolist()
        runner._advance_forward_vars()
        runner._gate_staging_reuse()
        publish_query_prefix(builder, batch, 4, tokens)
        for name in (
            "cu_seqlen_ks",
            "sparse_kv_indptr",
            "kv_last_page_lens",
        ):
            runner.forward_vars[name].gpu.fill_(-99)
        torch.cuda._sleep(3_000_000)
        md, _ = builder.prepare_prefill(batch, 4, batch.total_tokens_num)
        starts, bids, selected = [], [], []
        base = 0
        for i, (n, old) in enumerate(zip(counts, prefix)):
            for j in range(int(n)):
                starts.append(base)
                bids.append(i)
                selected.append(min(int(old) + j + 1, 8))
            base += int(n + old)
        saved.extend(
            [
                (md.cu_seqlen_ks.clone(), starts),
                # Published with the query prefix; `build()` hands it over.
                (
                    runner.forward_vars["batch_id_per_q_token"].gpu[:tokens].clone(),
                    bids,
                ),
                (md.sparse_kv_indptr.clone(), [0] + np.cumsum(selected).tolist()),
                (
                    runner.forward_vars["cu_seqlen_ks"].gpu[tokens:].clone(),
                    [-99] * (64 - tokens),
                ),
            ]
        )
        if cached:
            saved.append(
                (
                    runner.forward_vars["kv_last_page_lens"].gpu.clone(),
                    batch.last_block_num_tokens + [0, 0],
                )
            )
        assert_repeat_does_not_write(
            runner,
            lambda batch=batch: builder.prepare_prefill(
                batch, 4, batch.total_tokens_num
            ),
        )
        finish(runner)
    torch.cuda.synchronize()
    for actual, expected in saved:
        assert actual.cpu().tolist() == expected


@pytest.mark.parametrize("producer", ["prefill", "mrope_prefill", "mrope_decode"])
def test_attention_reentry_preserves_sources_and_first_consumer(monkeypatch, producer):
    from tests.test_h2d_runner_publication import runner_with_buffers

    if producer == "prefill":
        runner, builder = make_runner(monkeypatch, "mha", tbo=False)

        def produce(changed):
            batch = decode_batch(2)
            batch.total_seqs_num_decode = batch.total_tokens_num_decode = 0
            batch.total_seqs_num_prefill = 2
            batch.total_tokens_num_prefill = batch.total_tokens_num = 5
            batch.num_scheduled_tokens = np.array([3, 2], dtype=np.int32)
            batch.context_lens = np.array([12, 7], dtype=np.int32) + changed * np.array(
                [1, -1], dtype=np.int32
            )
            batch.num_cached_tokens = (
                batch.context_lens - batch.num_scheduled_tokens
            ).tolist()
            return builder.prepare_prefill(batch, 4, batch.total_tokens_num)

        names = ("cu_seqlens_k", "context_lens", "positions", "slot_mapping")
        cu = runner.forward_vars["cu_seqlens_q"]
        cu.cpu[:5] = torch.tensor([0, 3, 5, 5, 5])
        cu.gpu.copy_(cu.cpu)
    else:
        runner = runner_with_buffers(monkeypatch)
        builder = runner.attn_metadata_builder

        def produce(changed):
            ends = np.array([12, 22], dtype=np.int32) + changed
            batch = SimpleNamespace(
                total_tokens_num_decode=4,
                total_tokens_num_prefill=4,
                num_scheduled_tokens=np.full(2, 2, dtype=np.int32),
                req_ids=[10, 20],
                context_lens=ends,
                num_cached_tokens=ends - 2,
                mrope_positions_by_req={},
                mrope_position_deltas={},
            )
            prefill = producer == "mrope_prefill"
            group = builder.model_runner.h2d_groups[
                "prefill_metadata" if prefill else "decode_metadata"
            ]
            if prefill:
                positions = builder._build_mrope_prefill_positions(batch, 8, group)
            else:
                positions = builder._build_mrope_decode_positions(
                    batch, ends, group, running_tokens=8
                )
            group.publish()
            return positions

        names = ("mrope_positions",)

    runner._gate_staging_reuse()
    produce(0)
    runner._mark_staging_h2d_enqueued()
    torch.cuda.synchronize()
    sources = [runner.forward_vars[name] for name in names]
    expected = [buf.gpu.clone() for buf in sources]
    observed = [torch.empty_like(value) for value in expected]
    runner._gate_staging_reuse()
    torch.cuda._sleep(20_000_000)
    produce(0)
    host_before = [buf.cpu.clone() for buf in sources]
    for out, buf in zip(observed, sources):
        out.copy_(buf.gpu)
    try:
        with pytest.raises(PublicationError, match="republish_reason"):
            produce(-1)
    finally:
        runner._mark_staging_h2d_enqueued()
        torch.cuda.synchronize()
    for name, buf, before, actual, reference in zip(
        names, sources, host_before, observed, expected
    ):
        assert torch.equal(buf.cpu, before), name
        assert torch.equal(actual, reference), name


@pytest.mark.parametrize("kind", ["mha", "mla"])
@pytest.mark.parametrize("pp", [1, 2])
def test_shared_maps_skip_h2d_per_slot_and_tbo_buffer(monkeypatch, kind, pp):
    from atom.model_engine.sequence import new_block_table

    runner, builder = make_runner(monkeypatch, kind, pp=pp, tbo=True)
    rows = [new_block_table(row) for row in decode_batch(3).block_tables]
    published = {}
    observed = []
    for step in range(9):
        if step == 4:
            rows = list(reversed(rows))
        if step == 7:
            rows[0].append(31)
        batch = decode_batch(3, step=step % 4)
        batch.block_tables = rows
        start_decode(runner, builder, batch, 1)
        torch.cuda._sleep(2_000_000)
        builder.prepare_decode(
            batch, 4, 4, 1, cut_decode_ubatches(4, 4, np.ones(3, np.int32))
        )
        # Two requests a half: ubatch 0 holds the first two, ubatch 1 the
        # third and a padding request.
        for name, selected, width in (
            ("block_tables", rows, 4),
            ("ub0_block_tables", rows[0:2], 2),
            ("ub1_block_tables", rows[2:3], 2),
        ):
            buf = runner.forward_vars[name]
            mapping = tuple(tuple(row) for row in selected)
            key = (runner._fv_idx, name)
            changed = published.get(key) != mapping
            assert (buf._publication._epoch == runner.h2d_owner.epoch) == changed
            published[key] = mapping
            expected = torch.zeros_like(buf.cpu[:width])
            for i, row in enumerate(selected):
                expected[i, : len(row)] = torch.tensor(list(row), dtype=torch.int32)
            observed.append((buf.gpu[:width].clone(), expected))
        finish(runner)
    torch.cuda.synchronize()
    for actual, expected in observed:
        assert torch.equal(actual.cpu(), expected)


@pytest.mark.parametrize("has_drafter", [False, True])
def test_a_cold_prefill_uploads_the_block_tables_a_drafter_reads(has_drafter):
    """A drafter's steps after a prefill read the block tables as a decode
    does: with no cached prefix the prefill must still upload them, or the
    draft writes its KV through the previous step's pages."""
    from atom.model_ops.attentions import backends

    runner = SimpleNamespace(drafter=object()) if has_drafter else SimpleNamespace()
    stub = SimpleNamespace(model_runner=runner)
    rows = backends.CommonAttentionBuilder._prefill_block_table_rows(
        stub, None, 4, has_cached=False
    )
    assert rows == (4 if has_drafter else None)
