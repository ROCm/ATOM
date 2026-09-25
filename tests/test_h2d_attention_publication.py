# SPDX-License-Identifier: MIT
"""Persistent attention producers through real H2D owners and GPU consumers."""

import os
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from atom.utils import CpuGpuBuffer
from atom.utils.h2d import PublicationError

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
    tbo=True,
    base=None,
    bind=True,
    transport="direct",
):
    from atom.model_engine.model_runner import ModelRunner
    from atom.model_ops.attentions import aiter_attention, aiter_mla, backends

    monkeypatch.setenv("ATOM_H2D_BACKEND", transport)
    monkeypatch.setenv("ATOM_MLA_PAGE_SIZE", "1")
    monkeypatch.setenv("ATOM_USE_UNIFIED_ATTN", "0")
    monkeypatch.setenv("ATOM_USE_TRITON_MLA", "0")
    monkeypatch.setattr(backends, "tbo_enabled", lambda: tbo)
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
    runner.block_size = 16
    runner.kv_cache_dtype = "bf16"
    runner.num_spec_tokens = q - 1
    runner.has_mla_indexer = sparse
    runner.use_mrope = False
    runner.enforce_eager = True
    runner.arange_np = np.arange(64, dtype=np.int64)
    runner.config = SimpleNamespace(
        pipeline_parallel_size=pp,
        max_model_len=128,
        enable_tbo=tbo,
        enable_tbo_decode=tbo,
        kv_cache_dtype="bf16",
        index_cache_dtype="fp8",
        attn_prefill_chunk_size=0,
        compilation_config=SimpleNamespace(static_forward_context={}),
        speculative_config=SimpleNamespace(num_speculative_tokens=q - 1),
        hf_config=SimpleNamespace(
            num_attention_heads=16,
            num_key_value_heads=1,
            index_topk=8,
            index_kpool=1,
            indexer_compress_ratio=4,
            ngram_size=2,
            ple_layer_ids=[0],
            eos_token_id=2,
        ),
    )
    runner.tokenID_processor = SimpleNamespace(num_rejected=None)
    runner.forward_vars = {
        name: CpuGpuBuffer(64, dtype=torch.int32, device=runner.device)
        for name in ("input_ids", "decode_src")
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
        builder._set_ubatch_mla_buffers = lambda *a, **kw: None
        builder._publish_indexer_fp4_decode_schedule = lambda *a, **kw: None
        builder._publish_indexer_fp4_prefill_schedule = lambda *a, **kw: None
    runner.attn_metadata_builder = builder
    runner._init_forward_vars_ring()
    if bind:
        runner._init_h2d_publication()
    return runner, builder


def decode_batch(count, q=1, step=0):
    lens = np.array([17 + 3 * i + step for i in range(count)], dtype=np.int32)
    return SimpleNamespace(
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


def start_decode(runner, builder, batch, q):
    runner._advance_forward_vars()
    runner._gate_staging_reuse()
    builder.publish_cu_seqlens_q(batch, SimpleNamespace(running_bs=4))


def finish(runner):
    runner._mark_staging_h2d_enqueued()
    runner._record_forward_vars_event()


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


@pytest.mark.parametrize("transport", ["direct", "packed"])
@pytest.mark.parametrize("pp", [1, 2])
@pytest.mark.parametrize(
    "kind,q,dcp,sparse,tbo",
    [
        ("mha", 1, 1, False, True),
        ("mha", 3, 1, False, True),
        ("mla", 1, 1, False, True),
        ("mla", 3, 2, False, True),
        ("mla", 1, 2, True, True),
        ("mla", 3, 2, True, False),
    ],
)
def test_decode_csr_tbo_delayed_slot_reuse(
    monkeypatch, pp, kind, q, dcp, sparse, tbo, transport
):
    runner, builder = make_runner(
        monkeypatch,
        kind,
        pp=pp,
        q=q,
        dcp=dcp,
        sparse=sparse,
        tbo=tbo,
        transport=transport,
    )
    saved = []
    for step, count in enumerate((3, 1, 4, 2, 1, 3)):
        batch = decode_batch(count, q, step)
        start_decode(runner, builder, batch, q)
        torch.cuda._sleep(3_000_000)
        md, _ = builder.prepare_decode(batch, 4, 4 * q, q)
        names = ["kv_indptr"]
        if kind == "mla":
            names.append("kv_last_page_lens")
        if dcp > 1:
            names.append("g_kv_indptr")
        if sparse:
            names.extend(("sparse_kv_indptr", "dcp_local_context_lens"))
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
        if sparse:
            expected_local = [
                (int(n) - q + j + 1 + dcp - 1 - builder.dcp_rank) // dcp
                for n in lengths
                for j in range(q)
            ] + [0] * ((4 - count) * q)
            saved.append((md.dcp_local_context_lens.clone(), expected_local))
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
            for ub, lo in enumerate((0, 2)):
                n = max(0, min(count - lo, 2))
                p = f"ub{ub}_"
                expected_ub = [v - expected[lo] for v in expected[lo : lo + 3]]
                saved.append(
                    (runner.forward_vars[p + "kv_indptr"].gpu[:3].clone(), expected_ub)
                )
                saved.append(
                    (
                        runner.forward_vars[p + "cu_seqlens_q"].gpu[:3].clone(),
                        [0] + [min(j, n) * q for j in (1, 2)],
                    )
                )
            assert_repeat_does_not_write(
                runner,
                lambda count=count, lengths=lengths: builder._prepare_ubatch_decode(
                    count, 4, q, lengths
                ),
            )
        assert_repeat_does_not_write(
            runner,
            lambda count=count, step=step: builder.prepare_decode(
                decode_batch(count, q, step + 50), 4, 4 * q, q
            ),
        )
        finish(runner)
    torch.cuda.synchronize()
    for actual, expected in saved:
        assert actual.cpu().tolist() == expected


@pytest.mark.parametrize("kind", ["mla", "mha"])
def test_tbo_empty_batch_and_preflight_all_sources(monkeypatch, kind):
    runner, builder = make_runner(monkeypatch, kind)
    runner._gate_staging_reuse()
    builder._prepare_ubatch_decode(0, 4, 1, np.array([], dtype=np.int32))
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
        lambda: builder._prepare_ubatch_decode(0, 4, 1, np.array([], dtype=np.int32)),
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


def make_qwen_runner(monkeypatch, pp=1, *, bind=True):
    from atom.model_ops.attentions.gdn_attn import GDNAttentionMetadataBuilder
    from atom.model_ops.attentions.qwen4_exp_attn import Qwen4ExpMetadataBuilder

    runner, _ = make_runner(monkeypatch, "mha", pp=1, tbo=False, bind=False)

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
        args = (md, len(counts), 12, np.asarray(counts, dtype=np.int64), n)
        qsa = builder._build_qsa_metadata(*args)
        ple = builder._build_ple_metadata(batch, md, len(counts), is_prefill=True)
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
            runner, lambda args=args: builder._build_qsa_metadata(*args)
        )
        assert_repeat_does_not_write(
            runner,
            lambda batch=batch, md=md, counts=counts: builder._build_ple_metadata(
                batch, md, len(counts), is_prefill=True
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
    with torch.cuda.graph(graph), pytest.raises(PublicationError, match="capture"):
        builder._build_qsa_metadata(None, 0, 0, np.array([], dtype=np.int64), 0)
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
        builder.publish_cu_seqlens_q(batch, SimpleNamespace(running_bs=4))
        for name in (
            "cu_seqlen_ks",
            "cu_seqlen_ke",
            "sparse_kv_indptr",
            "kv_last_page_lens",
        ):
            runner.forward_vars[name].gpu.fill_(-99)
        torch.cuda._sleep(3_000_000)
        md, _ = builder.prepare_prefill(batch, 4)
        starts, ends, bids, selected = [], [], [], []
        base = 0
        for i, (n, old) in enumerate(zip(counts, prefix)):
            for j in range(int(n)):
                starts.append(base)
                ends.append(base + int(old) + j + 1)
                bids.append(i)
                selected.append(min(int(old) + j + 1, 8))
            base += int(n + old)
        saved.extend(
            [
                (md.cu_seqlen_ks.clone(), starts),
                (md.cu_seqlen_ke.clone(), ends),
                (md.batch_id_per_q_token.clone(), bids),
                (md.sparse_kv_indptr.clone(), [0] + np.cumsum(selected).tolist()),
                (
                    runner.forward_vars["cu_seqlen_ke"].gpu[tokens:].clone(),
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
            runner, lambda batch=batch: builder.prepare_prefill(batch, 4)
        )
        finish(runner)
    torch.cuda.synchronize()
    for actual, expected in saved:
        assert actual.cpu().tolist() == expected


@pytest.mark.parametrize("transport", ["direct", "packed"])
@pytest.mark.parametrize("producer", ["prefill", "mrope_prefill", "mrope_decode"])
def test_attention_reentry_preserves_sources_and_first_consumer(
    monkeypatch, transport, producer
):
    from tests.test_h2d_runner_publication import runner_with_buffers

    if producer == "prefill":
        runner, builder = make_runner(
            monkeypatch, "mha", tbo=False, transport=transport
        )

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
            return builder.prepare_prefill(batch, 4)

        names = ("cu_seqlens_k", "context_lens", "positions", "slot_mapping")
        cu = runner.forward_vars["cu_seqlens_q"]
        cu.cpu[:5] = torch.tensor([0, 3, 5, 5, 5])
        cu.gpu.copy_(cu.cpu)
    else:
        runner = runner_with_buffers(monkeypatch, transport)
        builder = runner.attn_metadata_builder

        def produce(changed):
            ends = np.array([12, 22], dtype=np.int32) + changed
            batch = SimpleNamespace(
                total_tokens_num_decode=4,
                total_tokens_num_prefill=4,
                req_ids=[10, 20],
                context_lens=ends,
                num_cached_tokens=ends - 2,
                mrope_positions_by_req={},
                mrope_position_deltas={},
            )
            if producer == "mrope_prefill":
                return builder._build_mrope_prefill_positions(batch)
            return builder._build_mrope_decode_positions(
                batch, ends, 2, running_tokens=8
            )

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
@pytest.mark.parametrize("transport", ["direct", "packed"])
@pytest.mark.parametrize("pp", [1, 2])
def test_shared_maps_skip_h2d_per_slot_and_tbo_buffer(monkeypatch, kind, transport, pp):
    from atom.model_engine.sequence import new_block_table

    runner, builder = make_runner(
        monkeypatch, kind, pp=pp, tbo=True, transport=transport
    )
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
        builder.prepare_decode(batch, 4, 4, 1)
        for name, selected, width in (
            ("block_tables", rows, 4),
            ("ub0_block_tables", rows[:2], 2),
            ("ub1_block_tables", rows[2:], 2),
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
