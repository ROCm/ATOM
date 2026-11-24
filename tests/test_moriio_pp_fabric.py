# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""MiniMax-M3.1 P/D from a PP4 x TP1 prefill into a TP4 decode over the fabric.

- Placement, byte for byte: each P stage's real pools (the M3 builder's NVFP4
  K/V and FP4 index, the DSpark draft on the last stage) are pushed into each
  of four TP4 decode ranks' real pools by `MoRIIOConnector._write_pages`, and
  every byte of each decode rank's blocks must be what a TP4 layout of the same
  KV holds there: KV head r of every layer, the whole index, head r of the
  draft, through the views the model reads, every destination byte written
  exactly once, nothing else touched. ATOM_MORIIO_VERIFY's digests must agree.
- Refusals before any write.
- Completion: a request is released on P only after every decode rank has
  every stage's report (RELEASE); decode is told only after every stage
  reported.

CPU only, on the aiter stand-ins, like `test_moriio_fabric.py`.
"""

from __future__ import annotations

import types
from functools import partial
from unittest import mock

import msgpack
import pytest
import torch
from aiter_stub import atom_under_stubs
from test_moriio_fabric import (
    BLOCK,
    M31_DRAFT_LAYERS,
    M31_HEAD_DIM,
    M31_INDEX_DIM,
    M31_KV_HEADS,
    M31_LAYERS,
    _DraftKVBinding,
    apply_batch,
    as_bytes,
    fabric_connector,
    handshake,
    kv_config,
)
from torch import nn

PARTITION = (17, 17, 17, 9)  # kv_balanced for M3.1 PP4 x TP1 with DSpark
D_TP = 4


@pytest.fixture(scope="module")
def stack():
    with atom_under_stubs() as load:
        aiter = load("aiter")
        aiter.dtypes.d_dtypes.update(
            fp8=torch.float8_e4m3fn, bf16=torch.bfloat16, fp32=torch.float32
        )
        yield types.SimpleNamespace(
            runner=load("atom.model_engine.model_runner"),
            attn=load("atom.model_ops.attentions.aiter_attention"),
            draft=load("atom.spec_decode.draft_kv"),
            conn=load("atom.kv_transfer.disaggregation.moriio.moriio_connector"),
            fabric=load("atom.kv_transfer.disaggregation.moriio.fabric"),
            common=load("atom.kv_transfer.disaggregation.moriio.moriio_common"),
            index_fp4=load("atom.model_ops.minimax_m3.index_fp4"),
            sparse_attn=load("atom.model_ops.minimax_m3.sparse_attn"),
        )


@pytest.fixture(autouse=True)
def _device():
    with mock.patch("torch.cuda.current_device", lambda: 0):
        yield


class _SparseAttention(nn.Module):
    """An M3.1 sparse attention layer as the builder walks it, on one rank
    holding ``heads`` KV heads."""

    base_attention = True
    use_mla = False

    def __init__(self, layer_num: int, heads: int):
        super().__init__()
        self.layer_num = layer_num
        self.num_kv_heads = heads
        self.head_dim = M31_HEAD_DIM
        self.impl = types.SimpleNamespace(
            is_indexed_sparse_attention=True, kv_format="nvfp4"
        )


def rank_pools(stack, layers, tp, draft, blocks, role):
    """One rank's pools: target ``layers`` at TP ``tp`` (all four KV heads at
    TP1), the DSpark draft if ``draft``. As `m31_runner`, generalised."""
    mr, attn, draft_kv = stack.runner, stack.attn, stack.draft
    hf = types.SimpleNamespace(
        text_config=types.SimpleNamespace(
            sparse_attention_config={"sparse_index_dim": M31_INDEX_DIM}
        )
    )
    draft_hf = types.SimpleNamespace(
        num_key_value_heads=M31_KV_HEADS, head_dim=M31_HEAD_DIM
    )
    runner = object.__new__(mr.ModelRunner)
    runner.config = types.SimpleNamespace(
        kv_transfer_config=kv_config(role),
        kv_cache_dtype="fp8",
        index_cache_dtype="fp4",
        hf_config=hf,
        speculative_config=(
            types.SimpleNamespace(num_speculative_tokens=7) if draft else None
        ),
    )
    runner.device = torch.device("cpu")
    runner.block_size = BLOCK
    runner.model = nn.ModuleList(
        _SparseAttention(i, M31_KV_HEADS // tp) for i in layers
    )
    runner.drafter = (
        types.SimpleNamespace(
            model=nn.ModuleList(
                _DraftKVBinding(M31_LAYERS + i) for i in range(M31_DRAFT_LAYERS)
            )
        )
        if draft
        else None
    )
    builder = object.__new__(attn.AiterAttentionMetadataBuilder)
    builder.model_runner = runner
    builder.block_size = BLOCK
    builder.block_ratio = 1
    builder._has_sparse_attention = True
    builder._pool_rows = None
    builder.kv_pools = {}
    runner.attn_metadata_builder = builder
    if draft:
        runner.draft_kv_builder = draft_kv.DraftKvBuilder(
            runner,
            partial(
                attn.AiterBackend.make_kv_pool,
                draft_hf,
                world_size=tp,
                kv_dtype=torch.float8_e4m3fn,
            ),
        )
    pools = []

    def cpu_fabric_pool(nbytes, device):
        pool = torch.zeros(nbytes, dtype=torch.uint8)
        stack.fabric._pool_range = (pool.data_ptr(), nbytes)
        pools.append(pool)
        return pool

    with mock.patch.object(stack.conn, "allocate_fabric_pool", cpu_fabric_pool):
        pool = runner._back_paged_pools(blocks)
    tt = builder.get_kv_transfer_tensors()
    if draft:
        tt.merge_pages(runner.draft_kv_builder.get_kv_transfer_tensors())
    tt.set_block_count(blocks)
    # `page_table` checks a rank's regions against the pool its process
    # allocated; each side here is a process of its own.
    return types.SimpleNamespace(
        runner=runner, tt=tt, pool=pool, pool_range=stack.fabric._pool_range
    )


def register(stack, conn, side):
    stack.fabric._pool_range = side.pool_range
    return conn._register_pages({}, side.tt, side.tt.num_blocks)


def stage_layers(stage, partition=PARTITION):
    start = sum(partition[:stage])
    return list(range(start, start + partition[stage]))


def stage_connector(stack, stage, side, partition=PARTITION, tp=1):
    conn = fabric_connector(stack, producer=True)
    conn.tp_size, conn.tp_rank = tp, 0
    conn.pp_size, conn.pp_rank = len(partition), stage
    first = sum(partition[:stage])
    last = stage == len(partition) - 1
    conn._stage_layers = (first, None if last else first + partition[stage])
    register(stack, conn, side)
    return conn


def decode_connector(stack, side, rank):
    conn = fabric_connector(stack, producer=False)
    conn.tp_rank = rank
    meta = register(stack, conn, side)
    return conn, meta


def fill(pool, seed):
    generator = torch.Generator().manual_seed(seed)
    pool.copy_(torch.randint(0, 256, pool.shape, generator=generator))


def push_request(rank, transfer_id=9):
    return {"transfer_id": transfer_id, "decode_tp_size": D_TP, "decode_tp_rank": rank}


def engine(rank):
    return f"decode:tp{rank}"


def target_views(pool, layer, payload, scale):
    """Every per-layer view of a target pool row, as the model binds them:
    K, V and their scales (heads on dim 1), then the index planes."""
    views = list(pool.kv_views(layer) + pool.scale_views(layer))
    index = [
        pool.field_view(field, layer, torch.uint8, (-1, BLOCK, width))
        for field, width in (("index", payload), ("index_scale", scale))
    ]
    return views, index


@pytest.fixture(scope="module")
def pd(stack):
    """Four P stages (TP1, ``PARTITION``) and four D ranks (TP4), random KV."""
    with mock.patch("torch.cuda.current_device", lambda: 0):
        stages = [
            rank_pools(stack, stage_layers(s), 1, s == 3, 7 + s, "kv_producer")
            for s in range(4)
        ]
        ranks = [
            rank_pools(stack, range(M31_LAYERS), D_TP, True, 12, "kv_consumer")
            for _ in range(D_TP)
        ]
    for i, side in enumerate(stages + ranks):
        fill(side.pool, i)
    return types.SimpleNamespace(stages=stages, ranks=ranks)


SRC, DST = [5, 6, 2, 0], [3, 4, 9, 1]


def push_all(stack, pd, verify=False, skip=()):
    """Every stage pushes every D rank; returns the written-byte counts per D
    rank and the stage connectors."""
    written = [torch.zeros(r.pool.numel(), dtype=torch.int32) for r in pd.ranks]
    consumers = [decode_connector(stack, r, i) for i, r in enumerate(pd.ranks)]
    producers = []
    with mock.patch("torch.cuda.current_device", lambda: 0):
        for s, side in enumerate(pd.stages):
            conn = stage_connector(stack, s, side)
            conn._verify_kv = verify
            if verify:
                conn._page_views = [p.view for p in side.tt.pages]
            producers.append(conn)
            for r, (_consumer, meta) in enumerate(consumers):
                if (s, r) in skip:
                    continue
                handshake(stack, conn, meta)
                conn.remote_moriio_metadata[engine(r)] = (
                    conn.remote_moriio_metadata.pop("decode")
                )
                request = push_request(r)
                conn.posted = []
                conn._write_pages(request, engine(r), SRC[: len(DST)], DST)
                assert len(conn.posted) == 1  # one batch per (stage, D rank)
                written[r] += apply_batch(conn.posted, side.pool, pd.ranks[r].pool)
                if verify:
                    consumers[r][0].stage_reports = getattr(
                        consumers[r][0], "stage_reports", []
                    ) + [(s, request["page_digest"])]
    return written, producers, consumers


class TestPlacement:
    def test_every_d_rank_holds_the_tp4_layout_of_the_same_kv(self, stack, pd):
        before = [r.pool.clone() for r in pd.ranks]
        written, _, _ = push_all(stack, pd)
        payload, scale = stack.index_fp4.fp4_index_row_bytes(M31_INDEX_DIM)
        for r, rank in enumerate(pd.ranks):
            d_pool = rank.runner.attn_metadata_builder.kv_pools
            [(_, d_kv)] = d_pool.items()
            for s, stage in enumerate(pd.stages):
                [(_, p_kv)] = stage.runner.attn_metadata_builder.kv_pools.items()
                for row, layer in enumerate(stage_layers(s)):
                    p_heads, p_index = target_views(p_kv, row, payload, scale)
                    d_heads, d_index = target_views(d_kv, layer, payload, scale)
                    for p, d in zip(p_heads, d_heads, strict=True):
                        # A TP4 rank's one KV head r of the TP1 stage's four.
                        assert d.shape[1] == 1 and p.shape[1] == M31_KV_HEADS
                        assert torch.equal(
                            as_bytes(d[DST][:, 0]), as_bytes(p[SRC][:, r])
                        ), (r, layer)
                    for p, d in zip(p_index, d_index, strict=True):
                        assert torch.equal(as_bytes(d[DST]), as_bytes(p[SRC]))
            # The draft through the page-16 SHUFFLE views DSpark reads and
            # writes it in (`_bound_cache`): eight 16-token pages a block.
            page16 = stack.sparse_attn.to_page16_shuffle
            src16 = [b * 8 + j for b in SRC for j in range(8)]
            dst16 = [b * 8 + j for b in DST for j in range(8)]
            p_draft = pd.stages[3].runner.draft_kv_builder.kv_pool
            d_draft = rank.runner.draft_kv_builder.kv_pool
            for layer in range(M31_DRAFT_LAYERS):
                for p, d in zip(
                    page16(*p_draft.kv_views(layer), *p_draft.scale_views(layer)),
                    page16(*d_draft.kv_views(layer), *d_draft.scale_views(layer)),
                    strict=True,
                ):
                    assert d.shape[1] == 1 and p.shape[1] == M31_KV_HEADS
                    assert torch.equal(
                        as_bytes(d[dst16][:, 0]), as_bytes(p[src16][:, r])
                    ), (r, layer)
            # Every byte of every region's destination blocks once, nothing else.
            expected = torch.zeros_like(written[r])
            base = rank.pool.data_ptr()
            for region in rank.tt.block_regions:
                off = region.base_addr - base
                for d in DST:
                    lo = off + d * region.unit_bytes
                    expected[lo : lo + region.unit_bytes] = 1
            assert torch.equal(written[r], expected)
            untouched = expected == 0
            assert torch.equal(rank.pool[untouched], before[r][untouched])

    def test_the_verify_digests_merge_over_stages_and_match(self, stack, pd, caplog):
        _, _, consumers = push_all(stack, pd, verify=True)
        for r, (consumer, _meta) in enumerate(consumers):
            prepare_consumer(consumer)
            consumer._verify_kv = True
            consumer._page_views = [p.view for p in pd.ranks[r].tt.pages]
            consumer._write_reports_expected["d"] = 4
            consumer._verify_page_dst["d"] = DST
            for s, digest in consumer.stage_reports:
                consumer._record_write_report("d", True, None, digest, stage=s)
            assert consumer._write_reports == {"d": True}
            with caplog.at_level("INFO", logger="atom"):
                caplog.clear()
                consumer._verify_page_landed("d")
            assert "match the producer's 4 stage(s)" in caplog.text, caplog.text

    def test_verify_names_a_stage_that_never_pushed(self, stack, pd, caplog):
        # Stage 2 skips rank 1: its regions there stay as they were.
        _, _, consumers = push_all(stack, pd, verify=True, skip={(2, 1)})
        consumer = consumers[1][0]
        prepare_consumer(consumer)
        consumer._verify_kv = True
        consumer._page_views = [p.view for p in pd.ranks[1].tt.pages]
        consumer._write_reports_expected["d"] = 3
        consumer._verify_page_dst["d"] = DST
        for i, (_s, digest) in enumerate(consumer.stage_reports):
            consumer._record_write_report("d", True, None, digest, stage=i)
        with caplog.at_level("INFO", logger="atom"):
            consumer._verify_page_landed("d")
        assert "were written by none of the 3 producer stages" in caplog.text

    def test_a_wrong_byte_is_caught_by_the_digest(self, stack, pd, caplog):
        _, _, consumers = push_all(stack, pd, verify=True)
        consumer = consumers[0][0]
        prepare_consumer(consumer)
        consumer._verify_kv = True
        consumer._page_views = [p.view for p in pd.ranks[0].tt.pages]
        consumer._write_reports_expected["d"] = 4
        consumer._verify_page_dst["d"] = DST
        for s, digest in consumer.stage_reports:
            consumer._record_write_report("d", True, None, digest, stage=s)
        pd.ranks[0].tt.pages[100].view[DST[0]].view(-1)[3] ^= 1
        with caplog.at_level("INFO", logger="atom"):
            consumer._verify_page_landed("d")
        assert "1 of 380 regions differ" in caplog.text, caplog.text


def prepare_consumer(conn):
    conn._write_reports_lock = __import__("threading").Lock()
    conn._write_reports_expected = {}
    conn._write_report_counts = {}
    conn._write_reports = {}
    conn._stage_digests = {}
    conn._write_page_digests = {}
    conn._write_state_digests = {}
    conn._verify_page_dst = {}


class TestRefusals:
    def setup(self, stack, layers, tp, draft_p, draft_d, stage, partition):
        p = rank_pools(stack, layers, tp, draft_p, 4, "kv_producer")
        d = rank_pools(stack, range(M31_LAYERS), D_TP, draft_d, 4, "kv_consumer")
        conn = stage_connector(stack, stage, p, partition, tp)
        _, meta = decode_connector(stack, d, 2)
        return conn, meta

    def refused(self, stack, conn, meta, match, tp=D_TP):
        handshake(stack, conn, meta)
        request = {"transfer_id": 9, "decode_tp_size": tp, "decode_tp_rank": 2}
        with pytest.raises(ValueError, match=match):
            conn._write_pages(request, "decode", [1], [1])
        assert conn.posted == []

    def test_a_tp2_stage_cannot_feed_tp4(self, stack):
        conn, meta = self.setup(stack, stage_layers(0), 2, False, True, 0, PARTITION)
        self.refused(stack, conn, meta, "only from TP1")

    def test_a_decode_without_the_draft(self, stack):
        conn, meta = self.setup(stack, stage_layers(3), 1, True, False, 3, PARTITION)
        self.refused(stack, conn, meta, "consumer has no unit")

    def test_a_last_stage_without_the_draft(self, stack):
        conn, meta = self.setup(stack, stage_layers(3), 1, False, True, 3, PARTITION)
        self.refused(stack, conn, meta, "not held here")

    def test_a_stage_split_unlike_its_layers(self, stack):
        # The stage builds layers 0-16 but believes it holds 0-17.
        conn, meta = self.setup(
            stack, stage_layers(0), 1, False, True, 0, (18, 17, 16, 9)
        )
        self.refused(stack, conn, meta, "not held here")

    def test_a_decode_that_names_no_units(self, stack):
        conn, meta = self.setup(stack, stage_layers(1), 1, False, True, 1, PARTITION)
        meta.page_units = None
        self.refused(stack, conn, meta, "names no unit")

    def test_another_index_format(self, stack):
        conn, meta = self.setup(stack, stage_layers(1), 1, False, True, 1, PARTITION)
        for unit in meta.page_units:
            if unit[1] == "index":
                unit[4] = "fp8"
        self.refused(stack, conn, meta, "unit format mismatch")

    def test_a_rdma_pp_producer_is_refused(self, stack):
        config = types.SimpleNamespace(
            kv_transfer_config={"kv_role": "kv_producer", "kv_connector": "moriio"},
            kv_cache_block_size=BLOCK,
            pipeline_parallel_size=4,
            parallel_config=types.SimpleNamespace(pipeline_parallel_rank=1),
            hf_config=types.SimpleNamespace(num_hidden_layers=M31_LAYERS),
        )
        group = lambda: types.SimpleNamespace(rank_in_group=0, world_size=1)
        with (
            mock.patch.object(stack.conn, "get_tp_group", group),
            mock.patch.object(stack.conn, "get_dp_group", group),
            mock.patch.object(stack.conn, "get_ip", lambda: "10.0.0.1"),
            pytest.raises(ValueError, match="fabric only"),
        ):
            stack.conn.MoRIIOConnector(config)


# ---------------------------------------------------------------------------
# Completion: P releases after every D rank's RELEASE; D after every stage
# ---------------------------------------------------------------------------


@pytest.fixture
def write_path(stack):
    import test_moriio_connector as moriio_tests

    host = types.SimpleNamespace(connector=stack.conn.MoRIIOConnector, torch=torch)
    return types.SimpleNamespace(
        make=partial(moriio_tests.WritePathTest.make, host),
        push_request=partial(moriio_tests.WritePathTest.push_request, host),
        recv_meta=partial(moriio_tests.WritePathTest.recv_meta, host),
        FakeStatus=moriio_tests.FakeStatus,
    )


def tp1_stage(write_path, stage):
    conn = write_path.make(producer=True)
    conn.tp_size, conn.tp_rank = 1, 0
    conn.pp_size, conn.pp_rank = 4, stage
    conn._write_target_ready = lambda request: True
    return conn


def serve(conn, write_path, ranks):
    """Each consumer rank asks; every push lands."""
    statuses = {}

    def write_blocks(request, engine_id):
        statuses[request["decode_tp_rank"]] = [write_path.FakeStatus("ok")]
        return statuses[request["decode_tp_rank"]]

    conn._write_blocks = write_blocks
    for r in ranks:
        conn._write_requests.put(
            write_path.push_request(decode_tp_size=4, decode_tp_rank=r)
        )
        conn._service_write_requests()


class TestProducerCompletion:
    def test_stage_0_releases_only_after_every_d_rank(self, write_path):
        conn = tp1_stage(write_path, 0)
        serve(conn, write_path, range(4))
        conn._service_write_requests()
        # Every push reported to its own D rank, as stage 0.
        reports = [(msgpack.loads(f[1]), port) for f, _, port in conn.sent]
        assert sorted(port for _, port in reports) == [7400, 7401, 7402, 7403]
        assert all(r["pp_rank"] == 0 and r["ok"] for r, _ in reports)
        assert conn.done_sending == set()
        for rank in (0, 1, 1, 2):
            conn._record_release(9, 4, rank)
        assert conn.done_sending == set()  # rank 1 twice counts once
        conn._record_release(9, 4, 3)
        assert conn.done_sending == {9}
        conn.done_sending.clear()
        conn._record_release(9, 4, 0)  # a late repeat frees nothing again
        assert conn.done_sending == set()

    def test_the_ready_event_outlives_all_but_the_last_push(self, write_path):
        conn = tp1_stage(write_path, 2)
        serve(conn, write_path, range(3))
        conn._service_write_requests()
        assert 9 in conn._kv_ready_events and conn._pushes_done == {9: 3}
        serve(conn, write_path, [3])
        conn._service_write_requests()
        assert 9 not in conn._kv_ready_events and conn._pushes_done == {}
        # A later stage never frees stage 0's block table.
        assert conn.done_sending == set()

    def test_a_second_ask_from_one_rank_is_not_a_second_push(self, write_path):
        conn = tp1_stage(write_path, 1)
        serve(conn, write_path, [2])
        conn._service_write_requests()
        serve(conn, write_path, [2])
        assert conn._pushes_done == {9: 1}

    def test_one_stage_tp1_into_tp4_waits_for_the_releases_too(self, write_path):
        conn = write_path.make(producer=True)
        conn.tp_size, conn.tp_rank = 1, 0
        conn._write_target_ready = lambda request: True
        serve(conn, write_path, range(4))
        conn._service_write_requests()
        assert conn.done_sending == set()
        for rank in range(4):
            conn._record_release(9, 4, rank)
        assert conn.done_sending == {9}

    def test_equal_tp_without_pp_releases_on_its_push(self, write_path):
        conn = write_path.make(producer=True)
        conn._write_target_ready = lambda request: True
        conn._write_blocks = lambda request, engine_id: [write_path.FakeStatus("ok")]
        conn._write_requests.put(write_path.push_request())
        conn._service_write_requests()
        conn._service_write_requests()
        assert conn.done_sending == {9}
        [(frames, _host, _port)] = conn.sent
        assert "pp_rank" not in msgpack.loads(frames[1])


class TestConsumerCompletion:
    def consumer(self, write_path):
        conn = write_path.make(producer=False)  # TP8 rank 1
        conn.request_id_to_transfer_id = {"d1": 9}
        meta = write_path.recv_meta([4, 5], [10, 11], remote_pp_size=4)
        meta.tp_size = 1
        return conn, meta

    def test_it_asks_every_stage_at_its_port(self, write_path):
        conn, meta = self.consumer(write_path)
        conn._request_write("d1", meta)
        asks = [(msgpack.loads(f[1]), port) for f, _, port in conn.sent]
        assert [port for _, port in asks] == [7301, 7302, 7303, 7304]
        assert {a["decode_tp_rank"] for a, _ in asks} == {1}
        assert conn._write_reports_expected == {"d1": 4}

    def test_done_only_after_every_stage_then_release_stage_0(self, write_path):
        conn, meta = self.consumer(write_path)
        conn._request_write("d1", meta)
        conn.sent.clear()
        for stage in (0, 3, 3, 1):  # stage 3 twice counts once
            conn._record_write_report("d1", True, stage=stage)
        done, failed = set(), set()
        conn._collect_write_reports(done, failed)
        assert done == set() and "d1" in conn._awaiting_write
        conn._record_write_report("d1", True, stage=2)
        conn._collect_write_reports(done, failed)
        assert done == {"d1"}
        [(frames, host, port)] = conn.sent
        assert frames[0] == b"release" and (host, port) == ("10.0.0.1", 7301)
        assert msgpack.loads(frames[1]) == {
            "transfer_id": 9,
            "consumers": 8,
            "decode_tp_rank": 1,
        }

    def test_one_failed_stage_fails_the_receive(self, write_path):
        conn, meta = self.consumer(write_path)
        conn._request_write("d1", meta)
        for stage in range(4):
            conn._record_write_report("d1", stage != 2, stage=stage)
        done, failed = set(), set()
        conn._collect_write_reports(done, failed)
        assert done == set() and "d1" in conn._held_failures

    def test_a_report_from_a_stage_not_asked_is_ignored(self, write_path):
        conn, meta = self.consumer(write_path)
        conn._request_write("d1", meta)
        conn._record_write_report("d1", True, stage=7)
        assert conn._write_report_counts == {}


class TestScheduler:
    def test_the_prefill_names_its_stages(self, stack):
        sched = object.__new__(stack.conn.MoRIIOConnectorScheduler)
        sched.__dict__.update(
            is_producer=True,
            pp_size=4,
            tp_size=1,
            dp_rank=0,
            engine_id="e",
            host_ip="10.0.0.1",
            handshake_port=1,
            base_handshake_port=6301,
            _awaiting_send=set(),
            request_id_to_transfer_id={},
            transfer_id_to_request_id={},
        )
        seq = types.SimpleNamespace(
            id=9,
            output_tokens=[1],
            block_table=[3, 4],
            kv_transfer_params={},
            leave_reason="max_tokens",
            is_finished=True,
        )
        with mock.patch.object(stack.conn, "produced_no_handoff", lambda s: False):
            sched.request_finished(seq)
        out = seq.kv_transfer_params_output
        assert out["remote_pp_size"] == 4 and out["tp_size"] == 1


def test_a_downstream_stage_polls_pushes_at_the_idle_cadence():
    from pathlib import Path

    source = (
        Path(__file__).resolve().parents[1] / "atom/model_engine/pp_engine_core.py"
    ).read_text()
    body = source[source.index("def _downstream_busy_loop") :]
    assert 'getattr(connector, "pushes_from_worker_poll", False)' in body[:1500]
