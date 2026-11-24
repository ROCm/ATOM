# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""CPU tests for MoRIIO over the GPU scale-up fabric (`moriio_backend` `fabric`).

- Region coverage for MiniMax-M3.1 (TP4, NVFP4 K/V, FP4 index, DSpark draft):
  the paged pool is carved by the runner's own `_back_paged_pools`, which
  asks the connector hook for it, its regions come from the M3 attention
  builder and the DSpark draft builder, and the transfer tensors are put
  together as `ModelRunner.allocate_kv_cache` does. Every PAGE region the
  Mooncake path moves must be registered by `_register_pages` and moved, once,
  by one fabric push.
- The fabric batch plan (`page_batch`) and its refusals.
- The config path: write on both ends, the connector building only the FABRIC
  backend, and the factory hook, `multi` included.
- Write-mode ordering: a push waits for its prefill's KV-ready event, which
  every producer records, and a consumer asks only after its queued forwards.

aiter, triton and mori are not needed. The model, runner and connector modules
import against the stand-ins in `aiter_stub`; mori's fabric allocator is
replaced by a CPU tensor, and its batch write by applying the batch's segments
to the CPU pools.
"""

from __future__ import annotations

import ast
import collections
import json
import sys
import types
import warnings
from functools import partial
from pathlib import Path
from unittest import mock

import msgpack
import msgspec
import pytest
import torch
from aiter_stub import atom_under_stubs
from torch import nn

ROOT = Path(__file__).resolve().parents[1]

# MiniMax-M3.1-Flash (config.json and dspark/config.json): 60 layers, all
# sparse (sparse_attention_freq is all 1), 4 KV heads of 128, a 128-wide index
# key; the DSpark draft has 5 sliding-window layers with 4 KV heads of 128.
M31_LAYERS = 60
M31_KV_HEADS = 4
M31_HEAD_DIM = 128
M31_INDEX_DIM = 128
M31_DRAFT_LAYERS = 5
TP = 4
BLOCK = 128


@pytest.fixture(scope="module")
def stack():
    """The modules under test, imported against the aiter / triton stand-ins.

    Every test runs inside this context: the builders and the factory import
    lazily (`from aiter import dtypes`, the connector registry), and those
    imports must keep resolving to the stand-ins.
    """
    with atom_under_stubs() as load:
        aiter = load("aiter")
        # The stand-in's table is empty; the M3 builder looks the KV dtype up.
        aiter.dtypes.d_dtypes.update(
            fp8=torch.float8_e4m3fn, bf16=torch.bfloat16, fp32=torch.float32
        )
        yield types.SimpleNamespace(
            runner=load("atom.model_engine.model_runner"),
            attn=load("atom.model_ops.attentions.aiter_attention"),
            draft=load("atom.spec_decode.draft_kv"),
            factory=load("atom.kv_transfer.disaggregation.factory"),
            conn=load("atom.kv_transfer.disaggregation.moriio.moriio_connector"),
            fabric=load("atom.kv_transfer.disaggregation.moriio.fabric"),
            common=load("atom.kv_transfer.disaggregation.moriio.moriio_common"),
            types=load("atom.kv_transfer.disaggregation.types"),
            multi=load("atom.kv_transfer.disaggregation.multi.multi_connector"),
            index_fp4=load("atom.model_ops.minimax_m3.index_fp4"),
            sub_pool=load("atom.model_ops.attentions.pool_layout.sub_pool_spec"),
        )


def kv_config(role: str, **extra) -> dict:
    """The --kv-transfer-config of the GPU run plan (pd1123/README-fabric.md)."""
    cfg = {
        "kv_role": role,
        "kv_connector": "moriio",
        "moriio_backend": "fabric",
        "moriio_mode": "write",
        "handshake_port": 6301,
    }
    cfg.update(extra)
    return cfg


# The prefill's --kv-transfer-config with the LMCache CPU tier, as the GPU plan
# (pd1123/README-fabric.md) passes it: no spaces, server.sh splits EXTRA_ARGS.
P_MULTI = (
    '{"kv_connector":"multi","connectors":[{"kv_connector":"moriio",'
    '"kv_role":"kv_producer","moriio_backend":"fabric","moriio_mode":"write",'
    '"handshake_port":6301},{"kv_connector":"lmcache_offload","kv_role":"offload"}]}'
)


class _SparseAttention(nn.Module):
    """What the M3 builder's walk sees of a MiniMax-M3 sparse attention layer:
    the TP4 shard (one KV head) and its NVFP4 sparse impl."""

    base_attention = True
    use_mla = False

    def __init__(self, layer_num: int):
        super().__init__()
        self.layer_num = layer_num
        self.num_kv_heads = M31_KV_HEADS // TP
        self.head_dim = M31_HEAD_DIM
        self.impl = types.SimpleNamespace(
            is_indexed_sparse_attention=True, kv_format="nvfp4"
        )


class _DraftKVBinding(nn.Module):
    """`atom.models.minimax_m3.dspark.KVCacheBinding`'s attributes (that module
    needs FlyDSL to import): paged, not MLA."""

    base_attention = None
    use_mla = False

    def __init__(self, layer_num: int):
        super().__init__()
        self.layer_num = layer_num


def m31_runner(stack, blocks: int, role: str, kv: dict | None = None):
    """A ModelRunner with M3.1's builders and pool backed for ``blocks`` blocks,
    under ``kv`` (default: the plan's fabric config for ``role``).

    Returns the runner, its transfer tensors and the pool tensor.
    """
    mr, attn, draft = stack.runner, stack.attn, stack.draft
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
        kv_transfer_config=kv if kv is not None else kv_config(role),
        kv_cache_dtype="fp8",
        index_cache_dtype="fp4",
        hf_config=hf,
        speculative_config=types.SimpleNamespace(num_speculative_tokens=7),
    )
    runner.device = torch.device("cpu")
    runner.block_size = BLOCK
    runner.model = nn.ModuleList(_SparseAttention(i) for i in range(M31_LAYERS))
    runner.drafter = types.SimpleNamespace(
        model=nn.ModuleList(
            _DraftKVBinding(M31_LAYERS + i) for i in range(M31_DRAFT_LAYERS)
        )
    )
    builder = object.__new__(attn.AiterAttentionMetadataBuilder)
    builder.model_runner = runner
    builder.block_size = BLOCK
    builder.block_ratio = 1
    builder._has_sparse_attention = True
    builder._pool_rows = None
    builder.kv_pools = {}
    runner.attn_metadata_builder = builder
    # `draft_kv_builder()` with the M3.1 draft's backend, which is the target's.
    runner.draft_kv_builder = draft.DraftKvBuilder(
        runner,
        partial(
            attn.AiterBackend.make_kv_pool,
            draft_hf,
            world_size=TP,
            kv_dtype=torch.float8_e4m3fn,
        ),
    )

    pools = []

    def cpu_fabric_pool(nbytes, device):
        # `allocate_fabric_pool` on a CPU tensor: the same record of the range.
        pool = torch.zeros(nbytes, dtype=torch.uint8)
        stack.fabric._pool_range = (pool.data_ptr(), nbytes)
        pools.append(pool)
        return pool

    with mock.patch.object(stack.conn, "allocate_fabric_pool", cpu_fabric_pool):
        pool = runner._back_paged_pools(blocks)
    assert pools and pool is pools[0], "the paged pool did not come from the hook"
    # As ModelRunner.allocate_kv_cache puts them together
    # (`test_the_runner_still_assembles_the_regions_this_way` pins it).
    tt = builder.get_kv_transfer_tensors()
    tt.merge_pages(runner.draft_kv_builder.get_kv_transfer_tensors())
    tt.set_block_count(blocks)
    return runner, tt, pool


def packed_desc(ptr: int, size: int, handle: bytes = b"\x01" * 64, vpod=16):
    """A packed mori MemoryDesc, fields in MSGPACK_DEFINE order."""
    fields = [b"key", 1, 0, b"bus", ptr, size, 1, b"", 0, 0, handle, vpod]
    return msgpack.packb(fields + [b"ppod", 0, size, {}])


def fabric_connector(stack, *, producer: bool, handle: bytes = b"\x01" * 64):
    """A MoRIIOConnector with what registration and a fabric push read."""
    conn = object.__new__(stack.conn.MoRIIOConnector)
    conn.is_producer = producer
    conn.use_fabric = True
    conn.transfer_mode = "write"
    conn.tp_size = TP
    conn.tp_rank = 0
    conn.engine_id = f"10.0.0.{1 if producer else 2}:6301"
    conn.kv_cache_block_size = BLOCK
    conn._verify_kv = False
    conn._built_sessions = {}
    conn.remote_moriio_metadata = {}
    conn.posted = []

    def write(sizes, l_offs, r_offs, session):
        conn.posted.append((session, list(l_offs), list(r_offs), list(sizes)))
        return "status"

    conn.moriio_wrapper = types.SimpleNamespace(
        register_local_buffer=lambda ptr, size, device: packed_desc(ptr, size, handle),
        get_agent_metadata=lambda: b"agent",
        get_unpack_memory_metadata=lambda packed: ("desc", packed),
        build_session=lambda local, remote: ("session", local[1], remote[1]),
        write_remote_data_status=write,
    )
    return conn


def handshake(stack, producer, consumer_meta):
    """What `_execute_handshake` stores: the peer's metadata after the wire."""
    wire = msgspec.msgpack.Encoder().encode(consumer_meta)
    meta = msgspec.msgpack.Decoder(stack.common.MoRIIOAgentMetadata).decode(wire)
    producer.remote_moriio_metadata["decode"] = meta


def as_bytes(tensor: torch.Tensor) -> torch.Tensor:
    """Compared as bytes: random fp32 scales hold NaNs, and fp8 has no eq."""
    return tensor.contiguous().view(torch.uint8)


def apply_batch(posted, src_pool, dst_pool):
    """Run a posted fabric batch on CPU pools; count the writes per byte."""
    written = torch.zeros(dst_pool.numel(), dtype=torch.int32)
    for _session, l_offs, r_offs, sizes in posted:
        for lo, ro, n in zip(l_offs, r_offs, sizes):
            dst_pool[ro : ro + n] = src_pool[lo : lo + n]
            written[ro : ro + n] += 1
    return written


# ---------------------------------------------------------------------------
# Region coverage: M3.1's real pools, all registered, all moved, once
# ---------------------------------------------------------------------------


class TestM31RegionCoverage:
    @pytest.fixture(autouse=True)
    def _device(self):
        # A CPU view has no device index; registration then asks the current
        # CUDA device, which a GPU rank has.
        with mock.patch("torch.cuda.current_device", lambda: 0):
            yield

    def test_the_runner_still_assembles_the_regions_this_way(self):
        # m31_runner repeats these lines of ModelRunner.allocate_kv_cache.
        source = (ROOT / "atom/model_engine/model_runner.py").read_text()
        body = source[source.index("    def allocate_kv_cache(self") :]
        body = body[: body.index("\n    def ", 10)]
        steps = [
            "pool = self._back_paged_pools(num_kvcache_blocks)",
            "transfer_tensors = self.attn_metadata_builder.get_kv_transfer_tensors()",
            "self.draft_kv_builder.get_kv_transfer_tensors()",
            "transfer_tensors.set_block_count(num_kvcache_blocks)",
        ]
        positions = [body.index(step) for step in steps]
        assert positions == sorted(positions)

    def test_the_pool_comes_from_the_fabric_hook_and_holds_every_region(self, stack):
        runner, tt, pool = m31_runner(stack, 6, "kv_producer")
        assert runner._kv_pool_outside_torch_bytes == pool.numel()
        base, end = pool.data_ptr(), pool.data_ptr() + pool.numel()
        for region in tt.block_regions:
            assert base <= region.base_addr
            assert region.base_addr + region.total_bytes <= end

    def test_the_regions_are_m31s_380(self, stack):
        _runner, tt, _pool = m31_runner(stack, 6, "kv_producer")
        roles = [r.semantic_role for r in tt.block_regions]
        target = [r for r in roles if r.startswith("mha.")]
        drafts = [r for r in roles if r.startswith("draft.")]
        assert len(roles) == len(set(roles)) == 380
        assert len(target) == M31_LAYERS * 6 and len(drafts) == M31_DRAFT_LAYERS * 4
        fields = collections.Counter(r.split(".")[-2] for r in roles)
        assert fields == {
            "k": 65,
            "v": 65,
            "k_scale": 65,
            "v_scale": 65,
            "index": 60,
            "index_scale": 60,
        }
        units = {r.semantic_role.split(".")[-2]: r.unit_bytes for r in tt.block_regions}
        # NVFP4 K/V (D/2 B a token, D/16 E4M3 scales), FP4 index (64 + 8 B a
        # token), and the draft's fp8 K/V with fp32 per-token scales.
        per_block = sum(r.unit_bytes for r in tt.block_regions)
        assert units["index"] == BLOCK * M31_INDEX_DIM // 2
        assert units["index_scale"] == BLOCK * M31_INDEX_DIM // 16
        assert per_block == M31_LAYERS * BLOCK * (3 * 64 + 3 * 8) + (
            M31_DRAFT_LAYERS * BLOCK * (2 * M31_HEAD_DIM + 2 * 4)
        )

    def test_registration_covers_every_region_the_mooncake_path_moves(self, stack):
        _runner, tt, pool = m31_runner(stack, 6, "kv_producer")
        conn = fabric_connector(stack, producer=True)
        meta = conn._register_pages({}, tt, tt.num_blocks)
        # Mooncake registers and moves `tt.block_regions`, in this order.
        assert conn._page_regions == [
            (r.base_addr - pool.data_ptr(), r.unit_bytes, r.semantic_role)
            for r in tt.block_regions
        ]
        # The whole pool, one descriptor.
        assert stack.fabric.desc_size(conn._page_desc) == pool.numel()
        assert meta.page_regions == [list(entry) for entry in conn._page_regions]
        assert meta.num_blocks == 6

    def test_one_push_moves_every_regions_blocks_exactly_once(self, stack):
        # Different block counts, so the two pools place their regions apart.
        _p, tt_p, pool_p = m31_runner(stack, 8, "kv_producer")
        producer = fabric_connector(stack, producer=True)
        producer._register_pages({}, tt_p, tt_p.num_blocks)
        _d, tt_d, pool_d = m31_runner(stack, 11, "kv_consumer")
        consumer = fabric_connector(stack, producer=False)
        handshake(stack, producer, consumer._register_pages({}, tt_d, tt_d.num_blocks))
        assert [o for o, _, _ in producer._page_regions] != [
            o for o, _, _ in consumer._page_regions
        ]

        generator = torch.Generator().manual_seed(0)
        pool_p.copy_(torch.randint(0, 256, pool_p.shape, generator=generator))
        pool_d.copy_(torch.randint(0, 256, pool_d.shape, generator=generator))
        before = pool_d.clone()
        src, dst = [5, 6, 7, 1], [2, 3, 4, 9]
        producer._write_pages(
            {"transfer_id": 9, "decode_tp_size": TP}, "decode", src, dst
        )
        # One batch, one segment per region and run (5-7 -> 2-4, then 1 -> 9).
        [(session, _l_offs, _r_offs, sizes)] = producer.posted
        assert session[0] == "session" and len(sizes) == 2 * 380
        written = apply_batch(producer.posted, pool_p, pool_d)
        assert int(written.max()) == 1

        expected = before.clone()
        for page_p, page_d in zip(tt_p.pages, tt_d.pages):
            for s, d in zip(src, dst):
                expected_view = expected[page_d.region.base_addr - pool_d.data_ptr() :][
                    : page_d.region.total_bytes
                ].view(-1, page_d.region.unit_bytes)
                expected_view[d] = page_p.view[s].reshape(-1)
        assert torch.equal(pool_d, expected)

        # Read back through the views the model binds (`build_kv_cache_tensor`).
        geometry_p, pool_kv_p = next(iter(_p.attn_metadata_builder.kv_pools.items()))
        pool_kv_d = _d.attn_metadata_builder.kv_pools[geometry_p]
        payload, scale = stack.index_fp4.fp4_index_row_bytes(M31_INDEX_DIM)
        for layer in (0, 31, M31_LAYERS - 1):
            views_p = list(pool_kv_p.kv_views(layer) + pool_kv_p.scale_views(layer))
            views_d = list(pool_kv_d.kv_views(layer) + pool_kv_d.scale_views(layer))
            for field, width in (("index", payload), ("index_scale", scale)):
                shape = (-1, BLOCK, width)
                views_p.append(pool_kv_p.field_view(field, layer, torch.uint8, shape))
                views_d.append(pool_kv_d.field_view(field, layer, torch.uint8, shape))
            for view_p, view_d in zip(views_p, views_d, strict=True):
                assert torch.equal(as_bytes(view_d[dst]), as_bytes(view_p[src]))
        draft_p, draft_d = _p.draft_kv_builder.kv_pool, _d.draft_kv_builder.kv_pool
        for layer in range(M31_DRAFT_LAYERS):
            for view_p, view_d in zip(
                draft_p.kv_views(layer) + draft_p.scale_views(layer),
                draft_d.kv_views(layer) + draft_d.scale_views(layer),
                strict=True,
            ):
                assert torch.equal(as_bytes(view_d[dst]), as_bytes(view_p[src]))

    def test_a_block_past_either_pool_is_refused_before_anything_is_posted(self, stack):
        _p, tt_p, _pool_p = m31_runner(stack, 8, "kv_producer")
        producer = fabric_connector(stack, producer=True)
        producer._register_pages({}, tt_p, tt_p.num_blocks)
        _d, tt_d, _pool_d = m31_runner(stack, 4, "kv_consumer")
        consumer = fabric_connector(stack, producer=False)
        handshake(stack, producer, consumer._register_pages({}, tt_d, tt_d.num_blocks))
        request = {"transfer_id": 9, "decode_tp_size": TP}
        for src, dst in (([1], [4]), ([8], [1]), ([1], [-1]), ([1, 2], [3, 3])):
            with pytest.raises(ValueError):
                producer._write_pages(request, "decode", src, dst)
        assert producer.posted == []

    def test_a_peer_missing_a_region_gets_no_session(self, stack):
        _p, tt_p, _pool_p = m31_runner(stack, 6, "kv_producer")
        producer = fabric_connector(stack, producer=True)
        producer._register_pages({}, tt_p, tt_p.num_blocks)
        _d, tt_d, _pool_d = m31_runner(stack, 6, "kv_consumer")
        consumer = fabric_connector(stack, producer=False)
        meta = consumer._register_pages({}, tt_d, tt_d.num_blocks)
        # A decode without the draft's regions (no DSpark there).
        meta.page_regions = meta.page_regions[:360]
        handshake(stack, producer, meta)
        with pytest.raises(ValueError, match="first difference at region 360"):
            producer._write_pages({"transfer_id": 9}, "decode", [1], [1])
        assert producer.posted == []

    def test_a_pool_without_a_fabric_handle_is_refused(self, stack):
        _p, tt_p, _pool_p = m31_runner(stack, 6, "kv_producer")
        conn = fabric_connector(stack, producer=True, handle=b"\x00" * 64)
        with pytest.raises(RuntimeError, match="no fabric handle"):
            conn._register_pages({}, tt_p, tt_p.num_blocks)

    def test_a_region_outside_the_fabric_pool_is_refused(self, stack):
        _p, tt_p, pool = m31_runner(stack, 6, "kv_producer")
        # The last region (the draft's) in memory of its own.
        stray = torch.zeros(tt_p.block_regions[-1].total_bytes, dtype=torch.uint8)
        tt_p.pages[-1].region.base_addr = stray.data_ptr()
        conn = fabric_connector(stack, producer=True)
        with pytest.raises(ValueError, match="not inside"):
            conn._register_pages({}, tt_p, tt_p.num_blocks)
        assert stack.fabric._pool_range == (pool.data_ptr(), pool.numel())


class TestDigests:
    """One digest helper for the three verify paths, the same sums as before."""

    @staticmethod
    def weighted(flat: torch.Tensor, length: int) -> int:
        weights = torch.arange(length, dtype=torch.int64) % 65521 + 1
        return int(torch.sum(flat.to(torch.int64) * weights[: flat.numel()]))

    def test_the_layer_digest_is_the_old_one(self, stack):
        cache = torch.randint(0, 256, (6, 2, 8), dtype=torch.uint8).view(torch.bfloat16)
        rows = torch.tensor([4, 1])
        old = cache.index_select(0, rows).view(torch.uint8).reshape(-1)
        assert stack.conn._byte_digests([cache.index_select(0, rows)]) == [
            self.weighted(old, old.numel())
        ]

    def test_state_rows_of_any_shape_including_0_d(self, stack):
        state = torch.randn(4, 3, 5).transpose(1, 2)  # not contiguous
        scalar = torch.randn(4)  # one value per slot: a 0-d row
        old = [
            state[2].reshape(-1).view(torch.uint8),
            scalar[2].reshape(-1).view(torch.uint8),
        ]
        length = max(row.numel() for row in old)
        assert stack.conn._byte_digests([state[2], scalar[2]]) == [
            self.weighted(row, length) for row in old
        ]


# ---------------------------------------------------------------------------
# The batch plan
# ---------------------------------------------------------------------------


class TestPageBatch:
    def test_runs_contiguous_on_both_sides_are_one_segment_per_region(self, stack):
        offs = stack.fabric.page_batch(
            [(0, 16), (1000, 4)], [(64, 16), (5000, 4)], [2, 3, 7], [3, 4, 0], 8, 8
        )
        assert offs == (
            [32, 112, 1008, 1028],
            [64 + 48, 64, 5012, 5000],
            [32, 16, 8, 4],
        )

    def test_a_gap_on_either_side_splits_the_run(self, stack):
        _, _, sizes = stack.fabric.page_batch(
            [(0, 1)], [(0, 1)], [1, 2, 4, 5], [1, 2, 3, 4], 8, 8
        )
        assert sizes == [2, 2]

    def test_refusals(self, stack):
        batch = partial(stack.fabric.page_batch, [(0, 4)], [(0, 4)])
        for args in (
            ([8], [0], 8, 8),  # source past its pool
            ([0], [8], 8, 8),  # destination past its pool
            ([-1], [0], 8, 8),
            ([0, 1], [3, 3], 8, 8),  # a destination twice
            ([0, 1], [3], 8, 8),
        ):
            with pytest.raises(ValueError):
                batch(*args)
        # A source read twice is not a write twice.
        batch([1, 1], [2, 3], 8, 8)
        with pytest.raises(ValueError):
            stack.fabric.page_batch([(0, 4)], [(0, 8)], [0], [0], 8, 8)
        with pytest.raises(ValueError):
            stack.fabric.page_batch([(0, 4)], [], [0], [0], 8, 8)

    def test_a_table_must_fit_its_pool_without_overlaps(self, stack):
        check = stack.fabric.check_page_table
        check([(0, 4, "k"), (32, 4, "v")], 8, 64, "local")
        for entries, pool in (
            ([(0, 4, "k"), (28, 4, "v")], 64),  # v starts inside k
            ([(0, 4, "k"), (40, 4, "v")], 64),  # v runs past the pool
            ([(-4, 4, "k")], 64),
        ):
            with pytest.raises(ValueError):
                check(entries, 8, pool, "local")


# ---------------------------------------------------------------------------
# The config path
# ---------------------------------------------------------------------------


class TestConfigPath:
    def config(self, **kv):
        return types.SimpleNamespace(kv_transfer_config=kv, kv_cache_block_size=BLOCK)

    def test_the_run_plans_configs_select_fabric(self, stack, monkeypatch):
        monkeypatch.delenv("ATOM_MORIIO_MODE", raising=False)
        backend = stack.conn.MoRIIOConnector._fabric_backend_from
        for role in ("kv_producer", "kv_consumer"):
            parsed = json.loads(json.dumps(kv_config(role)))
            assert backend(types.SimpleNamespace(kv_transfer_config=parsed))
        assert not backend(self.config(kv_connector="moriio", moriio_mode="write"))

    def test_fabric_refuses_read_mode_on_either_end(self, stack, monkeypatch):
        monkeypatch.delenv("ATOM_MORIIO_MODE", raising=False)
        backend = stack.conn.MoRIIOConnector._fabric_backend_from
        for role in ("kv_producer", "kv_consumer"):
            unset = kv_config(role)
            del unset["moriio_mode"]
            for kv in (unset, kv_config(role, moriio_mode="read")):
                with pytest.raises(ValueError, match="moriio_mode=write"):
                    backend(types.SimpleNamespace(kv_transfer_config=kv))
        monkeypatch.setenv("ATOM_MORIIO_MODE", "read")
        with pytest.raises(ValueError):
            backend(types.SimpleNamespace(kv_transfer_config=kv_config("kv_consumer")))

    def test_fabric_refuses_dcp_pcp_and_pp(self, stack, monkeypatch):
        monkeypatch.delenv("ATOM_MORIIO_MODE", raising=False)
        backend = stack.conn.MoRIIOConnector._fabric_backend_from
        for name in (
            "decode_context_parallel_size",
            "prefill_context_parallel_size",
            "pipeline_parallel_size",
        ):
            config = types.SimpleNamespace(
                kv_transfer_config=kv_config("kv_producer"), **{name: 2}
            )
            with pytest.raises(ValueError, match=name):
                backend(config)

    def test_the_factory_hook_reaches_moriio_inside_multi(self, stack, monkeypatch):
        monkeypatch.delenv("ATOM_MORIIO_MODE", raising=False)
        factory = stack.factory.KVConnectorFactory
        made = []

        def cpu_fabric_pool(nbytes, device):
            made.append(nbytes)
            return torch.zeros(nbytes, dtype=torch.uint8)

        offload = {"kv_connector": "lmcache_offload", "kv_role": "offload"}
        with mock.patch.object(stack.conn, "allocate_fabric_pool", cpu_fabric_pool):
            for kv in (
                kv_config("kv_producer"),
                {
                    "kv_connector": "multi",
                    "connectors": [offload, kv_config("kv_producer")],
                },
            ):
                pool = factory.allocate_kv_pool(
                    types.SimpleNamespace(kv_transfer_config=kv), 64, "cpu"
                )
                assert pool is not None and pool.numel() == 64
            for kv in (
                {"kv_connector": "moriio", "kv_role": "kv_producer"},
                {"kv_connector": "mooncake", "kv_role": "kv_producer"},
                {"kv_connector": "multi", "connectors": [offload]},
            ):
                assert (
                    factory.allocate_kv_pool(
                        types.SimpleNamespace(kv_transfer_config=kv), 64, "cpu"
                    )
                    is None
                )
        assert made == [64, 64]

    def test_the_runner_allocates_with_torch_without_fabric(self, stack, monkeypatch):
        monkeypatch.delenv("ATOM_MORIIO_MODE", raising=False)
        runner = object.__new__(stack.runner.ModelRunner)
        runner.device = torch.device("cpu")
        runner.attn_metadata_builder = types.SimpleNamespace(
            invalidate_pool_rows=lambda: None,
            paged_pool_bytes=lambda blocks: blocks * 512,
            sub_pool_specs=lambda: [stack.sub_pool.page_pool(512)],
        )
        refuse = lambda *a: pytest.fail("fabric pool without fabric")
        with mock.patch.object(stack.conn, "allocate_fabric_pool", refuse):
            for kv in (
                {},
                {"kv_connector": "mooncake", "kv_role": "kv_producer"},
                {"kv_connector": "moriio", "kv_role": "kv_producer"},
            ):
                runner.config = types.SimpleNamespace(kv_transfer_config=kv)
                buf, regions, _ = runner._carve_paged_pool(4)
                assert runner._kv_pool_outside_torch_bytes == 0
                assert buf.numel() == 2048 and regions[0].numel() == 2048

    def connector(self, stack, kv, create_backend):
        """MoRIIOConnector.__init__ on a fake mori whose engine's
        `create_backend` is ``create_backend``."""
        conn_mod = stack.conn
        engine = mock.MagicMock()
        engine.create_backend.side_effect = create_backend
        group = lambda: types.SimpleNamespace(rank_in_group=0, world_size=TP)
        fake_io = types.ModuleType("mori.io")
        fake_io.FabricBackendConfig = lambda streams, events: (
            "fabric",
            streams,
            events,
        )
        fake_mori = types.ModuleType("mori")
        fake_mori.io = fake_io
        patches = [
            mock.patch.object(conn_mod, "get_tp_group", group),
            mock.patch.object(conn_mod, "get_dp_group", group),
            mock.patch.object(conn_mod, "get_ip", lambda: "10.0.0.1"),
            mock.patch.object(conn_mod, "IOEngine", lambda *a: engine, create=True),
            mock.patch.object(
                conn_mod, "IOEngineConfig", mock.MagicMock(), create=True
            ),
            mock.patch.object(
                conn_mod,
                "BackendType",
                types.SimpleNamespace(FABRIC="FABRIC", RDMA="RDMA"),
                create=True,
            ),
            # `_create_fabric_backend` imports it from mori.io itself.
            mock.patch.dict(sys.modules, {"mori": fake_mori, "mori.io": fake_io}),
            mock.patch.object(
                conn_mod, "RdmaBackendConfig", mock.MagicMock(), create=True
            ),
            mock.patch.object(conn_mod, "PollCqMode", mock.MagicMock(), create=True),
        ]
        for patch in patches:
            patch.start()
        try:
            conn = conn_mod.MoRIIOConnector(
                types.SimpleNamespace(kv_transfer_config=kv, kv_cache_block_size=BLOCK)
            )
        finally:
            for patch in patches:
                patch.stop()
        conn.zmq_context.term()
        conn._handshake_executor.shutdown(wait=False)
        return conn

    def test_the_connector_builds_the_fabric_backend_alone(self, stack, monkeypatch):
        monkeypatch.delenv("ATOM_MORIIO_MODE", raising=False)
        created = []
        record = lambda kind, cfg: created.append((kind, cfg))
        conn = self.connector(stack, kv_config("kv_consumer"), record)
        assert conn.use_fabric and conn.transfer_mode == "write"
        assert created == [("FABRIC", ("fabric", 64, 64))]
        created.clear()
        self.connector(stack, kv_config("kv_producer", fabric_num_streams=32), record)
        assert created == [("FABRIC", ("fabric", 32, 64))]

    def test_a_fabric_copy_kernel_that_fails_to_load_fails_the_connector(
        self, stack, monkeypatch
    ):
        monkeypatch.delenv("ATOM_MORIIO_MODE", raising=False)

        def warn(kind, cfg):
            # mori.io.engine._load_fabric_copy_kernel only warns when it fails.
            warnings.warn_explicit(
                "Failed to load fabric_copy kernel (no hipcc)",
                RuntimeWarning,
                "/app/mori/python/mori/io/engine.py",
                113,
            )

        with pytest.raises(RuntimeError, match="fabric_copy"):
            self.connector(stack, kv_config("kv_producer"), warn)

    def test_another_threads_warning_does_not_fail_the_backend(
        self, stack, monkeypatch
    ):
        monkeypatch.delenv("ATOM_MORIIO_MODE", raising=False)

        def warn(kind, cfg):
            warnings.warn_explicit("overflow in exp", RuntimeWarning, "numpy.py", 1)

        # Shown again once the backend is built, not raised.
        with pytest.warns(RuntimeWarning, match="overflow in exp"):
            conn = self.connector(stack, kv_config("kv_producer"), warn)
        assert conn.use_fabric

    def test_a_mori_without_the_fabric_backend_still_imports_the_connector(self):
        # FabricBackendConfig is imported by `_create_fabric_backend`, not with
        # the RDMA names at module level.
        tree = ast.parse(
            (
                ROOT / "atom/kv_transfer/disaggregation/moriio/moriio_connector.py"
            ).read_text()
        )
        module_level = [
            alias.name
            for node in ast.walk(tree)
            if isinstance(node, ast.If)
            for stmt in node.body
            if isinstance(stmt, ast.ImportFrom) and stmt.module == "mori.io"
            for alias in stmt.names
        ]
        assert "IOEngine" in module_level
        assert "FabricBackendConfig" not in module_level

    def test_a_misspelt_backend_is_refused(self, stack):
        fabric = stack.fabric
        assert fabric.is_fabric_backend({"moriio_backend": " Fabric "})
        assert not fabric.is_fabric_backend({})
        assert not fabric.is_fabric_backend({"moriio_backend": "rdma"})
        with pytest.raises(ValueError, match="moriio_backend"):
            fabric.is_fabric_backend({"moriio_backend": "fabirc"})

    def test_a_fabric_consumer_in_read_mode_builds_no_engine(self, stack, monkeypatch):
        monkeypatch.delenv("ATOM_MORIIO_MODE", raising=False)
        never = lambda kind, cfg: pytest.fail("built a backend")
        with pytest.raises(ValueError, match="moriio_mode=write"):
            self.connector(stack, kv_config("kv_consumer", moriio_mode="read"), never)


# ---------------------------------------------------------------------------
# MoRIIO fabric next to the LMCache CPU tier on the prefill (`multi`)
# ---------------------------------------------------------------------------


class _OffloadWorker:
    """What MultiConnector calls on the lmcache_offload worker sub."""

    is_producer = False
    _state_tier = None

    def __init__(self, output_cls):
        self.output_cls = output_cls
        self.registered = None
        self.ready: list = []

    def register_kv_caches(self, kv_caches, transfer_tensors=None, num_blocks=None):
        self.registered = (kv_caches, transfer_tensors, num_blocks)

    def start_load_kv(self, metadata):
        pass

    def publish_prefill_chunks(self, batch):
        pass

    def record_kv_cache_ready(self, req_ids):
        self.ready.append(list(req_ids))

    def get_finished(self):
        return self.output_cls(finished_saving={"save-9"})


class _OffloadScheduler:
    """The lmcache_offload scheduler sub: a prefix hit and a save per request."""

    is_producer = False
    is_offload = True

    def __init__(self):
        self.saving: set = set()

    def get_num_new_matched_tokens(self, seq):
        return 256, True

    def cancel_pending_load(self, seq):
        pass

    def request_finished(self, seq):
        self.saving.add(seq.id)

    def should_defer_free(self, seq):
        return seq.id in self.saving

    def waits_for_transfer_report(self, seq):
        return False

    def admission_ready(self, seq):
        return True

    def send_finished(self, req_id):
        pass


class TestMultiWithLMCache:
    @pytest.fixture(autouse=True)
    def _env(self, monkeypatch):
        monkeypatch.delenv("ATOM_MORIIO_MODE", raising=False)
        with mock.patch("torch.cuda.current_device", lambda: 0):
            yield

    def test_the_prefills_json_selects_fabric_for_the_moriio_leg(self, stack):
        kv = json.loads(P_MULTI)
        assert stack.types.kv_config_has_producer(kv)
        config = types.SimpleNamespace(kv_transfer_config=kv)
        leaves = stack.factory.KVConnectorFactory.leaf_connectors(config)
        assert [name for name, _ in leaves] == ["moriio", "lmcache_offload"]
        assert stack.conn.MoRIIOConnector._fabric_backend_from(leaves[0][1])

    def test_the_runner_takes_the_fabric_pool_under_multi(self, stack):
        for kv in (
            json.loads(P_MULTI),
            # The offload leg first: it allocates nothing, the walk goes on.
            dict(
                json.loads(P_MULTI),
                connectors=json.loads(P_MULTI)["connectors"][::-1],
            ),
        ):
            runner, tt, pool = m31_runner(stack, 6, "kv_producer", kv=kv)
            assert runner._kv_pool_outside_torch_bytes == pool.numel()
            base, end = pool.data_ptr(), pool.data_ptr() + pool.numel()
            assert all(
                base <= r.base_addr and r.base_addr + r.total_bytes <= end
                for r in tt.block_regions
            )

    def test_the_composite_registers_records_and_pushes_through_moriio(
        self, stack, write_path
    ):
        # Each end registers against the pool its own process allocated last
        # (`fabric._pool_range`), so the decode registers before the prefill
        # allocates.
        _d, tt_d, pool_d = m31_runner(stack, 10, "kv_consumer")
        d_meta = fabric_connector(stack, producer=False)._register_pages(
            {}, tt_d, tt_d.num_blocks
        )
        _p, tt_p, pool_p = m31_runner(stack, 8, "kv_producer", kv=json.loads(P_MULTI))
        producer = write_path.make(producer=True)
        producer.use_fabric = True
        producer.tp_size = TP
        producer.kv_cache_block_size = BLOCK
        producer._kv_ready_events = {}
        producer.engine_id = "10.0.0.1:6301"
        producer.side_channel_port = 6301
        advertised = []
        producer._handshake_listener = lambda meta, *_rest: advertised.append(meta)
        status = write_path.FakeStatus()
        posted = []

        def write(sizes, l_offs, r_offs, session):
            posted.append((session, list(l_offs), list(r_offs), list(sizes)))
            return status

        wrapper = producer.moriio_wrapper
        wrapper.register_local_buffer = lambda ptr, size, dev: packed_desc(ptr, size)
        wrapper.get_unpack_memory_metadata = lambda packed: ("desc", packed)
        wrapper.build_session = lambda local, remote: ("session", local[1], remote[1])
        wrapper.write_remote_data_status = write
        offload = _OffloadWorker(stack.types.KVConnectorOutput)
        multi = object.__new__(stack.multi.MultiConnector)
        multi._connectors = [producer, offload]
        multi.is_producer = True
        multi._state_tier = None

        multi.register_kv_caches({}, tt_p, tt_p.num_blocks)
        producer._handshake_listener_thread.join(timeout=5)
        assert len(producer._page_regions) == 380
        assert len(advertised[0].page_regions) == 380  # what the decode is sent
        assert offload.registered[1] is tt_p
        event_patch, stream_patch = cuda_events()
        with event_patch, stream_patch:
            multi.record_kv_cache_ready([9])
        assert offload.ready == [[9]] and 9 in producer._kv_ready_events

        engine = "10.0.0.2:7400_dp0"  # push_request's decode end
        wire = msgspec.msgpack.Encoder().encode(d_meta)
        producer.remote_moriio_metadata[engine] = msgspec.msgpack.Decoder(
            stack.common.MoRIIOAgentMetadata
        ).decode(wire)
        producer._remote_agents[engine] = {"agent"}
        src, dst = [2, 3, 4], [7, 8, 9]
        producer._write_requests.put(
            write_path.push_request(
                decode_tp_size=TP, src_block_ids=src, dst_block_ids=dst
            )
        )
        out = multi.get_finished()  # the prefill's forward is still running
        assert posted == [] and not out.finished_sending
        producer._kv_ready_events[9].done = True
        out = multi.get_finished()
        assert len(posted) == 1 and not out.finished_sending
        assert out.finished_saving == {"save-9"}  # the offload leg, merged
        status.state = "ok"
        out = multi.get_finished()
        assert out.finished_sending == {9}

        generator = torch.Generator().manual_seed(1)
        pool_p.copy_(torch.randint(0, 256, pool_p.shape, generator=generator))
        before = pool_d.clone()
        written = apply_batch(posted, pool_p, pool_d)
        assert int(written.max()) == 1
        for page_p, page_d in zip(tt_p.pages, tt_d.pages, strict=True):
            assert torch.equal(page_d.view[dst], page_p.view[src])
        untouched = written == 0
        assert torch.equal(pool_d[untouched], before[untouched])

    def scheduler(self, stack):
        sched = object.__new__(stack.conn.MoRIIOConnectorScheduler)
        sched.is_producer = True
        sched.engine_id = "None"
        sched.host_ip = "10.0.0.1"
        sched.handshake_port = 5555
        sched.base_handshake_port = 6301
        sched.tp_size = TP
        sched.dp_rank = 0
        sched._awaiting_send = set()
        sched._reqs_need_recv = {}
        sched._reqs_need_save = {}
        sched.request_id_to_transfer_id = {}
        sched.transfer_id_to_request_id = {}
        offload = _OffloadScheduler()
        multi = object.__new__(stack.multi.MultiConnectorScheduler)
        multi._connectors = [sched, offload]
        multi.is_producer = True
        multi.is_offload = True
        multi._load_winner = {}
        return multi, offload

    def prefill(self, leave_reason: str = "max_tokens"):
        import numpy as np

        return types.SimpleNamespace(
            id=9,
            leave_reason=leave_reason,
            output_tokens=[42],
            spec_token_ids=np.array([], dtype=np.int32),
            block_table=[1, 2],
            state_slot=-1,
            prompt_token_ids=list(range(200)),
            kv_transfer_params={"do_remote_decode": True},
            prefix_cache_hit_tokens=0,
        )

    def test_the_source_is_held_until_the_push_and_the_save_are_done(self, stack):
        multi, offload = self.scheduler(stack)
        seq = self.prefill()
        # On the prefill MoRIIO answers no lookup; the CPU tier's hit wins.
        assert multi.get_num_new_matched_tokens(seq) == (256, True)
        assert multi.admission_ready(seq)
        multi.request_finished(seq)
        handoff = seq.kv_transfer_params_output
        assert handoff["do_remote_prefill"] and handoff["transfer_id"] == 9
        assert handoff["remote_block_ids"] == [1, 2] and handoff["tp_size"] == TP
        # The deferred T0 and the (empty: a producer never drafts) drafts.
        assert handoff["first_token_id"] == 42 and handoff["draft_token_ids"] == []
        assert multi.should_defer_free(seq)
        multi.send_finished(9)  # the push landed; the save has not
        assert multi.should_defer_free(seq)
        offload.saving.discard(9)
        assert not multi.should_defer_free(seq)

    def test_an_aborted_prefill_hands_nothing_off_and_claims_nothing(self, stack):
        multi, offload = self.scheduler(stack)
        seq = self.prefill("aborted")
        multi.request_finished(seq)
        assert seq.kv_transfer_params_output is None
        offload.saving.discard(9)
        assert not multi.should_defer_free(seq)


# ---------------------------------------------------------------------------
# Write-mode ordering
# ---------------------------------------------------------------------------


class FakeEvent:
    """Stands in for torch.cuda.Event; ``done`` is flipped by the test."""

    def __init__(self, *args, **kwargs):
        self.done = False

    def record(self, stream=None):
        pass

    def query(self):
        return self.done


@pytest.fixture
def write_path(stack):
    """`tests/test_moriio_connector.py`'s write-path connector, on the stand-ins."""
    import test_moriio_connector as moriio_tests

    host = types.SimpleNamespace(connector=stack.conn.MoRIIOConnector, torch=torch)
    make = partial(moriio_tests.WritePathTest.make, host)
    push_request = partial(moriio_tests.WritePathTest.push_request, host)
    return types.SimpleNamespace(
        make=make, push_request=push_request, FakeStatus=moriio_tests.FakeStatus
    )


def moriio_tests_module():
    import test_moriio_connector

    return test_moriio_connector


def cuda_events():
    return (
        mock.patch("torch.cuda.Event", FakeEvent),
        mock.patch("torch.cuda.current_stream", lambda *a: None),
    )


class TestWriteOrdering:
    def test_a_read_mode_producer_records_the_event_a_push_waits_for(self, write_path):
        # The prefill's own moriio_mode does not matter: the decode asks for
        # the push, and every producer records when its prefill is done.
        conn = write_path.make(producer=True)
        conn.transfer_mode = "read"
        conn._kv_ready_events = {}
        statuses = [write_path.FakeStatus()]
        conn._write_target_ready = lambda request: True
        conn._write_blocks = lambda request, engine_id: statuses
        conn._write_requests.put(write_path.push_request())
        conn._service_write_requests()
        assert conn._sending_writes == {} and len(conn._pending_writes) == 1

        event_patch, stream_patch = cuda_events()
        with event_patch, stream_patch:
            conn.record_kv_cache_ready([9])
        conn._service_write_requests()
        assert conn._sending_writes == {}
        conn._kv_ready_events[9].done = True
        conn._service_write_requests()
        assert 9 in conn._sending_writes

        statuses[0].state = "ok"
        conn._service_write_requests()
        assert conn.done_sending == {9}
        # Finished: a second ask is refused by `_write_ids`, so the event goes.
        assert 9 not in conn._kv_ready_events

    def test_a_consumer_records_no_ready_event(self, write_path):
        conn = write_path.make(producer=False)
        conn._kv_ready_events = {}
        event_patch, stream_patch = cuda_events()
        with event_patch, stream_patch:
            conn.record_kv_cache_ready([9])
        assert conn._kv_ready_events == {}

    def test_a_push_whose_prefill_never_reports_fails_after_the_write_timeout(
        self, write_path
    ):
        import time

        conn = write_path.make(producer=True)
        conn._kv_ready_events = {}
        conn._write_target_ready = lambda request: True
        conn._write_blocks = lambda request, engine_id: pytest.fail("pushed")
        request = write_path.push_request()
        conn._pending_writes = [(request, time.monotonic() - 601.0)]
        conn._service_write_requests()
        assert conn.done_sending == {9}
        [(frames, _host, _port)] = conn.sent
        assert msgpack.loads(frames[1]) == {"decode_req_id": "d1", "ok": False}

    def test_old_ready_events_expire_but_their_prefills_stay_finished(self, write_path):
        import time

        conn = write_path.make(producer=True)
        stale, kept = FakeEvent(), FakeEvent()
        conn._kv_ready_events = {1: stale, 2: kept}
        conn._kv_ready_batches.append((time.monotonic() - 1e6, stale, [1, 2]))
        event_patch, stream_patch = cuda_events()
        with event_patch, stream_patch:
            conn.record_kv_cache_ready([2, 3])
        assert 1 not in conn._kv_ready_events
        assert conn._source_ready(1)
        # Re-recorded since, so the stale batch must not drop it.
        assert conn._kv_ready_events[2] is conn._kv_ready_events[3]
        assert 2 not in conn._kv_ready_expired
        assert not conn._source_ready(2)

    def test_the_consumer_asks_once_its_queued_forwards_finished(self, write_path):
        conn = write_path.make(producer=False)
        asked = []
        conn._request_write = lambda req_id, meta: asked.append(req_id)
        event_patch, stream_patch = cuda_events()
        with event_patch, stream_patch:
            conn.start_load_kv(
                types.SimpleNamespace(
                    reqs_to_recv={"d1": None, "d2": None},
                    request_id_to_transfer_id={"d1": 1, "d2": 2},
                )
            )
            conn.start_load_kv(
                types.SimpleNamespace(
                    reqs_to_recv={"d3": None}, request_id_to_transfer_id={"d3": 3}
                )
            )
        first, second = (event for event, _ in conn._gated_writes)
        conn.get_finished()
        assert asked == []
        second.done = True
        conn.get_finished()
        # In order: a later step's requests wait behind an earlier step's.
        assert asked == []
        first.done = True
        conn.get_finished()
        assert asked == ["d1", "d2", "d3"]
        assert not conn._gated_writes

    def test_a_request_that_cannot_be_asked_for_fails_alone(self, write_path):
        conn = write_path.make(producer=False)
        conn.request_id_to_transfer_id = {"d1": 1, "d2": 2, "d3": 3}
        meta = moriio_tests_module().WritePathTest.recv_meta
        good = [meta(None, [4], [10]), meta(None, [6], [12])]
        bad = meta(None, [5], [11])
        bad.remote_handshake_port = "not a port"
        ready = FakeEvent()
        ready.done = True
        conn._gated_writes.append(
            (ready, [("d1", good[0]), ("d2", bad), ("d3", good[1])])
        )
        conn._request_gated_writes()
        asked = [
            msgpack.loads(frames[1])["decode_req_id"] for frames, _, _ in conn.sent
        ]
        assert asked == ["d1", "d3"]
        assert set(conn._awaiting_write) == {"d1", "d3"}
        assert conn._failed_before_read == {"d2"}
        assert not conn._gated_writes

    def test_a_finished_push_leaves_no_ready_entry_for_a_late_ask(self, write_path):
        conn = write_path.make(producer=True)
        conn._kv_ready_events = {}
        conn._kv_ready_expired = {9: None}
        statuses = [write_path.FakeStatus("ok")]
        conn._write_target_ready = lambda request: True
        conn._write_blocks = lambda request, engine_id: statuses
        conn._write_requests.put(write_path.push_request())
        conn._service_write_requests()
        conn._service_write_requests()
        assert conn.done_sending == {9}
        assert 9 not in conn._kv_ready_expired and not conn._source_ready(9)
        # Long after, `_forget_finished_writes` has dropped the id.
        conn._write_ids.clear()
        conn._write_blocks = lambda request, engine_id: pytest.fail("pushed again")
        conn._write_requests.put(write_path.push_request())
        conn._service_write_requests()
        assert len(conn._pending_writes) == 1

    def test_transfer_stats_split_a_push_into_queue_wait_and_write(
        self, stack, write_path, caplog
    ):
        conn = write_path.make(producer=True)
        conn._transfer_stats = True
        request = write_path.push_request()
        # Arrived on the listener, taken up a step later, posted, seen settled.
        request.update(
            _received_at=100.0, _accepted_at=100.25, _page_stats=(100.5, 10**9, 4, 760)
        )
        clock = types.SimpleNamespace(monotonic=lambda: 101.0, time=lambda: 0.0)
        with (
            mock.patch.object(stack.conn, "time", clock),
            caplog.at_level("INFO", logger="atom"),
        ):
            conn._finish_write(request, ok=True)
        [line] = [
            r.getMessage() for r in caplog.records if "transfer stats" in r.getMessage()
        ]
        assert "queue_ms=250.000 wait_ms=250.000 write_ms=500.000" in line
        assert "GB_per_s=2.0 segments=760" in line

    def test_the_ready_event_is_recorded_after_the_draft_kv_is_written(self):
        # DSpark writes the draft's context KV in `compute_draft_kv`; the push
        # gate is the event `_record_kv_cache_ready` records after it.
        source = (ROOT / "atom/model_engine/model_runner.py").read_text()
        body = source[source.index("    def forward(self, batch: ScheduledBatch)") :]
        body = body[: body.index("\n    @staticmethod")]
        draft_kv = body.index("drafter.compute_draft_kv(")
        assert draft_kv < body.index("self._record_kv_cache_ready(batch)")
        assert body.count("self._record_kv_cache_ready(batch)") == 2
