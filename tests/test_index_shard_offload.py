# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""The LMCache CPU tier over a sharded MiniMax-M3 index cache.

A rank's chunk of n blocks is the whole regions' bytes of every block, then the
index rows of the blocks it keeps (``b % P == rank``), ``ceil(n / P)`` slots
(`hybrid/m3/codec.py`). Checked here without a GPU: the chunk sizes LMCache
allocates, the bytes a save stages and a load scatters for every rank (the
`DSV4PageSlotCodec` copies replayed on CPU tensors), and the namespace that
keeps sharded and replicated objects apart.
"""

from types import SimpleNamespace

import pytest
import torch

from atom.config import DCPConfig
from atom.kv_transfer.disaggregation.types import KVTransferRegion
from atom.kv_transfer.offload import config as offcfg
from atom.kv_transfer.offload.hybrid.dsv4.codec import DSV4PageSlotCodec
from atom.kv_transfer.offload.hybrid.m3.codec import M3ShardedPageCodec
from atom.kv_transfer.offload.metadata import ATOMRawBytesLMCacheMetadata, chunk_bytes

P = 4
BLOCKS = 16
WHOLE_UNITS = (16, 8, 4)  # K, V, a draft region: every block
SHARD_UNITS = (12, 2)  # index payload and scales: a rank's share


def test_a_chunk_reserves_the_most_index_blocks_it_can_hold():
    # M3.1 TP4: 1,274,880 B of K/V and draft a block, 552,960 of index.
    whole, index = 1_274_880, 552_960
    assert chunk_bytes(2, whole, index, P) == 2 * whole + index  # 256 tokens
    assert chunk_bytes(1, whole, index, P) == whole + index  # 128: lazy
    assert chunk_bytes(4, whole, index, P) == 4 * whole + index
    assert chunk_bytes(5, whole, index, P) == 5 * whole + 2 * index
    # Per token: 12,120 B at 256-token chunks, 11,040 at 512, 14,280 at 128.
    assert [chunk_bytes(n, whole, index, P) // (128 * n) for n in (2, 4, 1)] == [
        12_120,
        11_040,
        14_280,
    ]
    assert chunk_bytes(3, 100) == 300


def test_lmcache_allocates_the_chunks_bytes():
    base = SimpleNamespace(chunk_size=256)
    meta = ATOMRawBytesLMCacheMetadata(
        base,
        atom_block_size=128,
        bytes_per_block=100,
        sharded_bytes_per_block=40,
        shard_world=P,
    )
    assert meta.get_shapes() == [torch.Size((2 * 100 + 40,))]
    assert meta.get_shapes(128) == [torch.Size((100 + 40,))]
    replicated = ATOMRawBytesLMCacheMetadata(
        base, atom_block_size=128, bytes_per_block=140
    )
    assert replicated.get_shapes() == [torch.Size((280,))]
    assert meta != replicated


class Pool:
    """One rank's CPU-backed PAGE regions and the codec over them."""

    def __init__(self, rank: int, seed: int | None):
        generator = torch.Generator().manual_seed(seed) if seed is not None else None

        def tensor(units, unit):
            if generator is None:
                return torch.zeros(units * unit, dtype=torch.uint8)
            return torch.randint(0, 256, (units * unit,), generator=generator).to(
                torch.uint8
            )

        self.whole = [tensor(BLOCKS, u) for u in WHOLE_UNITS]
        self.sharded = [tensor(BLOCKS // P, u) for u in SHARD_UNITS]
        regions = [
            KVTransferRegion(t.data_ptr(), t.numel(), u, semantic_role=f"whole.{i}")
            for i, (t, u) in enumerate(zip(self.whole, WHOLE_UNITS))
        ] + [
            KVTransferRegion(
                t.data_ptr(),
                t.numel(),
                u,
                semantic_role=f"index.{i}",
                shard_rank=rank,
                shard_world=P,
            )
            for i, (t, u) in enumerate(zip(self.sharded, SHARD_UNITS))
        ]
        self.regions = regions
        self.by_addr = {t.data_ptr(): t for t in self.whole + self.sharded}
        self.codec = M3ShardedPageCodec(regions, num_blocks=BLOCKS, device="cpu")


@pytest.fixture
def cpu_copies(monkeypatch):
    """`DSV4PageSlotCodec.gather` / `scatter` replayed on CPU: item i of a
    section is its regions' units, region after region, at ``buffer_offset + i
    * bytes_per_item`` (what the Triton kernels do)."""
    pools: list[Pool] = []

    def tensors(codec):
        return [
            next(p.by_addr for p in pools if r.base_addr in p.by_addr)[r.base_addr]
            for r in codec.page_regions
        ]

    def copy(codec, plan, buf, gather):
        regions = list(zip(codec.page_regions, tensors(codec)))
        for section in plan.sections:
            for i, item in enumerate(section.item_ids):
                at = section.buffer_offset + i * codec.bytes_per_block
                for region, tensor in regions:
                    unit = region.unit_bytes
                    src = tensor[item * unit : (item + 1) * unit]
                    dst = buf[at : at + unit]
                    (dst if gather else src).copy_(src if gather else dst)
                    at += unit

    monkeypatch.setattr(
        DSV4PageSlotCodec,
        "gather",
        lambda codec, plan, dst, stream=None: copy(codec, plan, dst, True),
    )
    monkeypatch.setattr(
        DSV4PageSlotCodec,
        "scatter",
        lambda codec, src, plan, stream=None: copy(codec, plan, src, False),
    )
    return pools


def _rows(pool, block, rank):
    """The bytes chunk-major staging holds for ``block``'s own index rows."""
    if block % P != rank:
        return torch.empty(0, dtype=torch.uint8)
    unit = block // P
    return torch.cat(
        [t[unit * u : (unit + 1) * u] for t, u in zip(pool.sharded, SHARD_UNITS)]
    )


def _whole(pool, block):
    return torch.cat(
        [t[block * u : (block + 1) * u] for t, u in zip(pool.whole, WHOLE_UNITS)]
    )


def _connector(codec, chunk_blocks):
    from atom.kv_transfer.offload._block_gpu_connector import BlockGPUConnector

    return BlockGPUConnector(codec, block_size=1, chunk_size=chunk_blocks)


def _store(connector, codec, objs, starts, ends, block_ids):
    """`batched_from_gpu`'s staged pipeline, group by group, on CPU: pack every
    chunk of a group into one staging buffer, then slice it into the objects."""
    groups = connector._ranges_to_block_ids(starts, ends, block_ids=block_ids)
    chunks = connector._iter_transfer_chunks(objs, groups, starts, ends)
    for group in connector._iter_transfer_groups(chunks):
        buf = torch.zeros(group.nbytes, dtype=torch.uint8)
        codec.gpu_to_chunk_major_device_buffer(buf, connector._group_block_ids(group))
        connector._slice_to_memory_objs(group, buf)


def _load(connector, codec, objs, starts, ends, block_ids):
    groups = connector._ranges_to_block_ids(starts, ends, block_ids=block_ids)
    chunks = connector._iter_transfer_chunks(objs, groups, starts, ends)
    for group in connector._iter_transfer_groups(chunks):
        buf = torch.zeros(group.nbytes, dtype=torch.uint8)
        connector._memory_objs_to_slice(group, buf)
        codec.chunk_major_device_buffer_to_gpu(buf, connector._group_block_ids(group))


# Logical blocks 0..5; block ids in the lanes of their positions, on both ends.
SRC_TABLE = [4, 9, 6, 11, 8, 13]
DST_TABLE = [12, 1, 2, 7, 0, 5]


@pytest.mark.parametrize("batched", [True, False])
@pytest.mark.parametrize("chunk_blocks", [1, 2])
@pytest.mark.parametrize("rank", range(P))
def test_a_ranks_chunks_are_exactly_its_bytes_and_round_trip(
    cpu_copies, monkeypatch, rank, chunk_blocks, batched
):
    from atom.kv_transfer.offload import _block_gpu_connector

    monkeypatch.setattr(_block_gpu_connector, "_BATCHED_HOST_COPY", batched)
    source, dest = Pool(rank, seed=rank), Pool(rank, seed=None)
    cpu_copies += [source, dest]
    whole, shard = sum(WHOLE_UNITS), sum(SHARD_UNITS)
    assert (source.codec.bytes_per_block, source.codec.sharded_bytes_per_block) == (
        whole,
        shard,
    )
    save = _connector(source.codec, chunk_blocks)
    # The staging chunk is the most a chunk can hold; a chunk's object is exact.
    assert save.gpu_staging_chunk_bytes == chunk_blocks * whole + shard
    n = len(SRC_TABLE) // chunk_blocks
    starts = [i * chunk_blocks for i in range(n)]
    ends = [s + chunk_blocks for s in starts]
    shapes = save.chunk_shapes(starts, ends, block_ids=SRC_TABLE)
    expected = [
        torch.cat(
            [_whole(source, b) for b in SRC_TABLE[s:e]]
            + [_rows(source, b, rank) for b in SRC_TABLE[s:e]]
        )
        for s, e in zip(starts, ends)
    ]
    assert [int(shape[0]) for shape in shapes] == [t.numel() for t in expected]
    objs = [
        SimpleNamespace(tensor=torch.zeros(shape, dtype=torch.uint8))
        for shape in shapes
    ]
    _store(save, source.codec, objs, starts, ends, SRC_TABLE)
    for obj, want in zip(objs, expected):
        assert torch.equal(obj.tensor, want)

    load = _connector(dest.codec, chunk_blocks)
    _load(load, dest.codec, objs, starts, ends, DST_TABLE)
    for s, d in zip(SRC_TABLE, DST_TABLE):
        assert torch.equal(_whole(dest, d), _whole(source, s))
        assert torch.equal(_rows(dest, d, rank), _rows(source, s, rank))
    # Nothing but the rank's own units was written.
    owned_dst = {d // P for d in DST_TABLE if d % P == rank}
    for td, u in zip(dest.sharded, SHARD_UNITS):
        for unit in set(range(BLOCKS // P)) - owned_dst:
            assert not td[unit * u : (unit + 1) * u].any()


def test_a_token_path_object_of_the_most_bytes_loads_the_same(cpu_copies):
    """LMCache's own store sizes a chunk by tokens (`get_shapes`): the object
    is the most a chunk holds and its bytes past the chunk's are never read."""
    rank = 1
    source, dest = Pool(rank, seed=7), Pool(rank, seed=None)
    cpu_copies += [source, dest]
    save, load = _connector(source.codec, 1), _connector(dest.codec, 1)
    starts, ends = list(range(6)), list(range(1, 7))
    big = [
        SimpleNamespace(
            tensor=torch.full((save.gpu_staging_chunk_bytes,), 0xAB, dtype=torch.uint8)
        )
        for _ in starts
    ]
    _store(save, source.codec, big, starts, ends, SRC_TABLE)
    _load(load, dest.codec, big, starts, ends, DST_TABLE)
    for s, d in zip(SRC_TABLE, DST_TABLE):
        assert torch.equal(_whole(dest, d), _whole(source, s))
        assert torch.equal(_rows(dest, d, rank), _rows(source, s, rank))


def test_an_object_smaller_than_its_chunk_is_refused(cpu_copies):
    """An object stored for a chunk that owns no index rows cannot load into
    one that does (both ends broke the lane rule): refused, not half-read."""
    rank = 0
    pool = Pool(rank, seed=None)
    cpu_copies.append(pool)
    connector = _connector(pool.codec, 1)
    small = SimpleNamespace(
        tensor=torch.zeros(pool.codec.bytes_per_block, dtype=torch.uint8)
    )
    with pytest.raises(ValueError, match="too small"):
        _load(connector, pool.codec, [small], [0], [1], [4])


def test_a_codec_needs_one_shard_and_whole_regions():
    t = torch.zeros(64, dtype=torch.uint8)
    whole = KVTransferRegion(t.data_ptr(), 64, 4, semantic_role="k")
    shard = KVTransferRegion(
        t.data_ptr(), 16, 4, semantic_role="index", shard_rank=1, shard_world=P
    )
    other = KVTransferRegion(
        t.data_ptr(), 16, 4, semantic_role="index2", shard_rank=2, shard_world=P
    )
    for regions in ([shard], [whole, shard, other]):
        with pytest.raises(ValueError, match="one shard"):
            M3ShardedPageCodec(regions, num_blocks=16, device="cpu")


def _namespace_config(**overrides):
    config = SimpleNamespace(
        model="m",
        model_tag="m",
        kv_cache_dtype="fp8",
        index_cache_dtype="fp4",
        kv_cache_block_size=128,
        tensor_parallel_size=4,
        decode_context_parallel_size=1,
        speculative_config=None,
        hf_config=SimpleNamespace(num_hidden_layers=60, head_dim=128),
    )
    for name, value in overrides.items():
        setattr(config, name, value)
    return config


def test_sharded_and_replicated_tiers_never_share_a_namespace():
    cfg = SimpleNamespace(chunk_size=256)
    plain = offcfg.build_page_namespace(_namespace_config(), cfg, 4)
    off = offcfg.build_page_namespace(
        _namespace_config(dcp_config=DCPConfig(indexer_dcp_only=True)), cfg, 4
    )
    sharded = offcfg.build_page_namespace(
        _namespace_config(
            dcp_config=DCPConfig(indexer_dcp_only=True, index_shard=True)
        ),
        cfg,
        4,
    )
    assert off == plain
    assert sharded != plain and sharded.startswith("m::atom-page-v3-")


def test_the_m3_worker_registers_a_ranks_chunks(monkeypatch):
    from test_lmcache_offload_connector import _install_registration_dependencies

    from atom.kv_transfer.offload.hybrid.m3.connector import M3OffloadConnector

    captured = _install_registration_dependencies(monkeypatch)  # chunk: 2 blocks
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)
    worker = M3OffloadConnector.__new__(M3OffloadConnector)
    worker._config = SimpleNamespace(kv_transfer_config={}, kv_cache_block_size=4)
    worker.block_size = worker.virtual_block_size = 4
    worker.chunk_size = None
    worker._do_save = worker._do_load = True
    worker._engine = worker._codec = worker._lookup_server = None
    pool = Pool(rank=2, seed=0)  # holds the regions' memory
    tensors = SimpleNamespace(block_regions=pool.regions, num_blocks=BLOCKS)

    worker.register_kv_caches({}, tensors, num_blocks=BLOCKS)

    assert isinstance(worker._codec, M3ShardedPageCodec)
    whole_bytes, shard_bytes = sum(WHOLE_UNITS), sum(SHARD_UNITS)
    assert captured["metadata"].get_shapes(8) == [
        torch.Size((2 * whole_bytes + shard_bytes,))
    ]
    assert captured["gpu_connector"].gpu_staging_chunk_bytes == (
        2 * whole_bytes + shard_bytes
    )


def test_the_prepared_path_reads_the_index_rows_past_the_whole_bytes(
    cpu_copies, monkeypatch
):
    """One block-id upload per transfer, as the replicated codec: each part's
    prepared group copies what `gpu_to_chunk_major_device_buffer` does."""

    def prepare(codec, groups, *, device, stream):
        return SimpleNamespace(
            groups=[list(g) for g in groups],
            upload_count=int(any(groups)),
            codec=codec,
        )

    def prepared(gather):
        def run(codec, buf, owner, index, *, stream=None):
            plan = codec.page_plan(owner.groups[index])
            (codec.gather(plan, buf) if gather else codec.scatter(buf, plan))

        return run

    monkeypatch.setattr(DSV4PageSlotCodec, "prepare_block_id_groups", prepare)
    monkeypatch.setattr(
        DSV4PageSlotCodec, "gpu_to_chunk_major_device_buffer_prepared", prepared(True)
    )
    monkeypatch.setattr(
        DSV4PageSlotCodec, "chunk_major_device_buffer_to_gpu_prepared", prepared(False)
    )
    rank = 3
    source, dest = Pool(rank, seed=3), Pool(rank, seed=None)
    cpu_copies += [source, dest]
    groups = [[[b] for b in SRC_TABLE[:3]], [[b] for b in SRC_TABLE[3:]]]
    owner = source.codec.prepare_block_id_groups(groups, device="cpu", stream=None)
    assert owner.group_count == 2 and owner.upload_count == 2
    for index, group in enumerate(groups):
        sizes, _ = source.codec.staging_layout(group)
        want = torch.zeros(sum(sizes), dtype=torch.uint8)
        got = torch.zeros(sum(sizes), dtype=torch.uint8)
        source.codec.gpu_to_chunk_major_device_buffer(want, group)
        source.codec.gpu_to_chunk_major_device_buffer_prepared(got, owner, index)
        assert torch.equal(got, want)
        dst_group = [[d] for d in DST_TABLE[3 * index : 3 * index + 3]]
        dst_owner = dest.codec.prepare_block_id_groups(
            [dst_group], device="cpu", stream=None
        )
        dest.codec.chunk_major_device_buffer_to_gpu_prepared(got, dst_owner, 0)
    for s, d in zip(SRC_TABLE, DST_TABLE):
        assert torch.equal(_whole(dest, d), _whole(source, s))
        assert torch.equal(_rows(dest, d, rank), _rows(source, s, rank))


@pytest.mark.parametrize("rank", range(P))
def test_a_prehashed_save_allocates_each_chunk_its_own_bytes(cpu_copies, rank):
    """The lazy policy's demotions (1-block chunks) are prehashed saves: each
    chunk is allocated its exact bytes, so 3 of 4 carry no index."""
    from test_offload_prehashed_save import CHUNK, _engine, _worker

    from atom.kv_transfer.disaggregation.types import SaveOperationId
    from atom.kv_transfer.offload.metadata import (
        LMCacheReqMeta,
        PrehashedSave,
        SaveSpec,
    )

    pool = Pool(rank, seed=None)
    cpu_copies.append(pool)
    engine = _engine()
    from atom.kv_transfer.offload._block_gpu_connector import BlockGPUConnector

    # One block a chunk, as the lazy policy runs.
    connector = BlockGPUConnector(pool.codec, block_size=CHUNK, chunk_size=CHUNK)
    asked = []
    engine.gpu_connector.chunk_shapes = connector.chunk_shapes
    allocate = engine.storage_manager.allocate

    def sized(shapes, dtypes, busy_loop=True, fmt=None):
        asked.append(int(shapes[0][0]))
        return allocate(shapes, dtypes, busy_loop, fmt)

    engine.storage_manager.allocate = sized
    worker = _worker(engine)
    blocks = SRC_TABLE[:4]  # lanes 0, 1, 2, 3
    req = LMCacheReqMeta(
        req_id=1,
        token_ids=[],
        block_ids=blocks,
        save_spec=SaveSpec(skip_leading_tokens=0),
        save_operation=SaveOperationId(1, 0),
        prehashed_save=PrehashedSave(start=0, hashes=(11, 12, 13, 14)),
    )
    worker._do_save_req(req)
    whole, shard = pool.codec.bytes_per_block, pool.codec.sharded_bytes_per_block
    assert asked == [whole + shard * (b % P == rank) for b in blocks]


def test_m31_tp4_bytes_per_token_and_the_tier_horizon(monkeypatch):
    """M3.1 TP4 per rank and token: 14,280 B replicated; 11,040 B on average
    with the index sharded (HBM, and the CPU tier with exact chunk objects);
    the tier index (`lmcache_tier_horizon_tokens`) counts the latter."""
    from test_lmcache_offload_connector import _m31_pp4_config

    from atom import config as config_mod
    from atom.models.pp_partition import minimax_m3_stage_kv_costs

    monkeypatch.setenv("ATOM_M3_SPARSE_KV_NVFP4", "1")
    monkeypatch.setattr(config_mod, "_is_minimax_m3_config", lambda hf: True)
    monkeypatch.setattr(
        offcfg,
        "build_lmcache_config",
        lambda _kvc=None: SimpleNamespace(max_local_cpu_size=430.0),
    )
    config = _m31_pp4_config(0)
    config.tensor_parallel_size, config.pipeline_parallel_size = 4, 1
    tokens = {}
    for shard in (False, True):
        config.dcp_config = DCPConfig(indexer_dcp_only=shard, index_shard=shard)
        layers, draft = minimax_m3_stage_kv_costs(config)
        tokens[shard] = sum(layers) + draft
    assert tokens == {False: 14_280, True: 11_040}
    # Exact chunk objects: a 1-block chunk of an owned block holds the whole
    # index, others none; on average 1/4 of it, as the HBM pool's price.
    whole, index = 1_274_880, 552_960
    assert (whole + index / P) / 128 == 11_040
    assert offcfg.lmcache_tier_horizon_tokens(config) == int(430 * 2**30 / 11_040)
