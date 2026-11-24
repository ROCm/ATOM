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


@pytest.mark.parametrize("rank", range(P))
def test_a_ranks_chunks_round_trip_its_own_index_rows(cpu_copies, rank):
    source, dest = Pool(rank, seed=rank), Pool(rank, seed=None)
    cpu_copies += [source, dest]
    codec = source.codec
    whole, shard = sum(WHOLE_UNITS), sum(SHARD_UNITS)
    assert (codec.bytes_per_block, codec.sharded_bytes_per_block) == (whole, shard)
    # Logical blocks 0..4 as 2-block chunks; block ids in the lanes of their
    # positions, on both ends.
    src_chunks = [[4, 9], [6, 11], [8]]
    dst_chunks = [[12, 1], [2, 7], [0]]
    sizes = [codec.chunk_bytes(len(c)) for c in src_chunks]
    assert sizes == [2 * whole + shard] * 2 + [whole + shard]
    buf = torch.zeros(sum(sizes), dtype=torch.uint8)
    codec.gpu_to_chunk_major_device_buffer(buf, src_chunks)

    # The bytes of chunk 0 are its blocks' whole regions, then its own rows.
    first = buf[: sizes[0]]
    own = [b for b in src_chunks[0] if b % P == rank]
    rows = [t[b // P * u : (b // P + 1) * u]
            for b in own for t, u in zip(source.sharded, SHARD_UNITS)]  # fmt: skip
    rows = torch.cat(rows) if rows else torch.empty(0, dtype=torch.uint8)
    assert torch.equal(first[2 * whole : 2 * whole + rows.numel()], rows)

    dest.codec.chunk_major_device_buffer_to_gpu(buf, dst_chunks)
    for src_ids, dst_ids in zip(src_chunks, dst_chunks):
        for s, d in zip(src_ids, dst_ids):
            for ts, td, u in zip(source.whole, dest.whole, WHOLE_UNITS):
                assert torch.equal(td[d * u : (d + 1) * u], ts[s * u : (s + 1) * u])
            for ts, td, u in zip(source.sharded, dest.sharded, SHARD_UNITS):
                landed = td[d // P * u : (d // P + 1) * u]
                if s % P == rank:
                    assert torch.equal(landed, ts[s // P * u : (s // P + 1) * u])
    # Nothing but the rank's own units was written.
    owned_dst = {d // P for c in dst_chunks for d in c if d % P == rank}
    for td, u in zip(dest.sharded, SHARD_UNITS):
        for unit in set(range(BLOCKS // P)) - owned_dst:
            assert not td[unit * u : (unit + 1) * u].any()


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
