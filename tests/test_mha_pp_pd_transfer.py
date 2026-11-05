# SPDX-License-Identifier: MIT
"""MHA KV from a PP prefill (TP1 stages) into a TP-sharded decode.

The MHA backend's own export numbers a stage's units as an unsplit model does
and says how each splits by KV head (MiniMax-M3: page-16 sparse layers, dense
layers, the replicated indexer keys). The byte test drives the real descriptor
builder against host buffers, so a wrong region or head offset shows up as a
wrong byte.
"""

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from atom.kv_transfer.offload.mp.page_views import _build_cache_views
from atom.model_ops.attentions.mha_kv_pool import MhaKvPool
from atom.model_ops.attentions.pool_layout.entry_arena import EntryField
from atom.model_ops.attentions.pool_layout.pool_rows import KvGeometry

# `exc_type`: collected after a test that stubs `sys.modules["aiter"]`, the
# import reaches the stub and fails on a symbol rather than on the module.
aiter_attention = pytest.importorskip(
    "atom.model_ops.attentions.aiter_attention", exc_type=ImportError
)
mc = pytest.importorskip(
    "atom.kv_transfer.disaggregation.mooncake.mooncake_connector",
    exc_type=ImportError,
)
from atom.model_ops.attention_mha import (
    PagedAttentionImpl,
    SparseMHAPagedAttentionImpl,
)

BLOCK, HEAD_DIM, LAYERS = 128, 128, 8
SPARSE = (0, 2, 6)  # the layers with an indexer
# Layers per stage, as VLLM_PP_LAYER_PARTITION gives them.
PARTITIONS = [(2, 2, 2, 2), (3, 2, 2, 1)]


class _Layer:
    """An attention module as the backend walks it, with its real impl class:
    the sparse one on an indexed layer, the paged one elsewhere."""

    def __init__(self, layer_num):
        self.layer_num = layer_num
        if layer_num in SPARSE:
            self.impl = object.__new__(SparseMHAPagedAttentionImpl)
            self.impl.sparse_layer_ordinal = SPARSE.index(layer_num)
        else:
            self.impl = object.__new__(PagedAttentionImpl)


def _builder(layers, heads, kv_dtype=torch.float8_e4m3fn):
    modules = [_Layer(n) for n in layers]
    indexed = [m for m in modules if m.layer_num in SPARSE]
    pool = MhaKvPool(
        layers=len(modules),
        block_size=BLOCK,
        num_kv_heads=heads,
        head_dim=HEAD_DIM,
        kv_dtype=kv_dtype,
        extra_fields=[EntryField("index", len(indexed), (BLOCK * 128,), torch.uint8)],
    )
    pool.allocate(4, "cpu")
    geometry = KvGeometry(heads, HEAD_DIM)
    builder = object.__new__(aiter_attention.AiterAttentionMetadataBuilder)
    builder.kv_pools = {geometry: pool}
    builder._pool_rows = {
        geometry: {m: row for row, m in enumerate(modules)},
        aiter_attention.INDEX_ROWS: {m: row for row, m in enumerate(indexed)},
    }
    builder.block_size, builder.block_ratio = BLOCK, 1
    builder.model_runner = SimpleNamespace(
        config=SimpleNamespace(
            hf_config=SimpleNamespace(num_hidden_layers=LAYERS),
            pipeline_parallel_size=1,
        )
    )
    return builder


def _pp(monkeypatch, builder, rank, size):
    builder.model_runner.config.pipeline_parallel_size = size
    monkeypatch.setattr(
        aiter_attention,
        "get_pp_group",
        lambda: SimpleNamespace(rank_in_group=rank, world_size=size),
    )
    return builder


@pytest.fixture
def export(monkeypatch):
    """The backend's export on stage `rank` of `partition`, or unsplit."""

    def _export(heads, rank=0, partition=None, kv_dtype=torch.float8_e4m3fn):
        sizes = partition or (LAYERS,)
        monkeypatch.setenv("VLLM_PP_LAYER_PARTITION", ",".join(map(str, sizes)))
        start = sum(sizes[:rank])
        builder = _builder(range(start, start + sizes[rank]), heads, kv_dtype)
        return _pp(monkeypatch, builder, rank, len(sizes)).get_kv_transfer_tensors()

    return _export


def _unit(role):
    """A region's name without the per-rank head count a head cut changes."""
    return "mha." + role.split(".", 2)[2] if role.startswith("mha.") else role


def _layout(tt):
    return [
        (p.region.semantic_role, p.head_major_runs, p.region.unit_bytes)
        for p in tt.pages
    ]


def test_an_unsplit_model_keeps_the_backend_names_and_splits_by_head(export):
    """Without PP the names are the ones a backend gave before, so TP4 -> TP4
    P/D and LMCache MP see the same units on either side of this change."""
    tt = export(heads=4)
    pool = _builder(range(LAYERS), 4).kv_pools[KvGeometry(4, HEAD_DIM)]
    geometry = KvGeometry(4, HEAD_DIM)
    assert [r.semantic_role for r in tt.block_regions] == [
        f"mha.{geometry}.{role}" for role, _ in pool.region_tensors()
    ]
    assert tt.block_region_consumer_indices is None
    runs = {p.region.semantic_role: p.head_major_runs for p in tt.pages}
    name = f"mha.{geometry}.{{}}.layer_{{}}".format
    assert runs[name("k", 0)] == runs[name("v_scale", 6)] == (8, 4)  # page-16
    assert runs[name("v", 1)] == (1, 4)  # dense: one page-128 run
    assert [runs[name("index", i)] for i in range(len(SPARSE))] == [None] * 3
    # LMCache MP takes the same units, still one stored shard per rank.
    assert tt.tp_replication_factor == 1
    tt.set_block_count(4)
    assert len(_build_cache_views(tt, num_blocks=4).tensors) == len(tt.pages)


@pytest.mark.parametrize("heads", [1, 4])
@pytest.mark.parametrize("kv_dtype", [torch.float8_e4m3fn, torch.bfloat16])
@pytest.mark.parametrize("partition", PARTITIONS)
def test_every_stage_maps_into_the_full_consumer_layout(
    export, heads, kv_dtype, partition
):
    consumer = [r.semantic_role for r in export(1, kv_dtype=kv_dtype).block_regions]
    claimed = []
    for rank in range(len(partition)):
        tt = export(heads, rank, partition, kv_dtype)
        indices = tt.block_region_consumer_indices
        assert len(indices) == len(tt.pages)
        for page, index in zip(tt.pages, indices):
            assert _unit(page.region.semantic_role) == _unit(consumer[index])
        claimed.extend(indices)
    assert sorted(claimed) == list(range(len(consumer)))


def test_a_stage_whose_layers_are_not_all_kv_keeps_local_rows(monkeypatch):
    """A hybrid's linear layers hold no KV row, so a layer's row is not the
    layer; such a stage keeps the rows it had."""
    builder = _pp(monkeypatch, _builder([2], 4), rank=1, size=4)
    tt = builder.get_kv_transfer_tensors()  # layer 3 holds no KV row
    assert tt.block_region_consumer_indices is None
    assert tt.pages[0].region.semantic_role.endswith(".k.layer_0")


def test_page16_runs_are_how_the_sparse_writer_lays_heads_out():
    sparse_attn = pytest.importorskip(
        "atom.model_ops.minimax_m3.sparse_attn", exc_type=ImportError
    )
    assert SparseMHAPagedAttentionImpl.shuffle_page_tokens == sparse_attn.ASM_PAGE_SIZE
    builder = _builder([0], heads=4)
    pool = next(iter(builder.kv_pools.values()))
    views = sparse_attn.to_page16_shuffle(*pool.kv_views(0), *pool.scale_views(0))
    for view in views:
        for head in range(4):
            view.view(torch.uint8)[:, head].fill_(head + 1)
    pages = [p for p in builder.get_kv_transfer_tensors().pages if p.head_major_runs]
    assert len(pages) == 4
    for page in pages:
        runs, heads = page.head_major_runs
        unit = page.view[2].view(runs, heads, -1)
        for head in range(heads):
            assert torch.all(unit[:, head] == head + 1)


class _Memory:
    """Host buffers at disjoint fake addresses, written by a fake RDMA."""

    def __init__(self):
        self.buffers, self.next = [], 1 << 40
        self.writes = 0

    def add(self, nbytes, fill):
        buf = np.zeros(nbytes, np.uint8)
        if fill:
            buf[:] = np.random.default_rng(nbytes + len(self.buffers)).integers(
                0, 256, nbytes, np.uint8
            )
        self.buffers.append((self.next, buf))
        self.next += 1 << 32
        return self.buffers[-1][0]

    def at(self, addr, n):
        for base, buf in self.buffers:
            if base <= addr and addr + n <= base + buf.size:
                return buf[addr - base : addr - base + n]
        raise AssertionError(hex(addr))

    def write(self, srcs, dsts, sizes):
        assert len(srcs) <= 7  # Exercise flushing during the head-cut path too.
        self.writes += 1
        for s, d, n in zip(srcs, dsts, sizes):
            self.at(d, n)[:] = self.at(s, n)


def _producer(tt, mem, blocks, heads):
    layout = _layout(tt)
    producer = object.__new__(mc.MooncakeConnector)
    producer.__dict__.update(
        kv_caches_base_addr=[mem.add(blocks * b, True) for _, _, b in layout],
        _per_block_bytes_list=[b for _, _, b in layout],
        _block_region_roles=[r for r, _, _ in layout],
        _block_head_major_runs=[runs for _, runs, _ in layout],
        _block_region_consumer_indices=tt.block_region_consumer_indices,
        tp_size=4 // heads,
        dcp_size=1,
        pp_size=4,
        block_size=BLOCK,
        _mla_staging=None,
        _MAX_RDMA_ENTRIES_PER_BATCH=7,
        _rdma_write=lambda target, s, d, n, *a, **k: mem.write(s, d, n) or True,
    )
    return producer


def _request(theirs, bases, rank, runs=True):
    request = {
        "consumer_base_addrs": bases,
        "consumer_region_roles": [r for r, _, _ in theirs],
        "consumer_block_bpb": [b for _, _, b in theirs],
        "consumer_tp_size": 4,
        "consumer_tp_rank": rank,
    }
    if runs:
        request["consumer_head_major_runs"] = [r for _, r, _ in theirs]
    return request


@pytest.mark.parametrize("heads", [1, 4])
@pytest.mark.parametrize("rank, partition", [(0, (2, 2, 2, 2)), (3, (3, 2, 2, 1))])
def test_stage_fills_each_tp4_rank_with_its_head(export, heads, rank, partition):
    """`heads` 4 is a TP1 stage, cut by head; 1 is a TP4 stage, copied whole."""
    mem, blocks, src_ids, dst_ids = _Memory(), 4, [3, 1], [0, 2]
    theirs = _layout(export(1))  # a TP4 decode rank: every layer, one head
    tt = export(heads, rank, partition)
    layout = _layout(tt)
    producer = _producer(tt, mem, blocks, heads)
    consumers = []
    for consumer_rank in range(4):
        bases = [mem.add(blocks * b, False) for _, _, b in theirs]
        consumers.append({_unit(r): (a, b) for (r, _, b), a in zip(theirs, bases)})
        assert producer._execute_block_transfer(
            _request(theirs, bases, consumer_rank),
            "consumer",
            src_ids,
            dst_ids,
            f"req-{consumer_rank}",
        )
    for (role, runs, bpb), base in zip(layout, producer.kv_caches_base_addr):
        src = mem.at(base, blocks * bpb).reshape(blocks, -1)
        for consumer_rank, regions in enumerate(consumers):
            dst_base, dst_bpb = regions[_unit(role)]
            dst = mem.at(dst_base, blocks * dst_bpb).reshape(blocks, -1)
            for s, d in zip(src_ids, dst_ids):
                want = (
                    src[s]
                    if runs is None or heads == 1
                    else src[s].reshape(runs[0], heads, -1)[:, consumer_rank]
                )
                assert np.array_equal(dst[d], want.ravel()), (role, consumer_rank)
            assert not dst[[1, 3]].any()
    held = {_unit(role) for role, _, _ in layout}
    for regions in consumers:
        for role, (base, bpb) in regions.items():
            if role not in held:
                assert not mem.at(base, blocks * bpb).any()
    assert mem.writes > 4


@pytest.mark.parametrize("bad_layout", ["reordered", "short"])
def test_incompatible_consumer_layout_is_refused(export, bad_layout):
    mem = _Memory()
    producer = _producer(export(4, 3, (3, 2, 2, 1)), mem, 4, 4)
    roles = [r.semantic_role for r in export(1).block_regions]
    bases = [0] * len(roles)
    if bad_layout == "reordered":
        roles.reverse()
    else:
        bases, roles = bases[:-1], roles[:-1]
    with pytest.raises((RuntimeError, ValueError), match="role mismatch|exceeds"):
        producer._execute_block_transfer(
            {
                "consumer_base_addrs": bases,
                "consumer_region_roles": roles,
                "consumer_tp_size": 1,
            },
            "consumer",
            [0],
            [0],
            "bad-layout",
        )
    assert mem.writes == 0


@pytest.mark.parametrize(
    "heads, refusal",
    [(4, "role mismatch|byte-size mismatch"), (2, "needs a TP1 producer")],
)
def test_an_older_consumer_is_refused_before_anything_is_written(
    export, heads, refusal
):
    """A consumer that does not describe its head runs gets no cut, so a TP1
    stage's layout checks name the region that differs. A producer that may
    not cut is refused whatever the consumer sends: rank `c` would pull
    another rank's heads from producer rank `c % P`."""
    mem = _Memory()
    producer = _producer(export(heads, 3, (3, 2, 2, 1)), mem, 4, heads)
    theirs = _layout(export(1))
    with pytest.raises(RuntimeError, match=refusal):
        producer._execute_block_transfer(
            _request(theirs, [mem.add(4 * b, False) for _, _, b in theirs], 0, False),
            "consumer",
            [0],
            [0],
            "old-consumer",
        )
    assert mem.writes == 0


def test_a_head_cut_matches_names_up_to_the_head_count_only():
    same = mc._same_unit_role
    assert same("mha.h4d128.k.layer_3", "mha.h1d128.k.layer_3", True)
    assert not same("mha.h4d128.k.layer_3", "mha.h1d128.k.layer_3", False)
    assert not same("mha.h4d128.k.layer_3", "mha.h1d128.v.layer_3", True)
    assert not same("mha.h4d128.k.layer_3", "mha.h1d128.k.layer_4", True)
    assert not same("draft.k.layer_0", "draft.v.layer_0", True)
