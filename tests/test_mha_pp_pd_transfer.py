# SPDX-License-Identifier: MIT
"""MHA KV from a PP prefill (TP1 stages) into a TP-sharded decode, byte by byte.

Drives the real Mooncake descriptor builder over host buffers, so a wrong
region or head offset shows up as a wrong byte. Unit layout and pairing alone
are in `test_page_unit_layout.py`, which needs no aiter.
"""

import msgpack
import numpy as np
import pytest
import torch

from atom.kv_transfer.disaggregation.page_region import page_region
from atom.model_ops.attentions.mha_kv_pool import MhaKvPool
from atom.model_ops.attentions.pool_layout.entry_arena import EntryField
from atom.model_ops.attentions.pool_layout.pool_rows import KvGeometry
from tests.test_page_unit_layout import (
    BLOCK,
    HEAD_DIM,
    LAYERS,
    MTP,
    SPARSE,
    stage_layers,
    stage_pages,
)

# `exc_type`: collected after a test that stubs `sys.modules["aiter"]`, the
# import reaches the stub and fails on a symbol rather than on the module.
mc = pytest.importorskip(
    "atom.kv_transfer.disaggregation.mooncake.mooncake_connector",
    exc_type=ImportError,
)
aiter_attention = pytest.importorskip(
    "atom.model_ops.attentions.aiter_attention", exc_type=ImportError
)
from atom.model_ops.attention_mha import (
    PagedAttentionImpl,
    SparseMHAPagedAttentionImpl,
)


class _Memory:
    """Host buffers at disjoint fake addresses, written by a fake RDMA."""

    def __init__(self):
        self.buffers, self.next = [], 1 << 40
        self.writes = 0

    def add(self, nbytes, fill):
        buf = np.zeros(nbytes, np.uint8)
        if fill:
            rng = np.random.default_rng(nbytes + len(self.buffers))
            buf[:] = rng.integers(0, 256, nbytes, np.uint8)
        self.buffers.append((self.next, buf))
        self.next += 1 << 32
        return self.buffers[-1][0]

    def at(self, addr, n):
        for base, buf in self.buffers:
            if base <= addr and addr + n <= base + buf.size:
                return buf[addr - base : addr - base + n]
        raise AssertionError(hex(addr))

    def write(self, srcs, dsts, sizes):
        assert len(srcs) <= 7  # Flushes mid-region on the head-cut path too.
        self.writes += 1
        for s, d, n in zip(srcs, dsts, sizes):
            self.at(d, n)[:] = self.at(s, n)


def _producer(pages, mem, heads, partition, rank, monkeypatch):
    monkeypatch.setenv("VLLM_PP_LAYER_PARTITION", ",".join(map(str, partition)))
    producer = object.__new__(mc.MooncakeConnector)
    bases = [mem.add(4 * p.region.unit_bytes, True) for p in pages]
    producer.__dict__.update(
        kv_caches_base_addr=bases,
        _block_regions=[(b, p.region.unit_bytes) for b, p in zip(bases, pages)],
        _per_block_bytes_list=[p.region.unit_bytes for p in pages],
        _block_region_roles=[p.region.semantic_role for p in pages],
        _block_head_major_runs=tuple(p.head_major_runs for p in pages),
        _block_unit_specs=(
            mc._pack_unit_layout(pages) and tuple(p.unit_spec for p in pages)
        ),
        _block_region_consumer_indices=None,
        _unit_plans={},
        _slot_regions=[],
        _swa_block_regions=[],
        _fp4_index_layout=False,
        _num_local_layers=len(pages),
        tp_size=4 // heads,
        dcp_size=1,
        pp_size=len(partition),
        pp_rank=rank,
        num_hidden_layers=LAYERS,
        _start_layer=sum(partition[:rank]),
        block_size=BLOCK,
        _mla_staging=None,
        _MAX_RDMA_ENTRIES_PER_BATCH=7,
        _rdma_write=lambda target, s, d, n, *a, **k: mem.write(s, d, n) or True,
    )
    return producer


def _request(theirs, bases, rank):
    request = {
        "consumer_base_addrs": bases,
        "consumer_region_roles": [p.region.semantic_role for p in theirs],
        "consumer_block_bpb": [p.region.unit_bytes for p in theirs],
        "consumer_unit_layout": mc._pack_unit_layout(theirs),
        "consumer_tp_size": 4,
        "consumer_tp_rank": rank,
    }
    # The producer receives what the wire carries, not our dataclasses.
    return msgpack.loads(msgpack.dumps(request))


@pytest.mark.parametrize("heads", [1, 4])
@pytest.mark.parametrize("rank, partition", [(0, (2, 2, 2, 2)), (3, (3, 2, 2, 1))])
@pytest.mark.parametrize("draft", [False, True])
def test_stage_fills_each_tp4_rank_with_its_head(
    heads, rank, partition, draft, monkeypatch
):
    """`heads` 4 is a TP1 stage, cut by head; 1 is a TP4 stage, copied whole.
    The last stage also holds the MTP row and, with `draft`, a draft pool."""
    mem, src_ids, dst_ids = _Memory(), [3, 1], [0, 2]
    theirs = stage_pages([*range(LAYERS), MTP], 1, draft=draft)
    layers, _, _ = stage_layers(partition, rank)
    last = rank == len(partition) - 1
    ours = stage_pages(layers, heads, draft=draft and last)
    producer = _producer(ours, mem, heads, partition, rank, monkeypatch)
    consumers = []
    for consumer_rank in range(4):
        bases = [mem.add(4 * p.region.unit_bytes, False) for p in theirs]
        consumers.append({p.unit_spec: (b, p) for p, b in zip(theirs, bases)})
        assert producer._execute_block_transfer(
            _request(theirs, bases, consumer_rank),
            "consumer",
            src_ids,
            dst_ids,
            f"req-{consumer_rank}",
        )
    for page, base in zip(ours, producer.kv_caches_base_addr):
        src = mem.at(base, 4 * page.region.unit_bytes).reshape(4, -1)
        runs = page.head_major_runs
        for consumer_rank, regions in enumerate(consumers):
            dst_base, dst_page = regions[page.unit_spec]
            dst = mem.at(dst_base, 4 * dst_page.region.unit_bytes).reshape(4, -1)
            for s, d in zip(src_ids, dst_ids):
                want = src[s]
                if runs is not None and heads > 1:
                    want = want.reshape(runs[0], heads, -1)[:, consumer_rank]
                assert np.array_equal(dst[d], want.ravel()), page.unit_spec
            assert not dst[[1, 3]].any()
    held = {p.unit_spec for p in ours}
    for regions in consumers:
        for spec, (base, page) in regions.items():
            if spec not in held:
                assert not mem.at(base, 4 * page.region.unit_bytes).any()
    assert len(producer._unit_plans) == 4  # one plan per consumer rank


def _refused(producer, request, mem, match, slot=False):
    with pytest.raises((RuntimeError, NotImplementedError), match=match):
        if slot:
            request.update(
                consumer_block_base_addrs=request["consumer_base_addrs"],
                consumer_slot_base_addrs=[],
                consumer_slot_bps=[],
                dst_slot_index=0,
            )
            producer._execute_block_slot_transfer(
                request, "consumer", [0], [0], {"slot_index": 0}, "slot", None
            )
        else:
            producer._execute_block_transfer(request, "consumer", [0], [0], "bad")
    assert mem.writes == 0


@pytest.mark.parametrize(
    "consumer, rank, match",
    [
        ("no_layout", 3, "names no PAGE units"),
        ("other_index_format", 2, "unit format mismatch"),  # holds layer 6's keys
        ("draft_only_there", 3, "not held here"),
        ("fewer_regions", 3, "names .* units for"),
    ],
)
def test_incompatible_consumer_is_refused_before_any_write(
    consumer, rank, match, monkeypatch
):
    mem = _Memory()
    layers, _, _ = stage_layers((3, 2, 2, 1), rank)
    producer = _producer(
        stage_pages(layers, 4), mem, 4, (3, 2, 2, 1), rank, monkeypatch
    )
    index = (64, torch.bfloat16) if consumer == "other_index_format" else None
    theirs = stage_pages(
        [*range(LAYERS), MTP],
        1,
        draft=consumer == "draft_only_there",
        **({"index": index} if index else {}),
    )
    request = _request(
        theirs, [mem.add(4 * p.region.unit_bytes, False) for p in theirs], 0
    )
    if consumer == "no_layout":
        del request["consumer_unit_layout"]
    if consumer == "fewer_regions":
        request["consumer_base_addrs"].pop()
    _refused(producer, request, mem, match)


@pytest.mark.parametrize("slot", [False, True])
def test_a_producer_that_may_not_cut_heads_is_refused(slot, monkeypatch):
    """Consumer rank `c` pulls from producer rank `c % P`, which holds other
    heads than `c` keeps, so only a TP1 producer may cut."""
    mem = _Memory()
    layers, _, _ = stage_layers((2, 2, 2, 2), 0)
    producer = _producer(stage_pages(layers, 2), mem, 2, (2, 2, 2, 2), 0, monkeypatch)
    theirs = stage_pages([*range(LAYERS), MTP], 1)
    request = _request(
        theirs, [mem.add(4 * p.region.unit_bytes, False) for p in theirs], 0
    )
    _refused(producer, request, mem, "needs a TP1 producer", slot)


def test_the_state_slot_path_refuses_a_head_cut(monkeypatch):
    """GDN hybrids' state slots are TP-sharded and never cut by head."""
    mem = _Memory()
    layers, _, _ = stage_layers((2, 2, 2, 2), 1)
    producer = _producer(stage_pages(layers, 4), mem, 4, (2, 2, 2, 2), 1, monkeypatch)
    theirs = stage_pages([*range(LAYERS), MTP], 1)
    request = _request(
        theirs, [mem.add(4 * p.region.unit_bytes, False) for p in theirs], 0
    )
    _refused(producer, request, mem, "state slots", slot=True)


class _Layer:
    """An attention module as the backend walks it, with its real impl class."""

    def __init__(self, layer_num):
        self.layer_num = layer_num
        impl = (
            SparseMHAPagedAttentionImpl if layer_num in SPARSE else PagedAttentionImpl
        )
        self.impl = object.__new__(impl)


def test_the_backend_views_sparse_layers_as_the_sparse_writer_lays_heads_out():
    sparse_attn = pytest.importorskip(
        "atom.model_ops.minimax_m3.sparse_attn", exc_type=ImportError
    )
    modules = [_Layer(n) for n in (5, 6)]  # a stage: one dense, one sparse layer
    pool = MhaKvPool(
        layers=2,
        block_size=BLOCK,
        num_kv_heads=4,
        head_dim=HEAD_DIM,
        kv_dtype=torch.float8_e4m3fn,
        extra_fields=[EntryField("index", 1, (BLOCK * 128,), torch.uint8)],
    )
    pool.allocate(4, "cpu")
    builder = object.__new__(aiter_attention.AiterAttentionMetadataBuilder)
    builder.kv_pools = {KvGeometry(4, HEAD_DIM): pool}
    builder._pool_rows = {
        KvGeometry(4, HEAD_DIM): {m: row for row, m in enumerate(modules)},
        aiter_attention.INDEX_ROWS: {modules[1]: 0},
    }
    builder._index_cache_spec = lambda: aiter_attention._IndexCacheSpec(
        1, 128, torch.float8_e4m3fn
    )
    pages = builder.get_kv_transfer_tensors().pages
    by_unit = {(p.unit_spec.field, p.unit_spec.layer): p for p in pages}
    assert by_unit["k", 5].head_major_runs == (1, 4)
    assert by_unit["index", 6].head_major_runs is None
    views = sparse_attn.to_page16_shuffle(*pool.kv_views(1), *pool.scale_views(1))
    for view in views:
        for head in range(4):
            view.view(torch.uint8)[:, head].fill_(head + 1)
    for field in ("k", "v", "k_scale", "v_scale"):
        page = by_unit[field, 6]
        runs, heads = page.head_major_runs
        assert runs == BLOCK // sparse_attn.ASM_PAGE_SIZE
        unit = page.view[2].view(runs, heads, -1)
        for head in range(heads):
            assert torch.all(unit[:, head] == head + 1)


def _mla_target_with_mha_draft(heads):
    """An MLA target names no units; its paged MHA (EAGLE3) draft does."""
    pool = MhaKvPool(
        layers=1,
        block_size=BLOCK,
        num_kv_heads=heads,
        head_dim=HEAD_DIM,
        kv_dtype=torch.bfloat16,
    )
    pool.allocate(4, "cpu")
    mla = page_region(torch.zeros(4, 576, dtype=torch.uint8), semantic_role="mla.kv")
    return [mla, *pool.transfer_pages("draft", "draft", [LAYERS + 1])]


@pytest.mark.parametrize("heads", [1, 4])
def test_an_mla_target_with_an_mha_draft_moves_by_position(heads, monkeypatch):
    """Same TP copies every region whole, as on main; another TP is refused
    rather than handing consumer rank c producer rank c % P's draft heads."""
    mem = _Memory()
    ours = _mla_target_with_mha_draft(heads)
    assert mc._pack_unit_layout(ours) is None
    producer = _producer(ours, mem, heads, (1,), 0, monkeypatch)
    theirs = _mla_target_with_mha_draft(1)
    bases = [mem.add(4 * p.region.unit_bytes, False) for p in theirs]
    request = _request(theirs, bases, 0)
    if heads == 4:  # a TP1 producer, a TP4 consumer
        _refused(producer, request, mem, "split by KV head")
        return
    assert producer._execute_block_transfer(request, "consumer", [2], [1], "mixed")
    for page, src_base, dst_base in zip(ours, producer.kv_caches_base_addr, bases):
        n = page.region.unit_bytes
        assert np.array_equal(
            mem.at(dst_base + n, n), mem.at(src_base + 2 * n, n)
        ), page.region.semantic_role
