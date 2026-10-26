# SPDX-License-Identifier: MIT
"""MiniMax-M3 P/D from a PP prefill (TP1 stages) into a TP-sharded decode.

The units are named by global layer and say how they split by KV head; the
producer pairs a stage's regions with the decode's by name and writes each
consumer rank its heads. The byte test drives the real descriptor builder
against host buffers, so a wrong offset shows up as a wrong byte.
"""

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from atom.model_ops.attentions.mha_kv_pool import MhaKvPool
from atom.model_ops.attentions.pool_layout.entry_arena import EntryField
from atom.model_ops.attentions.pool_layout.pool_rows import KvGeometry

aiter_attention = pytest.importorskip("atom.model_ops.attentions.aiter_attention")

BLOCK, HEAD_DIM = 128, 128


class _Layer:
    def __init__(self, layer_num, sparse):
        self.layer_num = layer_num
        self.impl = SimpleNamespace(is_indexed_sparse_attention=sparse)


def _m3_builder(layer_nums, sparse, heads):
    modules = [_Layer(n, s) for n, s in zip(layer_nums, sparse)]
    indexed = [m for m in modules if m.impl.is_indexed_sparse_attention]
    pool = MhaKvPool(
        layers=len(modules),
        block_size=BLOCK,
        num_kv_heads=heads,
        head_dim=HEAD_DIM,
        kv_dtype=torch.float8_e4m3fn,
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
    builder.model_runner = SimpleNamespace(block_size=BLOCK)
    builder._has_sparse_attention = True
    return builder


def test_m3_units_are_named_by_global_layer_and_split_by_head():
    tt = _m3_builder([15, 16], [True, False], heads=4).get_kv_transfer_tensors()
    assert tt.roles_are_global
    runs = {p.region.semantic_role: p.head_major_runs for p in tt.pages}
    assert runs["m3.layer_15.k"] == runs["m3.layer_15.v_scale"] == (8, 4)
    assert runs["m3.layer_16.v"] == (1, 4)  # dense: one page-128 run
    assert runs["m3.layer_15.index"] is None and "m3.layer_16.index" not in runs


def test_page16_runs_are_how_the_sparse_writer_lays_heads_out():
    sparse_attn = pytest.importorskip("atom.model_ops.minimax_m3.sparse_attn")
    builder = _m3_builder([15], [True], heads=4)
    pool = next(iter(builder.kv_pools.values()))
    views = sparse_attn.to_page16_shuffle(*pool.kv_views(0), *pool.scale_views(0))
    for view in (views[0].view(torch.uint8), views[1].view(torch.uint8)):
        for head in range(4):
            view[:, head].fill_(head + 1)
    for page in builder.get_kv_transfer_tensors().pages:
        if page.region.semantic_role in ("m3.layer_15.k", "m3.layer_15.v"):
            unit = page.view[2].view(8, 4, -1)
            for head in range(4):
                assert torch.all(unit[:, head] == head + 1)


class _Memory:
    """Host buffers at disjoint fake addresses, written by a fake RDMA."""

    def __init__(self):
        self.buffers, self.next = [], 1 << 40

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
        for s, d, n in zip(srcs, dsts, sizes):
            self.at(d, n)[:] = self.at(s, n)


def _layer(n, heads):
    kv, scale = heads * 8 * 2048, heads * 8 * 64
    return [
        (f"m3.layer_{n}.k", (8, heads), kv),
        (f"m3.layer_{n}.v_scale", (8, heads), scale),
        (f"m3.layer_{n}.index", None, BLOCK * 128),
    ]


def test_tp1_stage_fills_each_tp4_rank_with_its_head():
    mc = pytest.importorskip(
        "atom.kv_transfer.disaggregation.mooncake.mooncake_connector"
    )
    mem, blocks, src_ids, dst_ids = _Memory(), 6, [4, 1, 3], [0, 5, 2]
    layout = _layer(15, 4) + _layer(16, 4)
    producer = object.__new__(mc.MooncakeConnector)
    producer.__dict__.update(
        kv_caches_base_addr=[mem.add(blocks * b, True) for _, _, b in layout],
        _per_block_bytes_list=[b for _, _, b in layout],
        _block_region_roles=[r for r, _, _ in layout],
        _block_head_major_runs=[runs for _, runs, _ in layout],
        _roles_are_global=True,
        _block_region_consumer_indices=None,
        tp_size=1,
        dcp_size=1,
        pp_size=4,
        block_size=BLOCK,
        _mla_staging=None,
        _rdma_write=lambda target, s, d, n, *a, **k: mem.write(s, d, n) or True,
    )
    consumers = []
    for rank in range(4):
        # The decode holds every layer (and the draft), in its own order.
        theirs = (_layer(14, 1) + _layer(15, 1) + _layer(16, 1))[::-1]
        bases = [mem.add(blocks * b, False) for _, _, b in theirs]
        consumers.append({r: (a, b) for (r, _, b), a in zip(theirs, bases)})
        assert producer._execute_block_transfer(
            {
                "consumer_base_addrs": bases,
                "consumer_region_roles": [r for r, _, _ in theirs],
                "consumer_block_bpb": [b for _, _, b in theirs],
                "consumer_head_major_runs": [runs for _, runs, _ in theirs],
                "consumer_tp_size": 4,
                "consumer_tp_rank": rank,
            },
            "consumer",
            src_ids,
            dst_ids,
            f"req-{rank}",
        )
    for (role, runs, bpb), base in zip(layout, producer.kv_caches_base_addr):
        src = mem.at(base, blocks * bpb).reshape(blocks, -1)
        for rank, regions in enumerate(consumers):
            dst_base, dst_bpb = regions[role]
            dst = mem.at(dst_base, blocks * dst_bpb).reshape(blocks, -1)
            for s, d in zip(src_ids, dst_ids):
                want = src[s] if runs is None else src[s].reshape(8, 4, -1)[:, rank]
                assert np.array_equal(dst[d], want.ravel()), (role, rank)
            assert not dst[[1, 3, 4]].any()
    for regions in consumers:  # layer 14 is another stage's
        assert not mem.at(*regions["m3.layer_14.k"]).any()


def test_role_pairing_refuses_a_consumer_without_the_region():
    mc = pytest.importorskip(
        "atom.kv_transfer.disaggregation.mooncake.mooncake_connector"
    )
    conn = object.__new__(mc.MooncakeConnector)
    conn._block_region_roles = ["draft.k.layer_0", "m3.layer_59.k"]
    assert conn._consumer_indices_by_role(["m3.layer_59.k", "draft.k.layer_0"]) == [
        1,
        0,
    ]
    for roles in (["m3.layer_59.k"], ["m3.layer_59.k", "m3.layer_59.k"]):
        with pytest.raises(RuntimeError, match="Cannot pair"):
            conn._consumer_indices_by_role(roles)
