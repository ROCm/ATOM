# SPDX-License-Identifier: MIT
"""MHA KV from a PP prefill (TP1 stages) into a TP-sharded decode.

The MHA backend's own export numbers a stage's units as an unsplit model does
and says how each splits by KV head (MiniMax-M3: page-16 sparse layers, dense
layers, the replicated indexer keys); the last stage appends its draft pool's
(DSpark: page-16), which the consumer appends last too. The byte test drives
the real descriptor builder against host buffers, so a wrong region or head
offset shows up as a wrong byte.
"""

from dataclasses import asdict
from types import SimpleNamespace

import msgpack
import numpy as np
import pytest
import torch

from atom.kv_transfer.offload.mp.page_views import _build_cache_views
from atom.model_ops.attentions.mha_kv_pool import MhaKvPool
from atom.model_ops.attentions.pool_layout.entry_arena import EntryField
from atom.model_ops.attentions.pool_layout.pool_rows import KvGeometry
from atom.spec_decode.draft_kv import DRAFT_KV_ROWS, DraftKvBuilder

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
INDEX = (128, torch.float8_e4m3fn)  # the indexer keys' width and dtype
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


def _builder(
    layers, heads, kv_dtype=torch.float8_e4m3fn, head_dim=HEAD_DIM, index=INDEX
):
    modules = [_Layer(n) for n in layers]
    indexed = [m for m in modules if m.layer_num in SPARSE]
    index_dim, index_dtype = index
    index_bytes = BLOCK * index_dim * index_dtype.itemsize
    pool = MhaKvPool(
        layers=len(modules),
        block_size=BLOCK,
        num_kv_heads=heads,
        head_dim=head_dim,
        kv_dtype=kv_dtype,
        extra_fields=[EntryField("index", len(indexed), (index_bytes,), torch.uint8)],
    )
    pool.allocate(4, "cpu")
    geometry = KvGeometry(heads, head_dim)
    builder = object.__new__(aiter_attention.AiterAttentionMetadataBuilder)
    builder.kv_pools = {geometry: pool}
    builder._pool_rows = {
        geometry: {m: row for row, m in enumerate(modules)},
        aiter_attention.INDEX_ROWS: {m: row for row, m in enumerate(indexed)},
    }
    builder.block_size, builder.block_ratio = BLOCK, 1
    builder._index_cache_spec = lambda: aiter_attention._IndexCacheSpec(
        len(indexed), index_dim, index_dtype
    )
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


class _Binding:
    """A DSpark draft layer's cache binding, which reads and writes page-16."""

    shuffle_page_tokens = 16


def _draft(heads, kv_dtype):
    builder = object.__new__(DraftKvBuilder)
    builder._kv_pool = MhaKvPool(
        layers=2,
        block_size=BLOCK,
        num_kv_heads=heads,
        head_dim=HEAD_DIM,
        kv_dtype=kv_dtype,
    )
    builder._kv_pool.allocate(4, "cpu")
    builder._pool_rows = {DRAFT_KV_ROWS: {_Binding(): row for row in range(2)}}
    return builder.get_kv_transfer_tensors()


@pytest.fixture
def export(monkeypatch):
    """The runner's export on stage `rank` of `partition`, or unsplit: the
    backend's units, then the draft's on the last stage."""

    def _export(
        heads,
        rank=0,
        partition=None,
        kv_dtype=torch.float8_e4m3fn,
        head_dim=HEAD_DIM,
        index=INDEX,
        draft=False,
    ):
        sizes = partition or (LAYERS,)
        monkeypatch.setenv("VLLM_PP_LAYER_PARTITION", ",".join(map(str, sizes)))
        start = sum(sizes[:rank])
        layers = range(start, start + sizes[rank])
        builder = _builder(layers, heads, kv_dtype, head_dim, index)
        tt = _pp(monkeypatch, builder, rank, len(sizes)).get_kv_transfer_tensors()
        if draft and rank == len(sizes) - 1:
            tt.merge_pages(_draft(heads, kv_dtype))
        return tt

    return _export


def _layout(tt):
    return [
        (p.region.semantic_role, p.head_major_runs, p.region.unit_bytes, p.unit_spec)
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
@pytest.mark.parametrize("draft", [False, True])
@pytest.mark.parametrize("partition", PARTITIONS)
def test_every_stage_maps_into_the_full_consumer_layout(
    export, heads, kv_dtype, draft, partition
):
    consumer = [p.unit_spec for p in export(1, kv_dtype=kv_dtype, draft=draft).pages]
    claimed = []
    for rank in range(len(partition)):
        tt = export(heads, rank, partition, kv_dtype, draft=draft)
        assert len(tt.block_region_consumer_indices) == len(tt.pages)
        # Negative: counted from the consumer's end (the appended draft).
        indices = [i % len(consumer) for i in tt.block_region_consumer_indices]
        for page, index in zip(tt.pages, indices):
            assert page.unit_spec == consumer[index]
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
        kv_caches_base_addr=[mem.add(blocks * b, True) for _, _, b, _ in layout],
        _per_block_bytes_list=[b for _, _, b, _ in layout],
        _block_region_roles=[r for r, _, _, _ in layout],
        _block_head_major_runs=[runs for _, runs, _, _ in layout],
        _block_unit_specs=[spec for _, _, _, spec in layout],
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
        "consumer_region_roles": [r for r, _, _, _ in theirs],
        "consumer_block_bpb": [b for _, _, b, _ in theirs],
        "consumer_unit_specs": [asdict(spec) for _, _, _, spec in theirs],
        "consumer_tp_size": 4,
        "consumer_tp_rank": rank,
    }
    if runs:
        request["consumer_head_major_runs"] = [r for _, r, _, _ in theirs]
    # The producer receives dictionaries and lists, not our local dataclasses.
    return msgpack.loads(msgpack.dumps(request))


@pytest.mark.parametrize("heads", [1, 4])
@pytest.mark.parametrize("draft", [False, True])
@pytest.mark.parametrize("rank, partition", [(0, (2, 2, 2, 2)), (3, (3, 2, 2, 1))])
@pytest.mark.parametrize("rename", [False, True])
def test_stage_fills_each_tp4_rank_with_its_head(
    export, heads, draft, rank, partition, rename, monkeypatch
):
    """`heads` 4 is a TP1 stage, cut by head; 1 is a TP4 stage, copied whole."""
    mem, blocks, src_ids, dst_ids = _Memory(), 4, [3, 1], [0, 2]
    theirs = _layout(export(1, draft=draft))  # a TP4 decode rank: one head
    if rename:
        # A backend's display format can change without changing its bytes.
        monkeypatch.setattr(KvGeometry, "__str__", lambda self: "display-geometry")
    tt = export(heads, rank, partition, draft=draft)
    layout = _layout(tt)
    producer = _producer(tt, mem, blocks, heads)
    consumers = []
    for consumer_rank in range(4):
        bases = [mem.add(blocks * b, False) for _, _, b, _ in theirs]
        consumers.append({spec: (a, b) for (_, _, b, spec), a in zip(theirs, bases)})
        assert producer._execute_block_transfer(
            _request(theirs, bases, consumer_rank),
            "consumer",
            src_ids,
            dst_ids,
            f"req-{consumer_rank}",
        )
    for (role, runs, bpb, spec), base in zip(layout, producer.kv_caches_base_addr):
        src = mem.at(base, blocks * bpb).reshape(blocks, -1)
        for consumer_rank, regions in enumerate(consumers):
            dst_base, dst_bpb = regions[spec]
            dst = mem.at(dst_base, blocks * dst_bpb).reshape(blocks, -1)
            for s, d in zip(src_ids, dst_ids):
                want = (
                    src[s]
                    if runs is None or heads == 1
                    else src[s].reshape(runs[0], heads, -1)[:, consumer_rank]
                )
                assert np.array_equal(dst[d], want.ravel()), (role, consumer_rank)
            assert not dst[[1, 3]].any()
    held = {spec for _, _, _, spec in layout}
    for regions in consumers:
        for role, (base, bpb) in regions.items():
            if role not in held:
                assert not mem.at(base, blocks * bpb).any()
    assert mem.writes > 4


@pytest.mark.parametrize("bad_layout", ["reordered", "short", "missing_draft"])
def test_incompatible_consumer_layout_is_refused(export, bad_layout):
    mem = _Memory()
    producer = _producer(export(4, 3, (3, 2, 2, 1), draft=True), mem, 4, 4)
    consumer = export(1, draft=bad_layout != "missing_draft")
    roles = [r.semantic_role for r in consumer.block_regions]
    bases = [0] * len(roles)
    if bad_layout == "reordered":
        roles.reverse()
    elif bad_layout == "short":
        bases, roles = bases[:-1], roles[:-1]
    # A missing draft: its regions, numbered from the consumer's end, land
    # on target regions this stage already maps.
    with pytest.raises(
        (RuntimeError, ValueError), match="role mismatch|exceeds|two regions"
    ):
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
    [(4, "describes no head runs"), (2, "needs a TP1 producer")],
)
@pytest.mark.parametrize("layout_checks", [True, False])
def test_an_older_consumer_is_refused_before_anything_is_written(
    export, heads, refusal, layout_checks
):
    """A consumer at another TP that does not describe its head runs is
    refused, with or without its roles and byte widths. A producer that may
    not cut is refused whatever the consumer sends: rank `c` would pull
    another rank's heads from producer rank `c % P`."""
    mem = _Memory()
    producer = _producer(export(heads, 3, (3, 2, 2, 1)), mem, 4, heads)
    theirs = _layout(export(1))
    request = _request(
        theirs, [mem.add(4 * b, False) for _, _, b, _ in theirs], 0, False
    )
    del request["consumer_unit_specs"]
    if not layout_checks:
        del request["consumer_region_roles"], request["consumer_block_bpb"]
    with pytest.raises(RuntimeError, match=refusal):
        producer._execute_block_transfer(request, "consumer", [0], [0], "old")
    assert mem.writes == 0


@pytest.mark.parametrize("heads", [1, 4])
@pytest.mark.parametrize(
    "field, value",
    [
        ("namespace", "draft"),
        ("field", "v"),
        ("row", 100),
        ("head_dim", 64),
        ("dtype", "torch.float16"),
        ("layout", "token_head"),
    ],
)
def test_incompatible_last_unit_is_refused_before_any_write(
    export, heads, field, value
):
    mem = _Memory()
    tt = export(heads, 3, (3, 2, 2, 1))
    producer = _producer(tt, mem, 4, heads)
    theirs = _layout(export(1))
    request = _request(theirs, [mem.add(4 * b, False) for _, _, b, _ in theirs], 0)
    # Keep the names and byte lengths untouched. Even the last region must be
    # checked before the earlier regions can flush a batch of RDMA writes.
    index = tt.block_region_consumer_indices[-1]
    request["consumer_unit_specs"][index][field] = value
    with pytest.raises(RuntimeError, match="unit spec mismatch"):
        producer._execute_block_transfer(request, "consumer", [0], [0], "bad-spec")
    assert mem.writes == 0


@pytest.mark.parametrize(
    "producer_dim, consumer_dtype",
    [(64, torch.float8_e4m3fn), (128, torch.float16)],
)
def test_equal_byte_width_does_not_hide_an_incompatible_format(
    export, producer_dim, consumer_dtype
):
    mem = _Memory()
    tt = export(4, 0, (2, 2, 2, 2), torch.bfloat16, producer_dim)
    producer = _producer(tt, mem, 4, 4)
    theirs = _layout(export(1, kv_dtype=consumer_dtype))
    assert tt.pages[0].region.unit_bytes // 4 == theirs[0][2]
    request = _request(theirs, [mem.add(4 * b, False) for _, _, b, _ in theirs], 0)
    with pytest.raises(RuntimeError, match="unit spec mismatch"):
        producer._execute_block_transfer(request, "consumer", [0], [0], "equal-bytes")
    assert mem.writes == 0


@pytest.mark.parametrize(
    "producer_index",
    [(64, torch.bfloat16), (128, torch.float8_e4m3fnuz)],
)
def test_equal_byte_width_does_not_hide_an_incompatible_index_format(
    export, producer_index
):
    """The pool stores the indexer keys as uint8; their spec names the keys'
    own width and dtype, so equal bytes per token still cannot pass."""
    mem = _Memory()
    tt = export(4, 0, (2, 2, 2, 2), index=producer_index)
    producer = _producer(tt, mem, 4, 4)
    theirs = _layout(export(1))
    ours = [p for p in tt.pages if p.unit_spec.field == "index"]
    assert ours and all(
        p.region.unit_bytes == b
        for p in ours
        for *_, b, s in theirs
        if s.field == "index"
    )
    request = _request(theirs, [mem.add(4 * b, False) for _, _, b, _ in theirs], 0)
    with pytest.raises(RuntimeError, match="unit spec mismatch"):
        producer._execute_block_transfer(request, "consumer", [0], [0], "equal-index")
    assert mem.writes == 0


@pytest.mark.parametrize(
    "missing", ["consumer_unit_specs", "consumer_block_bpb", "one_unit_spec"]
)
def test_head_cut_requires_structured_metadata(export, missing):
    mem = _Memory()
    tt = export(4, 0, (2, 2, 2, 2))
    producer = _producer(tt, mem, 4, 4)
    theirs = _layout(export(1))
    request = _request(theirs, [mem.add(4 * b, False) for _, _, b, _ in theirs], 0)
    if missing == "one_unit_spec":
        request["consumer_unit_specs"][tt.block_region_consumer_indices[-1]] = None
    else:
        del request[missing]
    with pytest.raises(RuntimeError, match="requires.*unit specs"):
        producer._execute_block_transfer(request, "consumer", [0], [0], "missing-spec")
    assert mem.writes == 0


@pytest.mark.parametrize("heads", [1, 4])
def test_head_layout_mismatch_is_refused_before_any_write(export, heads):
    mem = _Memory()
    tt = export(heads, 0, (2, 2, 2, 2))
    producer = _producer(tt, mem, 4, heads)
    theirs = _layout(export(1))
    request = _request(theirs, [mem.add(4 * b, False) for _, _, b, _ in theirs], 0)
    index = next(
        index
        for page, index in reversed(
            list(zip(tt.pages, tt.block_region_consumer_indices))
        )
        if page.head_major_runs is not None
    )
    request["consumer_head_major_runs"][index] = [2, 1]
    with pytest.raises((RuntimeError, ValueError)):
        producer._execute_block_transfer(request, "consumer", [0], [0], "bad-runs")
    assert mem.writes == 0


def test_same_tp_keeps_legacy_name_and_width_checks(export):
    mem = _Memory()
    producer = _producer(export(1, 0, (2, 2, 2, 2)), mem, 4, 1)
    theirs = _layout(export(1))
    request = _request(theirs, [mem.add(4 * b, False) for _, _, b, _ in theirs], 0)
    del request["consumer_unit_specs"], request["consumer_head_major_runs"]
    assert producer._execute_block_transfer(request, "consumer", [0], [0], "legacy")
    mem.writes = 0
    request["consumer_region_roles"][0] = "wrong-region"
    with pytest.raises(RuntimeError, match="role mismatch"):
        producer._execute_block_transfer(request, "consumer", [0], [0], "legacy-bad")
    assert mem.writes == 0


@pytest.mark.parametrize("heads, refusal", [(4, "state slots"), (2, "TP1 producer")])
def test_a_producer_with_state_slots_refuses_another_consumer_tp(
    export, heads, refusal
):
    """A GDN hybrid's per-request state is TP-sharded and never cut, so its
    block+slot path refuses a consumer at another TP size before any write."""
    mem = _Memory()
    producer = _producer(export(heads, 3, (3, 2, 2, 1)), mem, 4, heads)
    theirs = _layout(export(1))
    request = _request(theirs, [mem.add(4 * b, False) for _, _, b, _ in theirs], 0)
    with pytest.raises((NotImplementedError, RuntimeError), match=refusal):
        producer._execute_block_slot_transfer(
            request, "consumer", [0], [0], {"slot_index": 0}, "slot-consumer"
        )
    assert mem.writes == 0
