# SPDX-License-Identifier: MIT
"""MiniMax-M3 P/D from a PP prefill (TP1 stages) into a TP-sharded decode.

The model annotates the backend's regions before the runner appends the draft.
The byte test drives the real descriptor builder against host buffers, so a
wrong region or head offset shows up as a wrong byte.
"""

import __future__

import ast
from pathlib import Path
from types import MethodType, SimpleNamespace

import numpy as np
import pytest
import torch

from atom.kv_transfer.offload.mp.page_views import _build_cache_views
from atom.model_ops.attentions.mha_kv_pool import MhaKvPool
from atom.model_ops.attentions.mla_kv_pool import MlaKvPool
from atom.model_ops.attentions.pool_layout.entry_arena import EntryField
from atom.model_ops.attentions.pool_layout.pool_rows import KvGeometry
from atom.spec_decode.draft_kv import DRAFT_KV_ROWS, DraftKvBuilder

# `exc_type`: collected after a test that stubs `sys.modules["aiter"]`, the
# import reaches the stub and fails on a symbol rather than on the module.
aiter_attention = pytest.importorskip(
    "atom.model_ops.attentions.aiter_attention", exc_type=ImportError
)

BLOCK, HEAD_DIM = 128, 128
CONFIG = SimpleNamespace(
    num_hidden_layers=8,
    sparse_attention_config={"sparse_attention_freq": [1, 0, 1, 0, 0, 0, 1, 0]},
)


def _cache_methods(path, cls_name, names):
    # Run the shipped cache-registration methods without loading unrelated
    # model kernels or initializing the GPU runner. Pools/exporters below are
    # real; only the owner objects stand in for weight-loaded models.
    source = Path(__file__).resolve().parents[1] / path
    tree = ast.parse(source.read_text())
    cls = next(
        n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == cls_name
    )
    methods = [
        n
        for n in tree.body + cls.body
        if isinstance(n, ast.FunctionDef) and n.name in names
    ]
    namespace = {}
    exec(  # noqa: S102 -- execute the shipped methods under the CPU fixture
        compile(
            ast.Module(body=methods, type_ignores=[]),
            str(source),
            "exec",
            flags=__future__.annotations.compiler_flag,
        ),
        namespace,
    )
    return namespace


_MODEL_SOURCE = "atom/models/minimax_m3/model.py"
_configure_model = _cache_methods(
    _MODEL_SOURCE,
    "MiniMaxM3SparseForCausalLM",
    (
        "_sparse_attention_layer_ids",
        "_sparse_attention_layer_ordinals",
        "configure_kv_transfer_tensors",
    ),
)["configure_kv_transfer_tensors"]
_configure_wrapper = _cache_methods(
    _MODEL_SOURCE,
    "MiniMaxM3SparseForConditionalGenerationTextOnly",
    ("configure_kv_transfer_tensors",),
)["configure_kv_transfer_tensors"]
_runner_transfer = _cache_methods(
    "atom/model_engine/model_runner.py", "ModelRunner", ("_get_kv_transfer_tensors",)
)["_get_kv_transfer_tensors"]
_mla_export = _cache_methods(
    "atom/model_ops/attentions/aiter_mla.py",
    "AiterMLAMetadataBuilder",
    ("get_kv_transfer_tensors",),
)["get_kv_transfer_tensors"]


class _Layer:
    def __init__(self, layer_num, sparse):
        self.layer_num = layer_num
        self.impl = SimpleNamespace(is_indexed_sparse_attention=sparse)


def _m3_builder(layer_nums, sparse, heads, config=None, kv_dtype=torch.float8_e4m3fn):
    modules = [_Layer(n, s) for n, s in zip(layer_nums, sparse)]
    indexed = [m for m in modules if m.impl.is_indexed_sparse_attention]
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
    if config is None:
        config = SimpleNamespace(
            num_hidden_layers=60,
            sparse_attention_config={"sparse_attention_freq": [0, 1] * 30},
        )
    model = SimpleNamespace(config=config)
    model.configure_kv_transfer_tensors = MethodType(_configure_model, model)
    builder.model_runner = SimpleNamespace(
        block_size=BLOCK,
        model=model,
        attn_metadata_builder=builder,
        config=SimpleNamespace(index_cache_dtype="fp8"),
    )
    return builder


def _export(builder, draft=None):
    if draft is not None:
        builder.model_runner.draft_kv_builder = SimpleNamespace(
            get_kv_transfer_tensors=lambda: draft
        )
    return _runner_transfer(builder.model_runner, 4)


def _draft_tensors(heads, kv_dtype=torch.float8_e4m3fn):
    builder = object.__new__(DraftKvBuilder)
    builder._kv_pool = MhaKvPool(
        layers=2,
        block_size=BLOCK,
        num_kv_heads=heads,
        head_dim=HEAD_DIM,
        kv_dtype=kv_dtype,
    )
    builder._kv_pool.allocate(4, "cpu")
    builder._pool_rows = {
        DRAFT_KV_ROWS: {
            type("Binding", (), {"shuffle_page_tokens": 16})(): row for row in range(2)
        }
    }
    return builder.get_kv_transfer_tensors()


def _transfer(layer_nums, heads, draft=False, kv_dtype=torch.float8_e4m3fn):
    sparse = [
        CONFIG.sparse_attention_config["sparse_attention_freq"][i] for i in layer_nums
    ]
    return _export(
        _m3_builder(layer_nums, sparse, heads, CONFIG, kv_dtype),
        _draft_tensors(heads, kv_dtype) if draft else None,
    )


def test_m3_units_are_named_by_global_layer_and_split_by_head():
    builder = _m3_builder([15, 16], [True, False], heads=4)
    raw = builder.get_kv_transfer_tensors()
    tt = _export(builder)
    for original, annotated in zip(raw.pages, tt.pages, strict=True):
        assert original.region.base_addr == annotated.region.base_addr
        assert original.region.total_bytes == annotated.region.total_bytes
        assert original.region.unit_bytes == annotated.region.unit_bytes
        assert original.view.data_ptr() == annotated.view.data_ptr()
    assert tt.state_backend is builder and tt.num_blocks == 4
    assert tt.block_region_consumer_indices == [15, 16, 75, 76, 135, 136, 195, 196, 247]
    assert tt.block_region_consumer_count == 270
    runs = {p.region.semantic_role: p.head_major_runs for p in tt.pages}
    assert runs["m3.layer_15.k"] == runs["m3.layer_15.v_scale"] == (8, 4)
    assert runs["m3.layer_16.v"] == (1, 4)  # dense: one page-128 run
    assert runs["m3.layer_15.index"] is None and "m3.layer_16.index" not in runs
    # LMCache MP takes the same units, still one stored shard per rank.
    assert tt.tp_replication_factor == 1
    tt.set_block_count(4)
    assert len(_build_cache_views(tt, num_blocks=4).tensors) == len(tt.pages)


@pytest.mark.parametrize("schedule", [{}, {"sparse_attention_freq": None}])
def test_no_sparse_schedule_has_no_global_index_regions(schedule):
    config = SimpleNamespace(
        num_hidden_layers=3,
        sparse_attention_config={"sparse_index_dim": 128, **schedule},
    )
    tt = _export(_m3_builder([1], [False], 4, config))
    assert tt.block_region_consumer_indices == [1, 4, 7, 10]
    assert tt.block_region_consumer_count == 12


def test_text_wrapper_forwards_layout_before_draft_is_appended():
    builder = _m3_builder([6, 7], [True, False], 4, CONFIG)
    wrapper = SimpleNamespace(language_model=builder.model_runner.model)
    wrapper.configure_kv_transfer_tensors = MethodType(_configure_wrapper, wrapper)
    builder.model_runner.model = wrapper
    tt = _export(builder, _draft_tensors(4))
    expected = _transfer([6, 7], 4, draft=True)
    assert tt.block_region_consumer_indices == expected.block_region_consumer_indices
    assert _layout(tt) == _layout(expected)


def test_runner_without_model_hook_keeps_generic_mha_layout():
    builder = _m3_builder([0, 1], [True, False], 4, CONFIG)
    builder.model_runner.model = SimpleNamespace()
    raw = builder.get_kv_transfer_tensors()
    tt = _export(builder, _draft_tensors(4))
    assert _layout(tt)[: len(raw.pages)] == _layout(raw)
    assert all(r.semantic_role.startswith("mha.") for r in raw.block_regions)
    assert tt.block_region_consumer_indices is None
    assert tt.block_region_consumer_count is None
    assert tt.state_backend is builder and tt.num_blocks == 4


def test_runner_without_transfer_regions():
    builder = _m3_builder([0], [True], 4, CONFIG)
    builder.kv_pools = {}
    assert _export(builder) is None


@pytest.mark.parametrize("hybrid", [False, True])
@pytest.mark.parametrize("fp4", [False, True])
@pytest.mark.parametrize("stage", [0, 1])
def test_mla_export_preserves_compact_layers_mtp_and_scale_offsets(
    monkeypatch, hybrid, fp4, stage
):
    from aiter.dist import parallel_state

    monkeypatch.setattr(
        parallel_state,
        "get_pp_group",
        lambda: SimpleNamespace(rank_in_group=stage, world_size=2),
    )
    target = [1, 3, 5] if hybrid else list(range(6))
    global_index = [1, 5, 6] if hybrid else [0, 3, 5, 6]
    local_kv = [i for i in target if stage * 3 <= i < (stage + 1) * 3]
    if stage == 1:
        local_kv.append(6)  # MTP is held only by the last stage.
    local_index = [i for i in global_index if i in local_kv]
    pool = MlaKvPool(
        layers=len(local_kv),
        block_size=64,
        entry_dim=16,
        kv_dtype=torch.float16,
        index_layers=len(local_index),
        index_rows_per_block=64,
        index_dim=144,
        index_dtype=torch.uint8,
        index_head_dim=128,
        index_fp4=fp4,
    )
    pool.allocate(4, "cpu")
    runner = SimpleNamespace(
        config=SimpleNamespace(
            hf_config=SimpleNamespace(num_hidden_layers=6),
            tensor_parallel_size=4,
            kv_transfer_config={"kv_connector": "lmcache_mp"},
        ),
        index_cache_layer_ids=tuple(local_index),
        full_attention_layers=target if hybrid else None,
    )
    builder = SimpleNamespace(
        model_runner=runner,
        kv_pool=pool,
        _indexer_fp4=fp4,
        _local_draft_rows=lambda: int(stage == 1),
        _index_cache_layout=lambda: (tuple(local_index), tuple(global_index)),
    )
    transfer = _mla_export(builder)
    kv_indices = (
        ([0] if stage == 0 else [1, 2, 3])
        if hybrid
        else ([0, 1, 2] if stage == 0 else [3, 4, 5, 6])
    )
    index_indices = (
        ([4] if stage == 0 else [5, 6])
        if hybrid
        else ([7] if stage == 0 else [8, 9, 10])
    )
    expected = kv_indices + index_indices
    if fp4:
        expected += [i + len(global_index) for i in index_indices]
    count = len(target) + 1 + len(global_index) * (2 if fp4 else 1)
    assert transfer.block_region_consumer_indices == expected
    assert transfer.block_region_consumer_count == count
    assert len(transfer.pages) == len(expected)
    # A separate contributor starts after the full target/MTP layout, even
    # though this stage holds only a subset of those regions.
    transfer.merge_pages(_draft_tensors(1))
    assert transfer.block_region_consumer_indices[len(expected) :] == list(
        range(count, count + 8)
    )


@pytest.mark.parametrize("heads", [1, 4])
@pytest.mark.parametrize("kv_dtype", [torch.float8_e4m3fn, torch.bfloat16])
@pytest.mark.parametrize("draft", [False, True])
@pytest.mark.parametrize(
    "stages", [([0, 1], [2, 3], [4, 5], [6, 7]), ([0, 1, 2], [3, 4], [5, 6], [7])]
)
def test_every_stage_maps_into_the_full_consumer_layout(heads, kv_dtype, draft, stages):
    consumer = _transfer(list(range(8)), 1, draft, kv_dtype)
    roles = [r.semantic_role for r in consumer.block_regions]
    all_indices = []
    for held in stages:
        tt = _transfer(held, heads, draft and held[-1] == 7, kv_dtype)
        assert len(tt.block_region_consumer_indices) == len(tt.pages)
        for page, idx in zip(tt.pages, tt.block_region_consumer_indices):
            assert page.region.semantic_role == roles[idx]
        all_indices.extend(tt.block_region_consumer_indices)
    assert sorted(all_indices) == list(range(len(consumer.pages)))
    assert consumer.block_region_consumer_count == len(consumer.pages)
    assert consumer.block_region_consumer_indices == list(range(len(consumer.pages)))


def test_page16_runs_are_how_the_sparse_writer_lays_heads_out():
    sparse_attn = pytest.importorskip(
        "atom.model_ops.minimax_m3.sparse_attn", exc_type=ImportError
    )
    builder = _m3_builder([15], [True], heads=4)
    pool = next(iter(builder.kv_pools.values()))
    views = sparse_attn.to_page16_shuffle(*pool.kv_views(0), *pool.scale_views(0))
    for view in (views[0].view(torch.uint8), views[1].view(torch.uint8)):
        for head in range(4):
            view[:, head].fill_(head + 1)
    for page in _export(builder).pages:
        if page.region.semantic_role in ("m3.layer_15.k", "m3.layer_15.v"):
            unit = page.view[2].view(8, 4, -1)
            for head in range(4):
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


def _layout(tt):
    return [
        (p.region.semantic_role, p.head_major_runs, p.region.unit_bytes)
        for p in tt.pages
    ]


def _producer(tt, mem, blocks, heads):
    mc = pytest.importorskip(
        "atom.kv_transfer.disaggregation.mooncake.mooncake_connector"
    )
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
        _kv_staging=None,
        _MAX_RDMA_ENTRIES_PER_BATCH=7,
        _rdma_write=lambda target, s, d, n, *a, **k: mem.write(s, d, n) or True,
    )
    return producer


@pytest.mark.parametrize("heads", [1, 4])
@pytest.mark.parametrize("held", [[0, 1], [2, 3], [4, 5], [6, 7], [7]])
def test_stage_fills_each_tp4_rank_with_its_head(heads, held):
    mem, blocks, src_ids, dst_ids = _Memory(), 4, [3, 1], [0, 2]
    tt = _transfer(held, heads, draft=held[-1] == 7)
    layout = _layout(tt)
    producer = _producer(tt, mem, blocks, heads)
    consumers = []
    for rank in range(4):
        # Both layouts come from the real exporters. The consumer has all layers.
        theirs = _layout(_transfer(list(range(8)), 1, draft=True))
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
                want = (
                    src[s]
                    if runs is None or heads == 1
                    else src[s].reshape(runs[0], heads, -1)[:, rank]
                )
                assert np.array_equal(dst[d], want.ravel()), (role, rank)
            assert not dst[[1, 3]].any()
    held_roles = {role for role, _, _ in layout}
    for regions in consumers:
        for role, (base, bpb) in regions.items():
            if role not in held_roles:
                assert not mem.at(base, blocks * bpb).any()
    assert mem.writes > 4


@pytest.mark.parametrize("bad_layout", ["reordered", "missing_draft"])
def test_incompatible_consumer_layout_is_refused(bad_layout):
    mem = _Memory()
    producer = _producer(_transfer([7], 4, draft=True), mem, 4, 4)
    consumer = _transfer(list(range(8)), 1, draft=True)
    roles = [r.semantic_role for r in consumer.block_regions]
    bases = [0] * len(roles)
    if bad_layout == "reordered":
        roles.reverse()
    else:
        bases = bases[:-1]
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
