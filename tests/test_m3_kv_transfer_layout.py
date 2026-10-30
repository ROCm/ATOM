# SPDX-License-Identifier: MIT
"""PP/M3 mixed-pool exports and real Mooncake copies, using CPU cache views."""

import __future__

import ast
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest
import test_pd_chunked_transfer as protocol
import torch

from atom.kv_transfer.disaggregation.chunked_prefill import ChunkedPrefill
from atom.kv_transfer.offload.mp.page_views import _build_cache_views
from atom.model_ops.attentions.mha_kv_pool import MhaKvPool
from atom.model_ops.attentions.pool_layout.entry_arena import EntryField
from atom.model_ops.attentions.pool_layout.pool_rows import KvGeometry
from atom.model_ops.minimax_m3.cache_layout import to_page16_shuffle
from atom.model_ops.minimax_m3.kv_transfer import configure_m3_kv_transfer_tensors
from atom.spec_decode.draft_kv import DRAFT_KV_ROWS, DraftKvBuilder

BLOCK, BLOCKS, DIM = 128, 4, 128
# Both dense-only and sparse-only stages; the first pool changes with the stage.
SPARSE = {1, 2, 6, 7}
STAGES = ([0], [1, 2], [3, 4, 5], [6, 7])


def _methods(path, cls_name, names):
    """Execute shipped registration methods without weight/kernel imports."""
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
    exec(  # noqa: S102 -- the same CPU fixture technique as test_m3_pp_pd_transfer
        compile(
            ast.Module(body=methods, type_ignores=[]),
            str(source),
            "exec",
            flags=__future__.annotations.compiler_flag,
        ),
        namespace,
    )
    return namespace


_annotate_model = _methods(
    "atom/models/minimax_m3/model.py",
    "MiniMaxM3SparseForCausalLM",
    {
        "_sparse_attention_layer_ids",
        "_sparse_attention_layer_ordinals",
        "_should_skip_minimax_m3_index_topk",
        "configure_kv_transfer_tensors",
    },
)["configure_kv_transfer_tensors"]
_export_backend = _methods(
    "atom/model_ops/attentions/aiter_attention.py",
    "AiterAttentionMetadataBuilder",
    {"get_kv_transfer_tensors"},
)["get_kv_transfer_tensors"]


def _target(held, heads, *, nvfp4=True, index_fp4=True, dtype=torch.float8_e4m3fn):
    modules = {layer: type("Layer", (), {"layer_num": layer})() for layer in held}
    groups = {}
    for layer in held:
        geometry = KvGeometry(heads, DIM, "nvfp4" if nvfp4 and layer in SPARSE else "")
        groups.setdefault(geometry, []).append(layer)
    index_layers = [layer for layer in held if layer in SPARSE]
    extra = [
        EntryField(
            "index",
            len(index_layers),
            (BLOCK * (64 if index_fp4 else DIM),),
            torch.uint8,
        )
    ]
    if index_fp4:
        extra.append(
            EntryField("index_scale", len(index_layers), (BLOCK * 8,), torch.uint8)
        )
    builder = SimpleNamespace(
        kv_pools={},
        pool_rows={"index": {modules[n]: i for i, n in enumerate(index_layers)}},
        model_runner=SimpleNamespace(
            block_size=BLOCK,
            config=SimpleNamespace(index_cache_dtype="fp4" if index_fp4 else "fp8"),
        ),
    )
    # Each value is [block, physical_page, head, ...] or [block, ...] for index.
    physical = {}
    for geometry, layers in groups.items():
        pool = MhaKvPool(
            layers=len(layers),
            block_size=BLOCK,
            num_kv_heads=heads,
            head_dim=DIM,
            kv_dtype=dtype,
            kv_format=geometry.kv_format,
            extra_fields=extra,
        )
        extra = ()
        pool.allocate(BLOCKS, "cpu")
        builder.kv_pools[geometry] = pool
        builder.pool_rows[geometry] = {modules[n]: i for i, n in enumerate(layers)}
        for row, layer in enumerate(layers):
            k, v = pool.kv_views(row)
            ks, vs = pool.scale_views(row) if pool.scale_fields else (None, None)
            if layer in SPARSE and not geometry.kv_format:
                k, v, ks, vs = to_page16_shuffle(k, v, ks, vs)
            for field, view in (("k", k), ("v", v), ("k_scale", ks), ("v_scale", vs)):
                if view is not None:
                    physical[f"m3.layer_{layer}.{field}"] = view.view(torch.uint8).view(
                        BLOCKS,
                        -1,
                        heads,
                        *view.shape[2:-1],
                        view.shape[-1] * view.element_size(),
                    )
        for field in pool.extra_fields:
            for row, layer in enumerate(index_layers):
                physical[f"m3.layer_{layer}.{field.name}"] = pool.field_view(
                    field.name, row, torch.uint8, (BLOCKS, -1)
                )
    raw = _export_backend(builder)
    return builder, raw, physical


def _configure(builder, raw, *, nvfp4, index_fp4):
    configure_m3_kv_transfer_tensors(
        raw,
        builder,
        num_layers=8,
        sparse_layers=SPARSE,
        index_fp4=index_fp4,
        sparse_nvfp4=nvfp4,
    )
    return raw


def _layout(
    held, heads, *, nvfp4=True, index_fp4=True, dtype=torch.float8_e4m3fn, draft=False
):
    builder, raw, views = _target(
        held, heads, nvfp4=nvfp4, index_fp4=index_fp4, dtype=dtype
    )
    tt = _configure(builder, raw, nvfp4=nvfp4, index_fp4=index_fp4)
    if draft:
        draft_builder = object.__new__(DraftKvBuilder)
        pool = MhaKvPool(
            layers=2,
            block_size=BLOCK,
            num_kv_heads=heads,
            head_dim=DIM,
            kv_dtype=torch.float8_e4m3fn,
        )
        pool.allocate(BLOCKS, "cpu")
        draft_builder._kv_pool = pool
        draft_builder._pool_rows = {
            DRAFT_KV_ROWS: {
                type("Binding", (), {"shuffle_page_tokens": 16})(): row
                for row in range(2)
            }
        }
        for row in range(2):
            k, v, ks, vs = to_page16_shuffle(
                *pool.kv_views(row), *pool.scale_views(row)
            )
            for field, view in (("k", k), ("v", v), ("k_scale", ks), ("v_scale", vs)):
                views[f"draft.{field}.layer_{row}"] = view.view(torch.uint8).view(
                    BLOCKS, 8, heads, -1
                )
        tt.merge_pages(draft_builder.get_kv_transfer_tensors())
    tt.set_block_count(BLOCKS)
    return tt, views


@pytest.mark.parametrize("nvfp4", [False, True])
@pytest.mark.parametrize("index_fp4", [False, True])
@pytest.mark.parametrize("dtype", [torch.float8_e4m3fn, torch.bfloat16])
def test_all_pp_stages_cover_consumer_once_including_draft(nvfp4, index_fp4, dtype):
    consumer, _ = _layout(
        range(8), 1, nvfp4=nvfp4, index_fp4=index_fp4, dtype=dtype, draft=True
    )
    roles = [p.region.semantic_role for p in consumer.pages]
    scale_layers = range(8) if dtype.itemsize == 1 else sorted(SPARSE) if nvfp4 else []
    expected = [
        f"m3.layer_{n}.{field}"
        for field, layers in (
            ("k", range(8)),
            ("v", range(8)),
            ("k_scale", scale_layers),
            ("v_scale", scale_layers),
            ("index", sorted(SPARSE)),
            ("index_scale", sorted(SPARSE) if index_fp4 else []),
        )
        for n in layers
    ]
    expected += [
        f"draft.{field}.layer_{n}"
        for field in ("k", "v", "k_scale", "v_scale")
        for n in range(2)
    ]
    assert roles == expected
    all_indices = []
    for held in STAGES:
        tt, _ = _layout(
            held, 4, nvfp4=nvfp4, index_fp4=index_fp4, dtype=dtype, draft=7 in held
        )
        assert [roles[i] for i in tt.block_region_consumer_indices] == [
            p.region.semantic_role for p in tt.pages
        ]
        all_indices.extend(tt.block_region_consumer_indices)
        assert len(_build_cache_views(tt, num_blocks=BLOCKS).tensors) == len(tt.pages)
    assert sorted(all_indices) == list(range(len(roles)))
    assert consumer.block_region_consumer_indices == list(range(len(roles)))
    assert consumer.block_region_consumer_count == len(roles)


def test_model_hook_uses_runtime_flags_and_keeps_every_byte_view(monkeypatch):
    monkeypatch.setenv("ATOM_M3_SPARSE_KV_NVFP4", "1")
    builder, raw, _ = _target([1, 2, 3], 4)
    pointers = {
        (
            p.region.base_addr,
            p.region.unit_bytes,
            p.region.total_bytes,
            p.view.data_ptr(),
        )
        for p in raw.pages
    }
    model = SimpleNamespace(
        config=SimpleNamespace(
            num_hidden_layers=8,
            sparse_attention_config={
                "sparse_attention_freq": [int(n in SPARSE) for n in range(8)]
            },
        )
    )
    _annotate_model(model, raw, builder)
    assert pointers == {
        (
            p.region.base_addr,
            p.region.unit_bytes,
            p.region.total_bytes,
            p.view.data_ptr(),
        )
        for p in raw.pages
    }
    assert raw.block_region_consumer_count == 40
    runs = {p.region.semantic_role: p.head_major_runs for p in raw.pages}
    assert runs["m3.layer_1.k"] == runs["m3.layer_1.k_scale"] == (1, 4)
    assert runs["m3.layer_2.index_scale"] is None


@pytest.mark.parametrize(
    "bad", ["reordered", "missing", "wrong_role", "duplicate", "missing_index_scale"]
)
def test_incomplete_or_misidentified_backend_pages_fail_before_publication(bad):
    builder, raw, _ = _target([1, 2, 3], 4)
    if bad == "reordered":
        raw.pages.reverse()
    elif bad == "missing":
        raw.pages.pop()
    elif bad == "wrong_role":
        raw.pages[0] = replace(
            raw.pages[0], region=replace(raw.pages[0].region, semantic_role="wrong")
        )
    elif bad == "duplicate":
        raw.pages[1] = raw.pages[0]
    else:
        builder, raw, _ = _target([1, 2, 3], 4, index_fp4=False)
    before = raw.pages
    with pytest.raises((RuntimeError, ValueError)):
        _configure(builder, raw, nvfp4=True, index_fp4=True)
    assert raw.pages is before
    assert raw.block_region_consumer_indices is None


def _fill(views, *, zeros=False):
    for ordinal, (role, view) in enumerate(views.items()):
        if not zeros:
            # Distinct head/token/plane/block data; compare with physical cache
            # views, not a second implementation of consumer_head_slices.
            view.copy_(
                torch.arange(view.numel())
                .reshape(view.shape)
                .add(ordinal * 17)
                .remainder(251)
                .to(torch.uint8)
            )
        else:
            view.zero_()


def _producer(tt, heads):
    c, _, req, _, _ = protocol.producer()
    c.tp_size = 4 // heads
    c.pp_size = 4
    c.block_size = BLOCK
    c._block_region_consumer_indices = tt.block_region_consumer_indices
    c._block_region_roles = [p.region.semantic_role for p in tt.pages]
    c._block_head_major_runs = [p.head_major_runs for p in tt.pages]
    c._per_block_bytes_list = [p.region.unit_bytes for p in tt.pages]
    c.kv_caches_base_addr = [p.region.base_addr for p in tt.pages]
    c._kv_staging = None
    c._MAX_RDMA_ENTRIES_PER_BATCH = 7
    return c, req


def _request(req, tt, rank):
    req.update(
        consumer_tp_size=4,
        consumer_tp_rank=rank,
        consumer_base_addrs=[p.region.base_addr for p in tt.pages],
        consumer_region_roles=[p.region.semantic_role for p in tt.pages],
        consumer_block_bpb=[p.region.unit_bytes for p in tt.pages],
        consumer_head_major_runs=[p.head_major_runs for p in tt.pages],
    )


def _assert_copy(source, destination, heads, rank, src_ids, dst_ids):
    for role, dst in destination.items():
        for src_block, dst_block in zip(src_ids, dst_ids):
            if role not in source:
                assert not dst[dst_block].any()
                continue
            src = source[role][src_block]
            if src.ndim > 1:
                src = src[
                    :, rank if heads == 4 else 0 : (rank if heads == 4 else 0) + 1
                ]
            assert torch.equal(src.reshape(-1), dst[dst_block].reshape(-1)), (
                role,
                rank,
            )


@pytest.mark.parametrize("held", STAGES)
@pytest.mark.parametrize("heads", [1, 4])
@pytest.mark.parametrize(
    "nvfp4,index_fp4", [(False, False), (False, True), (True, True)]
)
def test_full_transfer_matches_physical_target_and_draft_views(
    held, heads, nvfp4, index_fp4
):
    tt, source = _layout(held, heads, nvfp4=nvfp4, index_fp4=index_fp4, draft=7 in held)
    _fill(source)
    c, req = _producer(tt, heads)
    for rank in range(4):
        consumer, destination = _layout(
            range(8), 1, nvfp4=nvfp4, index_fp4=index_fp4, draft=True
        )
        _fill(destination, zeros=True)
        _request(req, consumer, rank)
        assert c._execute_block_transfer(req, "cpu", [3, 1], [0, 2], f"req-{rank}")
        _assert_copy(source, destination, heads, rank, [3, 1], [0, 2])
        assert all(not v[[1, 3]].any() for v in destination.values())


@pytest.mark.parametrize("held", STAGES)
@pytest.mark.parametrize("cached", [0, 1])
def test_chunked_pp_transfer_skips_prefix_and_waits_for_handoff(held, cached):
    tt, source = _layout(held, 4, draft=7 in held)
    _fill(source)
    consumer, destination = _layout(range(8), 1, draft=True)
    _fill(destination, zeros=True)
    c, req = _producer(tt, 4)
    c._MAX_RDMA_ENTRIES_PER_BATCH = 10000  # One write batch per published chunk.
    _request(req, consumer, 2)
    req.update(num_computed_blocks=cached, dst_block_ids=[2, 0, 3][cached:])
    state = ChunkedPrefill(7, [3, 1, 2], 258, BLOCK)
    c._chunked_prefills["xfer-a"] = state
    protocol.publish(state, (cached + 1) * BLOCK)
    with ThreadPoolExecutor(1) as executor:
        future = executor.submit(protocol.dispatch, c, req)
        try:
            for pos in range(cached, 3):
                if pos > cached:
                    protocol.publish(state, min((pos + 1) * BLOCK, 258))
                c.transfer_engine.writes.get(timeout=5)
                _assert_copy(
                    source,
                    destination,
                    4,
                    2,
                    [3, 1, 2][cached : pos + 1],
                    [2, 0, 3][cached : pos + 1],
                )
                assert all(
                    not v[[2, 0, 3][pos + 1 :]].any() for v in destination.values()
                )
                assert not future.done()
            handoff = {"first_token_id": 42, "draft_token_ids": []}
            state.finish(handoff)
            future.result(timeout=5)
            assert state.completed_readers == 1
            assert not state.source_safe()  # Other TP ranks still need this source.
        finally:
            state.cancel()
    assert req["prefill_handoff"] == handoff
    c._notify_transfer_result.assert_called_once_with(req, success=True)
    if cached:
        assert all(not v[2].any() for v in destination.values())
    assert state.source_safe()
