# SPDX-License-Identifier: MIT
"""Cross-feature contracts: compact MXFP4 PAGEs, PP head cuts and query reuse."""

from types import SimpleNamespace

import pytest
import test_m3_kv_transfer_layout as layout
import torch

from atom.kv_transfer.disaggregation.mha_head_transfer import consumer_head_slices
from atom.model_ops.attentions.mha_kv_pool import MhaKvPool
from atom.model_ops.attentions.pool_layout.entry_arena import EntryField
from atom.model_ops.attentions.pool_layout.pool_rows import KvGeometry
from atom.model_ops.minimax_m3.kv_mxfp4 import PAGE_BYTES
from atom.model_ops.minimax_m3.kv_transfer import configure_m3_kv_transfer_tensors


def _pages(held, heads, computing):
    geometry = KvGeometry(heads, 128, "mxfp4")
    modules = {i: type("Layer", (), {"layer_num": i})() for i in held}
    indexed = sorted(set(held) & computing)
    pool = MhaKvPool(
        layers=len(held),
        block_size=128,
        num_kv_heads=heads,
        head_dim=128,
        kv_dtype=torch.float8_e4m3fn,
        kv_format="mxfp4",
        extra_fields=[EntryField("index", len(indexed), (PAGE_BYTES,), torch.uint8)],
    )
    pool.allocate(2, "cpu")
    builder = SimpleNamespace(
        kv_pools={geometry: pool},
        pool_rows={
            geometry: {modules[i]: row for row, i in enumerate(held)},
            "index": {modules[i]: row for row, i in enumerate(indexed)},
        },
        model_runner=SimpleNamespace(block_size=128),
    )
    tt = layout._export_backend(builder)
    configure_m3_kv_transfer_tensors(
        tt,
        builder,
        num_layers=8,
        sparse_layers=set(range(8)),
        index_fp4=False,
        sparse_nvfp4=False,
        sparse_mxfp4=True,
        computing_index_layers=computing,
    )
    return tt, pool


@pytest.mark.parametrize("computing", [set(range(8)), {0, 4}])
def test_mxfp4_pp_pages_cover_all_kv_and_only_computing_index(computing):
    consumer, _ = _pages(list(range(8)), 1, computing)
    expected = [f"m3.layer_{n}.{field}" for field in ("k", "v") for n in range(8)]
    expected += [f"m3.layer_{n}.index" for n in sorted(computing)]
    assert [p.region.semantic_role for p in consumer.pages] == expected
    indices = []
    for held in ([0, 1], [2, 3], [4, 5], [6, 7]):
        tt, pool = _pages(held, 4, computing)
        indices += tt.block_region_consumer_indices
        for p, dst_index in zip(tt.pages, tt.block_region_consumer_indices):
            dst = consumer.pages[dst_index]
            assert p.region.semantic_role == dst.region.semantic_role
            if p.region.semantic_role.endswith(".index"):
                assert p.head_major_runs is None
                assert p.region.unit_bytes == dst.region.unit_bytes == PAGE_BYTES
            else:
                # One complete data+E8M0 scale page for each head, no page-16 cuts.
                assert p.head_major_runs == (1, 4)
                assert dst.head_major_runs == (1, 1)
                assert p.region.unit_bytes == 4 * PAGE_BYTES
        assert (
            pool.entry_bytes
            == len(held) * 8 * PAGE_BYTES + len(set(held) & computing) * PAGE_BYTES
        )
    assert sorted(indices) == list(range(len(expected)))


@pytest.mark.parametrize("rank", range(4))
def test_mxfp4_head_transfer_preserves_payload_and_embedded_scales(rank):
    source, _ = _pages([0, 1], 4, {0, 4})
    dest, _ = _pages(list(range(8)), 1, {0, 4})
    for ordinal, page in enumerate(source.pages):
        page.view.copy_(
            torch.arange(page.view.numel())
            .reshape(page.view.shape)
            .add(17 * ordinal)
            .remainder(251)
            .to(torch.uint8)
        )
    for page in dest.pages:
        page.view.zero_()
    for src, region in zip(source.pages, source.block_region_consumer_indices):
        dst = dest.pages[region]
        for src_off, dst_off, size in consumer_head_slices(
            src.region.unit_bytes,
            dst.region.unit_bytes,
            src.head_major_runs,
            dst.head_major_runs,
            rank,
            4,
        ):
            dst.view[1].reshape(-1)[dst_off : dst_off + size].copy_(
                src.view[1].reshape(-1)[src_off : src_off + size]
            )
        expected = (
            src.view[1].reshape(-1)
            if src.head_major_runs is None
            else src.view[1].reshape(4, PAGE_BYTES)[rank]
        )
        assert torch.equal(dst.view[1].reshape(-1), expected)
        assert not dst.view[0].any()  # cached-prefix bytes were not overwritten


@pytest.mark.parametrize("fp8_query", [False, True])
def test_mxfp4_reusing_layer_matches_computed_query_cache_key(monkeypatch, fp8_query):
    import sys
    from types import ModuleType
    from unittest.mock import Mock

    from test_minimax_m3_index_cache_rows import definitions

    from atom.utils import envs

    monkeypatch.setattr(envs, "ATOM_M3_INDEX_Q_FP8", fp8_query)
    sparse = ModuleType("atom.model_ops.minimax_m3.sparse_attn")
    sparse.to_page16_shuffle = Mock(side_effect=AssertionError("MXFP4 is not SHUFFLE"))
    monkeypatch.setitem(sys.modules, sparse.__name__, sparse)
    model = ModuleType("atom.models.minimax_m3.model")
    model._minimax_m3_cos_sin_cache = lambda *_: torch.ones(2, 64)
    monkeypatch.setitem(sys.modules, model.__name__, model)
    fused = Mock()
    ns = {
        "torch": torch,
        "aiter": SimpleNamespace(fused_qknorm_idxrqknorm=fused),
        "envs": envs,
        "ForwardContext": object,
    }
    definitions(
        "atom/model_ops/attention_mha.py",
        {"rope_cache", "_topk_cache_key"},
        ns,
        "SparseMHAPagedAttentionImpl",
    )
    norm = SimpleNamespace(weight=torch.ones(128), variance_epsilon=1e-6)
    impl = SimpleNamespace(
        layer_num=0,
        num_heads=1,
        num_kv_heads=1,
        num_idx_heads=1,
        head_dim=128,
        index_head_dim=128,
        index_q_size=128,
        kv_mxfp4=True,
        kv_nvfp4=False,
        index_mxfp4=True,
        kv_cache_dtype="fp8",
        index_cache_dtype="mxfp4",
        q_norm=norm,
        k_norm=norm,
        index_q_norm=norm,
        index_k_norm=norm,
        rotary_emb=SimpleNamespace(rotary_dim=64),
        index_scale=None,
        index_cache=torch.empty(2, PAGE_BYTES, dtype=torch.uint8),
        skip_index_topk=False,
        topk=16,
        init_blocks=1,
        local_blocks=1,
        indexer_cp_group=None,
    )
    ctx = SimpleNamespace(
        attn_metadata=SimpleNamespace(slot_mapping=torch.arange(2)),
        kv_cache_data={
            "layer_0": SimpleNamespace(
                k_cache=torch.empty(2, 1, PAGE_BYTES, dtype=torch.uint8),
                v_cache=torch.empty(2, 1, PAGE_BYTES, dtype=torch.uint8),
            )
        },
    )
    qkv = torch.zeros(2, 640, dtype=torch.bfloat16)
    tables, lengths = torch.zeros(2, 1, dtype=torch.int32), torch.ones(
        2, dtype=torch.int32
    )
    ns["rope_cache"](impl, None, None, None, qkv, torch.arange(2), ctx)
    computed = ns["_topk_cache_key"](
        impl, "decode", impl._index_q, tables, lengths, 1, 128
    )
    impl.skip_index_topk = True
    impl.index_cache = None
    ns["rope_cache"](impl, None, None, None, qkv, torch.arange(2), ctx)
    reused = ns["_topk_cache_key"](impl, "decode", None, tables, lengths, 1, 128)
    assert computed == reused
    assert fused.call_args.kwargs["skip_index_branch"] is True
    assert fused.call_args.args[15] is None
