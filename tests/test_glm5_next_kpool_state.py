# SPDX-License-Identifier: MIT

"""GLM-5.3 k-pool state contracts (CPU plus ROCm-only kernel cases)."""

import importlib
from types import SimpleNamespace

import pytest
import torch

from atom.utils import forward_context

GLM5 = pytest.importorskip(
    "atom.models.glm5_next",
    reason="the GLM-5.3 model imports the AITER runtime",
    exc_type=ImportError,
)
from atom.model_ops.glm5_next import indexer as KPOOL_INDEXER


def _config_for_normalize(**overrides):
    values = {
        "num_hidden_layers": 3,
        "num_attention_heads": 8,
        "linear_attn_config": {"kda_layers": [0, 2]},
        "layer_types": [
            "linear_attention",
            "deepseek_sparse_attention",
            "linear_attention",
        ],
        "n_routed_experts": 4,
        "qk_nope_head_dim": 256,
        "qk_rope_head_dim": 0,
        "kv_lora_rank": 512,
        "index_head_dim": 128,
        "index_kpool": 4,
        "index_topk": 2048,
        "index_kpool_always_select_tail": True,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def test_normalize_derives_a_missing_full_attention_list():
    config = _config_for_normalize()

    GLM5._normalize_glm5_next_config(config)

    assert config.glm5_kda_layers == [0, 2]
    assert config.glm5_full_attn_layers == [1]


def test_normalize_rejects_conflicting_attention_layouts():
    config = _config_for_normalize(
        linear_attn_config={
            "kda_layers": [0, 2],
            "full_attn_layers": [0, 1],
        }
    )

    with pytest.raises(ValueError, match="disjoint"):
        GLM5._normalize_glm5_next_config(config)


def test_normalize_rejects_non_nope_geometry():
    config = _config_for_normalize(qk_rope_head_dim=64)

    with pytest.raises(ValueError, match="qk_rope_head_dim"):
        GLM5._normalize_glm5_next_config(config)


def test_dummy_custom_op_result_is_fresh_fp32_like_its_fake(monkeypatch):
    """Runtime and fake implementations must have the same alias/dtype contract."""
    monkeypatch.setattr(
        forward_context,
        "get_forward_context",
        lambda: SimpleNamespace(
            attn_metadata=None,
            context=SimpleNamespace(is_dummy_run=True),
        ),
    )
    weights = torch.ones((2, 3), dtype=torch.bfloat16)
    out = KPOOL_INDEXER._sparse_attn_indexer_kpool(
        hidden_states=torch.empty((2, 4)),
        kv_cache=torch.empty(1),
        q_fp8=torch.empty((2, 1, 4)),
        k=torch.empty((2, 4)),
        gate_score=torch.empty((2, 4)),
        weights=weights,
        compress_ape=torch.empty((4, 4)),
        tail_cache=torch.empty((1, 2, 4, 4)),
        state_slot_idx_in=torch.zeros(2, dtype=torch.int32),
        state_slot_idx=torch.zeros(2, dtype=torch.int32),
        positions=torch.arange(2),
        sparse_kv_indices_buffer=torch.empty(1),
        topk_tokens=4,
        index_kpool=4,
        head_dim=4,
        max_model_len=16,
        topk_out_width=128,
        scale_fmt="ue8m0",
        stable_topk=False,
    )

    assert out.dtype == torch.float32
    assert out.data_ptr() != weights.data_ptr()
    assert torch.equal(out, weights.float())


def test_one_row_cached_chunk_can_close_a_pool_from_the_input_tail(monkeypatch):
    """A chunk starting at absolute position 103 closes pool 100..103 at row 0."""
    seen = {}

    def pool_and_rotate(pool_k, pool_gate, _ape):
        seen["pool_k"] = pool_k.clone()
        seen["pool_gate"] = pool_gate.clone()
        return pool_k.sum(dim=1)

    def pool_slot_mapping(_bt, pool_ids, _req_idx, _rows):
        seen["pool_ids"] = pool_ids.clone()
        return torch.where(pool_ids >= 0, torch.full_like(pool_ids, 7), pool_ids)

    def cache_write(pooled, _cache, slots, *_args, **_kwargs):
        seen["pooled"] = pooled.clone()
        seen["slots"] = slots.clone()

    monkeypatch.setattr(KPOOL_INDEXER.kpool, "pool_and_rotate", pool_and_rotate)
    monkeypatch.setattr(KPOOL_INDEXER.kpool, "pool_slot_mapping", pool_slot_mapping)
    monkeypatch.setattr(KPOOL_INDEXER, "indexer_k_quant_and_cache", cache_write)

    tail = torch.zeros((12, 2, 4, 2), dtype=torch.bfloat16)
    for phase in range(3):
        tail[5, 0, phase] = 100 + phase
        tail[5, 1, phase] = 200 + phase

    KPOOL_INDEXER._kpool_write_completed_pools(
        kv_cache=torch.empty(1),
        k=torch.full((1, 2), 103, dtype=torch.bfloat16),
        gate_score=torch.full((1, 2), 203, dtype=torch.bfloat16),
        positions=torch.tensor([103]),
        pool_bt=torch.empty((1, 1), dtype=torch.int32),
        req_idx=torch.tensor([0]),
        compress_ape=torch.zeros((4, 2)),
        index_kpool=4,
        head_dim=2,
        scale_fmt="ue8m0",
        pool_rows=16,
        chunk_start=torch.tensor([103]),
        tail_cache=tail,
        state_slot_idx_in=torch.tensor([5]),
        state_slot_idx=torch.tensor([9]),
    )

    assert seen["pool_k"][:, :, 0].tolist() == [[100, 101, 102, 103]]
    assert seen["pool_gate"][:, :, 0].tolist() == [[200, 201, 202, 203]]
    assert seen["pool_ids"].tolist() == [25]
    assert seen["slots"].tolist() == [7]


def test_invalid_output_slot_cannot_publish_a_completed_pool(monkeypatch):
    seen = {}
    monkeypatch.setattr(
        KPOOL_INDEXER.kpool,
        "pool_and_rotate",
        lambda pool_k, _pool_gate, _ape: pool_k.sum(dim=1),
    )

    def pool_slot_mapping(_bt, pool_ids, _req_idx, _rows):
        seen["pool_ids"] = pool_ids.clone()
        return pool_ids

    monkeypatch.setattr(KPOOL_INDEXER.kpool, "pool_slot_mapping", pool_slot_mapping)
    monkeypatch.setattr(
        KPOOL_INDEXER, "indexer_k_quant_and_cache", lambda *_args, **_kwargs: None
    )

    KPOOL_INDEXER._kpool_write_completed_pools(
        kv_cache=torch.empty(1),
        k=torch.ones((4, 2), dtype=torch.bfloat16),
        gate_score=torch.ones((4, 2), dtype=torch.bfloat16),
        positions=torch.arange(4),
        pool_bt=torch.empty((1, 1), dtype=torch.int32),
        req_idx=torch.zeros(4, dtype=torch.int64),
        compress_ape=torch.zeros((4, 2)),
        index_kpool=4,
        head_dim=2,
        scale_fmt="ue8m0",
        pool_rows=16,
        state_slot_idx=torch.tensor([-1]),
    )

    assert seen["pool_ids"].tolist() == [-1, -1, -1, -1]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="exercises Triton tail copy")
def test_prefill_tail_fork_copies_prior_rows_into_the_output_slot():
    device = torch.device("cuda")
    tail = torch.zeros((12, 2, 4, 128), dtype=torch.bfloat16, device=device)
    for phase in range(2):
        tail[5, 0, phase] = 100 + phase
        tail[5, 1, phase] = 200 + phase

    KPOOL_INDEXER.kpool.kpool_seed_tail(
        tail,
        torch.full((1, 128), 102, dtype=torch.bfloat16, device=device),
        torch.full((1, 128), 202, dtype=torch.bfloat16, device=device),
        torch.tensor([102], dtype=torch.int64, device=device),
        torch.tensor([0, 1], dtype=torch.int32, device=device),
        torch.tensor([9], dtype=torch.int32, device=device),
        4,
        slot_idx_in=torch.tensor([5], dtype=torch.int32, device=device),
    )

    assert tail[9, 0, :, 0].cpu().tolist() == [100, 101, 102, 0]
    assert tail[9, 1, :, 0].cpu().tolist() == [200, 201, 202, 0]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="exercises Triton tail copy")
def test_decode_tail_fork_reads_input_and_materializes_output_slot():
    device = torch.device("cuda")
    tail = torch.zeros((12, 2, 4, 128), dtype=torch.bfloat16, device=device)
    for phase in range(3):
        tail[5, 0, phase] = phase + 1
        tail[5, 1, phase] = phase + 11

    out = KPOOL_INDEXER.kpool.kpool_decode_stash_and_pool(
        tail,
        torch.full((1, 128), 4, dtype=torch.bfloat16, device=device),
        torch.full((1, 128), 14, dtype=torch.bfloat16, device=device),
        torch.tensor([3], dtype=torch.int64, device=device),
        torch.tensor([9], dtype=torch.int32, device=device),
        torch.zeros((4, 128), dtype=torch.float32, device=device),
        4,
        slot_idx_in=torch.tensor([5], dtype=torch.int32, device=device),
    )

    assert out.shape == (1, 128)
    assert tail[9, 0, :, 0].cpu().tolist() == [1, 2, 3, 4]
    assert tail[9, 1, :, 0].cpu().tolist() == [11, 12, 13, 14]


def test_cached_chunk_reads_the_tail_from_its_history_ring_rows(monkeypatch):
    """Under MTP the tail is the history ring: position `p` lives at `p % 8`."""
    seen = {}

    def pool_and_rotate(pool_k, pool_gate, _ape):
        seen["pool_k"] = pool_k.clone()
        seen["pool_gate"] = pool_gate.clone()
        return pool_k.sum(dim=1)

    monkeypatch.setattr(KPOOL_INDEXER.kpool, "pool_and_rotate", pool_and_rotate)
    monkeypatch.setattr(
        KPOOL_INDEXER.kpool,
        "pool_slot_mapping",
        lambda _bt, pool_ids, _req_idx, _rows: pool_ids,
    )
    monkeypatch.setattr(
        KPOOL_INDEXER, "indexer_k_quant_and_cache", lambda *_args, **_kwargs: None
    )

    tail = torch.full((12, 2, 8, 2), -1, dtype=torch.bfloat16)
    for position in range(100, 103):
        tail[5, 0, position % 8] = position
        tail[5, 1, position % 8] = position + 100

    KPOOL_INDEXER._kpool_write_completed_pools(
        kv_cache=torch.empty(1),
        k=torch.full((1, 2), 103, dtype=torch.bfloat16),
        gate_score=torch.full((1, 2), 203, dtype=torch.bfloat16),
        positions=torch.tensor([103]),
        pool_bt=torch.empty((1, 1), dtype=torch.int32),
        req_idx=torch.tensor([0]),
        compress_ape=torch.zeros((4, 2)),
        index_kpool=4,
        head_dim=2,
        scale_fmt="ue8m0",
        pool_rows=16,
        chunk_start=torch.tensor([103]),
        tail_cache=tail,
        state_slot_idx_in=torch.tensor([5]),
        state_slot_idx=torch.tensor([9]),
    )

    assert seen["pool_k"][:, :, 0].tolist() == [[100, 101, 102, 103]]
    assert seen["pool_gate"][:, :, 0].tolist() == [[200, 201, 202, 203]]


def _run_speculative_verify(monkeypatch, positions, max_model_len):
    """Verify one request's rows at `positions`, recording what the ops get."""
    from atom.model_ops.glm5_next import speculative as SPEC

    # The op imports these at call time; patch the defining modules. A dotted
    # string would resolve `pa_mqa_logits` to the package's re-exported name.
    aiter_cache = importlib.import_module("aiter.ops.cache")
    aiter_topk = importlib.import_module("aiter.ops.topk")
    aiter_logits = importlib.import_module("aiter.ops.triton.attention.pa_mqa_logits")
    seen = {}
    monkeypatch.setattr(
        SPEC,
        "build_speculative_pool_candidates",
        lambda _h, keys, *_args: (
            torch.zeros_like(keys),
            torch.zeros(keys.shape[0], dtype=torch.int64),
        ),
    )
    monkeypatch.setattr(
        SPEC.kpool,
        "pool_slot_mapping",
        lambda _bt, pool_ids, _req_idx, _rows: pool_ids,
    )
    monkeypatch.setattr(SPEC, "update_speculative_kpool_history", lambda *_a: None)
    monkeypatch.setattr(
        aiter_cache, "indexer_k_quant_and_cache", lambda *_a, **_k: None
    )

    def paged_logits(_q, _kv, _w, logits, lens, _bt, max_pools, **_kwargs):
        seen["logits_width"] = logits.shape[1]
        seen["max_pools"] = max_pools
        seen["logits_lens"] = lens.tolist()

    def top_k(_logits, _next_n, lens, *_args, **_kwargs):
        seen["topk_lens"] = lens.tolist()

    monkeypatch.setattr(aiter_logits, "deepgemm_fp8_paged_mqa_logits", paged_logits)
    monkeypatch.setattr(aiter_topk, "top_k_per_row_decode", top_k)
    monkeypatch.setattr(
        SPEC.kpool, "expand_pools_and_append_tail", lambda *_a, **_k: None
    )

    def map_to_slots(*_args):
        seen["mapped"] = True

    monkeypatch.setattr(SPEC, "map_token_indices_to_slots", map_to_slots)

    rows = len(positions)
    SPEC.run_speculative_kpool_indexer(
        SimpleNamespace(
            cu_seqlens_q=torch.tensor([0, rows], dtype=torch.int32),
            max_seqlen_k=0,
            block_tables=torch.zeros((1, 4), dtype=torch.int32),
            sparse_kv_indptr=torch.zeros(rows + 1, dtype=torch.int32),
        ),
        kv_cache=torch.zeros((16, 1, 132), dtype=torch.uint8),
        queries=torch.zeros((rows, 2, 128)),
        keys=torch.zeros((rows, 128), dtype=torch.bfloat16),
        gates=torch.zeros((rows, 128), dtype=torch.bfloat16),
        weights=torch.zeros((rows, 2)),
        pool_bias=torch.zeros((4, 128)),
        history=torch.zeros((1, 2, 16, 128), dtype=torch.bfloat16),
        source_slots=torch.tensor([0], dtype=torch.int32),
        destination_slots=torch.tensor([0], dtype=torch.int32),
        positions=torch.tensor(positions),
        sparse_kv_indices=torch.zeros(16, dtype=torch.int32),
        pool_size=4,
        topk_tokens=2048,
        output_width=2176,
        block_size=16,
        max_model_len=max_model_len,
        scale_fmt="ue8m0",
        stable_topk=False,
    )
    return seen


def test_speculative_verify_selects_below_index_topk_and_at_capture(monkeypatch):
    """MLA verify always reads sparse_kv_indices, so the indexer must write them.

    `max_seqlen_k == 0` is what CUDAGraph capture metadata carries; skipping on
    it would record the skip into every replay. The scoring scratch is sized
    from the model limit for the same reason.
    """
    seen = _run_speculative_verify(monkeypatch, [5, 6], max_model_len=8192)

    assert seen.get("mapped") is True
    assert seen["logits_width"] == seen["max_pools"] == 8192 // 4


def test_speculative_verify_scores_no_pool_past_the_model_limit(monkeypatch):
    """A 15-token request under a 16-token limit verifies 5 drafts at 15..19.

    Row 19 spans pools 0..4, one more than the 4 columns the scratch holds.
    """
    seen = _run_speculative_verify(
        monkeypatch, [14, 15, 16, 17, 18, 19], max_model_len=16
    )

    assert seen["max_pools"] == 4
    assert seen["logits_lens"] == seen["topk_lens"] == [3, 4, 4, 4, 4, 4]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="exercises Triton tail copy")
def test_prefill_tail_seed_writes_history_ring_rows():
    """Under MTP the tail has 8 rows per slot, not the pool's 4."""
    device = torch.device("cuda")
    tail = torch.zeros((12, 2, 8, 128), dtype=torch.bfloat16, device=device)

    KPOOL_INDEXER.kpool.kpool_seed_tail(
        tail,
        torch.arange(100, 103, device=device)[:, None].expand(-1, 128).bfloat16(),
        torch.arange(200, 203, device=device)[:, None].expand(-1, 128).bfloat16(),
        torch.arange(100, 103, dtype=torch.int64, device=device),
        torch.tensor([0, 3], dtype=torch.int32, device=device),
        torch.tensor([9], dtype=torch.int32, device=device),
        4,
    )

    assert tail[9, 0, :, 0].cpu().tolist() == [0, 0, 0, 0, 100, 101, 102, 0]
    assert tail[9, 1, :, 0].cpu().tolist() == [0, 0, 0, 0, 200, 201, 202, 0]
    tail[9] = 0
    assert not tail.any(), "the seed wrote outside its own slot"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="exercises Triton tail copy")
def test_decode_tail_stash_addresses_history_ring_rows():
    device = torch.device("cuda")
    tail = torch.zeros((12, 2, 8, 128), dtype=torch.bfloat16, device=device)
    for position in range(4, 7):
        tail[5, 0, position] = position
        tail[5, 1, position] = position + 10

    KPOOL_INDEXER.kpool.kpool_decode_stash_and_pool(
        tail,
        torch.full((1, 128), 7, dtype=torch.bfloat16, device=device),
        torch.full((1, 128), 17, dtype=torch.bfloat16, device=device),
        torch.tensor([7], dtype=torch.int64, device=device),
        torch.tensor([9], dtype=torch.int32, device=device),
        torch.zeros((4, 128), dtype=torch.float32, device=device),
        4,
        slot_idx_in=torch.tensor([5], dtype=torch.int32, device=device),
    )

    assert tail[9, 0, :, 0].cpu().tolist() == [0, 0, 0, 0, 4, 5, 6, 7]
    assert tail[9, 1, :, 0].cpu().tolist() == [0, 0, 0, 0, 14, 15, 16, 17]
